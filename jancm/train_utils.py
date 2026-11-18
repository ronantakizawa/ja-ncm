"""Small shared training utilities: LR schedule, optimizer groups, logging, checkpoints."""
import json
import math
import os
import shlex
import subprocess
import time

import torch


def lr_at(step, peak, floor, warmup, total):
    """Linear warmup, then cosine decay to ``floor``."""
    if step < warmup:
        return peak * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t))


def adamw(model, lr, weight_decay, betas):
    """Decay matrices only; norms, biases and scalars are not decayed."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    kw = {"fused": True} if torch.cuda.is_available() else {}
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas), **kw)


def is_cuda(device):
    return str(device).startswith("cuda")


def autocast(device, enabled):
    if enabled and is_cuda(device):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def sync(device):
    if is_cuda(device):
        torch.cuda.synchronize()
    elif str(device) == "mps":
        torch.mps.synchronize()


class RunLog:
    """Rank 0 appends JSON lines to metrics.jsonl and records GPU-hours for the cost table.

    GPU-hours = wall time x GPUs used by this job (the torchrun world size), not the number
    of GPUs on the machine. Wall time accumulates across resumes (run_stats.json).
    """

    def __init__(self, out_dir, n_gpu=1, enabled=True):
        self.enabled = enabled
        self.out_dir = out_dir
        self.n_gpu = n_gpu if torch.cuda.is_available() else 0
        self.t0 = time.time()
        self.prev_hours = 0.0
        if not enabled:
            return
        os.makedirs(out_dir, exist_ok=True)
        self.f = open(os.path.join(out_dir, "metrics.jsonl"), "a")
        self.gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        stats = os.path.join(out_dir, "run_stats.json")
        if os.path.exists(stats):
            with open(stats) as f:
                self.prev_hours = float(json.load(f).get("wall_hours", 0.0))

    def log(self, **kv):
        if not self.enabled:
            return
        kv["elapsed_s"] = round(time.time() - self.t0, 1)
        if torch.cuda.is_available():
            kv["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
        self.f.write(json.dumps(kv, ensure_ascii=False) + "\n")
        self.f.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kv.items()), flush=True)

    def finish(self, **kv):
        if not self.enabled:
            return
        hours = self.prev_hours + (time.time() - self.t0) / 3600
        stats = {"gpu": self.gpu, "n_gpu": self.n_gpu, "wall_hours": round(hours, 3),
                 "gpu_hours": round(hours * self.n_gpu, 3), **kv}
        with open(os.path.join(self.out_dir, "run_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[run] {stats}", flush=True)


class Checkpointer:
    """Rank-0 checkpoint writer for preemption-safe training.

    * last.pt (with optimizer state) every ``save_every_minutes`` and at every step-based
      save, so an interruption loses at most that many minutes;
    * generator_stepN.pt / tokenizer.pt snapshots every ``save_every`` steps;
    * an optional ``sync_cmd`` (e.g. "rsync -a {out_dir}/ /shared/run/") runs in the
      background after each save, because a preempted node loses its local disk.
    Writes are atomic (tmp + rename), so a crash mid-save keeps the previous file.
    """

    def __init__(self, out_dir, T, enabled=True):
        self.out_dir, self.enabled = out_dir, enabled
        self.every_steps = int(T.get("save_every", 0) or 0)
        self.every_s = 60 * float(T.get("save_every_minutes", 15) or 0)
        self.sync_cmd = str(T.get("sync_cmd", "") or "")
        self.last_t = time.time()
        self._proc = None

    def step_due(self, step, total):
        return self.enabled and (step == total or (self.every_steps and step % self.every_steps == 0))

    def time_due(self):
        return self.enabled and self.every_s > 0 and time.time() - self.last_t >= self.every_s

    def save(self, name, obj):
        if not self.enabled:
            return
        path = os.path.join(self.out_dir, name)
        torch.save(obj, path + ".tmp")
        os.replace(path + ".tmp", path)
        if name == "last.pt":
            self.last_t = time.time()

    def sync(self, wait=False):
        if not (self.enabled and self.sync_cmd):
            return
        if self._proc is not None and self._proc.poll() is None:
            if not wait:
                return                                  # previous sync still running
            self._proc.wait()
        cmd = self.sync_cmd.format(out_dir=shlex.quote(self.out_dir))
        self._proc = subprocess.Popen(cmd, shell=True)
        if wait:
            self._proc.wait()
