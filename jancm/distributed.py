"""Multi-GPU (torchrun / DDP) setup shared by the three trainers.

  torchrun --standalone --nproc_per_node 8 scripts/train_ncm.py --config ...
  torchrun --nnodes 2 --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint HOST:29500 scripts/train_ncm.py ...

Without torchrun (WORLD_SIZE unset) everything runs as one process, which the CPU smoke
tests use. Only rank 0 logs, evaluates and writes checkpoints.
"""
import datetime
import os
import signal
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Dist:
    def __init__(self, device_arg="", timeout_minutes=120):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.enabled = self.world > 1
        if device_arg:
            self.device = device_arg
        elif torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = f"cuda:{self.local_rank}"
        else:
            self.device = "cpu"
        self.is_cuda = self.device.startswith("cuda")
        if self.enabled and not dist.is_initialized():
            # Long timeout: rank 0 evaluates and saves while the other ranks wait in the
            # next all-reduce.
            dist.init_process_group("nccl" if self.is_cuda else "gloo",
                                    timeout=datetime.timedelta(minutes=timeout_minutes))
        self.is_main = self.rank == 0

    def wrap(self, module, find_unused=False):
        """DDP-wrap a module. Buffers are not re-broadcast each forward: the generator's are
        constants, and the tokenizer's EMA buffers are synchronised by the quantizer."""
        if not self.enabled:
            return module
        return DDP(module, device_ids=[self.local_rank] if self.is_cuda else None,
                   find_unused_parameters=find_unused, broadcast_buffers=False)

    def no_sync(self, model, last_micro_step):
        """Skip the gradient all-reduce on all but the last micro-step of an accumulation."""
        if self.enabled and not last_micro_step:
            return model.no_sync()
        return nullcontext()

    def mean(self, value):
        if not self.enabled:
            return float(value)
        t = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        dist.all_reduce(t)
        return float(t) / self.world

    def any(self, flag):
        if not self.enabled:
            return bool(flag)
        t = torch.tensor(1 if flag else 0, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())

    def broadcast_module(self, module):
        """Copy rank 0's parameters and buffers to every rank (after a non-DDP init step)."""
        if self.enabled:
            for t in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(t.data, 0)

    def barrier(self):
        if self.enabled:
            dist.barrier()

    def close(self):
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def resolve_batching(T, world):
    """``batch_size`` is the global batch per optimizer step. The per-GPU size comes from
    ``micro_batch_size`` (gradient accumulation fills the rest) or, if absent, from
    ``grad_accum``. The same config therefore gives the same optimisation on 1 or 16 GPUs."""
    glob = int(T.batch_size)
    if T.get("micro_batch_size"):
        micro = min(int(T.micro_batch_size), max(1, glob // world))
        if glob % (micro * world):
            raise ValueError(f"batch_size {glob} is not divisible by micro_batch_size {micro} x {world} GPUs")
        return micro, glob // (micro * world)
    accum = int(T.get("grad_accum", 1))
    if glob % (world * accum):
        raise ValueError(f"batch_size {glob} is not divisible by {world} GPUs x grad_accum {accum}")
    return glob // (world * accum), accum


class Preemption:
    """Turns SIGTERM / SIGUSR1 (preemption, torchrun shutdown, manual stop) into a
    flag that the training loop checks once per step, so every rank stops at the same step
    and rank 0 writes last.pt before exiting."""

    def __init__(self):
        self.flag = False
        for s in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(s, self._handler)

    def _handler(self, signum, frame):
        self.flag = True
        print(f"[preempt] signal {signum}: saving at the end of this step", flush=True)


def resolve_resume(arg, out_dir):
    """--resume auto -> <out_dir>/last.pt if it exists (restart a preempted job unchanged)."""
    if arg == "auto":
        p = os.path.join(out_dir, "last.pt")
        return p if os.path.exists(p) else ""
    return arg
