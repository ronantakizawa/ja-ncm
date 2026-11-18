"""Train a multiscale residual VQ-VAE concept tokenizer from scratch (Phase B1, Stage 0).
Multi-GPU with torchrun.

  torchrun --standalone --nproc_per_node 8 scripts/train_tokenizer.py --config configs/phase_b/ja_tokenizer.yaml
  torchrun ... scripts/train_tokenizer.py --config configs/stage0/lm1b_tokenizer.yaml        # B0 check
  Resume   add --resume auto
  Smoke    python scripts/train_tokenizer.py --config ... --device cpu tokenizer_train.steps=3 ...

Codebooks are EMA-updated once per optimizer step from statistics summed over every
micro-batch and every GPU (tokenizer_train.ema_per_optimizer_step, default true), so the
update matches one global batch of tokenizer_train.batch_size windows.

Writes tokenizer_train.out_dir/tokenizer.pt ({"model": state_dict}, loadable by the
upstream VQVAETokenizer and by every generator config through checkpoints.tokenizer),
plus last.pt (resume), best.pt, metrics.jsonl and run_stats.json (GPU-hours).
"""
import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402
from jancm.data import MixtureSampler, open_split  # noqa: E402
from jancm.distributed import Dist, Preemption, resolve_batching, resolve_resume  # noqa: E402
from jancm.loading import load_config, load_state, resolve  # noqa: E402
from jancm.text import pretokenizer_from_config  # noqa: E402
from jancm.tokenizer_objective import CodeCorruption, DualFineDrop, decoder_robustness, evaluate_tokenizer, kmeans_init_codebooks, tokenizer_loss  # noqa: E402
from jancm.train_utils import Checkpointer, RunLog, adamw, autocast, lr_at  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402


class TokenizerStep(torch.nn.Module):
    """Puts the loss computation inside forward() so DDP synchronises its gradients."""

    def __init__(self, vqvae, dual, chunk_rows, use_bf16, device, corrupt=None):
        super().__init__()
        self.vqvae, self.dual, self.chunk, self.corrupt = vqvae, dual, chunk_rows, corrupt
        self.use_bf16, self.dev = use_bf16, device

    def forward(self, x):
        return tokenizer_loss(self.vqvae, x, self.dual, self.chunk, autocast(self.dev, self.use_bf16), self.corrupt)


def state(vqvae, cfg, step, **extra):
    return {"model": vqvae.state_dict(), "step": step, "config": OmegaConf.to_container(cfg, resolve=True), **extra}


def dev_batches(dev, n, bs, device):
    for s in range(0, min(n, len(dev)), bs):
        yield torch.from_numpy(dev.windows(range(s, min(s + bs, n, len(dev))))).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="", help="checkpoint path, or 'auto' for <out_dir>/last.pt if present")
    ap.add_argument("--device", default="", help="force a device (e.g. cpu for smoke tests)")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()

    cfg = load_config(a.config, a.overrides)
    T = cfg.tokenizer_train
    D = Dist(a.device)
    device = D.device
    torch.manual_seed(int(T.seed))
    use_bf16 = bool(T.bf16) and D.is_cuda
    if D.is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = resolve(T.out_dir, ROOT)
    resume = resolve_resume(a.resume, out_dir)
    micro, accum = resolve_batching(T, D.world)

    pretok = pretokenizer_from_config(cfg)
    vcfg = cfg.tokenizer.vqvae
    vqvae = VQVAE(**vcfg).to(device)   # same construction as upstream VQVAETokenizer.from_config
    vq = vqvae.vq
    vq.defer_ema = bool(T.get("ema_per_optimizer_step", True))
    dual = DualFineDrop.from_vq_config(vcfg.vector_quantizer_config, len(vq.levels))
    corrupt = CodeCorruption.from_vq_config(vcfg.vector_quantizer_config)

    train = open_split(cfg.data, "train", root=ROOT)
    dev = open_split(cfg.data, "dev", root=ROOT)
    assert train.window == vqvae.context_length, f"windows of {train.window} vs context_length {vqvae.context_length}"

    opt = adamw(vqvae, float(T.lr), float(T.weight_decay), T.betas)
    step, best, seen = 0, -1.0, 0
    if resume:
        st = load_state(resume)
        vqvae.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        step, best, seen = int(st["step"]), float(st.get("best", best)), int(st.get("seen_tokens", 0))
        del st
    else:
        km = T.get("kmeans_init", None)
        if km and int(km.get("windows", 0)) > 0:
            if D.is_main:   # fit on rank 0, then copy the codebooks to every rank
                rows = np.random.default_rng(int(T.seed)).choice(len(train), min(int(km.windows), len(train)), replace=False)
                x0 = torch.from_numpy(train.windows(rows)).to(device)
                sizes = kmeans_init_codebooks(vqvae, x0, int(km.get("iters", 10)), int(km.get("max_vectors", 262144)))
                print(f"[tok] k-means codebook init on {len(rows)} windows; vectors per level: {sizes}", flush=True)
                del x0
            D.broadcast_module(vqvae)
    model = D.wrap(TokenizerStep(vqvae, dual, int(T.get("ce_chunk_rows", 4096)), use_bf16, device, corrupt),
                   find_unused=bool(T.get("ddp_find_unused", False)))
    if D.is_main:
        print(f"[tok] world={D.world} global_batch={T.batch_size} micro={micro} accum={accum} step={step} "
              f"levels={vq.levels} dual_dropout={dual is not None} code_corruption={vars(corrupt) if corrupt else None} "
              f"ema_per_step={vq.defer_ema} "
              f"train windows={len(train)} dev windows={len(dev)}", flush=True)

    sampler = MixtureSampler(train, {"corpus": 1.0}, micro, seed=int(T.seed) + 1_000_003 * D.rank + step)
    log = RunLog(out_dir, n_gpu=D.world, enabled=D.is_main)
    ckpt = Checkpointer(out_dir, T, enabled=D.is_main)
    if D.is_main:
        OmegaConf.save(cfg, os.path.join(out_dir, "config.yaml"))
    pre = Preemption()
    steps = int(T.steps)
    pad = pretok.pad_token_id
    chunk = int(T.get("ce_chunk_rows", 4096))

    vqvae.train()
    while step < steps:
        lr = lr_at(step, float(T.lr), float(T.lr_floor), int(T.warmup), steps)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        agg = {}
        for k in range(accum):
            x = torch.from_numpy(train.windows(next(sampler))).to(device)
            with D.no_sync(model, k == accum - 1):
                loss, st = model(x)
                (loss / accum).backward()
            for key, v in st.items():
                agg[key] = agg.get(key, 0.0) + v / accum
            seen += x.numel() * D.world
        gn = torch.nn.utils.clip_grad_norm_(vqvae.parameters(), float(T.grad_clip))
        opt.step()
        if vq.defer_ema:
            vq.apply_deferred_ema()      # collective: every rank calls it once per step
        step += 1
        if step % int(T.log_every) == 0 or step == 1:
            agg["ce"] = D.mean(agg["ce"])
            log.log(step=step, lr=lr, grad_norm=float(gn), seen_tokens=seen,
                    **{key: (round(v, 4) if isinstance(v, float) else v) for key, v in agg.items()})
        if D.is_main and (step % int(T.eval_every) == 0 or step == steps):
            # fp32 on purpose: codeword search in bf16 selects different codes from the fp32 encoding the
            # generator is trained on, so the gate metrics would describe slightly different codes.
            ev = evaluate_tokenizer(vqvae, dev_batches(dev, int(T.eval_n), int(T.eval_bs), device), pad, chunk)
            vqvae.eval()
            xr = torch.from_numpy(dev.windows(range(min(512, len(dev))))).to(device)
            ev.update(decoder_robustness(vqvae, xr, chunk_rows=chunk))
            vqvae.train()
            log.log(step=step, **ev)
            if ev["recon_acc_full"] > best:
                best = ev["recon_acc_full"]
                ckpt.save("best.pt", state(vqvae, cfg, step, dev=ev))
        if ckpt.step_due(step, steps):
            ckpt.save("tokenizer.pt", state(vqvae, cfg, step))
        stop = D.any(pre.flag)
        if ckpt.step_due(step, steps) or ckpt.time_due() or stop:
            ckpt.save("last.pt", state(vqvae, cfg, step, optimizer=opt.state_dict(), best=best, seen_tokens=seen))
            ckpt.sync(wait=stop)
        if stop:
            log.log(step=step, event="preempted; resume with --resume auto")
            break

    log.finish(steps=step, seen_tokens=seen, best_recon_acc_full=best, world=D.world)
    ckpt.sync(wait=True)
    D.close()


if __name__ == "__main__":
    main()
