"""Train the same-size masked diffusion baseline (jancm/mdlm.py) on the NCM corpus windows,
or (data.kind: pairs) fine-tune it as a conditional prefix MDLM for en->ja / summarisation,
initialised from checkpoints.init_mdlm (the unconditional run's EMA weights).

  torchrun --standalone --nproc_per_node 8 scripts/train_mdlm.py --config configs/phase_b/ja_uncond_mdlm.yaml
  Resume  add --resume auto
  Smoke   python scripts/train_mdlm.py --config ... --device cpu train.steps=3 train.batch_size=4 ...

Saves best.pt / last.pt / mdlm_stepN.pt with the raw weights ("model") and an EMA copy
("ema", decay train.ema, MDLM uses 0.9999). Unconditional runs evaluate with the EMA weights.
Conditional runs evaluate and select best.pt by dev BLEU with the raw weights, as the AR and
NCM fine-tunes do (they keep no EMA, and a 0.9999 EMA barely moves in a short fine-tune).
"""
import argparse
import copy
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402
from jancm.ar import right_to_left  # noqa: E402
from jancm.data import MixtureSampler, open_split, read_jsonl  # noqa: E402
from jancm.distributed import Dist, Preemption, resolve_batching, resolve_resume  # noqa: E402
from jancm.loading import load_config, load_state, resolve  # noqa: E402
from jancm.mdlm import build_mdlm, sample_mdlm_unconditional, translate_mdlm  # noqa: E402
from jancm.text import pretokenizer_from_config  # noqa: E402
from jancm.train_utils import Checkpointer, RunLog, adamw, autocast, lr_at  # noqa: E402


def state(net, ema, cfg, step, **extra):
    return {"model": net.state_dict(), "ema": ema.state_dict(), "step": step,
            "config": OmegaConf.to_container(cfg, resolve=True), **extra}


@torch.no_grad()
def evaluate(ema, cfg, pretok, dev, device, use_bf16):
    """Dev NELBO per token on a fixed t grid (deterministic), plus a few samples."""
    ema.eval()
    g = torch.Generator(device=device).manual_seed(0)
    tot, n, bs = 0.0, 0, int(cfg.eval.get("batch_size", 64))
    for s in range(0, min(len(dev), int(cfg.train.get("eval_loss_n", 4096))), bs):
        _, x0 = dev.take(range(s, min(s + bs, len(dev))), device)
        t = ema.eps + (1 - ema.eps) * (torch.arange(x0.shape[0], device=device) + 0.5) / x0.shape[0]
        with autocast(device, use_bf16):
            tot += float(ema(x0, t=t, generator=g)) * x0.shape[0]
        n += x0.shape[0]
    out = {"dev_nelbo": tot / n}
    k = int(cfg.train.get("eval_samples", 0))
    if k:
        with autocast(device, use_bf16):
            out["samples"] = [h[:120] for h in sample_mdlm_unconditional(ema, cfg, pretok, k, device, steps=64)[0]]
    return out


@torch.no_grad()
def evaluate_cond(net, cfg, pretok, dev, dev_text, device, use_bf16):
    """Conditional: dev NELBO per target token on a fixed t grid, and dev BLEU (raw weights)."""
    import sacrebleu
    from jancm.metrics_ja import bleu_tokenize_arg
    pad = pretok.pad_token_id
    net.eval()
    g = torch.Generator(device=device).manual_seed(0)
    tot, n, bs = 0.0, 0, int(cfg.eval.get("batch_size", 64))
    for s in range(0, min(len(dev), int(cfg.train.get("eval_loss_n", 4096))), bs):
        src, tgt = dev.take(range(s, min(s + bs, len(dev))), device)
        t = net.eps + (1 - net.eps) * (torch.arange(tgt.shape[0], device=device) + 0.5) / tgt.shape[0]
        with autocast(device, use_bf16):
            tot += float(net(tgt, t=t, generator=g, src_left=right_to_left(src, pad), src_pad_id=pad)) * tgt.shape[0]
        n += tgt.shape[0]
    out = {"dev_nelbo": tot / n}
    k = int(cfg.train.get("eval_bleu_n", 0))
    if k and dev_text:
        rows = dev_text[:k]
        with autocast(device, use_bf16):
            hyps = translate_mdlm(net, cfg, pretok, [r["src"] for r in rows], device, batch_size=bs)
        tok = bleu_tokenize_arg(str(cfg.eval.get("target_lang", "ja")))
        out["dev_bleu"] = sacrebleu.corpus_bleu(hyps, [[r["ref"] for r in rows]], tokenize=tok).score
        out["sample"] = f"{rows[0]['src'][:80]} => {hyps[0][:120]}"
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    T = cfg.train
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
    eos, pad = pretok.eos_token_id, pretok.pad_token_id
    cond = str(cfg.data.get("kind", "windows")) == "pairs"
    net = build_mdlm(cfg, device)
    net.ce_chunk_rows = int(T.get("ce_chunk_rows", 4096))
    init = cfg.checkpoints.get("init_mdlm", None)
    if init and not resume:
        ist = load_state(resolve(str(init), ROOT))
        w = str(cfg.checkpoints.get("init_mdlm_weights", "ema"))   # ema: a pre-trained run; raw: a fine-tune
        net.load_state_dict(ist["ema"] if (w == "ema" and "ema" in ist) else ist["model"])
        del ist
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    decay = float(T.get("ema", 0.9999))

    train = open_split(cfg.data, "train", prefix_token=eos, root=ROOT)
    dev = open_split(cfg.data, "dev", prefix_token=eos, root=ROOT)
    dev_text = read_jsonl(os.path.join(dev.dir, "dev.jsonl")) if cond else []
    if cond:
        assert train.meta["eos_id"] == eos and train.meta["pad_id"] == pad
    best_bleu = float("-inf")
    opt = adamw(net, float(T.lr), float(T.weight_decay), T.betas)
    step, best, seen = 0, float("inf"), 0
    if resume:
        st = load_state(resume)
        net.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["optimizer"])
        step, best, seen = int(st["step"]), float(st.get("best", best)), int(st.get("seen_tokens", 0))
        best_bleu = float(st.get("best_bleu", best_bleu))
        del st
    model = D.wrap(net)
    # Distinct randomness per rank (dropout, noise): the global seed above only makes initialisation
    # reproducible; DDP broadcasts rank 0's weights, so reseeding here cannot desynchronise them.
    torch.manual_seed(int(T.seed) + 1_000_003 * D.rank + step)
    if D.is_main:
        n_all = sum(p.numel() for p in net.parameters())
        print(f"[mdlm] world={D.world} global_batch={T.batch_size} micro={micro} accum={accum} step={step} "
              f"params={n_all / 1e6:.2f}M non-embedding={(n_all - net.wte.weight.numel()) / 1e6:.2f}M "
              f"ema={decay} cond={cond} init={init if init and not resume else ('resume' if resume else 'scratch')}", flush=True)
    mixture = dict(cfg.data.get("mixture", None) or {"corpus": 1.0})
    sampler = MixtureSampler(train, mixture, micro, seed=int(T.seed) + 1_000_003 * D.rank + step)
    gen = torch.Generator(device=device).manual_seed(int(T.seed) + 1_000_003 * D.rank + step)   # per-rank mask noise
    log = RunLog(out_dir, n_gpu=D.world, enabled=D.is_main)
    ckpt = Checkpointer(out_dir, T, enabled=D.is_main)
    if D.is_main:
        OmegaConf.save(cfg, os.path.join(out_dir, "config.yaml"))
    pre = Preemption()
    steps = int(T.steps)
    ema_params = [p for p in ema.parameters()]
    net_params = [p for p in net.parameters()]

    net.train()
    while step < steps:
        lr = lr_at(step, float(T.lr), float(T.lr_floor), int(T.warmup), steps)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for k in range(accum):
            src, x0 = train.take(next(sampler), device)
            with D.no_sync(model, k == accum - 1):
                with autocast(device, use_bf16):
                    if cond:
                        loss = model(x0, generator=gen, src_left=right_to_left(src, pad), src_pad_id=pad)
                    else:
                        loss = model(x0, generator=gen)
                (loss / accum).backward()
            tot += float(loss.detach()) / accum
            seen += (x0.numel() + (src.numel() if cond else 0)) * D.world
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), float(T.grad_clip))
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, net_params, 1.0 - decay)
        step += 1
        if step % int(T.log_every) == 0 or step == 1:
            log.log(step=step, loss=D.mean(tot), lr=lr, grad_norm=float(gn), seen_tokens=seen)
        if D.is_main and (step % int(T.eval_every) == 0 or step == steps):
            if cond:
                ev = evaluate_cond(net, cfg, pretok, dev, dev_text, device, use_bf16)
                log.log(step=step, **ev)
                score = ev.get("dev_bleu", -ev["dev_nelbo"])
                if score > best_bleu:
                    best_bleu, best = score, ev["dev_nelbo"]
                    ckpt.save("best.pt", state(net, ema, cfg, step, dev=ev))
            else:
                ev = evaluate(ema, cfg, pretok, dev, device, use_bf16)
                log.log(step=step, **ev)
                if ev["dev_nelbo"] < best:
                    best = ev["dev_nelbo"]
                    ckpt.save("best.pt", state(net, ema, cfg, step, dev=ev))
        if D.is_main and int(T.get("save_every", 0)) and step % int(T.save_every) == 0:
            # Keep raw weights too: before step*(1-decay) >= 5 the EMA is dominated by early
            # weights, and load_mdlm_checkpoint falls back to "model" for those points.
            ckpt.save(f"mdlm_step{step}.pt", {"model": net.state_dict(), "ema": ema.state_dict(),
                                               "step": step,
                                               "config": OmegaConf.to_container(cfg, resolve=True)})
        stop = D.any(pre.flag)
        if ckpt.step_due(step, steps) or ckpt.time_due() or stop:
            ckpt.save("last.pt", state(net, ema, cfg, step, optimizer=opt.state_dict(), best=best,
                                         best_bleu=best_bleu, seen_tokens=seen))
            ckpt.sync(wait=stop)
        if stop:
            log.log(step=step, event="preempted; resume with --resume auto")
            break
    log.finish(steps=step, seen_tokens=seen, best_dev_nelbo=best, world=D.world)
    ckpt.sync(wait=True)
    D.close()


if __name__ == "__main__":
    main()
