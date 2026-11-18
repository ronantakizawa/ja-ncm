"""Train or fine-tune an NCM generator on top of a frozen concept tokenizer.

Multi-GPU with torchrun (DDP); also runs as one process for CPU smoke tests.

  Phase A  torchrun --standalone --nproc_per_node 8 scripts/train_ncm.py --config configs/ja_en_ncm.yaml
  Phase B  torchrun ... scripts/train_ncm.py --config configs/phase_b/ja_uncond.yaml     # unconditional pre-training
           torchrun ... scripts/train_ncm.py --config configs/phase_b/ja_summ.yaml       # fine-tune from it
           torchrun ... scripts/train_ncm.py --config configs/phase_b/en_ja.yaml
  Stage 0  torchrun ... scripts/train_ncm.py --config configs/stage0/lm1b_small.yaml
  Resume   add --resume auto (picks up <out_dir>/last.pt after a preemption)
  Smoke    python scripts/train_ncm.py --config ... --device cpu train.steps=3 train.batch_size=4

train.batch_size is the global batch; train.micro_batch_size the per-GPU batch (gradient
accumulation makes up the difference). data.kind: pairs or windows (unconditional).
checkpoints.init_generator: optional; may come from a run with a different prefix_len.
Writes to train.out_dir: metrics.jsonl, run_stats.json (GPU-hours), last.pt (resume state,
every train.save_every_minutes), best.pt, generator_stepN.pt ({"model": state_dict}).
"""
import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402
from jancm.data import MixtureSampler, open_split, read_jsonl  # noqa: E402
from jancm.distributed import Dist, Preemption, resolve_batching, resolve_resume  # noqa: E402
from jancm.loading import build_generator, build_tokenizer, load_config, load_state, resolve  # noqa: E402
from jancm.ncm_infer import sample_unconditional, translate_ncm  # noqa: E402
from jancm.ncm_objective import Corruptor, build_batch, level_weights, ncm_loss, per_level_mean, predicted_levels  # noqa: E402
from jancm.train_utils import Checkpointer, RunLog, adamw, autocast, lr_at  # noqa: E402


def _bleu(hyps, refs, lang):
    import sacrebleu
    from jancm.metrics_ja import bleu_tokenize_arg
    return (sacrebleu.corpus_bleu(hyps, [refs], tokenize=bleu_tokenize_arg(lang)).score,
            sacrebleu.corpus_chrf(hyps, [refs]).score)


@torch.no_grad()
def evaluate(net, tk, dev, dev_text, cfg, device, use_bf16):
    """Rank 0 only. Dev loss is always the unweighted mean CE per code (alpha = 0), so runs
    with different loss weightings, and the released checkpoints, are comparable."""
    net.eval()
    T = cfg.train
    pad = tk.pre_tokenizer.pad_token_id
    levels = predicted_levels(net)
    w0 = level_weights(levels, 0.0, device)
    tot, per, n = 0.0, None, 0
    bs = int(cfg.eval.batch_size)
    for s in range(0, min(len(dev), int(T.get("eval_loss_n", 4096))), bs):
        src, tgt = dev.take(range(s, min(s + bs, len(dev))), device)
        inputs, codes = build_batch(net, tk.vqvae, src, tgt, pad)
        with autocast(device, use_bf16):
            logits = net(**inputs)
        loss, ce = ncm_loss(logits, codes, w0)
        pl = per_level_mean(ce, levels)
        b = src.shape[0]
        tot += float(loss) * b
        per = [p * b for p in pl] if per is None else [x + p * b for x, p in zip(per, pl)]
        n += b
    out = {"dev_loss": tot / n, "dev_ce_per_level": [round(p / n, 3) for p in per]}
    n_bleu = int(T.get("eval_bleu_n", 0))
    if n_bleu and dev_text:
        rows = dev_text[:n_bleu]
        with autocast(device, use_bf16):
            hyps = translate_ncm(net, tk, [r["src"] for r in rows], cfg, device, batch_size=bs)
        out["dev_bleu"], out["dev_chrf"] = _bleu(hyps, [r["ref"] for r in rows], str(cfg.eval.get("target_lang", "en")))
        out["sample"] = f"{rows[0]['src'][:80]} => {hyps[0]}"
    n_samp = int(T.get("eval_samples", 0))
    if n_samp and str(cfg.data.get("kind", "pairs")) == "windows":
        with autocast(device, use_bf16):
            out["samples"] = [h[:120] for h in sample_unconditional(net, tk, n_samp, cfg, device)]
    net.train()
    return out


def better(ev, best, T):
    """Checkpoint selection. dev_bleu (higher) when the eval produced it, else dev_loss
    (lower). Override with train.select_by: dev_loss | dev_bleu. Dev loss is a poor proxy
    for the AR baseline: on Phase A it bottomed out at step 8k while dev BLEU kept rising
    from 13.0 to 16.4 at step 30k."""
    key = str(T.get("select_by", "auto"))
    if key == "auto":
        key = "dev_bleu" if "dev_bleu" in ev else "dev_loss"
    score = ev[key] if key == "dev_bleu" else -ev[key]
    return (score > best[0], (score, key))


def state(net, cfg, step, **extra):
    return {"model": net.state_dict(), "step": step, "config": OmegaConf.to_container(cfg, resolve=True), **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "ja_en_ncm.yaml"))
    ap.add_argument("--resume", default="", help="checkpoint path, or 'auto' for <out_dir>/last.pt if present")
    ap.add_argument("--device", default="", help="force a device (e.g. cpu for smoke tests)")
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist, e.g. train.steps=100")
    a = ap.parse_args()

    cfg = load_config(a.config, a.overrides)
    T = cfg.train
    D = Dist(a.device)
    device = D.device
    torch.manual_seed(int(T.seed))
    use_bf16 = bool(T.bf16) and D.is_cuda
    if D.is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out_dir = resolve(T.out_dir, ROOT)
    resume = resolve_resume(a.resume, out_dir)
    micro, accum = resolve_batching(T, D.world)

    tk = build_tokenizer(cfg, resolve(cfg.checkpoints.tokenizer, ROOT), device)
    init = cfg.checkpoints.get("init_generator", None)
    init = None if (resume or not init) else resolve(str(init), ROOT)
    net = build_generator(cfg, tk, device, init_ckpt=init)
    vq = tk.vqvae.vq
    pretok = tk.pre_tokenizer
    pad = pretok.pad_token_id

    train = open_split(cfg.data, "train", prefix_token=pretok.eos_token_id, root=ROOT)
    dev = open_split(cfg.data, "dev", prefix_token=pretok.eos_token_id, root=ROOT)
    dev_text = []
    if str(cfg.data.get("kind", "pairs")) == "pairs":
        dev_text = read_jsonl(os.path.join(dev.dir, "dev.jsonl"))
        assert train.meta["prefix_len"] == cfg.model.prefix_len, "data prepared with a different prefix_len"
        assert train.meta["target_len"] == tk.vqvae.context_length, "data prepared with a different target window"
    else:
        assert train.window == tk.vqvae.context_length, "corpus windows differ from the tokenizer context"

    levels = predicted_levels(net)
    weights = level_weights(levels, float(T.loss_alpha), device)
    corrupt = Corruptor.from_config(len(vq.levels), T)

    opt = adamw(net, float(T.lr), float(T.weight_decay), T.betas)
    step, best, seen_tokens = 0, float("inf"), 0
    best_sel = (float("-inf"), "")
    if resume:
        st = load_state(resume)
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        step, best, seen_tokens = int(st["step"]), float(st.get("best", best)), int(st.get("seen_tokens", 0))
        best_sel = tuple(st.get("best_sel", best_sel))
        del st
    model = D.wrap(net, find_unused=bool(T.get("ddp_find_unused", False)))
    # Distinct randomness per rank (dropout, noise): the global seed above only makes initialisation
    # reproducible; DDP broadcasts rank 0's weights, so reseeding here cannot desynchronise them.
    torch.manual_seed(int(T.seed) + 1_000_003 * D.rank + step)
    if bool(T.compile):
        model = torch.compile(model)
    if D.is_main:
        print(f"[ncm] world={D.world} global_batch={T.batch_size} micro={micro} accum={accum} "
              f"init={'resume ' + resume if resume else (init or 'scratch')} step={step} "
              f"params={sum(p.numel() for p in net.parameters()) / 1e6:.2f}M", flush=True)
        print(f"[ncm] levels={vq.levels} corruption={corrupt.describe()} train={len(train)} dev={len(dev)}", flush=True)

    # Every rank samples its own rows; the seed depends on rank and step, so a resumed
    # run continues with fresh, reproducible draws.
    mixture = dict(cfg.data.get("mixture", None) or {"corpus": 1.0})
    seed = int(T.seed) + 1_000_003 * D.rank + step
    sampler = MixtureSampler(train, mixture, micro, seed=seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    log = RunLog(out_dir, n_gpu=D.world, enabled=D.is_main)
    ckpt = Checkpointer(out_dir, T, enabled=D.is_main)
    if D.is_main:
        OmegaConf.save(cfg, os.path.join(out_dir, "config.yaml"))
    pre = Preemption()
    steps = int(T.steps)

    net.train()
    while step < steps:
        lr = lr_at(step, float(T.lr), float(T.lr_floor), int(T.warmup), steps)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tot, ce_acc = 0.0, None
        for k in range(accum):
            src, tgt = train.take(next(sampler), device)
            inputs, codes = build_batch(net, tk.vqvae, src, tgt, pad, corrupt=corrupt, generator=gen)
            with D.no_sync(model, k == accum - 1):
                with autocast(device, use_bf16):
                    logits = model(**inputs)
                loss, ce = ncm_loss(logits, codes, weights)
                (loss / accum).backward()
            tot += float(loss.detach()) / accum
            ce_acc = ce if ce_acc is None else ce_acc + ce
            seen_tokens += (src.numel() + tgt.numel()) * D.world
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), float(T.grad_clip))
        opt.step()
        step += 1

        if step % int(T.log_every) == 0 or step == 1:
            loss_mean = D.mean(tot)
            log.log(step=step, loss=loss_mean, lr=lr, grad_norm=float(gn),
                    ce_per_level=[round(c, 3) for c in per_level_mean(ce_acc / accum, levels)],
                    seen_tokens=seen_tokens)
        if D.is_main and (step % int(T.eval_every) == 0 or step == steps):
            ev = evaluate(net, tk, dev, dev_text, cfg, device, use_bf16)
            log.log(step=step, **ev)
            improved, cand = better(ev, best_sel, T)
            if improved:
                best_sel = cand
                best = ev["dev_loss"]
                ckpt.save("best.pt", state(net, cfg, step, dev=ev, selected_by=cand[1]))
        if ckpt.step_due(step, steps):
            ckpt.save(f"generator_step{step}.pt", state(net, cfg, step))
        stop = D.any(pre.flag)
        if ckpt.step_due(step, steps) or ckpt.time_due() or stop:
            ckpt.save("last.pt", state(net, cfg, step, optimizer=opt.state_dict(), best=best, best_sel=list(best_sel), seen_tokens=seen_tokens))
            ckpt.sync(wait=stop)
        if stop:
            log.log(step=step, event="preempted; resume with --resume auto")
            break

    log.finish(steps=step, seen_tokens=seen_tokens, best_dev_loss=best, world=D.world)
    ckpt.sync(wait=True)
    D.close()


if __name__ == "__main__":
    main()
