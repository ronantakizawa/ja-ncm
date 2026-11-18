"""Train the same-size AR baseline on the data prepared for NCM. Multi-GPU with torchrun.

  Phase A  torchrun --standalone --nproc_per_node 8 scripts/train_ar.py --config configs/ja_en_ar.yaml
  Phase B  torchrun ... scripts/train_ar.py --config configs/phase_b/ja_uncond_ar.yaml   # pre-train
           torchrun ... scripts/train_ar.py --config configs/phase_b/ja_summ_ar.yaml     # fine-tune (init_ar)
  Resume   add --resume auto
  Smoke    python scripts/train_ar.py --config ... --device cpu train.steps=3 train.batch_size=4

checkpoints.init_ar: optional AR checkpoint to start from (same recipe as NCM's init_generator).
"""
import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402
from jancm.ar import right_to_left  # noqa: E402
from jancm.ar_setup import build_ar, load_ar_config, sample_ar_unconditional, special_ids, translate_ar  # noqa: E402
from jancm.data import MixtureSampler, open_split, read_jsonl  # noqa: E402
from jancm.metrics_ja import bleu_tokenize_arg  # noqa: E402
from jancm.distributed import Dist, Preemption, resolve_batching, resolve_resume  # noqa: E402
from jancm.loading import load_state, resolve  # noqa: E402
from jancm.text import pretokenizer_from_config  # noqa: E402
from jancm.train_utils import Checkpointer, RunLog, adamw, autocast, lr_at  # noqa: E402


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


@torch.no_grad()
def evaluate(net, cfg, pretok, dev, dev_text, ids, device, use_bf16, n_bleu):
    import sacrebleu
    sep, eos, pad = ids
    net.eval()
    tot, n, bs = 0.0, 0, int(cfg.eval.batch_size)
    for s in range(0, min(len(dev), int(cfg.train.get("eval_loss_n", 4096))), bs):
        src, tgt = dev.take(range(s, min(s + bs, len(dev))), device)
        with autocast(device, use_bf16):
            loss = net(right_to_left(src, pad), tgt, sep, pad, pad)
        tot += float(loss) * src.shape[0]
        n += src.shape[0]
    out = {"dev_loss": tot / n}
    if n_bleu and dev_text:
        rows = dev_text[:n_bleu]
        with autocast(device, use_bf16):
            hyps = translate_ar(net, cfg, pretok, [r["src"] for r in rows], device, batch_size=bs)
        tok = bleu_tokenize_arg(str(cfg.eval.get("target_lang", "en")))
        out["dev_bleu"] = sacrebleu.corpus_bleu(hyps, [[r["ref"] for r in rows]], tokenize=tok).score
        out["dev_chrf"] = sacrebleu.corpus_chrf(hyps, [[r["ref"] for r in rows]]).score
        out["sample"] = f"{rows[0]['src'][:80]} => {hyps[0]}"
    n_samp = int(cfg.train.get("eval_samples", 0))
    if n_samp and str(cfg.data.get("kind", "pairs")) == "windows":
        with autocast(device, use_bf16):
            out["samples"] = [h[:120] for h in sample_ar_unconditional(net, cfg, pretok, n_samp, device)[0]]
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "ja_en_ar.yaml"))
    ap.add_argument("--resume", default="", help="checkpoint path, or 'auto' for <out_dir>/last.pt if present")
    ap.add_argument("--device", default="", help="force a device (e.g. cpu for smoke tests)")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()

    cfg = load_ar_config(a.config, a.overrides)
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
    ids = special_ids(cfg, pretok)
    sep, eos, pad = ids
    net = build_ar(cfg, device)
    net.ce_chunk_rows = int(T.get("ce_chunk_rows", 4096))
    init = cfg.checkpoints.get("init_ar", None)
    if init and not resume:
        net.load_state_dict(load_state(resolve(str(init), ROOT))["model"])

    train = open_split(cfg.data, "train", prefix_token=eos, root=ROOT)
    dev = open_split(cfg.data, "dev", prefix_token=eos, root=ROOT)
    dev_text = []
    if str(cfg.data.get("kind", "pairs")) == "pairs":
        dev_text = read_jsonl(os.path.join(dev.dir, "dev.jsonl"))
        assert train.meta["eos_id"] == eos and train.meta["pad_id"] == pad

    opt = adamw(net, float(T.lr), float(T.weight_decay), T.betas)
    step, best, seen = 0, float("inf"), 0
    best_sel = (float("-inf"), "")
    if resume:
        st = load_state(resume)
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        step, best, seen = int(st["step"]), float(st.get("best", best)), int(st.get("seen_tokens", 0))
        best_sel = tuple(st.get("best_sel", best_sel))
        del st
    model = D.wrap(net, find_unused=bool(T.get("ddp_find_unused", False)))
    # Distinct randomness per rank (dropout, noise): the global seed above only makes initialisation
    # reproducible; DDP broadcasts rank 0's weights, so reseeding here cannot desynchronise them.
    torch.manual_seed(int(T.seed) + 1_000_003 * D.rank + step)
    if bool(T.compile):
        model = torch.compile(model)
    if D.is_main:
        n_all = sum(p.numel() for p in net.parameters())
        print(f"[ar] world={D.world} global_batch={T.batch_size} micro={micro} accum={accum} step={step} "
              f"init={'resume ' + resume if resume else (init or 'scratch')} params={n_all / 1e6:.2f}M "
              f"non-embedding={(n_all - net.wte.weight.numel()) / 1e6:.2f}M", flush=True)
    mixture = dict(cfg.data.get("mixture", None) or {"corpus": 1.0})
    sampler = MixtureSampler(train, mixture, micro, seed=int(T.seed) + 1_000_003 * D.rank + step)
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
        tot = 0.0
        for k in range(accum):
            src, tgt = train.take(next(sampler), device)
            with D.no_sync(model, k == accum - 1):
                with autocast(device, use_bf16):
                    loss = model(right_to_left(src, pad), tgt, sep, pad, pad)
                (loss / accum).backward()
            tot += float(loss.detach()) / accum
            seen += (src.numel() + tgt.numel()) * D.world
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), float(T.grad_clip))
        opt.step()
        step += 1
        if step % int(T.log_every) == 0 or step == 1:
            log.log(step=step, loss=D.mean(tot), lr=lr, grad_norm=float(gn), seen_tokens=seen)
        if D.is_main and (step % int(T.eval_every) == 0 or step == steps):
            ev = evaluate(net, cfg, pretok, dev, dev_text, ids, device, use_bf16, int(T.eval_bleu_n))
            log.log(step=step, **ev)
            improved, cand = better(ev, best_sel, T)
            if improved:
                best_sel = cand
                best = ev["dev_loss"]
                ckpt.save("best.pt", state(net, cfg, step, dev=ev, selected_by=cand[1]))
        if D.is_main and int(T.get("save_every", 0)) and step % int(T.save_every) == 0:
            ckpt.save(f"ar_step{step}.pt", state(net, cfg, step))   # training-token curve
        stop = D.any(pre.flag)
        if ckpt.step_due(step, steps) or ckpt.time_due() or stop:
            ckpt.save("last.pt", state(net, cfg, step, optimizer=opt.state_dict(), best=best, best_sel=list(best_sel), seen_tokens=seen))
            ckpt.sync(wait=stop)
        if stop:
            log.log(step=step, event="preempted; resume with --resume auto")
            break

    log.finish(steps=step, seen_tokens=seen, best_dev_loss=best, world=D.world)
    ckpt.sync(wait=True)
    D.close()


if __name__ == "__main__":
    main()
