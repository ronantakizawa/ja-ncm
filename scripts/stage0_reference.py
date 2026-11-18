"""Stage 0 (B0): score released or own LM1B checkpoints with the metrics our loops log.

Gate: our from-scratch generator's dev loss within ~5% of the released
generator's, and our tokenizer's reconstruction close to the released 99.86%.

  # reference numbers from the released checkpoints
  python scripts/stage0_reference.py
  # our runs (same metric, same dev windows)
  python scripts/stage0_reference.py --generator runs/stage0/lm1b_small/best.pt
  python scripts/stage0_reference.py --tokenizer runs/stage0/lm1b_tokenizer/tokenizer.pt --skip-generator

Dev loss = mean cross-entropy per code over all levels (alpha = 0, clean conditioning),
on the first --n LM1B validation windows. Writes results/stage0_<name>.json.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402
from jancm.data import open_split  # noqa: E402
from jancm.loading import build_generator, build_tokenizer, default_device, load_config, resolve  # noqa: E402
from jancm.ncm_objective import build_batch, level_weights, ncm_loss, per_level_mean, predicted_levels  # noqa: E402
from jancm.tokenizer_objective import evaluate_tokenizer  # noqa: E402
from jancm.train_utils import autocast  # noqa: E402


@torch.no_grad()
def generator_dev_loss(net, tk, dev, n, bs, device, use_bf16):
    net.eval()
    pad = tk.pre_tokenizer.pad_token_id
    levels = predicted_levels(net)
    w0 = level_weights(levels, 0.0, device)
    tot, per, m = 0.0, None, 0
    for s in range(0, min(n, len(dev)), bs):
        src, tgt = dev.take(range(s, min(s + bs, n, len(dev))), device)
        inputs, codes = build_batch(net, tk.vqvae, src, tgt, pad)
        with autocast(device, use_bf16):
            logits = net(**inputs)
        loss, ce = ncm_loss(logits, codes, w0)
        b = src.shape[0]
        pl = per_level_mean(ce, levels)
        tot += float(loss) * b
        per = [p * b for p in pl] if per is None else [x + p * b for x, p in zip(per, pl)]
        m += b
    return {"dev_loss": round(tot / m, 4), "dev_ce_per_level": [round(p / m, 3) for p in per], "n_windows": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "stage0", "lm1b_small.yaml"))
    ap.add_argument("--generator", default="", help="default: checkpoints.reference_generator")
    ap.add_argument("--tokenizer", default="", help="default: checkpoints.tokenizer (released)")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--skip-generator", action="store_true")
    ap.add_argument("--skip-tokenizer", action="store_true")
    ap.add_argument("--device", default="")
    ap.add_argument("--name", default="")
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = a.device or default_device()
    use_bf16 = device == "cuda"
    tok_path = a.tokenizer or resolve(cfg.checkpoints.tokenizer, ROOT)
    gen_path = a.generator or resolve(cfg.checkpoints.reference_generator, ROOT)
    tk = build_tokenizer(cfg, tok_path, device)
    dev = open_split(cfg.data, "dev", prefix_token=tk.pre_tokenizer.eos_token_id, root=ROOT)
    out = {"tokenizer": tok_path, "dev_file": dev.dir, "n": a.n}

    if not a.skip_tokenizer:
        def batches():
            for s in range(0, min(a.n, len(dev)), a.bs):
                yield torch.from_numpy(dev.windows(range(s, min(s + a.bs, a.n, len(dev))))).to(device)
        with autocast(device, use_bf16):
            out["tokenizer_eval"] = evaluate_tokenizer(tk.vqvae, batches(), tk.pre_tokenizer.pad_token_id)
        print(json.dumps({"tokenizer_eval": out["tokenizer_eval"]}))
    if not a.skip_generator:
        net = build_generator(cfg, tk, device, init_ckpt=gen_path)
        out["generator"] = gen_path
        out["generator_eval"] = generator_dev_loss(net, tk, dev, a.n, a.bs, device, use_bf16)
        print(json.dumps({"generator_eval": out["generator_eval"]}))

    name = a.name or ("released" if not (a.generator or a.tokenizer) else "own")
    path = os.path.join(ROOT, "results", f"stage0_{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
