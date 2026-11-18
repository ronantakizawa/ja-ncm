"""Measure how well the frozen English concept tokenizer reconstructs the ja->en targets.

Phase A freezes the WMT14 tokenizer, so this is the ceiling on achievable quality: if it
cannot reconstruct KFTT English (temple names, romanised terms), no generator can.
Reports per corpus: token accuracy over non-PAD positions after each level, exact-sequence
match at the full hierarchy, and per-level codebook utilisation.

  python scripts/check_tokenizer.py --split dev
  python scripts/check_tokenizer.py --split train --per-source 2000
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402
from jancm.data import PairSplit  # noqa: E402
from jancm.loading import build_tokenizer, default_device, load_config, resolve  # noqa: E402
from jancm.text import decode_target  # noqa: E402


@torch.no_grad()
def reconstruct(tk, tgt):
    """Return per-level argmax token ids [K, B, L] and the codes."""
    vq = tk.vqvae.vq
    codes = tk.vqvae.encode(tgt)
    (zh,) = vq.idx_to_zhats(codes, return_zlevels=False)
    return torch.stack([tk.vqvae.decoder(z).argmax(-1) for z in zh]), codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "ja_en_ncm.yaml"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--per-source", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--device", default="")
    ap.add_argument("--data-dir", default="", help="default: data.dir from the config")
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = a.device or default_device()
    tk = build_tokenizer(cfg, resolve(cfg.checkpoints.tokenizer, ROOT), device)
    pretok = tk.pre_tokenizer
    eos, pad = pretok.eos_token_id, pretok.pad_token_id
    vq = tk.vqvae.vq
    split = PairSplit(os.path.join(a.data_dir or resolve(cfg.data.dir, ROOT), a.split))
    rng = np.random.default_rng(0)
    report = {}

    for si, name in enumerate(split.names):
        rows = np.nonzero(split.source == si)[0]
        if not len(rows):
            continue
        rows = rng.choice(rows, min(len(rows), a.per_source), replace=False)
        correct = np.zeros(len(vq.levels))
        total, exact, n = 0, 0, 0
        used = [set() for _ in vq.levels]
        shown = []
        for s in range(0, len(rows), a.bs):
            _, tgt = split.take(rows[s:s + a.bs], device)
            rec, codes = reconstruct(tk, tgt)
            mask = tgt != pad
            total += int(mask.sum())
            for k in range(len(vq.levels)):
                correct[k] += int(((rec[k] == tgt) & mask).sum())
            exact += int(((rec[-1] == tgt) | ~mask).all(dim=1).sum())
            n += tgt.shape[0]
            off = 0
            for k, l in enumerate(vq.levels):
                used[k].update(codes[:, off:off + l].flatten().tolist())
                off += l
            for b in range(tgt.shape[0]):
                if len(shown) < a.examples and not bool(((rec[-1][b] == tgt[b]) | ~mask[b]).all()):
                    shown.append({"ref": decode_target(pretok, tgt[b].tolist(), eos, pad),
                                  "recon": decode_target(pretok, rec[-1][b].tolist(), eos, pad)})
        report[name] = {
            "n": n,
            "token_acc_per_level": [round(100 * c / total, 2) for c in correct],
            "token_acc_full": round(100 * correct[-1] / total, 3),
            "exact_sequence_pct": round(100 * exact / n, 2),
            "codebook_util_pct": [round(100 * len(u) / v, 1) for u, v in zip(used, vq.vqvae_vocab_sizes)],
            "failures": shown,
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False, indent=1))

    out = os.path.join(ROOT, "results", f"tokenizer_check_{os.path.basename(os.path.dirname(os.path.normpath(split.dir)))}_{a.split}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
