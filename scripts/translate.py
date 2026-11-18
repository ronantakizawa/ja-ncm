"""Translate Japanese sentences with an NCM checkpoint; optionally show every level (paper Fig. 3).

  python scripts/translate.py --ckpt runs/ja_en_ncm/best.pt --levels "京都は日本の古都である。"
  echo "猫が好きです。" | python scripts/translate.py --ckpt runs/ja_en_ncm/best.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf  # noqa: E402

from jancm import ROOT  # noqa: E402
from jancm.loading import build_generator, build_tokenizer, default_device, load_config, load_state, resolve  # noqa: E402
from jancm.ncm_infer import translate_ncm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--levels", action="store_true", help="print the decoded output after each level")
    ap.add_argument("--device", default="")
    ap.add_argument("text", nargs="*")
    a = ap.parse_args()

    st = load_state(a.ckpt)
    cfg = load_config(a.config) if a.config else (
        OmegaConf.create(st["config"]) if "config" in st else load_config(os.path.join(ROOT, "configs", "ja_en_ncm.yaml")))
    del st
    device = a.device or default_device()
    tk = build_tokenizer(cfg, resolve(cfg.checkpoints.tokenizer, ROOT), device)
    net = build_generator(cfg, tk, device, init_ckpt=a.ckpt).eval()
    texts = a.text or [line.strip() for line in sys.stdin if line.strip()]
    hyps, levels = translate_ncm(net, tk, texts, cfg, device, return_levels=True)
    for t, h, lv in zip(texts, hyps, levels):
        print(f"src: {t}\nout: {h}")
        if a.levels:
            for k, s in enumerate(lv, 1):
                print(f"  level {k:2d} ({tk.vqvae.vq.levels[k - 1]:2d} codes): {s}")
        print()


if __name__ == "__main__":
    main()
