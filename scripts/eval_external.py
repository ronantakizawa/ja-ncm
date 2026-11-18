"""Score external Hugging Face seq2seq MT baselines through our pipeline (Phase A2).

Same test files, same sacreBLEU settings (13a BLEU, chrF), same timing protocol as
scripts/evaluate.py (batch 16, one warm-up batch excluded, device synchronised), so the
rows are directly comparable with NCM and the AR baseline. Each model is decoded twice:
  default : the model's own generation_config (beam 4 for ElanMT, 12 FuguMT, 6 opus-mt)
  greedy  : num_beams=1, the same search as our NCM and AR rows

  python scripts/eval_external.py --models Mitsua/elan-mt-bt-ja-en,Mitsua/elan-mt-tiny-ja-en

Writes results/ext_<model>_<decode>/<test>.json, .hyps.jsonl and .hyp.txt.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402
from jancm.data import load_test_set  # noqa: E402
from jancm.train_utils import sync  # noqa: E402

DEFAULT_MODELS = ("Mitsua/elan-mt-bt-ja-en,Mitsua/elan-mt-base-ja-en,Mitsua/elan-mt-tiny-ja-en,"
                  "staka/fugumt-ja-en,Helsinki-NLP/opus-mt-ja-en")


@torch.no_grad()
def translate(model, tok, texts, device, bs, num_beams, max_new):
    out, steps = [], []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        kw = {"max_new_tokens": max_new, "max_length": None}
        if num_beams is not None:
            kw["num_beams"] = num_beams
        gen = model.generate(**enc, **kw)
        out += tok.batch_decode(gen, skip_special_tokens=True)
        steps.append(gen.shape[1] - 1)          # decoder forward passes for this batch
    return out, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--tests", default="flores,ntrex,kftt,alt")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "processed", "test"))
    ap.add_argument("--decodes", default="default,greedy")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--device", default="")
    ap.add_argument("--lang", choices=["en", "ja"], default="en", help="output language (BLEU tokenizer, length units)")
    a = ap.parse_args()
    import sacrebleu
    from jancm.metrics_ja import bleu_tokenize_arg, length_units
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for mid in [m for m in a.models.split(",") if m]:
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForSeq2SeqLM.from_pretrained(mid).to(device).eval()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        for decode in a.decodes.split(","):
            beams = None if decode == "default" else 1
            eff_beams = model.generation_config.num_beams if beams is None else 1
            out_dir = os.path.join(ROOT, "results", f"ext_{mid.split('/')[-1]}_{decode}")
            os.makedirs(out_dir, exist_ok=True)
            for name in [t for t in a.tests.split(",") if t]:
                srcs, refs = load_test_set(os.path.join(a.data_dir, name))
                translate(model, tok, srcs[:a.bs], device, a.bs, beams, a.max_new)    # warm-up
                sync(device)
                t0 = time.perf_counter()
                hyps, steps = translate(model, tok, srcs, device, a.bs, beams, a.max_new)
                sync(device)
                wall = time.perf_counter() - t0
                bm, cm = sacrebleu.BLEU(tokenize=bleu_tokenize_arg(a.lang)), sacrebleu.CHRF()
                b, c = bm.corpus_score(hyps, [refs]), cm.corpus_score(hyps, [refs])
                res = {"model": mid, "decode": decode, "num_beams": eff_beams, "test": name, "n": len(srcs),
                       "bleu": round(b.score, 2), "chrf": round(c.score, 2),
                       "len_ratio": round(sum(length_units(h, a.lang) for h in hyps) / max(1, sum(length_units(r, a.lang) for r in refs)), 3),
                       "wall_s": round(wall, 3), "ms_per_sentence": round(1000 * wall / len(srcs), 3),
                       "batch_size": a.bs, "nfe_mean": round(sum(steps) / len(steps), 2), "precision": "fp32",
                       "device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else device,
                       "params_M": round(n_params, 2), "bleu_signature": str(bm.get_signature()),
                       "chrf_signature": str(cm.get_signature())}
                print(json.dumps(res, ensure_ascii=False), flush=True)
                with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
                with open(os.path.join(out_dir, f"{name}.hyps.jsonl"), "w", encoding="utf-8") as f:
                    for h in hyps:
                        f.write(json.dumps(h, ensure_ascii=False) + "\n")
                with open(os.path.join(out_dir, f"{name}.hyp.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(h.replace("\n", " ") for h in hyps) + "\n")
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
