"""Unconditional Japanese generation quality: generative perplexity and unigram entropy.

Paper protocol (Section 4.2): seeds x samples, Gen. PPL under a reference AR model, mean
per-sample unigram entropy H over the generator's token ids, and the same two numbers on
held-out data for reference. GPT-2 is meaningless on Japanese, so the reference model is
``eval.ppl_model`` (default llm-jp/llm-jp-3-1.8b).

  python scripts/eval_uncond.py --model ncm --ckpt runs/phase_b/ja_uncond/best.pt
  python scripts/eval_uncond.py --model ar  --ckpt runs/phase_b/ja_uncond_ar/best.pt
  python scripts/eval_uncond.py --model data --config configs/phase_b/ja_uncond.yaml     # data reference row
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402
from jancm.loading import build_generator, build_tokenizer, default_device, load_config, load_state, resolve  # noqa: E402
from jancm.text import decode_target, encode_batch, pretokenizer_from_config  # noqa: E402
from jancm.train_utils import autocast, sync  # noqa: E402


def entropy(ids):
    c, m = Counter(ids), len(ids)
    return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ncm", "ar", "mdlm", "data"], required=True)
    ap.add_argument("--steps", type=int, default=128, help="MDLM sampling steps (NFE)")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--mdlm-weights", choices=["auto", "raw", "ema"], default="auto",
                    help="MDLM: auto = EMA once the average has warmed up (step*(1-decay) >= 5), else raw")
    ap.add_argument("--config", default="", help="default: config stored in the checkpoint")
    ap.add_argument("--n", type=int, default=256, help="samples per seed")
    ap.add_argument("--seeds", default="42,0,1,2,3")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--ppl-model", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    from omegaconf import OmegaConf

    device = a.device or default_device()
    st = load_state(a.ckpt) if a.ckpt else {}
    cfg = load_config(a.config) if a.config else OmegaConf.create(st["config"])
    if a.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    del st
    pretok = pretokenizer_from_config(cfg)
    eos, pad = pretok.eos_token_id, pretok.pad_token_id
    seeds = [int(s) for s in a.seeds.split(",")]

    if a.model == "ncm":
        from jancm.ncm_infer import sample_unconditional
        tk = build_tokenizer(cfg, resolve(cfg.checkpoints.tokenizer, ROOT), device)
        net = build_generator(cfg, tk, device, init_ckpt=a.ckpt).eval()

        def draw(n):
            return sample_unconditional(net, tk, n, cfg, device, batch_size=a.bs), [len(tk.vqvae.vq.levels)]
    elif a.model == "mdlm":
        from jancm.mdlm import load_mdlm_checkpoint, sample_mdlm_unconditional
        net, cfg, pretok = load_mdlm_checkpoint(a.ckpt, device, weights=a.mdlm_weights)
        if a.overrides:   # e.g. generation.temperature=0.9 (the checkpoint config replaced cfg above)
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
        print(f"[mdlm] sampling with {net.weights} weights, T={cfg.generation.get('temperature', 1.0)}", flush=True)
        T = float(cfg.generation.get("temperature", 1.0))

        def draw(n):
            return sample_mdlm_unconditional(net, cfg, pretok, n, device, a.bs, steps=a.steps, temperature=T)
    elif a.model == "ar":
        from jancm.ar_setup import load_ar_checkpoint, sample_ar_unconditional
        net, cfg, pretok = load_ar_checkpoint(a.ckpt, device)
        if a.overrides:   # e.g. generation.ar_temperature=0.9 (the checkpoint config replaced cfg above)
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
        T, P = float(cfg.generation.get("ar_temperature", 1.0)), float(cfg.generation.get("ar_top_p", 1.0))

        def draw(n):
            return sample_ar_unconditional(net, cfg, pretok, n, device, a.bs, temperature=T, top_p=P)
    else:
        from jancm.data import open_split
        dev = open_split(cfg.data, "dev", prefix_token=eos, root=ROOT)

        def draw(n):
            rows = np.random.default_rng(torch.initial_seed()).choice(len(dev), n, replace=False)
            cut = bool(cfg.generation.get("cut_at_eos", True))
            return [decode_target(pretok, w, eos, pad, cut) for w in dev.windows(rows).tolist()], [0]

    per, all_texts, wall = [], [], 0.0
    for seed in seeds:
        torch.manual_seed(seed)
        sync(device)
        t0 = time.perf_counter()
        with autocast(device, device == "cuda"):
            texts, nfes = draw(a.n)
        sync(device)
        wall += time.perf_counter() - t0
        per.append({"seed": seed, "texts": texts, "nfe": float(np.mean(nfes)),
                    "entropy": float(np.mean([entropy(i) for i in encode_batch(pretok, texts) if i])),
                    # character unigram entropy: comparable across text tokenizers (token entropy is not)
                    "char_entropy": float(np.mean([entropy(list(t)) for t in texts if t]))})
        all_texts += texts

    ref = a.ppl_model or str(cfg.eval.get("ppl_model", "llm-jp/llm-jp-3-1.8b"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.gen_ppl import generation_perplexity
    rtok = AutoTokenizer.from_pretrained(ref)
    rlm = AutoModelForCausalLM.from_pretrained(ref, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()
    for p in per:
        nonempty = [t for t in p["texts"] if t.strip()] or [" "]
        p["gen_ppl"] = float(generation_perplexity(rlm, rtok, [""] * len(nonempty), nonempty, device=device,
                                                   batch_size=16, max_length=512).mean_ppl)
        p["empty"] = len(p["texts"]) - len([t for t in p["texts"] if t.strip()])
    g = np.array([p["gen_ppl"] for p in per])
    empty_frac = sum(p["empty"] for p in per) / max(1, sum(len(p["texts"]) for p in per))
    if empty_frac > 0.01:
        print(f"[warn] {100 * empty_frac:.1f}% of samples are empty; gen-PPL covers only the non-empty ones", flush=True)
    res = {"model": a.model, "ckpt": a.ckpt, **({"weights": net.weights} if a.model == "mdlm" else {}), "ref_model": ref, "n_per_seed": a.n, "seeds": seeds,
           "gen_ppl_mean": round(float(g.mean()), 2), "gen_ppl_sd": round(float(g.std(ddof=1)) if len(g) > 1 else 0.0, 2),
           "entropy_mean": round(float(np.mean([p["entropy"] for p in per])), 4),
           "char_entropy_mean": round(float(np.mean([p["char_entropy"] for p in per])), 4),
           "empty_frac": round(empty_frac, 4), "nfe": per[0]["nfe"], "sampling_s_total": round(wall, 2),
           "per_seed": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items() if k != "texts"} for p in per],
           "examples": all_texts[:5]}
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, ensure_ascii=False))
    out = a.out or os.path.join(ROOT, "results", f"uncond_{a.model}_{os.path.basename(os.path.dirname(os.path.abspath(a.ckpt))) if a.ckpt else 'data'}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    with open(out.replace(".json", ".samples.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_texts) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
