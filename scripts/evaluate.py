"""Score NCM, the AR baseline or the conditional MDLM on conditional test sets: BLEU, chrF, ROUGE, wall-clock, NFE.

  Phase A (ja->en, English output: sacreBLEU 13a)
    python scripts/evaluate.py --model ncm --ckpt runs/ja_en_ncm/best.pt --tests flores,ntrex,kftt
    python scripts/evaluate.py --model ar  --ckpt runs/ja_en_ar/best.pt  --tests flores,ntrex,kftt
  Phase B (Japanese output: BLEU ja-mecab, ROUGE on MeCab words and characters)
    python scripts/evaluate.py --model ncm --ckpt runs/phase_b/en_ja/best.pt --lang ja \
        --data-dir data/phase_b/en_ja/test --tests flores,ntrex,kftt
    python scripts/evaluate.py --model ncm --ckpt runs/phase_b/ja_summ_xlsum/best.pt --lang ja --rouge \
        --data-dir data/phase_b/summ/test --tests xlsum --bs 16

Timing follows the paper:
batch size 16 by default, one warm-up batch excluded, device synchronised around the loop.
Writes results/<model>_<run>/<test>.json and <test>.hyp.txt.
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
from jancm.loading import build_generator, build_tokenizer, default_device, load_config, load_state, resolve  # noqa: E402
from jancm.train_utils import autocast, sync  # noqa: E402


def make_ncm(a, device):
    from omegaconf import OmegaConf
    st = load_state(a.ckpt)
    if a.config:
        cfg = load_config(a.config)
    elif "config" in st:
        cfg = OmegaConf.create(st["config"])
    else:  # released upstream checkpoints carry no config
        cfg = load_config(os.path.join(ROOT, "configs", "ja_en_ncm.yaml"))
    del st
    if a.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    tk = build_tokenizer(cfg, resolve(cfg.checkpoints.tokenizer, ROOT), device)
    net = build_generator(cfg, tk, device, init_ckpt=a.ckpt).eval()
    from jancm.ncm_infer import n_truncated, translate_ncm

    def run(texts, bs):
        return translate_ncm(net, tk, texts, cfg, device, batch_size=bs), [len(tk.vqvae.vq.levels)]

    n_params = sum(p.numel() for p in net.parameters())
    tok_params = sum(p.numel() for p in tk.vqvae.parameters())
    info = {"generator_params_M": round(n_params / 1e6, 2), "tokenizer_params_M": round(tok_params / 1e6, 2)}
    return run, cfg, (lambda texts: n_truncated(tk, texts, int(cfg.model.prefix_len))), info


def make_ar(a, device):
    from jancm.ar_setup import load_ar_checkpoint, translate_ar
    from jancm.text import encode_batch
    net, cfg, pretok = load_ar_checkpoint(a.ckpt, device)
    if a.overrides:
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    net.eval()

    def run(texts, bs):
        return translate_ar(net, cfg, pretok, texts, device, batch_size=bs, return_nfe=True)

    P = int(cfg.model.prefix_len)
    n_all = sum(p.numel() for p in net.parameters())
    info = {"params_M": round(n_all / 1e6, 2), "non_embedding_params_M": round((n_all - net.wte.weight.numel()) / 1e6, 2)}
    return run, cfg, (lambda texts: sum(len(i) > P for i in encode_batch(pretok, texts))), info


def make_mdlm(a, device):
    """Conditional prefix MDLM (scripts/train_mdlm.py with data.kind: pairs). Raw weights, as
    the AR and NCM fine-tunes. Steps / order / temperature via overrides, e.g.
    generation.mdlm_cond_steps=32 generation.mdlm_order=confidence."""
    from omegaconf import OmegaConf
    from jancm.mdlm import load_mdlm_checkpoint, translate_mdlm
    from jancm.text import encode_batch
    net, cfg, pretok = load_mdlm_checkpoint(a.ckpt, device, weights="raw")
    if a.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    net.eval()

    def run(texts, bs):
        return translate_mdlm(net, cfg, pretok, texts, device, batch_size=bs, return_nfe=True)

    P = int(cfg.model.prefix_len)
    n_all = sum(p.numel() for p in net.parameters())
    G = cfg.generation
    info = {"params_M": round(n_all / 1e6, 2), "non_embedding_params_M": round((n_all - net.wte.weight.numel()) / 1e6, 2),
            "mdlm_cond_steps": int(G.get("mdlm_cond_steps", 32)), "mdlm_order": str(G.get("mdlm_order", "random")),
            "mdlm_temperature": float(G.get("mdlm_temperature", 0.0))}
    return run, cfg, (lambda texts: sum(len(i) > P for i in encode_batch(pretok, texts))), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ncm", "ar", "mdlm"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="", help="NCM only; default: config stored in the checkpoint")
    ap.add_argument("--tests", default="flores,ntrex,kftt,alt")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "processed", "test"))
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--lang", choices=["en", "ja"], default="en", help="output language (metric tokenisation)")
    ap.add_argument("--rouge", action="store_true", help="also report ROUGE-1/2/L (summarisation)")
    ap.add_argument("overrides", nargs="*", help="dotlist, e.g. generation.use_kv_cache=false")
    a = ap.parse_args()
    import sacrebleu
    from jancm.metrics_ja import bleu_tokenize_arg, length_units, rouge_scores

    device = a.device or default_device()
    run, cfg, count_trunc, info = {"ncm": make_ncm, "ar": make_ar, "mdlm": make_mdlm}[a.model](a, device)
    run_name = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    out_dir = a.out or os.path.join(ROOT, "results", f"{a.model}_{run_name}")
    os.makedirs(out_dir, exist_ok=True)
    use_bf16 = not a.no_bf16

    for name in [t for t in a.tests.split(",") if t]:
        d = os.path.join(a.data_dir, name)
        if not os.path.exists(os.path.join(d, "test.jsonl")):
            print(f"[skip] {name}: no {d}/test.jsonl (run prepare_data.py)")
            continue
        srcs, refs = load_test_set(d)
        if a.n:
            srcs, refs = srcs[:a.n], refs[:a.n]
        with autocast(device, use_bf16):
            run(srcs[:a.bs], a.bs)                      # warm-up (kernels, allocator)
            sync(device)
            t0 = time.perf_counter()
            hyps, nfes = run(srcs, a.bs)
            sync(device)
            wall = time.perf_counter() - t0
        bleu_m, chrf_m = sacrebleu.BLEU(tokenize=bleu_tokenize_arg(a.lang)), sacrebleu.CHRF()
        bleu, chrf = bleu_m.corpus_score(hyps, [refs]), chrf_m.corpus_score(hyps, [refs])
        res = {
            "model": a.model, "ckpt": a.ckpt, "test": name, "n": len(srcs),
            "bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2),
            "len_ratio": round(sum(length_units(h, a.lang) for h in hyps) / max(1, sum(length_units(r, a.lang) for r in refs)), 3),
            "mean_len_" + ("chars" if a.lang == "ja" else "words"): round(sum(length_units(h, a.lang) for h in hyps) / len(hyps), 1),
            "wall_s": round(wall, 3), "ms_per_sentence": round(1000 * wall / len(srcs), 3),
            "batch_size": a.bs, "nfe_mean": round(sum(nfes) / len(nfes), 2),
            "src_truncated": count_trunc(srcs), "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else device,
            "bleu_signature": str(bleu_m.get_signature()), "chrf_signature": str(chrf_m.get_signature()),
            **info,
        }
        if a.rouge:
            res["rouge"] = {k: {m: round(v, 2) for m, v in d.items()} for k, d in rouge_scores(hyps, refs, a.lang).items()}
        print(json.dumps(res, ensure_ascii=False))
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        # .hyps.jsonl is the exact, line-aligned record (outputs may contain newlines);
        # .hyp.txt is for reading, with internal newlines flattened to spaces.
        with open(os.path.join(out_dir, f"{name}.hyps.jsonl"), "w", encoding="utf-8") as f:
            for h in hyps:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, f"{name}.hyp.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(h.replace("\n", " ") for h in hyps) + "\n")


if __name__ == "__main__":
    main()
