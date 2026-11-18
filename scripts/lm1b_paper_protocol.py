"""Score LM1B models with the paper's released protocol, so our MDLM can be placed on the paper's scale.

The paper (and its released pipeline, ncm/scripts/eval_shard.py + eval_merge.py with
artifacts/lm1b/small/config.yaml: eval.metric genppl_flm, ref_model gpt2-large) reports LM1B
Gen. PPL as follows: 5 seeds x 256 samples; each 128-token window decoded with the BERT
tokenizer keeping special tokens; GPT-2-large scores the first 128 GPT-2 tokens; corpus-level
perplexity per seed (token-weighted), averaged over seeds; entropy = mean per-sample unigram
entropy over the raw window ids. Paper, Figure 7: NCM 93.1, MDLM 115.5 (66B training tokens),
DUO 97.3. This differs from scripts/gates.py stage0 (per-sample mean PPL over decoded text,
specials dropped), which scored the released NCM 136.8 vs the paper's 93.05.

NCM samples come from the released eval_shard.py unchanged (its read-out penalties and
level-0 unigram included). This script draws the MDLM and data windows and scores everything
with a copy of eval_merge.genppl_flm.

  python scripts/lm1b_paper_protocol.py gen-mdlm --ckpt runs/stage0/lm1b_mdlm/last.pt --steps 128 \
      --temperature 1.0 --out results/lm1b_paper/mdlm_S128_T1.0.pkl
  python scripts/lm1b_paper_protocol.py gen-data --out results/lm1b_paper/data.pkl
  python scripts/lm1b_paper_protocol.py score --dir results/lm1b_paper --out results/lm1b_paper/scores.json
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402,F401

SEEDS = [42, 0, 1, 2, 3]
N = 256
L = 128


def gen_mdlm(a):
    from jancm.mdlm import load_mdlm_checkpoint
    net, cfg, _ = load_mdlm_checkpoint(a.ckpt, "cuda")
    net.eval()
    raw, seeds = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        for s in range(0, N, a.bs):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ids, _ = net.generate(min(a.bs, N - s), L, a.steps, "cuda", temperature=a.temperature)
            raw += ids.tolist()
            seeds += [seed] * ids.shape[0]
    pickle.dump({"raw_ids": raw, "seeds": seeds, "weights": net.weights, "steps": a.steps,
                 "temperature": a.temperature}, open(a.out, "wb"))
    print(f"-> {a.out} ({len(raw)} windows, {net.weights} weights)")


def gen_data(a):
    from jancm.data import WindowSplit
    dev = WindowSplit(a.valid, window=L, dtype="uint16")
    rng = np.random.default_rng(0)
    rows = rng.choice(len(dev), N * len(SEEDS), replace=False)
    raw = dev.windows(rows).tolist()
    pickle.dump({"raw_ids": raw, "seeds": [s for s in SEEDS for _ in range(N)]}, open(a.out, "wb"))
    print(f"-> {a.out} ({len(raw)} windows)")


def score(a):
    """eval_merge.genppl_flm, applied to every pickle in --dir (merging eval_shard shards by prefix)."""
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    gen_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rtok = AutoTokenizer.from_pretrained(a.ref_model)
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    rlm = AutoModelForCausalLM.from_pretrained(a.ref_model).to(dev).eval()

    def corpus_ppl(texts):
        enc = rtok(texts, return_tensors="pt", return_token_type_ids=False, truncation=True, padding=True, max_length=128)
        ids, attn = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        nll_sum = tok_sum = 0.0
        with torch.no_grad():
            for i in range(0, ids.shape[0], 16):
                s_, m_ = ids[i:i + 16], attn[i:i + 16]
                logits = rlm(s_, attention_mask=m_)[0].transpose(-1, -2)
                nll = F.cross_entropy(logits[..., :-1], s_[..., 1:], reduction="none")
                first_eos = (s_ == rtok.eos_token_id).cumsum(-1) == 1
                valid = (first_eos[..., 1:] + (s_ != rtok.eos_token_id)[..., 1:]).to(nll.dtype)
                nll_sum += float((nll * valid).sum())
                tok_sum += float(valid.sum())
        return float(np.exp(nll_sum / tok_sum))

    def entropy(ids):
        t = torch.tensor(ids)
        return float(torch.special.entr(torch.unique(t, return_counts=True)[1].float() / t.numel()).sum())

    groups = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*.pkl"))):
        name = os.path.basename(f)[:-4].split(".shard")[0]
        groups.setdefault(name, []).append(pickle.load(open(f, "rb")))
    out = {"ref_model": a.ref_model, "protocol": "eval_merge.genppl_flm (gpt2-large, first 128 tokens, corpus PPL per seed, specials kept)",
           "paper_figure7": {"NCM": 93.1, "MDLM (66B tokens)": 115.5, "DUO": 97.3}, "models": {}}
    for name, parts in groups.items():
        raw = [r for p in parts for r in p["raw_ids"]]
        seeds = [s for p in parts for s in p["seeds"]]
        per = []
        for seed in dict.fromkeys(seeds):
            sel = [k for k, s_ in enumerate(seeds) if s_ == seed]
            texts = gen_tok.batch_decode([raw[k] for k in sel])
            per.append((seed, corpus_ppl(texts), float(np.mean([entropy(raw[k]) for k in sel]))))
        g, e = np.array([p[1] for p in per]), np.array([p[2] for p in per])
        out["models"][name] = {"gen_ppl": round(float(g.mean()), 2), "gen_ppl_sd": round(float(g.std(ddof=1)), 2) if len(g) > 1 else 0.0,
                               "entropy": round(float(e.mean()), 4), "n": len(raw),
                               "per_seed": [[s, round(p, 2), round(h, 4)] for s, p, h in per]}
        print(f"{name:28s} gen-PPL {g.mean():8.2f}  H {e.mean():.4f}  ({len(raw)} samples)", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen-mdlm")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--steps", type=int, required=True)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--bs", type=int, default=128)
    g.add_argument("--out", required=True)
    d = sub.add_parser("gen-data")
    d.add_argument("--valid", default="data/upstream/lm1b/valid_bert_30522.bin")
    d.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--ref-model", default="gpt2-large", help="the paper's reference model; gpt2 only for smoke tests")
    a = ap.parse_args()
    {"gen-mdlm": gen_mdlm, "gen-data": gen_data, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
