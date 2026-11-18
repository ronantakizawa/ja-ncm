"""Check that the CUDA-graph replays timed by bench_latency.py (--modes capture) produce valid samples.

The speed numbers only count if a replayed graph does the same work as eager generation. For
each model this captures the same generation call bench_latency.py times, replays it, and
compares with eager generation on the same model:

  greedy  NCM (top_k=1) and AR (temperature 0) are deterministic, so every replayed sequence
          should equal the eager one token for token (match_rate; bf16 kernels may differ
          slightly, so a small mismatch rate is reported rather than asserted).
  sampled every model at temperature 1: gen-PPL under llm-jp-3-1.8b, character entropy, the
          share of distinct samples (replays must not repeat one batch) and of empty samples,
          for eager vs replay. Matching distributions = the graph samples like eager does.

  python scripts/check_capture.py --ncm runs/phase_b_v3/ja_uncond/last.pt \
      --mdlm runs/phase_b_v3/ja_uncond_mdlm/last.pt --mdlm-l runs/phase_b_v3/ja_uncond_mdlm_l/last.pt \
      --ar runs/phase_b_v3/ja_uncond_ar/last.pt --out results/v3/capture_check.json
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402,F401
from jancm.loading import build_generator, build_tokenizer, load_state, resolve  # noqa: E402


def char_entropy(t):
    c, m = Counter(t), len(t)
    return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0


def capture(fn, bs):
    """Capture fn(bs) as one CUDA graph exactly as bench_latency.timed_capture does; return (graph, output)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(bs)
            fn(bs)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn(bs)
    return g, out


def replay(g, out, n, bs):
    rows = []
    for _ in range(n // bs):
        g.replay()
        rows.append(out.clone())
    torch.cuda.synchronize()
    return torch.cat(rows)


def eager(fn, n, bs):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.cat([fn(bs) for _ in range(n // bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncm", required=True)
    ap.add_argument("--mdlm", required=True)
    ap.add_argument("--mdlm-l", required=True)
    ap.add_argument("--ar", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from omegaconf import OmegaConf
    from jancm.ar import left_pad
    from jancm.ar_setup import load_ar_checkpoint, special_ids
    from jancm.mdlm import generate_static, load_mdlm_checkpoint
    from jancm.text import decode_target
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    st = load_state(a.ncm)
    ncfg = OmegaConf.create(st["config"])
    del st
    tk = build_tokenizer(ncfg, resolve(ncfg.checkpoints.tokenizer, ROOT), dev)
    ncm = build_generator(ncfg, tk, dev, init_ckpt=a.ncm).eval()
    pretok = tk.pre_tokenizer
    eos, pad = pretok.eos_token_id, pretok.pad_token_id
    L = int(ncfg.tokenizer.vqvae.context_length)

    def ncm_fn(top_k):
        def f(bs):   # same call as bench_latency.ncm_gen
            codes = ncm.generate(None, torch.full((bs, 1), eos, device=dev), temperature=1.0, top_k=top_k,
                                 top_p=1.0, use_kv_cache=True)
            (zh,) = tk.vqvae.vq.idx_to_zhats(codes, return_zlevels=False)
            return tk.vqvae.decoder(zh[-1]).argmax(-1)
        return f

    mdlm = load_mdlm_checkpoint(a.mdlm, dev)[0].eval()
    mdlm_l = load_mdlm_checkpoint(a.mdlm_l, dev)[0].eval()
    ar, acfg, _ = load_ar_checkpoint(a.ar, dev)
    ar.eval()
    sep, _, apad = special_ids(acfg, pretok)
    P = int(acfg.model.prefix_len)
    prompts = {}

    def ar_fn(temperature):
        def f(bs):   # same call as bench_latency.ar_gen
            if bs not in prompts:
                prompts[bs] = torch.tensor(left_pad([[eos]] * bs, P, apad), device=dev)
            return ar.generate(prompts[bs], sep, apad, -1, max_new=L, use_kv_cache=True,
                               temperature=temperature, early_stop=False)[0]
        return f

    def texts(ids):
        return [decode_target(pretok, r, eos, pad) for r in ids.tolist()]

    report = {"n": a.n, "batch": a.bs, "gpu": torch.cuda.get_device_name(0), "greedy": {}, "sampled": {}}

    # 1. Greedy: replay must reproduce eager output token for token.
    for name, fn in [("ncm", ncm_fn(1)), ("ar", ar_fn(0.0))]:
        e = eager(fn, a.n, a.bs)
        g, out = capture(fn, a.bs)
        r = replay(g, out, a.n, a.bs)
        del g
        torch.cuda.empty_cache()
        seq_match = (e == r).all(1).float().mean().item()
        tok_match = (e == r).float().mean().item()
        report["greedy"][name] = {"sequence_match_rate": round(seq_match, 4), "token_match_rate": round(tok_match, 4),
                                  "example_eager": texts(e[:1])[0][:120], "example_replay": texts(r[:1])[0][:120]}
        print(name, "greedy", report["greedy"][name], flush=True)

    # 2. Sampled: replay and eager should give the same quality distribution.
    samples = {}
    for name, fn in [("ncm", ncm_fn(0)), ("ar", ar_fn(1.0)),
                     ("mdlm_T32", lambda bs: generate_static(mdlm, bs, L, 32, dev)),
                     ("mdlmL_T16", lambda bs: generate_static(mdlm_l, bs, L, 16, dev))]:
        torch.manual_seed(0)
        e = eager(fn, a.n, a.bs)
        g, out = capture(fn, a.bs)
        r = replay(g, out, a.n, a.bs)
        del g
        torch.cuda.empty_cache()
        samples[name] = {"eager": texts(e), "replay": texts(r)}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.gen_ppl import generation_perplexity
    ref = str(ncfg.eval.get("ppl_model", "llm-jp/llm-jp-3-1.8b"))
    rtok = AutoTokenizer.from_pretrained(ref)
    rlm = AutoModelForCausalLM.from_pretrained(ref, torch_dtype=torch.bfloat16).to(dev).eval()
    for name, d in samples.items():
        row = {}
        for kind, ts in d.items():
            ne = [t for t in ts if t.strip()] or [" "]
            ppl = float(generation_perplexity(rlm, rtok, [""] * len(ne), ne, device=dev, batch_size=16,
                                              max_length=512).mean_ppl)
            row[kind] = {"gen_ppl": round(ppl, 2), "char_entropy": round(sum(char_entropy(t) for t in ne) / len(ne), 4),
                         "distinct_frac": round(len(set(ts)) / len(ts), 4), "empty_frac": round(1 - len(ne) / len(ts), 4),
                         "example": ts[0][:120]}
        row["gen_ppl_ratio_replay_over_eager"] = round(row["replay"]["gen_ppl"] / row["eager"]["gen_ppl"], 3)
        report["sampled"][name] = row
        print(name, "sampled", json.dumps(row, ensure_ascii=False), flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
