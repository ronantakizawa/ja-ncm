"""Latency of NCM against an AR decoder of the SAME shape as the NCM generator (one GPU).

Our AR baseline is 12x768 while the v3 NCM generator is 24x1024, so bench_latency.py's
NCM-vs-AR ratio is not the same-shape comparison of GitHub issue #26. Decoding cost does not
depend on the weights, so (as in #26) the same-shape AR is randomly initialised: same token
table width, vocabulary, prefix and 128 new tokens as the trained AR baseline, generated
without early stopping. Timed with the same harness and modes as bench_latency.py, in one run
with NCM and the trained 12x768 AR for reference.

  python scripts/bench_ar_shape.py --ncm runs/phase_b_v3/ja_uncond/last.pt \
      --ar runs/phase_b_v3/ja_uncond_ar/last.pt --out results/v3/bench_ar_shape.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402
from jancm.loading import build_generator, build_tokenizer, load_state, resolve  # noqa: E402
from scripts.bench_latency import timed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncm", required=True)
    ap.add_argument("--ar", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batch", default="1,16,64,128")
    ap.add_argument("--modes", default="eager,capture")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from omegaconf import OmegaConf
    from jancm.ar import left_pad
    from jancm.ar_setup import build_ar, load_ar_checkpoint, special_ids
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    st = load_state(a.ncm)
    ncfg = OmegaConf.create(st["config"])
    del st
    tk = build_tokenizer(ncfg, resolve(ncfg.checkpoints.tokenizer, ROOT), dev)
    ncm = build_generator(ncfg, tk, dev, init_ckpt=a.ncm).eval()
    eos = tk.pre_tokenizer.eos_token_id
    L = int(ncfg.tokenizer.vqvae.context_length)
    M = ncfg.model

    def ncm_gen(bs):
        codes = ncm.generate(None, torch.full((bs, 1), eos, device=dev), temperature=1.0, top_k=0, top_p=1.0,
                             use_kv_cache=True)
        (zh,) = tk.vqvae.vq.idx_to_zhats(codes, return_zlevels=False)
        return tk.vqvae.decoder(zh[-1]).argmax(-1)

    ar_small, acfg, pretok = load_ar_checkpoint(a.ar, dev)
    ar_small.eval()
    sep, _, pad = special_ids(acfg, pretok)
    P = int(acfg.model.prefix_len)
    shape_cfg = OmegaConf.merge(acfg, OmegaConf.create({"ar": {"n_layer": int(M.n_layer), "n_head": int(M.n_head),
                                                               "n_embd": int(M.n_embd), "init_emb_from_vqvae": False}}))
    V, D = int(acfg.tokenizer.vqvae.in_vocab_size), int(acfg.tokenizer.vqvae.embed_dim)
    torch.manual_seed(0)
    ar_shape = build_ar(shape_cfg, dev, emb_init=torch.randn(V, D) * 0.02).eval()
    torch.nn.init.normal_(ar_shape.out_proj.weight, std=0.02)   # non-degenerate logits (zero init gives uniform)
    prompts = {}

    def ar_fn(net):
        def f(bs):
            if bs not in prompts:
                prompts[bs] = torch.tensor(left_pad([[eos]] * bs, P, pad), device=dev)
            return net.generate(prompts[bs], sep, pad, -1, max_new=L, use_kv_cache=True, temperature=1.0,
                                early_stop=False)[0]
        return f

    def params(net):
        return round(sum(p.numel() for p in net.parameters()) / 1e6, 1)

    shape = f"{int(M.n_layer)}x{int(M.n_embd)}"
    models = [("ncm", 12, ncm_gen, f"{shape} (trained)"),
              ("ar_same_shape", L, ar_fn(ar_shape), f"{shape} random weights, {params(ar_shape)}M params"),
              ("ar_trained_12x768", L, ar_fn(ar_small), f"12x768 trained, {params(ar_small)}M params")]
    rows = []
    for mode in a.modes.split(","):
        for bs in [int(x) for x in a.batch.split(",")]:
            n = a.n if bs > 1 else min(a.n, 128)
            n = max(bs, n - n % bs)
            for name, nfe, fn, desc in models:
                try:
                    secs = timed(fn, n, bs, mode)
                    row = {"model": name, "desc": desc, "nfe": nfe, "mode": mode, "batch": bs, "n_seq": n,
                           "seconds": round(secs, 3), "ms_per_seq": round(1000 * secs / n, 3)}
                except Exception as e:
                    row = {"model": name, "mode": mode, "batch": bs, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                torch._dynamo.reset()
                torch.cuda.empty_cache()
                print(json.dumps(row), flush=True)
                rows.append(row)
    by = {(r["model"], r["mode"], r["batch"]): r.get("ms_per_seq") for r in rows}
    ratios = {f"{m}_b{b}": round(by[("ar_same_shape", m, b)] / by[("ncm", m, b)], 2)
              for m in a.modes.split(",") for b in [int(x) for x in a.batch.split(",")]
              if by.get(("ar_same_shape", m, b)) and by.get(("ncm", m, b))}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": rows,
               "speedup_same_shape_ar_over_ncm": ratios}, open(a.out, "w"), indent=1)
    print("speedup (same-shape AR ms / NCM ms):", ratios)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
