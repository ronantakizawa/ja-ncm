"""Wall-clock generation benchmark: NCM vs MDLM (several step counts) vs AR, one GPU.

Paper setup (Fig. 1, issue #9): 1,024 sequences, batch 16, H100, KV cache + bf16 autocast +
CUDA graphs for every model. Here every model runs under the same settings, at several batch
sizes (issue #26 found NCM's lead over AR shrinks with batch size):

  eager   bf16 autocast, KV cache where the model has one (NCM per-level prefix cache, AR)
  graphs  the same, wrapped in torch.compile(mode="reduce-overhead") (CUDA graph trees); a model
          that cannot be captured is reported as such, not silently timed eagerly
  capture the same, but the whole generation call captured once as one torch.cuda.CUDAGraph
          and replayed (works where cudagraph trees reject state allocated inside the call)

AR always generates the full 128 tokens (no early EOS stop), so its cost does not depend on
sample length. One warm-up pass per setting, then timed with CUDA synchronisation.

  python scripts/bench_latency.py --ncm runs/phase_b_v3/ja_uncond/last.pt \
      --mdlm runs/phase_b_v3/ja_uncond_mdlm/last.pt --ar runs/phase_b_v3/ja_uncond_ar/last.pt \
      --out results/v3/bench.json
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402
from jancm.loading import build_generator, build_tokenizer, load_state, resolve  # noqa: E402


def timed_capture(fn, n_seq, bs):
    """Capture one whole generation call (all levels / steps / tokens) as a single CUDA graph
    with torch.cuda.CUDAGraph and time replays. Unlike torch.compile's cudagraph trees this
    allows state allocated during the call (NCM's per-level KV cache) to live in the graph's
    private pool. Sampling RNG is graph-safe (Philox offsets advance on replay). Autocast's
    weight cache must be off while capturing."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(bs)                                # warm-up outside capture (lazy init, allocator)
            fn(bs)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(bs)
        g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_seq // bs):
            g.replay()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    del g
    torch.cuda.empty_cache()
    return dt


def timed(fn, n_seq, bs, mode):
    if mode == "capture":
        return timed_capture(fn, n_seq, bs)
    if mode == "graphs":
        fn = torch.compile(fn, mode="reduce-overhead", dynamic=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        fn(bs)                                    # warm-up (compilation / graph capture)
        fn(bs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_seq // bs):
            fn(bs)
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncm", required=True)
    ap.add_argument("--mdlm", required=True)
    ap.add_argument("--mdlm-extra", action="append", default=[], metavar="NAME=PATH",
                    help="more MDLM checkpoints to time, e.g. mdlmL=runs/.../ja_uncond_mdlm_l/last.pt")
    ap.add_argument("--ar", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batch", default="1,16,64")
    ap.add_argument("--mdlm-steps", default="8,16,32,64,128")
    ap.add_argument("--modes", default="eager,graphs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from omegaconf import OmegaConf
    from jancm.ar_setup import load_ar_checkpoint, special_ids
    from jancm.ar import left_pad
    from jancm.mdlm import generate_static, load_mdlm_checkpoint
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    st = load_state(a.ncm)
    ncfg = OmegaConf.create(st["config"])
    del st
    tk = build_tokenizer(ncfg, resolve(ncfg.checkpoints.tokenizer, ROOT), dev)
    ncm = build_generator(ncfg, tk, dev, init_ckpt=a.ncm).eval()
    eos = tk.pre_tokenizer.eos_token_id
    L = int(ncfg.tokenizer.vqvae.context_length)

    def ncm_gen(bs):   # codes -> decoder read-out, as evaluation does
        codes = ncm.generate(None, torch.full((bs, 1), eos, device=dev), temperature=1.0, top_k=0, top_p=1.0,
                             use_kv_cache=True)
        (zh,) = tk.vqvae.vq.idx_to_zhats(codes, return_zlevels=False)
        return tk.vqvae.decoder(zh[-1]).argmax(-1)

    mdlms = [("mdlm", load_mdlm_checkpoint(a.mdlm, dev)[0].eval())]
    for spec in a.mdlm_extra:
        name, path = spec.split("=", 1)
        mdlms.append((name, load_mdlm_checkpoint(path, dev)[0].eval()))
    ar, acfg, apretok = load_ar_checkpoint(a.ar, dev)
    ar.eval()
    sep, _, pad = special_ids(acfg, apretok)
    P = int(acfg.model.prefix_len)

    ar_prompts = {}   # built once per batch size, outside timing and graph capture

    def ar_gen(bs):
        if bs not in ar_prompts:
            ar_prompts[bs] = torch.tensor(left_pad([[eos]] * bs, P, pad), device=dev)
        x = ar_prompts[bs]
        return ar.generate(x, sep, pad, -1, max_new=L, use_kv_cache=True, temperature=1.0, early_stop=False)[0]

    models = [("ncm", 12, ncm_gen), ("ar", L, ar_gen)]
    for mname, m in mdlms:
        for s in [int(x) for x in a.mdlm_steps.split(",")]:
            models.append((f"{mname}_T{s}", s, (lambda m, s: lambda bs: generate_static(m, bs, L, s, dev))(m, s)))
    rows = []
    for mode in a.modes.split(","):
        for bs in [int(x) for x in a.batch.split(",")]:
            n = a.n if bs > 1 else min(a.n, 128)
            for name, nfe, fn in models:
                try:
                    secs = timed(fn, n, bs, mode)
                    row = {"model": name, "nfe": nfe, "mode": mode, "batch": bs, "n_seq": n,
                           "seconds": round(secs, 3), "ms_per_seq": round(1000 * secs / n, 3),
                           "seconds_per_1024": round(secs * 1024 / n, 3)}
                except Exception as e:   # e.g. a data-dependent host sync that CUDA graphs cannot capture
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    row = {"model": name, "nfe": nfe, "mode": mode, "batch": bs, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                torch._dynamo.reset()
                print(json.dumps(row), flush=True)
                rows.append(row)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
