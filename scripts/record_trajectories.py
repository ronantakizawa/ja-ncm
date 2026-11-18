"""Record intermediate generation states for the demo video (scripts/render_race_video.py).

For each model, draws 16 unconditional samples, scores them with the gen-PPL reference model
and keeps the MEDIAN sample (the same rule for every model, so no model gets a cherry-picked
example). Records what the model holds after every network pass:

  NCM   decoded text after each of the 12 levels (cumulative coarse-to-fine reconstruction)
  AR    the tokens generated so far, after each of the 128 passes
  MDLM  the window after each denoising step, masked positions shown as None

Temperatures are the ones that put each model closest to the data's character entropy
(results/v3/uncond). Output: JSON with per-pass states and the measured batch-1 latency.

  python scripts/record_trajectories.py --out results/v3/trajectories.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402
from jancm.loading import build_generator, build_tokenizer, load_state, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncm", default="runs/phase_b_v3/ja_uncond/last.pt")
    ap.add_argument("--mdlm-l", default="runs/phase_b_v3/ja_uncond_mdlm_l/last.pt")
    ap.add_argument("--ar", default="runs/phase_b_v3/ja_uncond_ar/last.pt")
    ap.add_argument("--k", type=int, default=16, help="candidates per model; the median gen-PPL one is kept")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from omegaconf import OmegaConf
    from jancm.ar import left_pad
    from jancm.ar_setup import load_ar_checkpoint, special_ids
    from jancm.mdlm import load_mdlm_checkpoint
    from jancm.ncm_infer import generate_ids
    from jancm.text import decode_target
    dev = "cuda"

    st = load_state(a.ncm)
    ncfg = OmegaConf.create(st["config"])
    del st
    tk = build_tokenizer(ncfg, resolve(ncfg.checkpoints.tokenizer, ROOT), dev)
    ncm = build_generator(ncfg, tk, dev, init_ckpt=a.ncm).eval()
    pretok = tk.pre_tokenizer
    hf = pretok.hf_tokenizer
    eos, pad = pretok.eos_token_id, pretok.pad_token_id
    L = int(ncfg.tokenizer.vqvae.context_length)

    def spans(ids, masked):
        """Window -> list of (text, is_mask) runs; decoding runs (not single tokens) keeps UTF-8 whole."""
        out, run, cur = [], [], None
        for t, m in zip(ids, masked):
            if m != cur and run:
                out.append(("" if cur else hf.decode(run, skip_special_tokens=True), bool(cur), len(run)))
                run = []
            cur = m
            run.append(t)
            if not m and t == eos:
                break
        if run:
            out.append(("" if cur else hf.decode(run, skip_special_tokens=True), bool(cur), len(run)))
        return [{"text": s, "mask": m, "n": n} for s, m, n in out]

    # NCM: per-level reconstructions
    torch.manual_seed(a.seed)
    gcfg = OmegaConf.merge(ncfg, OmegaConf.from_dotlist(["generation.temperature=0.95"]))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        finals, levels = generate_ids(ncm, tk, [[eos]] * a.k, gcfg, dev, batch_size=a.k, return_levels=True)
    cands = {"ncm": [{"final": f, "states": [[{"text": t, "mask": False, "n": 0}] for t in lv]} for f, lv in zip(finals, levels)]}

    # AR: token-by-token
    ar, acfg, _ = load_ar_checkpoint(a.ar, dev)
    ar.eval()
    sep, _, apad = special_ids(acfg, pretok)
    P = int(acfg.model.prefix_len)
    torch.manual_seed(a.seed)
    x = torch.tensor(left_pad([[eos]] * a.k, P, apad), device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        toks = ar.generate(x, sep, apad, eos, max_new=L, use_kv_cache=True, temperature=0.95, early_stop=False)[0]
    cands["ar"] = []
    for row in toks.tolist():
        states = [spans(row[:i + 1], [False] * (i + 1)) for i in range(L)]
        cands["ar"].append({"final": decode_target(pretok, row, eos, pad), "states": states})

    # MDLM (same size as NCM): ancestral steps, recorded
    mdlm = load_mdlm_checkpoint(a.mdlm_l, dev)[0].eval()
    tab = mdlm.wte.weight[:mdlm.V]
    for steps, temp in [(16, 0.95), (128, 0.9)]:
        torch.manual_seed(a.seed)
        xs = torch.full((a.k, L), mdlm.mask_id, dtype=torch.long, device=dev)
        hist = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(steps):
                t, s = 1.0 - i / steps, 1.0 - (i + 1) / steps
                still = xs == mdlm.mask_id
                reveal = still & (torch.rand(xs.shape, device=dev) < ((t - s) / t if i < steps - 1 else 2.0))
                lg = (mdlm.out_proj(mdlm.hidden(xs)) @ tab.t()).float()
                u = torch.rand(lg.shape, device=dev).clamp_(1e-20, 1.0)
                xs = torch.where(reveal, (lg / temp - torch.log(-torch.log(u))).argmax(-1), xs)
                hist.append(xs.clone().cpu())
        rows = []
        for b in range(a.k):
            states = [spans(h[b].tolist(), (h[b] == mdlm.mask_id).tolist()) for h in hist]
            rows.append({"final": decode_target(pretok, hist[-1][b].tolist(), eos, pad), "states": states})
        cands[f"mdlmL_T{steps}"] = rows

    # Keep the median gen-PPL candidate of each model.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.gen_ppl import generation_perplexity
    ref = str(ncfg.eval.get("ppl_model", "llm-jp/llm-jp-3-1.8b"))
    rtok = AutoTokenizer.from_pretrained(ref)
    rlm = AutoModelForCausalLM.from_pretrained(ref, torch_dtype=torch.bfloat16).to(dev).eval()
    out = {"selection": f"median gen-PPL of {a.k} samples per model (seed {a.seed})", "models": {}}
    for name, rows in cands.items():
        ppl = []
        for r in rows:
            txt = r["final"] if r["final"].strip() else " "
            ppl.append(float(generation_perplexity(rlm, rtok, [""], [txt], device=dev, batch_size=1, max_length=512).mean_ppl))
        order = sorted(range(len(rows)), key=lambda i: ppl[i])
        pick = order[len(order) // 2]
        out["models"][name] = {"final": rows[pick]["final"], "gen_ppl": round(ppl[pick], 1),
                               "passes": len(rows[pick]["states"]), "states": rows[pick]["states"]}
        print(name, len(rows[pick]["states"]), round(ppl[pick], 1), rows[pick]["final"][:80].replace("\n", " "), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
