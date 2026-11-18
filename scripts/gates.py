"""Go/no-go gates for the Phase B queue (exit 0 = pass, 1 = fail; prints the evidence).

  python scripts/gates.py stage0      # own LM1B generator gen-PPL at matched entropy <= 1.25 x released
  python scripts/gates.py tokenizer   # Japanese tokenizer full reconstruction >= 95%
  python scripts/gates.py a2          # A2 results present and NCM rows sane
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    with open(os.path.join(ROOT, p)) as f:
        return json.load(f)


def ppl_at_entropy(points, h):
    """log gen-PPL interpolated linearly in entropy at h from (entropy, gen_ppl) points.

    Uses the two points bracketing h; outside the range it extrapolates from the nearest two
    (flagged by the caller)."""
    import math
    pts = sorted(points)
    lo = max([p for p in pts if p[0] <= h], default=pts[0])
    hi = min([p for p in pts if p[0] >= h], default=pts[-1])
    if lo == hi:
        i = pts.index(lo)
        lo, hi = (pts[i], pts[i + 1]) if i + 1 < len(pts) else (pts[i - 1], pts[i])
    t = (h - lo[0]) / (hi[0] - lo[0])
    return math.exp(math.log(lo[1]) + t * (math.log(hi[1]) - math.log(lo[1]))), pts[0][0] <= h <= pts[-1][0]


STAGE0_CKPT = "runs/stage0/lm1b_small_v2/generator_step100000.pt"
STAGE0_TEMPS = ("1.0", "0.9", "0.8", "0.7")


def stage0(ppl_tol=1.25, ckpt=STAGE0_CKPT, tag="v2"):
    """Generation-quality gate at matched entropy (the authors' rebuttal protocol).

    Teacher-forced dev loss is not comparable across generators (the released LM1B generator
    scores 4.1 nats on our codes yet samples fluently), and raw gen-PPL moves with sample
    entropy (authors: r = +0.79 across runs). So the released generator is sampled at T=1 and
    ours at several temperatures (2 seeds x 256, no decoding penalties, GPT-2-large); our
    gen-PPL is interpolated at the released generator's entropy.
    PASS: our matched-entropy gen-PPL <= ppl_tol x released.
    """
    import subprocess
    py, cfg = sys.executable, "configs/stage0/lm1b_small.yaml"
    common = ["--n", "256", "--seeds", "42,0", "--bs", "64", "--ppl-model", "gpt2-large"]
    runs = [("released", "artifacts/lm1b/small/generator.pt", "1.0")] + [(f"own_{tag}", ckpt, T) for T in STAGE0_TEMPS]
    for name, ck, T in runs:
        out = f"results/stage0_uncond_{name}_T{T}.json"
        if not os.path.exists(os.path.join(ROOT, out)):
            subprocess.run([py, "scripts/eval_uncond.py", "--model", "ncm", "--config", cfg, "--ckpt", ck,
                            "--out", out, *common, f"generation.temperature={T}"], cwd=ROOT, check=True)
    r = load("results/stage0_uncond_released_T1.0.json")
    own = [load(f"results/stage0_uncond_own_{tag}_T{T}.json") for T in STAGE0_TEMPS]
    pts = [(o["entropy_mean"], o["gen_ppl_mean"]) for o in own]
    ppl, inside = ppl_at_entropy(pts, r["entropy_mean"])
    ok = ppl <= ppl_tol * r["gen_ppl_mean"]
    curve = ", ".join(f"T{T}: {o['gen_ppl_mean']:.1f}@{o['entropy_mean']:.3f}" for T, o in zip(STAGE0_TEMPS, own))
    print(f"[gate stage0] released gen-PPL {r['gen_ppl_mean']:.1f} at entropy {r['entropy_mean']:.3f}; ours ({curve}) "
          f"-> {ppl:.1f} at matched entropy{'' if inside else ' (extrapolated)'} "
          f"(ratio {ppl / r['gen_ppl_mean']:.2f}, limit {ppl_tol}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def tokenizer(min_acc=95.0, run="runs/phase_b/ja_tokenizer"):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, run, "metrics.jsonl")) if "recon_acc_full" in l]
    best = max(rows, key=lambda d: d["recon_acc_full"])
    ok = best["recon_acc_full"] >= min_acc
    print(f"[gate tokenizer] best full reconstruction {best['recon_acc_full']:.2f}% at step {best['step']} "
          f"(per level {best['recon_acc_per_level']}), exact windows {best['exact_window_pct']}% "
          f"(limit {min_acc}%) -> {'PASS' if ok else 'FAIL'}")
    return ok


def a2():
    ok = True
    for sysname in ("ncm_final", "ar_final"):
        for t in ("flores", "ntrex", "kftt", "alt"):
            p = f"results/{sysname}/{t}.json"
            if not os.path.exists(os.path.join(ROOT, p)):
                print(f"[gate a2] missing {p}")
                ok = False
                continue
            d = load(p)
            if not (d["bleu"] > 1 and "comet22" in d):
                print(f"[gate a2] suspicious {p}: bleu={d['bleu']} comet={d.get('comet22')}")
                ok = False
    ext = glob.glob(os.path.join(ROOT, "results/ext_*/*.json"))
    print(f"[gate a2] {len(ext)} external-baseline result files -> {'PASS' if ok and ext else 'FAIL'}")
    return ok and bool(ext)


if __name__ == "__main__":
    fn = {"stage0": stage0, "tokenizer": tokenizer, "a2": a2}[sys.argv[1]]
    sys.exit(0 if (fn(run=sys.argv[2]) if len(sys.argv) > 2 else fn()) else 1)
