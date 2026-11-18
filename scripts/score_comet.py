"""COMET-22 (Unbabel/wmt22-comet-da, Apache-2.0, not gated) for every system in results/.

Runs in its own virtualenv because unbabel-comet pins numpy<2 and transformers<5:
  python3 -m venv .venv-comet && .venv-comet/bin/pip install -r requirements-comet.txt
  .venv-comet/bin/python scripts/score_comet.py

Reads results/<system>/<test>.hyps.jsonl (line-aligned) with the test set's sources and
references, adds "comet22" to results/<system>/<test>.json and writes results/comet22.json.
Uses only the standard library plus comet, so it does not import the project package.
"""
import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "processed", "test"))
    ap.add_argument("--model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated fnmatch patterns of system dirs to score")
    a = ap.parse_args()
    import fnmatch
    only = [p for p in a.only.split(",") if p]
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(a.model))
    table = {}
    for hp in sorted(glob.glob(os.path.join(a.results, "*", "*.hyps.jsonl"))):
        system, test = os.path.basename(os.path.dirname(hp)), os.path.basename(hp)[: -len(".hyps.jsonl")]
        tf = os.path.join(a.data_dir, test, "test.jsonl")
        if system.startswith("_") or not os.path.exists(tf):
            continue
        if only and not any(fnmatch.fnmatch(system, p) for p in only):
            continue
        rows, hyps = read_jsonl(tf), read_jsonl(hp)
        assert len(rows) == len(hyps), f"{hp}: {len(hyps)} outputs for {len(rows)} test rows"
        data = [{"src": r["src"], "mt": h, "ref": r["ref"]} for r, h in zip(rows, hyps)]
        # num_workers>0: comet sets a multiprocessing_context, which new torch rejects with 0 workers.
        out = model.predict(data, batch_size=a.bs, gpus=a.gpus, progress_bar=False, num_workers=2)
        score = round(100 * float(out.system_score), 2)
        table.setdefault(system, {})[test] = score
        jp = os.path.join(os.path.dirname(hp), f"{test}.json")
        if os.path.exists(jp):
            with open(jp) as f:
                res = json.load(f)
            res["comet22"] = score
            res["comet_model"] = a.model
            with open(jp, "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"{system:40s} {test:7s} COMET-22 {score:6.2f}", flush=True)
    out = os.path.join(a.results, "comet22.json")
    prev = json.load(open(out))["scores"] if os.path.exists(out) else {}
    prev.update(table)
    with open(out, "w") as f:
        json.dump({"model": a.model, "scores": prev}, f, indent=2)


if __name__ == "__main__":
    main()
