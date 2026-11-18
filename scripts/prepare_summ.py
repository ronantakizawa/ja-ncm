"""Prepare Japanese summarisation data (Phase B fine-tune): article prefix -> summary target.

Corpora:
  distill  hachi-intelligence/JapaneseSummarization-FW2EduJa-Distill (Apache 2.0). Summaries
           written by Qwen3-30B-A3B / gpt-oss-120b; ``--lengths`` picks the instruction
           length buckets (default "100": the 100-character summaries).
  xlsum    csebuetnlp/xlsum japanese (CC BY-NC-SA 4.0: research only). Human-written BBC
           summaries. train -> fine-tune stage 2, validation -> dev, test -> headline score.

Output (--out, default data/phase_b/summ), same layout as prepare_data.py:
  train/, dev/            src.npy [N, P], tgt.npy [N, L], source.npy, meta.json
  dev/dev.jsonl           XL-Sum validation as text
  test/xlsum/test.jsonl   {"src": article, "ref": summary}
  ../xlsum_eval_urls.txt  URLs of XL-Sum validation + test, for prepare_corpus.py decontamination

  python scripts/prepare_summ.py                       # all distill shards
  python scripts/prepare_summ.py --max-shards 1 --limit 5000   # smoke test
"""
import argparse
import io
import json
import os
import sys
import tarfile
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402
from jancm.loading import load_config  # noqa: E402
from jancm.text import encode_batch, pretokenizer_from_config, target_window  # noqa: E402

DISTILL = "hachi-intelligence/JapaneseSummarization-FW2EduJa-Distill"
SOURCES = ["distill", "xlsum"]


def read_xlsum(raw):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("csebuetnlp/xlsum", "data/japanese_XLSum_v2.0.tar.bz2", repo_type="dataset",
                           cache_dir=os.path.join(raw, "hf"))
    out = {}
    with tarfile.open(path, "r:bz2") as tar:
        for m in tar.getmembers():
            for split in ("train", "val", "test"):
                if m.name.endswith(f"_{split}.jsonl"):
                    rows = [json.loads(l) for l in io.TextIOWrapper(tar.extractfile(m), encoding="utf-8") if l.strip()]
                    out["dev" if split == "val" else split] = rows
    assert set(out) == {"train", "dev", "test"}, f"unexpected archive layout: {list(out)}"
    return out


def read_distill(raw, lengths, max_shards):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files(DISTILL, repo_type="dataset") if f.endswith(".parquet"))
    rows, seen_len = [], Counter()
    for f in files[: max_shards or None]:
        t = pq.read_table(hf_hub_download(DISTILL, f, repo_type="dataset", cache_dir=os.path.join(raw, "hf")),
                          columns=["input", "output", "length", "url"])
        for r in t.to_pylist():
            seen_len[r["length"]] += 1
            if r["length"] in lengths and r["input"] and r["output"]:
                rows.append(r)
    print(f"[distill] length buckets seen: {dict(seen_len)}; kept {len(rows)}")
    return rows


def tokenise(pairs, pretok, P, L, eos, pad, fill, drops):
    """Articles longer than P are truncated (lead is kept; news/web leads carry the summary)."""
    src_ids = encode_batch(pretok, [p[0] for p in pairs])
    tgt_ids = encode_batch(pretok, [p[1] for p in pairs])
    src, tgt, kept, n_trunc = [], [], [], 0
    for (a, s), si, ti in zip(pairs, src_ids, tgt_ids):
        tw = target_window(ti, L, eos, pad, True, fill)
        if tw is None:
            drops["summary_too_long"] += 1
            continue
        if len(si) > P:
            n_trunc += 1
        src.append(si[:P] + [pad] * (P - min(len(si), P)))
        tgt.append(tw)
        kept.append((a, s, len(si), len(ti)))
    return src, tgt, kept, n_trunc


def pct(xs):
    a = np.asarray(xs) if xs else np.zeros(1)
    return {f"p{q}": int(np.percentile(a, q)) for q in (50, 90, 99)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "phase_b", "ja_summ.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "phase_b", "summ"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--lengths", default="100", help="distill length buckets to keep, comma-separated")
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap rows per corpus split (smoke tests)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    pretok = pretokenizer_from_config(cfg)
    P, L = int(cfg.model.prefix_len), int(cfg.tokenizer.vqvae.context_length)
    eos, pad = pretok.eos_token_id, pretok.pad_token_id
    fill = str(cfg.data.get("target_fill", "pad"))
    print(f"[config] P={P} L={L} fill={fill}")

    xl = read_xlsum(a.raw)
    eval_urls = {r["url"] for s in ("dev", "test") for r in xl[s]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(os.path.join(os.path.dirname(a.out), "xlsum_eval_urls.txt"), "w") as f:
        f.write("\n".join(sorted(eval_urls)) + "\n")
    test = xl["test"][: a.limit or None]
    d = os.path.join(a.out, "test", "xlsum")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "test.jsonl"), "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps({"src": r["text"], "ref": r["summary"]}, ensure_ascii=False) + "\n")
    print(f"[test] xlsum: {len(test)}; {len(eval_urls)} eval URLs written for corpus decontamination")

    distill = read_distill(a.raw, set(a.lengths.split(",")), a.max_shards)
    distill = [r for r in distill if r["url"] not in eval_urls]
    corpora = {"train": {"distill": [(r["input"], r["output"]) for r in distill],
                         "xlsum": [(r["text"], r["summary"]) for r in xl["train"]]},
               "dev": {"xlsum": [(r["text"], r["summary"]) for r in xl["dev"]]}}
    meta = {"prefix_len": P, "target_len": L, "eos_id": eos, "pad_id": pad, "target_eos": True,
            "target_fill": fill, "pretokenizer": str(cfg.tokenizer.pre_tokenizer.model_id), "sources": SOURCES}
    for split, by_src in corpora.items():
        out_dir = os.path.join(a.out, split)
        os.makedirs(out_dir, exist_ok=True)
        S, T, src_id, stats, text = [], [], [], {}, []
        for i, name in enumerate(SOURCES):
            pairs = [(x.strip(), y.strip()) for x, y in by_src.get(name, [])][: a.limit or None]
            if not pairs:
                continue
            drops = Counter()
            s, t, kept, n_trunc = tokenise(pairs, pretok, P, L, eos, pad, fill, drops)
            S += s
            T += t
            src_id += [i] * len(s)
            text += [(k[0], k[1]) for k in kept]
            stats[name] = {"raw": len(pairs), "kept": len(s), "dropped": dict(drops), "article_truncated": n_trunc,
                           "article_tokens": pct([k[2] for k in kept]), "summary_tokens": pct([k[3] for k in kept])}
            print(f"[{split}] {name}: {stats[name]}", flush=True)
        np.save(os.path.join(out_dir, "src.npy"), np.asarray(S, dtype=np.int32))
        np.save(os.path.join(out_dir, "tgt.npy"), np.asarray(T, dtype=np.int32))
        np.save(os.path.join(out_dir, "source.npy"), np.asarray(src_id, dtype=np.uint8))
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({**meta, "counts": {k: v["kept"] for k, v in stats.items()}, "stats": stats}, f,
                      indent=2, ensure_ascii=False)
        if split == "dev":
            with open(os.path.join(out_dir, "dev.jsonl"), "w", encoding="utf-8") as f:
                for x, y in text:
                    f.write(json.dumps({"src": x, "ref": y}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
