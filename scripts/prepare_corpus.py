"""Tokenise hotchpotch/fineweb-2-edu-japanese into fixed 128-token windows (Phase B).

The same windows train the Japanese concept tokenizer (B1) and the unconditional generator
(B2), as the paper does per dataset.

Windowing: every document is tokenised with the Qwen3 pre-tokenizer, EOS is appended, and
the result is cut into consecutive L-token windows; the last window is right-padded with PAD.
Documents are never packed together, so the tokenizer also learns the "text, EOS, PAD"
layout that summaries and translations use as targets.

Decontamination: documents whose URL appears in the XL-Sum ja validation or test split are
skipped (write that list first with prepare_summ.py; a warning is printed otherwise).

Output (--out, default data/phase_b/corpus):
  train/windows_XXXXX.npy  int32 [n, L], one file per source parquet shard
  dev/windows_00000.npy    from the dataset's held-out test parquet
  {train,dev}/meta.json    shard list, window, counts, token totals

  python scripts/prepare_corpus.py --workers 16                  # all 139 sample_10BT shards (~37 GB download)
  python scripts/prepare_corpus.py --max-shards 2 --workers 2    # smoke test
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402

REPO = "hotchpotch/fineweb-2-edu-japanese"


def windows_from_ids(ids, L, eos, pad, min_tokens):
    """Split one document (ids + EOS) into L-token windows; drop a tail shorter than min_tokens
    unless it is the whole document."""
    ids = list(ids) + [eos]
    out = []
    for i in range(0, len(ids), L):
        w = ids[i:i + L]
        if len(w) < min_tokens and i > 0:
            break
        out.append(w + [pad] * (L - len(w)))
    return out


def process_shard(args):
    (filename, out_path, model_id, eos_tok, pad_tok, L, min_tokens, max_windows_per_doc,
     skip_urls_path, raw_dir, keep_raw, batch_docs) = args
    if os.path.exists(out_path):
        arr = np.load(out_path, mmap_mode="r")
        return filename, len(arr), None, "cached"
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    eos, pad = tok.convert_tokens_to_ids(eos_tok), tok.convert_tokens_to_ids(pad_tok)
    skip = set()
    if skip_urls_path and os.path.exists(skip_urls_path):
        skip = set(open(skip_urls_path, encoding="utf-8").read().split())
    path = hf_hub_download(REPO, filename, repo_type="dataset", local_dir=raw_dir)
    pf = pq.ParquetFile(path)
    # One int32 array per batch of documents: ~0.5 GB per worker for a 270 MB shard, instead
    # of several GB of Python int lists.
    chunks, n_docs, n_skipped, n_tokens = [], 0, 0, 0
    for batch in pf.iter_batches(batch_size=batch_docs, columns=["text", "url"]):
        texts, urls = batch.column("text").to_pylist(), batch.column("url").to_pylist()
        keep = [t for t, u in zip(texts, urls) if t and u not in skip]
        n_skipped += len(texts) - len(keep)
        rows = []
        for ids in tok(keep, add_special_tokens=False)["input_ids"]:
            n_docs += 1
            n_tokens += len(ids) + 1
            rows.extend(windows_from_ids(ids, L, eos, pad, min_tokens)[:max_windows_per_doc or None])
        if rows:
            chunks.append(np.asarray(rows, dtype=np.int32).reshape(-1, L))
    arr = np.concatenate(chunks) if chunks else np.zeros((0, L), dtype=np.int32)
    del chunks
    np.save(out_path + ".tmp.npy", arr)
    os.replace(out_path + ".tmp.npy", out_path)
    if not keep_raw:
        os.remove(path)
    return filename, len(arr), {"docs": n_docs, "skipped_url": n_skipped, "tokens": n_tokens}, "done"


def write_meta(d, L, results, extra):
    shards = sorted(os.path.basename(p) for p in os.listdir(d) if p.startswith("windows_") and p.endswith(".npy"))
    counts = {s: int(len(np.load(os.path.join(d, s), mmap_mode="r"))) for s in shards}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"kind": "windows", "window": L, "shards": shards, "windows": sum(counts.values()),
                   "per_shard": counts, "stats": results, **extra}, f, indent=2)
    return sum(counts.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "phase_b", "ja_tokenizer.yaml"))
    ap.add_argument("--subset", default="sample_10BT", help="sample_10BT (~10B tokens) or data (full, ~89B)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "phase_b", "corpus"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw", "fw2"))
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-tokens", type=int, default=16)
    ap.add_argument("--max-windows-per-doc", type=int, default=0, help="cap long documents (0 = no cap)")
    ap.add_argument("--skip-urls", default=os.path.join(ROOT, "data", "phase_b", "xlsum_eval_urls.txt"))
    ap.add_argument("--keep-raw", action="store_true", help="keep downloaded parquet shards")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    from jancm.loading import load_config
    cfg = load_config(a.config)
    pre = cfg.tokenizer.pre_tokenizer
    L = int(cfg.tokenizer.vqvae.context_length)
    if not os.path.exists(a.skip_urls):
        print(f"[warn] {a.skip_urls} missing: run prepare_summ.py first to decontaminate against XL-Sum ja")
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.startswith(a.subset + "/") and f.endswith(".parquet"))
    train_files = [f for f in files if "/train-" in f][: a.max_shards or None]
    test_files = [f for f in files if "/test-" in f]
    print(f"[corpus] {a.subset}: {len(train_files)} train shards, {len(test_files)} test shard(s), L={L}")

    jobs = []
    for split, fl in (("dev", test_files), ("train", train_files)):
        d = os.path.join(a.out, split)
        os.makedirs(d, exist_ok=True)
        for i, f in enumerate(fl):
            jobs.append((split, (f, os.path.join(d, f"windows_{i:05d}.npy"), str(pre.model_id),
                                 str(pre.special_tokens.eos_token), str(pre.special_tokens.pad_token), L,
                                 a.min_tokens, a.max_windows_per_doc, a.skip_urls, a.raw, a.keep_raw, 2048)))
    results = {"train": {}, "dev": {}}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_shard, j): s for s, j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            fname, n, st, status = fut.result()
            results[futs[fut]][fname] = st
            print(f"[{k}/{len(jobs)}] {fname}: {n} windows ({status}) {st or ''}", flush=True)
    for split in ("train", "dev"):
        n = write_meta(os.path.join(a.out, split), L, results[split],
                       {"source": f"{REPO}/{a.subset}", "pretokenizer": str(pre.model_id),
                        "eos": str(pre.special_tokens.eos_token), "pad": str(pre.special_tokens.pad_token)})
        print(f"[write] {split}: {n} windows = {n * L / 1e9:.2f}B positions")


if __name__ == "__main__":
    main()
