"""Download, filter, tokenise and write the ja->en (Phase A) or en->ja (Phase B) translation data.

Outputs (under --out, default data/processed):
  train/, dev/           src.npy [N, P] int32, tgt.npy [N, L] int32, source.npy, meta.json
  dev/dev.jsonl          dev pairs as text, for greedy dev BLEU during training
  test/<name>/test.jsonl {"src": ja, "ref": en}; untruncated, scored by scripts/evaluate.py

Corpora (licences in README.md):
  kftt     CC BY-SA 3.0  train / dev / test
  mitsua   CC0           train only: Wikidata descriptions (94,480 rows, noisy)
  mitsua_labels CC0      train only: Wikidata entity labels (1,476,205 rows; names, e.g. Belgium/ベルギー)
  tatoeba  CC BY 2.0 FR  train only (OPUS v2023-04-12)
  alt      CC BY 4.0     train / dev / test
  flores   CC BY-SA 4.0  test only (devtest). Gated: accept the terms on Hugging Face and log in
                         (hf auth login); skipped with a warning otherwise.
  ntrex    CC BY-SA 4.0  test only (newstest2019 ja reference -> original English source)

Usage:
  python scripts/prepare_data.py --config configs/ja_en_ncm.yaml
  python scripts/prepare_data.py --config configs/phase_b/en_ja.yaml --direction en-ja   # Phase B
  python scripts/prepare_data.py --config configs/ja_en_ncm.yaml --only alt,ntrex --limit 2000   # smoke test
"""
import argparse
import io
import json
import os
import sys
import tarfile
import unicodedata
import urllib.request
import zipfile
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm.loading import load_config, resolve  # noqa: E402
from jancm.text import encode_batch, pretokenizer_from_config, target_window  # noqa: E402
from jancm import ROOT  # noqa: E402

SOURCES = ["kftt", "mitsua", "tatoeba", "alt", "mitsua_labels"]   # order defines source.npy ids; append only
TESTS = ["flores", "ntrex", "kftt", "alt"]
KFTT_URL = "https://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz"
TATOEBA_URL = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-ja.txt.zip"
NTREX_URL = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128/newstest2019-{}.txt"


# ---------------------------------------------------------------- download helpers
def fetch(url, cache):
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if not os.path.exists(cache):
        print(f"[download] {url}", flush=True)
        tmp = cache + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, cache)
    return cache


def hf_file(repo, filename, raw):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset", cache_dir=os.path.join(raw, "hf"))


# ---------------------------------------------------------------- corpus readers -> {split: [(ja, en)]}
def read_kftt(raw):
    tar = tarfile.open(fetch(KFTT_URL, os.path.join(raw, "kftt-data-1.0.tar.gz")))

    def lines(name):
        return tar.extractfile(f"kftt-data-1.0/data/orig/{name}").read().decode("utf-8").splitlines()

    return {split: list(zip(lines(f"kyoto-{split}.ja"), lines(f"kyoto-{split}.en")))
            for split in ("train", "dev", "test")}


def read_mitsua(raw, filename="wikidata_descriptions_filtered_dedup_en-ja.jsonl"):
    path = hf_file("Mitsua/wikidata-parallel-descriptions-en-ja", filename, raw)
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["translation"]
            pairs.append((t["ja"], t["en"]))
    return {"train": pairs}


def read_tatoeba(raw):
    z = zipfile.ZipFile(fetch(TATOEBA_URL, os.path.join(raw, "tatoeba-en-ja.txt.zip")))
    ja = z.read("Tatoeba.en-ja.ja").decode("utf-8").splitlines()
    en = z.read("Tatoeba.en-ja.en").decode("utf-8").splitlines()
    return {"train": list(zip(ja, en))}


def read_alt(raw):
    import pyarrow.parquet as pq
    out = {}
    for split, name in (("train", "train"), ("dev", "validation"), ("test", "test")):
        t = pq.read_table(hf_file("mutiyama/alt", f"alt-parallel/{name}-00000-of-00001.parquet", raw))
        out[split] = [(r["ja"], r["en"]) for r in t.column("translation").to_pylist() if r.get("ja") and r.get("en")]
    return out


def read_flores(raw):
    def rows(lang):
        with open(hf_file("openlanguagedata/flores_plus", f"devtest/{lang}.jsonl", raw), encoding="utf-8") as f:
            return {r["id"]: r["text"] for r in map(json.loads, f)}
    ja, en = rows("jpn_Jpan"), rows("eng_Latn")
    return {"test": [(ja[i], en[i]) for i in sorted(ja) if i in en]}


def read_ntrex(raw):
    # NTREX-128 was translated from English, so the Japanese reference is our source.
    ja = open(fetch(NTREX_URL.format("ref.jpn"), os.path.join(raw, "ntrex.jpn.txt")), encoding="utf-8").read().splitlines()
    en = open(fetch(NTREX_URL.format("src.eng"), os.path.join(raw, "ntrex.eng.txt")), encoding="utf-8").read().splitlines()
    assert len(ja) == len(en), (len(ja), len(en))
    return {"test": list(zip(ja, en))}


def read_mitsua_labels(raw):
    return read_mitsua(raw, "wikidata_labels_filtered_en-ja.jsonl")


READERS = {"kftt": read_kftt, "mitsua": read_mitsua, "mitsua_labels": read_mitsua_labels, "tatoeba": read_tatoeba, "alt": read_alt,
           "flores": read_flores, "ntrex": read_ntrex}


# ---------------------------------------------------------------- filtering
def clean(s):
    return " ".join(s.replace("\u3000", " ").split()).strip()


def norm_key(s):
    return unicodedata.normalize("NFKC", s).lower().replace(" ", "")


def plausible(ja, en, name):
    if not ja or not en:
        return False
    if name == "mitsua":
        # Descriptions are short and noisy: drop near-empty, untranslated or wildly unbalanced pairs.
        if len(en) < 3 or norm_key(ja) == norm_key(en):
            return False
        ratio = len(ja) / max(len(en), 1)
        if not 0.05 <= ratio <= 3.0:
            return False
    if name == "mitsua_labels":
        # Names: keep only real transliterations/translations, not copied Latin strings.
        if norm_key(ja) == norm_key(en) or ja.isascii():
            return False
    return True


def orient(pairs, direction):
    """Pairs are (ja, en) internally; return (source, target) for the requested direction."""
    return list(pairs) if direction == "ja-en" else [(e, j) for j, e in pairs]


def tokenise_pairs(pairs, pretok, P, L, eos, pad, add_eos, fill, drops):
    """pairs: (source, target) texts."""
    ja_ids = encode_batch(pretok, [p[0] for p in pairs])
    en_ids = encode_batch(pretok, [p[1] for p in pairs])
    src, tgt, kept = [], [], []
    for (ja, en), s, t in zip(pairs, ja_ids, en_ids):
        if len(s) > P:
            drops["src_too_long"] += 1
            continue
        tw = target_window(t, L, eos, pad, add_eos, fill)
        if tw is None:
            drops["tgt_too_long"] += 1
            continue
        src.append(s + [pad] * (P - len(s)))
        tgt.append(tw)
        kept.append((ja, en, len(s), len(t)))
    return src, tgt, kept


def write_split(out_dir, rows_by_source, meta_extra):
    os.makedirs(out_dir, exist_ok=True)
    src, tgt, source, counts = [], [], [], {}
    for i, name in enumerate(SOURCES):
        s, t = rows_by_source.get(name, ([], []))
        src += s
        tgt += t
        source += [i] * len(s)
        counts[name] = len(s)
    np.save(os.path.join(out_dir, "src.npy"), np.asarray(src, dtype=np.int32))
    np.save(os.path.join(out_dir, "tgt.npy"), np.asarray(tgt, dtype=np.int32))
    np.save(os.path.join(out_dir, "source.npy"), np.asarray(source, dtype=np.uint8))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"sources": SOURCES, "counts": counts, **meta_extra}, f, indent=2, ensure_ascii=False)
    print(f"[write] {out_dir}: {counts}")


def pct(xs):
    if not xs:
        return {}
    a = np.asarray(xs)
    return {f"p{q}": int(np.percentile(a, q)) for q in (50, 90, 99)}


def write_jsonl(path, pairs):
    """pairs: (source, target) texts."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for src, ref in pairs:
            f.write(json.dumps({"src": src, "ref": ref}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "ja_en_ncm.yaml"))
    ap.add_argument("--out", default="")
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--only", default="", help="comma-separated subset of corpora to process")
    ap.add_argument("--limit", type=int, default=0, help="cap pairs per corpus split (smoke tests)")
    ap.add_argument("--max-per-source", default="", help="e.g. mitsua=500000")
    ap.add_argument("--direction", choices=["ja-en", "en-ja"], default="ja-en")
    a = ap.parse_args()

    cfg = load_config(a.config)
    out = a.out or resolve(cfg.data.dir, ROOT)
    pretok = pretokenizer_from_config(cfg)
    P, L = int(cfg.model.prefix_len), int(cfg.tokenizer.vqvae.context_length)
    eos, pad = pretok.eos_token_id, pretok.pad_token_id
    add_eos = bool(cfg.data.get("target_eos", True))
    fill = str(cfg.data.get("target_fill", "pad"))
    caps = {k: int(v) for k, v in (kv.split("=") for kv in a.max_per_source.split(",") if kv)}
    only = set(a.only.split(",")) if a.only else set(SOURCES + TESTS)
    print(f"[config] {a.direction} P={P} L={L} eos={eos} pad={pad} add_eos={add_eos} fill={fill}")

    corpora = {}
    for name in dict.fromkeys(SOURCES + TESTS):
        if name not in only:
            continue
        try:
            corpora[name] = READERS[name](a.raw)
        except Exception as e:  # gated or offline corpora should not stop the rest
            print(f"[warn] {name}: {type(e).__name__}: {str(e)[:200]} -- skipped", flush=True)

    # Test sets first, so training data can be decontaminated against them.
    test_keys = set()
    for name in TESTS:
        pairs = [(clean(j), clean(e)) for j, e in corpora.get(name, {}).get("test", [])]
        pairs = [p for p in pairs if p[0] and p[1]][: a.limit or None]
        if not pairs:
            continue
        write_jsonl(os.path.join(out, "test", name, "test.jsonl"), orient(pairs, a.direction))
        test_keys |= {norm_key(j) for j, _ in pairs} | {norm_key(e) for _, e in pairs}
        print(f"[test] {name}: {len(pairs)} pairs")

    meta = {"prefix_len": P, "target_len": L, "eos_id": eos, "pad_id": pad, "target_eos": add_eos, "target_fill": fill, "direction": a.direction,
            "pretokenizer": str(cfg.tokenizer.pre_tokenizer.model_id)}
    for split in ("train", "dev"):
        rows, stats, dev_text = {}, {}, []
        seen = set()
        for name in SOURCES:
            raw_pairs = corpora.get(name, {}).get(split, [])
            if not raw_pairs:
                continue
            drops = Counter()
            pairs = []
            for j, e in raw_pairs:
                j, e = clean(j), clean(e)
                if not plausible(j, e, name):
                    drops["implausible"] += 1
                    continue
                key = (norm_key(j), norm_key(e))
                if key in seen:
                    drops["duplicate"] += 1
                    continue
                if split == "train" and (key[0] in test_keys or key[1] in test_keys):
                    drops["test_overlap"] += 1
                    continue
                seen.add(key)
                pairs.append((j, e))
            cap = a.limit or caps.get(name, 0)
            if cap and len(pairs) > cap:
                idx = np.random.default_rng(42).choice(len(pairs), cap, replace=False)
                pairs = [pairs[i] for i in sorted(idx)]
            s, t, kept = tokenise_pairs(orient(pairs, a.direction), pretok, P, L, eos, pad, add_eos, fill, drops)
            rows[name] = (s, t)
            stats[name] = {"raw": len(raw_pairs), "kept": len(s), "dropped": dict(drops),
                           "src_tokens": pct([k[2] for k in kept]), "tgt_tokens": pct([k[3] for k in kept])}
            dev_text += [(k[0], k[1]) for k in kept]
            print(f"[{split}] {name}: {stats[name]}", flush=True)
        if rows:
            write_split(os.path.join(out, split), rows, {**meta, "stats": stats})
            if split == "dev":
                write_jsonl(os.path.join(out, "dev", "dev.jsonl"), dev_text)


if __name__ == "__main__":
    main()
