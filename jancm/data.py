"""Token-id datasets written by scripts/prepare_data.py, and a mixture batch sampler."""
import json
import os

import numpy as np
import torch


def pad_to(ids, length, pad_id):
    ids = list(ids)[:length]
    return ids + [pad_id] * (length - len(ids))


class PairSplit:
    """A split stored as src.npy [N, P], tgt.npy [N, L] (int32) and source.npy [N] (uint8)."""

    def __init__(self, directory):
        self.dir = directory
        self.src = np.load(os.path.join(directory, "src.npy"), mmap_mode="r")
        self.tgt = np.load(os.path.join(directory, "tgt.npy"), mmap_mode="r")
        self.source = np.load(os.path.join(directory, "source.npy"))
        with open(os.path.join(directory, "meta.json")) as f:
            self.meta = json.load(f)
        self.names = self.meta["sources"]
        assert len(self.src) == len(self.tgt) == len(self.source)

    def __len__(self):
        return len(self.src)

    def take(self, rows, device):
        rows = np.sort(np.asarray(rows))
        src = torch.from_numpy(np.ascontiguousarray(self.src[rows])).long().to(device)
        tgt = torch.from_numpy(np.ascontiguousarray(self.tgt[rows])).long().to(device)
        return src, tgt


class MixtureSampler:
    """Infinite iterator of row indices: pick a corpus by ``weights``, then a row uniformly.

    Weights are corpus proportions per batch, so a large noisy corpus (Mitsua Wikidata
    descriptions) cannot crowd out KFTT just by size. Corpora missing from the split are
    skipped and the remaining weights renormalised.
    """

    def __init__(self, split, weights, batch_size, seed=42):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.uniform = split.source is None
        self.n = len(split)
        if self.uniform:
            return
        self.pools, ws = [], []
        for i, name in enumerate(split.names):
            rows = np.nonzero(split.source == i)[0]
            w = float(weights.get(name, 0.0))
            if len(rows) and w > 0:
                self.pools.append(rows)
                ws.append(w)
        if not self.pools:
            raise ValueError(f"no corpus with positive weight among {split.names}")
        self.p = np.array(ws) / sum(ws)

    def __iter__(self):
        return self

    def __next__(self):
        if self.uniform:
            return self.rng.integers(0, self.n, size=self.batch_size)
        which = self.rng.choice(len(self.pools), size=self.batch_size, p=self.p)
        return np.array([pool[self.rng.integers(len(pool))] for pool in (self.pools[w] for w in which)])


class WindowSplit:
    """Fixed-length token windows for unconditional training (and tokenizer training).

    ``path`` is either a directory written by scripts/prepare_corpus.py (meta.json listing
    int32 shard files of shape [n, L]) or a flat token stream ``*.bin`` (upstream LM1B
    format: uint16, windows packed back to back). ``take`` returns the same (src, tgt)
    pair as PairSplit, with a one-token null prefix (paper Table 6: prefix length 1).
    """

    def __init__(self, path, window=None, dtype="uint16", prefix_token=None):
        self.dir = path
        if path.endswith(".bin"):
            assert window, "window length required for .bin streams"
            flat = np.memmap(path, dtype=np.dtype(dtype), mode="r")
            n = len(flat) // window
            self.shards = [flat[: n * window].reshape(n, window)]
            self.meta = {"kind": "windows", "window": window, "source_file": path}
        else:
            with open(os.path.join(path, "meta.json")) as f:
                self.meta = json.load(f)
            self.shards = [np.load(os.path.join(path, s), mmap_mode="r") for s in self.meta["shards"]]
        self.window = int(self.meta["window"])
        self.sizes = np.array([len(s) for s in self.shards])
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)])
        self.names = ["corpus"]
        self.source = None          # single corpus: MixtureSampler samples uniformly
        self.prefix_token = prefix_token

    def __len__(self):
        return int(self.starts[-1])

    def windows(self, rows):
        rows = np.sort(np.asarray(rows))
        shard = np.searchsorted(self.starts, rows, side="right") - 1
        out = np.empty((len(rows), self.window), dtype=np.int64)
        for s in np.unique(shard):
            m = shard == s
            out[m] = self.shards[s][rows[m] - self.starts[s]]
        return out

    def take(self, rows, device):
        tgt = torch.from_numpy(self.windows(rows)).to(device)
        assert self.prefix_token is not None, "WindowSplit needs prefix_token for generator training"
        src = torch.full((tgt.shape[0], 1), int(self.prefix_token), dtype=torch.long, device=device)
        return src, tgt


def open_split(cfg_data, split, prefix_token=None, root="."):
    """Open the train/dev split named in a config ``data`` block.

    kind: pairs   -> PairSplit(<dir>/<split>)
    kind: windows -> WindowSplit(<data.<split>> or <dir>/<split>)
    """
    kind = str(cfg_data.get("kind", "pairs"))
    if kind == "pairs":
        return PairSplit(os.path.join(_abs(cfg_data.dir, root), split))
    path = cfg_data.get(split) or os.path.join(str(cfg_data.dir), split)
    return WindowSplit(_abs(str(path), root), window=cfg_data.get("window"),
                       dtype=str(cfg_data.get("dtype", "uint16")), prefix_token=prefix_token)


def _abs(p, root):
    return p if os.path.isabs(p) else os.path.join(root, p)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_test_set(directory):
    """Test sets are JSONL rows of {"src": ja, "ref": en} (no pickle)."""
    rows = read_jsonl(os.path.join(directory, "test.jsonl"))
    return [r["src"] for r in rows], [r["ref"] for r in rows]
