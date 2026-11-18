"""Multi-process checks (torchrun, 2 CPU workers, gloo) and single-process EMA / batching checks."""
import os
import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf

from conftest import LEVELS, V_TOK
from jancm.distributed import resolve_batching, resolve_resume
from test_phase_b import tiny_trainable_vqvae

HERE = os.path.dirname(os.path.abspath(__file__))


def test_ddp_matches_global_batch():
    env = dict(os.environ, OMP_NUM_THREADS="1", MASTER_ADDR="127.0.0.1")
    # Explicit localhost rendezvous: --standalone resolves the hostname, which hangs in
    # sandboxes without DNS for it.
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--nproc_per_node", "2",
                        "--rdzv_backend", "c10d", "--rdzv_endpoint", "127.0.0.1:29533", "--local_addr", "127.0.0.1",
                        os.path.join(HERE, "ddp_worker.py")], capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0 and "DDP_OK" in r.stdout, (r.stdout[-3000:] + r.stderr[-3000:])


def test_deferred_ema_accumulates_like_one_large_batch():
    """Deferred EMA over two micro-batches == deferred EMA over their concatenation, and its
    level-0 update equals the original immediate update. (Later levels legitimately differ
    from the immediate path, which quantizes with the codebook it has just updated.)"""
    a, _ = tiny_trainable_vqvae(dual=False)
    b, _ = tiny_trainable_vqvae(dual=False)
    c, _ = tiny_trainable_vqvae(dual=False)
    for v in (a, b, c):
        v.vq.fine_drop_prob = 0.0
        v.train()
    b.vq.defer_ema = c.vq.defer_ema = True
    x = torch.randint(2, V_TOK, (8, LEVELS[-1]), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a.vq(a._encode_pre_quant(x).float())
        b.vq(b._encode_pre_quant(x).float())
        for half in (x[:4], x[4:]):
            c.vq(c._encode_pre_quant(half).float())
    assert not torch.equal(a.vq.embeddings[0].weight, b.vq.embeddings[0].weight), "deferred must wait"
    b.vq.apply_deferred_ema()
    c.vq.apply_deferred_ema()
    assert torch.allclose(a.vq.embeddings[0].weight, b.vq.embeddings[0].weight, atol=1e-5)
    for i in range(len(LEVELS)):
        assert torch.allclose(b.vq.embeddings[i].weight, c.vq.embeddings[i].weight, atol=1e-5), i
        assert torch.allclose(getattr(b.vq, f"ema_cluster_size_{i}"), getattr(c.vq, f"ema_cluster_size_{i}"))


def test_resolve_batching():
    T = OmegaConf.create({"batch_size": 512, "micro_batch_size": 64})
    assert resolve_batching(T, 1) == (64, 8) and resolve_batching(T, 8) == (64, 1) and resolve_batching(T, 16) == (32, 1)
    with pytest.raises(ValueError):
        resolve_batching(OmegaConf.create({"batch_size": 100, "micro_batch_size": 64}), 3)
    assert resolve_batching(OmegaConf.create({"batch_size": 256, "grad_accum": 2}), 4) == (32, 2)


def test_resolve_resume(tmp_path):
    assert resolve_resume("auto", str(tmp_path)) == ""
    (tmp_path / "last.pt").write_bytes(b"x")
    assert resolve_resume("auto", str(tmp_path)) == str(tmp_path / "last.pt")
    assert resolve_resume("other.pt", str(tmp_path)) == "other.pt"
