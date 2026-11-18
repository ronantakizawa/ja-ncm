"""Run by tests/test_distributed.py under torchrun (2 CPU processes, gloo).

Checks, for the NCM generator, the AR baseline and the tokenizer:
  * DDP gradients with 2 micro-steps per rank (no_sync accumulation) equal the gradients
    of one process on the concatenated global batch;
  * after the deferred EMA step, every rank holds identical codebooks, equal to a
    single-process EMA update over the global batch.
Prints "DDP_OK" on rank 0 when everything matches.
"""
import copy
import os
import sys

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from conftest import EOS, LEVELS, PAD, V_TOK, make_vqvae, padded_prefixes  # noqa: E402
from jancm.ar import ARTranslator, right_to_left  # noqa: E402
from jancm.distributed import Dist  # noqa: E402
from jancm.ncm_objective import build_batch, level_weights, ncm_loss, predicted_levels  # noqa: E402
from jancm.tokenizer_objective import tokenizer_loss  # noqa: E402
from test_phase_b import lm1b_style_net, tiny_trainable_vqvae  # noqa: E402

D = Dist("cpu")
assert D.world == 2, D.world
MICRO, ACCUM = 3, 2
G = MICRO * ACCUM * D.world


def global_data(seed):
    g = torch.Generator().manual_seed(seed)
    src = padded_prefixes(G, 6, seed=seed)
    tgt = torch.randint(2, V_TOK, (G, LEVELS[-1]), generator=g)
    return src, tgt


def my_slices(n_micro):
    """Rows of micro-step k on this rank (rank-major layout of the global batch)."""
    per_rank = MICRO * n_micro
    base = D.rank * per_rank
    return [slice(base + k * MICRO, base + (k + 1) * MICRO) for k in range(n_micro)]


def check_grads(name, ddp_model, ref_model, run_micro, run_ref):
    for p in ddp_model.parameters():
        p.grad = None
    for k, sl in enumerate(my_slices(ACCUM)):
        with D.no_sync(ddp_model, k == ACCUM - 1):
            (run_micro(ddp_model, sl) / ACCUM).backward()
    run_ref(ref_model).backward()
    bad = []
    for (n, p), (_, q) in zip(ddp_model.module.named_parameters(), ref_model.named_parameters()):
        if p.requires_grad and q.grad is not None:
            if p.grad is None or not torch.allclose(p.grad, q.grad, atol=2e-5, rtol=1e-4):
                bad.append(n)
    assert not bad, f"{name}: DDP gradients differ from the global batch for {bad[:5]}"


# ---- NCM generator (query_level0, LM1B-style) ----
vqvae = make_vqvae()
torch.manual_seed(0)
net = lm1b_style_net(vqvae, prefix_len=6).train()
for m in net.modules():
    if isinstance(m, torch.nn.Dropout):
        m.p = 0.0
ref = copy.deepcopy(net)
ref.vqvae_local = net.vqvae_local
model = D.wrap(net)
src, tgt = global_data(1)
w = level_weights(predicted_levels(net), 0.0)


def ncm_micro(m, sl):
    inputs, codes = build_batch(m.module, vqvae, src[sl], tgt[sl], PAD)
    return ncm_loss(m(**inputs), codes, w)[0]


def ncm_ref(m):
    inputs, codes = build_batch(m, vqvae, src, tgt, PAD)
    return ncm_loss(m(**inputs), codes, w)[0]


check_grads("ncm", model, ref, ncm_micro, ncm_ref)

# ---- AR baseline ----
torch.manual_seed(0)
ar = ARTranslator(V_TOK, 16, n_layer=2, n_head=4, n_embd=32, dropout=0.0, rope_base=10000, max_len=40).set_eos(EOS)
ar.ce_chunk_rows = 5     # exercise the chunked loss
torch.nn.init.normal_(ar.out_proj.weight, std=0.02)   # zero init would give zero gradients upstream of it
ar_ref = copy.deepcopy(ar)
ar_model = D.wrap(ar)
asrc = right_to_left(src, PAD)
atgt = tgt[:, :8].clone()
atgt[:, 6] = EOS
check_grads("ar", ar_model, ar_ref,
            lambda m, sl: m(asrc[sl], atgt[sl], 2, PAD, PAD),
            lambda m: m(asrc, atgt, 2, PAD, PAD))

# ---- tokenizer: gradients (nested dropout off, so both sides see the same graph) ----
tok, _ = tiny_trainable_vqvae(dual=False)
tok.vq.fine_drop_prob = 0.0
for m in tok.modules():
    if isinstance(m, torch.nn.Dropout):
        m.p = 0.0
tok.train()
tok_ref = copy.deepcopy(tok)
tok.vq.defer_ema = tok_ref.vq.defer_ema = True


class Step(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, x):
        return tokenizer_loss(self.v, x, None, chunk_rows=7)[0]


tok_model = D.wrap(Step(tok))
x_all = tgt


def tok_ref_loss(m):
    return tokenizer_loss(m, x_all, None, chunk_rows=7)[0]


for p in tok_model.parameters():
    p.grad = None
for k, sl in enumerate(my_slices(ACCUM)):
    with D.no_sync(tok_model, k == ACCUM - 1):
        (tok_model(x_all[sl]) / ACCUM).backward()
tok_ref_loss(tok_ref).backward()
bad = [n for (n, p), (_, q) in zip(tok.named_parameters(), tok_ref.named_parameters())
       if p.requires_grad and q.grad is not None and not torch.allclose(p.grad, q.grad, atol=2e-5, rtol=1e-4)]
assert not bad, f"tokenizer: DDP gradients differ for {bad[:5]}"

# ---- tokenizer: deferred EMA over micro-steps and ranks == one global update ----
tok.vq.apply_deferred_ema()        # stats from this rank's 2 micro-batches, summed over ranks
# The reference must not all-reduce: it already holds the whole global batch.
import models.quant as Q  # noqa: E402


class _NoDist:
    is_available = staticmethod(lambda: False)
    is_initialized = staticmethod(lambda: False)


_real, Q.dist = Q.dist, _NoDist
tok_ref.vq.apply_deferred_ema()    # stats from the whole global batch in one process
Q.dist = _real
for i in range(len(LEVELS)):
    a, b = tok.vq.embeddings[i].weight, tok_ref.vq.embeddings[i].weight
    assert torch.allclose(a, b, atol=1e-5), f"level {i}: deferred EMA differs from the global-batch update"
    gathered = [torch.zeros_like(a) for _ in range(D.world)]
    dist.all_gather(gathered, a.contiguous())
    assert all(torch.equal(gathered[0], t) for t in gathered), f"level {i}: codebooks differ across ranks"

if D.is_main:
    print("DDP_OK", flush=True)
D.close()
