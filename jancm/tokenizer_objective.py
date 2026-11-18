"""Multiscale residual VQ-VAE (concept tokenizer) training objective (paper Eq. 9).

L_tok = CE(x | z_hat^(k)) + beta * MSE(z, sg[z_hat^(k)]) + MSE(sg[z], z_hat^(k))

The VQ terms, EMA codebook updates, dead-code restarts, straight-through estimator and
plain nested dropout (``aux.fine_drop.prob``) already live in the vendored quantizer's
forward(). This module adds what the upstream file lacks:

  * the reconstruction cross-entropy, computed in row chunks with activation
    recomputation, because B x 128 x 151,680 logits do not fit in memory;
  * the "dual" nested-dropout recipe of the released LM1B config
    (``fine_drop: {dual: true, w_full: 0.7, w_trunc: 0.3}``): every step decodes both the
    complete hierarchy and a randomly truncated one;
  * per-level reconstruction metrics.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def decoder_hidden(dec, z):
    """ncm/models/vae.py Decoder.forward without the output projection."""
    if dec.up is not None:
        z = dec.up(z.transpose(1, 2)).transpose(1, 2)
    if dec.wpe is not None:
        z = z + dec.wpe(torch.arange(z.shape[1], dtype=torch.long, device=z.device))
    x = dec.middle_block(z)
    for blk in dec.extra_blocks:
        x = blk(x)
    return dec.ln_f(x)


def _ce_chunk(h, w, t, ignore_index):
    return F.cross_entropy((h @ w.t()).float(), t, reduction="sum", ignore_index=ignore_index)


def chunked_ce(hidden, weight, targets, chunk_rows=4096, ignore_index=-100):
    """Mean CE over rows of hidden [N, d] against weight [V, d], without storing N x V logits.

    Each chunk's logits are recomputed in backward (activation checkpointing), so peak
    memory is one chunk of logits instead of all of them. Targets equal to ignore_index
    are excluded from the sum and the count.
    """
    hidden = hidden.reshape(-1, hidden.shape[-1])
    targets = targets.reshape(-1)
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, hidden.shape[0], chunk_rows):
        h, t = hidden[i:i + chunk_rows], targets[i:i + chunk_rows]
        if torch.is_grad_enabled() and (h.requires_grad or weight.requires_grad):
            total = total + checkpoint(_ce_chunk, h, weight, t, ignore_index, use_reentrant=False)
        else:
            total = total + _ce_chunk(h, weight, t, ignore_index)
    return total / (targets != ignore_index).sum().clamp(min=1)


@torch.no_grad()
def decode_argmax(dec, z, chunk_rows=4096):
    h = decoder_hidden(dec, z)
    B, L, D = h.shape
    flat = h.reshape(-1, D)
    w = dec.output_layer.weight
    out = torch.cat([(flat[i:i + chunk_rows] @ w.t()).argmax(-1) for i in range(0, flat.shape[0], chunk_rows)])
    return out.view(B, L)


class DualFineDrop:
    """Nested dropout that always trains on the full hierarchy plus one truncation."""

    def __init__(self, n_levels, w_full=0.7, w_trunc=0.3, min_keep=1):
        self.w_full, self.w_trunc = float(w_full), float(w_trunc)
        self.kmin, self.kmax = max(int(min_keep) - 1, 0), n_levels - 2

    @classmethod
    def from_vq_config(cls, vq_cfg, n_levels):
        fd = (vq_cfg.get("aux") or {}).get("fine_drop") or {}
        if not fd.get("dual", False):
            return None
        return cls(n_levels, fd.get("w_full", 0.7), fd.get("w_trunc", 0.3), fd.get("min_keep", 1))

    def pick(self):
        return int(torch.randint(self.kmin, self.kmax + 1, ()))


class CodeCorruption:
    """``aux.code_corruption`` of the released LM1B tokenizer config (our first trainer ignored it).

    Each step the decoder also reconstructs x from codes in which a fraction r ~ U(low, high)
    per sequence (level 0 kept with skip_level0) is replaced by uniform random codes. The
    corrupted reconstruction enters the decoder straight-through (z + (z_c - z).detach()), so
    the encoder also gets gradients from it and learns codes that survive wrong neighbours,
    as generated codes will be. The released config has no weight for this term; ``weight``
    (ours, default 0.5) scales it next to the clean (dual) terms.
    """

    def __init__(self, low=0.0, high=1.0, skip_level0=True, weight=0.5):
        self.low, self.high, self.skip0, self.weight = float(low), float(high), bool(skip_level0), float(weight)

    @classmethod
    def from_vq_config(cls, vq_cfg):
        c = (vq_cfg.get("aux") or {}).get("code_corruption") or {}
        if not c.get("enabled", False):
            return None
        assert str(c.get("mode", "uniform")) == "uniform" and bool(c.get("random_rate", True)), c
        return cls(c.get("low", 0.0), c.get("high", 1.0), c.get("skip_level0", True), c.get("weight", 0.5))

    @torch.no_grad()
    def __call__(self, codes, levels, vocab_sizes):
        B = codes.shape[0]
        r = self.low + (self.high - self.low) * torch.rand(B, 1, device=codes.device)
        out, off = codes.clone(), 0
        for k, (l, v) in enumerate(zip(levels, vocab_sizes)):
            if not (k == 0 and self.skip0):
                seg = out[:, off:off + l]
                m = torch.rand(seg.shape, device=codes.device) < r
                out[:, off:off + l] = torch.where(m, torch.randint(0, v, seg.shape, device=codes.device), seg)
            off += l
        return out


@torch.no_grad()
def decoder_robustness(vqvae, x, rates=(0.3, 0.5), chunk_rows=4096):
    """Token accuracy (%) when a fraction of real codes is replaced: levels 1-6 ("mid") and
    every level but 0 ("all"). Released LM1B tokenizer at 0.5: mid 99.7, all 80.3."""
    vq = vqvae.vq
    K = len(vq.levels)
    g = torch.Generator(device=x.device).manual_seed(0)
    codes = vq.z_to_idx(vqvae._encode_pre_quant(x).float())
    codes = torch.cat(codes, dim=1) if isinstance(codes, (list, tuple)) else codes
    out = {}
    for scope, lv in (("mid", range(1, min(7, K))), ("all", range(1, K))):
        for r in rates:
            c, off = codes.clone(), 0
            for k, (l, v) in enumerate(zip(vq.levels, vq.vqvae_vocab_sizes)):
                if k in lv:
                    seg = c[:, off:off + l]
                    m = torch.rand(seg.shape, device=x.device, generator=g) < r
                    c[:, off:off + l] = torch.where(m, torch.randint(0, v, seg.shape, device=x.device, generator=g), seg)
                off += l
            z = vq.idx_to_zhats(c, return_zlevels=False)[0][-1]
            out[f"robust_{scope}_{r}"] = round(float((decode_argmax(vqvae.decoder, z, chunk_rows) == x).float().mean()) * 100, 3)
    return out


def tokenizer_loss(vqvae, x, dual=None, chunk_rows=4096, autocast_ctx=None, corrupt=None):
    """Return (loss, stats) for one batch of token windows x [B, L].

    The quantizer runs in fp32 even under bf16 autocast: nearest-codeword search in bf16
    changes the selected codes.
    """
    vq = vqvae.vq
    ctx = autocast_ctx if autocast_ctx is not None else torch.autocast("cpu", enabled=False)
    with ctx:
        z = vqvae._encode_pre_quant(x)
    vq.keep_cumulative = dual is not None
    if dual is not None:
        vq.fine_drop_prob = 0.0   # the dual recipe replaces the quantizer's own truncation
    with torch.autocast(device_type=x.device.type, enabled=False):
        z_q, vq_loss, util_levels, util_global, _ = vq(z.float())
    terms = [(1.0, z_q)]
    k_trunc = None
    if dual is not None:
        k_trunc = dual.pick()
        terms = [(dual.w_full, z_q), (dual.w_trunc, vq.last_cumulative[k_trunc])]
        vq.last_cumulative = None
    if corrupt is not None:
        with torch.no_grad():
            codes = vq.z_to_idx(z.detach().float())
            codes = torch.cat(codes, dim=1) if isinstance(codes, (list, tuple)) else codes
            z_c = vq.idx_to_zhats(corrupt(codes, vq.levels, vq.vqvae_vocab_sizes), return_zlevels=False)[0][-1]
        zf = z.float()
        terms.append((corrupt.weight, zf + (z_c - zf).detach()))
    dec = vqvae.decoder
    ce_terms = []
    with ctx:
        for w, zz in terms:
            h = decoder_hidden(dec, zz.to(z.dtype))
            ce_terms.append((w, chunked_ce(h, dec.output_layer.weight, x, chunk_rows)))
    ce = sum(w * c for w, c in ce_terms)
    loss = ce + vq_loss
    stats = {"ce": float(ce.detach()), "vq": float(vq_loss.detach()),
             "ce_full": float(ce_terms[0][1].detach()), "util_global_ever": round(float(util_global), 4)}
    if k_trunc is not None:
        stats.update(ce_trunc=float(ce_terms[1][1].detach()), trunc_levels=k_trunc + 1)
    if corrupt is not None:
        stats.update(ce_corrupt=float(ce_terms[-1][1].detach()))
    return loss, stats


@torch.no_grad()
def kmeans_init_codebooks(vqvae, x, iters=10, max_vectors=262144, chunk=65536):
    """Data-dependent codebook init, level by level on the real residuals.

    Without it the N(0, 1) codebooks (norm ~ sqrt(d)) sit far from the area-downsampled
    residuals (much smaller norms), one small-norm code wins every assignment, the EMA only
    moves that code, and every level collapses to about one code (observed on LM1B: 6%
    reconstruction, flat at the unigram CE). Uses the vendored quantizer's own
    ``kmeans_init_level``, which nothing upstream calls.

    x: [N, L] token windows; level 0 has one vector per window, so N should be >= its codebook.
    """
    vq = vqvae.vq
    was_training = vqvae.training
    vqvae.eval()
    z = vqvae._encode_pre_quant(x).float()
    B, L, C = z.shape
    z_rest = z.clone()
    sizes = []
    for i, level in enumerate(vq.levels):
        flat_BLC = vq._down_for_quant(z_rest, i, level) if i != len(vq.levels) - 1 else z_rest
        flat = flat_BLC.reshape(-1, C)
        fit = flat if flat.shape[0] <= max_vectors else flat[torch.randperm(flat.shape[0], device=flat.device)[:max_vectors]]
        vq.kmeans_init_level(i, fit.contiguous(), n_iters=iters)
        w = vq.embeddings[i].weight
        idx = torch.cat([vq._distances(flat[s:s + chunk], w).argmin(1) for s in range(0, flat.shape[0], chunk)]).view(B, level)
        q = vq.embeddings[i](idx)
        q = vq._up_to_full(q, i, L) if i != len(vq.levels) - 1 else q
        z_rest = z_rest - vq._apply_quant_resi(q, i)
        sizes.append(int(flat.shape[0]))
    vqvae.train(was_training)
    return sizes


@torch.no_grad()
def evaluate_tokenizer(vqvae, batches, pad_id, chunk_rows=4096):
    """Per-level exact-token reconstruction over non-PAD positions, exact-window match and
    per-level codebook utilisation, on an iterable of [B, L] windows (eval mode)."""
    vq = vqvae.vq
    K = len(vq.levels)
    correct = torch.zeros(K, dtype=torch.float64)
    total, exact, n = 0, 0, 0
    used = [torch.zeros(v, dtype=torch.bool) for v in vq.vqvae_vocab_sizes]
    was_training = vqvae.training
    vqvae.eval()
    for x in batches:
        codes = vqvae.encode(x)
        (zh,) = vq.idx_to_zhats(codes, return_zlevels=False)
        mask = x != pad_id
        total += int(mask.sum())
        for k, z in enumerate(zh):
            rec = decode_argmax(vqvae.decoder, z, chunk_rows)
            correct[k] += int(((rec == x) & mask).sum())
        exact += int(((rec == x) | ~mask).all(dim=1).sum())
        n += x.shape[0]
        off = 0
        for k, l in enumerate(vq.levels):
            used[k][codes[:, off:off + l].flatten().cpu()] = True
            off += l
    vqvae.train(was_training)
    acc = (100 * correct / max(total, 1)).tolist()
    return {"recon_acc_per_level": [round(a, 2) for a in acc], "recon_acc_full": round(acc[-1], 3),
            "exact_window_pct": round(100 * exact / max(n, 1), 2),
            "codebook_util_pct": [round(100 * float(u.float().mean()), 1) for u in used], "n_windows": n}
