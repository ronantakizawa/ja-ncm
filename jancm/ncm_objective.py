"""NCM generator training objective (paper Section 3.4, Algorithm 1).

The upstream repository ships inference only. This module rebuilds the training step so
that its input layout matches ``NextConceptTransformer.generate`` exactly:

  * prefix  = how generate() embeds prefix tokens: frozen VQ-VAE token embeddings
              (``raw_token_prefix``), the contextual encoder (``contextual_prefix``), the
              VQ-VAE encoder output, or a separate context encoder (``ctx_encoder``);
              right-padded with PAD and masked through ``prefix_pad_mask``;
  * targets = all K levels of codes from the frozen tokenizer, concatenated coarse to fine;
  * inputs  = cumulative reconstruction of levels 1..k-1, resized to l_k positions
              (``vq.idx_to_ncm_input``), built from *corrupted* codes;
  * level 0 = predicted from the last non-pad prefix position (``predict_level0``) or from
              learned queries prepended to the target inputs (``query_level0``).

tests/test_ncm_objective.py and tests/test_phase_b.py check that teacher forcing on
greedily generated codes reproduces those codes, for both level-0 modes.
"""
import torch
import torch.nn.functional as F


def level_weights(levels, alpha, device=None):
    """Per-position weights w_k = l_k^-alpha / sum_j l_j^(1-alpha) (paper Eq. 10).

    alpha=0 weights every code equally (the ELBO); alpha=1 weights every level equally.
    Returns a [sum(levels)] tensor that sums to 1.
    """
    lv = torch.tensor(levels, dtype=torch.float64)
    denom = (lv ** (1.0 - alpha)).sum()
    w = torch.cat([torch.full((int(l),), float(l) ** -alpha) for l in levels]).double() / denom
    return w.float().to(device)


def corruption_rates(n_levels, start, end):
    """Linearly decreasing replacement probability for the n_levels-1 conditioning levels.

    The finest level is never an input, so its rate is 0.
    """
    k = n_levels - 1
    if k <= 1:
        rates = [start] * k
    else:
        rates = [start + (end - start) * i / (k - 1) for i in range(k)]
    return rates + [0.0]


class Corruptor:
    """Conditioning-code corruption (paper Appendix F.3).

    mode=linear      : fixed per-level rates falling linearly from ``start`` to ``end``
                       (the TinyStories ablation recipe).
    mode=random_rate : one rate per sequence drawn from U(low, high), shared by its levels
                       (the ``aux.code_corruption`` block of the released LM1B config).
    mode=none        : clean conditioning.
    skip_level0 leaves the coarsest level's codes intact.
    """

    def __init__(self, n_levels, mode="linear", start=0.5, end=0.1, low=0.0, high=1.0, skip_level0=False,
                 self_correct=False):
        self.mode, self.low, self.high, self.skip0 = mode, low, high, skip_level0
        self.n_levels, self.self_correct = n_levels, bool(self_correct)
        self.rates = corruption_rates(n_levels, start, end) if mode == "linear" else None
        if self.rates is not None and skip_level0:
            self.rates[0] = 0.0

    @classmethod
    def from_config(cls, n_levels, T):
        c = T.get("corruption", None)
        if c is None:   # Phase A configs: corrupt_start / corrupt_end
            return cls(n_levels, "linear", float(T.corrupt_start), float(T.corrupt_end))
        return cls(n_levels, str(c.get("mode", "linear")), float(c.get("start", 0.5)), float(c.get("end", 0.1)),
                   float(c.get("low", 0.0)), float(c.get("high", 1.0)), bool(c.get("skip_level0", False)),
                   bool(c.get("self_correct", False)))

    def rate_matrix(self, B, device, generator=None):
        """[B, K] replacement probability per sequence and level (finest level always 0)."""
        if self.mode == "none":
            return torch.zeros(B, self.n_levels, device=device)
        if self.mode == "linear":
            return torch.tensor(self.rates, device=device, dtype=torch.float32)[None].expand(B, -1)
        r = self.low + (self.high - self.low) * torch.rand(B, 1, device=device, generator=generator)
        m = r.expand(B, self.n_levels).clone()
        m[:, -1] = 0.0
        if self.skip0:
            m[:, 0] = 0.0
        return m

    def describe(self):
        sc = " self_correct" if self.self_correct else ""
        if self.mode == "linear":
            return f"{[round(r, 3) for r in self.rates]}{sc}"
        return f"{self.mode} U({self.low},{self.high}) skip_level0={self.skip0}{sc}"

    @torch.no_grad()
    def __call__(self, idx, levels, vocab_sizes, generator=None):
        if self.mode == "none":
            return idx
        if self.mode == "linear":
            return corrupt_codes(idx, levels, vocab_sizes, self.rates, generator)
        B = idx.shape[0]
        r = self.low + (self.high - self.low) * torch.rand(B, 1, device=idx.device, generator=generator)
        out, off = idx.clone(), 0
        for k, (l, v) in enumerate(zip(levels, vocab_sizes)):
            if k < len(levels) - 1 and not (k == 0 and self.skip0):
                seg = out[:, off:off + l]
                m = torch.rand(seg.shape, device=idx.device, generator=generator) < r
                rnd = torch.randint(0, v, seg.shape, device=idx.device, generator=generator)
                out[:, off:off + l] = torch.where(m, rnd, seg)
            off += l
        return out


@torch.no_grad()
def corrupt_codes(idx, levels, vocab_sizes, rates, generator=None):
    """Replace codes uniformly at random within each level's codebook (targets stay clean)."""
    out = idx.clone()
    off = 0
    for l, v, r in zip(levels, vocab_sizes, rates):
        if r > 0:
            seg = out[:, off:off + l]
            m = torch.rand(seg.shape, device=idx.device, generator=generator) < r
            rnd = torch.randint(0, v, seg.shape, device=idx.device, generator=generator)
            out[:, off:off + l] = torch.where(m, rnd, seg)
        off += l
    return out


@torch.no_grad()
def self_correcting_codes(vqvae, x, rates, generator=None):
    """Targets that correct earlier errors (Infinity's bitwise self-correction, arXiv 2412.04431).

    Residual quantization is re-run level by level against the *corrupted* path: the target
    of level k is the nearest code to the residual z - z~(k-1), where z~ is the cumulative
    reconstruction of the conditioning codes actually fed to the generator (some replaced
    uniformly with probability rates[:, k]). With all rates 0 this is exactly vqvae.encode.
    The generator therefore learns to undo earlier mistakes instead of only tolerating them.
    Returns (targets, conditioning codes), both [B, sum(levels)].
    """
    vq = vqvae.vq
    z = vqvae._encode_pre_quant(x).float()
    B, L, C = z.shape
    K = len(vq.levels)
    z_hat = torch.zeros_like(z)
    tgt, cond = [], []
    for i, level in enumerate(vq.levels):
        z_rest = z - z_hat
        flat = (vq._down_for_quant(z_rest, i, level) if i != K - 1 else z_rest).reshape(-1, C)
        idx = torch.argmin(vq._distances(flat, vq.embeddings[i].weight), dim=1).view(B, level)
        c = idx
        if i < K - 1:
            r = rates[:, i:i + 1]
            if bool((r > 0).any()):
                m = torch.rand(idx.shape, device=x.device, generator=generator) < r
                rnd = torch.randint(0, vq.vqvae_vocab_sizes[i], idx.shape, device=x.device, generator=generator)
                c = torch.where(m, rnd, idx)
            q = vq._apply_quant_resi(vq._up_to_full(vq.embeddings[i](c), i, L), i)
            z_hat = z_hat + q
        tgt.append(idx)
        cond.append(c)
    return torch.cat(tgt, dim=1), torch.cat(cond, dim=1)


def predicted_levels(net):
    """Levels the generator produces logits for. Without predict_level0 / query_level0
    (upstream TinyStories), level 0 is sampled from a stored unigram and not trained."""
    lv = list(net.levels)
    return lv if (net.query_level0 or net.predict_level0) else lv[1:]


@torch.no_grad()
def embed_prefix(net, vqvae, src_ids):
    """Prefix embedding exactly as NextConceptTransformer.generate computes it (pre-wte)."""
    cfg, P = net.config, net.prefix_len
    if bool(getattr(cfg, "raw_token_prefix", False)):
        enc = vqvae.encoder
        if bool(getattr(cfg, "contextual_prefix", False)) and len(getattr(enc, "blocks", [])) > 0:
            return enc.forward_nopos(src_ids)[:, :P, :]
        return enc.wte(src_ids)[:, :P, :]
    return vqvae.encode_z(src_ids)[:, :P, :]


def build_batch(net, vqvae, src_ids, tgt_ids, pad_id, corrupt=None, generator=None):
    """Return (forward kwargs, target codes) for ``NextConceptTransformer.forward``.

    src_ids: [B, P] source tokens right-padded with pad_id (P = 1 null token if unconditional).
    tgt_ids: [B, L] target tokens, L = tokenizer context length.
    corrupt: a Corruptor, or a list of per-level rates (Phase A), or None for clean inputs.
    """
    vq = vqvae.vq
    with torch.no_grad():
        if isinstance(corrupt, Corruptor) and corrupt.self_correct:
            rates = corrupt.rate_matrix(tgt_ids.shape[0], tgt_ids.device, generator)
            codes, cond = self_correcting_codes(vqvae, tgt_ids, rates, generator)
        else:
            codes = vqvae.encode(tgt_ids)                           # [B, sum(levels)]
            cond = codes
        if isinstance(corrupt, Corruptor) and not corrupt.self_correct:
            cond = corrupt(codes, vq.levels, vq.vqvae_vocab_sizes, generator)
        elif corrupt is not None and not isinstance(corrupt, Corruptor) and any(r > 0 for r in corrupt):
            cond = corrupt_codes(codes, vq.levels, vq.vqvae_vocab_sizes, corrupt, generator)
        x_target = vq.idx_to_ncm_input(cond)                        # [B, sum(levels[1:]), d]
    if net.query_level0:                                            # learned queries carry gradients
        q = net.build_l0_query_input(tgt_ids.shape[0], x_target.dtype)
        x_target = torch.cat([q, x_target], dim=1)
    if not (net.query_level0 or net.predict_level0):
        codes = codes[:, vq.levels[0]:]
    pad_mask = src_ids[:, :net.prefix_len] == pad_id
    if net.use_ctx_encoder:
        return {"x": x_target, "prefix_ids": src_ids[:, :net.prefix_len], "prefix_pad_mask": pad_mask}, codes
    z_prefix = embed_prefix(net, vqvae, src_ids).to(x_target.dtype)
    return {"x": torch.cat([z_prefix, x_target], dim=1), "prefix_pad_mask": pad_mask}, codes


def ncm_loss(logits, codes, weights):
    """Weighted cross-entropy summed over positions, averaged over the batch.

    Also returns the unweighted CE per position for logging.
    """
    B, T, V = logits.shape
    assert T == codes.shape[1], f"logits cover {T} positions, targets {codes.shape[1]}"
    ce = F.cross_entropy(logits.float().reshape(-1, V), codes.reshape(-1), reduction="none").view(B, T)
    loss = (ce * weights[None, :]).sum(dim=1).mean()
    return loss, ce.detach()


def per_level_mean(ce, levels):
    out, off = [], 0
    for l in levels:
        out.append(float(ce[:, off:off + l].mean()))
        off += l
    return out
