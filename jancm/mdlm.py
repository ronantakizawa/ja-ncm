"""Same-size masked diffusion baseline (MDLM; Sahoo et al., 2024) for the Japanese comparison.

The paper's selling points are measured against diffusion LMs (MDLM, DUO), so the Japanese NCM
needs one trained on the same windows and budget. Choices:

* Network: the AR baseline's stack (jancm/ar.py): token table from the frozen VQ-VAE encoder
  (V x 384, tied with the output) projected to 12 x 768 blocks with RoPE and QK-norm, but with
  bidirectional attention and one extra row for the [MASK] token. No time conditioning: with
  absorbing-state diffusion the noise level is implied by the mask count, and MDLM reports
  no loss from dropping it.
* Objective: SUBS parameterisation with the log-linear schedule alpha_t = 1 - t, so the
  NELBO weight of a masked position is 1/t; t is drawn low-discrepancy across the batch.
  Logits never put mass on [MASK]; unmasked positions are carried over.
* Sampling: ancestral, T steps from t=1 to 0. A masked position is revealed with probability
  (t - s) / t, its token drawn from p(x0 | x_t); NFE = T. Logits are only computed for the
  positions still masked, in chunks, with Gumbel-max sampling (no [B, L, V] tensor).

The whole window is modelled (text, EOS, PAD fill), as NCM's tokenizer does.
"""
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import jancm  # noqa: F401
from models.common import Block, LayerNorm, precompute_rope_cos_sin


def _weighted_ce_chunk(h, w_out, w_tab, targets, weights):
    logits = (h @ w_out.t()) @ w_tab.t()
    return (F.cross_entropy(logits.float(), targets, reduction="none") * weights).sum()


class MDLM(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_layer=12, n_head=12, n_embd=768, dropout=0.1, bias=True,
                 rope_base=100000, max_len=128, use_qk_norm=True, emb_init=None, eps=1e-3):
        super().__init__()
        bc = SimpleNamespace(n_embd=n_embd, n_head=n_head, dropout=dropout, bias=bias, use_qk_norm=use_qk_norm)
        self.V = int(vocab_size)
        self.mask_id = self.V                                     # extra row: [MASK]
        self.wte = nn.Embedding(self.V + 1, emb_dim)
        if emb_init is not None:
            with torch.no_grad():
                self.wte.weight[:self.V].copy_(emb_init)
                self.wte.weight[self.V].normal_(0.0, float(emb_init.std()))
        self.in_proj = nn.Linear(emb_dim, n_embd, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(bc) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, emb_dim, bias=False)    # logits = out_proj(h) @ wte[:V]^T
        nn.init.zeros_(self.out_proj.weight)                      # uniform start (see ar.py)
        cos, sin = precompute_rope_cos_sin(torch.arange(max_len), n_embd // n_head, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.max_len, self.eps, self.ce_chunk_rows = int(max_len), float(eps), 4096
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def hidden(self, x, attn_mask=None):
        B, L = x.shape
        h = self.drop(self.in_proj(self.wte(x)))
        if attn_mask is None:
            attn_mask = torch.ones(1, 1, L, L, dtype=torch.bool, device=x.device)   # bidirectional
        # Full tables: with a (frozen) KV cache the attention offsets RoPE by the cached length.
        rope = (self.rope_cos, self.rope_sin)
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask, rope_cos_sin=rope)
        return self.ln_f(h)

    # ---------------------------------------------------------------- conditional (prefix) MDLM
    # Layout [src_left (P, left-padded) | target (L)]. The source is never masked and attends
    # only to valid source positions, so its K/V do not depend on the target and are computed
    # once at generation (frozen KV cache). Target positions attend to the valid source and to
    # the whole target block. Only target positions are noised and scored; the target window
    # is modelled whole (text, EOS, PAD fill), as the unconditional MDLM models its window.
    @staticmethod
    def cond_mask(src_valid, L):
        """[B, 1, P+L, P+L] bool. src_valid: [B, P]."""
        B, P = src_valid.shape
        S = P + L
        keys = torch.cat([src_valid, torch.ones(B, L, dtype=torch.bool, device=src_valid.device)], 1)
        m = keys[:, None, :].expand(B, S, S).clone()
        m[:, :P, P:] = False                                     # source does not see the target
        m |= torch.eye(S, dtype=torch.bool, device=src_valid.device)[None]   # pad rows stay finite
        return m[:, None]

    def forward_cond(self, src_left, tgt, src_pad_id, t=None, generator=None):
        """NELBO per target token, averaged over the batch."""
        B, L = tgt.shape
        P = src_left.shape[1]
        if t is None:
            t = self.sample_t(B, tgt.device, generator)
        masked = torch.rand(tgt.shape, device=tgt.device, generator=generator) < t[:, None]
        xt = torch.where(masked, torch.full_like(tgt, self.mask_id), tgt)
        h = self.hidden(torch.cat([src_left, xt], 1), self.cond_mask(src_left != src_pad_id, L))[:, P:]
        rows = masked.nonzero(as_tuple=True)
        hm, target = h[rows], tgt[rows]
        w = (1.0 / t)[rows[0]] / (L * B)
        total = h.new_zeros((), dtype=torch.float32)
        tab = self.wte.weight[:self.V]
        for i in range(0, hm.shape[0], self.ce_chunk_rows):
            sl = slice(i, i + self.ce_chunk_rows)
            args = (hm[sl], self.out_proj.weight, tab, target[sl], w[sl])
            if torch.is_grad_enabled():
                total = total + checkpoint(_weighted_ce_chunk, *args, use_reentrant=False)
            else:
                total = total + _weighted_ce_chunk(*args)
        return total

    def _cache(self, on, frozen=False):
        for blk in self.blocks:
            if on is not None:
                blk.attn.kv_caching(on)
            blk.attn.kv_freeze(frozen)

    @torch.no_grad()
    def generate_cond(self, src_left, src_pad_id, L, steps, temperature=0.0, order="confidence",
                      use_cache=True):
        """Denoise a length-L target given the source, `steps` network calls (NFE = steps).
        order="random": MDLM ancestral sampling (a masked position is revealed with probability
        (t - s) / t). order="confidence": reveal the same number of positions per step
        (L / steps), choosing the masked positions the model is most confident about (MaskGIT /
        LLaDA-style). No host syncs: CUDA-graph capturable. Returns ids [B, L]."""
        B, P = src_left.shape
        dev = src_left.device
        src_valid = src_left != src_pad_id
        x = torch.full((B, L), self.mask_id, dtype=torch.long, device=dev)
        tab = self.wte.weight[:self.V]
        keys = torch.cat([src_valid, torch.ones(B, L, dtype=torch.bool, device=dev)], 1)[:, None, None, :]
        full_mask = self.cond_mask(src_valid, L)
        if use_cache:
            self._cache(True)
            pm = src_valid[:, None, None, :] | torch.eye(P, dtype=torch.bool, device=dev)[None, None]
            self.hidden(src_left, pm)                            # fills each layer's cache with the source K/V
            self._cache(None, frozen=True)
        counts = [L * (i + 1) // steps - L * i // steps for i in range(steps)]
        try:
            for i in range(steps):
                t, s = 1.0 - i / steps, 1.0 - (i + 1) / steps
                still = x == self.mask_id
                if use_cache:
                    h = self.hidden(x, keys.expand(B, 1, L, P + L))
                else:
                    h = self.hidden(torch.cat([src_left, x], 1), full_mask)[:, P:]
                lg = (self.out_proj(h) @ tab.t()).float()
                if temperature > 0:
                    u = torch.rand(lg.shape, device=dev).clamp_(1e-20, 1.0)
                    lg = lg / temperature - torch.log(-torch.log(u))
                conf, tok = lg.max(-1)
                if order == "random":
                    reveal = still & (torch.rand(x.shape, device=dev) < ((t - s) / t if i < steps - 1 else 2.0))
                else:
                    k = counts[i]
                    if k == 0:
                        continue
                    score = torch.where(still, conf, torch.full_like(conf, float("-inf")))
                    idx = score.topk(k, dim=-1).indices
                    reveal = torch.zeros_like(still).scatter_(1, idx, True) & still
                x = torch.where(reveal, tok, x)
        finally:
            if use_cache:
                self._cache(False)
        return x

    def sample_t(self, B, device, generator=None):
        """Low-discrepancy t in [eps, 1] (one stratum per sequence, as MDLM)."""
        u = torch.rand((), device=device, generator=generator)
        t = (u + torch.arange(B, device=device) / B) % 1.0
        return self.eps + (1.0 - self.eps) * t

    def forward(self, x0, t=None, generator=None, src_left=None, src_pad_id=None):
        """NELBO per token, averaged over the batch. x0: [B, L] token ids (no [MASK]).
        With src_left ([B, P], left-padded) x0 is the target and the source is the condition."""
        if src_left is not None:
            return self.forward_cond(src_left, x0, src_pad_id, t=t, generator=generator)
        B, L = x0.shape
        if t is None:
            t = self.sample_t(B, x0.device, generator)
        masked = torch.rand(x0.shape, device=x0.device, generator=generator) < t[:, None]
        xt = torch.where(masked, torch.full_like(x0, self.mask_id), x0)
        h = self.hidden(xt)
        rows = masked.nonzero(as_tuple=True)
        hm, tgt = h[rows], x0[rows]
        w = (1.0 / t)[rows[0]] / (L * B)                          # alpha'/(1-alpha) = 1/t, per token, per batch
        total = h.new_zeros((), dtype=torch.float32)
        tab = self.wte.weight[:self.V]
        for i in range(0, hm.shape[0], self.ce_chunk_rows):
            sl = slice(i, i + self.ce_chunk_rows)
            args = (hm[sl], self.out_proj.weight, tab, tgt[sl], w[sl])
            if torch.is_grad_enabled():
                total = total + checkpoint(_weighted_ce_chunk, *args, use_reentrant=False)
            else:
                total = total + _weighted_ce_chunk(*args)
        return total

    @torch.no_grad()
    def _draw(self, h_rows, temperature, generator=None):
        """Sample token ids for hidden rows [N, d] chunk-wise (Gumbel-max at T>0, argmax at 0)."""
        tab = self.wte.weight[:self.V]
        out = []
        for i in range(0, h_rows.shape[0], self.ce_chunk_rows):
            lg = (self.out_proj(h_rows[i:i + self.ce_chunk_rows]) @ tab.t()).float()
            if temperature > 0:
                u = torch.rand(lg.shape, device=lg.device, generator=generator).clamp_(1e-20, 1.0)
                lg = lg / temperature - torch.log(-torch.log(u))
            out.append(lg.argmax(-1))
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.long, device=h_rows.device)

    @torch.no_grad()
    def generate(self, B, L, steps, device, temperature=1.0, generator=None):
        """Ancestral sampling with `steps` network calls. Returns ids [B, L], nfe."""
        x = torch.full((B, L), self.mask_id, dtype=torch.long, device=device)
        nfe = 0
        for i in range(steps):
            t, s = 1.0 - i / steps, 1.0 - (i + 1) / steps
            still = x == self.mask_id
            if not bool(still.any()):
                break
            reveal = still & (torch.rand(x.shape, device=device, generator=generator) < (t - s) / t)
            if i == steps - 1:
                reveal = still
            if not bool(reveal.any()):
                continue                                          # no position changes: skip the call
            h = self.hidden(x)
            nfe += 1
            rows = reveal.nonzero(as_tuple=True)
            x[rows] = self._draw(h[rows], temperature, generator)
        return x, nfe


@torch.no_grad()
def generate_static(net, B, L, steps, device, temperature=1.0):
    """Same sampler as MDLM.generate, without host syncs (no early exit, no skipped no-op
    steps, logits for every position): what CUDA graphs need, and the standard MDLM cost."""
    x = torch.full((B, L), net.mask_id, dtype=torch.long, device=device)
    tab = net.wte.weight[:net.V]
    for i in range(steps):
        t, s = 1.0 - i / steps, 1.0 - (i + 1) / steps
        still = x == net.mask_id
        reveal = still & (torch.rand(x.shape, device=device) < ((t - s) / t if i < steps - 1 else 2.0))
        lg = (net.out_proj(net.hidden(x)) @ tab.t()).float()
        u = torch.rand(lg.shape, device=device).clamp_(1e-20, 1.0)
        x = torch.where(reveal, (lg / temperature - torch.log(-torch.log(u))).argmax(-1), x)
    return x


def build_mdlm(cfg, device, emb_init=None):
    M = cfg.mdlm
    V, D = int(cfg.tokenizer.vqvae.in_vocab_size), int(cfg.tokenizer.vqvae.embed_dim)
    if emb_init is None and bool(M.get("init_emb_from_vqvae", True)):
        from jancm.ar_setup import vqvae_embedding
        emb_init = vqvae_embedding(cfg)
    L = int(cfg.tokenizer.vqvae.context_length)
    if str(cfg.get("data", {}).get("kind", "windows")) == "pairs":   # conditional: [source | target]
        L += int(cfg.model.prefix_len)
    net = MDLM(V, D, n_layer=int(M.n_layer), n_head=int(M.n_head), n_embd=int(M.n_embd), dropout=float(M.dropout),
               rope_base=int(M.rope_base), max_len=L, emb_init=emb_init)
    return net.to(device)


@torch.no_grad()
def translate_mdlm(net, cfg, pretok, texts, device, batch_size=64, return_nfe=False):
    """Conditional generation (translation / summarisation) with the prefix MDLM.
    generation.mdlm_cond_steps (NFE), generation.mdlm_order (confidence | random),
    generation.mdlm_temperature (0 = argmax)."""
    from jancm.ar import left_pad
    from jancm.text import decode_target, encode_batch
    P, L = int(cfg.model.prefix_len), int(cfg.tokenizer.vqvae.context_length)
    G = cfg.generation
    steps, order = int(G.get("mdlm_cond_steps", 32)), str(G.get("mdlm_order", "random"))
    temp = float(G.get("mdlm_temperature", 0.0))
    eos, pad = pretok.eos_token_id, pretok.pad_token_id
    ids = encode_batch(pretok, texts)
    hyps = []
    net.eval()
    for s in range(0, len(ids), batch_size):
        x = torch.tensor(left_pad(ids[s:s + batch_size], P, pad), device=device)
        out = net.generate_cond(x, pad, L, steps, temperature=temp, order=order)
        hyps += [decode_target(pretok, r, eos, pad) for r in out.tolist()]
    return (hyps, [steps]) if return_nfe else hyps


def load_mdlm_checkpoint(path, device, use_ema=True, weights="auto"):
    """weights: "auto" (EMA once step*(1-decay) >= 5, else raw; see below), "raw" or "ema"."""
    from omegaconf import OmegaConf
    from jancm.loading import load_state
    from jancm.text import pretokenizer_from_config
    st = load_state(path)
    cfg = OmegaConf.create(st["config"])
    net = build_mdlm(cfg, device, emb_init=torch.zeros(int(cfg.tokenizer.vqvae.in_vocab_size),
                                                       int(cfg.tokenizer.vqvae.embed_dim)))
    # An EMA with decay d averages over ~1/(1-d) steps and is not bias-corrected, so before
    # training has covered that window several times it is dominated by early weights
    # (10k-step pilot, d=0.9999: dev NELBO 6.43 with EMA vs train loss 4.98). Use it only
    # once step * (1 - d) >= 5; the 100k-step run (factor 10) keeps using it.
    decay = float(cfg.train.get("ema", 0.9999))
    ema_ok = "ema" in st and int(st.get("step", 0)) * (1.0 - decay) >= 5
    if weights == "raw":
        if "model" not in st:
            raise ValueError(f"{path} has no raw weights (EMA-only checkpoint)")
        net.load_state_dict(st["model"])
        net.weights = "raw"
    elif weights == "ema":
        net.load_state_dict(st["ema"])
        net.weights = "ema"
    elif use_ema and ema_ok:
        net.load_state_dict(st["ema"])
        net.weights = "ema"
    else:
        net.load_state_dict(st["model"] if "model" in st else st["ema"])
        net.weights = "raw" if "model" in st else "ema"
    return net, cfg, pretokenizer_from_config(cfg)


@torch.no_grad()
def sample_mdlm_unconditional(net, cfg, pretok, n, device, batch_size=64, steps=128, temperature=1.0):
    from jancm.text import decode_target
    L = int(cfg.tokenizer.vqvae.context_length)
    eos, pad = pretok.eos_token_id, pretok.pad_token_id
    cut = bool(cfg.generation.get("cut_at_eos", True))
    hyps, nfes = [], []
    net.eval()
    for s in range(0, n, batch_size):
        b = min(batch_size, n - s)
        ids, nfe = net.generate(b, L, steps, device, temperature)
        hyps += [decode_target(pretok, r, eos, pad, cut) for r in ids.tolist()]
        nfes.append(nfe)
    return hyps, nfes
