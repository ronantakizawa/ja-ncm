"""Same-size autoregressive (AR) baseline: a decoder-only prefix LM.

Layout:  [PAD ... PAD  ja tokens][SEP][en tokens ... EOS]
          left-padded to P         ^ start of decoding

Size parity with NCM: the token table is the frozen VQ-VAE encoder table NCM reads its
prefix from (151,680 x 384, tied with the output), projected to the same 12 x 768 stack
the NCM generator uses. Blocks, RoPE and QK-norm come from the vendored ncm/models/common.py.
"""
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import jancm  # noqa: F401
from models.common import Block, LayerNorm, precompute_rope_cos_sin
from jancm.tokenizer_objective import chunked_ce


class ARTranslator(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_layer=12, n_head=12, n_embd=768, dropout=0.1,
                 bias=True, rope_base=100000, max_len=256, use_qk_norm=True, emb_init=None,
                 freeze_emb=False):
        super().__init__()
        bc = SimpleNamespace(n_embd=n_embd, n_head=n_head, dropout=dropout, bias=bias, use_qk_norm=use_qk_norm)
        self.wte = nn.Embedding(vocab_size, emb_dim)
        if emb_init is not None:
            with torch.no_grad():
                self.wte.weight.copy_(emb_init)
        self.wte.weight.requires_grad_(not freeze_emb)
        self.in_proj = nn.Linear(emb_dim, n_embd, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(bc) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, emb_dim, bias=False)   # logits = out_proj(h) @ wte^T
        # The inherited VQ-VAE table has row norms ~17, so a default-initialised projection
        # starts with logit std ~10 (initial loss ~52 vs ln V = 11.9). Zero init starts uniform.
        nn.init.zeros_(self.out_proj.weight)
        self.n_head = n_head
        cos, sin = precompute_rope_cos_sin(torch.arange(max_len), n_embd // n_head, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.max_len = max_len
        self._eos = -1                                            # set by set_eos(); -1 disables
        self.ce_chunk_rows = 4096
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def set_eos(self, eos_id):
        self._eos = int(eos_id)
        return self

    def logits(self, h):
        return self.out_proj(h) @ self.wte.weight.t()

    def _stack(self, x, mask):
        h = self.drop(self.in_proj(self.wte(x)))
        rope = (self.rope_cos, self.rope_sin)
        for blk in self.blocks:
            h = blk(h, attn_mask=mask, rope_cos_sin=rope)
        return self.ln_f(h)

    @staticmethod
    def train_mask(key_valid):
        """Causal + key padding. Each position may always see itself, so pad rows stay finite."""
        B, S = key_valid.shape
        causal = torch.ones(S, S, dtype=torch.bool, device=key_valid.device).tril()
        m = causal[None] & key_valid[:, None, :]
        m = m | torch.eye(S, dtype=torch.bool, device=key_valid.device)[None]
        return m[:, None]                                         # [B, 1, S, S]

    @staticmethod
    def after_first(tgt, eos_id):
        """True strictly after the first EOS (targets may be EOS-filled to the window)."""
        is_eos = (tgt == eos_id).long()
        return (is_eos.cumsum(1) - is_eos) > 0

    def forward(self, src_left, tgt, sep_id, src_pad_id, tgt_pad_id):
        """Teacher-forced loss on target tokens. src_left: [B, P] left-padded; tgt: [B, L]."""
        B, P = src_left.shape
        sep = torch.full((B, 1), sep_id, dtype=src_left.dtype, device=src_left.device)
        x = torch.cat([src_left, sep, tgt[:, :-1]], dim=1)        # [B, P + L]
        valid = torch.cat([src_left != src_pad_id, torch.ones_like(sep, dtype=torch.bool),
                           torch.ones_like(tgt[:, :-1], dtype=torch.bool)], dim=1)
        h = self._stack(x, self.train_mask(valid))
        labels = tgt.masked_fill(self.after_first(tgt, self._eos) | (tgt == tgt_pad_id), -100)
        # logits = out_proj(h) @ wte^T over a 151,680-token vocabulary: computed in chunks with
        # recomputation instead of materialising [B, L, V] fp32 logits (~20 GB at B=256, L=128).
        return chunked_ce(self.out_proj(h[:, P:]), self.wte.weight, labels, self.ce_chunk_rows)

    def _set_cache(self, on):
        for blk in self.blocks:
            blk.attn.kv_caching(on)

    @staticmethod
    def _pick(logits, temperature, top_p, generator=None):
        if temperature <= 0:
            return logits.argmax(-1)
        logits = logits.float() / temperature
        if top_p < 1.0:
            sl, si = torch.sort(logits, descending=True, dim=-1)
            p = torch.softmax(sl, -1)
            sl = sl.masked_fill(p.cumsum(-1) - p > top_p, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)
        return torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).squeeze(-1)

    @torch.no_grad()
    def generate(self, src_left, sep_id, src_pad_id, eos_id, max_new=64, use_kv_cache=True,
                 temperature=0.0, top_p=1.0, generator=None, early_stop=True):
        """Greedy (temperature 0) or nucleus sampling. Returns (tokens [B, <=max_new], nfe)."""
        B, P = src_left.shape
        sep = torch.full((B, 1), sep_id, dtype=src_left.dtype, device=src_left.device)
        x = torch.cat([src_left, sep], dim=1)
        valid = torch.cat([src_left != src_pad_id, torch.ones_like(sep, dtype=torch.bool)], dim=1)
        done = torch.zeros(B, dtype=torch.bool, device=x.device)
        out, nfe = [], 0
        assert P + 1 + max_new <= self.max_len, "increase max_len"
        if use_kv_cache:
            self._set_cache(True)
        try:
            h = self._stack(x, self.train_mask(valid))
            nfe += 1
            for t in range(max_new):
                nxt = self._pick(self.logits(h[:, -1]), temperature, top_p, generator)
                nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
                out.append(nxt)
                done |= nxt == eos_id
                if t == max_new - 1 or (early_stop and bool(done.all())):   # early_stop=False: no host sync
                    break
                valid = torch.cat([valid, torch.ones(B, 1, dtype=torch.bool, device=x.device)], dim=1)
                if use_kv_cache:
                    h = self._stack(nxt[:, None], valid[:, None, None, :])
                else:
                    x = torch.cat([x, nxt[:, None]], dim=1)
                    h = self._stack(x, self.train_mask(valid))
                nfe += 1
        finally:
            if use_kv_cache:
                self._set_cache(False)
        return torch.stack(out, dim=1), nfe


def left_pad(ids_list, P, pad_id):
    """Truncate to the first P source tokens (as NCM does) and pad on the left."""
    rows = []
    for ids in ids_list:
        ids = list(ids)[:P]
        rows.append([pad_id] * (P - len(ids)) + ids)
    return rows


def right_to_left(src_right, pad_id):
    """Convert right-padded prefixes (as written by prepare_data.py) to left padding."""
    n_pad = (src_right == pad_id).sum(dim=1)
    P = src_right.shape[1]
    ar = torch.arange(P, device=src_right.device)[None, :]
    idx = (ar - n_pad[:, None]) % P
    return src_right.gather(1, idx)
