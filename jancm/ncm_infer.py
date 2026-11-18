"""Batched NCM generation: conditional (text prefix) and unconditional (null prefix)."""
import torch

from jancm.decoding import pick_tokens, readout_from_config, sample_tokens
from jancm.text import decode_target, encode_batch, source_prefix


def _gen_kwargs(G, eos):
    return dict(temperature=float(G.get("temperature", 0.0)), top_k=int(G.get("top_k", 1)),
                top_p=float(G.get("top_p", 0.0)), use_kv_cache=bool(G.get("use_kv_cache", False)),
                cfg_scale=float(G.get("cfg_scale", 0.0)), cfg_null_eos=eos)


@torch.no_grad()
def generate_ids(net, tk, prefix_rows, cfg, device, batch_size=64, return_levels=False, level0=None):
    """prefix_rows: list of length-P id lists. Returns decoded texts (and per-level texts).

    level0: optional [N, l0] codes for models that sample level 0 outside the generator.
    NFE = number of levels K (one forward pass per level).
    """
    pretok = tk.pre_tokenizer
    eos, pad = pretok.eos_token_id, pretok.pad_token_id
    kw = _gen_kwargs(cfg.generation, eos)
    cut = bool(cfg.generation.get("cut_at_eos", True))
    hyps, levels_out = [], []
    net.eval()
    level_t = cfg.generation.get("level_temperature", None)   # optional per-level temperatures (coarse -> fine)
    if level_t is not None:
        level_t = [float(t) for t in level_t]
        assert len(level_t) == len(net.levels), f"level_temperature needs {len(net.levels)} values"
        prev = net.__dict__.get("_sample_level_raw")                 # an instance override, if any
        raw, state = net._sample_level_raw, {"k": 0}

        def per_level(logits, temperature, top_k, top_p):
            k = state["k"]
            state["k"] = (k + 1) % len(level_t)
            return raw(logits, level_t[k], top_k, top_p)
        net._sample_level_raw = per_level
    readout = None   # (adjacency, utf8) built on the first batch, once the logit width is known
    for s in range(0, len(prefix_rows), batch_size):
        x = torch.tensor(prefix_rows[s:s + batch_size], device=device)
        l0 = None if level0 is None else level0[s:s + batch_size].to(device)
        codes = net.generate(l0, x, **kw)
        (zh,) = tk.vqvae.vq.idx_to_zhats(codes, return_zlevels=False)
        logits = tk.vqvae.decoder(zh[-1])
        if readout is None:
            readout = readout_from_config(cfg.generation, pretok, logits.shape[-1], logits.device)
        dec_t = float(cfg.generation.get("decoder_temperature", 0.0))   # 0 = argmax read-out (upstream)
        if dec_t > 0:
            out = sample_tokens(logits, dec_t, float(cfg.generation.get("decoder_top_p", 1.0))).tolist()
        else:
            out = pick_tokens(logits, *readout).tolist()
        hyps += [decode_target(pretok, r, eos, pad, cut) for r in out]
        if return_levels:
            # Full-length cumulative reconstruction after each level (paper Fig. 3); needs the
            # issue #11 patch, otherwise every entry aliases the final level.
            per_level = [tk.vqvae.decoder(z).argmax(-1).tolist() for z in zh]
            for b in range(len(out)):
                levels_out.append([decode_target(pretok, lv[b], eos, pad, cut) for lv in per_level])
    if level_t is not None:                                          # restore what was there before
        if prev is None:
            del net._sample_level_raw
        else:
            net._sample_level_raw = prev
    if return_levels:
        return hyps, levels_out
    return hyps


@torch.no_grad()
def translate_ncm(net, tk, texts, cfg, device, batch_size=64, return_levels=False):
    """Conditional generation from source texts (translation, summarisation)."""
    P = int(cfg.model.prefix_len)
    pad = tk.pre_tokenizer.pad_token_id
    rows = [source_prefix(i, P, pad) for i in encode_batch(tk.pre_tokenizer, texts)]
    return generate_ids(net, tk, rows, cfg, device, batch_size, return_levels)


@torch.no_grad()
def sample_unconditional(net, tk, n, cfg, device, batch_size=64, level0_unigram=None, return_ids=False):
    """Unconditional samples from the one-token null prefix (EOS), as upstream eval_shard."""
    pretok = tk.pre_tokenizer
    eos, pad = pretok.eos_token_id, pretok.pad_token_id
    P = int(cfg.model.prefix_len)
    rows = [[eos] + [pad] * (P - 1)] * n
    l0 = None
    if level0_unigram is not None and not (net.query_level0 or net.predict_level0):
        l0 = torch.multinomial(level0_unigram.float().cpu(), n * net.levels[0], replacement=True).view(n, -1)
    hyps = generate_ids(net, tk, rows, cfg, device, batch_size, level0=l0)
    if return_ids:
        return hyps, [encode_batch(pretok, [h])[0] for h in hyps]
    return hyps


def n_truncated(tk, texts, P):
    return sum(len(i) > P for i in encode_batch(tk.pre_tokenizer, texts))
