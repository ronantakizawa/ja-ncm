"""Config loading and model construction for the AR baseline."""
import torch
from omegaconf import OmegaConf

from jancm import ROOT
from jancm.ar import ARTranslator, left_pad
from jancm.loading import load_config, load_state, resolve
from jancm.text import decode_target, encode_batch, pretokenizer_from_config


def load_ar_config(path, overrides=None):
    return load_config(path, overrides)


def vqvae_embedding(cfg):
    """The frozen tokenizer's encoder token table, without building the whole VQ-VAE."""
    sd = load_state(resolve(cfg.checkpoints.tokenizer, ROOT))["model"]
    return sd["encoder.wte.weight"].float()


def build_ar(cfg, device, emb_init=None):
    A = cfg.ar
    P = int(cfg.model.prefix_len)
    # Room for the training window and for generation, whichever is longer.
    L = max(int(cfg.tokenizer.vqvae.context_length), int(cfg.generation.get("max_new_tokens", 0)))
    V, D = int(cfg.tokenizer.vqvae.in_vocab_size), int(cfg.tokenizer.vqvae.embed_dim)
    if emb_init is None and bool(A.get("init_emb_from_vqvae", True)):
        emb_init = vqvae_embedding(cfg)
    net = ARTranslator(V, D, n_layer=int(A.n_layer), n_head=int(A.n_head), n_embd=int(A.n_embd),
                       dropout=float(A.dropout), rope_base=int(A.rope_base), max_len=P + 1 + L,
                       emb_init=emb_init, freeze_emb=bool(A.get("freeze_emb", False)))
    pretok = pretokenizer_from_config(cfg)
    net.set_eos(pretok.eos_token_id)
    return net.to(device)


def special_ids(cfg, pretok):
    sep = pretok.hf_tokenizer.convert_tokens_to_ids(str(cfg.ar.sep_token))
    assert sep is not None and sep != pretok.hf_tokenizer.unk_token_id, f"unknown sep token {cfg.ar.sep_token}"
    return sep, pretok.eos_token_id, pretok.pad_token_id


@torch.no_grad()
def translate_ar(net, cfg, pretok, texts, device, batch_size=64, return_nfe=False):
    P = int(cfg.model.prefix_len)
    sep, eos, pad = special_ids(cfg, pretok)
    max_new = int(cfg.generation.get("max_new_tokens", 64))
    use_cache = bool(cfg.generation.get("use_kv_cache", True))
    temp, top_p = float(cfg.generation.get("ar_temperature", 0.0)), float(cfg.generation.get("ar_top_p", 1.0))
    ids = encode_batch(pretok, texts)
    hyps, nfes = [], []
    net.eval()
    for s in range(0, len(ids), batch_size):
        x = torch.tensor(left_pad(ids[s:s + batch_size], P, pad), device=device)
        out, nfe = net.generate(x, sep, pad, eos, max_new=max_new, use_kv_cache=use_cache,
                                temperature=temp, top_p=top_p)
        hyps += [decode_target(pretok, r, eos, pad) for r in out.tolist()]
        nfes.append(nfe)
    return (hyps, nfes) if return_nfe else hyps


@torch.no_grad()
def sample_ar_unconditional(net, cfg, pretok, n, device, batch_size=64, temperature=1.0, top_p=1.0):
    """Unconditional AR samples: null prefix [EOS] (as NCM), then SEP, then sampled tokens."""
    sep, eos, pad = special_ids(cfg, pretok)
    P = int(cfg.model.prefix_len)
    max_new = int(cfg.generation.get("max_new_tokens", 128))
    hyps, nfes = [], []
    net.eval()
    for s in range(0, n, batch_size):
        b = min(batch_size, n - s)
        x = torch.tensor(left_pad([[eos]] * b, P, pad), device=device)
        out, nfe = net.generate(x, sep, pad, eos, max_new=max_new, temperature=temperature, top_p=top_p)
        hyps += [decode_target(pretok, r, eos, pad) for r in out.tolist()]
        nfes.append(nfe)
    return hyps, nfes


def load_ar_checkpoint(path, device):
    st = load_state(path)
    cfg = OmegaConf.create(st["config"])
    net = build_ar(cfg, device, emb_init=torch.zeros(int(cfg.tokenizer.vqvae.in_vocab_size),
                                                     int(cfg.tokenizer.vqvae.embed_dim)))
    net.load_state_dict(st["model"])
    return net, cfg, pretokenizer_from_config(cfg)

