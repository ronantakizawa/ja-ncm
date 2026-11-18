"""Build the frozen tokenizer and the NCM generator from a config, and load weights safely."""
import os

import torch
from omegaconf import OmegaConf

import jancm  # noqa: F401  (sys.path setup)
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer

# Buffers that depend on prefix_len; they are rebuilt from the config, so a checkpoint
# trained with a different prefix length (96 for WMT14) can still initialise the model.
_REBUILT_BUFFERS = ("attn_mask_for_masking", "wp_prefix", "wp_1L")


def load_state(path):
    """Load a checkpoint without executing pickled code."""
    return torch.load(path, map_location="cpu", weights_only=True)


def load_config(path, overrides=None):
    """Load a YAML config. A top-level ``base:`` (path relative to the project root) is
    loaded first and this file is merged over it, recursively."""
    cfg = OmegaConf.load(path)
    if "base" in cfg and cfg.base:
        from jancm import ROOT
        bases = list(cfg.base) if OmegaConf.is_list(cfg.base) else [cfg.base]   # later bases override earlier
        merged = OmegaConf.merge(*[load_config(resolve(str(b), ROOT)) for b in bases]) if len(bases) > 1 \
            else load_config(resolve(str(bases[0]), ROOT))
        cfg = OmegaConf.merge(merged, cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def build_tokenizer(cfg, tokenizer_ckpt, device):
    """Frozen multiscale VQ-VAE plus its Qwen3 pre-tokenizer."""
    tcfg = cfg.tokenizer.copy()
    tcfg.vqvae.checkpoint_path = tokenizer_ckpt
    tk = VQVAETokenizer.from_config(tcfg)
    tk.vqvae = tk.vqvae.to(device).eval()
    for p in tk.vqvae.parameters():
        p.requires_grad_(False)
    return tk


def build_generator(cfg, tk, device, init_ckpt=None, strict=True):
    """NCM generator. ``init_ckpt`` may come from a run with a different prefix_len."""
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model)
    if init_ckpt:
        sd = load_state(init_ckpt)["model"]
        own = net.state_dict()
        dropped = [k for k in sd if k in _REBUILT_BUFFERS and k in own and sd[k].shape != own[k].shape]
        for k in dropped:
            sd.pop(k)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        missing = [k for k in missing if k not in dropped]
        if strict and (missing or unexpected):
            raise RuntimeError(f"init checkpoint mismatch: missing={missing} unexpected={unexpected}")
        if dropped:
            print(f"[load] rebuilt prefix-length buffers from config: {dropped}")
    return net.to(device)


def default_device():
    # MPS is not used: F.interpolate(mode="area") with non-divisible sizes (64 -> 14 codes)
    # is not implemented there, so Macs fall back to CPU for smoke tests.
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve(path, base):
    return path if os.path.isabs(path) else os.path.join(base, path)
