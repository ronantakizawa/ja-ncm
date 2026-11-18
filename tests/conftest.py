import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jancm  # noqa: E402,F401
from models.next_concept import NextConceptTransformer  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402

LEVELS = [1, 4, 8, 16]
V_CODE = 32
V_TOK = 100
PAD = 0
EOS = 1


def tiny_vq_config():
    return OmegaConf.create({
        "clss": "multiscale_residual_vector_quantizer",
        "decay": 0.99, "epsilon": 1e-5, "commitment_cost": 0.25,
        "learned_l1_sampling": True, "learned_l1_cascade": True, "learned_l1_cascade_base": 4,
        "learned_l1_cascade_norm": True, "learned_l1_cascade_norm_final": True,
        "learned_all_sampling": False,
        "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0, "learnable_ratio": False},
        "aux": {"fine_drop": {"prob": 0.0, "min_keep": 1}},
        "levels": {"use_manual_levels": True, "manual_levels": LEVELS},
    })


def tiny_model_config(prefix_len=10):
    return OmegaConf.create({
        "n_layer": 2, "n_head": 4, "n_embd": 64, "dropout": 0.0, "bias": True,
        "prefix_len": prefix_len, "raw_token_prefix": True, "prefix_pad_token_id": PAD,
        "vocab_size": [V_CODE] * (len(LEVELS) - 1), "predict_level0": True,
        "attn_pattern": "block_diagonal", "use_level_encoding": True, "use_rope": True,
        "rope_base": 10000, "use_qk_norm": True, "ctx_encoder": {"enabled": False},
        "use_flex_attention": False, "untied_head_bias": False,
    })


def make_vqvae(seed=0):
    torch.manual_seed(seed)
    vq = VQVAE(in_vocab_size=V_TOK, context_length=LEVELS[-1], embed_dim=32,
               vqvae_vocab_size=[V_CODE] * len(LEVELS), n_layers=4, vector_quantizer_config=tiny_vq_config(),
               compression_factor=1, dropout=0.0, pre_quant_groupnorm=8, use_qk_norm=True)
    return vq.eval()


def make_net(vqvae, prefix_len=10, seed=1):
    torch.manual_seed(seed)
    return NextConceptTransformer(vqvae_local=vqvae, config=tiny_model_config(prefix_len)).eval()


def padded_prefixes(B, P, seed=2):
    """Right-padded source ids with varying lengths (>= 1 real token each)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.full((B, P), PAD, dtype=torch.long)
    for b in range(B):
        n = int(torch.randint(1, P + 1, (1,), generator=g))
        x[b, :n] = torch.randint(2, V_TOK, (n,), generator=g)
    return x


@pytest.fixture
def vqvae():
    return make_vqvae()


@pytest.fixture
def net(vqvae):
    return make_net(vqvae)
