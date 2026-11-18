"""Shared text <-> token-id conventions, used by data prep, training and evaluation.

Keeping one implementation guarantees the prefix layout seen in training is the one
used at test time (upstream eval_shard.py: ``(ids[:P] + [PAD] * P)[:P]``).
"""
import jancm  # noqa: F401
from tokenization.hf_tokenizer import HFTokenizer


def pretokenizer_from_config(cfg):
    return HFTokenizer.from_config(cfg.tokenizer.pre_tokenizer)


def encode_batch(pretok, texts, batch_size=4096):
    """Plain token ids, no BOS/EOS. Qwen3 adds none by default (upstream ``src_tok(item)``);
    llm-jp-3 would prepend <s>, so special tokens are switched off explicitly."""
    out = []
    for i in range(0, len(texts), batch_size):
        out.extend(pretok.hf_tokenizer(list(texts[i:i + batch_size]), add_special_tokens=False)["input_ids"])
    return out


def source_prefix(ids, P, pad_id):
    """Right-pad / truncate source ids to the prefix length."""
    return (list(ids)[:P] + [pad_id] * P)[:P]


def target_window(ids, L, eos_id, pad_id, add_eos=True, fill="pad"):
    """Target ids + EOS, filled to the tokenizer window L. Returns None if too long.

    The released WMT14 tokenizer reconstructs PAD-filled windows better than EOS-filled
    ones on KFTT/ALT dev (97.0% vs 94.3% content-token accuracy, 50% vs 18% exact), and
    maps PAD positions to EOS on decode. Default is PAD; fill="eos" repeats EOS instead.
    """
    ids = list(ids) + ([eos_id] if add_eos else [])
    if len(ids) > L:
        return None
    return ids + [eos_id if fill == "eos" else pad_id] * (L - len(ids))


def decode_target(pretok, ids, eos_id, pad_id, cut_at_eos=True):
    """Cut at the first EOS and drop PAD, as upstream eval_shard.dec() does.

    cut_at_eos=False keeps everything (EOS tokens are dropped as specials): LM1B uses its
    EOS ([CLS]) as a sentence separator, so cutting would leave empty samples.
    """
    ids = list(ids)
    if cut_at_eos and eos_id in ids:
        ids = ids[:ids.index(eos_id)]
    return pretok.decode([t for t in ids if t not in (eos_id, pad_id)], skip_special_tokens=True)
