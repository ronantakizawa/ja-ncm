"""Metrics that work on Japanese output (used by the patched ncm/scripts/eval_merge.py).

Upstream problems this fixes:
  * rouge_score's default tokenizer keeps only [a-z0-9], so Japanese ROUGE is always 0;
  * ``len(h.split())`` counts one "word" per Japanese sentence;
  * sacreBLEU's 13a tokenizer does not segment Japanese.
"""
import re

_SPACE = re.compile(r"\s+")


class MeCabTokenizer:
    """rouge_score-compatible tokenizer (``.tokenize(text) -> list[str]``) using fugashi + unidic-lite."""

    def __init__(self):
        import fugashi  # optional dependency: pip install -r requirements-ja.txt
        self.tagger = fugashi.Tagger()

    def tokenize(self, text):
        return [w.surface for w in self.tagger(text) if w.surface.strip()]


class SacreBleuJaMecab:
    """sacreBLEU's ja-mecab segmentation (MeCab + IPAdic) as a rouge_score tokenizer.

    The same segmentation as BLEU(tokenize="ja-mecab") and llm-jp-eval's Japanese ROUGE,
    so BLEU and ROUGE agree on what a Japanese word is.
    """

    def __init__(self):
        from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
        self.tok = TokenizerJaMecab()

    def tokenize(self, text):
        return self.tok(text).split()


class CharTokenizer:
    """Character-level tokens, whitespace dropped. Needs no MeCab."""

    def tokenize(self, text):
        return list(_SPACE.sub("", text))


def length_units(text, lang):
    """Characters for Japanese (no spaces between words), whitespace words otherwise."""
    return len(_SPACE.sub("", text)) if lang == "ja" else len(text.split())


def bleu_tokenize_arg(lang):
    return "ja-mecab" if lang == "ja" else "13a"


def rouge_scores(hyps, refs, lang):
    """Mean ROUGE-1/2/L F1 (x100). For Japanese, returns MeCab-word and character variants."""
    from rouge_score import rouge_scorer
    keys = ["rouge1", "rouge2", "rougeL"]
    if lang == "ja":
        variants = {"char": CharTokenizer()}
        try:
            variants["mecab"] = SacreBleuJaMecab()      # headline: same segmentation as ja-mecab BLEU
        except Exception:
            print("[metrics_ja] sacrebleu[ja] (mecab-python3 + ipadic) missing; character-level ROUGE only")
    else:
        variants = {"word": None}
    out = {}
    for tag, tok in variants.items():
        sc = (rouge_scorer.RougeScorer(keys, tokenizer=tok) if tok is not None
              else rouge_scorer.RougeScorer(keys, use_stemmer=True))
        acc = {k: 0.0 for k in keys}
        for h, r in zip(hyps, refs):
            v = sc.score(r, h)
            for k in keys:
                acc[k] += v[k].fmeasure
        out[tag] = {k: 100 * acc[k] / max(1, len(hyps)) for k in keys}
    return out
