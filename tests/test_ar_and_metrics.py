import torch

from conftest import EOS, PAD, V_TOK, padded_prefixes
from jancm.ar import ARTranslator, left_pad, right_to_left
from jancm.metrics_ja import CharTokenizer, length_units, rouge_scores

SEP = 2


def tiny_ar():
    torch.manual_seed(0)
    return ARTranslator(V_TOK, 16, n_layer=2, n_head=4, n_embd=32, dropout=0.0, rope_base=10000, max_len=40).set_eos(EOS).eval()


def test_right_to_left_padding():
    x = torch.tensor([[5, 6, 7, PAD, PAD], [8, 9, 10, 11, 12]])
    assert right_to_left(x, PAD).tolist() == [[PAD, PAD, 5, 6, 7], [8, 9, 10, 11, 12]]
    assert left_pad([[5, 6, 7], [1, 2, 3, 4, 5, 6]], 5, PAD) == [[PAD, PAD, 5, 6, 7], [1, 2, 3, 4, 5]]


def test_ar_kv_cache_matches_full_recompute():
    net = tiny_ar()
    src = right_to_left(padded_prefixes(5, 10), PAD)
    a, nfe_a = net.generate(src, SEP, PAD, EOS, max_new=12, use_kv_cache=True)
    b, nfe_b = net.generate(src, SEP, PAD, EOS, max_new=12, use_kv_cache=False)
    assert torch.equal(a, b) and nfe_a == nfe_b


def test_ar_padding_does_not_change_output():
    """Left padding plus the key mask must make extra PADs invisible."""
    net = tiny_ar()
    src = torch.tensor([[PAD, PAD, 5, 6, 7]])
    more = torch.tensor([[PAD, PAD, PAD, PAD, PAD, 5, 6, 7]])
    a, _ = net.generate(src, SEP, PAD, EOS, max_new=8)
    b, _ = net.generate(more, SEP, PAD, EOS, max_new=8)
    assert torch.equal(a, b)


def test_ar_overfits_and_ignores_target_pad():
    net = tiny_ar().train()
    src = right_to_left(padded_prefixes(4, 10), PAD)
    tgt = torch.full((4, 8), PAD)
    tgt[:, :5] = torch.randint(3, V_TOK, (4, 5))
    tgt[:, 5] = EOS
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    first = None
    for _ in range(80):
        loss = net(src, tgt, SEP, PAD, PAD)
        assert torch.isfinite(loss)
        first = first or float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.2 * first
    out, _ = net.eval().generate(src, SEP, PAD, EOS, max_new=8)
    assert torch.equal(out[:, :6], tgt[:, :6])


def test_ar_labels_stop_after_first_eos():
    tgt = torch.tensor([[7, 8, EOS, EOS, EOS], [7, EOS, PAD, PAD, PAD]])
    assert ARTranslator.after_first(tgt, EOS).tolist() == [[False, False, False, True, True],
                                                            [False, False, True, True, True]]


def test_target_window_fill():
    from jancm.text import target_window
    assert target_window([5, 6], 5, EOS, PAD) == [5, 6, EOS, PAD, PAD]
    assert target_window([5, 6], 5, EOS, PAD, fill="eos") == [5, 6, EOS, EOS, EOS]
    assert target_window([5, 6, 7, 8, 9], 5, EOS, PAD) is None


def test_japanese_metrics_without_mecab():
    assert CharTokenizer().tokenize("京都 は 古都") == list("京都は古都")
    assert length_units("京都は古都です", "ja") == 7 and length_units("a b c", "en") == 3
    r = rouge_scores(["京都は日本の古都である。"], ["京都は日本の古都である。"], "ja")
    assert abs(r["char"]["rouge1"] - 100.0) < 1e-6
    r2 = rouge_scores(["東京は首都だ。"], ["京都は古都だ。"], "ja")
    assert 0 < r2["char"]["rouge1"] < 100


def test_mecab_rouge_and_bleu_if_installed():
    import pytest
    pytest.importorskip("fugashi")
    pytest.importorskip("MeCab")
    import sacrebleu
    from jancm.metrics_ja import MeCabTokenizer, bleu_tokenize_arg
    assert MeCabTokenizer().tokenize("京都は日本の古都である。")[:3] == ["京都", "は", "日本"]
    r = rouge_scores(["東京は日本の首都である。"], ["京都は日本の古都である。"], "ja")
    assert 0 < r["mecab"]["rouge1"] < 100
    hyp, ref = ["京都は日本の古都である。"], [["京都は日本の古都です。"]]
    assert sacrebleu.corpus_bleu(hyp, ref, tokenize=bleu_tokenize_arg("ja")).score > 0
    assert sacrebleu.corpus_bleu(hyp, ref).score == 0   # why upstream's 13a default fails on Japanese
