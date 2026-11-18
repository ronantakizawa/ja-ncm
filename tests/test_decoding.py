import torch

from jancm.decoding import Utf8Constraint, parse_adjacency, pick_tokens

# toy byte vocabulary: 0 'a', 1 'あ' (3 bytes), 2 first byte of 'あ', 3 last two bytes of 'あ',
# 4 lone continuation byte, 5 special (b""), 6 padded column (None)
A = "あ".encode()
VOCAB = [b"a", A, A[:1], A[1:], A[2:], b""]


def test_utf8_constraint_states():
    c = Utf8Constraint(VOCAB, vocab_size=7)
    assert c.allowed[0].tolist() == [True, True, True, False, False, True, False]
    assert c.nxt[0, 2] == 2 and c.allowed[2].tolist() == [False, False, False, True, True, False, False]
    assert c.nxt[2, 3] == 0 and c.nxt[2, 4] == 1


def test_pick_tokens_utf8_repairs_and_ends_whole():
    c = Utf8Constraint(VOCAB, vocab_size=7)
    lg = torch.full((1, 3, 7), -5.0)
    lg[0, 0, 4] = 9   # lone continuation byte first: invalid -> next best
    lg[0, 0, 2] = 5   # lead byte of あ
    lg[0, 1, 0] = 9   # 'a' would break the pending character
    lg[0, 1, 3] = 4
    lg[0, 2, 2] = 9   # a lead byte at the end would leave it incomplete
    lg[0, 2, 0] = 1
    assert pick_tokens(lg, utf8=c).tolist() == [[2, 3, 0]]
    assert pick_tokens(lg).tolist() == [[4, 0, 2]]


def test_adjacency_penalty_breaks_doubling():
    lg = torch.zeros(1, 3, 4)
    lg[0, :, 1] = 5.0
    lg[0, :, 2] = 4.0
    assert pick_tokens(lg).tolist() == [[1, 1, 1]]
    assert pick_tokens(lg, adjacency=parse_adjacency("10,4,0.5")).tolist() == [[1, 2, 0]]


def test_sample_tokens_low_temperature_is_argmax_and_high_is_diverse():
    from jancm.decoding import sample_tokens
    torch.manual_seed(0)
    lg = torch.randn(4, 8, 50)
    assert torch.equal(sample_tokens(lg, 1e-4), lg.argmax(-1))
    hot = sample_tokens(lg.repeat(64, 1, 1), 5.0)
    assert len(set(hot[:, 0].tolist())) > 5
    nuc = sample_tokens(lg, 1.0, top_p=1e-6)
    assert torch.equal(nuc, lg.argmax(-1))
