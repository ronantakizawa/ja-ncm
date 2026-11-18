import torch

from conftest import LEVELS, PAD, V_CODE, V_TOK, make_net, padded_prefixes
from jancm.loading import build_generator
from jancm.ncm_objective import build_batch, corrupt_codes, corruption_rates, level_weights, ncm_loss


def test_idx_to_zhats_levels_are_distinct(vqvae):
    """Upstream issue #11: every z_hats entry aliased the final reconstruction."""
    vq = vqvae.vq
    idx = vqvae.encode(torch.randint(0, V_TOK, (3, LEVELS[-1])))
    (zh,) = vq.idx_to_zhats(idx, return_zlevels=False)
    assert len(zh) == len(LEVELS)
    assert not torch.allclose(zh[0], zh[-1])
    assert torch.allclose(zh[-1], vq.idx_to_z_q(idx), atol=1e-5)
    # Each entry must equal the per-level cumulative reconstruction from the encoder path.
    z = vqvae.encode_z(torch.randint(0, V_TOK, (3, LEVELS[-1])))
    idx2, ref = vq.z_to_idx(z, return_zhats=True)
    (got,) = vq.idx_to_zhats(idx2, return_zlevels=False)
    for a, b in zip(got, ref):
        assert torch.allclose(a, b, atol=1e-5)


def test_training_layout_matches_generation(vqvae, net):
    """Teacher forcing on greedily generated codes must reproduce those codes.

    This fails if build_batch (training) feeds a different prefix, padding mask, level
    input or RoPE layout than NextConceptTransformer.generate (inference).
    """
    P = net.prefix_len
    prefix = padded_prefixes(6, P)
    codes = net.generate(None, prefix, temperature=0.0, use_kv_cache=False)
    x_t = vqvae.vq.idx_to_ncm_input(codes)
    x = torch.cat([vqvae.encoder.wte(prefix), x_t], dim=1)
    with torch.no_grad():
        logits = net(x, prefix_pad_mask=prefix == PAD)
    # build_batch must assemble exactly this layout from the codes of its targets.
    tgt = torch.randint(2, V_TOK, (prefix.shape[0], LEVELS[-1]))
    inputs, got = build_batch(net, vqvae, prefix, tgt, PAD)
    want = vqvae.encode(tgt)
    assert torch.equal(got, want)
    assert torch.equal(inputs["x"], torch.cat([vqvae.encoder.wte(prefix), vqvae.vq.idx_to_ncm_input(want)], 1))
    assert torch.equal(inputs["prefix_pad_mask"], prefix == PAD)
    assert logits.shape[:2] == codes.shape
    assert torch.equal(logits.argmax(-1), codes)


def test_kv_cache_matches_eager_with_padding(net):
    """ja-ncm patch: the cached path used to ignore the prefix pad mask."""
    prefix = padded_prefixes(6, net.prefix_len, seed=5)
    eager = net.generate(None, prefix, temperature=0.0, use_kv_cache=False)
    cached = net.generate(None, prefix, temperature=0.0, use_kv_cache=True)
    assert torch.equal(eager, cached)


def test_level_weights():
    w0 = level_weights(LEVELS, 0.0)
    assert torch.isclose(w0.sum(), torch.tensor(1.0)) and torch.allclose(w0, w0[0].expand_as(w0))
    w1 = level_weights(LEVELS, 1.0)
    per_level, off = [], 0
    for l in LEVELS:
        per_level.append(float(w1[off:off + l].sum()))
        off += l
    assert max(per_level) - min(per_level) < 1e-6


def test_corruption_keeps_finest_level_and_range(vqvae):
    rates = corruption_rates(len(LEVELS), 1.0, 1.0)
    assert rates[-1] == 0.0 and len(rates) == len(LEVELS)
    idx = vqvae.encode(torch.randint(0, V_TOK, (8, LEVELS[-1])))
    out = corrupt_codes(idx, LEVELS, [V_CODE] * len(LEVELS), rates, torch.Generator().manual_seed(0))
    fine = sum(LEVELS[:-1])
    assert torch.equal(out[:, fine:], idx[:, fine:])
    assert not torch.equal(out[:, :fine], idx[:, :fine])
    assert int(out.min()) >= 0 and int(out.max()) < V_CODE
    assert all(abs(a - b) < 1e-9 for a, b in zip(corruption_rates(4, 0.5, 0.1), [0.5, 0.3, 0.1, 0.0]))


def test_overfits_one_batch(vqvae):
    net = make_net(vqvae).train()
    src = padded_prefixes(8, net.prefix_len)
    tgt = torch.randint(2, V_TOK, (8, LEVELS[-1]))
    w = level_weights(LEVELS, 0.0)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        inputs, codes = build_batch(net, vqvae, src, tgt, PAD)
        loss, _ = ncm_loss(net(**inputs), codes, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert all(p.grad is None for p in vqvae.parameters()), "tokenizer must stay frozen"
    assert losses[-1] < 0.3 * losses[0], losses[::10]


def test_init_from_checkpoint_with_other_prefix_len(tmp_path, vqvae):
    """The WMT14 generator (prefix 96) must initialise a prefix-128 model."""
    src = make_net(vqvae, prefix_len=10)
    path = tmp_path / "g.pt"
    torch.save({"model": src.state_dict()}, path)

    class Cfg:  # minimal stand-in for the OmegaConf root used by build_generator
        pass

    from conftest import tiny_model_config

    class TK:
        pass

    tk = TK()
    tk.vqvae = vqvae
    cfg = Cfg()
    cfg.model = tiny_model_config(prefix_len=14)
    dst = build_generator(cfg, tk, "cpu", init_ckpt=str(path))
    assert dst.prefix_len == 14
    sp, dp = dict(src.named_parameters()), dict(dst.named_parameters())
    assert sp.keys() == dp.keys() and all(torch.equal(sp[k], dp[k]) for k in sp)
    codes = dst.generate(None, padded_prefixes(2, 14), temperature=0.0)
    assert codes.shape == (2, sum(LEVELS))


def test_self_correcting_codes_reduce_to_encode_and_correct_errors(vqvae):
    """Rates 0 -> exactly vqvae.encode; with corruption, level 0 targets are unchanged while
    later targets are re-quantised against the corrupted reconstruction."""
    from jancm.ncm_objective import self_correcting_codes
    g = torch.Generator().manual_seed(0)
    x = torch.randint(2, 30, (6, vqvae.vq.levels[-1]), generator=g)
    K = len(vqvae.vq.levels)
    tgt, cond = self_correcting_codes(vqvae, x, torch.zeros(6, K))
    assert torch.equal(tgt, vqvae.encode(x)) and torch.equal(cond, tgt)
    rates = torch.full((6, K), 1.0)
    rates[:, 0] = 0.0
    rates[:, -1] = 0.0
    tgt2, cond2 = self_correcting_codes(vqvae, x, rates, generator=torch.Generator().manual_seed(1))
    l0 = vqvae.vq.levels[0]
    assert torch.equal(tgt2[:, :l0], tgt[:, :l0]) and torch.equal(cond2[:, :l0], tgt2[:, :l0])
    assert not torch.equal(tgt2[:, l0:], tgt[:, l0:])                 # later targets follow the corrupted path
    lf = vqvae.vq.levels[-1]
    assert torch.equal(cond2[:, -lf:], tgt2[:, -lf:])                 # the finest level is never an input


def test_build_batch_self_correct_shapes(vqvae, net):
    from jancm.ncm_objective import Corruptor, build_batch
    K = len(vqvae.vq.levels)
    c = Corruptor(K, "linear", 0.5, 0.2, skip_level0=True, self_correct=True)
    x = torch.randint(2, 30, (3, vqvae.vq.levels[-1]))
    src = torch.zeros(3, net.prefix_len, dtype=torch.long)
    inputs_sc, codes_sc = build_batch(net, vqvae, src, x, 0, corrupt=c)
    c.self_correct = False
    inputs, codes = build_batch(net, vqvae, src, x, 0, corrupt=c)
    assert inputs_sc["x"].shape == inputs["x"].shape and codes_sc.shape == codes.shape
