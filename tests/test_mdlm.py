import torch

from jancm.mdlm import MDLM

V, L = 50, 16


def tiny():
    torch.manual_seed(0)
    return MDLM(V, 8, n_layer=2, n_head=2, n_embd=32, dropout=0.0, max_len=L, emb_init=torch.randn(V, 8))


def test_mdlm_learns_and_never_predicts_mask():
    net = tiny()
    x = torch.randint(0, V, (64, L))
    x[:, ::2] = 7
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    first = float(net(x).detach())
    for _ in range(200):
        loss = net(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < 0.7 * first
    net.eval()
    ids, nfe = net.generate(32, L, 8, "cpu")
    assert nfe <= 8 and int((ids == net.mask_id).sum()) == 0 and int(ids.max()) < V
    assert float((ids[:, ::2] == 7).float().mean()) > 0.3      # learned structure (chance 0.02)


def test_mdlm_loss_weights_only_masked_positions():
    net = tiny().eval()
    x = torch.randint(0, V, (4, L))
    t = torch.full((4,), 1e-6)                                  # almost nothing masked
    assert float(net(x, t=t)) < 1e-3 * float(net(x, t=torch.ones(4)))


def test_generate_static_fills_every_position():
    from jancm.mdlm import generate_static
    net = tiny().eval()
    for steps in (1, 4, 16):
        ids = generate_static(net, 8, L, steps, "cpu")
        assert int((ids == net.mask_id).sum()) == 0 and int(ids.max()) < V


# ---------------------------------------------------------------- conditional (prefix) MDLM
P, PAD = 6, 0


def tiny_cond():
    torch.manual_seed(0)
    return MDLM(V, 8, n_layer=2, n_head=2, n_embd=32, dropout=0.0, max_len=P + L, emb_init=torch.randn(V, 8))


def _src(B):
    src = torch.randint(1, V, (B, P))
    src[0, :2] = PAD                                            # left padding in one row
    return src


def test_cond_source_ignores_target():
    """Source rows must not see the target, so their K/V can be cached once."""
    net = tiny_cond().eval()
    src = _src(3)
    m = MDLM.cond_mask(src != PAD, L)
    a = net.hidden(torch.cat([src, torch.randint(0, V, (3, L))], 1), m)[:, :P]
    b = net.hidden(torch.cat([src, torch.randint(0, V, (3, L))], 1), m)[:, :P]
    assert torch.allclose(a, b, atol=1e-6)


def test_cond_cached_generation_equals_uncached():
    net = tiny_cond().eval()
    src = _src(4)
    for order in ("confidence", "random"):
        torch.manual_seed(1)
        a = net.generate_cond(src, PAD, L, 4, order=order, use_cache=True)
        torch.manual_seed(1)
        b = net.generate_cond(src, PAD, L, 4, order=order, use_cache=False)
        assert torch.equal(a, b), order
        assert int((a == net.mask_id).sum()) == 0 and int(a.max()) < V
    for blk in net.blocks:                                      # cache released after generation
        assert blk.attn.cached_k is None and not blk.attn.caching


def test_cond_learns_to_copy_source():
    """Target = source tokens repeated: only learnable by attending to the prefix."""
    net = tiny_cond()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    for _ in range(300):
        src = torch.randint(1, V, (64, P))
        tgt = src.repeat(1, L // P + 1)[:, :L]
        loss = net(tgt, src_left=src, src_pad_id=PAD)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    src = torch.randint(1, V, (32, P))
    out = net.generate_cond(src, PAD, L, 8)
    acc = float((out == src.repeat(1, L // P + 1)[:, :L]).float().mean())
    assert acc > 0.5, acc                                        # chance 0.02
