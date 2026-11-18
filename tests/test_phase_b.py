"""Phase B / Stage 0 code paths: query_level0 generators (LM1B-style), window data,
corruption recipes, tokenizer training objective and checkpoint round trips."""
import json

import numpy as np
import torch
from omegaconf import OmegaConf

from conftest import EOS, LEVELS, PAD, V_CODE, V_TOK, make_vqvae, padded_prefixes, tiny_model_config, tiny_vq_config
from jancm.data import MixtureSampler, WindowSplit
from jancm.ncm_objective import Corruptor, build_batch, level_weights, ncm_loss, predicted_levels
from jancm.tokenizer_objective import (DualFineDrop, chunked_ce, decode_argmax, decoder_hidden, evaluate_tokenizer,
                                       kmeans_init_codebooks, tokenizer_loss)
from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE


def lm1b_style_net(vqvae, prefix_len=1, seed=3):
    """query_level0 + post-upsample convs + shared output head with per-level bias (released LM1B)."""
    cfg = tiny_model_config(prefix_len)
    cfg.predict_level0 = False
    cfg.query_level0 = True
    cfg.vocab_size = [V_CODE] * len(LEVELS)
    cfg.post_upsample_conv = {"enabled": True, "kernel_size": 3}
    cfg.shared_output_head = True
    cfg.shared_head_per_level_bias = True
    torch.manual_seed(seed)
    net = NextConceptTransformer(vqvae_local=vqvae, config=cfg).eval()
    with torch.no_grad():   # zero-initialised convs would hide layout bugs; make them non-trivial
        for c in net.post_upsample_convs:
            c.weight.normal_(0, 0.05)
    return net


def test_query_level0_training_layout_matches_generation(vqvae):
    net = lm1b_style_net(vqvae, prefix_len=6)
    prefix = padded_prefixes(5, 6, seed=7)
    codes = net.generate(None, prefix, temperature=0.0, use_kv_cache=False)
    assert codes.shape == (5, sum(LEVELS))
    tgt = vqvae.decoder(vqvae.vq.idx_to_z_q(codes)).argmax(-1)
    # Decoding then re-encoding is not exact for random weights, so teacher-force the codes
    # themselves by swapping the target encoder for the generated codes.
    orig = vqvae.encode
    vqvae.encode = lambda _: codes
    try:
        inputs, got = build_batch(net, vqvae, prefix, tgt, PAD)
    finally:
        vqvae.encode = orig
    with torch.no_grad():
        logits = net(**inputs)
    assert torch.equal(got, codes)
    assert torch.equal(logits.argmax(-1), codes)
    assert predicted_levels(net) == LEVELS


def test_query_level0_kv_cache_matches_eager(vqvae):
    """ja-ncm patch: KV caching allowed with query_level0 (upstream asserted it off)."""
    net = lm1b_style_net(vqvae, prefix_len=6)
    prefix = padded_prefixes(5, 6, seed=8)
    assert torch.equal(net.generate(None, prefix, temperature=0.0, use_kv_cache=False),
                       net.generate(None, prefix, temperature=0.0, use_kv_cache=True))


def test_unconditional_prefix_and_finetune_init(tmp_path, vqvae):
    """Pre-train with a 1-token null prefix, then initialise a 12-token-prefix fine-tune."""
    src = lm1b_style_net(vqvae, prefix_len=1)
    null = torch.full((3, 1), EOS)
    assert src.generate(None, null, temperature=1.0, top_k=0, top_p=1.0).shape == (3, sum(LEVELS))
    torch.save({"model": src.state_dict()}, tmp_path / "g.pt")
    from jancm.loading import build_generator

    class C:
        pass

    cfg = C()
    cfg.model = lm1b_style_net(vqvae).config.copy()
    cfg.model.prefix_len = 12

    class TK:
        vqvae = None

    tk = TK()
    tk.vqvae = vqvae
    dst = build_generator(cfg, tk, "cpu", init_ckpt=str(tmp_path / "g.pt"))
    assert dst.prefix_len == 12
    assert all(torch.equal(a, b) for a, b in zip(src.parameters(), dst.parameters()))


def test_random_rate_corruption():
    c = Corruptor(len(LEVELS), "random_rate", low=1.0, high=1.0, skip_level0=True)
    idx = torch.randint(0, V_CODE, (16, sum(LEVELS)))
    out = c(idx, LEVELS, [V_CODE] * len(LEVELS), torch.Generator().manual_seed(0))
    l0, fine = LEVELS[0], sum(LEVELS[:-1])
    assert torch.equal(out[:, :l0], idx[:, :l0]), "skip_level0"
    assert torch.equal(out[:, fine:], idx[:, fine:]), "finest level is never an input"
    assert (out[:, l0:fine] != idx[:, l0:fine]).float().mean() > 0.9
    none = Corruptor(len(LEVELS), "random_rate", low=0.0, high=0.0)
    assert torch.equal(none(idx, LEVELS, [V_CODE] * len(LEVELS)), idx)


def test_window_split_bin_and_shards(tmp_path):
    stream = np.arange(5 * 16, dtype=np.uint16)
    stream.tofile(tmp_path / "x.bin")
    ws = WindowSplit(str(tmp_path / "x.bin"), window=16, dtype="uint16", prefix_token=EOS)
    assert len(ws) == 5 and ws.windows([3, 1]).tolist()[0] == list(range(16, 32))
    d = tmp_path / "shards"
    d.mkdir()
    np.save(d / "windows_00000.npy", np.zeros((3, 16), dtype=np.int32))
    np.save(d / "windows_00001.npy", np.ones((2, 16), dtype=np.int32))
    (d / "meta.json").write_text(json.dumps({"kind": "windows", "window": 16,
                                             "shards": ["windows_00000.npy", "windows_00001.npy"]}))
    ws2 = WindowSplit(str(d), prefix_token=EOS)
    assert len(ws2) == 5 and ws2.windows([4]).tolist() == [[1] * 16] and ws2.windows([2]).tolist() == [[0] * 16]
    src, tgt = ws2.take([0, 4], "cpu")
    assert src.tolist() == [[EOS], [EOS]] and tgt.shape == (2, 16)
    rows = next(MixtureSampler(ws2, {"corpus": 1.0}, 8))
    assert rows.min() >= 0 and rows.max() < 5


def test_chunked_ce_matches_full_and_backprops():
    torch.manual_seed(0)
    h = torch.randn(37, 8, requires_grad=True)
    w = torch.randn(50, 8, requires_grad=True)
    t = torch.randint(0, 50, (37,))
    full = torch.nn.functional.cross_entropy(h @ w.t(), t)
    got = chunked_ce(h, w, t, chunk_rows=10)
    assert torch.allclose(full, got, atol=1e-5)
    gh, gw = torch.autograd.grad(full, (h, w))
    gh2, gw2 = torch.autograd.grad(chunked_ce(h, w, t, chunk_rows=10), (h, w))
    assert torch.allclose(gh, gh2, atol=1e-5) and torch.allclose(gw, gw2, atol=1e-5)


def tiny_trainable_vqvae(dual=True):
    vqc = tiny_vq_config()
    vqc.aux = {"fine_drop": {"dual": dual, "prob": 1.0 if dual else 0.5, "min_keep": 1, "w_full": 0.7, "w_trunc": 0.3}}
    torch.manual_seed(0)
    return VQVAE(in_vocab_size=V_TOK, context_length=LEVELS[-1], embed_dim=32, vqvae_vocab_size=[V_CODE] * len(LEVELS),
                 n_layers=4, vector_quantizer_config=vqc, compression_factor=1, dropout=0.0,
                 pre_quant_groupnorm=8, use_qk_norm=True, dec_layers=2, dec_positional=True), vqc


def test_decoder_hidden_matches_decoder():
    vq, _ = tiny_trainable_vqvae()
    vq.eval()
    z = torch.randn(2, LEVELS[-1], 32)
    full = vq.decoder(z)
    assert torch.allclose(vq.decoder.output_layer(decoder_hidden(vq.decoder, z)), full, atol=1e-5)
    assert torch.equal(decode_argmax(vq.decoder, z, chunk_rows=5), full.argmax(-1))


def test_tokenizer_training_learns_and_reloads(tmp_path):
    """Our VQ-VAE loop (dual nested dropout, EMA codebooks, k-means init) must learn to
    reconstruct, and its checkpoint must load strictly into a fresh VQVAE."""
    vq, vqc = tiny_trainable_vqvae()
    dual = DualFineDrop.from_vq_config(OmegaConf.create(vqc), len(LEVELS))
    assert dual is not None
    g = torch.Generator().manual_seed(1)
    data = torch.randint(2, 20, (64, LEVELS[-1]), generator=g)   # small vocabulary: learnable quickly
    kmeans_init_codebooks(vq, data, iters=3)
    before = evaluate_tokenizer(vq, [data], PAD, chunk_rows=64)["recon_acc_full"]
    opt = torch.optim.AdamW([p for p in vq.parameters() if p.requires_grad], lr=3e-3)
    vq.train()
    for i in range(150):
        loss, st = tokenizer_loss(vq, data[(i * 16) % 64:(i * 16) % 64 + 16], dual, chunk_rows=64)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert "ce_trunc" in st
    after = evaluate_tokenizer(vq, [data], PAD, chunk_rows=64)
    assert after["recon_acc_full"] > before + 20, (before, after)
    acc = after["recon_acc_per_level"]
    assert acc[-1] >= acc[0], "finer levels should reconstruct at least as well as the coarsest"
    torch.save({"model": vq.state_dict()}, tmp_path / "t.pt")
    fresh, _ = tiny_trainable_vqvae()
    fresh.load_state_dict(torch.load(tmp_path / "t.pt", weights_only=True)["model"], strict=True)
    fresh.eval()
    assert torch.equal(fresh.encode(data[:4]), vq.eval().encode(data[:4]))


def test_quantizer_keep_cumulative_matches_zhats():
    vq = make_vqvae()
    vq.vq.keep_cumulative = True
    vq.vq.eval()
    z = vq.encode_z(torch.randint(0, V_TOK, (2, LEVELS[-1])))
    vq.vq(z)
    cum = vq.vq.last_cumulative
    _, ref = vq.vq.z_to_idx(z, return_zhats=True)
    assert len(cum) == len(LEVELS) and all(torch.allclose(a, b, atol=1e-5) for a, b in zip(cum, ref))


def test_uniform_weights_cover_query_level0_targets(vqvae):
    net = lm1b_style_net(vqvae, prefix_len=2)
    src = padded_prefixes(4, 2)
    tgt = torch.randint(2, V_TOK, (4, LEVELS[-1]))
    inputs, codes = build_batch(net, vqvae, src, tgt, PAD, corrupt=Corruptor(len(LEVELS), "random_rate"))
    w = level_weights(predicted_levels(net), 0.0)
    loss, ce = ncm_loss(net(**inputs), codes, w)
    assert torch.isfinite(loss) and ce.shape == codes.shape == (4, sum(LEVELS))


def xsum_style_net(vqvae, prefix_len=6, seed=4):
    """Released XSum generator block: query_level0 + output heads tied to the codebooks."""
    cfg = tiny_model_config(prefix_len)
    cfg.predict_level0 = False
    cfg.query_level0 = True
    cfg.vocab_size = [V_CODE] * len(LEVELS)
    cfg.tie_heads_to_codebook = True
    cfg.tie_heads_learn_scale = True
    torch.manual_seed(seed)
    net = NextConceptTransformer(vqvae_local=vqvae, config=cfg).eval()
    with torch.no_grad():   # a non-trivial learned temperature and projection
        for h in net.lm_heads:
            h.log_inv_T.fill_(0.7)
    return net


def test_xsum_style_layout_kv_and_finetune_init(tmp_path, vqvae):
    from jancm.loading import build_generator
    net = xsum_style_net(vqvae)
    assert type(net.lm_heads[0]).__name__ == "_TiedCodebookHead"
    prefix = padded_prefixes(5, 6, seed=9)
    codes = net.generate(None, prefix, temperature=0.0, use_kv_cache=False)
    assert torch.equal(codes, net.generate(None, prefix, temperature=0.0, use_kv_cache=True))
    orig = vqvae.encode
    vqvae.encode = lambda _: codes
    try:
        inputs, got = build_batch(net, vqvae, prefix, torch.zeros(5, LEVELS[-1], dtype=torch.long), PAD)
    finally:
        vqvae.encode = orig
    with torch.no_grad():
        assert torch.equal(net(**inputs).argmax(-1), codes)
    # unconditional pre-training (1-token prefix) -> summarisation fine-tune (12-token prefix)
    pre = xsum_style_net(vqvae, prefix_len=1)
    torch.save({"model": pre.state_dict()}, tmp_path / "x.pt")

    class C:
        pass

    cfg = C()
    cfg.model = pre.config.copy()
    cfg.model.prefix_len = 12

    class TK:
        vqvae = None

    tk = TK()
    tk.vqvae = vqvae
    ft = build_generator(cfg, tk, "cpu", init_ckpt=str(tmp_path / "x.pt"))
    assert all(torch.equal(a, b) for a, b in zip(pre.parameters(), ft.parameters()))
    assert torch.equal(ft.lm_heads[0].E, vqvae.vq.embeddings[0].weight), "heads tie to the frozen codebooks"


def test_code_corruption_term_trains_decoder_and_encoder():
    """aux.code_corruption: an extra straight-through reconstruction from randomly replaced
    codes; it must reach both the decoder and the encoder and keep level 0 intact."""
    from jancm.tokenizer_objective import CodeCorruption, decoder_robustness
    vq, vqc = tiny_trainable_vqvae()
    vqc = OmegaConf.create(vqc)
    assert CodeCorruption.from_vq_config(vqc) is None
    vqc.aux.code_corruption = {"enabled": True, "mode": "uniform", "random_rate": True, "low": 1.0, "high": 1.0,
                               "skip_level0": True}
    cc = CodeCorruption.from_vq_config(vqc)
    codes = torch.zeros(8, sum(LEVELS), dtype=torch.long)
    bad = cc(codes, LEVELS, [16] * len(LEVELS))
    assert torch.equal(bad[:, :LEVELS[0]], codes[:, :LEVELS[0]]) and (bad[:, LEVELS[0]:] != 0).any()
    data = torch.randint(2, 20, (16, LEVELS[-1]), generator=torch.Generator().manual_seed(2))
    vq.train()
    loss, st = tokenizer_loss(vq, data, None, chunk_rows=64, corrupt=cc)
    assert "ce_corrupt" in st
    loss.backward()
    enc = [p.grad for n, p in vq.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    dec = [p.grad for n, p in vq.named_parameters() if n.startswith("decoder.") and p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in enc) and any(g is not None and g.abs().sum() > 0 for g in dec)
    r = decoder_robustness(vq, data, rates=(0.5,), chunk_rows=64)
    assert set(r) == {"robust_mid_0.5", "robust_all_0.5"}



def test_level_temperature_schedule_is_applied_per_level(vqvae):
    """generation.level_temperature: one temperature per level, applied in level order."""
    net = lm1b_style_net(vqvae)
    seen = []
    raw = net._sample_level_raw

    def spy(logits, temperature, top_k, top_p):
        seen.append(temperature)
        return raw(logits, temperature, top_k, top_p)
    net._sample_level_raw = spy                     # what generate_ids wraps
    K = len(net.levels)
    temps = [round(0.5 + 0.1 * k, 2) for k in range(K)]

    class TK:   # minimal tokenizer facade for generate_ids
        pass
    from types import SimpleNamespace
    tk = SimpleNamespace(vqvae=vqvae, pre_tokenizer=SimpleNamespace(eos_token_id=1, pad_token_id=0,
                         decode=lambda ids, skip_special_tokens=True: " ".join(map(str, ids))))
    cfg = OmegaConf.create({"generation": {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "use_kv_cache": True,
                                           "level_temperature": temps, "cut_at_eos": False}})
    from jancm.ncm_infer import generate_ids
    generate_ids(net, tk, [[1] * net.prefix_len] * 2, cfg, "cpu", batch_size=2)
    assert seen == temps
    assert net._sample_level_raw is spy             # the wrapper is removed afterwards
