"""CPU checks of flint_hf on a tiny random Flint: the attention against an explicit softmax, causality, the window's
reach (it counts the current token), chunking invariance, packed documents kept apart, the sink, the MoE block
against a per-token reference, finite gradients under gradient checkpointing. In flint-sglang.sif:
PYTHONPATH=. python3 tests/test_flint_hf_tiny.py"""
import torch

import flint_hf.modeling_flint as M
from flint_hf import FlintConfig, FlintForCausalLM


def tiny(layer_types=("full_attention", "sliding_attention", "sliding_attention"), no_rope=(0,), window=5):
    cfg = FlintConfig(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=len(layer_types),
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, moe_intermediate_size=32,
                      shared_expert_intermediate_size=48, num_experts=8, num_experts_per_tok=2, sliding_window=window,
                      layer_types=list(layer_types), no_rope_layer_indices=list(no_rope), mlp_only_layers=[0])
    torch.manual_seed(0)
    m = FlintForCausalLM(cfg).float()
    for mod in m.modules():
        if isinstance(mod, M.FlintAttention):
            mod.softmax_offset.data.normal_(0, 1)
        if isinstance(mod, M.FlintRMSNorm):
            mod.weight.data.uniform_(0.5, 1.5)
    return m


def attn(q, k, v, sinks, window=None, cu=None):
    cu = torch.tensor([0, q.shape[0]]) if cu is None else cu
    return M.eager_sink_attention(q, k, v, sinks, 0.3, window, cu)


def test_attention_matches_explicit_softmax():
    T, H, D = 300, 4, 8
    q, k, v, sinks = torch.randn(T, H, D), torch.randn(T, 2, D), torch.randn(T, 2, D), torch.randn(H)
    kk, vv = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
    for window in (None, 17):
        sc = torch.einsum("thd,shd->hts", q, kk) * 0.3
        i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
        ok = (j <= i) if window is None else (j <= i) & (i - j < window)
        sc = sc.masked_fill(~ok, float("-inf"))
        p = torch.softmax(torch.cat([sc, sinks.view(H, 1, 1).expand(H, T, 1)], -1), -1)[..., :-1]
        ref = torch.einsum("hts,shd->thd", p, vv)
        old = M.SCORE_ELEMS
        for elems in (old, 1):   # one chunk, then 64-query chunks
            M.SCORE_ELEMS = elems
            assert torch.allclose(attn(q, k, v, sinks, window), ref, atol=1e-5), (window, elems)
        M.SCORE_ELEMS = old
    assert attn(q, k, v, sinks + 50).abs().max() < 1e-6   # a large sink takes the mass and carries no value


def test_window_counts_current_token():
    T, H, D = 12, 2, 4
    q, k, v = torch.randn(T, H, D), torch.randn(T, 1, D), torch.randn(T, 1, D)
    a = attn(q, k, v, torch.zeros(H), 3)
    v2 = v.clone(); v2[8] += 10   # position 11 sees 9, 10, 11; position 10 sees 8, 9, 10
    b = attn(q, k, v2, torch.zeros(H), 3)
    assert torch.allclose(a[11], b[11]) and not torch.allclose(a[10], b[10])


def test_causal_and_window_in_the_model():
    m = tiny(layer_types=("sliding_attention",), no_rope=())
    ids = torch.randint(0, 97, (40,))
    h = m.model(ids)
    ids2 = ids.clone(); ids2[30:] = (ids2[30:] + 1) % 97
    assert torch.allclose(m.model(ids2)[:30], h[:30], atol=1e-6), "future tokens leak"
    ids3 = ids.clone(); ids3[:20] = (ids3[:20] + 1) % 97   # one sliding layer of 5: position p sees p-4..p
    h3 = m.model(ids3)
    assert torch.allclose(h3[24:], h[24:], atol=1e-6) and not torch.allclose(h3[20:24], h[20:24], atol=1e-6)


def test_packed_documents_stay_apart():
    m = tiny()
    a, b = torch.randint(0, 97, (23,)), torch.randint(0, 97, (31,))
    pos = torch.cat([torch.arange(23), torch.arange(31)])
    packed = m.model(torch.cat([a, b])[None], pos[None])
    assert torch.allclose(packed[:23], m.model(a), atol=1e-5) and torch.allclose(packed[23:], m.model(b), atol=1e-5)
    cu = torch.tensor([0, 23, 54], dtype=torch.int32)   # miles v0.1.3 passes the boundaries as HF's cu_seq_lens_q
    assert torch.equal(m(torch.cat([a, b])[None], pos[None], None, cu_seq_lens_q=cu).logits[0], m.logits(packed))


def test_moe_block_matches_per_token_reference():
    blk = tiny().model.layers[1].mlp
    x = torch.randn(23, 64)
    s = torch.sigmoid(x @ blk.gate.weight.t())
    ids = s.topk(2, -1).indices
    w = s.gather(1, ids); w = w / w.sum(-1, keepdim=True)
    ref = torch.stack([sum(w[i, j] * blk.experts[ids[i, j]](x[i:i + 1])[0] for j in range(2)) for i in range(23)])
    assert torch.allclose(blk(x), ref + blk.shared_expert(x), atol=1e-5)


def test_backward_finite_with_checkpointing():
    m = tiny()
    m.gradient_checkpointing_enable()
    m.train()
    ids = torch.randint(0, 97, (33,))
    out = m(ids[None, :-1])
    lp = torch.log_softmax(out.logits[0], -1).gather(1, ids[1:, None])
    (-lp.mean()).backward()
    bad = [n for n, p in m.named_parameters()
           if ".experts." not in n and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not bad, bad   # (experts no token chose get no gradient)
    assert torch.allclose(m.token_logprobs(m.model(ids[:-1]), ids[1:]), lp[:, 0], atol=1e-5)


def test_save_and_from_pretrained_round_trip():
    import tempfile
    m = tiny()
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        m2 = FlintForCausalLM.from_pretrained(d, dtype=torch.float32)
    p2 = dict(m2.named_parameters())
    assert all(torch.equal(p, p2[n]) for n, p in m.named_parameters()), "from_pretrained re-initialized weights"
    assert all(float(b.abs().sum()) == 0 for n, b in m2.named_buffers() if n.endswith("expert_bias"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
