"""Flint for training: PyTorch modules on the transformers PreTrainedModel interface.

Ported from the SGLang implementation in Flint's serving kit (flint_sglang 0.1.4, model.py, built on SGLang's Qwen3Moe
classes), the only reference: Flint ships no transformers modeling code. Module and parameter names are the
checkpoint's (Qwen3Moe layout, experts one module each), so weights load and sync to SGLang by name. The numerics
follow the kit's validated serving path (FA3 attention, Triton MoE, FlashInfer CUDA RMSNorm, --enable-fp32-lm-head):

1. Attention: 32 query and 4 key/value heads of 128, per-head QK RMSNorm, a learned sink logit per head
   (``softmax_offset``) that joins the softmax denominator and carries no value. Sliding layers attend to the current
   token and the ``sliding_window - 1`` tokens before it (513 = current + 512). Layers in ``no_rope_layer_indices``
   use no positional encoding, the others neox-style RoPE.
2. MLP: dense SwiGLU on ``mlp_only_layers``; elsewhere a sparse MoE block: router logits from the bf16 gate, sigmoid
   scores, top-k on score + expert_bias (selection only; the checkpoint has no bias, so zero), weights = the chosen
   scores renormalized to sum to one, plus an ungated shared expert.
3. Residual stream as SGLang's fused add-RMSNorm: the sum in fp32, kept in bf16, normalized unrounded.
4. LM head in fp32.

Two parts are pluggable for the trainer. Attention: ``ATTENTION_FUNCTIONS[FlintAttention.attn_impl]``, called as
``fn(q, k, v, sinks, scale, window, cu_seqlens)``; ``"eager"`` is the exact fp32 reference used for parity. Experts:
``FlintSparseMoeBlock.experts_fn(experts, x, weights, ids)``, separate from the routing in ``FlintTopKRouter``, whose
``select`` is the expert-selection top-k a routing-replay hook rebinds.

Inputs are one packed sequence, as miles' FSDP backend sends them: ``input_ids`` [1, T] and ``position_ids`` [1, T]
restarting at 0 at each document; ``attention_mask`` must be None.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_flint import FlintConfig

SLIDING_ATTENTION = "sliding_attention"
SCORE_ELEMS = 1 << 28   # fp32 elements in one chunk of attention scores (1 GiB)


class FlintRMSNorm(nn.Module):
    """FlashInfer's CUDA RMSNorm: fp32 statistics and scaling, output in the input dtype. With a residual, the fused
    add-norm: returns (norm(x + residual), x + residual), the sum taken in fp32 and kept in the input dtype."""

    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def _norm(self, s: torch.Tensor, dtype) -> torch.Tensor:
        return (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(dtype)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return self._norm(x.float(), x.dtype)
        s = x.float() + residual.float()
        return self._norm(s, x.dtype), s.to(x.dtype)


def rope_cos_sin(position_ids: torch.Tensor, head_dim: int, theta: float):
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32) / head_dim)
    freqs = position_ids.float()[:, None] * inv_freq[None, :]
    return freqs.cos(), freqs.sin()   # [T, head_dim / 2]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Neox-style rotation (first half against second half) of [T, H, D], in fp32."""
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


def eager_sink_attention(q, k, v, sinks, scale: float, window: int | None, cu_seqlens: torch.Tensor):
    """Exact reference: causal attention with a learned sink logit per head, in fp32, document by document and in
    query chunks. q [T, H, D]; k, v [T, KV, D]; sinks [H]; window: a query attends to itself and the window - 1
    keys before it (None: every earlier key of its document); cu_seqlens [n + 1]: document boundaries."""
    T, H, D = q.shape
    rep = H // k.shape[1]
    qf = q.float().transpose(0, 1)                                   # [H, T, D]
    kf = k.float().repeat_interleave(rep, dim=1).transpose(0, 1)
    vf = v.float().repeat_interleave(rep, dim=1).transpose(0, 1)
    sink = sinks.float().view(H, 1, 1)
    out = []
    bounds = cu_seqlens.tolist()
    for a, b in zip(bounds[:-1], bounds[1:]):
        span = (b - a) if window is None else min(b - a, window + 2048)
        step = max(64, min(2048, SCORE_ELEMS // (H * span)))
        for s in range(a, b, step):
            e = min(b, s + step)
            lo = a if window is None else max(a, s - window + 1)
            sc = torch.matmul(qf[:, s:e], kf[:, lo:e].transpose(1, 2)) * scale    # [H, c, Lk]
            i = torch.arange(s, e, device=q.device)[:, None]
            j = torch.arange(lo, e, device=q.device)[None, :]
            allowed = (j <= i) if window is None else (j <= i) & (i - j < window)
            sc = torch.cat([sc.masked_fill(~allowed, float("-inf")), sink.expand(H, e - s, 1)], dim=-1)
            out.append(torch.matmul(torch.softmax(sc, dim=-1)[..., :-1], vf[:, lo:e]))
    return torch.cat(out, dim=1).transpose(0, 1).to(q.dtype)


ATTENTION_FUNCTIONS = {"eager": eager_sink_attention}


class FlintAttention(nn.Module):
    attn_impl = "eager"

    def __init__(self, config: FlintConfig, layer_idx: int):
        super().__init__()
        self.num_heads, self.num_kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.head_dim = config.head_dim
        h, bias = config.hidden_size, config.attention_bias
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=False)
        self.q_norm = FlintRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = FlintRMSNorm(self.head_dim, config.rms_norm_eps)
        self.softmax_offset = nn.Parameter(torch.zeros(self.num_heads))
        self.window = config.sliding_window if config.layer_types[layer_idx] == SLIDING_ATTENTION else None
        self.use_rope = layer_idx not in set(config.no_rope_layer_indices)
        self.scaling = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, rope, cu_seqlens: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        q = self.q_norm(self.q_proj(x).view(T, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(T, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).view(T, self.num_kv_heads, self.head_dim)
        if self.use_rope:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        o = ATTENTION_FUNCTIONS[self.attn_impl](q, k, v, self.softmax_offset, self.scaling, self.window, cu_seqlens)
        return self.o_proj(o.reshape(T, -1))


class FlintMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FlintTopKRouter(nn.Module):
    """``mlp.gate``: returns (router_logits, weights, ids). ``select`` picks the experts; routing replay rebinds it."""

    def __init__(self, config: FlintConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.weight = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size))
        self.register_buffer("expert_bias", torch.zeros(config.num_experts, dtype=torch.float32), persistent=False)

    def select(self, scores_for_choice: torch.Tensor, k: int) -> torch.Tensor:
        return torch.topk(scores_for_choice, k=k, dim=-1, sorted=False)[1]

    def forward(self, x: torch.Tensor):
        logits = F.linear(x, self.weight)
        scores = torch.sigmoid(logits.float())
        ids = self.select(scores.detach() + self.expert_bias, self.top_k)
        w = scores.gather(1, ids)
        return logits, w / (w.sum(dim=-1, keepdim=True) + 1e-20), ids


def experts_reference(experts: nn.ModuleList, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor):
    """Each expert on its tokens, weighted outputs summed in fp32, cast back to x's dtype."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    flat_ids, flat_w = ids.flatten(), weights.flatten()
    order = torch.argsort(flat_ids, stable=True)
    tok = order // ids.shape[1]
    start = 0
    for e, n in enumerate(torch.bincount(flat_ids, minlength=len(experts)).tolist()):
        if n:
            sel, t = order[start:start + n], tok[start:start + n]
            out.index_add_(0, t, experts[e](x[t]).float() * flat_w[sel, None])
            start += n
    return out.to(x.dtype)


class FlintSparseMoeBlock(nn.Module):
    experts_fn = staticmethod(experts_reference)

    def __init__(self, config: FlintConfig):
        super().__init__()
        self.gate = FlintTopKRouter(config)
        self.experts = nn.ModuleList(
            [FlintMLP(config.hidden_size, config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.shared_expert = (FlintMLP(config.hidden_size, config.shared_expert_intermediate_size)
                              if config.shared_expert_intermediate_size > 0 else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, weights, ids = self.gate(x)
        out = self.experts_fn(self.experts, x, weights, ids)
        return out + self.shared_expert(x) if self.shared_expert is not None else out


class FlintDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: FlintConfig, layer_idx: int):
        super().__init__()
        self.self_attn = FlintAttention(config, layer_idx)
        self.mlp = (FlintMLP(config.hidden_size, config.intermediate_size) if layer_idx in set(config.mlp_only_layers)
                    else FlintSparseMoeBlock(config))
        self.input_layernorm = FlintRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = FlintRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, residual, rope, cu_seqlens):
        if residual is None:
            hidden, residual = self.input_layernorm(hidden), hidden
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden = self.self_attn(hidden, rope, cu_seqlens)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        return self.mlp(hidden), residual


class FlintPreTrainedModel(PreTrainedModel):
    config_class = FlintConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FlintDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding, FlintTopKRouter)):
            module.weight.data.normal_(0.0, std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, FlintRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, FlintAttention):
            module.softmax_offset.data.zero_()


class FlintModel(FlintPreTrainedModel):
    def __init__(self, config: FlintConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([FlintDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = FlintRMSNorm(config.hidden_size, config.rms_norm_eps)
        rope = getattr(config, "rope_parameters", None) or {}
        self.rope_theta = float(rope.get("rope_theta", getattr(config, "rope_theta", 10000.0)))
        self.post_init()

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Final hidden states [T, hidden] of one packed sequence. The decoder layers may sit on several devices (a
        manual pipeline, see load_flint); the activations follow them."""
        if attention_mask is not None:
            raise ValueError("Flint takes packed sequences: attention_mask must be None, documents from position_ids")
        ids = input_ids.reshape(-1)
        pos = torch.arange(ids.numel(), device=ids.device) if position_ids is None else position_ids.reshape(-1)
        starts = torch.nonzero(pos == 0).flatten()
        if starts.numel() == 0 or starts[0] != 0:
            raise ValueError("position_ids must start each document at 0")
        cu_seqlens = torch.cat([starts, starts.new_tensor([ids.numel()])])
        hidden, residual, per_device = self.embed_tokens(ids), None, {}
        for layer in self.layers:
            dev = layer.input_layernorm.weight.device
            if dev not in per_device:
                p = pos.to(dev)
                per_device[dev] = (rope_cos_sin(p, self.config.head_dim, self.rope_theta), cu_seqlens.to(dev))
            if hidden.device != dev:
                hidden, residual = hidden.to(dev), (residual.to(dev) if residual is not None else None)
            hidden, residual = layer(hidden, residual, *per_device[dev])
        dev = self.norm.weight.device
        return self.norm(hidden.to(dev), residual.to(dev))[0]


class FlintForCausalLM(FlintPreTrainedModel):
    def __init__(self, config: FlintConfig):
        super().__init__(config)
        self.model = FlintModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """The fp32 LM head (the kit's --enable-fp32-lm-head)."""
        w = self.lm_head.weight
        return torch.matmul(hidden.to(w.device).float(), w.float().t())

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, **kwargs) -> CausalLMOutputWithPast:
        return CausalLMOutputWithPast(logits=self.logits(self.model(input_ids, position_ids, attention_mask))[None])

    def token_logprobs(self, hidden: torch.Tensor, targets: torch.Tensor, topk: int = 0, chunk: int = 2048):
        """log p(targets[i] | hidden[i]) without the full [T, vocab] logits; with topk, also the top-k head
        (logprobs, ids) at each position."""
        lps, tv, ti = [], [], []
        for s in range(0, hidden.shape[0], chunk):
            lp = torch.log_softmax(self.logits(hidden[s:s + chunk]), dim=-1)
            lps.append(lp.gather(1, targets[s:s + chunk, None].to(lp.device)).squeeze(1))
            if topk:
                v, i = lp.topk(topk, dim=-1)
                tv.append(v)
                ti.append(i)
        return (torch.cat(lps), torch.cat(tv), torch.cat(ti)) if topk else torch.cat(lps)


def load_flint(path: str, devices: list[str], dtype=torch.bfloat16) -> FlintForCausalLM:
    """Build Flint from a checkpoint directory with its decoder layers split into equal contiguous groups over
    ``devices`` (embeddings on the first, the final norm and LM head on the last): a manual pipeline for inference
    and checks on GPUs that cannot hold the whole model. Every checkpoint tensor must fill a parameter and every
    parameter must be filled."""
    from safetensors import safe_open

    path = Path(path)
    config = FlintConfig.from_pretrained(str(path))
    with torch.device("meta"):
        model = FlintForCausalLM(config)
    per = math.ceil(config.num_hidden_layers / len(devices))

    def device_of(name: str) -> str:
        if name.startswith("model.layers."):
            return devices[int(name.split(".")[2]) // per]
        return devices[0] if name.startswith("model.embed_tokens") else devices[-1]

    params = dict(model.named_parameters())
    index = json.load(open(path / "model.safetensors.index.json"))["weight_map"]
    by_file: dict[str, list[str]] = {}
    for name, f in index.items():
        by_file.setdefault(f, []).append(name)
    filled = set()
    for f, names in sorted(by_file.items()):
        with safe_open(str(path / f), "pt") as fh:
            for name in names:
                if name not in params:
                    raise RuntimeError(f"checkpoint tensor {name} fills no parameter")
                t = fh.get_tensor(name)
                if t.shape != params[name].shape:
                    raise RuntimeError(f"{name}: checkpoint {tuple(t.shape)}, model {tuple(params[name].shape)}")
                owner, attr = name.rsplit(".", 1)
                setattr(model.get_submodule(owner), attr, nn.Parameter(t.to(device_of(name), dtype)))
                filled.add(name)
    if filled != set(params):
        missing = sorted(set(params) - filled)
        raise RuntimeError(f"{len(missing)} parameters not in the checkpoint, e.g. {missing[:5]}")
    for name, buf in list(model.named_buffers()):
        owner, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(owner), attr, torch.zeros(buf.shape, dtype=buf.dtype, device=device_of(name)))
    return model
