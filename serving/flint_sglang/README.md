# flint_sglang for SGLang 0.5.23

Recursive's `flint_sglang` 0.1.4 (written for SGLang 0.5.16, `flint-eval/flint-sglang.sif`), ported to the SGLang
0.5.23.dev51 inside `miles-v0.1.3.sif`, so that miles' own rollout engines can serve Flint. The previous commit holds
the kit's files verbatim; `model.py` is the only module that changed:

- SGLang 0.5.23 replaced the `LayerCommunicator` with declared stage boundaries (`layers.layer_boundary`). The decoder
  layer now declares an attention stage and a dense or sparse FFN stage (`stage_facts`: dense on
  `config.mlp_only_layers`), as the stock Qwen3-MoE layer does.
- An FFN now returns a partial sum over tensor-parallel ranks, which the FFN boundary completes. The routed experts
  already do; the shared expert and the dense MLP are built with `reduce_results=False` so the sum is reduced once.
- `get_global_server_args()` is retired: the runtime checks read the operator's flags through `get_server_args()`, the
  pipeline group and DP-LM-head setting through `get_parallel()`.
- SGLang's true on-policy mode swaps the MoE router for softmax top-k, which is not Flint's sigmoid routing, so the
  model refuses to start in that mode.
- `load_weights` checks that every parameter was filled only on its first call (the startup load). Later calls are
  weight updates from an RL trainer, which carry a subset of the tensors; every call still refuses a tensor that
  matches no parameter.

Install into a directory of its own and put it on `PYTHONPATH` (its `sglang.srt.plugins` entry point registers the
model type, parsers and grammar backend): `serving/build_flint_sglang.sh <dir>`.
