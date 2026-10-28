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

Checks (`serving/check/`, H100, TP2, the eval servers' flags; jobs w533, w542, w546, w549 of 2026-10-28):

- Logprobs against the kit on 8 real long-game sequences of 2k-60k tokens: on assistant tokens the mean |delta| is
  0.032 nats, top-1 agreement 0.984, top-32 head KL 2.2e-3; flint_hf against the kit differs by 0.034, and one bf16
  rounding step inside flint_hf moves it 0.032-0.036. A batch of 8 gives bit-identical logprobs.
- Decode against prefill of the same sampled tokens (15 prompts, 512 tokens each): 0.073 nats, the kit 0.061.
- Real episodes (one Flint, 30k tokens): tool calls and reasoning parse; 1 of 42 turns had a markdown command block
  next to its tool call, a pattern the kit's own runs show at the same low rate.
- Decode throughput: 197.5 tokens/s for one request and 1,069 for eight, against the kit's 166.4 and 955.
- Weight updates through miles' non-colocated path (session frame, NCCL broadcast, 512 MiB buckets): all 36,734
  tensors sent back unchanged leave the served logprobs bit-identical, in 6.6 s; three changed tensors move the served
  logprobs as they move flint_hf's (served against flint_hf stays at 0.035 nats); the originals restore them bit for
  bit. The trainer must run with `NCCL_CUMEM_ENABLE=0`, as miles' Ray actors do: SGLang sets it to 0 in its own
  processes, and a communicator whose ranks disagree fails to set up.
