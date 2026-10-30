# Sourced by rl/rl_inner_ext.sh (the trainer) and rl/engine_inner.sh (the engine job): the one place for the
# arguments both read, so the engine serves with the flags the trainer's miles validates. Needs FLINT, NG (the
# trainer's GPU count; the engine job passes 8), ENGINE_TP, ENGINE_NGPUS (the GPUs of all engines) and
# ENGINE_MEM (the engines' --mem-fraction-static). The values equal rl/rl_inner.sh's run20 arrays except where noted.

# the FSDP actor: as rl/rl_inner.sh (score centering over the top-32 candidates, R3, one step per 8 episodes)
TRAIN=(--train-backend fsdp --hf-checkpoint "$FLINT" --attn-implementation eager
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NG" --seed 1 --fsdp-cpu-offload
  --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "11", "MKL_NUM_THREADS": "11"}'
  --advantage-estimator grpo --disable-grpo-std-normalization --kl-coef 0 --entropy-coef 0
  --loss-type score_centering --score-centering-is none --rollout-top-logprobs-num 32
  --optimizer adam --lr 2e-7 --lr-decay-style constant --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.98
  --gradient-checkpointing --use-dynamic-batch-size --max-tokens-per-gpu 65536
  --rollout-batch-size 1 --n-samples-per-prompt 8 --global-batch-size 8
  --use-rollout-routing-replay --log-routing-replay-mismatch)

# Flint's serving flags, as run20's 1 x TP8 engine (its sglang.launch_server line in runs/run20_p0/train.log), with
# three differences that come from not being colocated: no --enable-memory-saver (nothing to release for an actor),
# --mem-fraction-static ENGINE_MEM (0.85: the whole GPU is the engine's; run20 0.72), and no --gated-launch-port
# (nothing activates the gate). --sglang-cuda-graph-backend-prefill disabled is miles' colocate default, which run20
# served with; set here so the external engine serves the same way. The parsers are here, not in GAME, because the
# engine needs them.
SERVE=(--rollout-num-gpus "$ENGINE_NGPUS" --rollout-num-gpus-per-engine "$ENGINE_TP"
  --sglang-attention-backend fa3 --sglang-moe-runner-backend triton --sglang-enable-fp32-lm-head
  --sglang-context-length 262144 --sglang-chunked-prefill-size 4096 --sglang-swa-full-tokens-ratio 0.4
  --sglang-mem-fraction-static "$ENGINE_MEM" --sglang-grammar-backend flint --sglang-cuda-graph-backend-prefill disabled
  --sglang-tool-call-parser flint --sglang-reasoning-parser flint)
