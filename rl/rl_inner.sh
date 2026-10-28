#!/bin/bash
# Inside miles-v0.1.3.sif with a PID namespace of its own (rl/rl_job.sh): a job-private Ray head on job-derived ports,
# started with the run's environment (Ray's workers, the agent processes among them, take theirs from it), then miles'
# train.py. miles: our fork (branch flint-grpo of wenhaochai/miles-private, worktree $MILES); the training model is
# flint_hf from this repository. Usage: rl_inner.sh memprobe|test2|run20; env JID, OUT, PKG (the built Flint plugin),
# TMPDIR. Mode args: the argument check alone, off GPU.
set -uo pipefail
: "${JID:?}" "${OUT:?}" "${PKG:?}" "${TMPDIR:?}"
MODE=$1
REPO=/scratch/gpfs/GROUP/USER/project/mas-rl-rollout
FLINT=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint
MILES=/scratch/gpfs/GROUP/USER/project/miles-rollout
MEMPROBE_DATA=/scratch/gpfs/GROUP/USER/tmp/flint_s5/memprobe_data
export HOME=$TMPDIR/home TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export SGLANG_CACHE_DIR=$TMPDIR/sglang FLASHINFER_WORKSPACE_BASE=$TMPDIR/flashinfer
mkdir -p "$HOME"
export PYTHONPATH=$PKG:$REPO:$MILES PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1   # the container's writable layer is 64 MB
export RAY_USAGE_STATS_ENABLED=0 no_proxy=127.0.0.1,localhost MASTER_ADDR=127.0.0.1
export SGLANG_EXTERNAL_MODEL_PACKAGE=flint_sglang FLASHINFER_USE_CUDA_NORM=1 SGLANG_RETURN_ORIGINAL_LOGPROB=0
# the game: one Flint, 128k generated tokens per episode (budget end with forced restart, its defaults); the bash tool
# without strict (see flint/chat.py); the judge counts the user's tasks on the node (rl/rl_job.sh binds /proc there)
export MA_LG_N=1 MA_LG_TOKENS=128000 MA_AG_TOOL_STRICT=0 FCS_PROC=/host/proc
# flint.miles_agent.log_rollout: the run directory, and where the agents write traces (moved per rollout)
export MA_RL_OUT=$OUT MA_AG_TRACE_DIR=$OUT/traces_live
ulimit -n "$(ulimit -Hn)"
NG=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
JNUM=${JID//[!0-9]/}
P=$((30000 + (JNUM % 1000) * 16))

# the FSDP actor (S2's step, flint_hf through the fork's Flint adapter): full fine-tuning, fp32 master weights and Adam
# on the CPU (FSDP2 CPU offload), gradient checkpointing; GRPO advantages without std normalization; score centering
# over the sampler's top-32 candidates (support mode: top_k 20 <= 32), no importance weight; one optimizer step per
# rollout (8 episodes = 8 rollouts of miles' compact samples per step); routing replay of the rollout's experts, with
# the replayed-vs-own top-k mismatch logged per step. A sample longer than --max-tokens-per-gpu forms a micro-batch alone.
TRAIN=(--train-backend fsdp --hf-checkpoint "$FLINT" --attn-implementation eager
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NG" --seed 1 --fsdp-cpu-offload
  --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "11", "MKL_NUM_THREADS": "11"}'
  --advantage-estimator grpo --disable-grpo-std-normalization --kl-coef 0 --entropy-coef 0
  --loss-type score_centering --score-centering-is none --rollout-top-logprobs-num 32
  --optimizer adam --lr 2e-7 --lr-decay-style constant --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.98
  --gradient-checkpointing --use-dynamic-batch-size --max-tokens-per-gpu 131072
  --rollout-batch-size 1 --n-samples-per-prompt 8 --global-batch-size 8
  --use-rollout-routing-replay --log-routing-replay-mismatch)
# expandable segments only in the actor: torch_memory_saver, which frees the engines' memory while the actor trains,
# turns itself off in a process that has them

# Flint's serving flags (the eval servers: TP2, fa3, Triton MoE, fp32 LM head); 0.72 of each GPU for the engine (its
# weights ~37 GB, KV for 2 episodes of up to ~150k tokens), the rest for what the offloaded actor keeps on the GPU
SERVE=(--rollout-num-gpus "$NG" --rollout-num-gpus-per-engine 2
  --sglang-attention-backend fa3 --sglang-moe-runner-backend triton --sglang-enable-fp32-lm-head
  --sglang-context-length 262144 --sglang-chunked-prefill-size 4096 --sglang-swa-full-tokens-ratio 0.4
  --sglang-mem-fraction-static 0.72 --sglang-grammar-backend flint)
# the long game as a miles agent on problem 0 (flint/miles_agent.py), sampling as in the eval; the engines return each
# token's routed experts, incremental rows per turn (pause mode in_place); an episode the judge could not score takes
# its group out of the loss (drop_failed_groups) and fails the rollout's check (log_rollout), which ends the run
GAME=(--prompt-data "$OUT/data_p0.jsonl" --input-key prompt --label-key label --metadata-key metadata
  --rollout-temperature 1.0 --rollout-top-p 0.95 --rollout-top-k 20
  --rollout-max-response-len 32768 --max-seq-len 262144
  --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
  --custom-agent-function-path flint.miles_agent.run --use-session-server v2 --tito-model flint
  --session-message-matcher flint.miles_agent.message_matches
  --sglang-tool-call-parser flint --sglang-reasoning-parser flint --pause-generation-mode in_place
  --rollout-sample-filter-path flint.miles_agent.drop_failed_groups
  --custom-rollout-log-function-path flint.miles_agent.log_rollout
  --save-debug-rollout-data "$OUT/r{rollout_id}/rollout.pt")
# colocated: the engines release weights and KV while the actor trains, the actor goes to the CPU while they generate;
# the weights go to the engines after every step. The actor (model only, no optimizer) is saved once, after the last
# step: 1000 rows of one problem keep miles' epoch boundary, also a save point, out of the run.
RL=(--colocate --save "$OUT/actor" --save-interval 100000 --no-save-optim)

MEMPROBE=(--prompt-data "$OUT/data_p0.jsonl" --input-key prompt --label-key label --metadata-key metadata --num-rollout 2
  --load-debug-rollout-data "$MEMPROBE_DATA/rollout_{rollout_id}.pt" --rollout-temperature 1.0 --rollout-top-p 0.95
  --rollout-top-k 20)
if [ "$MODE" = args ]; then   # off-GPU: miles parses and checks each mode's arguments (rl/args_check.py)
  python3 "$REPO/rl/args_check.py" "${TRAIN[@]}" "${MEMPROBE[@]}" || exit 1
  python3 "$REPO/rl/schedule_check.py" "$MEMPROBE_DATA/rollout_0.pt" "${TRAIN[@]}" "${MEMPROBE[@]}" || exit 1
  python3 "$REPO/rl/args_check.py" "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" "${RL[@]}" --num-rollout 2
  exit
fi

echo "[s5] $(date -u +%FT%TZ) inner $MODE: $NG GPUs, ray port $P"
ray start --head --node-ip-address 127.0.0.1 --num-gpus "$NG" --port=$P --dashboard-port=$((P + 1)) \
  --dashboard-agent-listen-port=$((P + 2)) --dashboard-agent-grpc-port=$((P + 3)) --metrics-export-port=$((P + 4)) \
  --ray-client-server-port=$((P + 5)) --runtime-env-agent-port=$((P + 6)) --node-manager-port=$((P + 7)) \
  --object-manager-port=$((P + 8)) --include-dashboard=false --disable-usage-stats --temp-dir=/tmp/ray >/dev/null \
  || { echo "[s5] ray start failed"; exit 1; }
export RAY_ADDRESS=127.0.0.1:$P
trap 'ray stop --force >/dev/null 2>&1' EXIT   # this PID namespace's Ray only

t0=$SECONDS
case "$MODE" in
  memprobe)   # training alone, 2 steps on the same batch: memory at the real sequence lengths, R3 on real experts
    python3 "$MILES/train.py" "${TRAIN[@]}" "${MEMPROBE[@]}" > "$OUT/train.log" 2>&1 ;;
  test2) python3 "$MILES/train.py" "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" "${RL[@]}" --num-rollout 2 > "$OUT/train.log" 2>&1 ;;
  run20) python3 "$MILES/train.py" "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" "${RL[@]}" --num-rollout 20 > "$OUT/train.log" 2>&1 ;;
esac
rc=$?
echo "[s5] $(date -u +%FT%TZ) train.py exit $rc after $((SECONDS - t0)) s"
python3 "$REPO/rl/steps.py" "$OUT"
exit $rc
