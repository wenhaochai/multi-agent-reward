#!/bin/bash
# Inside miles-v0.1.3.sif with a PID namespace of its own (rl/rl_job.sh): a job-private Ray head on job-derived ports,
# started with the run's environment (Ray's workers, the agent processes among them, take theirs from it), then miles'
# train.py. miles: our fork (branch flint-grpo of wenhaochai/miles-private, worktree $MILES); the training model is
# flint_hf from this repository. Usage: rl_inner.sh smoke|test2|run20|args; env JID, OUT, PKG (the built Flint plugin),
# TMPDIR. Mode args: the argument check alone, off GPU.
set -uo pipefail
: "${JID:?}" "${OUT:?}" "${PKG:?}" "${TMPDIR:?}"
MODE=$1
REPO=/scratch/gpfs/GROUP/USER/project/mas-rl-rollout
FLINT=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint
MILES=/scratch/gpfs/GROUP/USER/project/miles-rollout
MEMPROBE_DATA=/scratch/gpfs/GROUP/USER/tmp/flint_s5/memprobe_data   # a recorded batch for the schedule check
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
# RL's context cap (NOT the eval's 400k characters and 32k-token turns): a training sample (one TITO leaf: the prompt
# plus one completion) must stay near the 64k tokens S2's actor step was measured at (55-60 GB per GPU), since the FSDP
# actor has no context parallelism. The prompt is <= 8.9k tokens (system, instance) + 0.31 tokens per history
# character (w562: 132,040 tokens at most with 400k characters) and the completion <= MA_AG_TURN, so 110k characters
# and 16k-token turns bound a sample at ~59k tokens (w562: 2 of 640 turns were over 16,384 tokens).
# flint.miles_agent.log_rollout ends the run before training if a sample is over MA_RL_MAX_SAMPLE_TOKENS.
export MA_LG_CTX_CHARS=110000 MA_AG_TURN=16384 MA_RL_MAX_SAMPLE_TOKENS=70000
# a wall cap per episode: 128k episodes take 17-44 min (w634, w647), and one at 67 min (w647 rollout 0) kept 3 engines
# at 0% for 45-50 min; at 45 min an episode ends as at its token budget (scored the same way), so an engine idles at
# most ~28 min. The game RL optimizes is then the time-limited one; it changes only episodes slower than 45 min.
export MA_LG_WALL_MIN=45
# flint.miles_agent.log_rollout: the run directory, and where the agents write traces (moved per rollout)
export MA_RL_OUT=$OUT MA_AG_TRACE_DIR=$OUT/traces_live
# smoke: the same path, short: 8 episodes x 4k tokens per step, the context cut to 20k characters (trims happen). The
# game's settings must be in the environment before `ray start` below: Ray's workers, the agents among them, take the
# environment Ray started with (w624's smoke set them after it, and its agents played 128k). Each episode checks its
# game's settings against $OUT/game_settings.json (flint.miles_agent.run) and stops the run if they differ.
[ "$MODE" = smoke ] && export MA_LG_TOKENS=4000 MA_LG_CTX_CHARS=20000
python3 -c "import json, os; json.dump({k: os.environ[k] for k in ('MA_LG_N', 'MA_LG_TOKENS', 'MA_LG_CTX_CHARS', 'MA_AG_TURN', 'MA_LG_WALL_MIN')}, open('$OUT/game_settings.json', 'w'))"
ulimit -n "$(ulimit -Hn)"
NG=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
JNUM=${JID//[!0-9]/}
P=$((30000 + (JNUM % 1000) * 16))

# the FSDP actor (S2's step, flint_hf through the fork's Flint adapter): full fine-tuning, fp32 master weights and Adam
# on the CPU (FSDP2 CPU offload), gradient checkpointing; GRPO advantages without std normalization; score centering
# over the sampler's top-32 candidates (support mode: top_k 20 <= 32), no importance weight; one optimizer step per
# rollout (8 episodes = 8 rollouts of miles' compact samples per step); routing replay of the rollout's experts, with
# the replayed-vs-own top-k mismatch logged per step. Micro-batches of at most 64k tokens (one sample alone if longer);
# a step with too few samples to give every rank its micro-batches gets zero-weight padding copies (the fork's schedule).
TRAIN=(--train-backend fsdp --hf-checkpoint "$FLINT" --attn-implementation eager
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NG" --seed 1 --fsdp-cpu-offload
  --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "11", "MKL_NUM_THREADS": "11"}'
  --advantage-estimator grpo --disable-grpo-std-normalization --kl-coef 0 --entropy-coef 0
  --loss-type score_centering --score-centering-is none --rollout-top-logprobs-num 32
  --optimizer adam --lr 2e-7 --lr-decay-style constant --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.98
  --gradient-checkpointing --use-dynamic-batch-size --max-tokens-per-gpu 65536
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
  --rollout-max-response-len 16384 --max-seq-len 262144
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
RL=(--colocate --save "$OUT/actor" --save-interval 100000 --no-save-optim --save-trigger-sentinel "$OUT/save_now")
# run20 resumes: actor, optimizer, scheduler, RNG and the data position every 2 steps (and at the end), loaded from the
# same directory on every start (miles skips the load when there is none yet); log_rollout keeps the newest 2
RUN20=(--colocate --save "$OUT/actor" --load "$OUT/actor" --save-interval 2 --save-trigger-sentinel "$OUT/save_now")

MEMPROBE=(--prompt-data "$OUT/data_p0.jsonl" --input-key prompt --label-key label --metadata-key metadata --num-rollout 2
  --load-debug-rollout-data "$MEMPROBE_DATA/rollout_{rollout_id}.pt" --rollout-temperature 1.0 --rollout-top-p 0.95
  --rollout-top-k 20)
if [ "$MODE" = args ]; then   # off-GPU: miles parses and checks each mode's arguments (rl/args_check.py)
  python3 "$REPO/rl/args_check.py" "${TRAIN[@]}" "${MEMPROBE[@]}" || exit 1
  python3 "$REPO/rl/schedule_check.py" "$MEMPROBE_DATA/rollout_0.pt" "${TRAIN[@]}" "${MEMPROBE[@]}" || exit 1
  python3 "$REPO/rl/args_check.py" "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" "${RL[@]}" --num-rollout 2 || exit 1
  python3 "$REPO/rl/args_check.py" "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" "${RUN20[@]}" --num-rollout 20
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

# train.py in the background; once the engines serve (the first rollout request), the served-weights probe runs once
# on the initial weights (flint.miles_agent.startup_probe): if it fails, train.py is stopped, minutes into the run
train() {
  if [ -n "${APPEND:-}" ]; then echo "=== segment $(date -u +%FT%TZ)" >> "$OUT/train.log"; else : > "$OUT/train.log"; fi
  local from=$(( $(wc -l < "$OUT/train.log") + 1 ))   # this segment's lines (run20 appends one segment after another)
  tail -n +"$from" -F "$OUT/train.log" > "$OUT/train_segment.log" 2>/dev/null &
  local tl=$!
  python3 "$MILES/train.py" "$@" >> "$OUT/train.log" 2>&1 &
  local tp=$!
  (
    until grep -q 'POST /v1/chat/completions' "$OUT/train_segment.log" 2>/dev/null; do kill -0 $tp 2>/dev/null || exit; sleep 15; done
    python3 -c "import sys; from flint.miles_agent import startup_probe as p; sys.exit(p('$OUT', '$OUT/train_segment.log', '$FLINT'))" \
      > "$OUT/probe_start.log" 2>&1 || { echo "[s5] startup probe failed: $(tail -1 "$OUT/probe_start.log")"; kill $tp; }
  ) &
  wait $tp
  local rc=$?
  kill $tl 2>/dev/null
  return $rc
}

t0=$SECONDS
case "$MODE" in
  smoke)   # 2 steps, no actor save: rollout, log_rollout (check, probe), training step, weight sync, rollout 2, probe
    train "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" --colocate --num-rollout 2 ;;
  test2) train "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" "${RL[@]}" --num-rollout 2 ;;
  run20)
    echo "[s5] run20 starts from checkpoint: $(cat "$OUT/actor/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)"
    APPEND=1 train "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" "${RUN20[@]}" --num-rollout 20 ;;
esac
rc=$?
echo "[s5] $(date -u +%FT%TZ) train.py exit $rc after $((SECONDS - t0)) s; $(tail -1 "$OUT/probe_start.log" 2>/dev/null)"
python3 "$REPO/rl/steps.py" "$OUT"
exit $rc
