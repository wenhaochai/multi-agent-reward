#!/bin/bash
# Inside miles-v0.1.3.sif with a PID namespace of its own (rl/rollout_job.sh). Each run gets a fresh job-private Ray
# head on job-derived ports, started with the run's environment (Ray's workers, among them the agent processes, take
# their environment from it), then miles' train.py rollout only and the check of its samples (rl/inspect_rollout.py).
# rl/site/sitecustomize.py loads SGLang's plugins (the Flint plugin) in every process that imports SGLang's arguments. Usage: rollout_inner.sh probe|full; env JID, OUT, PKG (the built Flint plugin), TMPDIR.
set -uo pipefail
: "${JID:?}" "${OUT:?}" "${PKG:?}" "${TMPDIR:?}"
MODE=$1
REPO=/scratch/gpfs/GROUP/USER/project/mas-rl-rollout
FLINT=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint
export HOME=$TMPDIR/home TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export SGLANG_CACHE_DIR=$TMPDIR/sglang FLASHINFER_WORKSPACE_BASE=$TMPDIR/flashinfer
mkdir -p "$HOME"
export PYTHONPATH=$PKG:$REPO/rl/site:$REPO:/root/miles PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1   # the container's writable layer is 64 MB
export RAY_USAGE_STATS_ENABLED=0 no_proxy=127.0.0.1,localhost MASTER_ADDR=127.0.0.1
export SGLANG_EXTERNAL_MODEL_PACKAGE=flint_sglang FLASHINFER_USE_CUDA_NORM=1 SGLANG_RETURN_ORIGINAL_LOGPROB=0
# the game: one Flint (budget end with forced restart, its defaults); the bash tool without strict (see flint/chat.py)
export MA_LG_N=1 MA_AG_TOOL_STRICT=0
NG=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
JNUM=${JID//[!0-9]/}
P=$((30000 + (JNUM % 1000) * 16))
echo "[s5a] $(date -u +%FT%TZ) inner $MODE: $NG GPUs, ray port $P"
python3 -c "import torch, sglang, transformers, miles; print('torch', torch.__version__, 'sglang', sglang.__version__, 'transformers', transformers.__version__)"

ray_up() {
  ray start --head --node-ip-address 127.0.0.1 --num-gpus "$NG" --port=$P --dashboard-port=$((P + 1)) \
    --dashboard-agent-listen-port=$((P + 2)) --dashboard-agent-grpc-port=$((P + 3)) --metrics-export-port=$((P + 4)) \
    --ray-client-server-port=$((P + 5)) --runtime-env-agent-port=$((P + 6)) --node-manager-port=$((P + 7)) \
    --object-manager-port=$((P + 8)) --include-dashboard=false --disable-usage-stats --temp-dir=/tmp/ray >/dev/null \
    && export RAY_ADDRESS=127.0.0.1:$P
}
ray_down() { ray stop --force >/dev/null 2>&1; }   # this PID namespace's Ray only

# Flint's serving flags (the kit's serve_h200.sh, the eval servers: TP2, fa3, Triton MoE, fp32 LM head)
SERVE=(--sglang-attention-backend fa3 --sglang-moe-runner-backend triton --sglang-enable-fp32-lm-head
  --sglang-context-length 262144 --sglang-chunked-prefill-size 4096 --sglang-swa-full-tokens-ratio 0.4
  --sglang-mem-fraction-static 0.85 --sglang-grammar-backend flint)
# the long game as a miles agent on the RL test problem (flint/miles_agent.py), sampling as in the eval; the engines
# return each token's routed experts (miles sets SGLang's --enable-return-routed-experts with routing replay)
GAME=(--prompt-data "$OUT/data_p0.jsonl" --input-key prompt --label-key label --metadata-key metadata
  --num-rollout 1 --rollout-batch-size 1 --rollout-temperature 1.0 --rollout-top-p 0.95 --rollout-top-k 20
  --rollout-max-response-len 32768 --max-seq-len 262144
  --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
  --custom-agent-function-path flint.miles_agent.run --use-session-server v2 --tito-model default
  --session-message-matcher flint.miles_agent.message_matches
  --sglang-tool-call-parser flint --sglang-reasoning-parser flint
  --use-rollout-routing-replay --pause-generation-mode in_place)
# --use-rollout-routing-replay: each token's routed experts, for routing replay in training (R3). Not 'retract' (the
# default), with which the session server returns every turn's full R3 rows (~1.2 kB per token, the whole context)
# instead of the new ones; the rollouts here are synchronous, so the pause mode changes nothing else.

rollout() {   # rollout NAME TOKENS SAMPLES LIMIT_S [train.py flags]: one group of SAMPLES episodes, then the check
  local name=$1 tokens=$2 n=$3 limit=$4
  shift 4
  local d=$OUT/$name
  mkdir -p "$d/traces"
  export MA_LG_TOKENS=$tokens MA_AG_TRACE_DIR=$d/traces
  ray_up || { echo "[s5a] $name: ray start failed"; return 1; }
  local t0=$SECONDS
  timeout -s KILL "$limit" python3 /root/miles/train.py --train-backend fsdp --hf-checkpoint "$FLINT" \
    --debug-rollout-only --rollout-num-gpus "$NG" --rollout-num-gpus-per-engine 2 "${SERVE[@]}" "${GAME[@]}" \
    --n-samples-per-prompt "$n" --global-batch-size "$n" --save-debug-rollout-data "$d/rollout_{rollout_id}.pt" \
    --save-debug-trajectory-data "$d/trajectory_{rollout_id}.jsonl" "$@" > "$d/train.log" 2>&1
  local rc=$?
  echo "[s5a] $(date -u +%FT%TZ) $name: train.py exit $rc after $((SECONDS - t0)) s"
  ray_down
  [ -e "$d/rollout_0.pt" ] && python3 "$REPO/rl/inspect_rollout.py" "$d" 2>&1 | tee "$d/inspect.txt"
  return $rc
}

case "$MODE" in
  probe)
    # miles' GPU TITO check for Flint (scripts/tools/verify_session_tito_tokenizer.py; its rollout workers read these)
    export MILES_TITO_MODEL=default MILES_SESSION_VERIFY_METRICS_PATH=$OUT/tito_metrics.jsonl
    ray_up || exit 1
    timeout -s KILL 1500 python3 "$REPO/rl/tito_verify.py" --hf-checkpoint "$FLINT" --tito-model default \
      --sglang-reasoning-parser flint --sglang-tool-call-parser flint --rollout-num-gpus-per-engine 2 \
      --actor-num-gpus-per-node "$NG" --n-samples-per-prompt 4 -- "${SERVE[@]}" > "$OUT/tito_verify.log" 2>&1
    v=$?
    ray_down
    grep -E "Verdict|Template append roles" "$OUT/tito_verify.log"
    echo "[s5a] $(date -u +%FT%TZ) tito verify exit $v"
    unset MILES_TITO_MODEL MILES_SESSION_VERIFY_METRICS_PATH
    # the path end to end, short: 8 episodes of 8k tokens, the context cut to 20k characters so trims happen
    export MA_LG_CTX_CHARS=20000
    rollout probe 8000 8 1800 --rollout-top-logprobs-num 32
    r=$?
    unset MA_LG_CTX_CHARS
    [ $v -eq 0 ] && [ $r -eq 0 ]
    ;;
  full)
    rollout topk32 128000 8 4200 --rollout-top-logprobs-num 32
    a=$?
    rollout notopk 128000 8 4200
    b=$?
    [ $a -eq 0 ] && [ $b -eq 0 ]
    ;;
esac
