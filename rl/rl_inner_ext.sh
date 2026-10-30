#!/bin/bash
# Inside miles-v0.1.3.sif (rl/rl_job_ext.sh): as rl/rl_inner.sh, with the rollout on engines in other jobs. A
# job-private Ray head bound to the node's routable address (miles' weight-push master address is Ray's node IP; the
# engines must reach it), then miles' train.py with --rollout-external-engine-addrs and NCCL broadcast. Usage:
# rl_inner_ext.sh smoke_ext|args. Env: JID, OUT, PKG, TMPDIR, IP, REPO, MILES, FLINT, ADDRS (host:port ...), ENGINE_TP,
# ENGINE_NGPUS. Mode args: the argument check alone, off GPU.
set -uo pipefail
: "${JID:?}" "${OUT:?}" "${PKG:?}" "${TMPDIR:?}" "${IP:?}" "${REPO:?}" "${MILES:?}" "${FLINT:?}" "${ADDRS:?}" \
  "${ENGINE_TP:?}" "${ENGINE_NGPUS:?}"
MODE=$1
export HOME=$TMPDIR/home TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export SGLANG_CACHE_DIR=$TMPDIR/sglang FLASHINFER_WORKSPACE_BASE=$TMPDIR/flashinfer
mkdir -p "$HOME"
export PYTHONPATH=$PKG:$REPO:$MILES PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
HOSTS=$(tr ' ' '\n' <<< "$ADDRS" | cut -d: -f1 | sort -u | paste -sd, -)
export RAY_USAGE_STATS_ENABLED=0 no_proxy=127.0.0.1,localhost,$IP,$HOSTS MASTER_ADDR=$IP
export SGLANG_EXTERNAL_MODEL_PACKAGE=flint_sglang FLASHINFER_USE_CUDA_NORM=1 SGLANG_RETURN_ORIGINAL_LOGPROB=0
# the weight push to engines in other apptainer instances (miles also sets NCCL_CUMEM_ENABLE=0 in its train actors)
export NCCL_CUMEM_ENABLE=0 NCCL_SOCKET_IFNAME=eno8303
# the game, as rl/rl_inner.sh (128k tokens, RL's context cap, the 45-min wall cap); smoke_ext as smoke (4k, 20k chars)
export MA_LG_N=1 MA_LG_TOKENS=128000 MA_AG_TOOL_STRICT=0 FCS_PROC=/host/proc
export MA_LG_CTX_CHARS=110000 MA_AG_TURN=16384 MA_RL_MAX_SAMPLE_TOKENS=70000
export MA_LG_WALL_MIN=45
export MA_RL_OUT=$OUT MA_AG_TRACE_DIR=$OUT/traces_live
case "$MODE" in smoke*) export MA_LG_TOKENS=4000 MA_LG_CTX_CHARS=20000;; esac
python3 -c "import json, os; json.dump({k: os.environ[k] for k in ('MA_LG_N', 'MA_LG_TOKENS', 'MA_LG_CTX_CHARS', 'MA_AG_TURN', 'MA_LG_WALL_MIN')}, open('$OUT/game_settings.json', 'w'))"
rm -rf "$OUT/settings_seen"   # this segment's agents write theirs (the startup probe's trigger and check)
ulimit -n "$(ulimit -Hn)"
NG=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
JNUM=${JID//[!0-9]/}
P=$((30000 + (JNUM % 1000) * 16))

ENGINE_MEM=0.85   # the engines' --mem-fraction-static (rl/engine_inner.sh), so the validators see the served value
source "$REPO/rl/ext_args.sh"   # TRAIN, SERVE
# shellcheck disable=SC2206
EXT=(--rollout-external-engine-addrs $ADDRS --update-weight-transfer-mode broadcast --update-weights-timeout 900)
GAME=(--prompt-data "$OUT/data_p0.jsonl" --input-key prompt --label-key label --metadata-key metadata
  --rollout-temperature 1.0 --rollout-top-p 0.95 --rollout-top-k 20
  --rollout-max-response-len 16384 --max-seq-len 262144
  --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
  --custom-agent-function-path flint.miles_agent.run --use-session-server v2 --tito-model flint
  --session-message-matcher flint.miles_agent.message_matches --pause-generation-mode in_place
  --rollout-sample-filter-path flint.miles_agent.drop_failed_groups
  --custom-rollout-log-function-path flint.miles_agent.log_rollout
  --save-debug-rollout-data "$OUT/r{rollout_id}/rollout.pt")
# smoke_ext: 2 steps; the checkpoint (actor, optimizer, data position) at the second, from where a later run resumes
SMOKE=(--save "$OUT/actor" --load "$OUT/actor" --save-interval 2 --save-trigger-sentinel "$OUT/save_now" --num-rollout 2)

if [ "$MODE" = args ]; then   # off-GPU: miles parses and checks the arguments (rl/args_check.py)
  python3 "$REPO/rl/args_check.py" "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" "${EXT[@]}" "${SMOKE[@]}"
  exit
fi

echo "[ext] $(date -u +%FT%TZ) inner $MODE: $NG GPUs, ray $IP:$P, engines $ADDRS"
timeout 300 ray start --head --node-ip-address "$IP" --num-gpus "$NG" --port=$P --dashboard-port=$((P + 1)) \
  --dashboard-agent-listen-port=$((P + 2)) --dashboard-agent-grpc-port=$((P + 3)) --metrics-export-port=$((P + 4)) \
  --ray-client-server-port=$((P + 5)) --runtime-env-agent-port=$((P + 6)) --node-manager-port=$((P + 7)) \
  --object-manager-port=$((P + 8)) --include-dashboard=false --disable-usage-stats --temp-dir=/tmp/ray >/dev/null \
  || { echo "[ext] ray start failed"; exit 1; }
export RAY_ADDRESS=$IP:$P
trap 'ray stop --force >/dev/null 2>&1' EXIT

# train.py; the startup probe (flint.miles_agent.startup_probe: the agents' game settings, the served log-probs of the
# initial weights, through miles' router) runs once the first agent has written its settings: the engines' request
# log is in the engine job, not here, so rl_inner.sh's trigger (the first /v1/chat/completions line) never appears
train() {
  : > "$OUT/train.log"
  tail -n +1 -F "$OUT/train.log" > "$OUT/train_segment.log" 2>/dev/null &
  local tl=$!
  python3 "$MILES/train.py" "$@" >> "$OUT/train.log" 2>&1 &
  local tp=$!
  (
    until ls "$OUT/settings_seen/"*.json >/dev/null 2>&1; do kill -0 $tp 2>/dev/null || exit; sleep 15; done
    python3 -c "import sys; from flint.miles_agent import startup_probe as p; sys.exit(p('$OUT', '$OUT/train_segment.log', '$FLINT'))" \
      > "$OUT/probe_start.log" 2>&1 || { echo "[ext] startup probe failed: $(tail -1 "$OUT/probe_start.log")"; kill $tp; }
  ) &
  wait $tp
  local rc=$?
  kill $tl 2>/dev/null
  return $rc
}

t0=$SECONDS
case "$MODE" in
  smoke_ext) train "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" "${EXT[@]}" "${SMOKE[@]}" ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
rc=$?
echo "[ext] $(date -u +%FT%TZ) train.py exit $rc after $((SECONDS - t0)) s; $(tail -1 "$OUT/probe_start.log" 2>/dev/null)"
[ $rc = 0 ] && { [ -e "$OUT/actor/latest_checkpointed_iteration.txt" ] || { echo "[ext] no checkpoint after the smoke"; rc=1; }; }
python3 "$REPO/rl/steps.py" "$OUT"
exit $rc
