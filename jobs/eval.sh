#!/bin/bash
#SBATCH --job-name=flint-eval
#SBATCH --gres=gpu:8
#SBATCH --mem=480G
#SBATCH --time=23:00:00
#SBATCH --comment=fetchable-partial
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
#FETCHABLE
# mas-rl: one eval of the long game on 8 GPUs (wq --partial). Usage: eval.sh SPLIT TOKENS EPISODES [key=value ...];
# SPLIT is val40 or train67, TOKENS the budget of each agent. Options (the script is self-contained: options are
# arguments, never variables from the submit shell):
#   agents=N   agents per episode (default 1; a team is 4: own repositories, must post)
#   conc=N     episodes at once (default 32 = 4 servers x 8: one TP2 server's KV pool holds about 8 trimmed contexts);
#              a team runs N agents per episode, so pass conc with N x conc <= 32
#   narrow=0|1 (default 1) when a segment starts with fewer than 4 episodes left, serve them on 2 servers at TP4 (2-3
#              left) or 1 at TP8 (1 left), so every server has an episode
#   idle=MIN   watchdog: when a server has had no batch for MIN minutes and episodes are left, stop the driver (default
#              60, under the 90-min idle-GPU host cancel; 0 = off). The episodes resume from their checkpoints.
#   rounds=N   MA_LG_MAX_ROUNDS (default 20000)
#   repo=PATH  code tree (default the clean worktree of main, miles-q38-build-clean); it must be clean
# Results: runs/<SPLIT>_t<TOKENS>_e<EPISODES>_budget_low0.75[_n<N>_own_mustpost] (the names of the same game before
# the cleanup of 2026-10-28, so a run resumes there). Exit: 0 when every episode is scored or given up; 99 (wq
# requeues) when episodes are left and the driver can resume them; 1 on a setup failure or a run without progress.
set -euo pipefail
# The servers bind 127.0.0.1, and SGLang's own warmup request goes there: a proxy in the environment (module
# proxy/default) answers it with 403 and the server exits (smoke 16159073). Nothing here needs the internet.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
P=/scratch/gpfs/GROUP/USER/project
FE=$P/flint-eval
MODEL=$P/labs-molt/_workspace/models/flint
SIF=$FE/flint-sglang.sif
PYSIF=$P/miles-q38-build/miles.sif   # python with aiohttp and yaml for the driver and the judge
SPLIT=${1:-} TOKENS=${2:-} EPISODES=${3:-}
[ $# -ge 3 ] || { echo "usage: eval.sh SPLIT TOKENS EPISODES [key=value ...]"; exit 1; }
shift 3
case "$SPLIT" in val40|train67) ;; *) echo "bad split $SPLIT"; exit 1;; esac
CONC=32 NARROW=1 IDLE_MIN=60 AGENTS=1 ROUNDS=20000 REPO=$P/miles-q38-build-clean
for a in "$@"; do
  case "$a" in
    agents=*) AGENTS=${a#*=};; conc=*) CONC=${a#*=};; narrow=*) NARROW=${a#*=};; idle=*) IDLE_MIN=${a#*=};;
    rounds=*) ROUNDS=${a#*=};; repo=*) REPO=${a#*=};;
    *) echo "unknown option $a"; exit 1;;
  esac
done
[[ "$TOKENS$EPISODES$CONC$IDLE_MIN$ROUNDS" =~ ^[0-9]+$ && "$NARROW" =~ ^[01]$ && "$AGENTS" =~ ^[1-8]$ ]] \
  || { echo "bad option value"; exit 1; }
JID=${FETCHED_JOB_ID:-${SLURM_JOB_ID:-}}
[ -n "$JID" ] || { echo "no job id"; exit 1; }
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
mkdir -p "$TMPDIR"
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "/scratch/gpfs/GROUP/USER/tmp/job/$JID" || true' EXIT
OUT=$FE/runs/${SPLIT}_t${TOKENS}_e${EPISODES}_budget_low0.75
[ "$AGENTS" = 1 ] || OUT=${OUT}_n${AGENTS}_own_mustpost
DATA=$REPO/data/fcs11_${SPLIT}_agent.jsonl
SEG=$(date +%m%d-%H%M%S)   # one stamp per segment (wq run): each segment keeps its own server logs
mkdir -p "$OUT/servers" "$FE/triton_cache"
echo "segment $SEG host $(hostname) job $JID slurm ${SLURM_JOB_ID:-none}"
echo "split $SPLIT tokens $TOKENS episodes $EPISODES agents $AGENTS conc $CONC narrow $NARROW idle ${IDLE_MIN}m rounds $ROUNDS repo $REPO"

# Provenance: record the commit of every segment; the tree must be clean
SHA=$(git -C "$REPO" rev-parse HEAD)
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { echo "tree $REPO is dirty: refusing"; exit 1; }
echo "commit $SHA"
PREV=$( (tail -1 "$OUT/segments.jsonl" 2>/dev/null || true) | sed -n 's/.*"commit": "\([0-9a-f]*\)".*/\1/p')
[ -z "$PREV" ] || [ "$PREV" = "$SHA" ] || echo "WARNING: resuming $OUT across commits $PREV -> $SHA"
printf '{"segment": "%s", "time": "%s", "job": "%s", "host": "%s", "commit": "%s", "dirty": false, "args": "%s"}\n' \
  "$SEG" "$(date -Is)" "$JID" "$(hostname)" "$SHA" "$SPLIT $TOKENS $EPISODES $*" >> "$OUT/segments.jsonl"

# (problem, episode) pairs without a score, by the rule of run_eval.py; "list" prints them to stderr.
remaining() {
  python3 -I - "$DATA" "$EPISODES" "$OUT/results.jsonl" "${1:-}" <<'PY'
import json, os, sys
data, E, res, verbose = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
pids = [json.loads(l)["metadata"]["pid"] for l in open(data) if l.strip()]
done = set()
if os.path.exists(res):
    for l in open(res):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get("S") is not None:
            done.add((r["pid"], r["episode"]))
miss = [(p, e) for e in range(E) for p in pids if (p, e) not in done]
if verbose and miss:
    print("missing", " ".join(f"{p}/ep{e}" for p, e in miss), file=sys.stderr)
print(len(miss))
PY
}

REM0=$(remaining list)
echo "remaining episodes $REM0"
[ "$REM0" -gt 0 ] || { echo EVAL_DONE; exit 0; }   # nothing to run: start no server
NS=4
if [ "$NARROW" = 1 ]; then
  if [ "$REM0" -eq 1 ]; then NS=1; elif [ "$REM0" -lt 4 ]; then NS=2; fi
fi
TP=$((8 / NS))
echo "layout: $NS servers at TP$TP for $REM0 episodes"
[ "$CONC" -ge "$NS" ] || { echo "conc $CONC < $NS servers: a server would idle from the start"; exit 1; }
if [ "$REM0" -lt "$NS" ] && [ "$IDLE_MIN" -gt 0 ]; then
  echo "EVAL_NEEDS_NARROW: $REM0 episodes left < $NS servers (narrow=0): resubmit with narrow=1"
  exit 1
fi

# Hard gate: the judge must run sandboxed (Landlock ABI >= 6) on this node
APPTAINERENV_PYTHONPATH=$REPO apptainer exec --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$PYSIF" python3 -c \
  "import sys, flint.judge as J; print('LANDLOCK_ABI', J.LANDLOCK_ABI); sys.exit(0 if J.LANDLOCK_ABI >= 6 else 1)" \
  || { echo "Landlock ABI < 6 on $(hostname): refusing to run unsandboxed"; exit 1; }

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
else mapfile -t G < <(nvidia-smi --query-gpu=index --format=csv,noheader); fi
[ "${#G[@]}" -eq 8 ] || { echo "need 8 GPUs, got ${#G[@]}: ${G[*]}"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Ports: a per-job START keeps co-hosted jobs apart (SGLang binds its HTTP port only ~3 min after launch, so a probe
# cannot see a sibling still loading); the probe skips ports any socket on this node holds and ports this job took.
JID_NUM=$(echo "$JID" | tr -dc 0-9)
START=$((20000 + (10#$JID_NUM % 500) * 20))
TAKEN=" "
port_busy() {
  if command -v ss >/dev/null 2>&1; then ss -Htan "sport = :$1" | grep -q .
  else (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; fi
}
pick_port() {   # sets PICKED (no subshell, so TAKEN persists)
  local p=$1
  while [ "$p" -lt 32768 ]; do
    if [[ "$TAKEN" != *" $p "* ]] && ! port_busy "$p"; then TAKEN+="$p "; PICKED=$p; return 0; fi
    p=$((p + 1))
  done
  echo "no free port from $1 below 32768"; exit 1
}
export APPTAINERENV_TMPDIR=$TMPDIR APPTAINERENV_XDG_CACHE_HOME=$TMPDIR/cache
PIDS=() URLS=() LOGS=()
for ((r = 0; r < NS; r++)); do
  pick_port "$START"; PORT=$PICKED
  pick_port $((START + 10)); NCCL=$PICKED
  GPUS=$(IFS=,; echo "${G[*]:$((r * TP)):$TP}")
  LOG="$OUT/servers/r$r-$JID-$SEG.log"; LOGS+=("$LOG")
  echo "server $r: GPUs $GPUS port $PORT nccl $NCCL log $LOG"
  APPTAINERENV_CUDA_VISIBLE_DEVICES=$GPUS APPTAINERENV_MODEL=$MODEL APPTAINERENV_PORT=$PORT \
  APPTAINERENV_FLINT_TRITON_CACHE_BASE=$FE/triton_cache \
    apptainer exec --nv --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$SIF" \
    bash "$MODEL/sglang/serve_h200.sh" --tp-size "$TP" --mem-fraction-static 0.85 --nccl-port "$NCCL" \
    --host 127.0.0.1 --enable-cache-report > "$LOG" 2>&1 &
  PIDS+=($!) URLS+=("http://127.0.0.1:$PORT")
done
# A server that died on a taken port is worth another run (exit 99); any other death is deterministic (exit 1).
srv_dead() {
  echo "server $1 $2"; tail -40 "${LOGS[$1]}"
  grep -qiE "address already in use|could not bind" "${LOGS[$1]}" && exit 99
  exit 1
}
for ((r = 0; r < NS; r++)); do
  for _ in $(seq 1 120); do
    curl -sf "${URLS[$r]}/health_generate" >/dev/null 2>&1 && break
    kill -0 "${PIDS[$r]}" 2>/dev/null || srv_dead "$r" died
    sleep 15
  done
  curl -sf "${URLS[$r]}/health_generate" >/dev/null || srv_dead "$r" "not ready"
done
echo "$NS servers up: ${URLS[*]}"

# Watchdog (every 60 s): all servers dead, or one server with no batch for IDLE_MIN minutes while episodes are left,
# stops the driver; the episodes in flight resume from their checkpoints in the next segment.
STOPF=$TMPDIR/watchdog_stop
watchdog() {
  set +e
  local t0 now last ts alive r rem
  t0=$(date +%s)
  while sleep 60; do
    kill -0 "$DRV" 2>/dev/null || return 0
    alive=0
    for r in "${PIDS[@]}"; do kill -0 "$r" 2>/dev/null && alive=$((alive + 1)); done
    if [ "$alive" -eq 0 ]; then echo "WATCHDOG: all $NS servers dead"; echo dead > "$STOPF"; kill "$DRV"; return 0; fi
    [ "$IDLE_MIN" -gt 0 ] || continue
    now=$(date +%s)
    [ $((now - t0)) -ge $((IDLE_MIN * 60)) ] || continue
    for ((r = 0; r < NS; r++)); do
      last=$(tac "${LOGS[$r]}" | grep -m1 -oE '^\[[0-9-]+ [0-9:]+ TP0\] (Prefill|Decode) batch' | cut -c2-20)
      ts=$t0; [ -n "$last" ] && ts=$(date -d "$last" +%s); [ "$ts" -ge "$t0" ] || ts=$t0
      [ $((now - ts)) -ge $((IDLE_MIN * 60)) ] || continue
      rem=$(remaining)
      if [ -n "$rem" ] && [ "$rem" -gt 0 ]; then
        echo "WATCHDOG: server $r had no batch for $(((now - ts) / 60)) min, $rem episodes left: stopping the driver"
        echo idle > "$STOPF"; kill "$DRV"; return 0
      fi
    done
  done
}

cd "$REPO"
APPTAINERENV_PYTHONPATH=$REPO APPTAINERENV_MA_LG_N=$AGENTS APPTAINERENV_MA_LG_TOKENS=$TOKENS \
APPTAINERENV_MA_LG_MAX_ROUNDS=$ROUNDS APPTAINERENV_FCS_RM_CONCURRENCY=32 \
APPTAINERENV_FCS_JUDGE_TMP=$TMPDIR APPTAINERENV_FCS_JUDGE_CACHE=$TMPDIR/fcs_judge_cache \
  apptainer exec --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$PYSIF" \
  python3 run_eval.py --base-url "$(IFS=,; echo "${URLS[*]}")" --model flint \
  --data "data/fcs11_${SPLIT}_agent.jsonl" --out "$OUT" --episodes "$EPISODES" --concurrency "$CONC" &
DRV=$!
watchdog &
WD=$!
rc=0
wait "$DRV" || rc=$?
kill "$WD" 2>/dev/null || true
echo "driver exit $rc"
for ((r = 0; r < NS; r++)); do echo "r$r $(grep -h 'gen throughput' "${LOGS[$r]}" | tail -1 || true)"; done
REM=$(remaining list)
echo "remaining episodes $REM (was $REM0)"
if [ "$REM" -eq 0 ]; then echo EVAL_DONE; exit 0; fi
[ ! -s "$STOPF" ] || echo "stopped by the watchdog: $(cat "$STOPF")"
if [ ! -s "$STOPF" ]; then
  case "$rc" in
    0) echo "EVAL_DONE_GIVEN_UP $REM: the driver gave these up after their tries"; exit 0;;
    3|99) echo "EVAL_PARTIAL $REM: the driver can resume them (exit $rc): requeue (exit 99)"; exit 99;;
  esac
fi
if [ -s "$STOPF" ] && ls "$OUT"/state/*.pkl >/dev/null 2>&1; then
  echo "EVAL_PARTIAL $REM: checkpoints saved before the stop: requeue (exit 99)"; exit 99
fi
if [ "$REM" -lt "$REM0" ]; then echo "EVAL_PARTIAL $REM: requeue (exit 99)"; exit 99; fi
echo "EVAL_PARTIAL $REM: this run finished no episode (driver exit $rc): failing instead of requeueing"
exit 1
