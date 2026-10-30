#!/bin/bash
#SBATCH --job-name=flint-engine
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/tmp/flint_s5/logs/%x-%j.out
#FETCHABLE
# One external Flint SGLang engine (1 x TP8, run20's serving flags: rl/engine_inner.sh) for an RL trainer in another
# job (rl/rl_job_ext.sh), stage 0 of tmp/elastic_rl/design.md. Usage: engine_job.sh RUN, where RUN is the run
# directory both jobs are given. The engine binds the node's eno8303 address, so the trainer's miles (router, weight
# push over NCCL) reaches it from its own node.
# Registry (rl/registry.py): RUN/engines/<job>.json, state "loading" from the start, "serving" once /health_generate,
# /server_info (8 GPUs) and a one-token /generate pass; its heartbeat is rewritten every 30 s; removed at exit.
# The job ends:
#   exit 0   RUN/STOP exists (the trainer ended the run);
#   exit 3   RUN/STOP exists at the start (the run is over; use a new RUN or delete it);
#   exit 4   the trainer's heartbeat (RUN/trainer.json), once seen, is older than 15 min (the trainer died);
#   exit 5   no trainer heartbeat 20 min after the engine began serving (the pair never met; it writes RUN/STOP so a
#            trainer that starts later fails at once instead of idling);
#   exit 1   the engine failed (launch, health, the server process ended, /health failing for 10 min);
#   exit 99  5 min before the segment's end (SLURM_JOB_END_TIME): wq requeues it when submitted --partial;
#   exit 143 a signal.
# Bounds: timeout -s KILL around the container, the GPU watch (rl/gpu_watch.sh: an engine idles while the trainer
# trains, ~5 min per step, and while it waits for the trainer, so 45 min with weights loaded; 10 min with nothing
# loaded), and cleanup of only the process groups this job started (serving/check/procs.sh).
set -uo pipefail
RUN=${1:?usage: engine_job.sh RUN}
LIMIT=${EXT_LIMIT_S:-10800}
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
[ -n "$JID" ] || { echo "no job id"; exit 1; }
REPO=${EXT_REPO:-/scratch/gpfs/GROUP/USER/project/mas-rl-elastic}
MILES=${EXT_MILES:-/scratch/gpfs/GROUP/USER/project/miles-rollout}
SIF=${EXT_SIF:-/scratch/gpfs/GROUP/USER/project/miles-q38-build/miles-v0.1.3.sif}
FLINT=${EXT_FLINT:-/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint}
GRACE=${EXT_TRAINER_GRACE_S:-1200} STALE=${EXT_TRAINER_STALE_S:-900} BEAT=${EXT_BEAT_S:-30}
REG=(timeout 60 python3 "$REPO/rl/registry.py")   # a scratch stall cannot hold the job
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
LOG=$RUN/engine_logs/$JID.log   # the server's own log
[ -e "$RUN/STOP" ] && { echo "[engine] $RUN/STOP exists: the run is over ($(cat "$RUN/STOP"))"; exit 3; }
mkdir -p "$RUN/engines" "$RUN/engine_logs" "$TMPDIR/tmp"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # the trainer and the engine talk over the cluster network
source "$REPO/serving/check/procs.sh"
source "$REPO/rl/gpu_watch.sh"
WD= SMI=
trap '"${REG[@]}" deregister "$RUN" "$JID"; stop "${STARTED[@]}"; kill $WD $SMI 2>/dev/null; rm -rf -- "/scratch/gpfs/GROUP/USER/tmp/job/$JID"' EXIT
trap 'exit 143' TERM INT
IP=$(ip -4 -o addr show eno8303 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
[ -n "$IP" ] || IP=$(hostname -i | tr ' ' '\n' | grep -E '^172\.' | head -1)
[ -n "$IP" ] || { echo "[engine] no routable address (eno8303)"; exit 1; }
# 41 free ports: nccl BASE, engine-info bootstrap BASE+1, dist-init BASE+8.., the server BASE+40; below Linux's
# ephemeral range, where the trainer's NCCL rendezvous port comes from
BASE=$(python3 - <<'PY'
import random, socket
for _ in range(500):
    b, socks = random.randrange(20000, 29900), []
    try:
        for p in range(b, b + 41):
            s = socket.socket(); socks.append(s); s.bind(("0.0.0.0", p))
        print(b); break
    except OSError:
        pass
    finally:
        for s in socks: s.close()
PY
)
[ -n "$BASE" ] || { echo "[engine] no free ports"; exit 1; }
PORT=$((BASE + 40)) URL=http://$IP:$((BASE + 40))
IFS=, read -r -a DEV <<< "${CUDA_VISIBLE_DEVICES:-}"
[ "${#DEV[@]}" -eq 8 ] || { echo "[engine] needs 8 GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"; exit 1; }
END=$(( $(date +%s) + LIMIT ))
[ -n "${SLURM_JOB_END_TIME:-}" ] && [ "$SLURM_JOB_END_TIME" -lt "$END" ] && END=$SLURM_JOB_END_TIME
T0=$(date +%s)
echo "[engine] $(date -u +%FT%TZ) run=$RUN node=$(hostname) job=$JID url=$URL gpus=$CUDA_VISIBLE_DEVICES end=$(date -d @"$END" +%FT%T)"
echo "mas-rl $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain -uno | wc -l)" \
  "miles $(git -C "$MILES" rev-parse --short HEAD) dirty=$(git -C "$MILES" status --porcelain -uno | wc -l)"
PKG=$TMPDIR/flint_pkg
timeout 900 bash "$REPO/serving/build_flint_sglang.sh" "$PKG" || { echo "[engine] plugin build failed"; exit 1; }
# the record (only this shell writes it, so a heartbeat never races a state change)
"${REG[@]}" register "$RUN" "$JID" "$(hostname)" "$URL,8,$(tr , : <<< "$CUDA_VISIBLE_DEVICES")" --state loading --log "$LOG" >/dev/null || exit 1
GPU_IDLE_MIN=45 gpu_watch $$ "$RUN/engine_logs/$JID.gpu.log" "grep -q '\"serving\"' '$RUN/engines/$JID.json' 2>/dev/null" 40 &
WD=$!
timeout "$LIMIT" nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits -l 10 \
  > "$RUN/engine_logs/$JID.gpu_mem.csv" 2>/dev/null &
SMI=$!
start timeout -s KILL "$LIMIT" apptainer exec --pid --nv --contain --cleanenv --writable-tmpfs \
  --bind /scratch/gpfs/GROUP --bind "$TMPDIR/tmp":/tmp --bind /dev/shm \
  --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env IP="$IP" --env PORT="$PORT" --env BASE="$BASE" --env PKG="$PKG" --env TMPDIR="$TMPDIR" --env REPO="$REPO" \
  --env MILES="$MILES" --env FLINT="$FLINT" "$SIF" bash "$REPO/rl/engine_inner.sh" serve > "$LOG" 2>&1
SRV=${STARTED[0]}

http() {   # http PATH [JSON BODY [TIMEOUT]] -> the HTTP status (000 when there is no answer)
  local a=(-s --noproxy '*' -m "${3:-10}" -o /dev/null -w '%{http_code}')
  [ -n "${2:-}" ] && a+=(-H 'Content-Type: application/json' -d "$2")
  curl "${a[@]}" "$URL$1"
}
# load: the server answers /health_generate, reports 8 GPUs, and generates one token (up to 40 min: 80 GB of weights)
until [ "$(http /health_generate)" = 200 ]; do
  kill -0 "$SRV" 2>/dev/null || { echo "[engine] the server ended while loading: $(tail -3 "$LOG")"; exit 1; }
  [ -e "$RUN/STOP" ] && { echo "[engine] $RUN/STOP appeared while loading"; exit 0; }
  [ $(( $(date +%s) - T0 )) -gt 2400 ] && { echo "[engine] not healthy after 40 min"; exit 1; }
  "${REG[@]}" beat "$RUN/engines/$JID.json"
  sleep 10
done
ws=$(curl -s --noproxy '*' -m 30 "$URL/server_info" | timeout 30 python3 -c 'import json,sys; print(json.load(sys.stdin)["internal_states"][0]["world_size"])' 2>/dev/null)
[ "$ws" = 8 ] || { echo "[engine] /server_info world_size '$ws', expected 8"; exit 1; }
code=$(http /generate '{"input_ids": [1, 2, 3, 4], "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}}' 300)
[ "$code" = 200 ] || { echo "[engine] one-token /generate answered $code"; exit 1; }
"${REG[@]}" register "$RUN" "$JID" "$(hostname)" "$URL,8,$(tr , : <<< "$CUDA_VISIBLE_DEVICES")" --state serving --log "$LOG" >/dev/null || exit 1
SINCE=$(date +%s)
echo "[engine] $(date -u +%FT%TZ) serving after $((SINCE - T0)) s; waiting for the trainer (at most $GRACE s)"

# serve: heartbeat every BEAT s; leave on STOP, a dead trainer, no trainer, a dead or unhealthy server, or the end
bad=0 unk=0
while sleep "$BEAT" & wait $!; do   # a background sleep, so a TERM is handled at once
  "${REG[@]}" beat "$RUN/engines/$JID.json" || "${REG[@]}" register "$RUN" "$JID" "$(hostname)" \
    "$URL,8,$(tr , : <<< "$CUDA_VISIBLE_DEVICES")" --state serving --log "$LOG" >/dev/null
  kill -0 "$SRV" 2>/dev/null || { echo "[engine] $(date -u +%FT%TZ) the server ended: $(tail -3 "$LOG")"; exit 1; }
  if [ "$(http /health 2>/dev/null)" = 200 ]; then bad=0; else bad=$((bad + 1)); fi
  [ $((bad * BEAT)) -ge 600 ] && { echo "[engine] $(date -u +%FT%TZ) /health failing for $((bad * BEAT)) s"; exit 1; }
  case $("${REG[@]}" engine-check "$RUN" "$SINCE" --stale "$STALE" --grace "$GRACE" --seen "$TMPDIR/trainer_seen") in
    serve) unk=0 ;;
    stop) echo "[engine] $(date -u +%FT%TZ) $RUN/STOP: $(cat "$RUN/STOP" 2>/dev/null); done"; exit 0 ;;
    gone) echo "[engine] $(date -u +%FT%TZ) the trainer's heartbeat is older than $STALE s: leaving"; exit 4 ;;
    alone) echo "[engine] $(date -u +%FT%TZ) no trainer heartbeat $GRACE s after serving: writing STOP, leaving"
           echo "engine $JID: no trainer within $GRACE s of serving ($(date -u +%FT%TZ))" > "$RUN/STOP"; exit 5 ;;
    *) unk=$((unk + 1))   # the registry gave no answer (a scratch stall, the 60-s timeout): at most GRACE s of that
       [ $((unk * BEAT)) -ge "$GRACE" ] && { echo "[engine] $(date -u +%FT%TZ) no registry answer for $((unk * BEAT)) s"; exit 1; } ;;
  esac
  [ "$(date +%s)" -gt $((END - 300)) ] && { echo "[engine] $(date -u +%FT%TZ) 5 min before the segment's end: leaving"; exit 99; }
done
