#!/bin/bash
#SBATCH --job-name=flint-elastic-engine
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=240G
#SBATCH --time=23:00:00
#SBATCH --comment=fetchable-partial
#SBATCH --output=/scratch/gpfs/GROUP/USER/tmp/flint_s5/logs/%x-%j.out
#FETCHABLE
# Extra rollout engines for an anchored elastic RL run (rl/rl_job.sh train67 ... elastic=1; tmp/elastic_rl/anchored.md):
# 8 / TP engines on this job's 8 GPUs (TP from the run's spec: 2 x TP4 with levers=tp4, else 1 x TP8), each serving
# exactly as the run's own colocated engines (rl/elastic_engine_argv.py). They register in RUN/engines (rl/registry.py);
# the trainer's miles adds each as a cell (flint/engine_registry.py) and routes to it from its next weight push on.
# Usage: wq submit --partial rl/elastic_engine.sh RUN (RUN: the trainer's run directory). Any number of these jobs may
# run, on any host, joining and leaving at any time; the run never waits for them.
# The job ends:
#   exit 0   RUN/STOP exists (the trainer ended the run);
#   exit 98  no live trainer (RUN/trainer.json older than 90 s) at the start, or the trainer's heartbeat went stale for
#            15 min while serving: wq puts the job back, not before 30 min, without counting the run;
#   exit 99  20 min before the host's end (SLURM_JOB_END_TIME): the record says "leaving", the run withdraws the
#            engines, they finish their running requests (5-20 min), then the job ends and wq requeues it;
#   exit 0   also after the same clean leave when RUN/engines/<job>.leave exists (an operator takes the engines away);
#   exit 1   an engine failed (launch, health, a server process ended, /health failing for 10 min);
#   exit 143 a signal.
# Bounds: timeout -s KILL around each container, the GPU watch (rl/gpu_watch.sh: a joining engine waits for the run's
# next weight push, up to one RL step, so 60 min at 0% with weights loaded; 10 min with nothing loaded), and cleanup of
# only the process groups this job started (serving/check/procs.sh).
set -uo pipefail
RUN=${1:?usage: elastic_engine.sh RUN}
LIMIT=${EL_LIMIT_S:-82800}
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
[ -n "$JID" ] || { echo "no job id"; exit 1; }
HERE=/scratch/gpfs/GROUP/USER/project/mas-rl-elastic2   # this job's own helpers (registry, GPU watch, procs)
SIF=/scratch/gpfs/GROUP/USER/project/miles-q38-build/miles-v0.1.3.sif
# timings (the EL_* variables only for tests/dryrun_elastic.sh; a wq job runs with the defaults)
BEAT=${EL_BEAT_S:-30} STALE=${EL_STALE_S:-900} DRAIN_MIN=${EL_DRAIN_MIN_S:-300} DRAIN_MAX=${EL_DRAIN_MAX_S:-1200}
LEAVE_BEFORE=${EL_LEAVE_BEFORE_S:-1200} MIN_LEFT=${EL_MIN_LEFT_S:-1800}
HERE=${EL_HERE:-$HERE} SIF=${EL_SIF:-$SIF}
REG=(timeout 60 python3 "$HERE/rl/registry.py")   # a scratch stall cannot hold the job
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
[ -e "$RUN/STOP" ] && { echo "[engine] $RUN/STOP exists: the run is over ($(cat "$RUN/STOP"))"; exit 0; }
trainer_age() { timeout 30 python3 -c "import json, sys, time; print(int(time.time() - json.load(open(sys.argv[1]))['heartbeat']))" "$RUN/trainer.json" 2>/dev/null || echo 999999; }
age=$(trainer_age)
[ "$age" -le 90 ] && [ -s "$RUN/engine_spec.json" ] \
  || { echo "[engine] no live trainer in $RUN (heartbeat age $age s, spec $(ls "$RUN/engine_spec.json" 2>/dev/null || echo missing)): back to the queue"; exit 98; }
mkdir -p "$RUN/engines" "$RUN/engine_logs" "$TMPDIR/tmp"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # the trainer and the engines talk over the cluster network
source "$HERE/serving/check/procs.sh"
source "$HERE/rl/gpu_watch.sh"
WD= SMI=
trap '"${REG[@]}" deregister "$RUN" "$JID"; stop "${STARTED[@]}"; kill $WD $SMI 2>/dev/null; rm -rf -- "/scratch/gpfs/GROUP/USER/tmp/job/$JID"' EXIT
trap 'exit 143' TERM INT
SPEC=$TMPDIR/engine_spec.json
cp "$RUN/engine_spec.json" "$SPEC"   # this job serves the spec it started with
read -r TP REPO MILES FLINT < <(python3 -c "import json, sys; s = json.load(open(sys.argv[1])); print(s['tp'], s['repo'], s['miles'], s['flint'])" "$SPEC")
IFS=, read -r -a DEV <<< "${CUDA_VISIBLE_DEVICES:-}"
[ "${#DEV[@]}" -eq 8 ] || { echo "[engine] needs 8 GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"; exit 1; }
case "$TP" in 4|8) ;; *) echo "[engine] the spec's TP is $TP; this job serves TP4 or TP8"; exit 1;; esac
N=$((8 / TP))
JNUM=${JID//[!0-9]/}; JNUM=$((10#${JNUM:-0} % 20))   # engine i of this job serves with the run's seed + 10 + 2 JNUM + i
IP=$(ip -4 -o addr show eno8303 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
[ -n "$IP" ] || IP=$(hostname -i | tr ' ' '\n' | grep -E '^172\.' | head -1)
[ -n "$IP" ] || { echo "[engine] no routable address (eno8303)"; exit 1; }
# 50 free ports per engine, below Linux's ephemeral range (the trainer's NCCL rendezvous port comes from there): engine
# i takes BASE + 50 i: nccl +0, engine-info bootstrap +1, dist-init +8.., the server +40
BASE=$(python3 - "$N" <<'PY'
import random, socket, sys
n = 50 * int(sys.argv[1])
for _ in range(500):
    b, socks = random.randrange(20000, 29000), []
    try:
        for p in range(b, b + n):
            s = socket.socket(); socks.append(s); s.bind(("0.0.0.0", p))
        print(b); break
    except OSError:
        pass
    finally:
        for s in socks: s.close()
PY
)
[ -n "$BASE" ] || { echo "[engine] no free ports"; exit 1; }
END=$(( $(date +%s) + LIMIT ))
[ -n "${SLURM_JOB_END_TIME:-}" ] && [ "$SLURM_JOB_END_TIME" -lt "$END" ] && END=$SLURM_JOB_END_TIME
T0=$(date +%s)
echo "[engine] $(date -u +%FT%TZ) run=$RUN node=$(hostname) job=$JID ${N} x TP$TP gpus=$CUDA_VISIBLE_DEVICES end=$(date -d @"$END" +%FT%T)"
echo "[engine] serving code: mas-rl $(git -C "$REPO" rev-parse --short HEAD) miles $(git -C "$MILES" rev-parse --short HEAD); spec $(md5sum < "$SPEC" | cut -c1-12)"
[ $((END - T0)) -gt $((LEAVE_BEFORE + MIN_LEFT)) ] || { echo "[engine] under $(((LEAVE_BEFORE + MIN_LEFT) / 60)) min left on this host: back to the queue"; exit 98; }
PKG=$TMPDIR/flint_pkg
timeout 900 bash "$REPO/serving/build_flint_sglang.sh" "$PKG" || { echo "[engine] plugin build failed"; exit 1; }
URLS=() SPECS=()
for i in $(seq 0 $((N - 1))); do
  URLS+=("http://$IP:$((BASE + 50 * i + 40))")
  SPECS+=("${URLS[i]},$TP,$(IFS=:; echo "${DEV[*]:$((i * TP)):$TP}")")
done
"${REG[@]}" register "$RUN" "$JID" "$(hostname)" "${SPECS[@]}" --state loading --log "$RUN/engine_logs/$JID.0.log" >/dev/null || exit 1
GPU_IDLE_MIN=60 gpu_watch $$ "$RUN/engine_logs/$JID.gpu.log" "grep -q '\"serving\"' '$RUN/engines/$JID.json' 2>/dev/null" 40 &
WD=$!
timeout "$LIMIT" nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits -l 10 \
  > "$RUN/engine_logs/$JID.gpu_mem.csv" 2>/dev/null &
SMI=$!
for i in $(seq 0 $((N - 1))); do
  start timeout -s KILL "$LIMIT" apptainer exec --pid --nv --contain --cleanenv --writable-tmpfs \
    --bind /scratch/gpfs/GROUP --bind "$TMPDIR/tmp":/tmp --bind /dev/shm \
    --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${DEV[*]:$((i * TP)):$TP}")" \
    --env IP="$IP" --env PORT="$((BASE + 50 * i + 40))" --env BASE="$((BASE + 50 * i))" --env SEED_OFFSET="$((10 + 2 * JNUM + i))" \
    --env PKG="$PKG" --env TMPDIR="$TMPDIR" --env SPEC="$SPEC" --env REPO="$REPO" --env MILES="$MILES" \
    "$SIF" bash "$REPO/rl/elastic_engine_inner.sh" serve > "$RUN/engine_logs/$JID.$i.log" 2>&1
done
SRV=("${STARTED[@]}")

http() {   # http URL PATH [JSON BODY [TIMEOUT]] -> the HTTP status (000 when there is no answer)
  local a=(-s --noproxy '*' -m "${4:-10}" -o /dev/null -w '%{http_code}')
  [ -n "${3:-}" ] && a+=(-H 'Content-Type: application/json' -d "$3")
  curl "${a[@]}" "$1$2"
}
alive() { local p; for p in "${SRV[@]}"; do kill -0 "$p" 2>/dev/null || return 1; done; }
# load: every engine answers /health_generate, reports TP GPUs, and generates one token (up to 40 min: 80 GB of weights)
for i in $(seq 0 $((N - 1))); do
  until [ "$(http "${URLS[i]}" /health_generate)" = 200 ]; do
    alive || { echo "[engine] a server ended while loading: $(tail -3 "$RUN/engine_logs/$JID."*.log)"; exit 1; }
    [ -e "$RUN/STOP" ] && { echo "[engine] $RUN/STOP appeared while loading"; exit 0; }
    [ $(( $(date +%s) - T0 )) -gt 2400 ] && { echo "[engine] engine $i not healthy after 40 min"; exit 1; }
    "${REG[@]}" beat "$RUN/engines/$JID.json"
    sleep 10
  done
  ws=$(curl -s --noproxy '*' -m 30 "${URLS[i]}/server_info" | timeout 30 python3 -c 'import json,sys; print(json.load(sys.stdin)["internal_states"][0]["world_size"])' 2>/dev/null)
  [ "$ws" = "$TP" ] || { echo "[engine] engine $i: /server_info world_size '$ws', expected $TP"; exit 1; }
  code=$(http "${URLS[i]}" /generate '{"input_ids": [1, 2, 3, 4], "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}}' 300)
  [ "$code" = 200 ] || { echo "[engine] engine $i: one-token /generate answered $code"; exit 1; }
done
"${REG[@]}" register "$RUN" "$JID" "$(hostname)" "${SPECS[@]}" --state serving --log "$RUN/engine_logs/$JID.0.log" >/dev/null || exit 1
echo "[engine] $(date -u +%FT%TZ) ${N} engine(s) serving after $(( $(date +%s) - T0 )) s: ${URLS[*]}"

running() {   # the requests running or waiting on this job's engines (SGLang /v1/loads; -1 when unknown)
  local i n t=0
  for i in $(seq 0 $((N - 1))); do
    n=$(curl -s --noproxy '*' -m 10 "${URLS[i]}/v1/loads" | timeout 15 python3 -c '
import json, sys
def walk(x):
    if isinstance(x, dict):
        return sum(v for k, v in x.items() if k in ("num_running_reqs", "num_waiting_reqs") and isinstance(v, int)) + sum(walk(v) for v in x.values() if isinstance(v, (dict, list)))
    return sum(walk(v) for v in x) if isinstance(x, list) else 0
print(walk(json.load(sys.stdin)))' 2>/dev/null) || n=
    [[ $n =~ ^[0-9]+$ ]] || { echo -1; return; }
    t=$((t + n))
  done
  echo "$t"
}
leave() {   # leave CODE REASON: the record says leaving (the run withdraws the engines), drain, then exit CODE
  echo "[engine] $(date -u +%FT%TZ) leaving ($2): withdrawn from the run, draining"
  timeout 60 python3 -c "
import json, os, sys, time
f = sys.argv[1]; r = json.load(open(f)); r['leaving'] = time.time()
t = f + '.tmp'; json.dump(r, open(t, 'w')); os.replace(t, f)" "$RUN/engines/$JID.json"
  local t0 n
  t0=$(date +%s)
  while sleep $((BEAT < 15 ? BEAT : 15)); do
    "${REG[@]}" beat "$RUN/engines/$JID.json" >/dev/null
    n=$(running)
    [ $(( $(date +%s) - t0 )) -ge "$DRAIN_MIN" ] && [ "$n" = 0 ] && break
    [ $(( $(date +%s) - t0 )) -ge "$DRAIN_MAX" ] && { echo "[engine] $n request(s) still running after $((DRAIN_MAX / 60)) min"; break; }
    alive || break
  done
  echo "[engine] $(date -u +%FT%TZ) drained after $(( $(date +%s) - t0 )) s; exit $1"
  exit "$1"
}

# serve: heartbeat every BEAT s; leave on STOP, a stale trainer, a dead or unhealthy server, or the host's end
bad=0 seen=
while sleep "$BEAT" & wait $!; do   # a background sleep, so a TERM is handled at once
  "${REG[@]}" beat "$RUN/engines/$JID.json" || "${REG[@]}" register "$RUN" "$JID" "$(hostname)" "${SPECS[@]}" \
    --state serving --log "$RUN/engine_logs/$JID.0.log" >/dev/null
  alive || { echo "[engine] $(date -u +%FT%TZ) a server ended: $(tail -3 "$RUN/engine_logs/$JID."*.log)"; exit 1; }
  ok=1; for u in "${URLS[@]}"; do [ "$(http "$u" /health)" = 200 ] || ok=; done
  if [ -n "$ok" ]; then bad=0; else bad=$((bad + 1)); fi
  [ $((bad * BEAT)) -ge 600 ] && { echo "[engine] $(date -u +%FT%TZ) /health failing for $((bad * BEAT)) s"; exit 1; }
  [ -e "$RUN/STOP" ] && { echo "[engine] $(date -u +%FT%TZ) $RUN/STOP: $(cat "$RUN/STOP" 2>/dev/null); done"; exit 0; }
  age=$(trainer_age)
  [ "$age" -le 90 ] && seen=1
  [ -n "$seen" ] && [ "$age" -gt "$STALE" ] && leave 98 "the trainer's heartbeat is $age s old"
  [ "$(date +%s)" -gt $((END - LEAVE_BEFORE)) ] && leave 99 "$((LEAVE_BEFORE / 60)) min before the host's end"
  [ -e "$RUN/engines/$JID.leave" ] && leave 0 "asked by $RUN/engines/$JID.leave"
done
