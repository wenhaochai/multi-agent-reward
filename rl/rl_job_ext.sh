#!/bin/bash
#SBATCH --job-name=flint-rl-ext
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=880G
#SBATCH --time=03:00:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/tmp/flint_s5/logs/%x-%j.out
#FETCHABLE
# RL of Flint as rl/rl_job.sh, with the rollout served by engine jobs on other nodes (rl/engine_job.sh), stage 0 of
# tmp/elastic_rl/design.md: this job holds the FSDP actor (8 GPUs), the router, the session servers and the agents;
# the engines are found in RUN/engines/ (rl/registry.py) and stay fixed for the run; weights go to them over NCCL
# (broadcast). Usage: rl_job_ext.sh smoke_ext RUN (RUN: the run directory the engine jobs are given too).
#   smoke_ext  2 steps of 8 episodes x 4k tokens, then a checkpoint (actor and optimizer, resumable) and exit 0
# Pairing: the job writes RUN/trainer.json (its heartbeat, every 30 s) from its start, waits for a serving engine
# (any engine record within 20 min, then up to 30 min for it to serve), and writes RUN/STOP when it ends (unless it
# exits 99 to be requeued), which ends the engine jobs. Exit codes: 0 done; 2 no engine (RUN/STOP written, so an
# engine job that starts later exits at once); 3 RUN/STOP existed at the start; 1 any other failure (an engine lost
# for 10 min among them); 143 a signal. Bounds: timeout -s KILL around the container; the GPU watch counts idle
# minutes only in training phases (actor load, log-probs, actor step, weight push: rl/gpu_watch.sh GPU_ACTIVE_CMD),
# with 80 min at 0% in any phase as the outer bound; cleanup of only the process groups this job started.
set -uo pipefail
MODE=${1:-} RUN=${2:-}
case "$MODE" in smoke_ext) LIMIT=${EXT_LIMIT_S:-10200};; *) echo "usage: rl_job_ext.sh smoke_ext RUN"; exit 1;; esac
[ -n "$RUN" ] || { echo "usage: rl_job_ext.sh smoke_ext RUN"; exit 1; }
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
[ -n "$JID" ] || { echo "no job id"; exit 1; }
REPO=${EXT_REPO:-/scratch/gpfs/GROUP/USER/project/mas-rl-elastic}
MILES=${EXT_MILES:-/scratch/gpfs/GROUP/USER/project/miles-rollout}
SIF=${EXT_SIF:-/scratch/gpfs/GROUP/USER/project/miles-q38-build/miles-v0.1.3.sif}
FLINT=${EXT_FLINT:-/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint}
WAIT_S=${EXT_WAIT_S:-1200} LOAD_WAIT_S=${EXT_LOAD_WAIT_S:-1800} LOST_S=${EXT_ENGINE_LOST_S:-600}
REG=(timeout 60 python3 "$REPO/rl/registry.py")   # a scratch stall cannot hold the job
OUT=$RUN
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
[ -e "$OUT/STOP" ] && { echo "[ext] $OUT/STOP exists: the run is over ($(cat "$OUT/STOP")); use a new RUN"; exit 3; }
mkdir -p "$OUT" "$TMPDIR/tmp" "$TMPDIR/datasets"
rm -f "$OUT/exit_code" "$OUT/stop_request" "$OUT/save_now" "$OUT/train_segment.log"
[ -d "$OUT/traces_live" ] && mv "$OUT/traces_live" "$OUT/traces_live.stale_$(date +%s)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # the engines are on the cluster network; nothing goes outside
source "$REPO/serving/check/procs.sh"
source "$REPO/rl/gpu_watch.sh"
WD= MEM= SMI= HB=
stop_file() {   # the run is over for the engines, except when this job asks wq for another segment (99)
  [ "$1" = 99 ] && return
  [ -e "$OUT/STOP" ] || echo "trainer $JID exit $1 ($(date -u +%FT%TZ))" > "$OUT/STOP"
}
trap 'rc=$?; stop_file $rc; stop "${STARTED[@]}"; kill $WD $MEM $SMI $HB 2>/dev/null; rm -rf -- "/scratch/gpfs/GROUP/USER/tmp/job/$JID"' EXIT
trap 'exit "$(cat "$OUT/exit_code" 2>/dev/null || echo 143)"' TERM INT
IP=$(ip -4 -o addr show eno8303 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
[ -n "$IP" ] || IP=$(hostname -i | tr ' ' '\n' | grep -E '^172\.' | head -1)
[ -n "$IP" ] || { echo "[ext] no routable address (eno8303)"; exit 1; }
T0=$(date +%s)
echo "[ext] $(date -u +%FT%TZ) $MODE run=$OUT node=$(hostname) ip=$IP job=$JID gpus=${CUDA_VISIBLE_DEVICES:-}"
# the heartbeat the engine jobs watch (they leave 15 min after it stops), from the start: an engine that registers
# while this job waits sees a trainer
( while kill -0 $$ 2>/dev/null; do "${REG[@]}" trainer-beat "$OUT" "$JID" "$(hostname)" "$IP" "$T0"; sleep 30; done ) &
HB=$!
END=$(( T0 + LIMIT ))
[ -n "${SLURM_JOB_END_TIME:-}" ] && [ "$SLURM_JOB_END_TIME" -lt "$END" ] && END=$SLURM_JOB_END_TIME
echo "mas-rl $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain -uno | wc -l)" \
  "miles $(git -C "$MILES" rev-parse --short HEAD) dirty=$(git -C "$MILES" status --porcelain -uno | wc -l)" | tee "$OUT/code"
# GPU watch from the start: until the actor is loaded (its first "train_wait" timer) the job is loading, which here
# includes the wait for an engine (<= 20 + 30 min) and the actor load (~8 min): at most 70 min, under the host's 90.
# Afterwards idle minutes count only while the actor works (its last timer line is the start of train, log_probs,
# actor_train, update_weights or update_weights_implementation): the trainer's GPUs idle by design while the remote
# engines generate (~30-45 min per step).
TIMERS="grep -a 'actor_cell00000_rank00000.*timer.py' '$OUT/train_segment.log' 2>/dev/null | tail -1"
GPU_ACTIVE_CMD="$TIMERS | grep -qE 'Timer (train|log_probs|actor_train|update_weights|update_weights_implementation) start'" \
GPU_IDLE_MIN=20 GPU_IDLE_MAX_MIN=80 gpu_watch $$ "$OUT/gpu.log" \
  "grep -aq 'rank00000.*Timer train_wait start' '$OUT/train_segment.log' 2>/dev/null" 70 &
WD=$!

# the engines: >= 1 serving engine of one TP size; fixed for the run (stage 0)
timeout $((WAIT_S + LOAD_WAIT_S + 120)) python3 "$REPO/rl/registry.py" wait "$OUT" --min 1 --timeout "$WAIT_S" --load-timeout "$LOAD_WAIT_S" > "$TMPDIR/engines.txt" &
wait $!   # in the background, so a TERM is handled at once
rc=$?
got=$(cat "$TMPDIR/engines.txt")
case $rc in
  0) ;;
  3) echo "[ext] $OUT/STOP appeared while waiting: $(cat "$OUT/STOP")"; exit 3 ;;
  *) echo "[ext] $(date -u +%FT%TZ) no engine: $( [ $rc = 2 ] && echo "none serving in time" || echo "registry error $rc")"
     echo "trainer $JID: no engine serving within the wait ($(date -u +%FT%TZ))" > "$OUT/STOP"; exit 2 ;;
esac
IFS=$'\t' read -r ADDRS ENGINE_TP ENGINE_NGPUS ENGINE_JOBS <<< "$got"
echo "[ext] $(date -u +%FT%TZ) engines after $(( $(date +%s) - T0 )) s: $ADDRS (TP $ENGINE_TP, $ENGINE_NGPUS GPUs; jobs $ENGINE_JOBS)" | tee "$OUT/engines_used.txt"

python3 - "$FLINT" <<'PY' || exit 1
import json, os, sys
d = sys.argv[1]
files = sorted(set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values()))
missing = [f for f in files + ["config.json", "tokenizer.json"] if not os.path.isfile(os.path.join(d, f))]
sys.exit(f"[ext] PREFLIGHT_FAIL {d}: {missing}") if missing else print(f"[ext] preflight: {len(files)} weight files")
PY
PKG=$TMPDIR/flint_pkg
timeout 900 bash "$REPO/serving/build_flint_sglang.sh" "$PKG" || { echo "[ext] plugin build failed"; exit 1; }
python3 - "$REPO/data/fcs11_val40_agent.jsonl" "$OUT/data_p0.jsonl" <<'PY' || exit 1
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1]) if json.loads(x)["metadata"]["pid"] == "0"]
assert len(rows) == 1
rows[0]["metadata"]["label"] = rows[0]["label"]
open(sys.argv[2], "w").write((json.dumps(rows[0]) + "\n") * 1000)
PY
# host memory every 10 s and the planned stops (as rl/rl_job.sh): over 860 GB a checkpoint then exit 3, over 875 GB
# exit 3 at once; and an engine job whose record is not fresh and serving for LOST_S seconds (10 min) stops the run (1): the
# engine set is fixed, so the run cannot go on without it (LOST_S leaves time to see what the router and the agents do)
CG=/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup | sed -E 's#(/job_[0-9]+)/.*#\1#')/memory.current
( asked= lost=0
  request() { [ -e "$OUT/stop_request" ] || echo "$1" > "$OUT/stop_request"; }
  finish() { echo "$1" | cut -d' ' -f1 > "$OUT/exit_code"; echo "PLANNED_STOP: $1" >&2; kill -TERM $$; }
  while sleep 10 && kill -0 $$ 2>/dev/null; do
    cg=$(( $(cat "$CG" 2>/dev/null || echo 0) / 1048576 ))
    echo "$(date +%s) $(free -m | awk '/Mem:/{print $3}') $cg"
    if [ "$cg" -gt $((875 * 1024)) ]; then finish "3 host memory $cg MiB over 875 GB"; break; fi
    [ "$cg" -gt $((860 * 1024)) ] && request "3 host memory $cg MiB over 860 GB"
    # shellcheck disable=SC2086
    if "${REG[@]}" trainer-check "$OUT" $ENGINE_JOBS; then lost=0; else lost=$((lost + 1)); fi
    [ "$lost" = 1 ] && echo "$(date -u +%FT%TZ) engine record lost (jobs $ENGINE_JOBS)" >&2
    if [ $((lost * 10)) -ge "$LOST_S" ]; then finish "1 engine lost: no fresh serving record of $ENGINE_JOBS for $LOST_S s"; break; fi
    [ "$(date +%s)" -gt $((END - 600)) ] && request "1 time: less than 10 min left before a smoke ended"
    if [ -e "$OUT/stop_request" ] && [ -z "$asked" ]; then touch "$OUT/save_now"; asked=1; fi
    if [ -n "$asked" ] && [ ! -e "$OUT/save_now" ]; then finish "$(cat "$OUT/stop_request")"; break; fi
  done ) >> "$OUT/host_mem.txt" 2>> "$OUT/stops.log" &
MEM=$!
timeout "$LIMIT" nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 2 > "$OUT/gpu_mem.csv" 2>/dev/null &
SMI=$!
start timeout -k 300 "$LIMIT" apptainer exec --pid --nv --contain --cleanenv --writable-tmpfs \
  --bind /scratch/gpfs/GROUP --bind "$TMPDIR/tmp":/tmp --bind "$TMPDIR/datasets":/root/datasets --bind /proc:/host/proc \
  --bind /dev/shm --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  --env JID="$JID" --env OUT="$OUT" --env PKG="$PKG" --env TMPDIR="$TMPDIR" --env IP="$IP" --env REPO="$REPO" \
  --env MILES="$MILES" --env FLINT="$FLINT" --env ADDRS="$ADDRS" --env ENGINE_TP="$ENGINE_TP" \
  --env ENGINE_NGPUS="$ENGINE_NGPUS" "$SIF" bash "$REPO/rl/rl_inner_ext.sh" "$MODE"
wait "${STARTED[0]}"
rc=$?
echo "[ext] $(date -u +%FT%TZ) container exit $rc"
python3 - "$OUT" <<'PY'
import collections, sys
out = sys.argv[1]
peak = collections.defaultdict(int)
for line in open(f"{out}/gpu_mem.csv"):
    p = [x.strip() for x in line.split(",")]
    if len(p) == 3 and p[2].isdigit():
        peak[int(p[1])] = max(peak[int(p[1])], int(p[2]))
host = max((int(l.split()[-1]) for l in open(f"{out}/host_mem.txt") if len(l.split()) >= 2), default=0)
print("[ext] peak GPU memory (2 s samples, MiB):", dict(sorted(peak.items())), "| peak job memory (cgroup, 10 s samples, MiB):", host)
PY
exit $rc
