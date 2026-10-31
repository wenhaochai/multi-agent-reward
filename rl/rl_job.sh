#!/bin/bash
#SBATCH --job-name=flint-rl
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=880G
#SBATCH --time=03:00:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/tmp/flint_s5/logs/%x-%j.out
#FETCHABLE
# RL of Flint on the long game through miles v0.1.3 (our fork), on one 8-GPU block: GRPO without std normalization,
# score centering over the sampler's top-32 candidates, routing replay (R3), one optimizer step per rollout (problem 0,
# 8 episodes of 128k tokens), the rollout engines (4 x TP2) and the FSDP actor (CPU-offloaded parameters and Adam)
# colocated. Usage: rl_job.sh smoke|test2|run20 (wq --prio 10; smoke and run20 with their own --time). smoke: the
# same path in ~50 min (2 steps of 8 episodes x 4k tokens, no actor save); run it before test2. The game runs with RL's context cap
# (rl/rl_inner.sh: 110k characters, 16k-token turns) so that a training sample stays near 64k tokens.
#   test2     2 colocated RL steps, the actor saved at the end
#   run20     20 colocated RL steps, the actor saved at the end
# train67 modes (Frontier-CS v1.1's 67 training problems, data/fcs11_train67_agent.jsonl, pre-shuffled with a fixed seed
# by rl/train67_data.py): P problems per step x G episodes each, one GRPO group per problem, one TP8 engine; usage
# rl_job.sh train67[_test|_smoke] [P G [STEPS]] [set=NAME] (defaults 4 8, train67 40 steps; set=NAME trains on the
# problems of data/train67_sets/NAME.txt only, default all 67; the run dirs then hold _NAME):
#   train67_smoke  B = P x G episodes of 4k tokens, 2 steps, no save (~1 h)
#   train67_test   B episodes of 128k, 2 steps, a full checkpoint after each (host memory at its peak: the end of the
#                  optimizer step and the save; step time), resumable in runs/train67_test[_NAME]_p<P>g<G>, exit 0
#                  when done
#   train67        the run, resumable in runs/train67[_NAME]_p<P>g<G>: checkpoints every 2 steps, the newest 2 kept
# The first failure ends the job: nothing is retried. Results: tmp/flint_s5/runs/<job>/. One container with a PID
# namespace of its own (--pid) holds everything (see rl/rollout_job.sh); bounds: timeout -s KILL around it, the GPU
# watch (rl/gpu_watch.sh) and cleanup of only the process groups this job started (serving/check/procs.sh).
set -uo pipefail
MODE=${1:-}
# run20: wq --partial --time 23:00:00; train67: wq --partial --time 23:00:00; train67_test: wq --partial --time 6:00:00
# (2 steps of 128k at B = 32: ~20 min start, 2 x (<= 60 min rollout + ~16 min train + sync and save) ~ 3 h; its time
# stop comes 105 min before the end, at 4.25 h); train67_smoke: wq --time 1:45:00
case "$MODE" in smoke|smoke_tp4|smoke_tp8) LIMIT=4800;; train67_smoke) LIMIT=6000;; base_tp8) LIMIT=8400;; test2) LIMIT=9600;;
  run20|train67) LIMIT=81000;; train67_test) LIMIT=21600;;
  *) echo "usage: rl_job.sh smoke[_tp4|_tp8]|base_tp8|test2|run20|train67[_test|_smoke] [P G [STEPS]] [set=NAME]"; exit 1;; esac
# train67: set=NAME at any place after the mode; the other arguments are P G STEPS in this order
T67_SET= POS=()
for x in "${@:2}"; do
  case "$x" in set=*) T67_SET=${x#set=};; *) POS+=("$x");; esac
done
T67_P=${POS[0]:-4} T67_G=${POS[1]:-8} T67_STEPS=${POS[2]:-40}
case "$MODE" in train67*) [[ $T67_P =~ ^[1-9][0-9]*$ && $T67_G =~ ^[1-9][0-9]*$ && $T67_STEPS =~ ^[1-9][0-9]*$ ]] \
  || { echo "P G STEPS must be positive integers: $T67_P $T67_G $T67_STEPS"; exit 1; }
  [ ${#POS[@]} -le 3 ] || { echo "too many arguments: ${POS[*]}"; exit 1; }
  [ -z "$T67_SET" ] || [[ $T67_SET =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "set=NAME: letters, digits, _ and -: $T67_SET"; exit 1; }
  ;;
  *) [ -z "$T67_SET" ] || { echo "set=NAME is for the train67 modes only, not $MODE"; exit 1; };; esac
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
[ -n "$JID" ] || { echo "no job id"; exit 1; }
REPO=/scratch/gpfs/GROUP/USER/project/mas-rl-train67b
MILES=/scratch/gpfs/GROUP/USER/project/miles-rollout
SIF=/scratch/gpfs/GROUP/USER/project/miles-q38-build/miles-v0.1.3.sif
FLINT=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint
OUT=/scratch/gpfs/GROUP/USER/tmp/flint_s5/runs/$JID
# run20 lives in one fixed directory, so that any submission resumes it (the checkpoints, steps.jsonl, the traces)
[ "$MODE" = run20 ] && OUT=/scratch/gpfs/GROUP/USER/tmp/flint_s5/runs/run20_p0
# so do the resumable train67 runs, one directory per problem set and batch shape (a set run never resumes a run on all
# 67 problems, nor the reverse)
T67_TAG=${T67_SET:+_$T67_SET}
[ "$MODE" = train67 ] && OUT=/scratch/gpfs/GROUP/USER/tmp/flint_s5/runs/train67${T67_TAG}_p${T67_P}g${T67_G}
[ "$MODE" = train67_test ] && OUT=/scratch/gpfs/GROUP/USER/tmp/flint_s5/runs/train67_test${T67_TAG}_p${T67_P}g${T67_G}
RESUMABLE=; case "$MODE" in run20|train67|train67_test) RESUMABLE=1;; esac
# the planned time stop: one step and a save must fit after it (run20: 43 min a step; train67 at B = 32: up to the
# 60-min wall cap, ~4 x run20's 4-min train time, sync and save, ~85 min)
STOP_S=4500; case "$MODE" in train67*) STOP_S=6300;; esac
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
mkdir -p "$OUT" "$TMPDIR/tmp" "$TMPDIR/datasets"
rm -f "$OUT/exit_code" "$OUT/stop_request" "$OUT/save_now" "$OUT/train_segment.log"   # this segment's stop state (run20 resumes in the same directory)
FRESH=; [ -e "$OUT/actor/latest_checkpointed_iteration.txt" ] || FRESH=1
# an interrupted segment's episode traces belong to no rollout of this one
[ -d "$OUT/traces_live" ] && mv "$OUT/traces_live" "$OUT/traces_live.stale_$(date +%s)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
source "$REPO/serving/check/procs.sh"
source "$REPO/rl/gpu_watch.sh"
WD= MEM= SMI=
trap 'stop "${STARTED[@]}"; kill $WD $MEM $SMI 2>/dev/null; rm -rf -- "/scratch/gpfs/GROUP/USER/tmp/job/$JID"' EXIT
trap 'exit "$(cat "$OUT/exit_code" 2>/dev/null || echo 143)"' TERM INT   # a planned stop leaves its exit code there
echo "[s5] $(date -u +%FT%TZ) $MODE node=$(hostname) job=$JID gpus=${CUDA_VISIBLE_DEVICES:-}"
# this segment's end: Slurm's (or a fetching host's) SLURM_JOB_END_TIME, else the time limit of the parent timeout (a
# fetched guest runs under `timeout <s>s`), else LIMIT; the planned time stop comes 75 min before it
END=$(( $(date +%s) + LIMIT )) SRC=LIMIT
if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
  [ "$SLURM_JOB_END_TIME" -lt "$END" ] && END=$SLURM_JOB_END_TIME; SRC=SLURM_JOB_END_TIME
elif d=$(tr '\0' '\n' < /proc/$PPID/cmdline 2>/dev/null | grep -xE '[0-9]+s' | tail -1 | tr -d s) && [ -n "$d" ]; then
  [ $(( $(date +%s) + d )) -lt "$END" ] && END=$(( $(date +%s) + d )); SRC="parent timeout ${d}s"
fi
echo "[s5] segment end $(date -d @"$END" +%FT%T) ($((END - $(date +%s))) s left; from $SRC); planned stop $((STOP_S / 60)) min before it"
echo "mas-rl $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain -uno | wc -l)" \
  "miles $(git -C "$MILES" rev-parse --short HEAD) dirty=$(git -C "$MILES" status --porcelain -uno | wc -l)" | tee "$OUT/code"
python3 - "$FLINT" <<'PY' || exit 1
import json, os, sys
d = sys.argv[1]
files = sorted(set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values()))
missing = [f for f in files + ["config.json", "tokenizer.json"] if not os.path.isfile(os.path.join(d, f))]
sys.exit(f"[s5] PREFLIGHT_FAIL {d}: {missing}") if missing else print(f"[s5] preflight: {len(files)} weight files")
PY
PKG=$TMPDIR/flint_pkg
bash "$REPO/serving/build_flint_sglang.sh" "$PKG" || exit 1
# the RL problem: val40 problem 0, its directory also in the metadata (miles hands the agent only the metadata); the
# row 1000 times, so that miles' epoch (one pass over the rows, a save point) never ends inside a run
python3 - "$REPO/data/fcs11_val40_agent.jsonl" "$OUT/data_p0.jsonl" <<'PY' || exit 1
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1]) if json.loads(x)["metadata"]["pid"] == "0"]
assert len(rows) == 1
rows[0]["metadata"]["label"] = rows[0]["label"]
open(sys.argv[2], "w").write((json.dumps(rows[0]) + "\n") * 1000)
PY
# train67: the training problems (all 67, or set=NAME's) in a fixed order (seed 67, 30 epochs: miles' epoch end, a save
# point, comes after n x 30 / P steps, 255 for 34 problems at P = 4, outside any run). A resumed run must read the same
# file: an existing one is checked, not rebuilt.
if [[ $MODE == train67* ]]; then
  echo "[s5] problem set: ${T67_SET:-none (all 67)}"   # train67_data.py's line below gives its size
  python3 "$REPO/rl/train67_data.py" "$REPO/data/fcs11_train67_agent.jsonl" "$OUT/data_train67.new.jsonl" "$T67_P" \
    ${T67_SET:+--set "$T67_SET"} || exit 1
  if [ -e "$OUT/data_train67.jsonl" ] && ! cmp -s "$OUT/data_train67.jsonl" "$OUT/data_train67.new.jsonl"; then
    echo "[s5] $OUT/data_train67.jsonl differs from the one this code builds: a resumed run would see other problems"; exit 1
  fi
  mv "$OUT/data_train67.new.jsonl" "$OUT/data_train67.jsonl"
fi
# one line per GPU and minute; stops the job when a GPU idles (rl/gpu_watch.sh), counting from the first rollout
# request: the start (actor weights 8 min in w583, then the engines; GPUs idle 11 min) may take up to 35 min. An
# engine whose episodes are done idles until the group's slowest episode ends (128k episodes take 17-34 min; w611's
# 8k episodes 18-28 min and one over 31), so a loaded GPU may idle 45 min (the host is cancelled at 90)
# train67: 32 episodes keep the one engine busy until the last one ends (up to the 60-min cap): 30 min at 0% is a hang
IDLE=45; [[ $MODE == train67* ]] && IDLE=30
GPU_IDLE_MIN=$IDLE gpu_watch $$ "$OUT/gpu.log" "grep -q 'POST /v1/chat/completions' '$OUT/train_segment.log' 2>/dev/null" 35 &
WD=$!
# host memory every 10 s (the node's used MiB, this job's cgroup MiB: what --mem bounds) and the planned stops. A
# stop request ($OUT/stop_request: "<exit code> <reason>", from here or from flint.miles_agent.log_rollout's rules)
# asks miles for a checkpoint at its next save point (--save-trigger-sentinel $OUT/save_now); once miles has written it
# (the sentinel is gone) the job stops with that exit code. Requests: cgroup memory over 860 GB (3: a failure); less
# than 75 min left of LIMIT (99: wq requeues it, the next segment resumes from the checkpoint); run20's first segment
# stops after its first checkpoint (99), so the second one proves the resume. Over 875 GB the job stops at once (3).
CG=/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup | sed -E 's#(/job_[0-9]+)/.*#\1#')/memory.current   # the job's
TRACKER=$OUT/actor/latest_checkpointed_iteration.txt
( t0=$SECONDS asked=
  request() { [ -e "$OUT/stop_request" ] || echo "$1" > "$OUT/stop_request"; }
  finish() { echo "$1" | cut -d' ' -f1 > "$OUT/exit_code"; echo "PLANNED_STOP: $1" >&2; kill -TERM $$; }
  while sleep 10; do
    cg=$(( $(cat "$CG" 2>/dev/null || echo 0) / 1048576 ))
    echo "$(date +%s) $(free -m | awk '/Mem:/{print $3}') $cg"
    if [ "$cg" -gt $((875 * 1024)) ]; then finish "3 host memory $cg MiB over 875 GB"; break; fi
    [ "$cg" -gt $((860 * 1024)) ] && request "3 host memory $cg MiB over 860 GB"
    [ -n "$RESUMABLE" ] && [ "$(date +%s)" -gt $((END - STOP_S)) ] && request "99 time: less than $((STOP_S / 60)) min left"
    if [ "$MODE" = run20 ] && [ -n "$FRESH" ] && [ -e "$TRACKER" ]; then finish "99 first segment: checkpoint $(cat "$TRACKER") written, resume next"; break; fi
    if [ -e "$OUT/stop_request" ] && [ -z "$asked" ]; then touch "$OUT/save_now"; asked=1; fi
    if [ -n "$asked" ] && [ ! -e "$OUT/save_now" ]; then finish "$(cat "$OUT/stop_request")"; break; fi
  done ) >> "$OUT/host_mem.txt" 2>> "$OUT/stops.log" &
MEM=$!
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 2 > "$OUT/gpu_mem.csv" 2>/dev/null &
SMI=$!
start timeout -k 300 "$LIMIT" apptainer exec --pid --nv --contain --cleanenv --writable-tmpfs \
  --bind /scratch/gpfs/GROUP --bind "$TMPDIR/tmp":/tmp --bind "$TMPDIR/datasets":/root/datasets --bind /proc:/host/proc \
  --bind /dev/shm --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  --env JID="$JID" --env OUT="$OUT" --env PKG="$PKG" --env TMPDIR="$TMPDIR" --env T67_P="$T67_P" --env T67_G="$T67_G" \
  --env T67_STEPS="$T67_STEPS" "$SIF" bash "$REPO/rl/rl_inner.sh" "$MODE"
wait "${STARTED[0]}"
rc=$?
echo "[s5] $(date -u +%FT%TZ) container exit $rc"
python3 - "$OUT" <<'PY'
import collections, sys
out = sys.argv[1]
peak = collections.defaultdict(int)
for line in open(f"{out}/gpu_mem.csv"):
    p = [x.strip() for x in line.split(",")]
    if len(p) == 3 and p[2].isdigit():
        peak[int(p[1])] = max(peak[int(p[1])], int(p[2]))
host = max((int(l.split()[-1]) for l in open(f"{out}/host_mem.txt") if len(l.split()) >= 2), default=0)
print("[s5] peak GPU memory (2 s samples, MiB):", dict(sorted(peak.items())), "| peak job memory (cgroup, 10 s samples, MiB):", host)
PY
exit $rc
