#!/bin/bash
#SBATCH --job-name=flint-rollout
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/tmp/flint_s5a/logs/%x-%j.out
#FETCHABLE
# Long-game rollouts of Flint through miles v0.1.3, rollout only, on 4 GPUs (2 engines at TP2, Flint's H100 layout; 8
# concurrent episodes, 4 per engine). Usage: rollout_job.sh probe|full (wq --prio 10).
#   probe  miles' GPU TITO check for Flint, then 8 episodes of 8k tokens on the RL test problem with the context cut to
#          20k characters (trims within the episode): the whole path, short
#   full   8 episodes of 128k tokens on the RL test problem (val40 problem 0) with the top-32 candidates, then 8 without
#          them (what recording them costs)
# Results: tmp/flint_s5a/runs/<job>/. Everything runs in one container with a PID namespace of its own (--pid), so
# miles' and Ray's cleanup sees only this job and the container's process group holds all of it. Bounds: timeout -s
# KILL around the container, a GPU-idle watchdog (10 min at 0%), and cleanup of only the process groups this job
# started (serving/check/procs.sh).
mkdir -p /scratch/gpfs/GROUP/USER/tmp/flint_s5a/logs
set -uo pipefail
MODE=${1:-}
case "$MODE" in probe) LIMIT=3300;; full) LIMIT=9600;; *) echo "usage: rollout_job.sh probe|full"; exit 1;; esac
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
[ -n "$JID" ] || { echo "no job id"; exit 1; }
REPO=/scratch/gpfs/GROUP/USER/project/mas-rl-rollout
SIF=/scratch/gpfs/GROUP/USER/project/miles-q38-build/miles-v0.1.3.sif
FLINT=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint
OUT=/scratch/gpfs/GROUP/USER/tmp/flint_s5a/runs/$JID
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
mkdir -p "$OUT" "$TMPDIR/tmp"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
source "$REPO/serving/check/procs.sh"
source "$REPO/serving/check/gpu_watchdog.sh"
WD=
trap 'stop "${STARTED[@]}"; [ -z "$WD" ] || kill "$WD" 2>/dev/null; rm -rf -- "/scratch/gpfs/GROUP/USER/tmp/job/$JID"' EXIT
trap 'exit 143' TERM INT
echo "[s5a] $(date -u +%FT%TZ) $MODE node=$(hostname) job=$JID gpus=${CUDA_VISIBLE_DEVICES:-} code=$(git -C "$REPO" rev-parse --short HEAD)"
# pre-flight: every weight file of the checkpoint is a real file (an evicted cache leaves dangling links)
python3 - "$FLINT" <<'PY' || exit 1
import json, os, sys
d = sys.argv[1]
files = sorted(set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values()))
missing = [f for f in files + ["config.json", "tokenizer.json"] if not os.path.isfile(os.path.join(d, f))]
sys.exit(f"[s5a] PREFLIGHT_FAIL {d}: {missing}") if missing else print(f"[s5a] preflight: {len(files)} weight files")
PY
# the Flint plugin for SGLang 0.5.23, built into this job's directory
PKG=$TMPDIR/flint_pkg
bash "$REPO/serving/build_flint_sglang.sh" "$PKG" || exit 1
# the RL test problem: val40 problem 0, its directory also in the metadata (miles hands the agent only the metadata)
python3 - "$REPO/data/fcs11_val40_agent.jsonl" "$OUT/data_p0.jsonl" <<'PY' || exit 1
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1]) if json.loads(x)["metadata"]["pid"] == "0"]
assert len(rows) == 1
rows[0]["metadata"]["label"] = rows[0]["label"]
open(sys.argv[2], "w").write(json.dumps(rows[0]) + "\n")
PY
# 20 min: the engines load 80 GB at 0% first, and an engine whose episodes ended early waits for the group's last
IDLE_MAX=20 gpu_watchdog $$ &
WD=$!
start timeout -s KILL "$LIMIT" apptainer exec --pid --nv --contain --cleanenv --writable-tmpfs \
  --bind /scratch/gpfs/GROUP --bind "$TMPDIR/tmp":/tmp --bind /dev/shm \
  --env PYTHONNOUSERSITE=1 --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" --env JID="$JID" --env OUT="$OUT" \
  --env PKG="$PKG" --env TMPDIR="$TMPDIR" "$SIF" bash "$REPO/rl/rollout_inner.sh" "$MODE"
wait "${STARTED[0]}"
rc=$?
echo "[s5a] $(date -u +%FT%TZ) container exit $rc"
exit $rc
