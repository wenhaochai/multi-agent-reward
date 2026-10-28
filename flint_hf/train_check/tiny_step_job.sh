#!/bin/bash
#SBATCH --job-name=flint-s2-tiny
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
# S2 integration run before the 40B step: step_job.sh's two GRPO + score-centering steps through miles' FSDP actor on a
# tiny random Flint (make_tiny.py: Flint's vocabulary and structure, 80M parameters) on 4 GPUs, same batch and flags.
# Usage: tiny_step_job.sh DATA_DIR OUT_DIR --hf-checkpoint TINY_DIR
set -uo pipefail
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
P=/scratch/gpfs/GROUP/USER/project
SIF=$P/miles-q38-build/miles-v0.1.3.sif
MODEL=$P/labs-molt/_workspace/models/flint
DATA=${1:?data dir} OUT=${2:?out dir}
shift 2
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
JOBTMP=/tmp/miles_$JID
mkdir -p "$TMPDIR" "$JOBTMP" "$OUT"
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "/scratch/gpfs/GROUP/USER/tmp/job/$JID" "/tmp/miles_$JID" || true' EXIT
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
# pre-flight: every weight file is a real file (never the evictable HF cache), and the batch exists
python3 - "$MODEL" "$DATA" <<'PY' || exit 1
import json, os, sys
d, data = sys.argv[1], sys.argv[2]
files = sorted(set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values()))
missing = [f for f in files + ["config.json", "tokenizer.json"] if not os.path.isfile(os.path.join(d, f))]
missing += [f for f in ("rollout_0.pt", "rollout_1.pt") if not os.path.isfile(os.path.join(data, f))]
sys.exit(f"PREFLIGHT_FAIL missing {missing}") if missing else print(f"preflight: {len(files)} weight files, batch present")
PY
echo "[s2] $(date -u +%FT%TZ) node=$(hostname) job=$JID gpus=$CUDA_VISIBLE_DEVICES"
# only this job's GPUs: on a shared host the others belong to another guest
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,memory.used --format=csv,noheader -l 2 > "$OUT/gpu_mem.csv" 2>/dev/null &
( while true; do echo "$(date +%s) $(free -m | awk '/Mem:/{print $3}')"; sleep 2; done ) > "$OUT/host_mem.txt" &
# in its own process group, with a hard limit and an idle-GPU watchdog (safe_run.sh)
$P/mas-rl-flintrl/flint_hf/train_check/safe_run.sh 1500 15 apptainer exec --nv --contain --cleanenv --writable-tmpfs \
  --bind /scratch/gpfs/GROUP/USER --bind "$JOBTMP":/tmp --bind "$TMPDIR" --bind /dev/shm \
  --env PYTHONNOUSERSITE=1 --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" --env JID="$JID" --env TMPDIR="$TMPDIR" \
  "$SIF" bash $P/mas-rl-flintrl/flint_hf/train_check/step_inner.sh "$DATA" "$OUT" "$@"
rc=$?
python3 - "$OUT" <<'PY'
import collections, sys
out = sys.argv[1]
peak = collections.defaultdict(int)
for line in open(f"{out}/gpu_mem.csv"):
    parts = [x.strip() for x in line.split(",")]
    if len(parts) == 3 and parts[2].endswith("MiB"):
        peak[int(parts[1])] = max(peak[int(parts[1])], int(parts[2].split()[0]))
host = max((int(l.split()[1]) for l in open(f"{out}/host_mem.txt") if len(l.split()) == 2), default=0)
print("[s2] peak GPU memory (nvidia-smi, MiB):", dict(sorted(peak.items())), "| peak host used (MiB):", host)
PY
echo "[s2] exit $rc"
exit $rc
