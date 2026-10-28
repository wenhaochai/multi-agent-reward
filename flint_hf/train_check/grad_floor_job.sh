#!/bin/bash
#SBATCH --job-name=flint-s2-gradfloor
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
# S2: grad_floor_check.py on 4 GPUs in miles-v0.1.3.sif: per-layer gradient agreement of two correct computations (the floor).
# Usage: grad_floor_job.sh SEQS_JSON OUT_LOG
set -uo pipefail
P=/scratch/gpfs/GROUP/USER/project
W=$P/mas-rl-flintrl
MILES=$P/miles-flint
SIF=$P/miles-q38-build/miles-v0.1.3.sif
SEQS=${1:?seqs} LOG=${2:?log}
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
echo "job $JID host $(hostname) flint_hf $(git -C $W rev-parse --short HEAD) miles $(git -C $MILES rev-parse --short HEAD) GPUs $CUDA_VISIBLE_DEVICES"
APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES APPTAINERENV_TMPDIR=$TMPDIR APPTAINERENV_PYTHONPATH=$W:$MILES \
APPTAINERENV_TRITON_CACHE_DIR=$TMPDIR/triton APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  apptainer exec --nv --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$SIF" \
  python3 $W/flint_hf/train_check/grad_floor_check.py "$SEQS" > "$LOG" 2>&1
rc=$?
grep -v -E "Warning|warn" "$LOG" | tail -60
exit $rc
