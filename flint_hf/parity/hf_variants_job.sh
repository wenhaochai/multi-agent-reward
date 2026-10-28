#!/bin/bash
#SBATCH --job-name=flint-parity-variants
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
# hf_variants.py on 4 GPUs (flint_hf's manual pipeline). Usage: hf_variants_job.sh OUTDIR
set -uo pipefail
P=/scratch/gpfs/GROUP/USER/project
W=$P/mas-rl-flintrl
OUT=${1:?outdir}
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
echo "job $JID host $(hostname) commit $(git -C $W rev-parse HEAD) GPUs $CUDA_VISIBLE_DEVICES"
APPTAINERENV_PYTHONPATH=$W APPTAINERENV_TMPDIR=$TMPDIR apptainer exec --nv --home "$TMPDIR" \
  --bind /scratch/gpfs/GROUP "$P/flint-eval/flint-sglang.sif" \
  python3 $W/flint_hf/parity/hf_variants.py /scratch/gpfs/GROUP/USER/tmp/flint_parity_seqs.json "$OUT" 2>&1 \
  | grep --line-buffered -v "model of type" | tee "$OUT/variants.txt"
