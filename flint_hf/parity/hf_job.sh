#!/bin/bash
#SBATCH --job-name=flint-parity-hf
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
# HF half of the parity check, after sglang_job.sh: flint_hf in a manual pipeline over the 4 GPUs (hf_score.py: logprobs
# against SGLang A's heads, a backward on 8k tokens, from_pretrained), then the table (compare.py) in OUTDIR/table.txt.
# Usage: hf_job.sh OUTDIR
set -uo pipefail
P=/scratch/gpfs/GROUP/USER/project
W=$P/mas-rl-flintrl
FSIF=$P/flint-eval/flint-sglang.sif
OUT=${1:?outdir}
SEQS=/scratch/gpfs/GROUP/USER/tmp/flint_parity_seqs.json
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
[ -f "$OUT/sglang_A.pt" ] || { echo "no $OUT/sglang_A.pt"; exit 1; }
echo "job $JID host $(hostname) commit $(git -C $W rev-parse HEAD) GPUs $CUDA_VISIBLE_DEVICES"
run() { APPTAINERENV_PYTHONPATH=$W APPTAINERENV_TMPDIR=$TMPDIR apptainer exec --nv --home "$TMPDIR" \
          --bind /scratch/gpfs/GROUP "$FSIF" python3 "$@"; }
run $W/flint_hf/parity/hf_score.py "$SEQS" "$OUT/sglang_A.pt" "$OUT/hf.pt" 2>&1 | grep -v "model of type" \
  | tee "$OUT/hf.log"
[ -f "$OUT/hf.pt" ] || { echo "PARITY_HF_FAIL"; exit 1; }
run $W/flint_hf/parity/compare.py "$SEQS" "$OUT" | tee "$OUT/table.txt"
