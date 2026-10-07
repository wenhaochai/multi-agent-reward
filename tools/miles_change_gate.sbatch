#!/bin/bash
#SBATCH --job-name=miles-fcs_oracle_smoke_team_diff_q9b-fcs-sft_nopad
#SBATCH --account=group --partition=pli-c --qos=pli-cp --time=01:00:00
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=64 --mem=640G
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/miles-fcs_oracle_smoke_team_diff_q9b-fcs-sft_nopad-%j.out
set -euo pipefail
# Gate for a change to the miles worktree the Frontier-CS chains import (packing rule 2026-10-07: smoke and gate in one
# job, each task through packed_task.sh, ledger logs/packed/<job>.tsv). Task 1: the no-pad team smoke resumes from its
# own checkpoint with the candidate miles (MILES_SRC) and trains one live rollout. Task 2: tools/resume_fix_gate.py on
# this job's log (the candidate ran, the first step after the resume is sane, the saved actor checkpoint is nonzero).
# Then miles-easyppo is fast-forwarded to the candidate; on any failure it stays where it is.
#   MILES_SRC=<candidate worktree> sbatch tools/miles_change_gate.sbatch <candidate sha> <checkpoint iteration it saves>
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
T=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
FIX=$1; IT=$2
R=fcs_oracle_smoke_team_diff_q9b-fcs-sft_nopad
LOG=$B/logs/miles-$R-$SLURM_JOB_ID.out
export RUN_NAME=$R GPUS=8 MILES_MODEL_TYPE=qwen3.5-9B MILES_SRC
export FCS_RM_CONCURRENCY=56 FCS_CASE_WORKERS=1 FCS_CASE_POOL=1
export MA_FO_GAME=team MA_FO_REWARD=diff MA_FO_SUBS=4 MA_FO_ROUNDS=5 MA_FO_BUDGET=4096 MA_FO_PLAN_BUDGET=2048 \
  MA_FO_SUB_TESTS=3 MA_FO_CF=0 MA_FO_REPORT_BUDGET=4096 MA_FO_MAX_LEN=36864 MA_FO_TRACE_DIR=$B/runs/$R/traces MA_FO_TRACE_EVERY=32
mapfile -t ARGS < $B/runs/$R/resume_gate_args
$T smoke - other 8 45m -- bash $B/sbatch_miles.sh "${ARGS[@]}"
$T gate - other 0 10m -- apptainer exec --contain --cleanenv --bind /scratch/gpfs/GROUP/USER $B/miles.sif \
  python3 $B/tools/resume_fix_gate.py "$LOG" $B/runs/$R/ckpt/iter_$(printf %07d "$IT") "$FIX" "$IT"
W=/scratch/gpfs/GROUP/USER/project/miles-easyppo
git -C $W merge --ff-only "$FIX"
test "$(git -C $W rev-parse HEAD)" = "$(git -C $W rev-parse "$FIX")"
echo "miles-easyppo now at $(git -C $W log --oneline -1)"
