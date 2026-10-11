#!/bin/bash
# One packed job (D9, owner 2026-10-07): the smoke copy of an oracle arm, then its first training segment, each through
# packed_task.sh (the record splits the job's H100-hours by its ledger). A failed smoke ends the job before training.
#   SMOKE_DIR, MAIN_DIR: EMIT dirs of submit_fcs_oracle.sh (run, env, args); MQB_CODE: the code worktree
#   JOB_WALL_S: the job's --time in seconds (default 24 h); the training task gets the rest minus 15 min, so its ledger
#   line is written before Slurm ends the job
#SBATCH --account=group
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/%x-%j.out
set -euo pipefail
T0=$(date +%s)
: "${SLURM_JOB_ID:?}" "${SMOKE_DIR:?}" "${MAIN_DIR:?}" "${MQB_CODE:?}"
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
PT=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
export PACKED_LEDGER_DIR=$B/logs/packed
task() {  # task NAME RECORD_RUN KIND LIMIT EMIT_DIR
  local -a E A; mapfile -t E < "$5/env"; mapfile -t A < "$5/args"
  echo "[packed] $(date -u +%FT%TZ) task $1 run $(cat "$5/run") limit $4"
  env "${E[@]}" MQB_CODE="$MQB_CODE" QTASK="$1" bash "$PT" "$1" "$2" "$3" 8 "$4" -- bash "$MQB_CODE/sbatch_miles.sh" "${A[@]}"
}
task smoke - other 100m "$SMOKE_DIR" || { echo "[packed] smoke failed: no training"; exit 1; }
task train "$(cat "$MAIN_DIR/run")" main "$(( ${JOB_WALL_S:-86400} - ($(date +%s) - T0) - 900 ))s" "$MAIN_DIR"
