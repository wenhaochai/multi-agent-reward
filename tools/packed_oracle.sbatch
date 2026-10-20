#!/bin/bash
# One packed job (D9, owner 2026-10-07): the smoke copy of an oracle arm, then its first training segment, each through
# packed_task.sh (the record splits the job's H100-hours by its ledger). A failed smoke ends the job before training.
#   SMOKE_DIR, MAIN_DIR: EMIT dirs of submit_fcs_oracle.sh (run, env, args); MQB_CODE: the code worktree
#   JOB_WALL_S: the training task's wall in seconds from job start (default 24 h); it gets the rest minus 15 min, so its
#   ledger line is written before Slurm ends the job
# Submit it through submit_packed_oracle.sh, which writes these settings into a copy of this script: fetch_pending.sh
# (owner 2026-10-15) runs a fetched job's batch script without its --export environment, in the host's environment.
# The job ends with fetch_pending.sh, which runs my pending fetchable jobs in the time left.
#SBATCH --account=group
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/%x-%j.out
set -euo pipefail
# a fetched copy without baked settings would read the host job's SMOKE_DIR and MAIN_DIR and rerun the host's run
[[ -n ${FETCHED_JOB_ID:-} && ${PACKED_BAKED:-} != 1 ]] && { echo "[packed] fetched without baked settings: refusing"; exit 1; }
T0=$(date +%s)
: "${SLURM_JOB_ID:?}" "${SMOKE_DIR:?}" "${MAIN_DIR:?}" "${MQB_CODE:?}"
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
PT=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
export PACKED_LEDGER_DIR=$B/logs/packed
NT=0
task() {  # task NAME RECORD_RUN KIND LIMIT EMIT_DIR
  # QTASK is a number: run_miles_in_container.sh puts it in the Ray port arithmetic (QTASK=smoke killed 15375871)
  local -a E A; mapfile -t E < "$5/env"; mapfile -t A < "$5/args"; NT=$((NT + 1))
  echo "[packed] $(date -u +%FT%TZ) task $1 run $(cat "$5/run") limit $4"
  env "${E[@]}" MQB_CODE="$MQB_CODE" QTASK="$NT" bash "$PT" "$1" "$2" "$3" 8 "$4" -- bash "$MQB_CODE/sbatch_miles.sh" "${A[@]}"
}
task smoke - other 100m "$SMOKE_DIR" || { echo "[packed] smoke failed: no training"; exit 1; }
# a fetched head (owner 2026-10-20: every job runs in a host) gets its smoke ledger line under its own job id, where its
# segments' smoke gate reads it: packed_task.sh writes ${FETCHED_JOB_ID:-$SLURM_JOB_ID}.tsv (controller, 2026-10-20). An
# earlier copy from the host's file here died under set -e in a pilot host, which has no ledger file
rc=0; task train "$(cat "$MAIN_DIR/run")" main "$(( ${JOB_WALL_S:-86400} - ($(date +%s) - T0) - 900 ))s" "$MAIN_DIR" || rc=$?
# PACKED_STRICT=1 (a producer that afterok jobs wait on): end with the training task's status, before fetch_pending,
# so a failed producer never releases its dependents and they do not wait for the host's leftover hours
[[ ${PACKED_STRICT:-} == 1 ]] && { echo "[packed] strict: exit $rc"; exit $rc; }
# a training task that crashed (not done, not its wall: 124) ends the host at once, so the next segment (afterany)
# resumes from the last checkpoint instead of waiting out the host's 24 h (review 2026-10-19)
if [[ $rc -ne 0 && $rc -ne 124 ]]; then echo "[packed] train task failed (rc $rc): no fetch_pending"; exit $rc; fi
# only the host fetches: a fetched copy that fetched again ran guests under its own --time and killed one (15686944
# inside 15625117 fetched 15696862, exit 143 at the guest's wall, 2026-10-16); the host's loop continues after it
if [[ -z ${FETCHED_JOB_ID:-} ]]; then bash /scratch/gpfs/GROUP/USER/tools/fetch_pending.sh; fi   # not a && list: its 1 would be the exit status
