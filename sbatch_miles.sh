#!/bin/bash
# Host-side SLURM wrapper for a miles run on Della: RUN_NAME, GPUS, MILES_MODEL_TYPE from the env; train args forwarded.
# MA_* env vars (the team game's knobs, miles_team/team_core.py) and FCS_* (the Frontier-CS judge, miles_team/fcs_rm.py)
# pass through --cleanenv to the Ray workers.
#SBATCH --account=group
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/%x-%j.out
set -euo pipefail
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
: "${RUN_NAME:?}" "${GPUS:?}" "${MILES_MODEL_TYPE:?}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
# /tmp on a compute node is small and shared (owner rule 2026-10-08): the job's temp dir is on /scratch, bound into the
# container as TMPDIR (run_miles_in_container.sh puts the judge and the caches there). The container's /tmp holds only
# Ray's session dir (unix socket paths must stay < 108 bytes); it is a per-job dir on the node, removed at exit too
[ -n "$SLURM_JOB_ID" ]
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID${QTASK:+_q$QTASK}   # QTASK: a queue-worker task
export APPTAINERENV_TMPDIR=$TMPDIR
JOBTMP=/tmp/miles_$SLURM_JOB_ID${QTASK:+_q$QTASK}
mkdir -p "$TMPDIR" "$JOBTMP"
trap 'rm -rf "/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID${QTASK:+_q$QTASK}" "/tmp/miles_$SLURM_JOB_ID${QTASK:+_q$QTASK}" || true' EXIT
echo "[sbatch] $(date -u +%FT%TZ) node=$(hostname) job=$SLURM_JOB_ID run=$RUN_NAME gpus=$GPUS"
# MA_*/FCS_* reach the container as APPTAINERENV_* (passes --cleanenv verbatim; --env %q turned "0,1,3" into "0\,1\,3",
# audit 2026-10-11); each is logged, since a variable exported in the submitting shell also lands here
for v in $(compgen -e | grep "^MA_\|^FCS_"); do export "APPTAINERENV_$v=${!v}"; echo "[sbatch] env $v=${!v}"; done
apptainer exec --nv --contain --cleanenv --writable-tmpfs \
  --bind /scratch/gpfs/GROUP/USER \
  --bind "$JOBTMP":/tmp --bind "$TMPDIR" \
  --bind /dev/shm \
  --bind "$B/cache/root":/root/.cache \
  --env PYTHONNOUSERSITE=1 --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  --env SLURM_JOB_ID="$SLURM_JOB_ID" --env RUN_NAME="$RUN_NAME" --env GPUS="$GPUS" \
  --env MILES_MODEL_TYPE="$MILES_MODEL_TYPE" ${MILES_SRC:+--env MILES_SRC="$MILES_SRC"} ${QTASK:+--env QTASK="$QTASK"} \
  ${MQB_CODE:+--env MQB_CODE="$MQB_CODE"} "$B/miles.sif" bash "${MQB_CODE:-$B}/run_miles_in_container.sh" "$@"
