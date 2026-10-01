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
JOBTMP=${TMPDIR:-/tmp}/miles_$SLURM_JOB_ID
mkdir -p "$JOBTMP"
trap 'rm -rf "$JOBTMP"' EXIT
echo "[sbatch] $(date -u +%FT%TZ) node=$(hostname) job=$SLURM_JOB_ID run=$RUN_NAME gpus=$GPUS"
apptainer exec --nv --contain --cleanenv --writable-tmpfs \
  --bind /scratch/gpfs/GROUP/USER \
  --bind "$JOBTMP":/tmp \
  --bind /dev/shm \
  --bind "$B/cache/root":/root/.cache \
  --env PYTHONNOUSERSITE=1 --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  --env SLURM_JOB_ID="$SLURM_JOB_ID" --env RUN_NAME="$RUN_NAME" --env GPUS="$GPUS" \
  --env MILES_MODEL_TYPE="$MILES_MODEL_TYPE" ${MILES_SRC:+--env MILES_SRC="$MILES_SRC"} \
  $(for v in $(compgen -e | grep "^MA_\|^FCS_"); do printf -- "--env %s=%q " "$v" "${!v}"; done) \
  "$B/miles.sif" bash "$B/run_miles_in_container.sh" "$@"
