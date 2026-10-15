#!/bin/bash
# Submit packed_oracle.sbatch as a self-contained copy (owner rule 2026-10-15: 24 h packed jobs that end with
# fetch_pending.sh). SMOKE_DIR, MAIN_DIR, MQB_CODE and JOB_WALL_S are written into the copy as plain shell variables,
# because fetch_pending.sh runs a fetched batch script without its --export environment.
#   submit_packed_oracle.sh NAME SMOKE_DIR MAIN_DIR [TIME [JOB_WALL_S [sbatch args...]]]
#   TIME defaults to 24:00:00 and JOB_WALL_S to 86400; a JOB_WALL_S below TIME caps the main task and leaves the rest
#   of the 24 h to fetched jobs. Prints the job id.
set -euo pipefail
name=$1 smoke=$2 main=$3 time=${4:-24:00:00} wall=${5:-86400}; shift $(( $# < 5 ? $# : 5 ))
code=$(cd "$(dirname "$0")/.." && pwd)
for d in "$smoke" "$main"; do for f in run env args; do [[ -s $d/$f ]] || { echo "missing $d/$f" >&2; exit 1; }; done; done
dir=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/packed/scripts; mkdir -p "$dir"
out=$dir/$name.$(date +%Y%m%d-%H%M%S).sbatch
awk -v e="PACKED_BAKED=1 SMOKE_DIR=$smoke MAIN_DIR=$main MQB_CODE=$code JOB_WALL_S=$wall" \
  '{print} /^set -euo pipefail$/ && !done {print e; done=1}' "$code/tools/packed_oracle.sbatch" > "$out"
grep -q '^PACKED_BAKED=1 ' "$out"
sbatch --parsable --partition=pli-c --account=group --qos=pli-short --time="$time" --gres=gpu:8 --cpus-per-task=96 \
  --mem=900G --job-name="miles-$name" "$@" "$out"
