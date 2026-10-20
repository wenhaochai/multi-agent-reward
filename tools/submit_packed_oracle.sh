#!/bin/bash
# Submit packed_oracle.sbatch as a self-contained copy (owner rule 2026-10-15: 24 h packed jobs that end with
# fetch_pending.sh). SMOKE_DIR, MAIN_DIR, MQB_CODE and JOB_WALL_S are written into the copy as plain shell variables,
# because fetch_pending.sh runs a fetched batch script without its --export environment.
#   submit_packed_oracle.sh NAME SMOKE_DIR MAIN_DIR [TIME [JOB_WALL_S [sbatch args...]]]
#   TIME defaults to 24:00:00 and JOB_WALL_S to 86400; a JOB_WALL_S below TIME caps the main task and leaves the rest
#   of the 24 h to fetched jobs. Prints the job id.
#   Fetchable by default (owner rule 2026-10-17): the job gets --comment=fetchable and the copy the #FETCHABLE line
#   that fetch_pending.sh requires. Pass --no-fetchable for a job that heads a dependency chain (its segments wait on
#   its id and gate on its ledger, runs/<run>/smoke_job, which a fetched run writes under the host's id).
set -euo pipefail
fetch=1 comment=fetchable; args=()
for a in "$@"; do case $a in --no-fetchable) fetch=0 ;; --partial) comment=fetchable-partial ;; *) args+=("$a") ;; esac; done
set -- "${args[@]}"
name=$1 smoke=$2 main=$3 time=${4:-24:00:00} wall=${5:-86400}; shift $(( $# < 5 ? $# : 5 ))
code=$(cd "$(dirname "$0")/.." && pwd)
for d in "$smoke" "$main"; do for f in run env args; do [[ -s $d/$f ]] || { echo "missing $d/$f" >&2; exit 1; }; done; done
dir=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/packed/scripts; mkdir -p "$dir"
out=$dir/$name.$(date +%Y%m%d-%H%M%S).sbatch
awk -v e="PACKED_BAKED=1 SMOKE_DIR=$smoke MAIN_DIR=$main MQB_CODE=$code JOB_WALL_S=$wall" \
  '{print} /^set -euo pipefail$/ && !done {print e; done=1}' "$code/tools/packed_oracle.sbatch" > "$out"
grep -q '^PACKED_BAKED=1 ' "$out"
if (( fetch )); then
  sed -i '1a #FETCHABLE: settings baked in by submit_packed_oracle.sh, no --export' "$out"
  set -- "$@" --comment=$comment
fi
sbatch --parsable --partition=pli-c --account=group --qos=pli-short --time="$time" --gres=gpu:8 --cpus-per-task=96 \
  --mem=900G --job-name="miles-$name" "$@" "$out"
