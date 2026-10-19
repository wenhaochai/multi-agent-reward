#!/bin/bash
# Submit the EMIT dirs of make_oracle_seed.sh (user 2026-10-19 popup: a second seed of the five oracle arms):
#   1. the team value-pretraining producer: one packed job (its smoke copy, then the 11-rollout producer), strict mode
#      (ends with the producer's status, no fetch_pending), --no-fetchable: team shared and bonus wait on it afterok
#   2. per arm, the packed 24 h host (its smoke copy, then segment 1, then fetch_pending), --no-fetchable because it heads
#      the chain; runs/<run>/smoke_job names it, so every later segment is gated on its smoke ledger line
#   3. per arm, NSEG resume segments: self-contained, fetchable-partial (owner 2026-10-19), afterany on the previous job
#      (a segment that finds the run complete exits at once)
#   bash tools/submit_oracle_seed.sh EMIT_ROOT [DRY=1]
set -euo pipefail
O=$(cd "$1" && pwd)
code=$(cd "$(dirname "$0")/.." && pwd)
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
run() { if [[ -n ${DRY:-} ]]; then echo "DRY: $*" >&2; echo 0; else "$@"; fi; }
for a in vp team_shared team_bonus team_diff_cfk4 seq_shared par_indiv; do
  n=$(cat "$O/$a/main/run")
  [[ -e $B/runs/$n/ckpt/latest_checkpointed_iteration.txt || -e $B/runs/$n/smoke_job ]] && { echo "$n was submitted before" >&2; exit 1; }
done
# producer: s42's took 7.3 h with a step-0 eval this one skips; 10 h wall, the producer task gets 10 h - smoke - 15 min
vp=$(cat "$O/vp/main/run")
prod=$(run env PACKED_STRICT=1 bash "$code/tools/submit_packed_oracle.sh" "$vp" "$O/vp/smoke" "$O/vp/main" 10:00:00 36000 --no-fetchable)
[[ -n ${DRY:-} ]] || echo "$prod" > "$B/runs/$vp/smoke_job"
echo "producer $vp $prod"
for spec in team_shared:2:prod team_bonus:2:prod team_diff_cfk4:3: seq_shared:2: par_indiv:2:; do
  IFS=: read -r a nseg dep <<< "$spec"
  n=$(cat "$O/$a/main/run")
  extra=(); [[ $dep == prod ]] && extra=(--dependency=afterok:$prod)
  host=$(run bash "$code/tools/submit_packed_oracle.sh" "$n" "$O/$a/smoke" "$O/$a/main" 24:00:00 86400 --no-fetchable "${extra[@]}")
  [[ -n ${DRY:-} ]] || echo "$host" > "$B/runs/$n/smoke_job"
  seg=$(bash "$code/tools/make_agent_segment.sh" "$O/$a/main")
  prev=$host segs=()
  for _ in $(seq "$nseg"); do
    prev=$(run sbatch --parsable --partition=pli-c --account=group --qos=pli-short --time=24:00:00 --comment=fetchable-partial \
      --dependency=afterany:$prev --gres=gpu:8 --cpus-per-task=96 --mem=900G --job-name="miles-$n" "$seg")
    segs+=("$prev")
  done
  echo "$a $n host $host segments ${segs[*]} script $seg"
done
