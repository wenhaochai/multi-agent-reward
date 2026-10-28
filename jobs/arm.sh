#!/bin/bash
#SBATCH --job-name=flint-arm
#SBATCH --gres=gpu:8
#SBATCH --mem=480G
#SBATCH --time=23:00:00
#SBATCH --comment=fetchable-partial
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
#FETCHABLE
# mas-rl: one arm of the main comparison on val40 (wq --partial). Usage: arm.sh SETUP BUDGET, BUDGET the generated
# tokens per problem (500k, 1m or 5m):
#   solo   one Flint with the whole budget
#   apart  four independent Flints per problem, a quarter each (4 episodes; the best provisional is picked offline)
#   team   four Flints with a quarter each, a repository each, must post after each submission
# Step 1 (once per job): a smoke of the same setup, unless smoke_wq.sh already passed it for this job
# (runs/smoke_pass_w<n>): on all 8 GPUs at TP8 (no GPU idles), 2 episodes at 150k tokens per agent; a fail exits 1.
# Step 2: eval.sh on all 8 GPUs.
set -uo pipefail
FE=/scratch/gpfs/GROUP/USER/project/flint-eval
JOBS=/scratch/gpfs/GROUP/USER/project/miles-q38-build-clean/jobs
case "${2:-}" in 500k) B=500000;; 1m) B=1000000;; 5m) B=5000000;; *) echo "usage: arm.sh solo|apart|team 500k|1m|5m"; exit 1;; esac
case "${1:-}" in
  solo)  SM="agents=1"; EV="val40 $B 1";;
  apart) SM="agents=1"; EV="val40 $((B / 4)) 4";;
  team)  SM="agents=4"; EV="val40 $((B / 4)) 1 agents=4 conc=8";;
  *) echo "usage: arm.sh solo|apart|team 500k|1m|5m"; exit 1;;
esac
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
[ -n "$JID" ] || { echo "no job id"; exit 1; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:?no GPUs}"
[ "${#G[@]}" -eq 8 ] || { echo "need 8 GPUs, got ${G[*]}"; exit 1; }
MARK=$FE/runs/smoke_pass_${JID}
if [ ! -e "$MARK" ]; then
  L=$FE/logs/flint-arm-$1-$2-smoke-$JID.log
  FETCHED_JOB_ID=${JID}9 bash $JOBS/smoke.sh $SM n=2 tokens=150000 > "$L" 2>&1
  r=$?
  grep -E "SMOKE_|POST rule|driver exit" "$L" | tail -4
  [ $r -eq 0 ] || { echo "SMOKE_FAIL $1 $2 exit $r, log $L"; exit 1; }
  touch "$MARK"
fi
exec bash $JOBS/eval.sh $EV
