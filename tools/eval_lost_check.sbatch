#!/bin/bash
#SBATCH --job-name=fcs-eval-lost
#SBATCH --account=group --partition=cpu --qos=short
#SBATCH --nodes=1 --cpus-per-task=1 --mem=1G --time=00:05:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/fcs_eval_lost-%j.out
set -euo pipefail
# After a training segment ends badly (submitted with --dependency=afternotok:<segment>; the next segment also waits on
# this job): if the run's newest checkpoint is an eval step and the segment's log has no eval for it, the segment died
# in that eval, so the next segment redoes it before training (runs/<run>/eval_before_train_once, run_miles_in_container.sh).
#   sbatch --dependency=afternotok:<segment> tools/eval_lost_check.sbatch <run name> <segment log> <eval interval>
R=$1; LOG=$2; EVERY=$3
D=/scratch/gpfs/GROUP/USER/project/miles-q38-build/runs/$R
K=$(cat $D/ckpt/latest_checkpointed_iteration.txt)
echo "run $R, newest checkpoint $K, eval every $EVERY"
if [ $(( (K + 1) % EVERY )) -eq 0 ] && ! grep -q "metrics.py:[0-9]* - eval $K:" "$LOG"; then
  touch $D/eval_before_train_once; echo "eval $K was lost: the next segment redoes it before training"
else
  echo "nothing to redo"
fi
