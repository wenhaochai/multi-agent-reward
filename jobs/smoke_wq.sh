#!/bin/bash
#SBATCH --job-name=flint-smoke-wq
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=02:45:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
#FETCHABLE
# mas-rl: step 1 of arm.sh as its own 4-GPU wq job (one TP4 server; a 2-GPU job strands half a block), so the arms
# skip their in-job smoke. Usage: smoke_wq.sh solo|apart|team w<a> [w<b> ...], the queued arm.sh jobs of that setup.
# A pass leaves runs/smoke_pass_w<n> for each; a fail cancels those still queued.
set -uo pipefail
FE=/scratch/gpfs/GROUP/USER/project/flint-eval
JOBS=/scratch/gpfs/GROUP/USER/project/miles-q38-build-clean/jobs
W=/scratch/gpfs/GROUP/USER/tools/wq/wq
case "${1:-}" in solo|apart) SM="agents=1";; team) SM="agents=4";; *) echo "usage: smoke_wq.sh solo|apart|team w<n> ..."; exit 1;; esac
shift
[ $# -ge 1 ] || { echo "no target"; exit 1; }
for t in "$@"; do [[ "$t" =~ ^w[0-9]+$ ]] || { echo "bad target $t"; exit 1; }; done
bash $JOBS/smoke.sh $SM n=2 tokens=150000
r=$?
if [ $r -eq 0 ]; then
  for t in "$@"; do touch $FE/runs/smoke_pass_$t; done; echo "SMOKE_PASS for $*"; exit 0
fi
echo "SMOKE_FAIL exit $r: cancel the queued targets"
for t in "$@"; do [ "$($W state $t)" = queued ] && $W cancel $t; done
exit 1
