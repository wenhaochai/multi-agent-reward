#!/bin/bash
#SBATCH --job-name=fcs-rescore
#SBATCH --account=group --partition=cpu --qos=short
#SBATCH --nodes=1 --cpus-per-task=16 --mem=48G --time=04:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/fcs_rescore-%j.out
set -euo pipefail
# re-judge probe answers with the current judge and the last-code-block rule: sbatch tools/rescore_probe.sbatch <probe dirs>
# judges = the job's cores - 1 (fcs_judge.job_cpus), one case at a time each (the wall limit is 2 x the time limit)
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
LT=/tmp/fcs_$SLURM_JOB_ID; mkdir -p $LT; trap 'rm -rf $LT' EXIT
apptainer exec --contain --cleanenv --bind /scratch/gpfs/GROUP/USER,$LT:/tmp \
  --env HOME=/tmp/home,FCS_JUDGE_CACHE=/tmp/fcsj_cache,PYTHONPATH=$B/pyfix,FCS_EXTRACT=last,FCS_CASE_WORKERS=1 \
  $B/miles.sif python3 $B/tools/rescore_probe.py "$@"
