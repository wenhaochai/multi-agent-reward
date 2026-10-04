#!/bin/bash
#SBATCH --job-name=fcs-rescore
#SBATCH --account=group --partition=cpu --qos=short
#SBATCH --nodes=1 --cpus-per-task=32 --mem=96G --time=03:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/fcs_rescore-%j.out
# re-judge the three Qwen3.5-9B probes with the current judge and the last-code-block rule
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
LT=/tmp/fcs_$SLURM_JOB_ID; mkdir -p $LT; trap 'rm -rf $LT' EXIT
apptainer exec --contain --cleanenv --bind /scratch/gpfs/GROUP/USER,$LT:/tmp \
  --env HOME=/tmp/home,FCS_JUDGE_CACHE=/tmp/fcsj_cache,PYTHONPATH=$B/pyfix,FCS_EXTRACT=last \
  $B/miles.sif python3 $B/tools/rescore_probe.py $B/runs/fcs_probe_qwen35_9b_sft $B/runs/fcs_probe_qwen35_9b \
  $B/runs/fcs_probe_qwen35_9b_nothink
