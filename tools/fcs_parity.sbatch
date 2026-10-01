#!/bin/bash
#SBATCH --job-name=fcs-parity
#SBATCH --account=group --qos=short --partition=cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G
#SBATCH --time=01:01:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/fcs_parity-%j.out
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
W=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace
LT=/tmp/fcs_$SLURM_JOB_ID; mkdir -p $LT  # container /tmp under --contain is a tiny tmpfs (parity 14849538: ENOSPC)
trap 'rm -rf $LT' EXIT
apptainer exec --contain --cleanenv --bind /scratch/gpfs/GROUP/USER,$LT:/tmp --env FCS_JUDGE_CACHE=/tmp/fcsj_cache \
  $W/sif/molt.sif python3 $B/tools/fcs_parity.py $B/logs/fcs_parity.jsonl 25 8
