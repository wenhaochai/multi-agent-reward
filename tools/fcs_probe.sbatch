#!/bin/bash
#SBATCH --job-name=fcs-probe
#SBATCH --account=group --partition=pli-c --qos=pli-cp
#SBATCH --nodes=1 --gres=gpu:8 --cpus-per-task=64 --mem=500G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/fcs_probe-%j.out
# difficulty probe: base model on val172 x5 and train200 x4 (EasyPPO eval settings), judged locally
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
MODEL=${MODEL:-/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/Qwen3.5-9B}
OUT=${OUT:-$B/runs/fcs_probe_qwen35_9b}
apptainer exec --nv --contain --cleanenv --bind /scratch/gpfs/GROUP/USER \
  --env HOME=/tmp/home,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,FCS_JUDGE_CACHE=/tmp/fcsj_cache,PYTHONPATH=$B/pyfix \
  $B/miles.sif python3 $B/tools/fcs_probe.py $MODEL $OUT val:$B/data/fcs_val172.jsonl:5 train:$B/data/fcs_train200.jsonl:4
