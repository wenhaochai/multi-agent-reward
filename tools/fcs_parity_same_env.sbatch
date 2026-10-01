#!/bin/bash
#SBATCH --job-name=fcs-parity2
#SBATCH --account=group --qos=short --partition=cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/fcs_parity2-%j.out
# official judge (host go-judge + node) and the local judge on the same node, one after the other
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
G=/scratch/gpfs/GROUP/USER/project/gojudge
W=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace
GJP=$((20000 + SLURM_JOB_ID % 5000)); PORT=$((GJP + 5000))
GJP=$GJP PORT=$PORT JUDGE_WORKERS=4 GJ_PARALLELISM=16 $G/serve_inner.sh > $B/logs/fcs_parity2_serve-$SLURM_JOB_ID.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do curl -s 127.0.0.1:$PORT/problems > /dev/null && break; sleep 2; done
grep -i "cgroup\|fallback\|fail" $G/gj_$GJP.log | head -5
LT=/tmp/fcs_$SLURM_JOB_ID; mkdir -p $LT; trap 'rm -rf $LT; pkill -P $SRV; kill $SRV' EXIT
apptainer exec --contain --cleanenv --bind /scratch/gpfs/GROUP/USER,$LT:/tmp --env FCS_JUDGE_CACHE=/tmp/fcsj_cache \
  $W/sif/molt.sif python3 $B/tools/fcs_parity_same_env.py $B/logs/fcs_parity.jsonl $B/logs/fcs_parity2.jsonl $PORT 4
