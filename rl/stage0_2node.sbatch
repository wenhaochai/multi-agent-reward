#!/bin/bash
#SBATCH --job-name=flint-elastic0
#SBATCH --account=group
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=880G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/tmp/flint_s5/logs/%x-%j.out
# Elastic RL stage 0 as one 2-node job (controller 2026-10-30: wq has no gang scheduling): node 0 runs the external
# engine job, node 1 the trainer, started together by Slurm. Each role gets its own job id (FETCHED_JOB_ID
# <jobid>e / <jobid>t) so their TMPDIRs and registry records differ. Usage: sbatch stage0_2node.sbatch RUN_DIR
RUN=${1:?usage: stage0_2node.sbatch RUN_DIR}
E=$(cd "$(dirname "$0")" && pwd)
[ -f "$E/engine_job.sh" ] || E=/scratch/gpfs/GROUP/USER/project/mas-rl-elastic/rl
srun --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=96 --mem=0 --relative=0 --export=ALL,FETCHED_JOB_ID=${SLURM_JOB_ID}e bash "$E/engine_job.sh" "$RUN" &
pe=$!
sleep 60
srun --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=96 --mem=0 --relative=1 --export=ALL,FETCHED_JOB_ID=${SLURM_JOB_ID}t bash "$E/rl_job_ext.sh" smoke_ext "$RUN" &
pt=$!
wait $pt; rt=$?
echo "trainer exit $rt"
wait $pe; re=$?
echo "engine exit $re"
exit $rt
