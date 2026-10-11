#!/bin/bash
#SBATCH --job-name=miles-fcs_easyppo_smoke_q9b-fcs-sft
#SBATCH --account=group --partition=pli-c --qos=pli-cp --time=01:30:00
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=64 --mem=640G
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/miles-fcs_easyppo_smoke_q9b-fcs-sft-%j.out
set -euo pipefail
# /tmp on a compute node is small and shared (owner rule 2026-10-08): this job's temp files, compiles and caches go to
# a job dir on /scratch, bound into any container as TMPDIR, removed at exit
: "${SLURM_JOB_ID:?}"   # an empty id would point TMPDIR and the EXIT rm at tmp/job/ itself
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID; export APPTAINERENV_TMPDIR=$TMPDIR
mkdir -p "$TMPDIR"; trap 'rm -rf "/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID" || true' EXIT
# Gate for eval judged after training resumes (miles_team.fcs_rm.log_eval; packing rule 2026-10-07: smoke and gate in
# one job, each task through packed_task.sh, ledger logs/packed/<job>.tsv). Task 1: the solo EasyPPO smoke
# (submit_fcs.sh SMOKE=1, 4 rollouts, evals after rollouts 1 and 3) with the candidate code (MQB_CODE), the eval set
# with fcs_defer_score and the two log functions, W&B offline. Task 2: tools/eval_defer_gate.py on its log and run dir.
# On PASS: main is fast-forwarded to the candidate and the two lastblock baselines get the three flags in
# runs/<run>/extra_args (read at their next segment). The smoke's checkpoints are deleted either way.
#   MQB_CODE=<candidate worktree> sbatch tools/eval_defer_smoke.sbatch <candidate sha>
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
T=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
: "${MQB_CODE:?}"; CAND=$1
R=fcs_easyppo_smoke_q9b-fcs-sft
LOG=$B/logs/miles-$R-$SLURM_JOB_ID.out
DEFER=(--eval-prompt-data fcs_val $B/data/fcs_val172_defer.jsonl
       --custom-eval-rollout-log-function-path miles_team.fcs_rm.log_eval
       --custom-rollout-log-function-path miles_team.fcs_rm.log_rollout)
# the miles the Frontier-CS runs use (submit_fcs.sh); without it the launcher takes the image's miles, which lacks the
# EasyPPO flags (job 15182362 failed at argparse)
MILES_SRC=/scratch/gpfs/GROUP/USER/project/miles-easyppo
MILES_SHA=$(git -C $MILES_SRC rev-parse HEAD)
export RUN_NAME=$R GPUS=8 MILES_MODEL_TYPE=qwen3.5-9B MQB_CODE MILES_SRC
export FCS_RM_CONCURRENCY=56 FCS_CASE_WORKERS=1 FCS_CASE_POOL=1
mapfile -t ARGS < $B/runs/$R/smoke_args
rc=0
$T smoke - other 8 60m -- bash $MQB_CODE/sbatch_miles.sh "${ARGS[@]}" "${DEFER[@]}" \
  --use-wandb --wandb-mode offline --wandb-dir $B/runs/$R --wandb-project fcs_easyppo_smoke --wandb-run-id $R-$SLURM_JOB_ID \
  --disable-wandb-random-suffix || rc=$?
# the gate re-judges eval 1: its compiles need a real /tmp (under --contain, /tmp is apptainer's small session tmpfs;
# job 15204506 ran out of space there)
GT=$TMPDIR/eval_defer_gate; mkdir -p "$GT"
[ $rc -eq 0 ] && { $T gate - other 0 15m -- apptainer exec --contain --cleanenv --bind /scratch/gpfs/GROUP/USER \
  --bind "$GT":/tmp --env MQB_CODE=$MQB_CODE $B/miles.sif python3 $MQB_CODE/tools/eval_defer_gate.py "$LOG" $B/runs/$R "$CAND" "$MILES_SHA" || rc=$?; }
rm -r "$TMPDIR/eval_defer_gate" 2>/dev/null || true
rm -r "$B/runs/fcs_easyppo_smoke_q9b-fcs-sft/ckpt" "$B/runs/fcs_easyppo_smoke_q9b-fcs-sft/ckpt_critic" 2>/dev/null || true
[ $rc -eq 0 ] || { echo "eval-defer gate failed (rc=$rc): main and the baselines unchanged"; exit $rc; }
git -C $B merge --ff-only "$CAND"
test "$(git -C $B rev-parse HEAD)" = "$(git -C $B rev-parse "$CAND")"
for r in fcs_easyppo_lastblock_q9b-fcs-sft_s42 fcs_easyppo_lastblock_q9b-fcs-sft_s50; do
  f=$B/runs/$r/extra_args
  grep -q "fcs_rm.log_eval" "$f" || { echo "# 2026-10-07 (owner): eval judged after training resumes (miles_team.fcs_rm.log_eval, gate job $SLURM_JOB_ID)"; printf '%s ' "${DEFER[@]}"; echo; } >> "$f"
done
echo "main now at $(git -C $B log --oneline -1); lastblock s42/s50 judge their evals in the background from their next segment"
