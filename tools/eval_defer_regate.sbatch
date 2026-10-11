#!/bin/bash
#SBATCH --job-name=eval-defer-regate
#SBATCH --account=group --qos=short --partition=cpu --time=00:45:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=64G
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/eval-defer-regate-%j.out
set -euo pipefail
# /tmp on a compute node is small and shared (owner rule 2026-10-08): this job's temp files, compiles and caches go to
# a job dir on /scratch, bound into any container as TMPDIR, removed at exit
: "${SLURM_JOB_ID:?}"   # an empty id would point TMPDIR and the EXIT rm at tmp/job/ itself
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID; export APPTAINERENV_TMPDIR=$TMPDIR
mkdir -p "$TMPDIR"; trap 'rm -rf "/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID" || true' EXIT
# The eval-defer gate again, on a CPU node, over the outputs of a smoke whose own part passed and whose gate failed in
# the gate itself (job 15204506: the gate's re-judge ran out of apptainer's session /tmp, and it expected 172 kept
# replies and a kept file for the inline last eval). RAN is the candidate the smoke ran (its code line); CAND is the
# commit main moves to, which differs from RAN only in the gate tools (checked here). On PASS: as
# tools/eval_defer_smoke.sbatch (main fast-forwards to CAND; lastblock s42/s50 get the three flags).
#   MQB_CODE=<candidate worktree> sbatch tools/eval_defer_regate.sbatch <smoke job id> <RAN sha> <CAND sha>
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
: "${MQB_CODE:?}"; SMOKE=$1; RAN=$2; CAND=$3
R=fcs_easyppo_smoke_q9b-fcs-sft
LOG=$B/logs/miles-$R-$SMOKE.out
changed=$(git -C "$MQB_CODE" diff --name-only "$RAN" "$CAND" | grep -v -E '^tools/eval_defer_(gate\.py|smoke\.sbatch|regate\.sbatch)$' || true)
[ -z "$changed" ] || { echo "CAND differs from the smoke's code beyond the gate tools: $changed"; exit 1; }
MILES_SHA=$(sed -n 's/.*\[miles-run\] code: miles \([0-9a-f]*\) .*/\1/p' "$LOG" | head -1)
test "$(git -C /scratch/gpfs/GROUP/USER/project/miles-easyppo rev-parse HEAD | cut -c1-${#MILES_SHA})" = "$MILES_SHA"
DEFER=(--eval-prompt-data fcs_val $B/data/fcs_val172_defer.jsonl
       --custom-eval-rollout-log-function-path miles_team.fcs_rm.log_eval
       --custom-rollout-log-function-path miles_team.fcs_rm.log_rollout)
GT=$TMPDIR/eval_defer_gate; mkdir -p "$GT"
rc=0
apptainer exec --contain --cleanenv --bind /scratch/gpfs/GROUP/USER --bind "$GT":/tmp --env MQB_CODE=$MQB_CODE \
  $B/miles.sif python3 $MQB_CODE/tools/eval_defer_gate.py "$LOG" $B/runs/$R "$RAN" "$MILES_SHA" || rc=$?
rm -r "$GT" 2>/dev/null || true
[ $rc -eq 0 ] || { echo "eval-defer gate failed (rc=$rc): main and the baselines unchanged"; exit $rc; }
git -C $B merge --ff-only "$CAND"
test "$(git -C $B rev-parse HEAD)" = "$(git -C $B rev-parse "$CAND")"
for r in fcs_easyppo_lastblock_q9b-fcs-sft_s42 fcs_easyppo_lastblock_q9b-fcs-sft_s50; do
  f=$B/runs/$r/extra_args
  grep -q "fcs_rm.log_eval" "$f" || { echo "# 2026-10-08 (owner): eval judged after training resumes (miles_team.fcs_rm.log_eval, gate job $SLURM_JOB_ID on smoke $SMOKE)"; printf '%s ' "${DEFER[@]}"; echo; } >> "$f"
done
echo "main now at $(git -C $B log --oneline -1); lastblock s42/s50 judge their evals in the background from their next segment"
