#!/bin/bash
#SBATCH --job-name=fcs-resume-gate
#SBATCH --account=group --partition=cpu --qos=short
#SBATCH --nodes=1 --cpus-per-task=2 --mem=16G --time=00:20:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/fcs_resume_gate-%j.out
set -euo pipefail
# the actor resume-fix gate: submitted with --dependency=afterok:<resume smoke>; the resumed Frontier-CS chains depend
# afterok on this job. Checks (tools/resume_fix_gate.py, in miles.sif), then fast-forwards miles-easyppo to the fix.
#   sbatch --dependency=afterok:<smoke> tools/resume_fix_gate.sbatch <smoke log> <saved actor ckpt iter dir> <fix sha>
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
apptainer exec --contain --cleanenv --bind /scratch/gpfs/GROUP/USER $B/miles.sif python3 $B/tools/resume_fix_gate.py "$@"
W=/scratch/gpfs/GROUP/USER/project/miles-easyppo
git -C $W merge --ff-only "$3"
test "$(git -C $W rev-parse HEAD)" = "$(git -C $W rev-parse "$3")"
echo "miles-easyppo now at $(git -C $W log --oneline -1)"
