#!/bin/bash
#SBATCH --job-name=fcs-nopad-gate
#SBATCH --account=group --partition=cpu --qos=short
#SBATCH --nodes=1 --cpus-per-task=1 --mem=2G --time=00:10:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/fcs_nopad_gate-%j.out
# the no-pad gate (tools/nopad_gate.py): submitted with --dependency=afterok:<smoke B>; the held Frontier-CS chains
# depend afterok on this job, so a failure here cancels them (kill_invalid_depend)
#   sbatch --dependency=afterok:<smoke B> tools/nopad_gate.sbatch <smoke A log> <smoke B log>
set -euo pipefail
exec python3 /scratch/gpfs/GROUP/USER/project/miles-q38-build/tools/nopad_gate.py "$@"
