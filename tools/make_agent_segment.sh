#!/bin/bash
# Write a self-contained batch script for one training segment of an EMIT main dir (user 2026-10-19: fetchable-partial
# segments). It reads env and args from the EMIT dir at run time, so a host's fetch_pending.sh can run it in a clean
# login env; the run resumes from its latest checkpoint. Prints the script path.
#   bash tools/make_agent_segment.sh MAIN_DIR
set -euo pipefail
M=$(cd "$1" && pwd) code=$(cd "$(dirname "$0")/.." && pwd)
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
[[ -s $M/env && -s $M/args && -s $M/run ]]
seg=$B/logs/packed/scripts/$(cat "$M/run").seg.$(date +%Y%m%d-%H%M%S).sbatch
cat > "$seg" <<SEG
#!/bin/bash
#FETCHABLE: agentic training segment (fetchable-partial); settings baked in by make_agent_segment.sh, no --export
#SBATCH --account=group
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=$B/logs/%x-%j.out
set -euo pipefail
M=$M CODE=$code
mapfile -t E < "\$M/env"; mapfile -t A < "\$M/args"
exec env "\${E[@]}" MQB_CODE="\$CODE" bash "\$CODE/sbatch_miles.sh" "\${A[@]}"
SEG
echo "$seg"
