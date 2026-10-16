#!/bin/bash
# Submit the agentic single-agent RL run (owner step 2, 2026-10-16) from the EMIT dirs of tools/make_agent_train.py:
#   1. a quick smoke on pli-cp under its own run name (a copy of the smoke EMIT with "_cp" in the name, so it never
#      shares a run dir with the packed job's smoke copy)
#   2. the packed 24 h host job on pli-short (tools/packed_oracle.sbatch through submit_packed_oracle.sh: its own smoke
#      copy, then the first training segment, then fetch_pending.sh); runs/<run>/smoke_job names it, so every later
#      segment is gated on its smoke ledger line (run_miles_in_container.sh)
#   3. NSEG resume segments, sbatch_miles.sh afterany on the previous job (a segment that finds the run complete exits)
#   bash tools/submit_agent_solo.sh EMIT_DIR [NSEG=3]   (review 2026-10-16: ~29M training tokens per rollout; a segment
#   that finds the run complete exits at once)
set -euo pipefail
O=$1 nseg=${2:-3}
code=$(cd "$(dirname "$0")/.." && pwd)
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
name=$(cat "$O/main/run") smoke=$(cat "$O/smoke/run")
[[ -e $B/runs/$name/ckpt/latest_checkpointed_iteration.txt ]] && { echo "$name already has a checkpoint" >&2; exit 1; }
cp_dir=$O/smoke_cp
if [[ ! -d $cp_dir ]]; then
  mkdir -p "$cp_dir"
  for f in run env args; do sed "s/$smoke/${smoke}_cp/g" "$O/smoke/$f" > "$cp_dir/$f"; done
fi
mkdir -p "$B/runs/${smoke}_cp" && cp -n "$B/runs/$smoke/extra_env" "$B/runs/${smoke}_cp/extra_env"   # NVLS off, judge pool
mapfile -t E < "$cp_dir/env"; mapfile -t A < "$cp_dir/args"
cp_job=$(env "${E[@]}" MQB_CODE="$code" sbatch --parsable --partition=pli-c --account=group --qos=pli-cp --time=01:45:00 \
  --gres=gpu:8 --cpus-per-task=96 --mem=900G --job-name="miles-$(cat "$cp_dir/run")" "$code/sbatch_miles.sh" "${A[@]}")
host=$(bash "$code/tools/submit_packed_oracle.sh" "$name" "$O/smoke" "$O/main")
mkdir -p "$B/runs/$name" && echo "$host" > "$B/runs/$name/smoke_job"
mapfile -t E < "$O/main/env"; mapfile -t A < "$O/main/args"
prev=$host segs=()
for _ in $(seq "$nseg"); do
  prev=$(env "${E[@]}" MQB_CODE="$code" sbatch --parsable --partition=pli-c --account=group --qos=pli-short --time=24:00:00 \
    --dependency=afterany:$prev --gres=gpu:8 --cpus-per-task=96 --mem=900G --job-name="miles-$name" \
    "$code/sbatch_miles.sh" "${A[@]}")
  segs+=("$prev")
done
echo "smoke_cp $cp_job host $host segments ${segs[*]}"
