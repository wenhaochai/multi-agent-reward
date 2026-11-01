#!/bin/bash
# Off-GPU checks of every RL mode's train.py argv (rl/rl_inner.sh args: run20's and the three train67 ones), the
# train67 schedule of one B-episode step, in miles-v0.1.3.sif on della-vis1 (not a login node). Usage:
# ssh della-vis1 'cd <worktree> && nice -n 19 bash rl/args_all.sh [P G] [set=NAME] [seed=N]' (set=NAME, seed=N: the
# data file and the seeds of that problem set and replicate, as rl/rl_job.sh builds them; levers=LIST: the engine levers,
# whose serving flags and environment also reach the elastic spec check). Prints ARGS_OK per argv,
# SCHEDULE_OK.
set -euo pipefail
SET= SEED=67 LEVERS= POS=()
for x in "$@"; do case "$x" in set=*) SET=${x#set=};; seed=*) SEED=${x#seed=};; levers=*) LEVERS=${x#levers=};; *) POS+=("$x");; esac; done
set -- "${POS[@]}"
cd "$(dirname "$0")/.."
REPO=$(pwd)
SIF=/scratch/gpfs/GROUP/USER/project/miles-q38-build/miles-v0.1.3.sif
W=$(mktemp -d /scratch/gpfs/GROUP/USER/tmp/args_all.XXXXXX)
trap 'rm -rf -- "$W"' EXIT
mkdir -p "$W/out" "$W/tmp"
bash serving/build_flint_sglang.sh "$W/pkg" >/dev/null
python3 rl/train67_data.py data/fcs11_train67_agent.jsonl "$W/out/data_train67.jsonl" "${1:-4}" "$SEED" ${SET:+--set "$SET"}
python3 -c "import json; r = json.loads(open('data/fcs11_val40_agent.jsonl').readline()); open('$W/out/data_p0.jsonl', 'w').write(json.dumps(r) + '\n')"
apptainer exec --contain --cleanenv --bind /scratch/gpfs/GROUP --bind "$W/tmp":/tmp \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --env JID=args0 --env OUT="$W/out" --env PKG="$W/pkg" --env TMPDIR="$W" \
  --env T67_P="${1:-4}" --env T67_G="${2:-8}" --env T67_SEED="$SEED" --env T67_LEVERS="$LEVERS" "$SIF" bash "$REPO/rl/rl_inner.sh" args
