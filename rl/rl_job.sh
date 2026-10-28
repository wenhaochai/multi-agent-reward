#!/bin/bash
#SBATCH --job-name=flint-rl
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=880G
#SBATCH --time=03:00:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/tmp/flint_s5/logs/%x-%j.out
#FETCHABLE
# RL of Flint on the long game through miles v0.1.3 (our fork), on one 8-GPU block: GRPO without std normalization,
# score centering over the sampler's top-32 candidates, routing replay (R3), one optimizer step per rollout (problem 0,
# 8 episodes of 128k tokens), the rollout engines (4 x TP2) and the FSDP actor (CPU-offloaded parameters and Adam)
# colocated. Usage: rl_job.sh test2|run20 (wq --prio 10; run20's own --time). The game runs with RL's context cap
# (rl/rl_inner.sh: 110k characters, 16k-token turns) so that a training sample stays near 64k tokens.
#   test2     2 colocated RL steps, the actor saved at the end
#   run20     20 colocated RL steps, the actor saved at the end
# The first failure ends the job: nothing is retried. Results: tmp/flint_s5/runs/<job>/. One container with a PID
# namespace of its own (--pid) holds everything (see rl/rollout_job.sh); bounds: timeout -s KILL around it, the GPU
# watch (rl/gpu_watch.sh) and cleanup of only the process groups this job started (serving/check/procs.sh).
set -uo pipefail
MODE=${1:-}
case "$MODE" in test2) LIMIT=9600;; run20) LIMIT=81000;; *) echo "usage: rl_job.sh test2|run20"; exit 1;; esac
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
[ -n "$JID" ] || { echo "no job id"; exit 1; }
REPO=/scratch/gpfs/GROUP/USER/project/mas-rl-rollout
MILES=/scratch/gpfs/GROUP/USER/project/miles-rollout
SIF=/scratch/gpfs/GROUP/USER/project/miles-q38-build/miles-v0.1.3.sif
FLINT=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint
OUT=/scratch/gpfs/GROUP/USER/tmp/flint_s5/runs/$JID
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
mkdir -p "$OUT" "$TMPDIR/tmp" "$TMPDIR/datasets"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
source "$REPO/serving/check/procs.sh"
source "$REPO/rl/gpu_watch.sh"
WD= MEM= SMI=
trap 'stop "${STARTED[@]}"; kill $WD $MEM $SMI 2>/dev/null; rm -rf -- "/scratch/gpfs/GROUP/USER/tmp/job/$JID"' EXIT
trap 'exit 143' TERM INT
echo "[s5] $(date -u +%FT%TZ) $MODE node=$(hostname) job=$JID gpus=${CUDA_VISIBLE_DEVICES:-}"
echo "mas-rl $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain -uno | wc -l)" \
  "miles $(git -C "$MILES" rev-parse --short HEAD) dirty=$(git -C "$MILES" status --porcelain -uno | wc -l)" | tee "$OUT/code"
python3 - "$FLINT" <<'PY' || exit 1
import json, os, sys
d = sys.argv[1]
files = sorted(set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values()))
missing = [f for f in files + ["config.json", "tokenizer.json"] if not os.path.isfile(os.path.join(d, f))]
sys.exit(f"[s5] PREFLIGHT_FAIL {d}: {missing}") if missing else print(f"[s5] preflight: {len(files)} weight files")
PY
PKG=$TMPDIR/flint_pkg
bash "$REPO/serving/build_flint_sglang.sh" "$PKG" || exit 1
# the RL problem: val40 problem 0, its directory also in the metadata (miles hands the agent only the metadata); the
# row 1000 times, so that miles' epoch (one pass over the rows, a save point) never ends inside a run
python3 - "$REPO/data/fcs11_val40_agent.jsonl" "$OUT/data_p0.jsonl" <<'PY' || exit 1
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1]) if json.loads(x)["metadata"]["pid"] == "0"]
assert len(rows) == 1
rows[0]["metadata"]["label"] = rows[0]["label"]
open(sys.argv[2], "w").write((json.dumps(rows[0]) + "\n") * 1000)
PY
# one line per GPU and minute; stops the job when a GPU idles (rl/gpu_watch.sh), counting from the first rollout
# request: the start (actor weights 8 min in w583, then the engines; GPUs idle 11 min) may take up to 35 min
gpu_watch $$ "$OUT/gpu.log" "grep -q 'POST /v1/chat/completions' '$OUT/train.log' 2>/dev/null" 35 &
WD=$!
( while sleep 10; do echo "$(date +%s) $(free -m | awk '/Mem:/{print $3}')"; done ) > "$OUT/host_mem.txt" &
MEM=$!
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 2 > "$OUT/gpu_mem.csv" 2>/dev/null &
SMI=$!
start timeout -s KILL "$LIMIT" apptainer exec --pid --nv --contain --cleanenv --writable-tmpfs \
  --bind /scratch/gpfs/GROUP --bind "$TMPDIR/tmp":/tmp --bind "$TMPDIR/datasets":/root/datasets --bind /proc:/host/proc \
  --bind /dev/shm --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  --env JID="$JID" --env OUT="$OUT" --env PKG="$PKG" --env TMPDIR="$TMPDIR" "$SIF" bash "$REPO/rl/rl_inner.sh" "$MODE"
wait "${STARTED[0]}"
rc=$?
echo "[s5] $(date -u +%FT%TZ) container exit $rc"
python3 - "$OUT" <<'PY'
import collections, sys
out = sys.argv[1]
peak = collections.defaultdict(int)
for line in open(f"{out}/gpu_mem.csv"):
    p = [x.strip() for x in line.split(",")]
    if len(p) == 3 and p[2].isdigit():
        peak[int(p[1])] = max(peak[int(p[1])], int(p[2]))
host = max((int(l.split()[1]) for l in open(f"{out}/host_mem.txt") if len(l.split()) == 2), default=0)
print("[s5] peak GPU memory (2 s samples, MiB):", dict(sorted(peak.items())), "| peak host used (10 s samples, MiB):", host)
PY
exit $rc
