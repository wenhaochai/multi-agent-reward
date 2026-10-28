#!/bin/bash
#SBATCH --job-name=flint-smoke
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
#FETCHABLE
# mas-rl: a smoke of the long game on one server (TP = the job's GPUs: 2, 4 or 8). Usage: smoke.sh [key=value ...]
#   n=N        episodes, all at once, on the first N problems of data/fcs11_val40_agent_smoke8.jsonl (N <= 8;
#              more: the first N of val40) (default 2)
#   agents=N   agents per episode (default 4)   tokens=T  tokens per agent (default 150000)
#   repo=PATH  code tree (default the clean worktree of main, miles-q38-build-clean)
# Gate (exit 0): every episode ended clean, and in a team every owed post was paid (after each submission the agent's
# next command that ran was a post with a message; refused commands between are fine). How many posts were owed is
# printed: a short smoke may have no submission, and then the post rule is untested.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY   # see eval.sh
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
P=/scratch/gpfs/GROUP/USER/project
FE=$P/flint-eval
MODEL=$P/labs-molt/_workspace/models/flint
SIF=$FE/flint-sglang.sif
PYSIF=$P/miles-q38-build/miles.sif
N=2 AGENTS=4 TOKENS=150000 REPO=$P/miles-q38-build-clean
for a in "$@"; do
  case "$a" in
    n=*) N=${a#*=};; agents=*) AGENTS=${a#*=};; tokens=*) TOKENS=${a#*=};; repo=*) REPO=${a#*=};;
    *) echo "unknown option $a"; exit 1;;
  esac
done
[[ "$N$TOKENS" =~ ^[0-9]+$ && "$AGENTS" =~ ^[1-8]$ ]] || { echo "bad option value"; exit 1; }
JID=${FETCHED_JOB_ID:-${SLURM_JOB_ID:-}}
[ -n "$JID" ] || { echo "no job id"; exit 1; }
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
mkdir -p "$TMPDIR"
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "$TMPDIR" || true' EXIT
OUT=$FE/runs/smoke_$JID
SLOG=$OUT/server-$(date +%Y%m%dT%H%M%S).log
mkdir -p "$OUT" "$FE/triton_cache"
echo "run $(date -Is) host $(hostname) job $JID: n $N agents $AGENTS tokens $TOKENS repo $REPO" \
     "commit $(git -C "$REPO" rev-parse HEAD)"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "no CUDA_VISIBLE_DEVICES"; exit 1; }
GPUS=$CUDA_VISIBLE_DEVICES
TP=$(awk -F, '{print NF}' <<< "$GPUS")
case "$TP" in 2|4|8) ;; *) echo "need 2, 4 or 8 GPUs, got $TP: $GPUS"; exit 1;; esac
if [ "$N" -le 8 ]; then DATA=data/fcs11_val40_agent_smoke8.jsonl; else DATA=data/fcs11_val40_agent.jsonl; fi

JID_NUM=$(echo "$JID" | tr -dc 0-9)   # ports as in eval.sh
PORT=$((20000 + (10#$JID_NUM % 500) * 20))
while ss -Htan "sport = :$PORT" | grep -q .; do PORT=$((PORT + 1)); done
NCCL=$((PORT + 10))
while ss -Htan "sport = :$NCCL" | grep -q .; do NCCL=$((NCCL + 1)); done
echo "TP$TP on GPUs $GPUS port $PORT nccl $NCCL log $SLOG"
export APPTAINERENV_TMPDIR=$TMPDIR APPTAINERENV_XDG_CACHE_HOME=$TMPDIR/cache
APPTAINERENV_CUDA_VISIBLE_DEVICES=$GPUS APPTAINERENV_MODEL=$MODEL APPTAINERENV_PORT=$PORT \
APPTAINERENV_FLINT_TRITON_CACHE_BASE=$FE/triton_cache \
  apptainer exec --nv --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$SIF" \
  bash "$MODEL/sglang/serve_h200.sh" --tp-size "$TP" --mem-fraction-static 0.85 --nccl-port "$NCCL" \
  --host 127.0.0.1 --enable-cache-report > "$SLOG" 2>&1 &
SRV=$!
for _ in $(seq 1 120); do
  curl -sf "127.0.0.1:$PORT/health_generate" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "server died"; tail -60 "$SLOG"; exit 1; }
  sleep 15
done
curl -sf "127.0.0.1:$PORT/health_generate" >/dev/null || { echo "server not ready after 30 min"; tail -60 "$SLOG"; exit 1; }

cd "$REPO"
set +e
APPTAINERENV_PYTHONPATH=$REPO APPTAINERENV_MA_LG_N=$AGENTS APPTAINERENV_MA_LG_TOKENS=$TOKENS \
APPTAINERENV_FCS_RM_CONCURRENCY=8 APPTAINERENV_FCS_JUDGE_TMP=$TMPDIR APPTAINERENV_FCS_JUDGE_CACHE=$TMPDIR/fcs_judge_cache \
  timeout 140m apptainer exec --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$PYSIF" \
  python3 run_eval.py --base-url "http://127.0.0.1:$PORT" --model flint --data "$DATA" --out "$OUT" \
  --limit "$N" --concurrency "$N" 2>&1 | tee -a "$OUT/driver.log"
rc=${PIPESTATUS[0]}
set -e
echo "driver exit $rc"
python3 - "$OUT" "$N" "$AGENTS" <<'PY'
import glob, json, re, sys
out, n_ep, team = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) > 1
recs = [json.loads(l) for f in glob.glob(out + "/traces/*.jsonl") for l in open(f)]
POST_RE = re.compile(r"\s*post\s+(.+?)\s*", re.S)   # flint.game.POST_RE
bad = paid = 0
for r in recs:
    print(r["label"].split("/")[-1], "S", r["S"], "subs", len(r["submissions"]), "rounds", r["rounds"],
          "agents", [(a["exit"], a["submits"], a["posts"], a.get("post_refused"), a["restarts"], a["generated"])
                     for a in r["agents"]])
    for a in r["agents"] if team else ():
        owe = False
        for t in a["turns"]:
            c = t.get("command")
            if c is None or t.get("post_refused"):
                continue
            m = POST_RE.fullmatch(c)
            ok_post = bool(m and m.group(1).strip().strip("'\"").strip())
            bad += owe and not ok_post
            paid += owe and ok_post
            owe = "judge_s" in t
subs = sum(len(r["submissions"]) for r in recs)
if team:
    print("POST rule broken", bad, "paid", paid, "submissions", subs, "(untested: no submission)" if not subs else "")
ok = len(recs) == n_ep and bad == 0
print("SMOKE_PASS" if ok else "SMOKE_FAIL", f"{len(recs)}/{n_ep} episodes")
sys.exit(0 if ok else 1)
PY
