"""Our judge on the Frontier-CS v1.1 problems: re-judge earlier submissions (the untrained SFT agent's 4-episode eval,
one submission per problem, the best one) under the old checkout and under v1.1, for every classic v1.1 problem that
has one. Pass: no infra error and no checker build failure on v1.1. Scores may differ (v1.1 rescales above 60)."""
import json, sys, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/miles-q38-build-long")
from miles_team.fcs_rm import judge_full
OLD = "/scratch/gpfs/GROUP/USER/project/Frontier-CS/algorithmic/problems/"
NEW = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/"
T = "/scratch/gpfs/GROUP/USER/project/miles-q38-build/runs/fcs_evalonly_agentev4_sft/traces/agent_102957.jsonl"
best = {}
for l in open(T):
    r = json.loads(l); pid = r["label"].rstrip("/").split("/")[-1]
    if r.get("submission") and (pid not in best or r["S"] > best[pid][0]):
        best[pid] = (r["S"], r["submission"])
split = json.load(open("/scratch/gpfs/GROUP/USER/project/miles-q38-build-long/data_v11/fcs11_split.json"))
pids = [p for p in split["val"] + split["train"] if p in best]
def run(p):
    o = judge_full(OLD + p, best[p][1]); n = judge_full(NEW + p, best[p][1])
    return p, o, n
st = collections.Counter(); bad = []; rows = []
with ThreadPoolExecutor(6) as ex:
    for p, o, n in ex.map(run, pids):
        st[(o["status"], n["status"])] += 1
        rows.append((p, 100 * best[p][0], 100 * o["score"], 100 * n["score"], o["status"], n["status"]))
        if n.get("infra_error") or "checker" in str(n.get("msg", "")).lower() and n["status"] != "done":
            bad.append((p, n["status"], str(n.get("msg", ""))[:200]))
for r in sorted(rows, key=lambda r: int(r[0])):
    print("%4s trace %6.2f old %6.2f v1.1 %6.2f  %s -> %s" % r)
print(st); print("BAD", bad)
print("PASS" if not bad else "FAIL")
