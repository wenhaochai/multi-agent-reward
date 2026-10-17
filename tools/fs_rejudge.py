"""Re-judge training programs of the 10 FrontierSmith problems fixed by FrontierCS/FrontierSmith PR #4 with the merged
checkers: per problem up to N programs from the EasyPPO lastblock training traces (old reward = what training paid),
plus an empty output. Prints old reward vs new score per program and a per-problem summary; writes JSONL.
  python3 fs_rejudge.py OUT_JSONL [N=12]"""
import glob, json, re, sys, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/miles-q38-build-agentic")
from miles_team.fcs_rm import extract_cpp, judge_full

IDS = {5, 18, 52, 58, 100, 106, 108, 125, 183, 185}
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
B = "/scratch/gpfs/GROUP/USER/project/miles-q38-build/runs"
pool = collections.defaultdict(list)
for f in sorted(glob.glob(f"{B}/fcs_easyppo_lastblock_q9b-fcs-sft_s*/traces/*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        m = re.search(r"frontiersmith_(\d+)$", r["label"].rstrip("/"))
        if m and int(m.group(1)) in IDS and r.get("text") and not r.get("truncated"):
            pool[int(m.group(1))].append(r)
jobs = []
for pid in sorted(IDS):
    rs = pool[pid]
    full = [r for r in rs if r["reward"] >= 0.999][: N // 2]          # the programs the old checker paid in full
    rest = [r for r in rs if r["reward"] < 0.999][: N - len(full)]
    label = (rs[0]["label"] if rs else f"/scratch/gpfs/GROUP/USER/project/frontiersmith-200/Frontier-CS/algorithmic/problems/frontiersmith_{pid}")
    jobs += [(pid, label, r["reward"], extract_cpp(r["text"])) for r in full + rest]
    jobs.append((pid, label, None, 'int main(){return 0;}'))         # empty output
    print(pid, "traced", len(rs), "old full", sum(r["reward"] >= 0.999 for r in rs), flush=True)

def one(j):
    pid, label, old, code = j
    if not code:
        return dict(pid=pid, old=old, new=None, status="no code")
    res = judge_full(label, code)
    return dict(pid=pid, old=old, new=res["score"], status=res["status"], infra=res["infra_error"])

with ThreadPoolExecutor(3) as ex, open(sys.argv[1], "w") as out:
    res = list(ex.map(one, jobs))
    for r in res:
        out.write(json.dumps(r) + "\n")
by = collections.defaultdict(list)
for r in res:
    by[r["pid"]].append(r)
for pid in sorted(by):
    rs = by[pid]; tr = [r for r in rs if r["old"] is not None and r["new"] is not None]
    emp = [r for r in rs if r["old"] is None][0]
    paid = [r for r in tr if r["old"] >= 0.999]
    print(f"fs_{pid}: {len(tr)} programs | old mean {sum(r['old'] for r in tr)/max(1,len(tr)):.2f} new mean "
          f"{sum(r['new'] for r in tr)/max(1,len(tr)):.2f} | old-full {len(paid)} -> new {[round(r['new'],2) for r in paid]} "
          f"| empty output new {emp['new']} ({emp['status']}) | infra {sum(bool(r.get('infra')) for r in rs)}")
