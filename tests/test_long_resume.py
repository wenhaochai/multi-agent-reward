"""fcs_agent_long's checkpoint (episode(..., ckpt=Checkpoint(path))): a two-agent episode paused after a round, or
killed in the middle of one (after the other agent's command ran), resumes from the last saved round and ends exactly
as the uninterrupted run: the same chats (every prompt byte for byte), submissions, board, rounds, tokens, git log and
score. Fake Session (replies keyed on the agent's turn count), stubbed judge, real problem 0 for the case count."""
import asyncio
import glob
import json
import os
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("FCS_CASE_PROC", "0")
os.environ["MA_LG_N"] = "2"
os.environ.setdefault("MA_LG_TOKENS", "100000")
os.environ["TMPDIR"] = tempfile.mkdtemp()
TRACE = os.environ["MA_AG_TRACE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from miles_team import fcs_agent as A  # noqa: E402
from miles_team import fcs_agent_long as L  # noqa: E402

LABEL = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/0"
FIN = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SCRIPT = {
    "agent1": ["printf 'int main(){return 0;}\\n' > solution.cpp", "submit", "post I wrote a stub",
               "echo 'note 1' >> notes.md", "printf 'int main(){int x;return 0;}\\n' > solution.cpp", FIN,
               "cat notes.md BOARD.md SCORES.md", FIN],
    "agent2": ["ls", "echo 'from agent2' > b.txt", "cat BOARD.md", "printf 'int main(){return 1;}\\n' > sol2.cpp",
               "submit sol2.cpp", "cat SCORES.md b.txt", "ls", FIN, FIN],
}
hook = {"stop_at": None, "ckpt": None, "judge_crash": None}
prompts, judged = {}, []


class FakeSession:
    def __init__(self, input, name, think, max_len=None):
        self.input, self.name, self.segments, self.last_messages = input, name, [], None
        self.used, self.queue, self.cont, self.n_cont, self.last_reasoning, self.fresh = 0, [], False, 0, "", False

    async def turn(self, messages, max_tokens):
        k = sum(m["role"] == "assistant" for m in messages)   # this agent's turns so far: the same after a resume
        prompts[(self.name, k)] = json.dumps(messages)
        if hook["stop_at"] == (self.name, k):
            hook["ckpt"].stop = True
        self.used += 7
        self.segments.append(SimpleNamespace(loss_mask=[1] * 10))
        cmd = SCRIPT[self.name][k] if k < len(SCRIPT[self.name]) else FIN
        return f"<think>r{k}</think>step {k}\n\n```mswea_bash_command\n{cmd}\n```", False


async def fake_judge(label, code):
    judged.append(code)
    if hook["judge_crash"] == len(judged):
        raise RuntimeError("killed in the middle of a round")
    return {"cases": [(len(code) % 7) / 7] * 10, "status": "ok", "infra_error": "", "msg": ""}


L.Session = FakeSession
L.judge_code = fake_judge
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK  " if cond else "FAIL"), name, "" if cond else detail)


# wall-clock numbers differ between any two runs; everything else must match
CLOCK = {"t", "think_s", "seconds", "cmd_wall_s", "cmd_queue_s", "harness_s", "judge_s", "llm_s", "t_req_start"}


def run(ckpt=None):
    prompts.clear()
    inp = SimpleNamespace(sample=SimpleNamespace(index=3, group_index=1), evaluation=True)
    agents, S, info = asyncio.run(L.episode(inp, [{"role": "user", "content": "problem"}], LABEL, ckpt=ckpt))
    files = sorted(glob.glob(os.path.join(TRACE, "*.jsonl")))
    tr = json.loads(open(files[-1]).read().splitlines()[-1])
    return {"chats": [json.dumps(a.chat) for a in agents], "S": S,
            "info": {k: v for k, v in info.items() if not k.endswith("_min")},
            "subs": [{k: v for k, v in s.items() if k not in ("rev", "judge_s")} for s in tr["submissions"]],
            "board": tr["board"], "rounds": tr["rounds"], "git_log": tr["git_log"], "blame": tr["blame"],
            "agents": [{k: v for k, v in x.items() if k != "turns"} for x in tr["agents"]],
            "turns": [[{k: v for k, v in t.items() if k not in CLOCK} for t in x["turns"]] for x in tr["agents"]],
            "used": [a.sess.used for a in agents], "prompts": dict(prompts)}


def same(a, b):
    keys = [k for k in a if k != "prompts" and a[k] != b[k]]
    return not keys, keys


ref = run()
print("reference", ref["rounds"], [a["exit"] for a in ref["agents"]], [(s["agent"], s["file"]) for s in ref["subs"]])
check("reference episode ran", ref["rounds"] >= 8 and len(ref["subs"]) == 4 and ref["board"], (ref["rounds"], ref["subs"]))

# pause: stop at agent1's 4th turn; the episode saves the end of that round and raises Paused
path = os.path.join(tempfile.mkdtemp(), "state", "0_e1")
ck = L.Checkpoint(path, every=1e9)
hook.update(stop_at=("agent1", 3), ckpt=ck)
try:
    run(ck)
    paused = False
except L.Paused:
    paused = True
hook["stop_at"] = None
tars = glob.glob(path + ".r*.tgz")
st = L.Checkpoint(path).load() or {"agents": [{}]}
check("the state has the game's counters and the episode's wall time", "idle_fin" in st["agents"][0].get("agent", {})
      and st.get("wall_ep", 0) > 0 and st["config"]["END"] == L.END and st["config"]["HARNESS"] == L.HARNESS, st.keys())
check("pause saves one state and one tar", paused and ck.saves >= 1 and os.path.exists(path + ".pkl") and len(tars) == 1,
      (paused, ck.saves, tars))
ck2 = L.Checkpoint(path, every=1e9)
res = run(ck2)
eq, diff = same(ref, res)
check("resume after a pause ends as the uninterrupted run", eq, diff)
check("resumed prompts are byte-identical", res["prompts"] and all(ref["prompts"][k] == v for k, v in res["prompts"].items())
      and ("agent1", 4) in res["prompts"] and ("agent1", 3) not in res["prompts"], sorted(res["prompts"]))
check("wall time carried over", ck2.prior_s > 0)
ck2.clear()
check("clear removes the files", not glob.glob(path + "*"), glob.glob(path + "*"))

# kill in the middle of a round: the 3rd judge call raises after the other agent's command may have run
ck = L.Checkpoint(path, every=0)   # save after every round
judged.clear()
hook["judge_crash"] = 3
try:
    run(ck)
    crashed = False
except RuntimeError:
    crashed = True
hook["judge_crash"] = None
check("mid-round kill leaves the last round's state", crashed and os.path.exists(path + ".pkl") and len(glob.glob(path + ".r*.tgz")) == 1)
judged.clear()
res = run(L.Checkpoint(path, every=0))
eq, diff = same(ref, res)
check("resume after a mid-round kill ends as the uninterrupted run", eq, diff)

# a checkpoint of other settings is ignored
st = L.Checkpoint(path).load()
check("finished episode leaves its last save (the driver clears it)", st is not None)
L.TOKENS += 1
res = run(L.Checkpoint(path, every=1e9))
L.TOKENS -= 1
check("other settings: fresh start", res["rounds"] == ref["rounds"] and len(res["prompts"]) == len(ref["prompts"]))

# a save that fails (a full quota, a file that shrinks while the tar reads it) logs; the episode goes on and the last
# good save stays
import errno  # noqa: E402
import pickle  # noqa: E402
import tarfile  # noqa: E402
from types import SimpleNamespace as NS  # noqa: E402


def quota(*a, **k):
    raise OSError(errno.EDQUOT, "Disk quota exceeded")


real_tar, real_pickle = L.tarfile, L.pickle
p3 = os.path.join(tempfile.mkdtemp(), "state", "0_e1")
L.tarfile = NS(open=quota, data_filter=tarfile.data_filter, FilterError=tarfile.FilterError)
ck = L.Checkpoint(p3, every=0)
res = run(ck)
L.tarfile = real_tar
eq, diff = same(ref, res)
check("every tar fails: the episode ends as the uninterrupted run, nothing left on disk", eq and ck.failed > 0
      and ck.saves == 0 and not glob.glob(p3 + "*"), (diff, ck.failed, ck.saves, glob.glob(p3 + "*")))
ck = L.Checkpoint(p3, every=1e9)
hook.update(stop_at=("agent1", 3), ckpt=ck)
try:
    run(ck)
except L.Paused:
    pass
hook["stop_at"] = None
good = L.Checkpoint(p3).load()
L.pickle = NS(dump=quota, load=pickle.load, HIGHEST_PROTOCOL=pickle.HIGHEST_PROTOCOL)
ck = L.Checkpoint(p3, every=0)
res = run(ck)
L.pickle = real_pickle
st = L.Checkpoint(p3).load()
eq, diff = same(ref, res)
check("every pickle fails after a resume: the episode ends as the uninterrupted run, the good save stays", eq
      and ck.failed > 0 and st and st["rounds"] == good["rounds"] and glob.glob(p3 + ".r*.tgz") == [
          os.path.join(os.path.dirname(p3), good["tar"])] and not glob.glob(p3 + "*.tmp"), (diff, ck.failed))

# nor one of a setting that changes the game: a paused state, then one setting changed, starts fresh
for name, val in [("CTX_LOW", 0.75), ("HARNESS", "v2"), ("END", "budget"), ("VERDICTS", True)]:
    p2 = os.path.join(tempfile.mkdtemp(), "state", "0_e1")
    ck = L.Checkpoint(p2, every=1e9)
    hook.update(stop_at=("agent1", 3), ckpt=ck)
    try:
        run(ck)
    except L.Paused:
        pass
    hook["stop_at"] = None
    old = getattr(L, name)
    setattr(L, name, val)
    res = run(L.Checkpoint(p2, every=1e9))
    setattr(L, name, old)
    check(f"other {name}: fresh start", ck.saves and ("agent1", 0) in res["prompts"], sorted(res["prompts"]))
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
