"""An episode's checkpoint (episode(..., ckpt=Checkpoint(path))): a team episode paused after a round, or killed in the
middle of one (after the other agent's command ran), resumes from the last saved round and ends exactly as the
uninterrupted run: the same chats (every prompt byte for byte), submissions, board, rounds, tokens, git log and score.
A checkpoint of other settings, or one whose tar is unreadable, starts fresh; a save that fails keeps the last good
one. Fake session (replies keyed on the agent's turn count; 1000 tokens a turn and a 10,000-token budget), stubbed
judge, real problem 0. Git commit hashes differ between runs (commit times): they are compared as H."""
import asyncio
import errno
import glob
import json
import os
import pickle
import re
import tarfile
import tempfile
from types import SimpleNamespace as NS

from helpers import CLOCK, FIN, LABEL, MSG, block, check, done, env, inp, sessions, traces

env(MA_LG_N=2, MA_LG_TOKENS=10000)
from flint import game as G  # noqa: E402

SCRIPT = {
    "agent1": ["printf 'int main(){return 0;}\\n' > solution.cpp", "submit", "post I wrote a stub",
               "echo 'note 1' >> notes.md", "printf 'int main(){int x;return 0;}\\n' > solution.cpp", FIN, "post v2",
               "cat notes.md BOARD.md SCORES.md", "git log --oneline team/agent2 | wc -l", "true"],
    "agent2": ["ls", "echo 'from agent2' > b.txt", "cat BOARD.md", "printf 'int main(){return 1;}\\n' > sol2.cpp",
               "submit sol2.cpp", "post agent2 sol2", "cat SCORES.md b.txt",
               "git checkout team/agent1 -- solution.cpp && echo took", "ls", "true"],
}
hook = {"stop_at": None, "ckpt": None, "judge_crash": None}
prompts, judged = {}, []
HEX = re.compile(r"\b[0-9a-f]{7,40}\b")


async def reply(sess, messages):
    k = sum(m["role"] == "assistant" for m in messages)   # this agent's turns so far: the same after a resume
    prompts[(sess.name, k)] = json.dumps(h(messages))
    if hook["stop_at"] == (sess.name, k):
        hook["ckpt"].stop = True
    return block(SCRIPT[sess.name][k] if k < len(SCRIPT[sess.name]) else "true", k), False


async def fake_judge(label, code):
    judged.append(code)
    if hook["judge_crash"] == len(judged):
        raise RuntimeError("killed in the middle of a round")
    return {"cases": [(len(code) % 7) / 7] * 10, "status": "ok", "infra_error": "", "msg": ""}


G.judge_code = fake_judge


def h(x):
    """x with every git hash in its strings as H."""
    if isinstance(x, dict):
        return {k: h(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [h(v) for v in x]
    return HEX.sub("H", x) if isinstance(x, str) else x


def run(ckpt=None):
    prompts.clear()
    agents, S, info = asyncio.run(G.episode(inp(3, 1), MSG, LABEL, sessions(reply, tokens=1000), ckpt=ckpt))
    tr = traces()[-1]
    return {"chats": [h(a.chat) for a in agents], "S": S,
            "info": {k: v for k, v in info.items() if not k.endswith("_min")},
            "subs": [h({k: v for k, v in s.items() if k not in ("rev", "judge_s")}) for s in tr["submissions"]],
            "board": tr["board"], "rounds": tr["rounds"], "git_log": sorted(tr["git_log"].splitlines()),
            "blame": tr["blame"], "adoptions": tr["adoptions"],
            "agents": [{k: v for k, v in x.items() if k != "turns"} for x in tr["agents"]],
            "turns": [[h({k: v for k, v in t.items() if k not in CLOCK}) for t in x["turns"]] for x in tr["agents"]],
            "used": [a.sess.used for a in agents], "prompts": dict(prompts)}


def same(a, b):
    keys = [k for k in a if k != "prompts" and a[k] != b[k]]
    return not keys, keys


def pause(path, at=("agent1", 3)):
    ck = G.Checkpoint(path, every=1e9)
    hook.update(stop_at=at, ckpt=ck)
    try:
        run(ck)
        paused = False
    except G.Paused:
        paused = True
    hook["stop_at"] = None
    return ck, paused


ref = run()
print("reference", ref["rounds"], [a["exit"] for a in ref["agents"]], [(s["agent"], s["file"]) for s in ref["subs"]])
check("reference episode ran (10 turns each, then a round that ends both)", ref["rounds"] == 11 and len(ref["subs"]) == 3 and len(ref["board"]) == 3
      and ref["adoptions"], (ref["rounds"], ref["subs"], ref["board"]))
path = os.path.join(tempfile.mkdtemp(), "state", "0_e1")
ck, paused = pause(path)
tars = glob.glob(path + ".r*.tgz")
st = G.Checkpoint(path).load() or {"agents": [{}]}
check("the state has the game's counters, the team's git record and the episode's wall time",
      "idle_fin" in st["agents"][0].get("agent", {}) and "origin" in st and st.get("wall_ep", 0) > 0
      and st["config"]["END"] == "budget" and st["config"]["REPOS"] == "own", st.keys())
check("pause saves one state and one tar", paused and ck.saves >= 1 and os.path.exists(path + ".pkl") and len(tars) == 1,
      (paused, ck.saves, tars))
ck2 = G.Checkpoint(path, every=1e9)
res = run(ck2)
eq, diff = same(ref, res)
check("resume after a pause ends as the uninterrupted run", eq, diff)
check("resumed prompts are byte-identical", res["prompts"] and all(ref["prompts"][k] == v for k, v in res["prompts"].items())
      and ("agent1", 4) in res["prompts"] and ("agent1", 3) not in res["prompts"], sorted(res["prompts"]))
check("wall time carried over", ck2.prior_s > 0)
ck2.clear()
check("clear removes the files", not glob.glob(path + "*"), glob.glob(path + "*"))

# kill in the middle of a round: the 3rd judge call raises after the other agent's command may have run
ck = G.Checkpoint(path, every=0)   # save after every round
judged.clear()
hook["judge_crash"] = 3
try:
    run(ck)
    crashed = False
except RuntimeError:
    crashed = True
hook["judge_crash"] = None
check("mid-round kill leaves the last round's state", crashed and os.path.exists(path + ".pkl")
      and len(glob.glob(path + ".r*.tgz")) == 1)
judged.clear()
res = run(G.Checkpoint(path, every=0))
eq, diff = same(ref, res)
check("resume after a mid-round kill ends as the uninterrupted run", eq, diff)

# a checkpoint of other settings is ignored
check("a finished episode leaves its last save (the driver clears it)", G.Checkpoint(path).load() is not None)
G.CTX_LOW = 0.5
ck = G.Checkpoint(path, every=1e9)
res = run(ck)
G.CTX_LOW = 0.75
check("other settings: fresh start, no earlier wall time", res["rounds"] == ref["rounds"]
      and len(res["prompts"]) == len(ref["prompts"]) and ck.prior_s == 0.0, (res["rounds"], ck.prior_s))
# a state whose tar is unreadable: fresh start, no earlier wall time
pause(path)
for t in glob.glob(path + ".r*.tgz"):
    open(t, "wb").write(b"not a tar")
ck = G.Checkpoint(path, every=1e9)
res = run(ck)
check("unreadable tar: fresh start, no earlier wall time", ("agent1", 0) in res["prompts"] and ck.prior_s == 0.0
      and same(ref, res)[0], (ck.prior_s, same(ref, res)[1]))


def quota(*a, **k):
    raise OSError(errno.EDQUOT, "Disk quota exceeded")


# a save that fails (a full quota, a file that shrinks while the tar reads it) logs; the episode goes on and the last
# good save stays
p3 = os.path.join(tempfile.mkdtemp(), "state", "0_e1")
G.tarfile = NS(open=quota, data_filter=tarfile.data_filter, FilterError=tarfile.FilterError)
ck = G.Checkpoint(p3, every=0)
res = run(ck)
G.tarfile = tarfile
eq, diff = same(ref, res)
check("every tar fails: the episode ends as the uninterrupted run, nothing left on disk", eq and ck.failed > 0
      and ck.saves == 0 and not glob.glob(p3 + "*"), (diff, ck.failed, ck.saves, glob.glob(p3 + "*")))
pause(p3)
good = G.Checkpoint(p3).load()
G.pickle = NS(dump=quota, load=pickle.load, HIGHEST_PROTOCOL=pickle.HIGHEST_PROTOCOL)
ck = G.Checkpoint(p3, every=0)
res = run(ck)
G.pickle = pickle
st = G.Checkpoint(p3).load()
eq, diff = same(ref, res)
check("every pickle fails after a resume: the episode ends as the uninterrupted run, the good save stays", eq
      and ck.failed > 0 and st and st["rounds"] == good["rounds"] and glob.glob(p3 + ".r*.tgz") == [
          os.path.join(os.path.dirname(p3), good["tar"])] and not glob.glob(p3 + "*.tmp"), (diff, ck.failed))
done()
