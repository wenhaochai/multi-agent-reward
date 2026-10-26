"""fcs_agent_long with MA_LG_POST=submit (owner 2026-10-26 popup; four team smokes, 32 agents, had no post). After each
of its submissions (submit, or a finish command that submitted) an agent's next command must be `post <message>` with a
message: any other command, an empty post and a finish command included, is not run, and the reply asks for the post.
The rule is in the prompt and after each submission's output; refusals are counted, flagged on their turns and kept
across a checkpoint; the harness tag and the trace name the rule. A solo game or an unknown value is refused at import.
Fake Session (replies keyed on the agent's turn count), stubbed judge, real problem 0 for the case count."""
import asyncio
import glob
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("FCS_CASE_PROC", "0")
os.environ["MA_LG_N"] = "2"
os.environ["MA_LG_POST"] = "submit"
os.environ["MA_LG_TOKENS"] = "100000"
os.environ["TMPDIR"] = tempfile.mkdtemp()
TRACE = os.environ["MA_AG_TRACE_DIR"] = tempfile.mkdtemp()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from miles_team import fcs_agent_long as L  # noqa: E402

LABEL = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/0"
FIN = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SCRIPT = {
    "agent1": ["printf 'int main(){return 0;}\\n' > solution.cpp", "submit", "touch marker1", "post ''",
               "post a stub, provisional 14; next a real algorithm", "ls marker1 2>&1; echo rc=$?",
               "printf 'int main(){int y;return 0;}\\n' > solution.cpp", FIN, FIN, "post second version", FIN],
    "agent2": ["ls", "echo hi > b.txt", "cat BOARD.md", "submit b.txt", "cat SCORES.md", "post agent2 tried b.txt"],
}
hook = {"stop_at": None, "ckpt": None}
judged = []
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK  " if cond else "FAIL"), name, "" if cond else detail)


class FakeSession:
    def __init__(self, input, name, think, max_len=None):
        self.input, self.name, self.segments, self.last_messages = input, name, [], None
        self.used, self.queue, self.cont, self.n_cont, self.last_reasoning, self.fresh = 0, [], False, 0, "", False

    async def turn(self, messages, max_tokens):
        k = sum(m["role"] == "assistant" for m in messages)   # this agent's turns so far: the same after a resume
        if hook["stop_at"] == (self.name, k):
            hook["ckpt"].stop = True
        self.segments.append(SimpleNamespace(loss_mask=[1] * 10))
        s = SCRIPT[self.name]
        # past the script: the finish command, then a post, in turn, so every agent ends (an unchanged finish)
        cmd = s[k] if k < len(s) else (FIN if (k - len(s)) % 2 == 0 else "post done")
        return f"<think>r{k}</think>step {k}\n\n```mswea_bash_command\n{cmd}\n```", False


async def fake_judge(label, code):
    judged.append(code)
    return {"cases": [(len(code) % 7) / 7] * 10, "status": "ok", "infra_error": "", "msg": ""}


L.Session, L.judge_code = FakeSession, fake_judge
INP = SimpleNamespace(sample=SimpleNamespace(index=0), evaluation=True)
MSG = [{"role": "user", "content": "problem"}]


def turn_of(a, cmd):
    return next(t for t in a.turns if t.get("command") == cmd)


def broken(rec) -> int:
    """The smoke gate's check (flint_smoke_v2.sh): a command that ran while a post was owed."""
    bad = 0
    for a in rec["agents"]:
        owe = False
        for t in a["turns"]:
            c = t.get("command")
            if c is None or t.get("post_refused"):
                continue
            bad += owe and not c.lstrip().startswith("post")
            owe = "judge_s" in t
    return bad


agents, S, info = asyncio.run(L.episode(INP, MSG, LABEL))
a1, a2 = agents
recs = [json.loads(x) for f in glob.glob(TRACE + "/*.jsonl") for x in open(f)]
rec = recs[-1]
check("the prompt states the rule", L._POST_RULE in a1.chat[1]["content"] and L._POST_RULE in a2.chat[1]["content"],
      a1.chat[1]["content"][-1500:])
t = turn_of(a1, "submit")
check("a submission's output ends by asking for the post", t["output"].endswith(L._POST_NEXT), t["output"][-300:])
t = turn_of(a1, "touch marker1")
check("a command while a post is owed is not run: flagged, exit 1, the reply asks for the post",
      t.get("post_refused") and t["returncode"] == 1 and t["output"].endswith(L._POST_REFUSED), t)
t = turn_of(a1, "post ''")
check("an empty post is refused too, and not posted", t.get("post_refused")
      and all(m.strip() for _, n, m in rec["board"] if n == "agent1"), (t, rec["board"]))
t = turn_of(a1, "ls marker1 2>&1; echo rc=$?")
check("after the post, commands run again; the refused one never ran", not t.get("post_refused")
      and "No such file" in t["output"] and "rc=2" in t["output"], t["output"])
fins = [t for t in a1.turns if t.get("command") == FIN]
check("a finish command that submitted makes the next one wait for a post (an unchanged one would end the work)",
      len(fins) == 3 and "judge_s" in fins[0] and fins[0]["output"].endswith(L._POST_NEXT)
      and fins[1].get("post_refused") and not fins[2].get("post_refused") and a1.exit == "Finished",
      [(t.get("judge_s"), t.get("post_refused"), t["output"][:80]) for t in fins] + [a1.exit])
check("refusals are counted per agent", a1.post_refused == 3 and a2.post_refused == 1,
      (a1.post_refused, a2.post_refused))
check("the board holds the posts in order", [(n, m) for _, n, m in rec["board"] if m != "done"] ==
      [("agent1", "a stub, provisional 14; next a real algorithm"), ("agent2", "agent2 tried b.txt"),
       ("agent1", "second version")], rec["board"])
check("every agent ended; no command ran while a post was owed (the smoke gate's check)",
      a2.exit == "Finished" and broken(rec) == 0 and a1.owe_post is False, (a2.exit, broken(rec), a1.owe_post))
check("info, the harness tag and the trace name the rule", info["fo_lg_must_post"] == 1.0
      and info["fo_lg_post_refused"] == 2.0 and info["lg_harness"] == "long-v1-ctxlow1-mustpost"
      and rec["post"] == "submit" and [x["post_refused"] for x in rec["agents"]] == [3, 1],
      (info.get("fo_lg_post_refused"), info["lg_harness"], rec.get("post")))

# a checkpoint taken right after the submission keeps the owed post: the resumed run refuses the next command and ends
# exactly as the uninterrupted run
ref = (len(rec["submissions"]), [(n, m) for _, n, m in rec["board"]], [x["post_refused"] for x in rec["agents"]],
       rec["rounds"], rec["S"])
ck = L.Checkpoint(os.path.join(tempfile.mkdtemp(), "ep"), every=1e9)
hook.update(stop_at=("agent1", 1), ckpt=ck)
try:
    asyncio.run(L.episode(INP, MSG, LABEL, ckpt=ck))
    paused = False
except L.Paused:
    paused = True
st = ck.load()
check("paused right after the submission, with the post owed in the saved state", paused and st and
      st["agents"][0]["agent"]["owe_post"] is True and st["config"].get("POST") == "submit",
      (paused, st and st["agents"][0]["agent"].get("owe_post"), st and st["config"]))
hook.update(stop_at=None)
ck.stop = False
agents, S, info = asyncio.run(L.episode(INP, MSG, LABEL, ckpt=ck))
rec2 = [json.loads(x) for f in glob.glob(TRACE + "/*.jsonl") for x in open(f)][-1]
got = (len(rec2["submissions"]), [(n, m) for _, n, m in rec2["board"]], [x["post_refused"] for x in rec2["agents"]],
       rec2["rounds"], rec2["S"])
check("the resumed run refuses the next command and ends as the uninterrupted one",
      got == ref and turn_of(agents[0], "touch marker1").get("post_refused"), (got, ref))

for env, what in (({"MA_LG_N": "1", "MA_LG_POST": "submit"}, "a solo game"),
                  ({"MA_LG_N": "2", "MA_LG_POST": "always"}, "an unknown value")):
    r = subprocess.run([sys.executable, "-c", "import miles_team.fcs_agent_long"], cwd=ROOT,
                       env={**os.environ, **env}, capture_output=True, text=True)
    check(f"MA_LG_POST=submit refuses {what} at import", r.returncode != 0 and "AssertionError" in r.stderr,
          r.stderr[-300:])
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
