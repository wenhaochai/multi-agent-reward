"""fcs_agent_long with MA_LG_HARNESS=v2 (audit 2026-10-24): the budget line ends every observation and error message
(after the rendering, so the trace's output is unchanged), the format error says what the finish command does here, a
finish command combined with others says nothing was submitted, and the prompt states the shell's limits and the
single-case rule. The real _Agent.think and observe with a stub session; stubbed judge."""
import asyncio
import os
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("FCS_CASE_PROC", "0")
os.environ["MA_LG_HARNESS"] = "v2"
os.environ["MA_LG_N"] = "1"
os.environ["MA_LG_TOKENS"] = "100000"
os.environ["TMPDIR"] = tempfile.mkdtemp()
os.environ["MA_AG_TRACE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from miles_team import fcs_agent_long as L  # noqa: E402

LABEL = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/0"
FIN = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
judged = []
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK  " if cond else "FAIL"), name, "" if cond else detail)


async def fake_judge(label, code):
    judged.append(code)
    return {"cases": [0.5] * 10, "status": "ok", "infra_error": "", "msg": ""}


def reply(cmd):
    return "<think>r</think>THOUGHT: go." + (f"\n\n```mswea_bash_command\n{cmd}\n```" if cmd else "")


SCRIPT = [reply("printf 'int main(){}\\n' > solution.cpp"), reply(None), reply("ls; " + FIN), reply("grep -c x solution.cpp"),
          reply(FIN), reply("submit"), reply(FIN)]


class Stub:
    def __init__(self, input, name, think, max_len=None):
        self.input, self.segments, self.last_messages, self.script = input, [], None, list(SCRIPT)

    async def turn(self, messages, max_tokens):
        self.segments.append(SimpleNamespace(loss_mask=[1] * 1000))
        return self.script.pop(0), False


L.judge_code, L.Session = fake_judge, Stub
inp = SimpleNamespace(sample=SimpleNamespace(index=0), evaluation=True)
agents, S, info = asyncio.run(L.episode(inp, [{"role": "user", "content": "problem"}], LABEL))
a = agents[0]
users = [m["content"] for m in a.chat[2:] if m["role"] == "user"]
check("the episode ends with Finished after 7 steps", a.exit == "Finished" and a.steps == 7, (a.exit, a.steps))
check("every observation and error ends with the budget line", len(users) == 6 and all(
    u.rstrip().endswith("left.</budget>") for u in users), users)
check("the line counts the tokens so far", "<budget>1,000 of 100,000 generated tokens used; about 99,000 left." in users[0]
      and "6,000 of 100,000" in users[-1], users)
check("the trace's output has no budget line", all("<budget>" not in t.get("output", "") for t in a.turns), a.turns)
check("format error: the finish command submits here", L._FORMAT_ERROR_END_V2 in users[1]
      and "not be able to continue" not in users[1], users[1])
check("finish combined with ls: nothing submitted, the note says so", "Nothing was submitted" in a.turns[2]["output"]
      and len(judged) == 2, (a.turns[2], judged))
check("a command without the finish command gets no note", "Nothing was submitted" not in a.turns[3]["output"], a.turns[3])
check("the finish reply leaves the token count to the budget line", "tokens are left" not in a.turns[4]["output"]
      and "Your work goes on." in a.turns[4]["output"], a.turns[4]["output"])
a.generated = L.TOKENS + 5
check("'left' never goes below 0", "about 0 left" in a._usage())
check("info and trace name the harness", info["lg_harness"] == "long-v2-ctxlow1", info["lg_harness"])

pr = L.instance("Read n {and} print it.", "agent1")
check("prompt: the budget line is announced", "The end of each observation shows how many tokens" in pr)
check("prompt: /tmp, ASan and the auto-commits", L._ENV_ASAN.strip() in pr and L._ENV_COMMITS.strip() in pr
      and ((L._ENV_FS.strip() in pr) == bool(L.C.J.LANDLOCK_ABI)), pr[:3000])
check("prompt: the provisional and final halves", "OTHER half" in pr and L._SHARED_FINAL not in pr)
ps = L.instance("Read n {and} print it.", "agent1", shared=True)
check("prompt, single case: one rule, no halves", L._SHARED_FINAL in ps and "OTHER half" not in ps
      and "half of the hidden tests" not in ps and "single hidden test, with any compiler errors" in ps, ps[-3000:])
L.N = 4
pt = L.instance("Read n {and} print it.", "agent2")
check("team prompt formats, with the shell facts and without the solo commit line", "You are agent2, one of 4 agents" in pt
      and L._ENV_ASAN.strip() in pt and L._ENV_COMMITS.strip() not in pt and "{" + "name}" not in pt)
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
