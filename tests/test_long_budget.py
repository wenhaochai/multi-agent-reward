"""fcs_agent_long with MA_LG_END=budget (audit 2026-10-24): only the budget ends the work. The finish command on an
unchanged solution.cpp is not judged and the work goes on, however many come in a row (owner 2026-10-25: the reply
says the work goes on until the budget is used up, with next steps; each is counted); the submit reply says how many
tokens are left; the prompt and the format error drop the end sentence, and a run of format errors does not end it.
Scripted agent (and the real think for the format errors), stubbed judge."""
import asyncio
import os
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("FCS_CASE_PROC", "0")
os.environ["MA_LG_END"] = "budget"
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


Real = L._Agent


class Scripted(Real):
    script = []

    def __init__(self, input, j, problem, shared=False):
        self.name = f"agent{j + 1}"
        self.steps = self.fmt_run = self.fmt_total = self.timeouts = self.cut_n = 0
        self.done, self.exit, self.turns, self.segs = False, "", [], []
        self.generated, self.dropped, self.seen, self.last_seg = 0, 0, 0, None
        self.n_submit = self.n_post = self.idle_fin = 0
        self.last_sha = None
        self.chat = [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
        self.outs = []

    async def think(self):
        if not Scripted.script:
            self.done, self.exit = True, "TokensExceeded"
            return None
        self.steps += 1
        cmd = Scripted.script.pop(0)
        self._pending = (cmd, {"step": self.steps, "text": cmd, "cut": False})
        return cmd

    def observe(self, cmd, out, rnd, order):
        self.outs.append(out["output"])
        Real.observe(self, cmd, out, rnd, order)


def run(script):
    Scripted.script = list(script)
    judged.clear()
    inp = SimpleNamespace(sample=SimpleNamespace(index=0), evaluation=True)
    agents, S, info = asyncio.run(L.episode(inp, [{"role": "user", "content": "problem"}], LABEL))
    return agents[0], info


L.judge_code, L._Agent = fake_judge, Scripted
a, info = run(["printf 'int main(){}\\n' > solution.cpp", FIN, FIN, "true", FIN, FIN, FIN, "true"])
check("an unchanged finish is not judged and the work goes on", len(judged) == 1 and not a.outs[2].startswith(L.A.MARK)
      and "nothing was submitted" in a.outs[2] and "tokens are left" in a.outs[2]
      and "Your work goes on until your token budget is used up" in a.outs[2] and "brute-force" in a.outs[2], (judged, a.outs))
check("three unchanged finishes in a row do not end it: the script runs out (TokensExceeded)", a.steps == 8
      and a.exit == "TokensExceeded" and info["fo_ag_exit_IdleFinish"] == 0.0 and info["fo_ag_exit_Finished"] == 0.0,
      (a.exit, a.steps, info))
check("the unchanged finishes are counted", info["fo_lg_idle_finishes"] == 4.0
      and [bool(t.get("idle_finish")) for t in a.turns] == [False, False, True, False, True, True, True, False],
      (info.get("fo_lg_idle_finishes"), [t.get("idle_finish") for t in a.turns]))
check("the finish reply on a changed file names no way to end", "ends your work" not in a.outs[1], a.outs[1])
check("the harness tag names the protocol", info["lg_harness"] == "long-v1-budget-ctxlow1", info["lg_harness"])
a, info = run(["printf 'int main(){}\\n' > solution.cpp", "submit", "printf 'int main(){return 0;}\\n' > solution.cpp",
               FIN, FIN])
check("the submit reply says how many tokens are left", "of your 100,000 tokens are left" in a.outs[1], a.outs[1])
check("a changed file is submitted, the next unchanged finish goes on", len(judged) == 2 and a.exit == "TokensExceeded"
      and a.steps == 5, (judged, a.exit, a.steps))
pr = L.instance("Read n {and} print it.", "agent1")
check("prompt: no end sentence, the budget ends the work", L._FIN_END.strip() not in pr and "ends your work" not in pr
      and "When your token budget is used up" in pr and "your work then goes on" in pr, pr[-4000:])


class Stub:
    def __init__(self, input, name, think, max_len=None):
        self.input, self.segments, self.last_messages = input, [], None

    async def turn(self, messages, max_tokens):
        self.segments.append(SimpleNamespace(loss_mask=[1] * 10))
        return "<think>r</think>no command here", False


L.Session = Stub
ag = Real(SimpleNamespace(sample=SimpleNamespace(index=0), evaluation=True), 0, "problem")
asyncio.run(ag.think())
check("format error: submit alone, no end sentence", L._FORMAT_ERROR_END_BUDGET in ag.chat[-1]["content"]
      and "not be able to continue" not in ag.chat[-1]["content"], ag.chat[-1]["content"][-300:])
for _ in range(4):
    asyncio.run(ag.think())
check("five format errors in a row do not end it, and the reply says the work goes on", not ag.done and ag.fmt_run == 5
      and "Your work goes on until your token budget is used up" in ag.chat[-1]["content"], (ag.done, ag.exit, ag.fmt_run))
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
