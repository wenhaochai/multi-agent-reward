"""The long game's finish command (fcs_agent_long.finish): `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` submits
solution.cpp and the agent goes on; on an unchanged solution.cpp it ends the agent's work; with no solution.cpp it
says so and goes on. Scripted agent, stubbed judge, real problem 0 for the case count."""
import asyncio
import os
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("FCS_CASE_PROC", "0")
os.environ.setdefault("MA_LG_N", "1")
os.environ.setdefault("MA_LG_TOKENS", "100000")
os.environ["TMPDIR"] = tempfile.mkdtemp()
os.environ["MA_AG_TRACE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from miles_team import fcs_agent_long as L  # noqa: E402

LABEL = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/0"
FIN = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
judged = []


async def fake_judge(label, code):
    judged.append(code)
    return {"cases": [0.9 if "return 1" in code else 0.5] * 10, "status": "ok", "infra_error": "", "msg": ""}


Base = L._Agent


class Scripted(Base):
    script = []

    def __init__(self, input, j, problem, shared=False):
        self.name = f"agent{j + 1}"
        self.steps = self.fmt_run = self.fmt_total = self.timeouts = self.cut_n = 0
        self.done, self.exit, self.turns, self.segs = False, "", [], []
        self.generated, self.dropped, self.seen, self.last_seg = 0, 0, 0, None
        self.n_submit = self.n_post = 0
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
        Base.observe(self, cmd, out, rnd, order)


L.judge_code = fake_judge
L._Agent = Scripted
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK  " if cond else "FAIL"), name, "" if cond else detail)


def run(script, full=False):
    Scripted.script = list(script)
    judged.clear()
    inp = SimpleNamespace(sample=SimpleNamespace(index=0), evaluation=True)
    agents, S, info = asyncio.run(L.episode(inp, [{"role": "user", "content": "problem"}], LABEL))
    return (agents[0], S, info) if full else agents[0]


a = run([FIN, "printf 'int main(){}\\n' > solution.cpp", FIN, FIN])
check("no solution.cpp: told, goes on", "no solution.cpp" in a.outs[0] and a.steps == 4, a.outs)
check("finish submits solution.cpp", len(judged) == 1 and a.n_submit == 1 and "Your work goes on" in a.outs[2], a.outs)
check("finish on unchanged file ends", a.exit == "Finished" and a.done, a.exit)

a = run(["printf 'int main(){}\\n' > solution.cpp", FIN, "printf 'int main(){return 0;}\\n' > solution.cpp", FIN, FIN,
         "true"])
check("changed file is submitted again", len(judged) == 2 and a.n_submit == 2, judged)
check("then unchanged ends, nothing after", a.exit == "Finished" and a.steps == 5, (a.exit, a.steps))

a = run(["printf 'int main(){}\\n' > solution.cpp", "submit", FIN])
check("explicit submit counts as the last submission", len(judged) == 1 and a.exit == "Finished", (judged, a.exit))

a = run(["printf 'int main(){}\\n' > solution.cpp", "submit sol2.cpp", FIN])
check("submit of another file does not", len(judged) == 1 and a.exit != "Finished" and "Your work goes on" in a.outs[2],
      (judged, a.exit, a.outs))
# the budget runs out (the scripted agent's TokensExceeded) with a better solution.cpp unsubmitted: S keeps the rule,
# fo_lg_S_auto judges it once apart
a, S, info = run(["printf 'int main(){}\\n' > solution.cpp", "submit", "printf 'int main(){return 1;}\\n' > solution.cpp"],
                 True)
check("budget end: S counts only the submission", abs(S - 0.5) < 1e-9 and len(judged) == 2, (S, judged))
check("budget end: the unsubmitted file is judged apart", abs(info["fo_lg_S_auto"] - 0.9) < 1e-9
      and info["fo_lg_auto_judged"] == 1.0 and info["fo_lg_submissions"] == 1, info)
a, S, info = run(["printf 'int main(){return 1;}\\n' > solution.cpp"], True)
check("budget end, no submission: S 0, S_auto > 0", S == 0.0 and info["fo_lg_S_auto"] > 0, (S, info))
a, S, info = run(["printf 'int main(){}\\n' > solution.cpp", "submit"], True)
check("budget end, file already submitted: no extra judge", len(judged) == 1 and info["fo_lg_auto_judged"] == 0.0
      and info["fo_lg_S_auto"] == S, (judged, info))
a, S, info = run(["printf 'int main(){return 1;}\\n' > solution.cpp", FIN, "printf 'int main(){}\\n' > solution.cpp",
                  "submit", FIN], True)
check("finished: no extra judge, S_auto = S", a.exit == "Finished" and info["fo_lg_auto_judged"] == 0.0
      and info["fo_lg_S_auto"] == S and info["fo_lg_shared_case"] == 0.0, info)

# N = 2: agent2 posts in the round in which agent1 runs the finish command on an unchanged file; the team update
# must not hide agent1's MARK, so agent1 ends (fcs_agent_long.episode's `ends`)
L.N = 2


class Pair(Scripted):
    scripts = {}

    async def think(self):
        sc = Pair.scripts[self.name]
        if not sc:
            self.done, self.exit = True, "TokensExceeded"
            return None
        self.steps += 1
        cmd = sc.pop(0)
        self._pending = (cmd, {"step": self.steps, "text": cmd, "cut": False})
        return cmd


L._Agent = Pair
Pair.scripts = {"agent1": ["printf 'int main(){}\\n' > solution.cpp", FIN, FIN, "true", "true"],
                "agent2": ["post a", "post b", "post c", "post d", "true"]}
judged.clear()
inp = SimpleNamespace(sample=SimpleNamespace(index=0), evaluation=True)
agents, S, info = asyncio.run(L.episode(inp, [{"role": "user", "content": "problem"}], LABEL))
a1 = agents[0]
check("N=2: finish on an unchanged file ends the agent despite a team update", a1.done and a1.exit == "Finished"
      and a1.steps == 3, (a1.exit, a1.steps, a1.outs))
L.N, L._Agent = 1, Scripted
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
