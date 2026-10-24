"""The long game's finish command (fcs_agent_long.finish): `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` submits
solution.cpp and the agent goes on; on an unchanged solution.cpp it ends the agent's work; with no solution.cpp it
says so and goes on. Scripted agent, stubbed judge, real problem 0 for the case count."""
import asyncio
import glob
import json
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


SECRET = "wrong answer expected 4242, found 7"   # a checker line that quotes the expected output


async def fake_judge(label, code):
    judged.append(code)
    await asyncio.sleep(0.05)
    return {"cases": [0.5] * 10, "status": "ok", "infra_error": "", "msg": "", "case_info": [SECRET] * 10}


Base = L._Agent


class Scripted(Base):
    script = []

    def __init__(self, input, j, problem):
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


def run(script):
    Scripted.script = list(script)
    judged.clear()
    inp = SimpleNamespace(sample=SimpleNamespace(index=0), evaluation=True)
    agents, S, info = asyncio.run(L.episode(inp, [{"role": "user", "content": "problem"}], LABEL))
    run.info = info
    return agents[0]


def last_trace() -> dict:
    lines = [l for f in glob.glob(os.path.join(os.environ["MA_AG_TRACE_DIR"], "*.jsonl")) for l in open(f)]
    return json.loads(lines[-1])


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
# judge time, every case and the judge's per-case lines reach the trace, never the agent (they would leak final tests)
a = run(["printf 'int main(){}\\n' > solution.cpp", "submit"])
sub = last_trace()["submissions"][0]
check("trace has judge_s, cases, case_info", sub.get("judge_s", 0) >= 0.05 and sub.get("cases") == [0.5] * 10
      and sub.get("case_info") == [SECRET] * 10, sub)
check("agent never sees case_info", not any("4242" in o for o in a.outs), a.outs)
check("fo_lg_judge_min in info", abs(run.info["fo_lg_judge_min"] - sub["judge_s"] / 60) < 1e-9, run.info)
check("judge time is not command time", run.info["fo_ag_run_min"] * 60 < sub["judge_s"] + 5, run.info)
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
