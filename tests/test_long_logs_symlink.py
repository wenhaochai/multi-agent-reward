"""fcs_agent_long's SCORES.md and BOARD.md writer does not follow what the agent puts at those paths (audit
2026-10-24): symlinks out of the repository are replaced, not written through; a directory is replaced; no temp file is
left; nothing raises. Also a symlinked or unreadable (chmod 000) solution.cpp is not read for submit, the finish
command or the auto-judge. Scripted agent, stubbed judge."""
import asyncio
import os
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("FCS_CASE_PROC", "0")
os.environ["MA_LG_N"] = "1"
os.environ["TMPDIR"] = tempfile.mkdtemp()
os.environ["MA_AG_TRACE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from miles_team import fcs_agent_long as L  # noqa: E402

LABEL = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/0"
outside = tempfile.mkdtemp(dir=os.environ["TMPDIR"])
victim = os.path.join(outside, "v")
open(victim, "w").write("precious\n")
judged, seen = [], {}


async def fake_judge(label, code):
    judged.append(code)
    return {"cases": [0.5] * 10, "status": "ok", "infra_error": "", "msg": ""}


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


FIN = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
Scripted.script = [f"ln -sf {victim} SCORES.md; ln -sf {victim}.b BOARD.md; printf 'int main(){{}}\\n' > solution.cpp",
                   "submit",
                   "ls -la SCORES.md BOARD.md; ls -a | grep -c '^\\.SCORES\\|^\\.BOARD' ; cat SCORES.md | head -3",
                   "rm -f SCORES.md && mkdir -p SCORES.md/x && echo hi > SCORES.md/x/y",
                   "post hello",
                   "ls -la SCORES.md; cat SCORES.md | head -1",
                   f"rm solution.cpp && ln -s {victim} solution.cpp",
                   FIN]
inp = SimpleNamespace(sample=SimpleNamespace(index=0), evaluation=True)
agents, S, info = asyncio.run(L.episode(inp, [{"role": "user", "content": "problem"}], LABEL))
a = agents[0]
check("the symlink target is unchanged", open(victim).read() == "precious\n")
check("no file appears through the BOARD.md symlink", not os.path.exists(victim + ".b"))
check("SCORES.md became a regular file with the submission", a.outs[2].lstrip().startswith("-") and "\n0\n" in a.outs[2]
      and "# Submissions" in a.outs[2], a.outs[2])
check("a directory at SCORES.md is replaced by the file", a.outs[5].lstrip().startswith("-") and "# Submissions" in a.outs[5],
      a.outs[5])
check("a symlink out of the repository is not read as solution.cpp", len(judged) == 1 and "no solution.cpp" in a.outs[7],
      (judged, a.outs[7]))
check("episode ran to the end without an exception", a.steps == 8 and S == 0.5, (a.steps, S))

# an unreadable solution.cpp (chmod 000; the harness has the same uid) is "no file" for submit, the finish command and
# the auto-judge at a budget end, and does not raise PermissionError out of episode() (review 2026-10-24)
judged.clear()
Scripted.script = ["printf 'int main(){return 0;}\\n' > solution.cpp && chmod 000 solution.cpp", "submit", FIN]
try:
    agents, S, info = asyncio.run(L.episode(inp, [{"role": "user", "content": "problem"}], LABEL))
    a, err = agents[0], None
except Exception as e:  # noqa: BLE001
    err = e
check("chmod 000 solution.cpp: episode ends without an exception", err is None, repr(err))
if err is None:
    check("chmod 000: submit says no file", "no file solution.cpp" in a.outs[1], a.outs[1])
    check("chmod 000: the finish command says no solution.cpp", "no solution.cpp" in a.outs[2], a.outs[2])
    check("chmod 000: nothing judged, S 0, S_auto 0", not judged and S == 0.0 and info["fo_lg_S_auto"] == 0.0
          and info["fo_lg_auto_judged"] == 0.0, (judged, S, info))
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
