"""The writer of SCORES.md and BOARD.md does not follow what the agent puts at those paths (audit 2026-10-24): symlinks
out of the repository are replaced, not written through; a directory is replaced; no temp file is left; nothing
raises. A symlinked or unreadable (chmod 000) solution.cpp is not read for submit, the finish command or the
auto-judge. Scripted agent, stubbed judge."""
import asyncio
import os
import tempfile

from helpers import FIN, LABEL, MSG, check, done, env, inp, scripted, sessions

env(MA_LG_N=1)
from flint import game as G  # noqa: E402

outside = tempfile.mkdtemp(dir=os.environ["TMPDIR"])
victim = os.path.join(outside, "v")
open(victim, "w").write("precious\n")
judged = []


async def fake_judge(label, code):
    judged.append(code)
    return {"cases": [0.5] * 10, "status": "ok", "infra_error": "", "msg": ""}


Scripted = scripted(G)
G.judge_code, G.Agent = fake_judge, Scripted
Scripted.script = [f"ln -sf {victim} SCORES.md; ln -sf {victim}.b BOARD.md; printf 'int main(){{}}\\n' > solution.cpp",
                   "submit",
                   "ls -la SCORES.md BOARD.md; ls -a | grep -c '^\\.SCORES\\|^\\.BOARD' ; cat SCORES.md | head -3",
                   "rm -f SCORES.md && mkdir -p SCORES.md/x && echo hi > SCORES.md/x/y",
                   "post hello",
                   "ls -la SCORES.md; cat SCORES.md | head -1",
                   f"rm solution.cpp && ln -s {victim} solution.cpp",
                   FIN]
agents, S, info = asyncio.run(G.episode(inp(), MSG, LABEL, sessions(None)))
a = agents[0]
check("the symlink target is unchanged", open(victim).read() == "precious\n")
check("no file appears through the BOARD.md symlink", not os.path.exists(victim + ".b"))
check("SCORES.md became a regular file with the submission", a.outs[2].lstrip().startswith("-") and "\n0\n" in a.outs[2]
      and "# Submissions" in a.outs[2], a.outs[2])
check("a directory at SCORES.md is replaced by the file", a.outs[5].lstrip().startswith("-")
      and "# Submissions" in a.outs[5], a.outs[5])
check("a symlink out of the repository is not read as solution.cpp", len(judged) == 1 and "no solution.cpp" in a.outs[7],
      (judged, a.outs[7]))
check("episode ran to the end without an exception", a.steps == 8 and S == 0.5, (a.steps, S))

# an unreadable solution.cpp (chmod 000; the harness has the same uid) is "no file" for submit, the finish command and
# the auto-judge at the budget's end, and does not raise PermissionError out of episode() (review 2026-10-24)
judged.clear()
Scripted.script = ["printf 'int main(){return 0;}\\n' > solution.cpp && chmod 000 solution.cpp", "submit", FIN]
try:
    agents, S, info = asyncio.run(G.episode(inp(), MSG, LABEL, sessions(None)))
    a, err = agents[0], None
except Exception as e:  # noqa: BLE001
    err = e
check("chmod 000 solution.cpp: episode ends without an exception", err is None, repr(err))
if err is None:
    check("chmod 000: submit says no file", "no file solution.cpp" in a.outs[1], a.outs[1])
    check("chmod 000: the finish command says no solution.cpp", "no solution.cpp" in a.outs[2], a.outs[2])
    check("chmod 000: nothing judged, S 0, S_auto 0", not judged and S == 0.0 and info["fo_lg_S_auto"] == 0.0
          and info["fo_lg_auto_judged"] == 0.0, (judged, S, info))
done()
