"""fcs_agent_team without GPUs: the team prompt, the harness's commits and git blame credit (run in miles.sif)."""
import os, sys, tempfile
from miles_team import fcs_agent_team as T
from miles_team import fcs_agent as A
ok = True
def check(c, what):
    global ok
    print(("PASS " if c else "FAIL ") + what); ok &= bool(c)
check("one of {n} agents" in T.TEAM_INSTANCE and "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`." in T.TEAM_INSTANCE
      and "&& cat solution.cpp" not in T.TEAM_INSTANCE, "team instance template")
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as w:
    T.repo_init(w)
    A.run_command("printf 'int main(){\\n  return 0;\\n}\\n' > solution.cpp && g++ -o solution solution.cpp", w)
    T.commit_as(w, "agent1", 1)
    A.run_command("sed -i 's/return 0;/int x = 1;\\n  return x - 1;/' solution.cpp", w)
    T.commit_as(w, "agent2", 1)
    sh = T.blame_shares(w)
    print("   shares", sh)
    check(abs(sh.get("agent1", 0) - 0.5) < 1e-9 and abs(sh.get("agent2", 0) - 0.5) < 1e-9, "blame: 2 of 4 lines each")
    log = T._git(w, "log", "--format=%an %s").stdout
    check("agent2 agent2 step 1" in log and "agent1 agent1 step 1" in log, "commits under agent names")
    files = T._git(w, "ls-files").stdout.split()
    check(files == ["solution.cpp"], f"only sources committed (binary left out): {files}")
print("ALL OK" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
