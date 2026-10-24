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
pr = T.team_prompt("Read n {and} print it.", "agent3")
check("You are agent3, one of 4 agents" in pr and "Read n {and} print it." in pr and "{name}" not in pr
      and "{n}" not in pr, "team prompt: name, team size and the problem text (braces kept) filled in")
check(("You have at most %d steps." % A.STEPS in pr) if A.STEPS > 0 else ("You have at most" not in pr),
      f"team prompt names the step limit only when there is one (MA_AG_STEPS={A.STEPS})")
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
    text, rev = T.snapshot(w)
    A.run_command("printf 'broken(\\n' > solution.cpp", w)
    T.commit_as(w, "agent3", 2)
    sh3 = T.blame_shares(w, "solution.cpp", rev)
    check("return x - 1;" in text and abs(sh3.get("agent1", 0) - 0.5) < 1e-9 and "agent3" not in sh3,
          "snapshot: text and blame at the submit commit ignore a later rewrite")
    A.run_command("git checkout -q HEAD~1 -- solution.cpp", w)
    T.commit_as(w, "agent3", 3)
    hook = os.path.join(w, ".git", "hooks", "pre-commit"); mark = os.path.join(w, "HOOK_RAN")
    open(hook, "w").write("#!/bin/sh\ntouch %s\n" % mark); os.chmod(hook, 0o755)
    open(os.path.join(w, "big.txt"), "w").write("x" * (T.MAX_COMMIT_BYTES + 1))
    A.run_command("echo '// note' >> solution.cpp", w)
    T.commit_as(w, "agent4", 1)
    check(not os.path.exists(mark), "an agent-written git hook does not run in the harness commit")
    check("big.txt" not in T._git(w, "ls-files").stdout.split(), "a file over MAX_COMMIT_BYTES stays uncommitted")
    os.remove(os.path.join(w, "big.txt"))
    files = T._git(w, "ls-files").stdout.split()
    check(files == ["solution.cpp"], f"only sources committed (binary left out): {files}")
# repo config an agent writes runs in its sandbox, not with the harness's rights (audit 2026-10-24); run this file with
# FCS_CASE_PROC=1 and =0
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as w, \
        tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as outside:
    T.repo_init(w)
    out_f, out_g = os.path.join(outside, "FILTER_RAN"), os.path.join(outside, "GPG_RAN")
    r = A.run_command(f"git config filter.x.clean 'touch FILTER_IN; touch {out_f}; cat' && echo '*.cpp filter=x' > "
                      f".gitattributes && printf '#!/bin/sh\\ntouch {out_g}\\n' > g.sh && chmod +x g.sh && "
                      f"git config gpg.program $PWD/g.sh && git config commit.gpgsign true && git config core.pager "
                      f"'touch {out_f}' && printf 'int main(){{}}\\n' > solution.cpp", w)
    check(r["returncode"] == 0, "planted filter, gpg and pager config: " + r["output"][-200:])
    T.commit_as(w, "agent1", 1)
    check(os.path.exists(os.path.join(w, "FILTER_IN")), "the planted filter ran, inside the sandbox")
    check(not os.path.exists(out_f) and not os.path.exists(out_g), "the planted filter and gpg.program wrote nothing outside")
    log = T._git(w, "log", "--format=%an %s").stdout
    check("agent1 agent1 step 1" in log, "the harness commit still lands: " + log)
    text, rev = T.snapshot(w)
    check(len(rev) == 40 and T.blame_shares(w, "solution.cpp", rev) == {"agent1": 1.0}, "rev-parse and blame still work")
    r = A.run_command("git -c user.name=me -c user.email=me@x -c commit.gpgsign=false commit -q --allow-empty -m mine && git log -1 --format=%s", w)
    check(r["output"].strip().endswith("mine"), "the agent's own git commit works in the sandbox: " + r["output"][-200:])
    A.run_command("printf x > \"$(printf 'bad\\377name.txt')\" && echo '// 2' >> solution.cpp", w)
    T.commit_as(w, "agent2", 2)
    check("agent2 agent2 step 2" in T._git(w, "log", "--format=%an %s").stdout, "a file name that is not UTF-8 is committed")
print("ALL OK" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
