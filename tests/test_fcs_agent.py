"""fcs_agent without GPUs: the sandboxed shell, the prompt pieces and the action parser (run inside miles.sif with
PYTHONPATH=<miles worktree>:<this repo>; MA_AG_TIMEOUT=3)."""
import json, os, sys, tempfile, time
os.environ.setdefault("MA_AG_TIMEOUT", "3")
from miles_team import fcs_agent as A
from miles_team import fcs_judge as J

ok = True
def check(c, what):
    global ok
    print(("PASS " if c else "FAIL ") + what); ok &= bool(c)

print("landlock ABI", J.LANDLOCK_ABI)
row = json.loads(open("/scratch/gpfs/GROUP/USER/project/miles-q38-build/data/fcs_val172_fo_par.jsonl").readline())
pt = A.problem_text(row["metadata"]["messages"])
check(not pt.startswith("You are a competitive programmer") and not pt.endswith("Generate solution code:") and "## Example" in pt, "problem text stripped")
hidden = os.path.join(row["label"], "testdata")
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as w:
    r = A.run_command("echo hello && pwd", w); check(r["returncode"] == 0 and "hello" in r["output"], "echo")
    r = A.run_command("sleep 1.2", w); check(1.1 < r["seconds"] < 3, f"run time measured: {r['seconds']:.2f} s")
    r = A.run_command("cat > a.cpp <<'EOF'\n#include <cstdio>\nint main(){int a,b;scanf(\"%d %d\",&a,&b);printf(\"%d\\n\",a+b);}\nEOF\ng++ -O2 -std=gnu++17 -o a a.cpp && echo '2 3' | ./a", w)
    check(r["returncode"] == 0 and r["output"].strip().endswith("5"), "compile and run: " + r["output"][-80:])
    r = A.run_command(f"ls {hidden} | head -2; cat {hidden}/* 2>&1 | head -2", w)
    check("Permission denied" in r["output"] or "cannot open" in r["output"], "hidden tests unreadable: " + r["output"][:120])
    r = A.run_command("python3 -c \"import socket; s=socket.socket(); s.settimeout(2); s.connect(('1.1.1.1', 80)); print('CONNECTED')\"", w)
    check("CONNECTED" not in r["output"], "no TCP: " + r["output"][-120:])
    r = A.run_command("echo x > /tmp/agent_probe_$$ && echo WROTE", w)
    check("WROTE" not in r["output"], "no /tmp write")
    t = time.time(); r = A.run_command("sleep 100", w)
    check(r["returncode"] == -1 and "timed out" in r["exception_info"] and time.time() - t < 10, "timeout kills")
    r = A.run_command("(sleep 30; touch late_file) & echo started", w); time.sleep(4)
    check(not os.path.exists(os.path.join(w, "late_file")), "background child killed")
    r = A.run_command("python3 -c \"import os,signal; os.kill(os.getppid(), 0); print('PPID_OK')\" ; kill -0 1 && echo INIT_OK", w)
    print("   signal probe:", r["output"].strip()[:120])
    r = A.run_command(f"echo {A.MARK} && cat a.cpp", w)
    lines = r["output"].lstrip().splitlines(keepends=True)
    check(lines[0].strip() == A.MARK and "int main" in "".join(lines[1:]), "submit marker")
obs = A.observation({"output": "x" * 12000, "returncode": 0, "exception_info": ""})
check("<elided_chars>\n2000 characters elided" in obs, "elision")
txt = "THOUGHT: compile.\n\n```mswea_bash_command\ng++ a.cpp\n```"
check(A.ACTION_RE.findall(txt) == ["g++ a.cpp"], "action regex")
check(len(A.ACTION_RE.findall(txt + "\n" + txt)) == 2, "two actions found")
print("ALL OK" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
