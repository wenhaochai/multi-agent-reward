"""fcs_judge's work dirs and helper cache follow TMPDIR when FCS_JUDGE_TMP / FCS_JUDGE_CACHE are unset (a job's per-job
scratch dir, removed by its EXIT trap), explicit variables still win, and with neither the paths stay /tmp. A judged
program scores the same under TMPDIR as under /tmp, and its checker lands in TMPDIR's cache."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LABEL = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/0"
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK  " if cond else "FAIL"), name, "" if cond else detail)


def run(env_set: dict, code: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("TMPDIR", "FCS_JUDGE_TMP", "FCS_JUDGE_CACHE")}
    env.update(env_set)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip().splitlines()[-1]


PATHS = "import miles_team.fcs_judge as J; print(J.TMP, J.CACHE)"
scratch = tempfile.mkdtemp(dir="/scratch/gpfs/GROUP/USER/tmp")
check("no TMPDIR: /tmp as before", run({}, PATHS) == "/tmp /tmp/fcs_judge_cache", run({}, PATHS))
check("TMPDIR: both follow it", run({"TMPDIR": scratch}, PATHS) == f"{scratch} {scratch}/fcs_judge_cache")
check("explicit variables win", run({"TMPDIR": scratch, "FCS_JUDGE_TMP": "/a", "FCS_JUDGE_CACHE": "/b"}, PATHS) == "/a /b")
check("TEMP and TMP without TMPDIR: /tmp", run({"TEMP": scratch, "TMP": scratch}, PATHS) == "/tmp /tmp/fcs_judge_cache")
check("fcs_judge_v1 follows TMPDIR the same way", run({"TMPDIR": scratch, "TEMP": "/x"}, PATHS.replace("fcs_judge ", "fcs_judge_v1 "))
      == f"{scratch} {scratch}/fcs_judge_cache" and run({"TMP": scratch}, PATHS.replace("fcs_judge ", "fcs_judge_v1 "))
      == "/tmp /tmp/fcs_judge_cache")
check("FCS_JUDGE_TMP alone sets the cache's parent", run({"FCS_JUDGE_TMP": "/a"}, PATHS) == "/a /a/fcs_judge_cache")

# a program that reads the input and prints a constant: some cases fail, so the scores are not all zero or all one
SRC = "#include <bits/stdc++.h>\nint main(){std::string s; while(std::cin>>s){} std::cout<<0<<std::endl;}\n"
JUDGE = (f"import json, os, miles_team.fcs_judge as J; r = J.judge({LABEL!r}, {SRC!r}); "
         "print(json.dumps({'s': r['score'], 'c': r['cases'], 'st': r['status'], "
         "'left': sorted(p for p in os.listdir(J.TMP) if p.startswith('fcsj_')), 'cache': sorted(os.listdir(J.CACHE))}))")
a = json.loads(run({"FCS_CASE_PROC": "0"}, JUDGE))
b = json.loads(run({"FCS_CASE_PROC": "0", "TMPDIR": scratch}, JUDGE))
check("same verdict under TMPDIR as under /tmp", (a["s"], a["c"], a["st"]) == (b["s"], b["c"], b["st"]), (a, b))
check("checker compiled into TMPDIR's cache", any(p.startswith("0_chk_") and not p.endswith(".lock") for p in b["cache"]),
      b["cache"])
check("no work dir left in TMPDIR", b["left"] == [], b["left"])
if os.path.basename(scratch).startswith("tmp"):   # this test's own mkdtemp dir only
    shutil.rmtree(scratch, ignore_errors=True)
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
