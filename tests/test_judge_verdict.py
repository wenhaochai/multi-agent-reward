"""fcs_judge.verdict reads a case's verdict from the case_info line the judge wrote, and fcs_rm.judge_full returns one
per case ("verdicts"): unit tests on the line formats, then real judges on val problems 6 (classic, 1 s) and 10
(interactive, 512 MiB) for each verdict, and the w93 pid6 best program when its trace is there."""
import json
import os
import sys
import tempfile

os.environ["TMPDIR"] = tempfile.mkdtemp(dir="/scratch/gpfs/GROUP/USER/tmp")   # 128 MiB outputs off node /tmp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from miles_team.fcs_judge import verdict  # noqa: E402
from miles_team.fcs_rm import judge_full  # noqa: E402

P = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/"
W93 = "/scratch/gpfs/GROUP/USER/project/flint-eval/runs/smoke_tc_w93/traces/agent_419086.jsonl"
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK  " if cond else "FAIL"), name, "" if cond else detail)


UNIT = [("chk=7 cpu=0.00/1.0 points 0.0 Ratio: 0.000000000000", "OK"), ("chk=0 cpu=0.01/1.0 ok", "OK"),
        ("chk=1 cpu=0.01/1.0 wrong answer Edge not exist", "WA"), ("chk=2 cpu=0.01/1.0 wrong output format", "WA"),
        ("chk=3 cpu=0.01/1.0 FAIL bad answer file", "JUDGE"), ("chk=-9 cpu=0.01/1.0 ", "JUDGE"),
        ("run: code=None cpu=1.98/1.0 killed=True", "TLE"), ("run: code=0 cpu=1.30/1.0 killed=False", "TLE"),
        ("run: code=-24 cpu=2.00/1.0 killed=False", "TLE"), ("run: code=-11 cpu=0.01/1.0 killed=False", "RE"),
        ("run: code=1 cpu=0.01/1.0 killed=False", "RE"), ("run: code=-25 cpu=0.30/1.0 killed=False", "OLE"),
        ("run: code=0 cpu=0.01/1.0 killed=False", "OLE"), ("INFRA exception: [Errno 12] Cannot allocate memory", "INFRA"),
        ("sol code=0 cpu=0.10/2.0 rss=800MiB/512 killed=False; inter code=1 killed=False wrong answer", "MLE"),
        ("sol code=None cpu=0.00/2.0 rss=3MiB/512 killed=True; inter code=1 killed=False", "TLE"),
        ("sol code=-11 cpu=0.00/2.0 rss=3MiB/512 killed=False; inter code=0 killed=False", "RE"),
        ("sol code=-13 cpu=0.00/2.0 rss=3MiB/512 killed=False; inter code=1 killed=False wrong", "WA"),
        ("sol code=0 cpu=0.10/2.0 rss=3MiB/512 killed=False; inter code=7 killed=False points 0.5", "OK"),
        ("sol code=0 cpu=0.10/2.0 rss=3MiB/512 killed=False; inter code=2 killed=False", "WA"),
        ("sol code=0 cpu=0.10/2.0 rss=3MiB/512 killed=False; inter code=None killed=True", "JUDGE"),
        ("sol code=0 cpu=0.10/2.0 rss=3MiB/512 killed=False; inter code=3 killed=False fail", "JUDGE"),
        ("sol code=0 cpu=0.00/2.0 rss=13MiB/512 killed=False; inter code=-13 killed=False ", "WA"),
        ("something else", "JUDGE")]
bad = [(i, verdict(i), v) for i, v in UNIT if verdict(i) != v]
check("verdict() on every line format", not bad, bad)

r = judge_full(P + "6", "")
check("no code: '-' per case", r["verdicts"] == ["-"] * 10, r["verdicts"])
r = judge_full(P + "6", "int main(){X;}\n")
check("compile error: CE per case", r["status"] == "compile error" and r["verdicts"] == ["CE"] * 10, r["verdicts"])
PROGS = {"TLE": "int main(){volatile long x=0; while(1) x++;}\n",
         "RE": "int main(){volatile int*p=0; *p=1;}\n",
         "OLE": "#include <cstdio>\n#include <cstring>\nint main(){static char b[1<<20]; memset(b,'1',sizeof b); "
                "for(int i=0;i<200;i++) fwrite(b,1,sizeof b,stdout);}\n"}
for want, src in PROGS.items():
    r = judge_full(P + "6", src)
    check(f"classic {want}", r["status"] == "done" and r["verdicts"] == [want] * 10, (r["verdicts"], r["case_info"][:2]))
r = judge_full(P + "6", "int main(){}\n")
check("classic empty output: checked (WA)", set(r["verdicts"]) == {"WA"}, (r["verdicts"], r["case_info"][:2]))
r = judge_full(P + "10", "#include <bits/stdc++.h>\nint main(){std::vector<char> v(800u<<20); "
                         "for(size_t i=0;i<v.size();i+=4096) v[i]=1; printf(\"%d\\n\", v[12345]);}\n")
check("interactive MLE", r["status"] == "done" and r["verdicts"] == ["MLE"] * len(r["cases"]),
      (r["verdicts"], r["case_info"][:2]))
r = judge_full(P + "10", "int main(){}\n")
check("interactive empty program: the interactor's verdict", r["status"] == "done"
      and set(r["verdicts"]) <= {"WA", "OK"}, (r["verdicts"], r["case_info"][:2]))

if os.path.exists(W93):   # the w93 smoke's pid6 best: provisional (even) cases OK, OK, WA, OK, WA
    best = [json.loads(l)["best"] for l in open(W93) if json.loads(l)["label"].endswith("/6")][0]
    r = judge_full(P + "6", best["code"])
    check("w93 pid6 best: provisional verdicts", r["verdicts"][0::2] == ["OK", "OK", "WA", "OK", "WA"]
          and abs(sum(r["cases"][0::2]) / 5 - best["prov"]) < 1e-9, (r["verdicts"], r["cases"], best["prov"]))
else:
    print("SKIP w93 pid6 (no trace)")
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
