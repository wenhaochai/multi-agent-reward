"""judge.judge_full on runs that fail, on val problems 6 (classic, 1 s) and 10 (interactive, 512 MiB): no code, a
compile error, a time-out, a crash, too much output, an empty output, too much memory, an empty interactive program;
each scores 0 and its case_info says why. Then the w93 smoke's best program for problem 6 scores its stored
provisional cases (a required fixture)."""
import json
import os
import re

from helpers import check, done, env

env()
from flint.judge import judge_full  # noqa: E402

P = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/"
W93 = "/scratch/gpfs/GROUP/USER/project/flint-eval/runs/smoke_tc_w93/traces/agent_419086.jsonl"


def zero(r) -> bool:
    return r["status"] == "done" and r["score"] == 0.0 and r["cases"] == [0.0] * len(r["cases"])


def timed_out(info: str) -> bool:
    """Killed at 2x the time limit, or stopped by RLIMIT_CPU (SIGXCPU or SIGKILL) after the limit."""
    m = re.match(r"run: code=\S+ cpu=([\d.]+)/([\d.]+) killed=(\w+)", info)
    return bool(m) and (m.group(3) == "True" or float(m.group(1)) > float(m.group(2)))


r = judge_full(P + "6", "")
check("no code: no run, 10 zeros", r["status"] == "no code" and r["cases"] == [0.0] * 10 and not r["case_info"], r)
r = judge_full(P + "6", "int main(){X;}\n")
check("compile error: 10 zeros and the compiler output", r["status"] == "compile error" and r["cases"] == [0.0] * 10
      and "X" in r["msg"], r["status"])
PROGS = {"time-out": ("int main(){volatile long x=0; while(1) x++;}\n", timed_out),
         "crash": ("int main(){volatile int*p=0; *p=1;}\n", lambda i: i.startswith("run: code=-11 ")),
         "too much output": ("#include <cstdio>\n#include <cstring>\nint main(){static char b[1<<20]; memset(b,'1',"
                             "sizeof b); for(int i=0;i<200;i++) fwrite(b,1,sizeof b,stdout);}\n",
                             lambda i: i.startswith("run: code=-25 "))}   # SIGXFSZ past 128 MiB
for what, (src, why) in PROGS.items():
    r = judge_full(P + "6", src)
    check(f"classic {what}: 0, and why", zero(r) and all(map(why, r["case_info"])), r["case_info"])
r = judge_full(P + "6", "int main(){}\n")
check("classic empty output: the checker rejects it", zero(r)
      and all(re.match(r"chk=[12] ", i) for i in r["case_info"]), r["case_info"][:2])
r = judge_full(P + "10", "#include <bits/stdc++.h>\nint main(){std::vector<char> v(800u<<20); "
                         "for(size_t i=0;i<v.size();i+=4096) v[i]=1; printf(\"%d\\n\", v[12345]);}\n")
check("interactive memory over the limit: 0, the peak above 512 MiB", zero(r)
      and all(int(re.search(r"rss=(-?\d+)MiB/512", i).group(1)) > 512 for i in r["case_info"]), r["case_info"][:2])
r = judge_full(P + "10", "int main(){}\n")
check("interactive empty program: the interactor's verdict, 0", zero(r)
      and all(re.search(r"sol code=0 .*; inter code=-?\d+ killed=False", i) for i in r["case_info"]),
      r["case_info"][:2])

check("w93 trace present (a required fixture)", os.path.exists(W93), W93)
if os.path.exists(W93):   # the w93 smoke's pid6 best: provisional (even) cases scored as stored
    best = [json.loads(l)["best"] for l in open(W93) if json.loads(l)["label"].endswith("/6")][0]
    r = judge_full(P + "6", best["code"])
    check("w93 pid6 best: its provisional score", r["status"] == "done"
          and abs(sum(r["cases"][0::2]) / 5 - best["prov"]) < 1e-9, (r["cases"], best["prov"]))
done()
