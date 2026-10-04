"""Checks of miles_team/fcs_judge.py (audit 2026-10-04): Landlock is active; untrusted code cannot read test
answers (neither by #include at compile time nor by fopen at run time), other processes' environments, or write to
scratch; ratio parsing (val: the official regex, train: full float clamped to [0, 1]); an interactive-program MLE is
scored from the peak RSS. Run in miles.sif:  python3 tools/judge_sandbox_test.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import miles_team.fcs_judge as J  # noqa: E402

VAL0 = J.FCS_ROOT / "problems" / "0"
ANS = sorted((VAL0 / "testdata").glob("*.ans"))[0]
SCRATCH_PROBE = Path("/scratch/gpfs/GROUP/USER/project/miles-q38-build/runs/.sandbox_probe_should_not_exist")
fails = []


def check(name, cond, detail=""):
    print(("[ok] " if cond else "[FAIL] ") + name + (f": {detail}" if detail else ""), flush=True)
    if not cond:
        fails.append(name)


check("landlock active", J.LANDLOCK_ABI > 0, f"ABI {J.LANDLOCK_ABI}")

# 1) compile-time exfiltration: #include a test answer -> must not compile under the sandbox
src_inc = f'#include <cstdio>\nconst char* a = R"(\n#include "{ANS}"\n)";\nint main(){{puts(a);}}\n'
src_inc = f'#include "{ANS}"\nint main(){{return 0;}}\n'
r = J.judge(str(VAL0), src_inc)
check("#include of a test answer does not compile", r["status"] == "compile error", r["status"] + " " + r.get("msg", "")[-160:])

# 2) run-time attacks: read an answer file, another process's environ, write to scratch
attack = r'''
#include <cstdio>
#include <unistd.h>
int main(){
  FILE* f = fopen("%s", "r"); printf("ans=%%d\n", f != nullptr);
  char p[64]; snprintf(p, sizeof p, "/proc/%%d/environ", (int)getppid());
  FILE* e = fopen(p, "r"); printf("environ=%%d\n", e != nullptr);
  FILE* w = fopen("%s", "w"); printf("write=%%d\n", w != nullptr);
  FILE* s = fopen("/proc/self/status", "r"); printf("self=%%d\n", s != nullptr);
  FILE* u = fopen("/dev/urandom", "r"); printf("urandom=%%d\n", u != nullptr);
  return 0;
}
''' % (ANS, SCRATCH_PROBE)
SCRATCH_PROBE.unlink(missing_ok=True)
work = Path(tempfile.mkdtemp(prefix="sbt_"))
(work / "a.cpp").write_text(attack)
subprocess.run(J.COMPILE + ["-o", str(work / "a"), str(work / "a.cpp")], check=True)
out = subprocess.run([str(work / "a")], capture_output=True, text=True, cwd=work, env=J._CLEAN_ENV,
                     preexec_fn=J._limits(2, 256 << 20, sandbox=J._program_sandbox(work / "a"))).stdout
got = dict(l.split("=") for l in out.split())
check("program cannot read a test answer", got.get("ans") == "0", out.replace("\n", " "))
check("program cannot read another process's environ", got.get("environ") == "0")
check("program cannot write to scratch", got.get("write") == "0" and not SCRATCH_PROBE.exists())
check("program can read /proc/self and /dev/urandom", got.get("self") == "1" and got.get("urandom") == "1")
out2 = subprocess.run([str(work / "a")], capture_output=True, text=True, cwd=work).stdout
check("control: without the sandbox the same program can read the answer", "ans=1" in out2)
SCRATCH_PROBE.unlink(missing_ok=True)  # the unsandboxed control did create it

# 3) a normal program still compiles, runs and is scored (A+B style: echo the input sum is problem-specific, so only
#    check that a program that reads stdin and prints something gets a "done" status and per-case results)
r = J.judge(str(VAL0), "#include <cstdio>\nint main(){int c; while((c=getchar())!=EOF){} puts(\"0\");}\n")
check("normal program is judged", r["status"] == "done" and len(r["cases"]) > 0 and not r["infra"], r["status"])

# 4) ratio parsing
check("val keeps the official regex (5e-05 -> 5)", J._ratio("points 1 Ratio: 5e-05", True, True, False) == 5.0)
check("train parses exponents", abs(J._ratio("points 1 Ratio: 5e-05", True, True, True) - 5e-05) < 1e-12)
check("train clamps above 1 and below 0", J._ratio("Ratio: 1.7", True, True, True) == 1.0
      and J._ratio("Ratio: -0.3", True, True, True) == 0.0 and J._ratio("Ratio: nan", True, True, True) == 0.0)
check("untrusted exit ignores the ratio", J._ratio("Ratio: 0.9", False, False, True) == 0.0)

# 5) interactive MLE from the peak RSS (a program that touches 2 x ML)
inter = next(p for p in sorted((J.FCS_ROOT / "problems").iterdir(), key=lambda p: (len(p.name), p.name))
             if p.is_dir() and (p / "config.yaml").exists() and J.load_problem(p)["interactive"])
pr = J.load_problem(inter)
ml = J.to_bytes(pr["cases"][0][3])
hog = ("#include <cstdio>\n#include <cstdlib>\nint main(){size_t n=%d; volatile char* p=(volatile char*)malloc(n);"
       " if(!p) return 3; for(size_t i=0;i<n;i+=4096) p[i]=1; long s=0; for(size_t i=0;i<n;i+=4096) s+=p[i];"
       " printf(\"%%ld\\n\", s); return 0;}\n" % (2 * ml))
r = J.judge(str(inter), hog)
check(f"interactive MLE from peak RSS (problem {inter.name}, ML {ml >> 20} MiB)",
      r["status"] == "done" and max(r["cases"]) == 0.0 and any("rss=" in i for i in r["case_info"]),
      (r.get("case_info") or [""])[0][:120])

# 6) process limit: ~128 new tasks (a fork bomb is contained), a few threads still work
thr = r"""
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv){ int n = atoi(argv[1]), ok = 0; std::vector<std::thread> v;
  try { for (int i = 0; i < n; i++) { v.emplace_back([]{ volatile long s = 0; for (int k = 0; k < 2000000; k++) s += k; }); ok++; } }
  catch (...) { }
  for (auto& t : v) t.join(); printf("%d\n", ok); return 0; }
"""
(work / "t.cpp").write_text(thr)
subprocess.run(J.COMPILE + ["-pthread", "-o", str(work / "t"), str(work / "t.cpp")], check=True)
run = lambda n: int(subprocess.run([str(work / "t"), str(n)], capture_output=True, text=True, cwd=work, env=J._CLEAN_ENV,
                                   preexec_fn=J._limits(10, 2 << 30, stack_b=8 << 20,
                                                        sandbox=J._program_sandbox(work / "t"),
                                                        nproc=J._nproc_limit())).stdout or -1)
# (an 8 MiB stack isolates the process limit: with stack = address space, as the judge sets them like the official
# engine, every new thread reserves the whole address space and no thread starts at all)
few, many = run(8), run(400)
check("a program may start a few threads", few == 8, str(few))
check("a program cannot start ~400 threads (procLimit ~128)", 0 < many <= J.NPROC_MARGIN + 8, str(many))

print("ALL OK" if not fails else f"FAILED: {fails}")
