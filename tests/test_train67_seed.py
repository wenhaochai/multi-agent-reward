"""train67 seed=N (rl/rl_job.sh train67* ... seed=N, default 67): (a) rl/train67_data.py: without a seed and with seed 67
the file and printed line are byte-identical to 60e6d34's (all 67 problems and signal34); seed 68 gives another order,
still each epoch a permutation of the set (epoch e: random.Random(68 + e)) and P different problems per step; (b) the
full argv of every python3 call of rl/rl_inner.sh args (run20's, the memprobe's and the three train67 modes' train.py
argv), recorded by a python3 stub: seed 67 byte-identical to 60e6d34's rl_inner.sh, seed 68 the same plus
--seed 101 --rollout-seed 142 on each train67 argv only; (c) rl/rl_job.sh's argument parsing (RL_JOB_PRINT_ONLY): the run
dirs (seed 67 unchanged, seed 68 with _s68 after the set name), seed= in any place after the mode, bad seeds refused.
Runs in miles-v0.1.3.sif: tests/run_all.sh (miles' own parse of these argvs: rl/args_all.sh seed=68). Usage alone:
python3 tests/test_train67_seed.py (any python 3.8+ with git)."""
import json
import os
import random
import subprocess
import sys
import tempfile

from helpers import ROOT, check, done

sys.path.insert(0, os.path.join(ROOT, "rl"))
import train67_data as T  # noqa: E402

BASE = "60e6d34"
SRC = os.path.join(ROOT, "data", "fcs11_train67_agent.jsonl")
tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR") or None)
old = os.path.join(tmp, "old")
r = subprocess.run(f"mkdir -p {old} && git -C {ROOT} archive {BASE} rl flint/checkers data/train67_sets | tar -x -C {old}",
                   shell=True, capture_output=True, text=True)
check(f"{BASE}'s rl/ exported", r.returncode == 0, r.stderr)


def data(tree, out, *args):
    return subprocess.run([sys.executable, os.path.join(tree, "rl", "train67_data.py"), SRC, out, *args],
                          capture_output=True, text=True)


# (a) the data order
for extra in ([], ["--set", "signal34"]):
    for P in ("4", "3"):
        a, b, c = (os.path.join(tmp, f"{k}{P}{len(extra)}.jsonl") for k in "abc")
        o1, o2, o3 = data(old, a, P, *extra), data(ROOT, b, P, *extra), data(ROOT, c, P, "67", *extra)
        same = all(o.returncode == 0 for o in (o1, o2, o3)) and open(a, "rb").read() == open(b, "rb").read() \
            == open(c, "rb").read() and o1.stdout == o2.stdout == o3.stdout
        check(f"seed 67 (default and explicit), P={P} {' '.join(extra)}: file and line byte-identical to {BASE}'s", same,
              (o1.stderr, o2.stderr, o3.stderr))
rows = T.select([json.loads(x) for x in open(SRC)], T.load_set("signal34"))
n = len(rows)
for P in (4, 3, 8):
    s67, s68 = T.order(n, P, 67, 30), T.order(n, P, 68, 30)
    check(f"seed 68 P={P}: another order than seed 67", s67 != s68 and s67[:P] != s68[:P])
    check(f"seed 68 P={P}: every epoch a permutation of the {n}",
          all(sorted(s68[e * n:(e + 1) * n]) == list(range(n)) for e in range(30)))
    check(f"seed 68 P={P}: every step {P} different problems",
          all(len(set(s68[i:i + P])) == P for i in range(0, len(s68) - P + 1, P)))
perm = list(range(n))
random.Random(68).shuffle(perm)
check("seed 68: epoch 0 is random.Random(68)'s permutation", T.order(n, 4, 68, 30)[:32] == perm[:32])
o = data(ROOT, os.path.join(tmp, "d68.jsonl"), "4", "68", "--set", "signal34")
check("cli seed 68: runs and reports seed 68", o.returncode == 0 and "seed 68," in o.stdout, o.stdout + o.stderr)
got = [json.loads(x)["metadata"]["pid"] for x in open(os.path.join(tmp, "d68.jsonl"))]
check("cli seed 68: the file holds seed 68's order", got == [rows[i]["metadata"]["pid"] for i in T.order(n, 4, 68, 30)])

# (b) every python3 argv of rl_inner.sh args, through a stub that records it
stub = os.path.join(tmp, "bin")
os.makedirs(stub)
open(os.path.join(stub, "python3"), "w").write('#!/bin/bash\n{ printf "%s\\0" "$@"; printf "\\n"; } >> "$ARGV_LOG"\n')
os.chmod(os.path.join(stub, "python3"), 0o755)


def argvs(script, seed):
    """[argv...] of rl_inner.sh args, REPO pointed at one fixed path (the script's own path, not an argument)."""
    src = open(script).read()
    lines = [x if not x.startswith("REPO=") else "REPO=/REPO" for x in src.splitlines()]
    sh = os.path.join(tmp, f"inner_{os.path.basename(os.path.dirname(os.path.dirname(script)))}_{seed}.sh")
    open(sh, "w").write("\n".join(lines) + "\n")
    log = sh + ".argv"
    work = tempfile.mkdtemp(dir=tmp)
    e = {**os.environ, "PATH": stub + ":" + os.environ["PATH"], "ARGV_LOG": log, "JID": "args0", "OUT": "/OUT",
         "PKG": "/PKG", "TMPDIR": work, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "T67_P": "4", "T67_G": "8"}
    if seed is not None:
        e["T67_SEED"] = str(seed)
    p = subprocess.run(["bash", sh, "args"], env=e, capture_output=True, text=True)
    out = [x.split("\0")[:-1] for x in open(log).read().split("\n") if x] if os.path.exists(log) else []
    return p.returncode, out, p.stdout + p.stderr


rc0, a0, l0 = argvs(os.path.join(old, "rl", "rl_inner.sh"), None)
rc1, a1, l1 = argvs(os.path.join(ROOT, "rl", "rl_inner.sh"), None)
rc2, a2, l2 = argvs(os.path.join(ROOT, "rl", "rl_inner.sh"), 67)
rc3, a3, l3 = argvs(os.path.join(ROOT, "rl", "rl_inner.sh"), 68)
# elastic-v2's args mode adds the elastic and async checks (new python3 calls: an engine provider, --fully-async, an
# engine job's argv from its spec); every call 60e6d34 makes must stay byte-identical, so compare without the new ones
NEW = ("--extra-inference-engine-provider-path", "--fully-async")


def base_only(av):
    return [a for a in av if not any(x in a for x in NEW) and not any(w.endswith("elastic_engine_argv.py") for w in a)]


n_new = len(a1) - len(base_only(a1))
check("argv: the elastic and async checks add python3 calls (an engine provider or --fully-async)", n_new > 0, n_new)
a1, a2, a3 = base_only(a1), base_only(a2), base_only(a3)
check(f"argv: {BASE}'s rl_inner.sh args gives 9 python3 calls (game settings, 4 run20/memprobe, 3 train67, schedule)",
      rc0 == 0 and len(a0) == 9, (rc0, len(a0), l0[-500:]))
check(f"argv seed unset: every python3 argv byte-identical to {BASE}'s", rc1 == 0 and a1 == a0, l1[-500:])
check(f"argv seed=67: every python3 argv byte-identical to {BASE}'s", rc2 == 0 and a2 == a0, l2[-500:])
t67 = [i for i, a in enumerate(a0) if "--rollout-batch-size" in a and "--global-batch-size" in a
       and a[len(a) - 1 - a[::-1].index("--global-batch-size") + 1] == "32"]
check("argv: the train67 argvs found (3 modes' args_check, the B=32 schedule check)", len(t67) == 4, t67)
extra = ["--seed", "101", "--rollout-seed", "142"]
exp = [a + extra if i in t67 else a for i, a in enumerate(a0)]
# the train67 argvs: TRAIN67 sits before SERVE; the seeds are appended to TRAIN67, so compare as multisets of the
# flag list's order with the extra flags removed, and check where they are
ok = rc3 == 0 and len(a3) == len(a0)
for i, (x, y) in enumerate(zip(a0, a3)):
    if i in t67:
        k = len(y) - 1 - y[::-1].index("--global-batch-size") + 2 if "--global-batch-size" in y else -1
        ok &= y[k:k + 4] == extra and y[:k] + y[k + 4:] == x
    else:
        ok &= x == y
check("argv seed=68: each train67 argv gains --seed 101 --rollout-seed 142 after TRAIN67 (after TRAIN's --seed 1: the "
      "last wins), every other argv unchanged", ok, l3[-500:])


def last(argv, flag):
    return argv[len(argv) - 1 - argv[::-1].index(flag) + 1] if flag in argv else None


check("argv: the last --seed / --rollout-seed of each train67 argv: seed 67 1 / (miles' default 42), seed 68 101 / 142",
      all(last(a0[i], "--seed") == "1" and last(a0[i], "--rollout-seed") is None
          and last(a3[i], "--seed") == "101" and last(a3[i], "--rollout-seed") == "142" for i in t67))

# (c) rl_job.sh's parsing
R = "/scratch/gpfs/GROUP/USER/tmp/flint_s5/runs"


def job(*args):
    p = subprocess.run(["bash", os.path.join(ROOT, "rl", "rl_job.sh"), *args], capture_output=True, text=True,
                       env={**os.environ, "RL_JOB_PRINT_ONLY": "1", "FETCHED_JOB_ID": "t1"})
    return p.returncode, p.stdout.strip()


cases = {("train67", "4", "8", "80", "set=signal34"): f"SEED=67 OUT={R}/train67_signal34_p4g8",
         ("train67", "4", "8", "80", "set=signal34", "seed=67"): f"SEED=67 OUT={R}/train67_signal34_p4g8",
         ("train67", "4", "8", "80", "set=signal34", "seed=68"): f"SEED=68 OUT={R}/train67_signal34_s68_p4g8",
         ("train67", "seed=68", "4", "set=signal34", "8", "80"): f"SEED=68 OUT={R}/train67_signal34_s68_p4g8",
         ("train67", "seed=68"): f"SEED=68 OUT={R}/train67_s68_p4g8",
         ("train67",): f"SEED=67 OUT={R}/train67_p4g8",
         ("train67_test", "seed=68", "set=signal34"): f"SEED=68 OUT={R}/train67_test_signal34_s68_p4g8",
         ("train67_test", "set=signal34"): f"SEED=67 OUT={R}/train67_test_signal34_p4g8"}
for a, want in cases.items():
    rc, out = job(*a)
    check(f"rl_job {' '.join(a)}: {want.split('/')[-1]}", rc == 0 and out.endswith(want), out)
rc, out = job("train67", "4", "8", "80", "set=signal34", "seed=68")
check("rl_job seed=68: P G STEPS still parsed", "P=4 G=8 STEPS=80 SET=signal34" in out, out)
for a in (("train67", "seed=abc"), ("train67", "seed=5"), ("train67", "seed=068"), ("train67", "seed="),
          ("train67", "seed=10000"), ("run20", "seed=68"), ("smoke", "seed=67")):
    rc, out = job(*a)
    check(f"rl_job {' '.join(a)}: refused", rc != 0 and "OUT=" not in out, out)
# levers=LIST (engine levers, rl/rl_inner.sh): sorted and deduplicated, their own run dir, no levers = as before
lev = {("train67", "2", "8", "160", "set=signal34", "levers=tp4,gate"):
           f"SEED=67 OUT={R}/train67_signal34_kgate-tp4_p2g8 LEVERS=gate,tp4",
       ("train67_test", "2", "8", "set=signal34", "levers=graphs,gate,tp4,gate"):
           f"SEED=67 OUT={R}/train67_test_signal34_kgate-graphs-tp4_p2g8 LEVERS=gate,graphs,tp4",
       ("train67", "seed=68", "levers=tp4"): f"SEED=68 OUT={R}/train67_s68_ktp4_p4g8 LEVERS=tp4",
       ("train67", "levers=pgraph,arfusion", "2", "8"): f"SEED=67 OUT={R}/train67_karfusion-pgraph_p2g8 LEVERS=arfusion,pgraph",
       ("train67_test", "2", "8", "set=signal34", "levers=moecfg,gate"):
           f"SEED=67 OUT={R}/train67_test_signal34_kgate-moecfg_p2g8 LEVERS=gate,moecfg"}
for a, want in lev.items():
    rc, out = job(*a)
    check(f"rl_job {' '.join(a)}: {want.split('/')[-1]}", rc == 0 and out.endswith(want), out)
rc, out = job("train67", "2", "8", "160", "set=signal34")
check("rl_job without levers: no LEVERS field, the w765 run dir", rc == 0 and "LEVERS" not in out
      and out.endswith(f"OUT={R}/train67_signal34_p2g8"), out)
for a in (("train67", "levers="), ("train67", "levers=foo"), ("train67", "levers=tp4,,gate"), ("train67", "levers=TP4"),
          ("train67", "levers=tp4 gate"), ("run20", "levers=tp4"), ("smoke", "levers=gate")):
    rc, out = job(*a)
    check(f"rl_job {' '.join(a)}: refused", rc != 0 and "OUT=" not in out, out)
# cmdtimeout=N (the game's per-command limit): 60 is the default run; another value gets its own run dir _t<N>
cmdt = {("train67", "2", "8", "10", "set=signal34", "cmdtimeout=30"):
            f"SEED=67 OUT={R}/train67_signal34_t30_p2g8 CMDTIMEOUT=30",
        ("train67", "2", "8", "10", "set=signal34", "cmdtimeout=60"): f"SEED=67 OUT={R}/train67_signal34_p2g8",
        ("train67", "cmdtimeout=30", "levers=gate", "seed=68"): f"SEED=68 OUT={R}/train67_s68_kgate_t30_p4g8 LEVERS=gate CMDTIMEOUT=30"}
for a, want in cmdt.items():
    rc, out = job(*a)
    check(f"rl_job {' '.join(a)}: {want.split('/')[-1]}", rc == 0 and out.endswith(want), out)
for a in (("train67", "cmdtimeout=3"), ("train67", "cmdtimeout=abc"), ("train67", "cmdtimeout=0060"), ("train67", "cmdtimeout="),
          ("train67", "cmdtimeout=601"), ("run20", "cmdtimeout=30")):
    rc, out = job(*a)
    check(f"rl_job {' '.join(a)}: refused", rc != 0 and "OUT=" not in out, out)
# place=2numa (async only; the run dir stays the same, so a run can change placement at a resume), save=0 (train67_test only)
async_tag = f"{R}/train67_signal34_kgate-graphs-tp4_e_a1_p2g8"
place = {("train67", "2", "8", "40", "set=signal34", "async=1", "stale=1", "elastic=1", "levers=gate,graphs,tp4", "place=2numa"):
             f"SEED=67 OUT={async_tag} LEVERS=gate,graphs,tp4 PLACE=2numa ELASTIC=1 ASYNC=1 STALE=1",
         ("train67", "2", "8", "40", "set=signal34", "async=1", "stale=1", "elastic=1", "levers=gate,graphs,tp4"):
             f"SEED=67 OUT={async_tag} LEVERS=gate,graphs,tp4 ELASTIC=1 ASYNC=1 STALE=1",
         ("train67_test", "1", "2", "set=signal34", "async=1", "stale=1", "elastic=1", "levers=gate,graphs,tp4", "place=2numa", "save=0"):
             f"SEED=67 OUT={R}/train67_test_signal34_kgate-graphs-tp4_e_a1_p1g2 LEVERS=gate,graphs,tp4 PLACE=2numa NOSAVE=1 ELASTIC=1 ASYNC=1 STALE=1"}
for a, want in place.items():
    rc, out = job(*a)
    check(f"rl_job {' '.join(a)}: {want.split('/')[-1]}", rc == 0 and out.endswith(want), out)
rc, out = job("train67", "2", "8", "40", "set=signal34", "async=1", "stale=1", "elastic=1", "levers=gate,graphs,tp4", "place=2numa", "inflight=32")
check("rl_job inflight=32: the same run dir", rc == 0 and out.endswith(f"SEED=67 OUT={async_tag} LEVERS=gate,graphs,tp4 PLACE=2numa INFLIGHT=32 ELASTIC=1 ASYNC=1 STALE=1"), out)
for a in (("train67", "inflight=32"), ("train67", "async=1", "inflight=x"), ("train67", "async=1", "inflight=0"), ("train67", "async=1", "inflight="), ("run20", "inflight=32"),
          ("train67", "place=2numa"), ("train67", "async=1", "place=x"), ("train67", "async=1", "place="), ("run20", "place=2numa"),
          ("train67", "save=0"), ("train67", "async=1", "save=0"), ("run20", "save=0")):
    rc, out = job(*a)
    check(f"rl_job {' '.join(a)}: refused", rc != 0 and "OUT=" not in out, out)
# malloc=keep: the same run dir (a run can take it at a resume); the training environment gains exactly the three glibc variables
rc, out = job("train67", "2", "8", "40", "set=signal34", "levers=gate,graphs,tp4", "malloc=keep")
check("rl_job malloc=keep: the same run dir", rc == 0 and out.endswith(f"OUT={R}/train67_signal34_kgate-graphs-tp4_p2g8 LEVERS=gate,graphs,tp4 MALLOC=keep"), out)
for a in (("train67", "malloc=x"), ("train67", "malloc="), ("run20", "malloc=keep")):
    rc, out = job(*a)
    check(f"rl_job {' '.join(a)}: refused", rc != 0 and "OUT=" not in out, out)


def train_env(**env):
    sh = "".join(l for l in open(os.path.join(ROOT, "rl", "rl_inner.sh")) if l.startswith("TRAIN_ENV") or "T67_MALLOC" in l and l.startswith("["))
    r = subprocess.run(["bash", "-c", sh + 'printf %s "$TRAIN_ENV"'], capture_output=True, text=True, env={"PATH": os.environ["PATH"], **env})
    return json.loads(r.stdout)


plain = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "11", "MKL_NUM_THREADS": "11"}
check("rl_inner: no malloc=keep, the environment as before", train_env() == plain)
check("rl_inner: malloc=keep adds the glibc variables and the run-time switch", train_env(T67_MALLOC="keep") == {**plain, "MALLOC_ARENA_MAX": "1", "MALLOC_TRIM_THRESHOLD_": "107374182400", "MILES_KEEP_FREED_AFTER_FIRST_STEP": "1"})
# order=oldest: async only, with a staleness bound; the same run dir (a run can take it at a resume). stale=2 has its own run dir.
rc, out = job("train67", "2", "8", "40", "set=signal34", "async=1", "stale=1", "elastic=1", "levers=gate,graphs,tp4", "order=oldest")
check("rl_job order=oldest: the s=1 run dir", rc == 0 and f"OUT={R}/train67_signal34_kgate-graphs-tp4_e_a1_p2g8" in out and "ORDER=oldest" in out, out)
rc, out = job("train67", "2", "8", "40", "set=signal34", "async=1", "stale=2", "elastic=1", "levers=gate,graphs,tp4", "place=2numa", "inflight=48", "malloc=keep", "order=oldest")
check("rl_job stale=2 inflight=48: its own run dir", rc == 0 and f"OUT={R}/train67_signal34_kgate-graphs-tp4_e_a2_p2g8" in out and "MALLOC=keep" in out and "ORDER=oldest" in out, out)
for a in (("train67", "order=oldest"), ("train67", "async=1", "order=oldest"), ("train67", "async=1", "stale=1", "order=x"), ("run20", "order=oldest")):
    rc, out = job(*a)
    check(f"rl_job {' '.join(a)}: refused", rc != 0 and "OUT=" not in out, out)
# one repo everywhere: rl_job.sh is a snapshot in wq and the others are read live; another worktree's name means another worktree's code at run time (elastic-v7's audit)
import re  # noqa: E402
_src = {n: open(os.path.join(ROOT, "rl", n)).read() for n in ("rl_job.sh", "rl_inner.sh", "elastic_engine.sh") if os.path.exists(os.path.join(ROOT, "rl", n))}
_names = {n: re.findall(r"^(?:REPO|HERE)=(\S+)", t, re.M) for n, t in _src.items()}
check("scripts: every script names this worktree", all(v and v[0] == os.path.realpath(ROOT) for v in _names.values()), _names)
_miles = {n: re.findall(r"^MILES=(\S+)", _src[n], re.M) for n in ("rl_job.sh", "rl_inner.sh")}
check("scripts: rl_job.sh and rl_inner.sh name one miles", _miles["rl_job.sh"] == _miles["rl_inner.sh"] and _miles["rl_job.sh"], _miles)
done()
