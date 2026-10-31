"""Submits of unchanged code (MA_LG_DEDUP_SUBMIT=1, the train67 RL modes; off by default). (1) Off changes nothing: the
scripted episodes of tests/dedup_ab_run.py (real ChatSession over a fake HTTP server, stubbed judge) on rl-train67's
code (864a904, exported from git) and on this tree give the same request bodies byte for byte and the same traces,
info, chats and judge calls, for a team of 4 and a lone agent, with and without trims, with MA_LG_DEDUP_SUBMIT unset,
empty or 0. On, the same episodes call the judge once less for each submit of the agent's own judged code (a
teammate's code is judged), and S and the best provisional stay. (2) A lone agent that submits the same code 3 times,
then changes it and submits: the judge runs twice, the replies name the earlier submission and its provisional score
and ask for a change, S and the best provisional equal those of the run that judges every submit, the trace and info
count the duplicates; an infra-failed submission is judged again; a resubmission under another file name is a
duplicate too; the finish command keeps its own rule. (3) A checkpoint in the middle keeps the dedup state (the
submissions and the counter): the resumed run ends as the uninterrupted one. (4) The setting is checked at import."""
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile

from helpers import CLOCK, FIN, LABEL, MSG, ROOT, block, check, done, env, inp, scripted, sessions, traces

env(MA_LG_N=1, MA_LG_TOKENS=10000, MA_LG_DEDUP_SUBMIT=1)
from flint import game as G  # noqa: E402

BASE = "864a904"


def clean_env(**kw):
    e = {k: v for k, v in os.environ.items() if not k.startswith("MA_")}
    e.update({k: str(v) for k, v in kw.items()})
    return e


# (4) the setting at import
for v, ok in (("", True), ("0", True), ("1", True), ("2", False), ("true", False), ("yes", False), (" 1", False)):
    p = subprocess.run([sys.executable, "-c", "import flint.game as G; print(G.DEDUP_SUBMIT)"], cwd=ROOT,
                       env=clean_env(MA_LG_DEDUP_SUBMIT=v), capture_output=True, text=True)
    check(f"import with MA_LG_DEDUP_SUBMIT={v!r}: {'accepted' if ok else 'an error'}",
          (p.returncode == 0 and p.stdout.strip() == str(v == "1")) if ok else
          (p.returncode != 0 and "MA_LG_DEDUP_SUBMIT" in p.stderr), (p.stdout, p.stderr[-300:]))

# (1) off: byte-identical to 864a904; on: fewer judge calls, the same S
tree = tempfile.mkdtemp(dir=os.environ["TMPDIR"])
r = subprocess.run(f"git -C {ROOT} archive {BASE} flint | tar -x -C {tree}", shell=True, capture_output=True, text=True)
check(f"rl-train67's code ({BASE}) exported", r.returncode == 0 and os.path.isfile(os.path.join(tree, "flint", "game.py")),
      r.stderr)


def run_ab(t, **kw):
    out = tempfile.mktemp(dir=os.environ["TMPDIR"], suffix=".json")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "dedup_ab_run.py"), t, out], env=clean_env(**kw),
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-3000:])
        return None
    print("  ", "base" if t == tree else "tree", kw, p.stdout.strip()[:300])
    return json.load(open(out))


def diff(a, b):
    if a is None or b is None:
        return "a run failed"
    keys = [k for k in a if a[k] != b.get(k)] + [k for k in b if k not in a]
    if keys == ["trace"]:
        keys = ["trace." + k for k in a["trace"] if a["trace"][k] != b["trace"].get(k)]
        k = keys[0][6:]
        return f"{keys}: {str(a['trace'][k])[:1500]} vs {str(b['trace'][k])[:1500]}"
    for k in keys[:1]:
        x, y = a.get(k), b.get(k)
        if isinstance(x, list) and isinstance(y, list):
            i = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
            return f"{keys}: first difference at {k}[{i}] of {len(x)}/{len(y)}: {str(x[i:i + 1])[:600]} vs {str(y[i:i + 1])[:600]}"
    return keys


for name, kw, dups in [("team of 4", dict(MA_LG_N=4, MA_LG_TOKENS=10000), 2),
                       ("team of 4 with trims", dict(MA_LG_N=4, MA_LG_TOKENS=10000, MA_LG_CTX_CHARS=1200), 2),
                       ("lone agent", dict(MA_LG_N=1, MA_LG_TOKENS=10000), 1),
                       ("lone agent with trims", dict(MA_LG_N=1, MA_LG_TOKENS=10000, MA_LG_CTX_CHARS=1200), 1)]:
    ref = run_ab(tree, **kw)
    check(f"{name}: {BASE} ran (requests, submissions)", ref is not None and len(ref["requests"]) >= 10
          and ref["trace"]["submissions"], ref and len(ref["requests"]))
    for v in (None, "", "0"):
        got = run_ab(ROOT, **kw, **({} if v is None else {"MA_LG_DEDUP_SUBMIT": v}))
        d = diff(ref, got)
        check(f"{name}, MA_LG_DEDUP_SUBMIT {'unset' if v is None else repr(v)}: requests, trace, info, chats and judge "
              f"calls as on {BASE}", not d, d)
    on = run_ab(ROOT, **kw, MA_LG_DEDUP_SUBMIT=1)
    if ref is None or on is None:
        check(f"{name}, dedup on: ran", False)
        continue
    t0, t1 = ref["trace"], on["trace"]
    check(f"{name}, dedup on: {dups} judge call(s) fewer, as many fewer submissions", on["judged"] == ref["judged"] - dups
          and len(t1["submissions"]) == len(t0["submissions"]) - dups, (ref["judged"], on["judged"]))
    check(f"{name}, dedup on: the same S, best provisional and best submission", on["S"] == ref["S"]
          and on["info"]["fo_lg_best_prov"] == ref["info"]["fo_lg_best_prov"]
          and (t1["best"] or {}).get("sha") == (t0["best"] or {}).get("sha"), (on["S"], ref["S"]))
    check(f"{name}, dedup on: counted in the trace and info, the harness tag ends in -dedup",
          sum(a.get("dup_submits", 0) for a in t1["agents"]) == dups
          and on["info"]["fo_lg_dup_submits"] == dups / kw["MA_LG_N"] and "fo_lg_dup_submits" not in ref["info"]
          and t1["harness"] == t0["harness"] + "-dedup", ([a.get("dup_submits") for a in t1["agents"]], on["info"].get("fo_lg_dup_submits")))
    if kw["MA_LG_N"] == 4:
        a1 = next(a for a in t1["agents"] if a["name"] == "agent1")
        check(f"{name}, dedup on: no post owed after a duplicate (nothing was submitted)", not any(
            "must be `post" in t["output"] for t in a1["turns"] if t.get("dup_submit")), a1["turns"][3]["output"][:300])

# (2) a lone agent, scripted, in this process (G.DEDUP_SUBMIT switched for the run that judges every submit)
judged = []
RESULT = None


async def fake_judge(label, code):
    judged.append(code)
    if RESULT is not None:
        return RESULT(code)
    m = re.search(r"P=(\d+)", code)
    return {"cases": [int(m.group(1)) / 100 if m else 0.1] * 10, "status": "ok", "infra_error": "", "msg": ""}


Scripted = scripted(G)
G.judge_code, G.INFRA_BACKOFF = fake_judge, 0.01
real_agent = G.Agent


def run(script, dedup=True):
    G.Agent, G.DEDUP_SUBMIT = Scripted, dedup
    Scripted.script = list(script)
    judged.clear()
    agents, S, info = asyncio.run(G.episode(inp(), MSG, LABEL, sessions(None)))
    G.Agent, G.DEDUP_SUBMIT = real_agent, True
    return agents[0], S, info, traces()[-1]


A = "printf 'int main(){return 0;}// P=40\\n' > solution.cpp"
B = "printf 'int main(){return 1;}// P=55\\n' > solution.cpp"
SC = [A, "submit", "submit", "submit", B, "submit"]
a, S, info, tr = run(SC)
check("3 x A then B: the judge runs 2 times", len(judged) == 2 and len(tr["submissions"]) == 2, len(judged))
want = ("submit: solution.cpp is unchanged since your submission #1 (provisional 40.00), so it was not judged again and "
        "no new submission was recorded. Change solution.cpp before you submit it again.\n")
check("the duplicates' replies: unchanged since #1 (its provisional), not judged, change it first",
      a.outs[2] == want and a.outs[3] == want, a.outs[2:4])
check("the change is judged as submission #2", a.outs[5].startswith("Submission #2 by agent1: ok. Provisional score 55.00")
      and "Best provisional so far: 55.00 (#2, agent1)" in a.outs[5], a.outs[5])
check("the duplicates counted: trace dup_submits 2, info fo_lg_dup_submits 2, turns flagged",
      tr["agents"][0]["dup_submits"] == 2 and info["fo_lg_dup_submits"] == 2.0 and a.dup_submits == 2
      and [bool(t.get("dup_submit")) for t in tr["agents"][0]["turns"]] == [False, False, True, True, False, False]
      and a.n_submit == 2 and info["fo_lg_submissions"] == 2, (tr["agents"][0]["dup_submits"], info.get("fo_lg_dup_submits")))
a0, S0, info0, tr0 = run(SC, dedup=False)
check("the run that judges every submit: 4 judge calls, 4 submissions, no dup counts", len(judged) == 4
      and len(tr0["submissions"]) == 4 and "dup_submits" not in tr0["agents"][0] and "fo_lg_dup_submits" not in info0)
check("S and the best provisional equal those of the run that judges every submit",
      S == S0 == 0.55 and info["fo_lg_best_prov"] == info0["fo_lg_best_prov"] == 0.55
      and info["fo_lg_S_auto"] == info0["fo_lg_S_auto"] and info["fo_lg_oracle_final"] == info0["fo_lg_oracle_final"],
      (S, S0, info["fo_lg_best_prov"], info0["fo_lg_best_prov"]))
SC2 = [A, "submit", B, "submit", A, "submit", "submit"]
a, S, info, tr = run(SC2)
a0, S0, info0, tr0 = run(SC2, dedup=False)
check("A, B, back to A: A again is a duplicate of #1 (an earlier submission, not only the last)",
      a.outs[5].startswith("submit: solution.cpp is unchanged since your submission #1 (provisional 40.00)")
      and a.dup_submits == 2 and S == S0 and info["fo_lg_best_prov"] == info0["fo_lg_best_prov"], a.outs[5])
a, S, info, tr = run([A, "submit", "cp solution.cpp s2.cpp", "submit s2.cpp", "echo x >> s2.cpp", "submit s2.cpp"])
check("another file with the same text is a duplicate; changed, it is judged", a.outs[3].startswith(
      "submit: s2.cpp is unchanged since your submission #1 (provisional 40.00)") and "Change s2.cpp before" in a.outs[3]
      and len(judged) == 2 and a.outs[5].startswith("Submission #2"), a.outs[3:])
a, S, info, tr = run([A, "submit", "submit", FIN])
n_dedup = len(judged)
a0, S0, info0, tr0 = run([A, "submit", "submit", FIN], dedup=False)
check("the finish command after a duplicate: its own unchanged rule, as in the run that judges every submit",
      a.restarts == a0.restarts == 1 and a.outs[3].startswith("solution.cpp is unchanged since your submission #1")
      and a0.outs[3].startswith("solution.cpp is unchanged since your submission #2") and n_dedup == 1
      and len(judged) == 2, (a.outs[3], a0.outs[3], n_dedup, len(judged)))
a, S, info, tr = run([A, "submit sol.cpp", "submit"])
check("a missing file: the submit's own error, not a duplicate", a.outs[1].startswith("submit: no file sol.cpp")
      and a.dup_submits == 0 and len(judged) == 1, a.outs[1])
RESULT = lambda code: ({"cases": [0.0] * 10, "status": "done", "infra_error": True, "msg": ""} if "P=40" in code  # noqa: E731
                       else {"cases": [0.5] * 10, "status": "ok", "infra_error": "", "msg": ""})
a, S, info, tr = run([A, "submit", "submit"])
check("an infra-failed submission is judged again (3 tries each), not a duplicate", len(judged) == 6 and a.dup_submits == 0
      and len(tr["submissions"]) == 2, (len(judged), a.dup_submits))
RESULT = None

# (3) a checkpoint in the middle: the real Agent with a fake session (replies keyed on the agent's turn count)
SCRIPT = [A, "submit", "submit", "true", "submit", B, "submit", "submit", "true"]
hook = {"stop_at": None, "ckpt": None}
HEX = re.compile(r"\b[0-9a-f]{7,40}\b")


async def reply(sess, messages):
    k = sum(m["role"] == "assistant" for m in messages)
    if hook["stop_at"] == k:
        hook["ckpt"].stop = True
    return block(SCRIPT[k] if k < len(SCRIPT) else "true", k), False


def h(x):
    if isinstance(x, dict):
        return {k: h(v) for k, v in x.items() if k not in CLOCK and k not in ("rev", "time")}
    if isinstance(x, (list, tuple)):
        return [h(v) for v in x]
    return HEX.sub("H", x) if isinstance(x, str) else x


def episode(ckpt=None):
    agents, S, info = asyncio.run(G.episode(inp(1, 2), MSG, LABEL, sessions(reply, tokens=1000), ckpt=ckpt))
    tr = traces()[-1]
    tr["git_log"] = sorted(tr["git_log"].splitlines())
    return {"S": S, "info": {k: v for k, v in info.items() if not k.endswith("_min")}, "trace": h(tr),
            "chats": h([a.chat for a in agents])}


judged.clear()
ref = episode()
n_ref = len(judged)
check("checkpoint reference: 2 judge calls, 3 duplicates", n_ref == 2 and ref["info"]["fo_lg_dup_submits"] == 3.0,
      (n_ref, ref["info"].get("fo_lg_dup_submits")))
path = os.path.join(tempfile.mkdtemp(dir=os.environ["TMPDIR"]), "state", "1_e2")
ck = G.Checkpoint(path, every=1e9)
hook.update(stop_at=3, ckpt=ck)
judged.clear()
try:
    episode(ck)
    paused = False
except G.Paused:
    paused = True
hook["stop_at"] = None
st = G.Checkpoint(path).load() or {"agents": [{"agent": {}}], "config": {}}
n1 = len(judged)
check("paused after 1 judged submit and 1 duplicate; the state holds the counter and the dedup setting",
      paused and n1 == 1 and st["agents"][0]["agent"].get("dup_submits") == 1 and st["config"].get("DEDUP_SUBMIT") is True
      and len(st["subs"]) == 1, (paused, n1, st["agents"][0]["agent"].get("dup_submits"), st["config"]))
judged.clear()
res = episode(G.Checkpoint(path, every=1e9))
check("resumed: the submit of A after the restore is a duplicate (1 more judge call: B), the run ends as the "
      "uninterrupted one", len(judged) == 1 and n1 + len(judged) == n_ref and res == ref,
      (len(judged), diff(ref, res)))
# a checkpoint of the game without dedup does not resume into the game with it (its config has no DEDUP_SUBMIT)
G.DEDUP_SUBMIT = False
ck = G.Checkpoint(path + "_off", every=1e9)
hook.update(stop_at=3, ckpt=ck)
try:
    episode(ck)
except G.Paused:
    pass
hook["stop_at"] = None
G.DEDUP_SUBMIT = True
judged.clear()
res = episode(G.Checkpoint(path + "_off", every=1e9))
check("a checkpoint saved without dedup starts fresh with dedup on", res == ref and len(judged) == n_ref, len(judged))
done()
