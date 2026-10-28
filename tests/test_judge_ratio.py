"""The case ratio follows the official Frontier-CS v1.1 engine except for one deviation (an unreadable `Ratio:` value
scores 0, where the engine gives 1 to a helper that exited 0): unit cases with our ratio and the engine's; the engine's
own isScoringVerdict / parseRatio / applyScoreRatio (cut from judge_engine.js and run in node) must give the engine's
ratio on the unit cases, so ours differs from it on exactly the intended cases, and must equal ours on every distinct
checker or interactor message stored in flint-eval/runs traces; real judges: a program that prints `Ratio:1` on
problem 247 scores 0 (its checker echoes the token after its own Ratio), and a stored program on problem 0 scores what
it scored then."""
import glob
import json
import os
import re
import subprocess

from helpers import check, done, env

env()
from flint import judge as J  # noqa: E402

P = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/"
NODE = os.environ.get("FCS_NODE", "/scratch/gpfs/GROUP/USER/nvm/versions/node/v22.17.1/bin/node")
ENGINE = J.FCS_ROOT / "judge" / "src" / "judge_engine.js"
RUNS = "/scratch/gpfs/GROUP/USER/project/flint-eval/runs"
W1532 = RUNS + "/smoke_v2_w1532/traces/agent_3114114.jsonl"   # problem 0, first submission: deterministic program
MSG_247 = ("points 0.0 Value: 0. Ratio: 0.000000000000, RatioUnbounded: 0.000000000000. Output must start with 'Yes' "
           "or 'No', found 'Ratio:1'")

# (helper exit code, message, our ratio, the engine's ratio): they differ only where the engine's fallback gives an
# accepted helper 1 for a Ratio label it cannot read (the auditor's weakness messages, tmp/audit_main/official_repro.mjs)
UNIT = [(7, "points 0.5 Ratio: 0.500000000000, RatioUnbounded: 0.500000000000", 0.5, 0.5),
        (7, MSG_247, 0.0, 0.0),                                  # the first Ratio is the checker's own
        (7, "points 1.5 Ratio: 1.500000000000", 1.5, 1.5),       # not clamped
        (7, "points 5e-05 Ratio: 5e-05", 5.0, 5.0),              # digits and dots only
        (7, "points 0.0 Ratio: 0.000000.", 0.0, 0.0),            # parseFloat's longest decimal prefix
        (7, "points -1e-15 Ratio: -0.000000000000", 0.0, 0.0),   # a label without a value: 0 either way
        (0, "ok Ratio: -0.000000000000", 0.0, 1.0),              # the deviation: the engine's fallback gives 1
        (0, "ok Ratio: -0.250000000000", 0.0, 1.0),
        (0, "ok Ratio: nan, RatioUnbounded: nan", 0.0, 1.0),
        (0, "ok Ratio: " + "9" * 400, 0.0, 1.0),                 # overflows to inf
        (7, "points 1 Ratio: " + "9" * 400, 0.0, 0.0),
        (0, "ok RatioUnbounded: 0.500000000000", 1.0, 1.0),      # no `Ratio:` label: accepted
        (7, "points 0.3", 0.0, 0.0), (0, "ok", 1.0, 1.0),        # no label
        (1, "wrong answer Ratio: 1.000000000000", 0.0, 0.0),     # an untrusted message is never read
        (2, "wrong output format Ratio: 1", 0.0, 0.0)]
bad = [(e, m[:60], J._ratio(m, e in (0, 7), e == 0), want) for e, m, want, _ in UNIT
       if J._ratio(m, e in (0, 7), e == 0) != want]
check("unit cases: our ratio", not bad, bad)


def stored_messages() -> set:
    """(helper exit, message) of every scored case in the stored traces: a classic case_info is 'chk=<exit> cpu=<t>/<tl>
    <message>', an interactive one '... inter code=<exit> killed=<bool> <message>' (killed: not a scoring verdict)."""
    out = set()
    chk = re.compile(r"chk=(-?\d+) cpu=\S+ ?(.*)", re.S)
    inter = re.compile(r".*?; inter code=(-?\d+) killed=False ?(.*)", re.S)
    for f in glob.glob(RUNS + "/*/traces/*.jsonl"):
        for line in open(f):
            for s in json.loads(line).get("submissions") or []:
                for info in s.get("case_info") or []:
                    m = chk.fullmatch(info) or inter.fullmatch(info)
                    if m:
                        out.add((int(m.group(1)), m.group(2)))
    return out


def engine(pairs: list) -> list:
    """The engine's scoreRatio for each (exit, message), from its own functions run in node."""
    src = ENGINE.read_text()
    body = src[src.index("const TESTLIB_OK_EXIT_CODE"):src.index("export class JudgeEngine")]
    js = (body + """
const pairs = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = pairs.map(([exit, msg]) => {
  const res = { status: exit === 0 ? 'Accepted' : 'Nonzero Exit Status', exitStatus: exit };
  const ok = res.status === 'Accepted' && res.exitStatus === 0;
  return applyScoreRatio({ ok, judgeMsg: isScoringVerdict(res) ? msg : '' }).scoreRatio;
});
process.stdout.write(JSON.stringify(out));
""")
    r = subprocess.run([NODE, "-e", js], input=json.dumps(pairs), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


if not os.access(NODE, os.X_OK):
    print(f"SKIP engine parity: no node at {NODE} (set FCS_NODE)")
else:
    unit = engine([(e, m) for e, m, _, _ in UNIT])
    check("the engine gives the expected ratio on every unit case", unit == [x for _, _, _, x in UNIT],
          [(e, m[:60], got, x) for (e, m, _, x), got in zip(UNIT, unit) if got != x])
    dev = [(e, m[:60]) for (e, m, ours, _), x in zip(UNIT, unit) if ours != x]
    print(f"unit cases: {len(dev)} of {len(UNIT)} differ from the engine, all by the intended deviation: {dev}")
    check("unit cases: ours differs from the engine only where intended", dev == [(e, m[:60]) for e, m, o, x in UNIT
                                                                                    if o != x] and len(dev) == 4, dev)
    stored = sorted(stored_messages())
    bad = [(e, m[:80], J._ratio(m, e in (0, 7), e == 0), x) for (e, m), x in zip(stored, engine(stored))
           if J._ratio(m, e in (0, 7), e == 0) != x]
    print(f"stored messages: {len(bad)} of {len(stored)} distinct differ from the engine")
    check("stored messages: ours equals the engine on every one", not bad, bad[:5])
    check("stored messages found", len(stored) > 1000, len(stored))

r = J.judge_full(P + "247", "#include <cstdio>\nint main(){ printf(\"Ratio:1\\n\"); }\n")
check("247: a program that prints Ratio:1 scores 0", r["status"] == "done" and r["score"] == 0.0
      and sum(i.startswith("chk=7 ") for i in r["case_info"]) >= 10, (r["score"], r["case_info"][:1]))
# a case the checker scored scores as stored; a run the node's load killed (wall over 2x the limit) is not compared
sub = [json.loads(l) for l in open(W1532) if json.loads(l)["label"].endswith("/0")][0]["submissions"][0]
r = J.judge_full(P + "0", sub["code"])
same = [round(c, 4) == s for c, s, i in zip(r["cases"], sub["cases"], r["case_info"]) if i.startswith("chk=")]
check("problem 0: a stored program scores its stored cases", r["status"] == "done" and len(same) >= 35 and all(same),
      (len(same), sum(same), r["cases"][:5], sub["cases"][:5]))
done()
