"""rl/registry.py: the engine/trainer file registry of the external-engine RL jobs (rl/engine_job.sh,
rl/rl_job_ext.sh). Plain python3 (no container). Usage: python3 tests/test_registry.py; exit 0 when all pass."""
import io
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl"))
import registry as R  # noqa: E402

fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def raises(fn, msg):
    try:
        fn()
    except (AssertionError, ValueError):
        check(True, msg)
        return
    check(False, msg + " (no error)")


T = 1_800_000_000.0
run = tempfile.mkdtemp(prefix="reg_")

# parse_engine: routable host:port, TP equal to the GPU count
e = R.parse_engine("172.17.8.3:20040,8,0:1:2:3:4:5:6:7")
check(e == {"url": "http://172.17.8.3:20040", "tp": 8, "gpus": list(range(8))}, "parse_engine host:port")
raises(lambda: R.parse_engine("127.0.0.1:20040,8,0:1:2:3:4:5:6:7"), "parse_engine refuses 127.0.0.1")
raises(lambda: R.parse_engine("localhost:20040,8,0:1:2:3:4:5:6:7"), "parse_engine refuses localhost")
raises(lambda: R.parse_engine("172.17.8.3:20040,8,0:1:2:3"), "parse_engine refuses TP != GPU count")
raises(lambda: R.parse_engine("172.17.8.3,8,0:1:2:3:4:5:6:7"), "parse_engine refuses a missing port")

# register (loading) -> not live; serving -> live; start kept across re-registration
R.register(run, "w1", "nodeA", [e], state="loading", now=T)
check(R.live_engines(run, now=T + 1) == [], "a loading engine is not live")
check(len(R.records(run, now=T + 1, states=("loading", "serving"))) == 1, "a loading record is a record")
R.register(run, "w1", "nodeA", [e], state="serving", now=T + 600)
rec = json.load(open(R.engine_file(run, "w1")))
check(rec["start"] == T and rec["registered"] == T + 600 and rec["state"] == "serving", "re-registration keeps start")
check([x["url"] for x in R.live_engines(run, now=T + 610)] == ["http://172.17.8.3:20040"], "a serving engine is live")
check(R.live_engines(run, now=T + 600 + R.FRESH + 1) == [], "an engine without a fresh heartbeat is dead")

# heartbeat: beat refreshes; beat on a missing file reports it
check(R.beat(R.engine_file(run, "w1"), now=T + 2000), "beat on an existing record")
check(len(R.live_engines(run, now=T + 2050)) == 1, "a beat makes a stale engine live again")
check(not R.beat(os.path.join(run, "engines", "nope.json"), now=T), "beat on a missing record returns False")
check(not [n for n in os.listdir(os.path.join(run, "engines")) if ".tmp." in n], "no temporary files left")

# a corrupt or half-written file is skipped, not fatal
open(os.path.join(run, "engines", "bad.json"), "w").write("{not json")
check(len(R.live_engines(run, now=T + 2050)) == 1, "a corrupt record is skipped")
os.remove(os.path.join(run, "engines", "bad.json"))

# order and pick: registration order; the TP size with most engines; duplicates collapse
e2 = R.parse_engine("172.17.8.4:20040,8,0:1:2:3:4:5:6:7")
e4a, e4b = R.parse_engine("172.17.8.5:20040,4,0:1:2:3"), R.parse_engine("172.17.8.5:20080,4,4:5:6:7")
R.register(run, "w2", "nodeB", [e2], now=T + 2010)
R.register(run, "w3", "nodeC", [e4a, e4b], now=T + 2020)
live = R.live_engines(run, now=T + 2050)
check([x["job"] for x in live] == ["w1", "w2", "w3", "w3"], "live engines come in registration order")
got = R.pick(live, 1)
check(got is not None and got[1] in (4, 8), "pick returns one TP size")
check(R.pick([e | {"job": "w1"}, e2 | {"job": "w2"}], 3) is None, "pick needs n_min engines")
check(R.pick([e | {"job": "w1"}, e | {"job": "w1"}], 1)[0] == ["172.17.8.3:20040"], "pick drops duplicate urls")
p8 = R.pick([e | {"job": "w1"}, e2 | {"job": "w2"}, e4a | {"job": "w3"}], 1)
check(p8 == (["172.17.8.3:20040", "172.17.8.4:20040"], 8, 16, ["w1", "w2"]), "pick: addrs, TP, GPUs, jobs")
check(R.pick([], 1) is None, "pick on nothing")

# trainer_check: every named engine job live and serving
check(R.trainer_check(run, ["w1", "w2"], now=T + 2050), "trainer_check: both live")
check(not R.trainer_check(run, ["w1", "w9"], now=T + 2050), "trainer_check: an unknown job is not live")
R.deregister(run, "w2")
R.deregister(run, "w2")   # twice is fine
check(not R.trainer_check(run, ["w1", "w2"], now=T + 2050), "trainer_check: a deregistered job is not live")
R.deregister(run, "w3")

# engine_check: serve while young; alone after grace without a trainer; serve when the trainer beats; gone when its
# beat stops for > stale; stop on STOP
seen = os.path.join(run, "seen")
check(R.engine_check(run, T, seen_file=seen, now=T + 60) == "serve", "engine_check: young engine waits")
check(R.engine_check(run, T, seen_file=seen, now=T + R.TRAINER_GRACE + 1) == "alone", "engine_check: alone after grace")
R.trainer_beat(run, "w5", "nodeT", "172.17.9.9", T, now=T + 100)
check(R.engine_check(run, T, seen_file=seen, now=T + 120) == "serve" and os.path.exists(seen), "engine_check: sees the trainer")
check(R.engine_check(run, T, seen_file=seen, now=T + 100 + 300) == "serve", "engine_check: a 5-min-old beat is fine once seen")
check(R.engine_check(run, T, seen_file=seen, now=T + 100 + R.TRAINER_STALE + 1) == "gone", "engine_check: gone after stale")
check(R.engine_check(run, T, seen_file=None, now=T + 100 + 300) == "serve", "engine_check without memory, in grace")
# a stale trainer.json from an earlier trainer does not count as seen
seen2 = os.path.join(run, "seen2")
check(R.engine_check(run, T + 5000, seen_file=seen2, now=T + 5000 + 60) == "serve" and not os.path.exists(seen2),
      "engine_check: an old trainer heartbeat is not a sighting")
check(R.engine_check(run, T + 5000, seen_file=seen2, now=T + 5000 + R.TRAINER_GRACE + 1) == "alone",
      "engine_check: an old trainer heartbeat does not keep an engine alive")
# stale < FRESH: a beat older than stale but fresh is a trainer that was there and died, not "no trainer yet"
seen3 = os.path.join(run, "seen3")
R.trainer_beat(run, "w6", "nodeT", "172.17.9.9", T, now=T + 100)
check(R.engine_check(run, T, stale=8, seen_file=seen3, now=T + 100 + 20) == "gone" and os.path.exists(seen3),
      "engine_check: stale < FRESH, a 20-s-old beat with stale 8 -> gone")
check(R.engine_check(run, T, stale=8, seen_file=seen3, now=T + 100 + 5) == "serve", "engine_check: stale 8, 5-s beat -> serve")
open(os.path.join(run, "STOP"), "w").write("x")
check(R.engine_check(run, T, seen_file=seen, now=T + 120) == "stop", "engine_check: STOP wins")
os.remove(os.path.join(run, "STOP"))

# wait: a fake clock; returns as soon as an engine serves; times out without records; extends while one loads
clock = [T]
tick = lambda s: clock.__setitem__(0, clock[0] + s)  # noqa: E731
run2 = tempfile.mkdtemp(prefix="reg_")
out = io.StringIO()
check(R.wait(run2, timeout=100, load_timeout=300, poll=15, clock=lambda: clock[0], sleep=tick, out=out) == 2,
      "wait: no record -> 2 after the timeout")
check(clock[0] - T <= 100 + 15, "wait: gives up within one poll of the timeout")
clock[0] = T
R.register(run2, "w7", "n", [e], state="loading", now=T + 50)


def sleep_and_serve(s):
    tick(s)
    if clock[0] >= T + 400:   # the engine serves 400 s in: beyond --timeout but inside --load-timeout
        R.register(run2, "w7", "n", [e], state="serving", now=clock[0])
    else:
        R.beat(R.engine_file(run2, "w7"), now=clock[0])


out = io.StringIO()
rc = R.wait(run2, timeout=100, load_timeout=600, poll=15, clock=lambda: clock[0], sleep=sleep_and_serve, out=out)
check(rc == 0 and out.getvalue().strip() == "172.17.8.3:20040\t8\t8\tw7", f"wait: a loading engine extends the wait ({rc}, {out.getvalue()!r})")
clock[0] = T
R.register(run2, "w7", "n", [e], state="loading", now=T)
rc = R.wait(run2, timeout=100, load_timeout=200, poll=15, clock=lambda: clock[0],
            sleep=lambda s: (tick(s), R.beat(R.engine_file(run2, "w7"), now=clock[0])), out=io.StringIO())
check(rc == 2, "wait: a record that never serves -> 2 after the load timeout")
open(os.path.join(run2, "STOP"), "w").write("x")
check(R.wait(run2, clock=lambda: clock[0], sleep=tick, out=io.StringIO()) == 3, "wait: STOP -> 3")

# the CLI, as the job scripts call it
run3 = tempfile.mkdtemp(prefix="reg_")
cli = [sys.executable, os.path.join(os.path.dirname(R.__file__), "registry.py")]
sh = lambda *a: subprocess.run(cli + list(a), capture_output=True, text=True)  # noqa: E731
r = sh("register", run3, "w8", "n", "172.17.8.3:20040,8,0:1:2:3:4:5:6:7", "--state", "serving", "--log", "/x")
check(r.returncode == 0 and json.loads(r.stdout)["log"] == "/x", "cli register")
r = sh("wait", run3, "--timeout", "5", "--poll", "1")
check(r.returncode == 0 and r.stdout == "172.17.8.3:20040\t8\t8\tw8\n", f"cli wait: {r.stdout!r} {r.stderr!r}")
check(sh("beat", R.engine_file(run3, "w8")).returncode == 0, "cli beat")
check(sh("trainer-check", run3, "w8").returncode == 0 and sh("trainer-check", run3, "w9").returncode == 1, "cli trainer-check")
r = sh("register", run3, "w9", "n", "127.0.0.1:1,1,0")
check(r.returncode != 0, "cli register refuses 127.0.0.1")
check(sh("trainer-beat", run3, "w5", "n", "172.17.9.9", str(T)).returncode == 0, "cli trainer-beat")
check(sh("engine-check", run3, str(T)).stdout.strip() == "serve", "cli engine-check")
check(sh("deregister", run3, "w8").returncode == 0 and not os.path.exists(R.engine_file(run3, "w8")), "cli deregister")
r = sh("wait", run3, "--timeout", "2", "--poll", "1")
check(r.returncode == 2, "cli wait times out with 2")

import shutil  # noqa: E402
for d in (run, run2, run3):
    shutil.rmtree(d, ignore_errors=True)
print(f"{len(fails)} failed" + (f": {fails}" if fails else ""))
sys.exit(1 if fails else 0)
