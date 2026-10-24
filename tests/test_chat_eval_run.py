"""tools/chat_eval_long.py's run outcome without a server (L.episode stubbed): an exception in one episode becomes its
aborted row and the others finish; rows still missing give exit 99 and the rerun resumes only those; an episode is
given up after --max-episode-tries rows that are not clean; a run whose episodes all crash exits 1; a judge-infra row
reruns; a 400 or a reply without choices aborts one episode instead of the run."""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp())
DATA = TMP / "data.jsonl"
DATA.write_text("".join(json.dumps({"label": f"L{i}", "metadata": {"pid": i, "messages": [{"role": "user", "content": "p"}]}})
                        + "\n" for i in range(4)))
src = (ROOT / "tools/chat_eval_long.py").read_text()
assert src.rstrip().endswith("asyncio.run(main())")
sys.argv = ["x", "--base-url", "http://a,http://b", "--model", "flint", "--data", str(DATA), "--out", str(TMP / "o0")]
g = {"__file__": str(ROOT / "tools/chat_eval_long.py"), "__name__": "chat_eval_long"}
exec(compile(src.rstrip()[: -len("asyncio.run(main())")], "chat_eval_long", "exec"), g)
L = g["L"]
ok = True


def check(c, what, detail=""):
    global ok
    print(("PASS " if c else "FAIL ") + what + ("" if c else f"  {detail}"))
    ok &= bool(c)


mode = {}   # (pid, episode) -> "raise" | "abort" | "infra"; default a clean score


async def fake_episode(inp, messages, label, ckpt=None):
    await asyncio.sleep(0)
    m = mode.get((inp.sample.metadata["pid"], inp.sample.group_index))
    if m == "raise":
        raise RuntimeError("boom")
    if m == "abort":
        raise g["Aborted"]("server gone")
    return None, 0.5, {"fo_lg_submissions": 1, "fo_judge_infra_error": float(m == "infra")}


L.episode = fake_episode


def run(out):
    g["_stop"][0] = None
    g["OUT"] = TMP / out
    g["OUT"].mkdir(exist_ok=True)
    try:
        asyncio.run(g["main"]())
        return 0
    except SystemExit as ex:
        return ex.code


def rows(out):
    return g["_read"](TMP / out / "results.jsonl")


mode = {(1, 0): "raise"}
c = run("o1")
r = rows("o1")
check(c == 99 and len(r) == 4 and sum(x["S"] is not None for x in r) == 3, "one crash: other episodes finish, exit 99", (c, r))
bad = [x for x in r if x["S"] is None]
check(bad and bad[0]["crash"] and "RuntimeError: boom" in bad[0]["aborted"] and "Traceback" in bad[0]["traceback"],
      "crash row keeps the error and the traceback", bad)
mode = {}
c = run("o1")
r = rows("o1")
check(c == 0 and len(r) == 5 and r[-1]["pid"] == 1 and r[-1]["S"] == 0.5, "rerun resumes only the crashed episode, exit 0", (c, r))

mode = {(i, 0): "raise" for i in range(4)}
check(run("o2") == 1, "every episode crashes: exit 1")

mode = {(2, 0): "abort"}
codes = [run("o3") for _ in range(4)]
r = rows("o3")
check(codes == [99, 99, 0, 0] and len(r) == 6, "aborted 3 times: given up, exit 0, no fourth try", (codes, len(r)))

mode = {(3, 0): "infra"}
c1 = run("o4")
mode = {}
c2 = run("o4")
r = rows("o4")
check(c1 == 99 and c2 == 0 and len(r) == 5 and r[-1]["pid"] == 3, "judge-infra row reruns", (c1, c2, r))
line, code = g["outcome"]([(i, 0) for i in range(4)], r, [], 3, 1)
check(code == 0 and line.startswith("DONE 4/4") and "mean final score 50.00" in line, "summary over clean rows", line)
line, code = g["outcome"]([(i, 0) for i in range(5)], r, [], 3, 1)
check(code == 99 and line.startswith("INCOMPLETE 4/5") and "to rerun 1" in line, "missing key: INCOMPLETE, 99", line)

with open(TMP / "o4/results.jsonl", "a") as f:
    f.write('{"pid": 0, "epis')   # a line cut short by a kill
check(len(rows("o4")) == 5 and run("o4") == 0, "a cut line is skipped")

# checkpoints through main(): episodes that save every round and pause when asked
import os  # noqa: E402
import time  # noqa: E402

starts = {}   # (pid, episode) -> the round each start resumed from, over all runs
ROUNDS = {}   # pid -> rounds the episode takes (default 5)


async def ck_episode(inp, messages, label, ckpt=None):
    key = (inp.sample.metadata["pid"], inp.sample.group_index)
    st = ckpt.load() if ckpt is not None else None
    r0 = st["rounds"] if st else 0
    starts.setdefault(key, []).append(r0)
    work = TMP / f"work_{key[0]}"
    work.mkdir(exist_ok=True)
    (work / "f.txt").write_text(str(r0))
    for rnd in range(r0 + 1, ROUNDS.get(key[0], 5) + 1):
        await asyncio.sleep(0.01)
        m = mode.get(key)
        if m == "abort" and rnd == 3:
            raise g["Aborted"]("server gone")
        if m == "raise" and rnd == 3:
            raise RuntimeError("boom")
        if ckpt is not None:
            ckpt.save({"rounds": rnd}, str(work))
            if ckpt.stop:
                raise L.Paused(f"paused after round {rnd}")
    return None, 0.5, {"fo_lg_submissions": 1}


L.episode = ck_episode
g["WATCH_S"], g["STOP_GRACE_S"] = 0.05, 5
mode = {}
ROUNDS = {i: 100000 for i in range(4)}   # long episodes: only the time guard ends this run
os.environ["SLURM_JOB_END_TIME"] = str(int(time.time()) + g["STOP_MARGIN_S"] + 1)
c = run("c1")
del os.environ["SLURM_JOB_END_TIME"]
st = sorted(p.name for p in (TMP / "c1/state").iterdir())
check(c == 99 and not rows("c1") and len([x for x in st if x.endswith(".pkl")]) == 4 and len([x for x in st if x.endswith(".tgz")]) == 4,
      "end-time guard: every episode paused at a save, no rows, exit 99", (c, st))
ROUNDS = {}
for k in starts:
    ROUNDS[k[0]] = starts[k][-1] + 3
saved = {k: v for k, v in starts.items()}
c = run("c1")
r = rows("c1")
check(c == 0 and len(r) == 4 and all(starts[k][-1] > 0 for k in saved) and not list((TMP / "c1/state").iterdir())
      and all("minutes" in x for x in r), "rerun resumes every episode from its save, exit 0, states removed", (c, starts, r))

starts.clear()
ROUNDS = {3: 100000}
g["args"].tail_exit_min = 0.002   # 0.12 s
c = run("c2")
g["args"].tail_exit_min = 0
r = rows("c2")
check(c == 3 and sorted(x["pid"] for x in r) == [0, 1, 2] and (TMP / "c2/state/3_e0.pkl").exists(),
      "tail exit: a server without episodes stops the run, exit 3, the long one keeps its state", (c, r))
ROUNDS = {3: starts[(3, 0)][-1] + 2}
check(run("c2") == 0 and starts[(3, 0)][-1] > 0 and len(rows("c2")) == 4, "tail: the rerun resumes it")

g["args"].no_ckpt = True
ROUNDS = {i: 100000 for i in range(4)}
os.environ["SLURM_JOB_END_TIME"] = str(int(time.time()) + g["STOP_MARGIN_S"] + 1)
c = run("c3")
del os.environ["SLURM_JOB_END_TIME"]
g["args"].no_ckpt = False
check(c == 1 and not rows("c3") and not (TMP / "c3/state").exists(), "no row and no save in a run: exit 1", c)

ROUNDS = {}
mode = {(1, 0): "abort", (2, 0): "raise"}
c = run("c4")
check(c == 99 and (TMP / "c4/state/1_e0.pkl").exists() and not (TMP / "c4/state/2_e0.pkl").exists(),
      "transport abort keeps the state, a crash removes it", c)
mode = {}
starts.clear()
c = run("c4")
check(c == 0 and starts[(1, 0)] == [2] and starts[(2, 0)] == [0], "abort resumes at round 2, crash starts fresh", starts)

# ChatSession: a 400 that is not about the context, and a reply without choices
from types import SimpleNamespace  # noqa: E402


class _Resp:
    def __init__(self, status, d):
        self.status, self.d = status, d

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    async def text(self):
        return str(self.d)

    async def json(self):
        if self.d == "bad":
            raise json.JSONDecodeError("x", "bad", 0)
        return self.d


posts = []


class _Http:
    def __init__(self, status, d):
        self.status, self.d = status, d

    def post(self, url, json):
        posts.append(url)
        return _Resp(self.status, self.d)

    def get(self, url, timeout=None):   # /health: the server is up, so the abort is this episode's alone
        return _Resp(200, {})


_sleep = asyncio.sleep


async def _fast(t, *a, **k):
    await _sleep(0)


asyncio.sleep = _fast
hist = [{"role": "system", "content": "s"}, {"role": "user", "content": "inst"}]
for status, d, what in [(400, "grammar compile failed", "400"), (200, {"error": "x"}, "no choices"), (200, "bad", "bad JSON")]:
    g["_http"] = _Http(status, d)
    posts.clear()
    sess = g["ChatSession"](SimpleNamespace(evaluation=True, base_url="http://b"), "a0", True, max_len=262144)
    try:
        asyncio.run(sess.turn(hist, 100))
        err = None
    except Exception as ex:
        err = ex
    check(isinstance(err, g["Aborted"]) and posts and set(posts) == {"http://b/v1/chat/completions"}
          and len(posts) == (1 if status == 400 else 4), f"{what}: Aborted after {len(posts)} posts", repr(err))
check(isinstance(err, g["_HttpAborted"]) and err.url == "http://b", "transport abort names its server")
asyncio.sleep = _sleep
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
