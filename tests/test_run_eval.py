"""run_eval.py's run outcome without a server (game.episode stubbed): an exception in one episode becomes its aborted
row and the others finish; rows still missing give exit 99 and the rerun resumes only those; an episode is given up
after --max-episode-tries rows that are not clean; a run whose episodes all crash exits 1; a judge-infra row reruns;
checkpoints pause at the job's end and resume; a tail stop exits 3. With fake HTTP: (c) after a run whose episodes
moved, every server's count is 0; (e) a dead server (failed retries and /health_generate) gets no new episodes, its
episode goes on elsewhere, and the run exits 99; the rerun resumes and exits 0. A 400 or a reply without choices
aborts one episode instead of the run."""
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import aiohttp

from helpers import check, done, env

env(MA_LG_N=1)
import run_eval as RE  # noqa: E402
from flint import Aborted  # noqa: E402
from flint import chat as CH  # noqa: E402
from flint import game as G  # noqa: E402

TMP = Path(tempfile.mkdtemp())
DATA = TMP / "data.jsonl"
DATA.write_text("".join(json.dumps({"label": f"L{i}", "metadata": {"pid": i, "messages": [{"role": "user", "content": "p"}]}})
                        + "\n" for i in range(4)))
args = RE.parse(["--base-url", "http://a,http://b", "--model", "flint", "--data", str(DATA), "--out", str(TMP / "o0")])
mode = {}   # (pid, episode) -> "raise" | "abort" | "infra"; default a clean score


async def fake_episode(inp, messages, label, new_session, ckpt=None):
    await asyncio.sleep(0)
    m = mode.get((inp.sample.metadata["pid"], inp.sample.group_index))
    if m == "raise":
        raise RuntimeError("boom")
    if m == "abort":
        raise Aborted("server gone")
    return None, 0.5, {"fo_lg_submissions": 1, "fo_judge_infra_error": float(m == "infra")}


G.episode = fake_episode


def run(out, client=None):
    args.out = str(TMP / out)
    return asyncio.run(RE.main(args, client))


def rows(out):
    return RE._read(TMP / out / "results.jsonl")


mode = {(1, 0): "raise"}
c = run("o1")
r = rows("o1")
check("one crash: other episodes finish, exit 99", c == 99 and len(r) == 4 and sum(x["S"] is not None for x in r) == 3,
      (c, r))
bad = [x for x in r if x["S"] is None]
check("crash row keeps the error and the traceback", bad and bad[0]["crash"] and "RuntimeError: boom" in bad[0]["aborted"]
      and "Traceback" in bad[0]["traceback"], bad)
mode = {}
c = run("o1")
r = rows("o1")
check("rerun resumes only the crashed episode, exit 0", c == 0 and len(r) == 5 and r[-1]["pid"] == 1
      and r[-1]["S"] == 0.5, (c, r))
mode = {(i, 0): "raise" for i in range(4)}
check("every episode crashes: exit 1", run("o2") == 1)
mode = {(2, 0): "abort"}
codes = [run("o3") for _ in range(4)]
check("aborted 3 times: given up, exit 0, no fourth try", codes == [99, 99, 0, 0] and len(rows("o3")) == 6,
      (codes, len(rows("o3"))))
mode = {(3, 0): "infra"}
c1 = run("o4")
mode = {}
c2 = run("o4")
r = rows("o4")
check("judge-infra row reruns", c1 == 99 and c2 == 0 and len(r) == 5 and r[-1]["pid"] == 3, (c1, c2, r))
line, code = RE.outcome([(i, 0) for i in range(4)], r, [], 3, 1)
check("summary over clean rows", code == 0 and line.startswith("DONE 4/4") and "mean final score 50.00" in line, line)
line, code = RE.outcome([(i, 0) for i in range(5)], r, [], 3, 1)
check("missing key: INCOMPLETE, 99", code == 99 and line.startswith("INCOMPLETE 4/5") and "to rerun 1" in line, line)
with open(TMP / "o4/results.jsonl", "a") as f:
    f.write('{"pid": 0, "epis')   # a line cut short by a kill
check("a cut line is skipped", len(rows("o4")) == 5 and run("o4") == 0)
mode = {(0, 1): "infra"}   # a second episode per problem: new rows go after the cut line
args.episodes = 2
c = run("o4")
args.episodes = 1
r = rows("o4")
check("the row after a cut line is kept whole (the cut line is ended first)", c == 99 and len(r) == 9
      and sorted(x["pid"] for x in r[5:]) == [0, 1, 2, 3] and r[5]["episode"] == 1, (c, len(r), r[5:]))
mode = {}

# checkpoints through main(): episodes that save every round and pause when asked
starts = {}   # (pid, episode) -> the round each start resumed from, over all runs
ROUNDS = {}   # pid -> rounds the episode takes (default 5)
FAIL_AT = [3]   # the round at which "abort" and "raise" happen


async def ck_episode(inp, messages, label, new_session, ckpt=None):
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
        if m == "abort" and rnd == FAIL_AT[0]:
            raise Aborted("server gone")
        if m == "raise" and rnd == FAIL_AT[0]:
            raise RuntimeError("boom")
        if ckpt is not None:
            ckpt.save({"rounds": rnd}, str(work))
            if ckpt.stop:
                raise G.Paused(f"paused after round {rnd}")
    return None, 0.5, {"fo_lg_submissions": 1}


G.episode = ck_episode
RE.WATCH_S, RE.STOP_GRACE_S = 0.05, 5
ROUNDS = {i: 100000 for i in range(4)}   # long episodes: only the time guard ends this run
os.environ["SLURM_JOB_END_TIME"] = str(int(time.time()) + RE.STOP_MARGIN_S + 1)
c = run("c1")
del os.environ["SLURM_JOB_END_TIME"]
st = sorted(p.name for p in (TMP / "c1/state").iterdir())
check("end-time guard: every episode paused at a save, no rows, exit 99", c == 99 and not rows("c1")
      and len([x for x in st if x.endswith(".pkl")]) == 4 and len([x for x in st if x.endswith(".tgz")]) == 4, (c, st))
ROUNDS = {k[0]: starts[k][-1] + 3 for k in starts}
saved = dict(starts)
c = run("c1")
r = rows("c1")
check("rerun resumes every episode from its save, exit 0, states removed", c == 0 and len(r) == 4
      and all(starts[k][-1] > 0 for k in saved) and not list((TMP / "c1/state").iterdir())
      and all("minutes" in x for x in r), (c, starts, r))
starts.clear()
ROUNDS = {3: 100000}
args.tail_exit_min = 0.002   # 0.12 s
c = run("c2")
args.tail_exit_min = 0
r = rows("c2")
check("tail exit: a server without episodes stops the run, exit 3, the long one keeps its state", c == 3
      and sorted(x["pid"] for x in r) == [0, 1, 2] and (TMP / "c2/state/3_e0.pkl").exists(), (c, r))
ROUNDS = {3: starts[(3, 0)][-1] + 2}
check("tail: the rerun resumes it", run("c2") == 0 and starts[(3, 0)][-1] > 0 and len(rows("c2")) == 4)
args.no_ckpt = True
ROUNDS = {i: 100000 for i in range(4)}
os.environ["SLURM_JOB_END_TIME"] = str(int(time.time()) + RE.STOP_MARGIN_S + 1)
c = run("c3")
del os.environ["SLURM_JOB_END_TIME"]
args.no_ckpt = False
check("no row and no save in a run: exit 1", c == 1 and not rows("c3") and not (TMP / "c3/state").exists(), c)
ROUNDS = {}
mode = {(1, 0): "abort", (2, 0): "raise"}
c = run("c4")
check("transport abort keeps the state, a crash removes it", c == 99 and (TMP / "c4/state/1_e0.pkl").exists()
      and not (TMP / "c4/state/2_e0.pkl").exists(), c)
mode = {}
starts.clear()
c = run("c4")
check("abort resumes at round 2, crash starts fresh", c == 0 and starts[(1, 0)] == [2] and starts[(2, 0)] == [0], starts)
mode = {(0, 0): "raise"}
ROUNDS = {i: 100000 for i in range(4)}
FAIL_AT[0] = 1   # pid 0 crashes at its first round, before the stop (2-3 s ahead) can pause it
os.environ["SLURM_JOB_END_TIME"] = str(int(time.time()) + RE.STOP_MARGIN_S + 3)
c = run("c5")
del os.environ["SLURM_JOB_END_TIME"]
FAIL_AT[0] = 3
r = rows("c5")
st = sorted(p.name for p in (TMP / "c5/state").iterdir())
check("time stop: one crash and three paused episodes, exit 99", c == 99 and [(x["pid"], x.get("crash")) for x in r]
      == [(0, True)] and len([x for x in st if x.endswith(".pkl")]) == 3, (c, r, st))
mode = {(i, 0): "raise" for i in range(4)}
ROUNDS = {}
c = run("c6")
check("every episode crashes after its saves: exit 1", c == 1 and len(rows("c6")) == 4
      and not list((TMP / "c6/state").iterdir()), (c, rows("c6")))
mode = {}


# fake HTTP: servers that answer, refuse connections, or reply with errors
class _Resp:
    def __init__(self, status, d):
        self.status, self.d = status, d

    async def __aenter__(self):
        if self.status is None:
            raise aiohttp.ClientConnectionError("connection refused")
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


REPLY = {"choices": [{"message": {"content": "plan", "tool_calls": [{"function": {"name": "bash", "arguments":
         '{"command": "ls"}'}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
posts, dead, probes = [], set(), []


class _Http:
    def __init__(self, status=200, d=REPLY):
        self.status, self.d = status, d

    def post(self, url, json):
        base = url.rsplit("/v1/", 1)[0]
        posts.append(base)
        return _Resp(None if base in dead else self.status, self.d)

    def get(self, url, timeout=None):
        probes.append((url, timeout.total if timeout else None))
        return _Resp(None if url.rsplit("/health", 1)[0] in dead else 200, {})

    async def close(self):
        pass


_sleep = asyncio.sleep


async def _fast(t, *a, **k):
    await _sleep(0)

CH.asyncio = SimpleNamespace(**{k: getattr(asyncio, k) for k in dir(asyncio) if not k.startswith("__")})
CH.asyncio.sleep = _fast
CH.aiohttp = SimpleNamespace(ClientSession=lambda **k: _Http(), ClientTimeout=aiohttp.ClientTimeout,
                             ClientError=aiohttp.ClientError)
HIST = [{"role": "system", "content": "s"}, {"role": "user", "content": "inst"}]


async def http_episode(inp, messages, label, new_session, ckpt=None):   # a few model turns each
    s = new_session(inp, "agent1")
    for _ in range(5):
        s.used = 0
        await s.turn(HIST, 100)
        await _sleep(0)
    return None, 0.5, {"fo_lg_submissions": 1}


G.episode = http_episode
DATA.write_text("".join(json.dumps({"label": f"L{i}", "metadata": {"pid": i, "messages": [{"role": "user", "content": "p"}]}})
                        + "\n" for i in range(6)))
args.base_url, args.concurrency = ",".join(f"http://s{k}" for k in range(4)), 3
cl = RE.make_client(args)
c = run("h1", cl)
check("(c) all counts 0 after the run", c == 0 and all(v == 0 for v in cl.active.values()), (c, cl.active))
dead.add("http://s1")   # s1 refuses connections: its episode fails the retries and /health_generate, then goes on elsewhere
posts.clear()
cl = RE.make_client(args)
c = run("h2", cl)
r = rows("h2")
check("(e) dead server marked down, never posted to after", cl.down == {"http://s1"}
      and posts.count("http://s1") == 4, (cl.down, posts.count("http://s1")))
check("(e) the probe is /health_generate (it runs the engine), 30 s",
      probes and all(u == "http://s1/health_generate" and t == 30 for u, t in probes), probes)
check("(e) its episode finished elsewhere; no new episodes after; exit 99 while rows are missing",
      c == 99 and 0 < len(r) < 6 and all(x["S"] == 0.5 for x in r) and all(v == 0 for v in cl.active.values()),
      (c, r, cl.active))
dead.clear()
c = run("h2", RE.make_client(args))
check("(e) rerun resumes the rest and exits 0", c == 0 and len(rows("h2")) == 6, c)

# a 400 that is not about the context, a reply without choices, a bad JSON reply: this episode is aborted
for status, d, what in [(400, "grammar compile failed", "400"), (200, {"error": "x"}, "no choices"),
                        (200, "bad", "bad JSON")]:
    cl = CH.Client(["http://b"], "flint")
    cl.http = _Http(status, d)
    posts.clear()
    sess = CH.ChatSession(cl, SimpleNamespace(evaluation=True, base_url="http://b"), "a0", 262144)
    try:
        asyncio.run(sess.turn(HIST, 100))
        err = None
    except Exception as ex:
        err = ex
    check(f"{what}: Aborted after {len(posts)} posts", isinstance(err, Aborted) and set(posts) == {"http://b"}
          and len(posts) == (1 if status == 400 else 4), repr(err))
check("transport abort names its server", isinstance(err, CH.HttpAborted) and err.url == "http://b")
done()
