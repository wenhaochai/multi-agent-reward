"""tools/chat_eval_long.py's per-turn routing without a server (fake clock, fake HTTP): (a) when one server's episodes
end, the next turns move episodes until no two servers differ by 2 or more; (b) a lone episode on 4 servers keeps
every server from idling longer than the bound, without moving on every turn; (c) after a run whose episodes moved,
every server's count is 0; (d) the agents of one episode move together; (e) a dead server (failed retries and
/health_generate) gets no new episodes, its episode goes on elsewhere, and the run exits 99; the rerun resumes and exits 0."""
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp())
DATA = TMP / "data.jsonl"
DATA.write_text("".join(json.dumps({"label": f"L{i}", "metadata": {"pid": i, "messages": [{"role": "user", "content": "p"}]}})
                        + "\n" for i in range(6)))
U = [f"http://s{k}" for k in range(4)]
src = (ROOT / "tools/chat_eval_long.py").read_text()
assert src.rstrip().endswith("asyncio.run(main())")
sys.argv = ["x", "--base-url", ",".join(U), "--model", "flint", "--data", str(DATA), "--out", str(TMP / "o0"),
            "--concurrency", "3"]
g = {"__file__": str(ROOT / "tools/chat_eval_long.py"), "__name__": "chat_eval_long"}
exec(compile(src.rstrip()[: -len("asyncio.run(main())")], "chat_eval_long", "exec"), g)
L = g["L"]
ok = True


def check(c, what, detail=""):
    global ok
    print(("PASS " if c else "FAIL ") + what + ("" if c else f"  {detail}"))
    ok &= bool(c)


clock = [1e6]
g["time"] = SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0])   # llm_s reads monotonic
TURN_S = 60.0
posts, dead, probes = [], set(), []   # posts: (url, start, end); probes: (url, timeout)


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

    async def json(self):
        return self.d


REPLY = {"choices": [{"message": {"content": "plan", "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
                      "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


class _Http:
    def post(self, url, json):
        base = url.rsplit("/v1/", 1)[0]
        t0 = clock[0]
        if base in dead:
            return _Resp(None, None)
        clock[0] += TURN_S
        posts.append((base, t0, clock[0]))
        return _Resp(200, REPLY)

    def get(self, url, timeout=None):
        probes.append((url, timeout.total if timeout else None))
        return _Resp(None if url.rsplit("/health", 1)[0] in dead else 200, {})

    async def close(self):
        pass


_real_sleep = asyncio.sleep


async def _fast(t, *a, **k):
    clock[0] += t
    await _real_sleep(0)


g["_http"] = _Http()
g["asyncio"] = SimpleNamespace(**{k: getattr(asyncio, k) for k in dir(asyncio) if not k.startswith("__")})
g["asyncio"].sleep = _fast
CS = g["ChatSession"]
_active, _last = g["_active"], g["_last"]
HIST = [{"role": "system", "content": "s"}, {"role": "user", "content": "inst"}]


def reset():
    for u in U:
        _active[u], _last[u] = 0, clock[0]
    g["DOWN"].clear()
    posts.clear()


def episode_inp(url, k=0):
    _active[url] += 1
    return SimpleNamespace(evaluation=True, base_url=url, moved_at=clock[0], moves=0,
                           sample=SimpleNamespace(index=k, group_index=0, metadata={"pid": k}))


# (a) 40 episodes, 10 per server; the 10 on s0 end; one turn each moves episodes until the gap is under 2
reset()
inps = [episode_inp(U[k % 4], k) for k in range(40)]
for x in [x for x in inps if x.base_url == U[0]]:
    _active[x.base_url] -= 1
    inps.remove(x)
for x in inps:
    asyncio.run(CS(x, "agent1", True, max_len=262144).turn(HIST, 100))
cnt = {u: sum(x.base_url == u for x in inps) for u in U}
check(max(_active.values()) - min(_active.values()) <= 1 and cnt == _active and sum(_active.values()) == 30
      and sum(x.moves for x in inps) == 7, "(a) balanced after one pass, counts match", (_active, cnt))

g["args"].no_route = True   # --no-route: an imbalance moves nothing
reset()
x = episode_inp(U[0])
for k in range(3):
    episode_inp(U[0], k + 1)
asyncio.run(CS(x, "agent1", True, max_len=262144).turn(HIST, 100))
check(x.moves == 0 and x.base_url == U[0] and _active[U[0]] == 4, "--no-route: no move", _active)
g["args"].no_route = False

# (b) one episode, 4 servers, 3 h of fake time
reset()
x = episode_inp(U[0])
sess = CS(x, "agent1", True, max_len=262144)
t_start, n_turns = clock[0], 0
while clock[0] - t_start < 3 * 3600:
    sess.used = 0
    asyncio.run(sess.turn(HIST, 100))
    clock[0] += 30   # the command
    n_turns += 1
gaps = {}
for u in U:
    ts = [(s, e) for v, s, e in posts if v == u]
    edges = [t_start] + [t for se in ts for t in se] + [clock[0]]
    gaps[u] = max(edges[i + 1] - edges[i] for i in range(0, len(edges), 2))
IDLE, DW = g["IDLE_S"], g["DWELL_S"]
check(max(gaps.values()) <= IDLE + 3 * DW + TURN_S + 30 and x.moves < n_turns / 4,
      f"(b) no server idles over IDLE_S + 3*DWELL_S + a turn (max {max(gaps.values()):.0f} s), "
      f"{x.moves} moves in {n_turns} turns", gaps)
late = {}
for u in U:   # after the first hour the visits are spread: the bound is IDLE_S + DWELL_S + a turn
    ts = [(s, e) for v, s, e in posts if v == u and s > t_start + 3600]
    late[u] = max(ts[i + 1][0] - ts[i][1] for i in range(len(ts) - 1))
check(max(late.values()) <= IDLE + DW + TURN_S + 30, f"(b) steady state: max idle {max(late.values()):.0f} s", late)
check(_active[x.base_url] == 1 and sum(_active.values()) == 1, "(b) one episode counted once", _active)

# (d) an episode with two agents: both go to the same new server, and the counts move by one
reset()
for k in range(3):
    episode_inp(U[0], k)
x = episode_inp(U[0], 9)
s1, s2 = CS(x, "agent1", True, max_len=262144), CS(x, "agent2", True, max_len=262144)


async def both():
    return await asyncio.gather(s1.turn(HIST, 100), s2.turn(HIST, 100))


asyncio.run(both())
check(x.moves == 1 and [p[0] for p in posts] == [x.base_url] * 2 and _active[U[0]] == 3 and _active[x.base_url] == 1,
      "(d) both agents move together", (posts, _active))


# (c), (e): whole runs through main(), episodes stubbed to make a few turns each
async def fake_episode(inp, messages, label, ckpt=None):
    s = CS(inp, "agent1", True, max_len=262144)
    for _ in range(5):
        s.used = 0
        await s.turn(HIST, 100)
        await asyncio.sleep(0)
    return None, 0.5, {"fo_lg_submissions": 1}


L.episode = fake_episode
g["aiohttp"] = SimpleNamespace(ClientSession=lambda **k: _Http(), ClientTimeout=aiohttp.ClientTimeout,
                               ClientError=aiohttp.ClientError)


def run(out):
    g["_stop"][0] = None
    g["OUT"] = TMP / out
    g["OUT"].mkdir(exist_ok=True)
    try:
        asyncio.run(g["main"]())
        return 0
    except SystemExit as ex:
        return ex.code


reset()
c = run("o1")
check(c == 0 and all(v == 0 for v in _active.values()), "(c) all counts 0 after the run", (c, _active))

reset()
dead.add(U[1])   # s1 refuses connections: its episode fails the retries and /health_generate, then goes on elsewhere
c = run("o2")
rows = g["_read"](TMP / "o2/results.jsonl")
after = [p for p in posts if p[0] == U[1]]
check(g["DOWN"] == {U[1]} and not after, "(e) dead server marked down, never posted to", (g["DOWN"], after))
check(probes and all(u == U[1] + "/health_generate" and t == 30 for u, t in probes),
      "(e) the probe is /health_generate (it runs the engine), 30 s", probes)
check(c == 99 and 0 < len(rows) < 6 and all(r["S"] == 0.5 for r in rows) and all(v == 0 for v in _active.values()),
      "(e) its episode finished elsewhere; no new episodes after; exit 99 while rows are missing", (c, rows, _active))
dead.clear()
g["DOWN"].clear()
c = run("o2")
check(c == 0 and len(g["_read"](TMP / "o2/results.jsonl")) == 6, "(e) rerun resumes the rest and exits 0", c)
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
