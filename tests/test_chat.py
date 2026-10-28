"""flint.chat without a server: the tool-call prompt, the history conversion to tool calls, a tool-call response turned
into an action (extra calls queued for the next rounds), the model call's numbers on the agent's turn records, and the
routing of episodes over servers (fake clock, fake HTTP): (a) when one server's episodes end, the next turns move
episodes until no two servers differ by 2 or more; (b) a lone episode on 4 servers keeps every server from idling longer
than the bound, without moving on every turn; (d) the agents of one episode move together."""
import asyncio
import json
from types import SimpleNamespace

from helpers import check, done, env

env(MA_LG_N=1)
from flint import chat as CH  # noqa: E402
from flint import game as G  # noqa: E402
from flint import prompts as P  # noqa: E402

pr = G.instance("Read n and print it.", "agent1")
check("prompt: tool-call rules", "mswea_bash_command" not in pr and "AT LEAST ONE bash tool call" in pr and "```bash" in pr)
check("prompt: long-work tools", "`submit` or `submit <file>`" in pr and pr.index("## Long work") < pr.index("## Important Rules"))
check("prompt: workflow and problem", "7. `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` also submits solution.cpp" in pr
      and "Read n and print it." in pr)
check("system: mini.yaml", P.SYSTEM.startswith("You are a helpful assistant") and "format_example" not in P.SYSTEM)
check("observation: JSON", json.loads(P.observation({"output": "hi\n", "returncode": 0, "exception_info": ""}))
      == {"returncode": 0, "output": "hi\n"})
chat = [{"role": "system", "content": "s"}, {"role": "user", "content": "inst"},
        {"role": "assistant", "content": "plan" + CH.block("ls -la"), "reasoning_content": "r1"},
        {"role": "user", "content": '{"returncode": 0}'},
        {"role": "assistant", "content": "no call", "reasoning_content": "r2"},
        {"role": "user", "content": "Tool call error: ..."}]
api = CH.to_api(chat)
check("history: block -> tool call, reasoning kept", api[2]["content"] == "plan"
      and api[2]["tool_calls"][0]["function"]["arguments"] == json.dumps({"command": "ls -la"})
      and api[2]["reasoning_content"] == "r1")
check("history: observation -> tool msg", api[3]["role"] == "tool" and api[3]["tool_call_id"] == api[2]["tool_calls"][0]["id"])
check("history: a format error stays a user message",
      api[4] == {"role": "assistant", "content": "no call", "reasoning_content": "r2"} and api[5]["role"] == "user")
m = CH.ACTION_RE.findall("x" + CH.block("submit"))
check("response: tool call -> one action the game parses", m == ["submit"] and G.SUBMIT_RE.fullmatch(m[0].strip()))
check("tool: strict, so Flint's grammar holds the arguments", CH.BASH_TOOL["function"]["strict"] is True)


class _Resp:
    def __init__(self, d, status=200):
        self.d, self.status = d, status

    async def __aenter__(self):
        if self.status is None:
            import aiohttp
            raise aiohttp.ClientConnectionError("connection refused")
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return self.d


def tool_reply(cmds, usage, args=None, reasoning="R"):
    tcs = [{"id": "x", "type": "function", "function": {"name": "bash", "arguments": a}}
           for a in (args or ['{"command": "%s"}' % c for c in cmds])]
    return {"choices": [{"message": {"reasoning_content": reasoning, "content": "plan", "tool_calls": tcs},
                         "finish_reason": "tool_calls"}], "usage": usage}


class _Http:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def post(self, url, json):
        self.calls.append(json)
        return _Resp(self.reply)


# a response with two calls: the first runs now, the second a round later without a model call
client = CH.Client(["http://x"], "flint")
client.http = _Http(tool_reply(["g++ a.cpp", "submit a.cpp"], {"prompt_tokens": 100, "completion_tokens": 50}))
sess = CH.ChatSession(client, SimpleNamespace(evaluation=True), "a0", 262144)
hist = [{"role": "system", "content": "s"}, {"role": "user", "content": "inst"}]
t1, c1 = asyncio.run(sess.turn(hist, 1000))
check("turn 1: first call is the action, second queued", CH.ACTION_RE.findall(t1) == ["g++ a.cpp"]
      and t1.startswith("<think>R</think>plan") and not c1 and [q[0] for q in sess.queue] == ["submit a.cpp"]
      and sess.completion == 50)
lc = sess.last_call
check(f"turn 1: last_call {lc}", lc["prompt_tokens"] == 100 and lc["completion_tokens"] == 50 and lc["tries"] == 1
      and lc["llm_s"] >= 0 and lc["cached_tokens"] is None and lc["finish_reason"] == "tool_calls")
check("the request carries the tool and the sampling settings", client.http.calls[0]["tools"] == [CH.BASH_TOOL]
      and client.http.calls[0]["temperature"] == 1.0 and client.http.calls[0]["top_k"] == 20)
hist += [{"role": "assistant", "content": "plan" + CH.block("g++ a.cpp"), "reasoning_content": "R"},
         {"role": "user", "content": '{"returncode": 0}'}]
t2, c2 = asyncio.run(sess.turn(hist, 1000))
check("turn 2: queued call, no model call, no tokens", CH.ACTION_RE.findall(t2) == ["submit a.cpp"]
      and len(client.http.calls) == 1 and sess.completion is None and sess.cont and not sess.fresh
      and sess.last_call == {"llm_s": 0.0, "queued_call": True})
hist += [{"role": "assistant", "content": CH.block("submit a.cpp"), "_cont": True},
         {"role": "user", "content": '{"returncode": 0, "output": "provisional 61"}'}]
api = CH.to_api(hist)
check("history: the queued call folds back into the turn that made it",
      [m["role"] for m in api] == ["system", "user", "assistant", "tool", "tool"]
      and [tc["function"]["arguments"] for tc in api[2]["tool_calls"]] == [json.dumps({"command": "g++ a.cpp"}),
                                                                         json.dumps({"command": "submit a.cpp"})]
      and [m["tool_call_id"] for m in api[3:]] == [tc["id"] for tc in api[2]["tool_calls"]]
      and all("_cont" not in m for m in api))
asyncio.run(sess.turn(hist, 1000))
check("turn 3: back to the model", len(client.http.calls) == 2 and sess.n_cont == 0 and not sess.cont)
api = CH.to_api([{"role": "user", "content": "[trimmed]"}, hist[-2], hist[-1]])
check("history: orphan _cont", api[1]["role"] == "assistant" and len(api[1]["tool_calls"]) == 1 and api[2]["role"] == "tool")

# the model call's numbers merged flat into the agent's turn records: server, tries, wall time, tokens, cached tokens
# (None without the cache report); the reasoning kept in the history
OUT0 = {"output": "x\n", "returncode": 0, "exception_info": ""}
for details, want in [({"cached_tokens": 96}, 96), (None, None)]:
    client.http = _Http(tool_reply(["ls", "pwd"], {"prompt_tokens": 120, "completion_tokens": 30,
                                                   "prompt_tokens_details": details}))
    ag = G.Agent(CH.ChatSession(client, SimpleNamespace(evaluation=True, base_url="http://x"), "agent1", G.MAX_LEN), 0,
                 "Read n.")
    c1 = asyncio.run(ag.think())
    ag.observe(c1, OUT0, 1, 0)
    c2 = asyncio.run(ag.think())
    ag.observe(c2, OUT0, 2, 0)
    u = ag.turns[0]
    check(f"last_call on the turn record, cached {want}", c1 == "ls" and u.get("cached_tokens") == want
          and u.get("prompt_tokens") == 120 and u.get("completion_tokens") == 30 and u.get("base_url") == "http://x"
          and u.get("tries") == 1 and "llm_s" in u and u["tokens"] == 30 and ag.generated == 30, u)
    check("queued call: flagged, no model call numbers, no tokens", c2 == "pwd" and ag.turns[1].get("queued_call")
          and ag.turns[1]["llm_s"] == 0.0 and "base_url" not in ag.turns[1] and ag.turns[1]["tokens"] == 0, ag.turns[1])
    check("the reasoning stays in the history, the queued call joins its turn",
          ag.chat[2].get("reasoning_content") == "R" and ag.chat[4].get("_cont") and "reasoning_content" not in ag.chat[4])

# a command with lines that start with ``` (a Markdown heredoc) comes back whole, and goes back to the model as sent
CMD = "cat > notes.md <<'EOF'\n# plan\n```\nint x;\n```\nEOF\necho ok"
RAW = '{"command":  ' + json.dumps(CMD) + '}'   # Flint's own spacing, kept byte for byte
client.http = _Http(tool_reply(None, {"prompt_tokens": 10, "completion_tokens": 5}, args=[RAW, RAW, "{bad"]))
ag = G.Agent(CH.ChatSession(client, SimpleNamespace(evaluation=True, base_url="http://x"), "agent1", G.MAX_LEN), 0,
             "Read n.")
cmds = []
for rnd in (1, 2, 3):
    c = asyncio.run(ag.think())
    cmds.append(c)
    ag.observe(c, OUT0, rnd, 0)
check("heredoc with ``` lines: whole command from think (first and queued call)", cmds[:2] == [CMD, CMD], cmds)
api = CH.to_api(ag.chat)
am = [m for m in api if m["role"] == "assistant"]
fns = [tc["function"] for tc in am[0]["tool_calls"]]
check("to_api: the arguments Flint sent; a bad one re-encoded", len(am) == 1 and am[0]["content"] == "plan"
      and [f["arguments"] for f in fns[:2]] == [RAW, RAW] and json.loads(fns[2]["arguments"]) == {"command": ""}, fns)

# routing, with a fake clock
clock = [1e6]
CH.time = SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0])   # llm_s reads monotonic
TURN_S = 60.0
posts = []
REPLY = {"choices": [{"message": {"content": "plan", "tool_calls": [{"function": {"name": "bash", "arguments":
         '{"command": "ls"}'}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


class _Timed:
    def post(self, url, json):
        t0 = clock[0]
        clock[0] += TURN_S
        posts.append((url.rsplit("/v1/", 1)[0], t0, clock[0]))
        return _Resp(REPLY)


U = [f"http://s{k}" for k in range(4)]
client = CH.Client(U, "flint")
client.http = _Timed()
HIST = [{"role": "system", "content": "s"}, {"role": "user", "content": "inst"}]


def reset():
    for u in U:
        client.active[u], client.last[u] = 0, clock[0]
    client.down.clear()
    posts.clear()


def episode_inp(url, k=0):
    client.active[url] += 1
    return SimpleNamespace(evaluation=True, base_url=url, moved_at=clock[0], moves=0,
                           sample=SimpleNamespace(index=k, group_index=0, metadata={"pid": k}))


def cs(x, name="agent1"):
    return CH.ChatSession(client, x, name, 262144)


reset()   # (a) 40 episodes, 10 per server; the 10 on s0 end; one turn each moves episodes until the gap is under 2
inps = [episode_inp(U[k % 4], k) for k in range(40)]
for x in [x for x in inps if x.base_url == U[0]]:
    client.active[x.base_url] -= 1
    inps.remove(x)
for x in inps:
    asyncio.run(cs(x).turn(HIST, 100))
cnt = {u: sum(x.base_url == u for x in inps) for u in U}
check("(a) balanced after one pass, counts match", max(client.active.values()) - min(client.active.values()) <= 1
      and cnt == client.active and sum(client.active.values()) == 30 and sum(x.moves for x in inps) == 7,
      (client.active, cnt))
client.no_route = True   # --no-route: an imbalance moves nothing
reset()
x = episode_inp(U[0])
for k in range(3):
    episode_inp(U[0], k + 1)
asyncio.run(cs(x).turn(HIST, 100))
check("--no-route: no move", x.moves == 0 and x.base_url == U[0] and client.active[U[0]] == 4, client.active)
client.no_route = False

reset()   # (b) one episode, 4 servers, 3 h of fake time
x = episode_inp(U[0])
sess = cs(x)
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
IDLE, DW = CH.IDLE_S, CH.DWELL_S
check(f"(b) no server idles over IDLE_S + 3*DWELL_S + a turn (max {max(gaps.values()):.0f} s), {x.moves} moves in "
      f"{n_turns} turns", max(gaps.values()) <= IDLE + 3 * DW + TURN_S + 30 and x.moves < n_turns / 4, gaps)
late = {}
for u in U:   # after the first hour the visits are spread: the bound is IDLE_S + DWELL_S + a turn
    ts = [(s, e) for v, s, e in posts if v == u and s > t_start + 3600]
    late[u] = max(ts[i + 1][0] - ts[i][1] for i in range(len(ts) - 1))
check(f"(b) steady state: max idle {max(late.values()):.0f} s", max(late.values()) <= IDLE + DW + TURN_S + 30, late)
check("(b) one episode counted once", client.active[x.base_url] == 1 and sum(client.active.values()) == 1, client.active)

reset()   # (d) an episode with two agents: both go to the same new server, and the counts move by one
for k in range(3):
    episode_inp(U[0], k)
x = episode_inp(U[0], 9)
s1, s2 = cs(x, "agent1"), cs(x, "agent2")


async def both():
    return await asyncio.gather(s1.turn(HIST, 100), s2.turn(HIST, 100))

asyncio.run(both())
check("(d) both agents move together", x.moves == 1 and [p[0] for p in posts] == [x.base_url] * 2
      and client.active[U[0]] == 3 and client.active[x.base_url] == 1, (posts, client.active))
done()
