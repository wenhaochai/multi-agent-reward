"""flint.miles_agent without miles' servers: (a) message_matches accepts the game's replay of a recorded assistant turn
(its own tool-call ids, the text around the call stripped) and nothing else; (b) one episode against a fake session
endpoint: every request extends the previous request and its reply under message_matches (one sample), except after
a turn cut at its token limit, which the game replays as "(cut off)" (a branch at the cut turn's parent); the
returned reward is the episode's S. Runs in miles-v0.1.3.sif (miles' strict matcher): tests/run_all.sh."""
import asyncio
import json

from aiohttp import web

from helpers import LABEL, MSG, check, done, env, traces

env(MA_LG_N=1, MA_LG_TOKENS=90, MA_AG_TOOL_STRICT=0)
from flint import chat as CH  # noqa: E402
from flint import miles_agent as MA  # noqa: E402

M = MA.message_matches


def call(cmd, cid="srv_1"):
    return {"id": cid, "type": "function", "index": 0, "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}


rec = {"role": "assistant", "content": "Let me look.\n\n", "reasoning_content": "think", "tool_calls": [call("ls")]}
rep = {"role": "assistant", "content": "Let me look.", "reasoning_content": "think",
       "tool_calls": [{"id": "call_2_0", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}
check("match: own ids, stripped text, re-spaced arguments", M(rec, rep) is True)
check("match: strict-equal messages", M(rec, dict(rec)) is True)
check("no match: other command", M(rec, {**rep, "tool_calls": [call("pwd", "x")]}) is False)
check("no match: other text", M(rec, {**rep, "content": "Let me see."}) is False)
check("no match: other reasoning", M(rec, {**rep, "reasoning_content": "other"}) is False)
check("no match: the cut-off replay", M({**rec, "tool_calls": []}, {"role": "assistant", "content": "(cut off)",
                                                                   "reasoning_content": "think"}) is False)
check("no match: a user message with stripped text", M({"role": "user", "content": "a\n"}, {"role": "user", "content": "a"}) is False)
check("STRICT off", CH.BASH_TOOL["function"]["strict"] is False)

# (b) the fake endpoint: turn 1 a command, turn 2 a reply cut at its limit, turn 3 a reply without a tool call
# (format error), then commands until the budget (90 tokens, 20 per turn) ends the episode
seen, replies = [], []


async def chat(request):
    body = await request.json()
    seen.append(body)
    k = len(seen)
    msg = {"role": "assistant", "content": f"Step {k}.\n", "reasoning_content": f"r{k}", "tool_calls": [call(f"echo {k}", f"srv_{k}")]}
    fin = "tool_calls"
    if k == 2:
        msg, fin = {"role": "assistant", "content": None, "reasoning_content": "long", "tool_calls": None}, "length"
    elif k == 3:
        msg = {"role": "assistant", "content": "I will think more.\n", "reasoning_content": "r3", "tool_calls": None}
        fin = "stop"
    replies.append(msg)
    return web.json_response({"choices": [{"message": msg, "finish_reason": fin}],
                              "usage": {"prompt_tokens": 100, "completion_tokens": 20}})


async def main():
    app = web.Application()
    app.router.add_post("/sessions/s1/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        return await MA.run(f"http://127.0.0.1:{port}/sessions/s1", MSG,
                            {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 32768},
                            {"pid": "0", "label": LABEL, "messages": MSG})
    finally:
        await runner.cleanup()


out = asyncio.run(main())
check("episode: reward is S", out.get("reward") == 0.0 and out.get("fo_lg_submissions") == 0, out)
check("episode: no abort", "flint_aborted" not in out, out)
check("requests: sampling settings as given, tool not strict",
      all(b["temperature"] == 1.0 and b["top_p"] == 0.95 and b["top_k"] == 20 and not b["tools"][0]["function"]["strict"]
          for b in seen))


def extends(prev, reply, req):
    """req's messages start with prev's messages and then the reply, under message_matches."""
    m = req["messages"]
    return (len(m) > len(prev["messages"]) and all(M(a, b) for a, b in zip(prev["messages"], m))
            and M(reply, m[len(prev["messages"])]))


links = [extends(seen[i], replies[i], seen[i + 1]) for i in range(len(seen) - 1)]
check("requests: 5 model calls", len(seen) == 5, len(seen))
check("tree: the cut turn is not extended (a branch at its parent), every other turn is", links == [True, False, True, True], links)
check("tree: after the cut the request extends the cut turn's parent",
      all(M(a, b) for a, b in zip(seen[1]["messages"], seen[2]["messages"])), seen[2]["messages"][:4])
t = traces()
check("trace: one episode record", len(t) == 1 and t[0]["label"] == LABEL, len(t))
done()
