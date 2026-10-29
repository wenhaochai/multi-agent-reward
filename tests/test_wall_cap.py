"""MA_LG_WALL_MIN: an episode still playing after that many minutes of wall time ends there, as at its token budget
(exit WallExceeded, scored the same way). A fake chat endpoint answers every turn with a command after 0.4 s; the cap
is 0.03 min (1.8 s) and the budget far away, so the episode ends after a few turns. Runs in miles-v0.1.3.sif."""
import asyncio
import json
import time

from aiohttp import web

from helpers import LABEL, MSG, check, done, env

env(MA_LG_N=1, MA_LG_TOKENS=100000, MA_AG_TOOL_STRICT=0, MA_LG_WALL_MIN=0.03)
from flint import miles_agent as MA  # noqa: E402
from flint import game as G  # noqa: E402

calls = []


async def chat(request):
    await request.json()
    calls.append(time.monotonic())
    await asyncio.sleep(0.4)
    k = len(calls)
    msg = {"role": "assistant", "content": f"Step {k}.\n", "reasoning_content": "r",
           "tool_calls": [{"id": f"c{k}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "true"})}}]}
    return web.json_response({"choices": [{"message": msg, "finish_reason": "tool_calls"}],
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
        t0 = time.monotonic()
        out = await MA.run(f"http://127.0.0.1:{port}/sessions/s1", MSG, {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
                           {"pid": "0", "label": LABEL, "messages": MSG})
        return out, time.monotonic() - t0
    finally:
        await runner.cleanup()


out, wall = asyncio.run(main())
check("wall cap: read from the environment", G.WALL_MIN == 0.03, G.WALL_MIN)
check("wall cap: the episode ends after a few turns, long before its budget", 2 <= len(calls) <= 8 and wall < 30, (len(calls), wall))
check("wall cap: it ends like the budget end (a reward, no failure)", "flint_failed" not in out and out.get("reward") == 0.0, out)
done()
