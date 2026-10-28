"""flint.miles_agent without miles' servers: (a) message_matches accepts the game's replay of a recorded assistant turn
(its own tool-call ids, the text around the call stripped) and nothing else; (b) one episode against a fake session
endpoint: every request extends the previous request and its reply under message_matches (one sample), except after
a turn cut at its token limit, which the game replays as "(cut off)" (a branch at the cut turn's parent); the
returned reward is the episode's S; drop_failed_groups takes a failed episode's group out of the loss; (c) log_rollout
moves the traces, probes the served log-probs against a fake router, writes the rollout's line and fails on a failed
sample check. Runs in miles-v0.1.3.sif (miles' strict matcher): tests/run_all.sh."""
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
check("episode: not failed", "flint_failed" not in out, out)

# drop_failed_groups: a group with a failed episode leaves the loss whole, the others stay
from types import SimpleNamespace  # noqa: E402

groups = [[SimpleNamespace(metadata={"flint_failed": "judge infra error"}, remove_sample=False),
           SimpleNamespace(metadata={}, remove_sample=False)],
          [SimpleNamespace(metadata={}, remove_sample=False), SimpleNamespace(metadata=None, remove_sample=False)]]
MA.drop_failed_groups(None, groups)
check("filter: a failed episode's group leaves the loss, others stay",
      [[x.remove_sample for x in g] for g in groups] == [[True, True], [False, False]])
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

# (c) log_rollout: two rollouts against a fake router whose served log-probs change between them
import os  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

shift = [0.0]
probes = []


class Router(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        probes.append(body)
        lps = [[None, body["input_ids"][0], None]] + [[-1.0 - shift[0], t, None] for t in body["input_ids"][1:]]
        data = json.dumps({"meta_info": {"input_token_logprobs": lps}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


srv = HTTPServer(("127.0.0.1", 0), Router)
threading.Thread(target=srv.serve_forever, daemon=True).start()
args = SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=srv.server_port)
out_dir = tempfile.mkdtemp()
os.environ["MA_RL_OUT"] = out_dir
os.environ["MA_AG_TRACE_DIR"] = os.path.join(out_dir, "traces_live")
smp = [SimpleNamespace(tokens=list(range(5000)), reward=0.5, metadata={"flint_session": "a", "fo_lg_submissions": 2}),
       SimpleNamespace(tokens=list(range(10)), reward=0.5, metadata={"flint_session": "a", "fo_lg_submissions": 2}),
       SimpleNamespace(tokens=list(range(20)), reward=0.1, metadata={"flint_session": "b", "fo_lg_submissions": 1})]
runs = []
real_run = subprocess.run
subprocess.run = lambda cmd, **kw: runs.append(cmd) or SimpleNamespace(returncode=0, stdout="INSPECT_OK\n", stderr="")
for rid in (0, 1):
    os.makedirs(os.environ["MA_AG_TRACE_DIR"])
    open(os.path.join(os.environ["MA_AG_TRACE_DIR"], "agent_1.jsonl"), "w").write("{}\n")
    MA.log_rollout(rid, args, smp, {}, 12.0)
    shift[0] = 0.01
lines = [json.loads(x) for x in open(os.path.join(out_dir, "steps.jsonl"))]
check("log: traces moved per rollout", all(os.path.exists(os.path.join(out_dir, f"r{i}", "traces", "agent_1.jsonl")) for i in (0, 1)))
check("log: the probe is rollout 0's first 4096 tokens, twice per rollout",
      len(probes) == 4 and all(b["input_ids"] == list(range(4096)) for b in probes), len(probes))
check("log: one line per rollout, episodes' S and submissions",
      [(x["episodes"], x["samples"], x["S"], x["submissions"]) for x in lines] == [(2, 3, [0.1, 0.5], [2, 1])] * 2, lines)
check("log: the probe change after the update", "probe_change_mean_abs" not in lines[0]
      and abs(lines[1]["probe_change_mean_abs"] - 0.01) < 1e-9 and lines[1]["probe_repeat_max_abs"] == 0.0, lines[1])
check("log: the sample check runs on the rollout's dump", runs[0][-2:] == [os.path.join(out_dir, "r0"), "rollout.pt"], runs[0])
subprocess.run = lambda cmd, **kw: SimpleNamespace(returncode=1, stdout="INSPECT_FAIL {}\n", stderr="")
try:
    MA.log_rollout(2, args, smp, {}, 12.0)
    raised = False
except RuntimeError:
    raised = True
subprocess.run = real_run
srv.shutdown()
check("log: a failed sample check ends the run", raised)
done()
