"""flint.miles_agent without miles' servers: (a) message_matches accepts the game's replay of a recorded assistant turn
(its own tool-call ids, the text around the call stripped) and nothing else; (b) one episode against a fake session
endpoint: every request extends the previous request and its reply under message_matches (one sample), except after
a turn cut at its token limit, which the game replays as "(cut off)" (a branch at the cut turn's parent); the
returned reward is the episode's S; drop_failed_groups takes a failed episode's group out of the loss; (c) log_rollout
moves the traces, probes the served log-probs against a fake router, writes the rollout's line and fails on a failed
sample check. Runs in miles-v0.1.3.sif (miles' strict matcher): tests/run_all.sh."""
import os
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
# as miles hands it over (w583): each group holds one list of samples per episode
nested = [[[SimpleNamespace(metadata={}, remove_sample=False)],
           [SimpleNamespace(metadata={}, remove_sample=False), SimpleNamespace(metadata={"flint_failed": "aborted"}, remove_sample=False)]],
          [[SimpleNamespace(metadata={}, remove_sample=False)], [SimpleNamespace(metadata={}, remove_sample=False)]]]
MA.drop_failed_groups(None, nested)
check("filter: episodes as lists of samples", [[[x.remove_sample for x in e] for e in g] for g in nested]
      == [[[True], [True, True]], [[False], [False]]])
# (2026-11-02) a group with a sample over MA_RL_MAX_SAMPLE_TOKENS leaves the loss whole, as a failed one does
os.environ["MA_RL_MAX_SAMPLE_TOKENS"] = "10"
capped = [[[SimpleNamespace(metadata={}, tokens=[0] * 11, remove_sample=False)],
           [SimpleNamespace(metadata={}, tokens=[0] * 3, remove_sample=False)]],
          [[SimpleNamespace(metadata={}, tokens=[0] * 10, remove_sample=False)],
           [SimpleNamespace(metadata={}, tokens=[0] * 4, remove_sample=False)]]]
MA.drop_failed_groups(None, capped)
check("filter: a group with a sample over the cap leaves the loss, one at the cap stays",
      [[[x.remove_sample for x in e] for e in g] for g in capped] == [[[True], [True]], [[False], [False]]])
del os.environ["MA_RL_MAX_SAMPLE_TOKENS"]
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
probes, chats = [], []


refuse = [False]


class Router(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        code = 200
        if self.headers.get("X-SMG-Routing-Key") != "flint-probe":   # sgl-router's consistent hashing (w612)
            code, reply = 400, {"code": "missing_routing_key"}
        elif self.path == "/generate" and refuse[0]:
            code, reply = 400, {"error": "no input log-probs here"}
        elif self.path == "/generate":
            probes.append(body)
            lps = [[None, body["input_ids"][0], None]] + [[-1.0 - shift[0], t, None] for t in body["input_ids"][1:]]
            reply = {"meta_info": {"input_token_logprobs": lps}}
        else:
            chats.append(body)
            reply = {"choices": [{"logprobs": {"content": [{"top_logprobs": [
                {"token": "a", "logprob": -0.1 - shift[0]}, {"token": "b", "logprob": -2.5}]}]}}]}
        data = json.dumps(reply).encode()
        self.send_response(code)
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
os.environ["MA_AG_TURN"] = "90"   # the turn cap the hook counts hits of
open(os.path.join(out_dir, "data_p0.jsonl"), "w").write(json.dumps({"prompt": MSG}) + "\n")
os.environ["MA_AG_TRACE_DIR"] = os.path.join(out_dir, "traces_live")
smp = [SimpleNamespace(tokens=list(range(5000)), reward=0.5, metadata={"flint_session": "a", "fo_lg_submissions": 2}),
       SimpleNamespace(tokens=list(range(10)), reward=0.5, metadata={"flint_session": "a", "fo_lg_submissions": 2}),
       SimpleNamespace(tokens=list(range(20)), reward=0.1, metadata={"flint_session": "b", "fo_lg_submissions": 1})]
runs = []
real_run = subprocess.run
subprocess.run = lambda cmd, **kw: runs.append(cmd) or SimpleNamespace(returncode=0, stdout="INSPECT_OK\n", stderr="")
for rid in (0, 1):
    os.makedirs(os.environ["MA_AG_TRACE_DIR"])
    open(os.path.join(os.environ["MA_AG_TRACE_DIR"], "agent_1.jsonl"), "w").write(json.dumps(
        {"S": 0.5, "submissions": [{}, {}], "agents": [{"restarts": 1, "turns": [
            {"cut": False, "dropped": 0, "prompt_tokens": 10}, {"cut": True, "dropped": 0, "prompt_tokens": 30, "completion_tokens": 90},
            {"cut": False, "dropped": 4, "prompt_tokens": 20}]}]}) + "\n")
    MA.log_rollout(rid, args, smp, {}, 12.0)
    shift[0] = 0.01
lines = [json.loads(x) for x in open(os.path.join(out_dir, "steps.jsonl"))]
check("log: traces moved per rollout", all(os.path.exists(os.path.join(out_dir, f"r{i}", "traces", "agent_1.jsonl")) for i in (0, 1)))
check("log: the probe is rollout 0's first 4096 tokens, twice per rollout",
      len(probes) == 4 and all(b["input_ids"] == list(range(4096)) for b in probes), len(probes))
check("log: one line per rollout, episodes' S and submissions",
      [(x["episodes"], x["samples"], x["S"], x["submissions"]) for x in lines] == [(2, 3, [0.1, 0.5], [2, 1])] * 2, lines)
check("log: per-episode trace numbers", lines[0]["trace_episodes"] == [{"S": 0.5, "cut_turns": 1, "turn_cap_hits": 1, "turns": 3, "trims": 1,
      "submissions": 2, "restarts": 1, "exit": None, "max_prompt_tokens": 30}] and lines[0]["cut_turns"] == 1
      and lines[0]["wall_exceeded"] == 0, lines[0])
check("log: the probe change after the update", "probe_change_mean_abs" not in lines[0]
      and abs(lines[1]["probe_change_mean_abs"] - 0.01) < 1e-9 and lines[1]["probe_repeat_max_abs"] == 0.0, lines[1])
check("log: the sample check runs on the rollout's dump", runs[0][-2:] == [os.path.join(out_dir, "r0"), "rollout.pt"], runs[0])
# /generate refused (w606: 400): the probe falls back to the first token's top log-probs from the chat endpoint
refuse[0] = True
for rid in (2, 3):
    os.makedirs(os.environ["MA_AG_TRACE_DIR"])
    MA.log_rollout(rid, args, smp, {}, 12.0)
    shift[0] = 0.05
lines = [json.loads(x) for x in open(os.path.join(out_dir, "steps.jsonl"))]
check("log: /generate refused -> chat probe, the 400 body kept, the change measured",
      lines[3]["probe_kind"] == "chat" and "no input log-probs" in lines[3]["probe_generate_refused"]
      and abs(lines[3]["probe_change_max_abs"] - 0.04) < 1e-9 and len(chats) == 4 and chats[0]["messages"] == MSG, lines[3])
# startup_probe: the router from miles' log line, the prompt as Flint's chat template renders it
refuse[0] = False
probes.clear()
log = os.path.join(out_dir, "train.log")
open(log, "w").write(f"x router_manager.py:64 - Router ready at host='127.0.0.1' port={srv.server_port} external_host=None\n")
FLINT = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint"
want = {"MA_LG_N": 1, "MA_LG_TOKENS": 4000, "MA_LG_CTX_CHARS": 20000, "MA_AG_TURN": 16384, "MA_LG_WALL_MIN": 45}
json.dump(want, open(os.path.join(out_dir, "game_settings.json"), "w"))
os.makedirs(os.path.join(out_dir, "settings_seen"))
json.dump({**want, "MA_LG_TOKENS": 128000}, open(os.path.join(out_dir, "settings_seen", "1.json"), "w"))
check("startup probe: an agent's game settings that are not the job's stop the run",
      MA.startup_probe(out_dir, log, FLINT) == 1 and not probes)
json.dump(want, open(os.path.join(out_dir, "settings_seen", "1.json"), "w"))
rc = MA.startup_probe(out_dir, log, FLINT)
ids = json.load(open(os.path.join(out_dir, "probe_ids.json")))
check("startup probe: works, the templated prompt, written for log_rollout", rc == 0 and len(probes) == 2
      and probes[0]["input_ids"] == ids and len(ids) >= 3, (rc, len(ids)))
# _steady: a read batched with another request's prefill differs on the same weights (w922's rollout 17): two in a row must agree
def _reads(*vals):
    it, n = iter(vals), []
    def read():
        n.append(1)
        return {"kind": "generate", "lp": [next(it)]}
    return read, n
read, n = _reads(1.0, 1.0, 9.0)
check("steady: the first two reads agree, no more reads", [x["lp"] for x in MA._steady(read)] == [[1.0], [1.0]] and len(n) == 2)
read, n = _reads(1.5, 1.0, 1.0, 9.0)
check("steady: the odd first read is passed over", [x["lp"] for x in MA._steady(read)] == [[1.0], [1.0]] and len(n) == 3)
read, n = _reads(*[float(i) for i in range(20)])
r = MA._steady(read)
check("steady: reads that never agree stop at the limit and show it", len(n) == MA.PROBE_TRIES and r[0]["lp"] != r[1]["lp"], (len(n), r))
# the engines' probe through the log: an odd read passes, an engine on other weights stops the run, and reads that never agree show nothing either way
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402
import threading  # noqa: E402
engine = {}


class _Fake(BaseHTTPRequestHandler):
    calls = {"router": 0, "direct": 0}

    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"workers": [{"url": f"http://127.0.0.1:{self.server.server_port}/direct@0"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        who = "direct" if self.path.startswith("/direct") else "router"
        k = _Fake.calls[who]
        _Fake.calls[who] += 1
        v = engine[who](k)
        self._send({"meta_info": {"input_token_logprobs": [[0.0], [v], [v]]}})


fsrv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
threading.Thread(target=fsrv.serve_forever, daemon=True).start()
fargs = SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=fsrv.server_port)
engine["router"], engine["direct"] = (lambda k: 1.0), (lambda k: 1.5 if k == 0 else 1.0)
per, uns = MA._probe_engines(fargs, [1, 2, 3])
check("engines: an odd read beside other requests is passed over", list(per.values()) == [[1.0, 1.0]] and uns == [], (per, uns))
_Fake.calls.update(router=0, direct=0)
engine["direct"] = lambda k: float(k)
per, uns = MA._probe_engines(fargs, [1, 2, 3])
check("engines: reads that never agree are named", len(uns) == 1 and _Fake.calls["direct"] == MA.PROBE_TRIES, (uns, _Fake.calls))


def _log_engines(rid, router, direct):
    """log_rollout with the engines' check on the fake router: the error's text, or None; and the rollout's line."""
    _Fake.calls.update(router=0, direct=0)
    engine["router"], engine["direct"] = router, direct
    os.makedirs(os.environ["MA_AG_TRACE_DIR"])
    open(os.path.join(os.environ["MA_AG_TRACE_DIR"], "agent_1.jsonl"), "w").write(json.dumps(
        {"S": 0.5, "submissions": [{}, {}], "agents": [{"restarts": 1, "turns": [{"cut": False, "dropped": 0, "prompt_tokens": 10}]}]}) + "\n")
    os.environ["MA_ENGINE_REGISTRY"] = out_dir
    try:
        MA.log_rollout(rid, fargs, smp, {}, 12.0)
        return None, [json.loads(x) for x in open(os.path.join(out_dir, "steps.jsonl"))][-1]
    except RuntimeError as e:
        import shutil
        shutil.rmtree(os.environ["MA_AG_TRACE_DIR"], ignore_errors=True)
        return str(e), None
    finally:
        del os.environ["MA_ENGINE_REGISTRY"]


err, line = _log_engines(10, lambda k: 1.5 if k == 0 else 1.0, lambda k: 1.0)
check("log: the router's odd first read does not stop the run (w922's rollout 17)", err is None and line["probe_repeat_max_abs"] == 0.0 and line["probe_engines_mean_abs"] and "probe_unsteady" not in line, (err, line))
err, line = _log_engines(11, lambda k: 1.0, lambda k: 1.5)
check("log: an engine that steadily differs stops the run", "other weights" in (err or ""), err)
err, line = _log_engines(12, lambda k: 1.0, lambda k: 1.5 + k)
check("log: reads that never agree do not stop the run, and the line says so", err is None and line["probe_unsteady"] == [f"http://127.0.0.1:{fsrv.server_port}/direct"], (err, line))
fsrv.shutdown()
# run20's stop rules and checkpoint retention
rd = tempfile.mkdtemp()
def steplog(*rows):
    open(os.path.join(rd, "train.log"), "w").write("".join(
        f"x log_utils.py:554 - step {i}: {{'train/grad_norm': {g}, 'train/train_rollout_kl': {k}}}\n" for i, g, k in rows))
def rolls(*S):
    open(os.path.join(rd, "steps.jsonl"), "w").write("".join(json.dumps({"rollout_id": i, "S": s}) + "\n" for i, s in enumerate(S)))
steplog((0, 20.0, 0.0003), (1, 50.0, 0.0003)); rolls([0.4, 0], [0.5, 0])
check("rules: a healthy run goes on", MA._stop_rule(rd, 1) is None)
steplog((0, 20.0, 0.0003), (1, 50.0, 0.02))
check("rules: KL over 0.01 stops", "KL" in (MA._stop_rule(rd, 1) or ""))
steplog((0, 20.0, 0.0003), (1, 250.0, 0.0003), (2, 210.0, 0.0003))
check("rules: grad norm over 10x step 0's twice stops", "grad norm" in (MA._stop_rule(rd, 2) or ""))
steplog((0, 20.0, 0.0003), (1, 250.0, 0.0003), (2, 150.0, 0.0003))
check("rules: once is not enough", MA._stop_rule(rd, 2) is None)
steplog((0, 20.0, 0.0003)); rolls([0.4, 0], [0, 0], [0, 0], [0, 0])
check("rules: 3 rollouts all at 0 stop", "S = 0" in (MA._stop_rule(rd, 3) or ""))
check("rules: 2 are not enough", MA._stop_rule(rd, 2) is None)
open(os.path.join(rd, "steps.jsonl"), "w").write("".join(json.dumps({"rollout_id": i, "S": [0.4], "wall_exceeded": w}) + "\n"
                                                         for i, w in enumerate([2, 1, 2, 3])))
check("rules: the wall cap binding for more than 1 episode in 2 rollouts stops", "wall cap" in (MA._stop_rule(rd, 3) or "")
      and MA._stop_rule(rd, 2) is None and MA._stop_rule(rd, 1) is None)
ck = os.path.join(rd, "actor")
for it in (1, 3, 5, 7):
    os.makedirs(os.path.join(ck, f"iter_{it:07d}"))
os.makedirs(os.path.join(ck, "rollout"))
open(os.path.join(ck, "latest_checkpointed_iteration.txt"), "w").write("7")
MA._keep_newest_checkpoints(ck, 2)
check("retention: the newest 2 checkpoints stay, nothing else is touched",
      sorted(os.listdir(ck)) == ["iter_0000005", "iter_0000007", "latest_checkpointed_iteration.txt", "rollout"], os.listdir(ck))
import pickle  # noqa: E402
subprocess.run = lambda cmd, **kw: SimpleNamespace(returncode=1, stdout="INSPECT_FAIL {}\n", stderr="")
try:
    MA.log_rollout(4, args, smp, {}, 12.0)
    raised = False
except RuntimeError as e:
    raised = type(pickle.loads(pickle.dumps(e))) is RuntimeError
subprocess.run = real_run
srv.shutdown()
check("log: a failed sample check ends the run, with an exception Ray can pickle", raised)
# async, a resumed run: a group restored from the checkpoint has its traces in RUN/traces_live.stale_*; a new segment's PID may
# equal an old one, so the session decides which file is whose
with tempfile.TemporaryDirectory() as td:
    live, old1, old2, dest = (os.path.join(td, n) for n in ("traces_live", "traces_live.stale_100", "traces_live.stale_200", "r7/traces"))
    for x in (live, old1, old2):
        os.makedirs(x)
    def put(where, pid, session):
        open(os.path.join(where, f"agent_{pid}.jsonl"), "w").write(json.dumps({"base_url": f"http://e/sessions/{session}", "s": session}) + "\n")
    put(live, 1, "new1")          # a fresh episode of this batch
    before = os.path.join(td, "r7", "traces.before_123")
    os.makedirs(before)
    put(before, 6, "logged6")     # a restored episode whose rollout was logged once before the run stopped
    put(old2, 2, "restored2")     # a restored episode, moved away at the segment's start
    put(old1, 3, "restored3")     # ... at an earlier segment's start
    put(live, 4, "later4")        # another batch's episode with the PID a restored one had
    put(old2, 4, "restored4")
    put(live, 5, "later5")        # another batch's episode, not asked for
    samples = [SimpleNamespace(metadata={"flint_trace_pid": p, "flint_session": x}) for p, x in
               ((1, "new1"), (2, "restored2"), (3, "restored3"), (4, "restored4"), (6, "logged6"), (9, "gone9"))] + [SimpleNamespace(metadata=None)]
    MA._take_async_traces(samples, live, dest, td)
    got = {f: json.loads(open(os.path.join(dest, f)).readline())["s"] for f in sorted(os.listdir(dest))}
    check("async traces: fresh, restored from any stale dir, and the session decides on a PID clash",
          got == {"agent_1.jsonl": "new1", "agent_2.jsonl": "restored2", "agent_3.jsonl": "restored3", "agent_4.jsonl": "restored4", "agent_6.jsonl": "logged6"}, got)
    check("async traces: another batch's files stay live", sorted(os.listdir(live)) == ["agent_4.jsonl", "agent_5.jsonl"], os.listdir(live))
done()
