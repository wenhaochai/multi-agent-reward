"""tools/chat_eval_long.py in tool-calling mode, without a server: the patched prompt (mini.yaml rules, our workflow and
long-work tools), the history conversion to tool calls, and a tool-call response turned into an action block."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "tools/chat_eval_long.py").read_text()
assert src.rstrip().endswith("asyncio.run(main())")
sys.argv = ["x", "--base-url", "http://x", "--model", "flint", "--data", "x", "--out", "/tmp/_tc_test"]
g = {"__file__": str(ROOT / "tools/chat_eval_long.py"), "__name__": "chat_eval_long"}
exec(compile(src.rstrip()[: -len("asyncio.run(main())")], "chat_eval_long", "exec"), g)
A, L = g["A"], g["L"]
ok = True


def check(c, what):
    global ok
    print(("PASS " if c else "FAIL ") + what)
    ok &= bool(c)


pr = L.instance("Read n and print it.", "agent1")
check("mswea_bash_command" not in pr and "AT LEAST ONE bash tool call" in pr and "```bash" in pr, "prompt: tool-call rules")
check("`submit` or `submit <file>`" in pr and pr.index("## Long work") < pr.index("## Important Rules"), "prompt: long-work tools")
check("7. `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` also submits solution.cpp" in pr and "Read n and print it." in pr, "prompt: workflow and problem")
check(A.SYSTEM.startswith("You are a helpful assistant") and "format_example" not in A.SYSTEM, "system: mini.yaml")
obs = json.loads(A.observation({"output": "hi\n", "returncode": 0, "exception_info": ""}))
check(obs == {"returncode": 0, "output": "hi\n"}, "observation: JSON")
chat = [{"role": "system", "content": "s"}, {"role": "user", "content": "inst"},
        {"role": "assistant", "content": "plan" + g["_block"]("ls -la"), "reasoning_content": "r1"},
        {"role": "user", "content": '{"returncode": 0}'},
        {"role": "assistant", "content": "no call", "reasoning_content": "r2"},
        {"role": "user", "content": "Tool call error: ..."}]
api = g["to_api"](chat)
check(api[2]["content"] == "plan" and api[2]["tool_calls"][0]["function"]["arguments"] == json.dumps({"command": "ls -la"})
      and api[2]["reasoning_content"] == "r1", "history: block -> tool call, reasoning kept")
check(api[3]["role"] == "tool" and api[3]["tool_call_id"] == api[2]["tool_calls"][0]["id"], "history: observation -> tool msg")
check(api[4] == {"role": "assistant", "content": "no call", "reasoning_content": "r2"} and api[5]["role"] == "user",
      "history: a format error stays a user message")
m = A.ACTION_RE.findall("x" + g["_block"]("submit"))
check(m == ["submit"] and L.SUBMIT_RE.fullmatch(m[0].strip()), "response: tool call -> one action the game parses")
check(g["BASH_TOOL"]["function"]["strict"] is True, "tool: strict, so Flint's grammar holds the arguments")

# a response with two calls: the first runs now, the second a round later without a model call
import asyncio
from types import SimpleNamespace


class _Resp:
    def __init__(self, d):
        self.d, self.status = d, 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return self.d


calls = []


class _Http:
    def post(self, url, json):
        calls.append(json)
        tcs = [{"id": "x", "type": "function", "function": {"name": "bash", "arguments": '{"command": "%s"}' % c}}
               for c in ("g++ a.cpp", "submit a.cpp")]
        return _Resp({"choices": [{"message": {"reasoning_content": "R", "content": "plan", "tool_calls": tcs},
                                   "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 100, "completion_tokens": 50}})


g["_http"] = _Http()
sess = g["ChatSession"](SimpleNamespace(evaluation=True), "a0", True, max_len=262144)
hist = [{"role": "system", "content": "s"}, {"role": "user", "content": "inst"}]
t1, c1 = asyncio.run(sess.turn(hist, 1000))
check(A.ACTION_RE.findall(t1) == ["g++ a.cpp"] and t1.startswith("<think>R</think>plan") and not c1 and sess.queue == ["submit a.cpp"],
      "turn 1: first call is the action, second queued")
hist += [{"role": "assistant", "content": "plan" + g["_block"]("g++ a.cpp"), "reasoning_content": "R"},
         {"role": "user", "content": '{"returncode": 0}'}]
n_seg = len(sess.segments)
t2, c2 = asyncio.run(sess.turn(hist, 1000))
check(A.ACTION_RE.findall(t2) == ["submit a.cpp"] and len(calls) == 1 and len(sess.segments) == n_seg and sess.cont
      and not sess.fresh, "turn 2: queued call, no model call, no tokens")
hist += [{"role": "assistant", "content": g["_block"]("submit a.cpp"), "_cont": True},
         {"role": "user", "content": '{"returncode": 0, "output": "provisional 61"}'}]
api = g["to_api"](hist)
check([m["role"] for m in api] == ["system", "user", "assistant", "tool", "tool"]
      and [tc["function"]["arguments"] for tc in api[2]["tool_calls"]] == [json.dumps({"command": "g++ a.cpp"}),
                                                                         json.dumps({"command": "submit a.cpp"})]
      and [m["tool_call_id"] for m in api[3:]] == [tc["id"] for tc in api[2]["tool_calls"]]
      and all("_cont" not in m for m in api), "history: the queued call folds back into the turn that made it")
asyncio.run(sess.turn(hist, 1000))
check(len(calls) == 2 and sess.n_cont == 0 and not sess.cont, "turn 3: back to the model")
# a _cont message after a trimmed parent stands alone
api = g["to_api"]([{"role": "user", "content": "[trimmed]"}, hist[-2], hist[-1]])
check(api[1]["role"] == "assistant" and len(api[1]["tool_calls"]) == 1 and api[2]["role"] == "tool", "history: orphan _cont")

# per-turn usage on the agent's turn records: server, latency, tokens, cached tokens (None without the cache report)
class _Http2:
    def __init__(self, details):
        self.details = details

    def post(self, url, json):
        tcs = [{"id": "x", "type": "function", "function": {"name": "bash", "arguments": '{"command": "%s"}' % c}}
               for c in ("ls", "pwd")]
        return _Resp({"choices": [{"message": {"reasoning_content": "R", "content": "plan", "tool_calls": tcs},
                                   "finish_reason": "tool_calls"}],
                      "usage": {"prompt_tokens": 120, "completion_tokens": 30, "prompt_tokens_details": self.details}})


OUT0 = {"output": "x\n", "returncode": 0, "exception_info": ""}
for details, want in [({"cached_tokens": 96}, 96), (None, None)]:
    g["_http"] = _Http2(details)
    ag = g["_Agent"](SimpleNamespace(evaluation=True, base_url="http://x"), 0, "Read n.")
    c1 = asyncio.run(ag.think())
    ag.observe(c1, OUT0, 1, 0)
    c2 = asyncio.run(ag.think())
    ag.observe(c2, OUT0, 2, 0)
    u = ag.turns[0].get("usage") or {}
    check(c1 == "ls" and u.get("cached_tokens") == want and u.get("prompt_tokens") == 120 and u.get("completion_tokens") == 30
          and u.get("base_url") == "http://x" and u.get("tries") == 1 and "latency_s" in u, f"usage on the turn record, cached {want}")
    check(c2 == "pwd" and "usage" not in ag.turns[1], "queued call: no usage")
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
