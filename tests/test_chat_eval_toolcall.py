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
check("mswea_bash_command" not in pr and "EXACTLY ONE bash tool call" in pr and "```bash" in pr, "prompt: tool-call rules")
check("`submit` or `submit <file>`" in pr and pr.index("## Long work") < pr.index("## Important Rules"), "prompt: long-work tools")
check("7. When you cannot improve" in pr and "Read n and print it." in pr, "prompt: workflow and problem")
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
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
