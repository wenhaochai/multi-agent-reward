"""Evaluate a served model in the long-horizon game (miles_team.fcs_agent_long) without miles: the game, the shell, the
submit command and the judge are the module's own; only the model call goes to an OpenAI-compatible chat endpoint
(SGLang with a reasoning parser). Built for Recursive Flint (user 2026-10-21: "我们先支持推理呗，先看看flint水平如何"),
which miles cannot train or serve (no Megatron or Transformers code). The same driver serves Qwen3.5-9B-FCS-SFT, so
both models play the same game.

The reasoning of every turn goes back into the history as `reasoning_content` (Flint's card asks for it; the earlier
shell runs lost points without it), and fcs_agent_long.trim_chat counts it against MA_LG_CTX_CHARS.

  python3 tools/chat_eval_long.py --base-url http://127.0.0.1:8000 --model flint --data data_v11/fcs11_val40_agent.jsonl \
      --out <dir> [--episodes 1] [--concurrency 40]
Game settings come from the environment as in miles: MA_LG_N, MA_LG_TOKENS, MA_LG_CTX_CHARS, MA_LG_CTX_LOW, MA_LG_HARNESS,
MA_AG_TURN, MA_AG_MAX_LEN, MA_AG_TRACE_DIR (defaults to <out>/traces). Output: <out>/results.jsonl, one line per finished episode (resumable)."""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True, help="one URL, or several comma-separated (one per server replica)")
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--episodes", type=int, default=1)
ap.add_argument("--concurrency", type=int, default=40)
ap.add_argument("--limit", type=int, default=0, help="first N problems only (smoke)")
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--top-k", type=int, default=20)
ap.add_argument("--request-timeout", type=float, default=3600)
ap.add_argument("--textbased", action="store_true", help="mini_textbased.yaml format instead of tool calls")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MA_AG_TRACE_DIR", str(OUT / "traces"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aiohttp  # noqa: E402

from miles_team import fcs_agent as A  # noqa: E402
from miles_team import fcs_agent_long as L  # noqa: E402
from miles_team.team_rollout import Aborted  # noqa: E402

_http: aiohttp.ClientSession | None = None
URLS = [u.strip().rstrip("/") for u in args.base_url.split(",") if u.strip()]
_active = {u: 0 for u in URLS}   # episodes running per server; a new episode takes the least busy one
TOOLCALL = not args.textbased

# ---- mini-swe-agent 2.4.6 mini.yaml (the tool-calling config; Flint's SWE-Bench evals ran mini-swe-agent) ------------
# Verbatim: the system template, the command execution rules and example response, the ```bash command examples, the
# JSON observation template, the tool-call error and length templates, and BASH_TOOL. Ours: the task and workflow
# (fcs_agent.INSTANCE's). The smoke test 15968124 showed why: in the text-based format Flint wrote its native
# tool-call syntax instead of the mswea_bash_command block and lost turns to format errors. We follow Flint's own
# format (user 2026-10-22, "Flint自己爱用什么格式我们就适配他的"): the kit's tool parser and grammar turn its native
# calls into OpenAI tool calls, "strict" makes the grammar hold the arguments to the schema, and a response may make
# several calls, as mini.yaml and Flint's template allow. The game runs one action per round, so ChatSession hands the
# extra calls out one per round without a model call, and to_api folds them back into the turn that made them.
BASH_TOOL = {"type": "function", "function": {"name": "bash", "strict": True, "description": "Execute a bash command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string",
    "description": "The bash command to execute"}}, "required": ["command"]}}}
_TC_RULES = """## Important Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load \
environment variables from files
- Each command may run for at most {timeout} seconds.{steps}

Example of a CORRECT response:
<example_response>
I need to understand the structure of the repository first. Let me check what files are in the current directory to \
get a better understanding of the codebase.

[Makes bash tool call with {{"command": "ls -la"}} as arguments]
</example_response>

## Useful command examples

### Create a new file:

```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

```bash
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```bash
anything
```
"""
_TC_FORMAT_ERROR = """Tool call error:

<error>
{error}
</error>

Here is general guidance on how to submit correct toolcalls:

Every response needs to use the 'bash' tool at least once to execute commands.

Call the bash tool with your command as the argument:
- Tool: bash
- Arguments: {{"command": "your_command_here"}}

If you want to end the task, please issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
without any other command."""
_TC_LENGTH_ERROR = ("Your previous response reached the output token limit (finish_reason=length) before you produced "
                    "a tool call, so it was cut off. Respond more concisely and finish with exactly one bash tool call. "
                    "If you need to think more, do so briefly.")


def _tc_observation(out: dict) -> str:
    """mini.yaml's observation template: JSON with the return code and the output (head and tail when long)."""
    o = out["output"]
    if len(o) < A.OUT_CHARS:
        d = {"returncode": out["returncode"], "output": o}
    else:
        d = {"returncode": out["returncode"], "output_head": o[:5000], "output_tail": o[-5000:],
             "elided_chars": len(o) - 10000, "warning": "Output too long."}
    if out.get("exception_info"):
        d["exception_info"] = out["exception_info"]
    return json.dumps(d, indent=2, ensure_ascii=False)


if TOOLCALL:
    A.INSTANCE = A.INSTANCE[: A.INSTANCE.index("## Important Rules")] + _TC_RULES
    A.SYSTEM = "You are a helpful assistant that can interact with a computer.\n"
    A.FORMAT_ERROR, A.LENGTH_ERROR, A.observation = _TC_FORMAT_ERROR, _TC_LENGTH_ERROR, _tc_observation
    L._FLAT = A.INSTANCE.replace("\\\n", "")
    assert L._LONE_ENV in L._FLAT and L._LONE_SUBMIT in L._FLAT
    assert L._FORMAT_ERROR_ENDS[1] in _TC_FORMAT_ERROR   # v2 rewrites this last line (fcs_agent_long.think)


def _block(cmd: str) -> str:
    return f"\n\n```mswea_bash_command\n{cmd}\n```"


def to_api(messages: list) -> list:
    """The game's history (commands as mswea blocks in the assistant text, observations as user messages) as a
    tool-calling chat: each block becomes a bash tool call and the next user message its tool result. A message marked
    _cont holds a call that an earlier response made (ChatSession ran it a round later); it joins that response's
    tool_calls when only its tool results lie between them, so the model sees the turn it wrote."""
    out, pending, last = [], [], None   # last: the assistant message that only tool results have followed
    for m in messages:
        if m["role"] == "assistant":
            cmds = [c.strip() for c in A.ACTION_RE.findall(m["content"])]
            if m.get("_cont") and last is not None and cmds:
                pending = [f"call_{len(out)}_{j}" for j in range(len(cmds))]
                last["tool_calls"] += [{"id": t, "type": "function", "function": {
                    "name": "bash", "arguments": json.dumps({"command": c})}} for t, c in zip(pending, cmds)]
                continue
            am = {"role": "assistant", "content": A.ACTION_RE.sub("", m["content"]).strip()}
            if m.get("reasoning_content"):
                am["reasoning_content"] = m["reasoning_content"]
            if cmds:
                pending = [f"call_{len(out)}_{j}" for j in range(len(cmds))]
                am["tool_calls"] = [{"id": t, "type": "function", "function": {
                    "name": "bash", "arguments": json.dumps({"command": c})}} for t, c in zip(pending, cmds)]
            out.append(am)
            last = am if cmds else None
            continue
        if m["role"] == "user" and pending:
            out.append({"role": "tool", "tool_call_id": pending[0], "content": m["content"]})
            out += [{"role": "tool", "tool_call_id": t, "content": "Not executed: one bash tool call per response."}
                    for t in pending[1:]]
            pending = []
        else:
            out.append(m)
            last = None
    return out


class ChatSession:
    """team_rollout.Session's interface over the chat API: turn(messages, max_tokens) -> (text, cut). The text is
    '<think>reasoning</think>answer' so fcs_agent's answer() splits it as for the in-process engines; the reasoning
    of the last turn waits in last_reasoning for _Agent to store in the history."""

    def __init__(self, input, name: str, think: bool, max_len: int | None = None):
        self.input, self.name, self.think = input, name, think
        self.max_len = max_len or A.MAX_LEN
        self.segments, self.last_messages = [], None
        self.last_reasoning, self.fresh, self.used = "", False, 0   # used: prompt + completion tokens of the last call
        self.last_call = {}   # the last turn's numbers for the trace (see turn)
        self.queue, self.cont, self.n_cont = [], False, 0   # calls a response made beyond its first; cont: this turn
        # handed one out; n_cont: how many since the last model call

    async def turn(self, messages: list, max_tokens: int) -> tuple[str, bool]:
        """last_call: this turn's numbers for the trace (fcs_agent_long merges them into the turn record)."""
        self.last_call = {"llm_s": 0.0}
        if self.queue:   # the next call of the last response: no model call, no tokens
            self.last_call["queued_call"] = True
            self.cont, self.fresh, self.n_cont = True, False, self.n_cont + 1
            block = _block(self.queue.pop(0))
            return (f"<think></think>{block}" if self.think else block), False
        # the messages since the last model call (its observation, then each handed-out call and its observation),
        # ~3 chars/token
        est = self.used + sum(len(m["content"]) for m in messages[-(1 + 2 * self.n_cont):]) // 3
        self.cont, self.n_cont = False, 0
        room = self.max_len - est
        if room <= 256:
            self.last_call["no_room"] = True
            return "", True
        body = {"model": args.model, "messages": to_api(messages) if TOOLCALL else messages,
                "max_tokens": min(max_tokens, room), "temperature": args.temperature, "top_p": args.top_p,
                "top_k": args.top_k}
        if TOOLCALL:
            body["tools"] = [BASH_TOOL]
        out, t_req, t0 = None, round(time.time(), 1), time.monotonic()
        for attempt in range(4):
            try:
                async with _http.post(f"{getattr(self.input, 'base_url', URLS[0])}/v1/chat/completions", json=body) as r:
                    if r.status == 400:
                        err = await r.text()
                        if "context" in err.lower() or "length" in err.lower():
                            self.last_call.update(llm_s=round(time.monotonic() - t0, 3), context_400=True)
                            return "", True
                        raise RuntimeError(f"400: {err[:500]}")
                    r.raise_for_status()
                    out = await r.json()
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == 3:
                    raise Aborted(f"{self.name}: {e!r}")
                await asyncio.sleep(10 * (attempt + 1))
        ch, usage = out["choices"][0], out.get("usage") or {}
        # cached_tokens needs SGLang's --enable-cache-report; None means not reported, not zero
        self.last_call = {"t_req_start": t_req, "llm_s": round(time.monotonic() - t0, 3), "attempts": attempt + 1,
                          "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                          "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                          "finish_reason": ch.get("finish_reason")}
        msg = ch["message"]
        reasoning, content = msg.get("reasoning_content") or "", msg.get("content") or ""
        cmds = []
        for tc in msg.get("tool_calls") or []:   # a bad call is still an action
            fn = tc.get("function") or {}
            try:
                cmd = json.loads(fn.get("arguments") or "{}").get("command") if fn.get("name") == "bash" else None
            except (json.JSONDecodeError, AttributeError):
                cmd = None
            cmds.append(cmd if isinstance(cmd, str) else "")
        cut = ch.get("finish_reason") == "length"
        if cmds:   # the first call is this round's action; the rest wait for the next rounds (none from a cut turn)
            content += _block(cmds[0])
            self.queue = [] if cut else cmds[1:]
        n_out = int(usage.get("completion_tokens") or 0)
        self.used = int(usage.get("prompt_tokens") or 0) + n_out
        self.segments.append(SimpleNamespace(loss_mask=[1] * n_out))
        self.last_reasoning, self.fresh = reasoning, True
        if self.think:
            text = f"<think>{reasoning}</think>{content}" if (content or not cut) else f"<think>{reasoning}"
        else:
            text = content
        return text, cut


class _Agent(L._Agent):
    """fcs_agent_long's agent; the assistant message a turn adds keeps that turn's reasoning."""

    def _keep_reasoning(self):
        m = self.chat[-2] if len(self.chat) >= 4 else None
        if self.sess.fresh and m is not None and m.get("role") == "assistant" and "reasoning_content" not in m:
            self.chat[-2] = {**m, "reasoning_content": self.sess.last_reasoning}
            self.sess.fresh = False

    async def think(self):
        cmd = await super().think()
        if cmd is None:   # a format or length error added (assistant, error) now
            self._keep_reasoning()
        return cmd

    def observe(self, cmd, out, rnd, order):
        super().observe(cmd, out, rnd, order)
        self._keep_reasoning()
        if self.sess.cont and len(self.chat) >= 4 and self.chat[-2].get("role") == "assistant":
            self.chat[-2] = {**self.chat[-2], "_cont": True}


L.Session = ChatSession
L._Agent = _Agent


async def main():
    global _http
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    res_path = OUT / "results.jsonl"
    done = set()
    if res_path.exists():
        done = {(r["pid"], r["episode"]) for r in map(json.loads, res_path.read_text().splitlines()) if r.get("S") is not None}
    todo = [(i, e, r) for e in range(args.episodes) for i, r in enumerate(rows) if (r["metadata"]["pid"], e) not in done]
    print(f"{len(rows)} problems x {args.episodes} episodes; {len(done)} done, {len(todo)} to run; model {args.model}; "
          f"N={L.N} tokens={L.TOKENS} ctx_chars={L.CTX_CHARS} ctx_low={L.CTX_LOW} turn={A.TURN} max_len={A.MAX_LEN} "
          f"harness={L.harness_tag()}", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    _http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout))

    async def one(i, e, r):
        async with sem:
            t0 = time.time()
            url = min(URLS, key=_active.get)
            _active[url] += 1
            inp = SimpleNamespace(evaluation=True, sampling_params={}, base_url=url,
                                  sample=SimpleNamespace(index=i, group_index=e, label=r["label"], metadata=r["metadata"]))
            rec = {"pid": r["metadata"]["pid"], "episode": e, "model": args.model}
            try:
                _, S, info = await L.episode(inp, r["metadata"]["messages"], r["label"])
                rec.update(S=S, info=info)
            except Aborted as ex:
                rec.update(S=None, aborted=str(ex))
            finally:
                _active[url] -= 1
            rec["minutes"] = round((time.time() - t0) / 60, 1)
            with open(res_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{rec['pid']} ep{e}] S={rec.get('S')} subs={(rec.get('info') or {}).get('fo_lg_submissions')} "
                  f"tokens={(rec.get('info') or {}).get('fo_lg_tokens')} {rec['minutes']} min", flush=True)

    try:
        await asyncio.gather(*(one(i, e, r) for i, e, r in todo))
    finally:
        await _http.close()
    recs = [json.loads(x) for x in res_path.read_text().splitlines()]
    ok = [r for r in recs if r.get("S") is not None]
    if ok:
        print(f"DONE {len(ok)} episodes, mean final score {100 * sum(r['S'] for r in ok) / len(ok):.2f}", flush=True)
        # a problem with one test case scores the case the agent saw (fcs_agent_long.split_cases): the held-out mean
        # leaves those out (runs before fo_lg_shared_case: train67 pids 43, 83, 137)
        ho = [r for r in ok if not (r.get("info") or {}).get("fo_lg_shared_case")]
        if ho:
            print(f"HELD-OUT {len(ho)} episodes ({len(ok) - len(ho)} with a single shared case left out), mean final "
                  f"score {100 * sum(r['S'] for r in ho) / len(ho):.2f}", flush=True)


asyncio.run(main())
