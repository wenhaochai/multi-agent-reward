"""Evaluate a served model in the long-horizon game (miles_team.fcs_agent_long) without miles: the game, the shell, the
submit command and the judge are the module's own; only the model call goes to an OpenAI-compatible chat endpoint
(SGLang with a reasoning parser). Built for Recursive Flint (user 2026-10-21: "我们先支持推理呗，先看看flint水平如何"),
which miles cannot train or serve (no Megatron or Transformers code). The same driver serves Qwen3.5-9B-FCS-SFT, so
both models play the same game.

The reasoning of every turn goes back into the history as `reasoning_content` (Flint's card asks for it; the earlier
shell runs lost points without it), and fcs_agent_long.trim_chat counts it against MA_LG_CTX_CHARS.

  python3 tools/chat_eval_long.py --base-url http://127.0.0.1:8000 --model flint --data data_v11/fcs11_val40_agent.jsonl \
      --out <dir> [--episodes 1] [--concurrency 40]
Game settings come from the environment as in miles: MA_LG_N, MA_LG_TOKENS, MA_LG_CTX_CHARS, MA_AG_TURN, MA_AG_MAX_LEN,
MA_AG_TRACE_DIR (defaults to <out>/traces). Output: <out>/results.jsonl, one line per finished episode (resumable).

Checkpoints (default on, --no-ckpt off): each episode saves its state to <out>/state/<pid>_e<episode>.{pkl,r<round>.tgz}
at the end of a round every MA_LG_CKPT_S seconds (300) and after every round with a submission
(fcs_agent_long.Checkpoint), and the next run resumes it there. A transport abort keeps the state; a crash or a
finished row removes it. The run stops early, pausing every episode at the end of its round (up to STOP_GRACE_S, then
cancelled at its last save), 10 min before SLURM_JOB_END_TIME, when a server is down (its GPUs idle; the requeue
starts fresh ones), or with --tail-exit-min M when every episode has started and a server has had none for M minutes.

Exit status (wq requeues a --partial job on 99): 0 when every episode has a clean score or has failed
--max-episode-tries times; 99 when some episode can still be retried or resumed, so the next run takes only those;
3 after a --tail-exit-min stop (for a job script that resubmits on fewer GPUs); 1 when every episode this run started
crashed (a bug: fail fast instead of looping) or the run wrote no row and saved no checkpoint. An exception in one
episode (a 400 reply, a bad JSON reply, a bug) becomes that episode's aborted row and no longer stops the others."""
import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from collections import Counter
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
ap.add_argument("--max-episode-tries", type=int, default=3,
                help="an episode whose rows are all aborted or judge-infra after this many runs is given up")
ap.add_argument("--no-ckpt", action="store_true", help="no episode checkpoints: a stopped run restarts its episodes")
ap.add_argument("--tail-exit-min", type=float, default=0,
                help="stop (exit 3) once every episode has started and a server has had none for this many minutes")
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
_last = {u: time.time() for u in URLS}   # the last request start or end on each server
DOWN: set = set()   # servers that failed the retries and a /health probe: no request goes to them again in this run
# An episode moves to another server (one cold prefill) when its server has 2 or more episodes more than the least
# busy one, or when a server without episodes has had no request for IDLE_S seconds (the cluster mails at 30 min of
# 0% GPU and cancels the whole host at 90) and the episode has stayed DWELL_S where it is. A lone episode then visits
# every server at least every IDLE_S + (servers - 1) * DWELL_S + one turn (36 min with 4 servers).
IDLE_S = float(os.environ.get("MA_CE_IDLE_S", 1200))
DWELL_S = 300
WATCH_S, STOP_MARGIN_S, STOP_GRACE_S = 60, 600, 420   # stop checks; stop this long before the end; pause wait
_stop: list = [None]   # why the run stops early: "time", "down" or "tail"
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


def _route(inp) -> str:
    """The server for an episode's next request; moves the episode (all its agents share `inp`) as above. No await
    between the check and the update, so two episodes never interleave a move."""
    cur = getattr(inp, "base_url", None)
    if cur not in _active:   # a session made outside main() (tests)
        return cur or URLS[0]
    live = [u for u in URLS if u not in DOWN]
    now, new = time.time(), None
    if live:
        low = min(live, key=_active.get)
        if cur in DOWN or _active[cur] - _active[low] >= 2:
            new = low
        else:
            idle = [u for u in live if u != cur and _active[u] == 0 and now - _last[u] > IDLE_S]
            if idle and now - getattr(inp, "moved_at", 0) > DWELL_S:
                new = min(idle, key=_last.get)
    if new is not None and new != cur:
        _active[cur] -= 1
        _active[new] += 1
        inp.base_url, inp.moved_at, inp.moves = new, now, getattr(inp, "moves", 0) + 1
        s = getattr(inp, "sample", None)
        print(f"[{(getattr(s, 'metadata', None) or {}).get('pid')} ep{getattr(s, 'group_index', '?')}] move {cur} -> "
              f"{new} (active {_active})", flush=True)
    _last[inp.base_url] = now
    return inp.base_url


async def _healthy(url: str) -> bool:
    try:
        async with _http.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=10)) as r:
            return r.status == 200
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return False


class _BadReply(Exception):
    """A 200 reply without a usable body: retried as a transport error."""


class _HttpAborted(Aborted):
    """The server did not answer after the retries; url is the server it was sent to."""

    def __init__(self, msg: str, url: str):
        super().__init__(msg)
        self.url = url


def _block(cmd: str) -> str:
    return f"\n\n```mswea_bash_command\n{cmd}\n```"


def _calls(m: dict) -> tuple[str, list]:
    """(text, tool-call functions) of an assistant message. A message from a tool call keeps the command (_cmds) and
    the function Flint sent (_fn): the block re-parsed would end at the first line of the command that starts with ```
    (a heredoc of Markdown), so the model would see a command it did not write. Arguments that are not a JSON object
    are re-encoded, so the server's chat template always parses them. Other messages (format errors, text mode): the
    blocks."""
    if "_cmds" in m:
        fn = m.get("_fn") or {}
        try:
            good = isinstance(json.loads(fn.get("arguments")), dict)
        except (TypeError, ValueError):
            good = False
        return m["content"].split("```mswea_bash_command")[0].strip(), [
            {"name": fn.get("name") or "bash", "arguments": fn["arguments"]} if good
            else {"name": "bash", "arguments": json.dumps({"command": c})} for c in m["_cmds"]]
    return A.ACTION_RE.sub("", m["content"]).strip(), [
        {"name": "bash", "arguments": json.dumps({"command": c.strip()})} for c in A.ACTION_RE.findall(m["content"])]


def to_api(messages: list) -> list:
    """The game's history (commands as mswea blocks in the assistant text, observations as user messages) as a
    tool-calling chat: each block becomes a bash tool call and the next user message its tool result. A message marked
    _cont holds a call that an earlier response made (ChatSession ran it a round later); it joins that response's
    tool_calls when only its tool results lie between them, so the model sees the turn it wrote."""
    out, pending, last = [], [], None   # last: the assistant message that only tool results have followed
    for m in messages:
        if m["role"] == "assistant":
            text, fns = _calls(m)
            if m.get("_cont") and last is not None and fns:
                pending = [f"call_{len(out)}_{j}" for j in range(len(fns))]
                last["tool_calls"] += [{"id": t, "type": "function", "function": f} for t, f in zip(pending, fns)]
                continue
            am = {"role": "assistant", "content": text}
            if m.get("reasoning_content"):
                am["reasoning_content"] = m["reasoning_content"]
            if fns:
                pending = [f"call_{len(out)}_{j}" for j in range(len(fns))]
                am["tool_calls"] = [{"id": t, "type": "function", "function": f} for t, f in zip(pending, fns)]
            out.append(am)
            last = am if fns else None
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
        self.queue, self.cont, self.n_cont = [], False, 0   # calls a response made beyond its first; cont: this turn
        # handed one out; n_cont: how many since the last model call
        self.last_call = None   # the last turn's model call: server, latency, tries, tokens, cached tokens (None: no call)
        self.last_tc = None   # the last turn's tool call (command, function as sent); None for a turn without one

    async def turn(self, messages: list, max_tokens: int) -> tuple[str, bool]:
        self.last_call = self.last_tc = None
        if self.queue:   # the next call of the last response: no model call, no tokens
            self.cont, self.fresh, self.n_cont = True, False, self.n_cont + 1
            self.last_tc = self.queue.pop(0)
            block = _block(self.last_tc[0])
            return (f"<think></think>{block}" if self.think else block), False
        # the messages since the last model call (its observation, then each handed-out call and its observation),
        # ~3 chars/token
        est = self.used + sum(len(m["content"]) for m in messages[-(1 + 2 * self.n_cont):]) // 3
        self.cont, self.n_cont = False, 0
        room = self.max_len - est
        if room <= 256:
            return "", True
        body = {"model": args.model, "messages": to_api(messages) if TOOLCALL else messages,
                "max_tokens": min(max_tokens, room), "temperature": args.temperature, "top_p": args.top_p,
                "top_k": args.top_k}
        if TOOLCALL:
            body["tools"] = [BASH_TOOL]
        out, attempt, tries, t0 = None, 0, 0, time.time()
        while True:
            url = _route(self.input)
            tries += 1
            try:
                async with _http.post(f"{url}/v1/chat/completions", json=body) as r:
                    if r.status == 400:
                        err = await r.text()
                        if "context" in err.lower() or "length" in err.lower():
                            return "", True
                        # e.g. a strict-grammar compile failure (the kit fails the request): this episode only
                        raise Aborted(f"{self.name}: 400: {err[:500]}")
                    r.raise_for_status()
                    try:
                        out = await r.json()
                    except ValueError as e:   # malformed JSON (a wrong content type is a ClientError already)
                        raise _BadReply(f"bad JSON: {e!r}")
                    if not isinstance(out, dict) or not out.get("choices"):
                        raise _BadReply(f"no choices: {str(out)[:300]}")
                break
            except (aiohttp.ClientError, asyncio.TimeoutError, _BadReply) as e:
                attempt += 1
                if attempt < 4:
                    await asyncio.sleep(10 * attempt)
                    continue
                # ~60 s of failures: a server that also fails /health is down, and the episode goes on elsewhere
                if url in _active and url not in DOWN and not await _healthy(url):
                    DOWN.add(url)
                    print(f"server {url} is down after {e!r}: no new episodes start in this run", flush=True)
                if url in DOWN and any(u not in DOWN for u in URLS):
                    attempt = 0
                    continue
                raise _HttpAborted(f"{self.name}: {e!r}", url)
            finally:
                if url in _last:   # a long decode does not look idle
                    _last[url] = time.time()
        ch, usage = out["choices"][0], out.get("usage") or {}
        msg = ch["message"]
        reasoning, content = msg.get("reasoning_content") or "", msg.get("content") or ""
        calls = []
        for tc in msg.get("tool_calls") or []:   # a bad call is still an action
            fn = tc.get("function") or {}
            try:
                cmd = json.loads(fn.get("arguments") or "{}").get("command") if fn.get("name") == "bash" else None
            except (json.JSONDecodeError, AttributeError, TypeError):
                cmd = None
            calls.append((cmd if isinstance(cmd, str) else "",
                          {"name": fn.get("name") or "bash", "arguments": fn.get("arguments") or "{}"}))
        cut = ch.get("finish_reason") == "length"
        if calls:   # the first call is this round's action; the rest wait for the next rounds (none from a cut turn)
            content += _block(calls[0][0])
            self.last_tc = calls[0]
            self.queue = [] if cut else calls[1:]
        n_out = int(usage.get("completion_tokens") or 0)
        # cached_tokens needs the servers' --enable-cache-report; None without it
        self.last_call = {"base_url": url, "latency_s": round(time.time() - t0, 2), "tries": tries,
                          "prompt_tokens": int(usage.get("prompt_tokens") or 0), "completion_tokens": n_out,
                          "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                          "finish_reason": ch.get("finish_reason")}
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
        self.sess.last_call = self.sess.last_tc = None   # a turn that ends before the model call (token budget)
        cmd = await super().think()
        if cmd is not None and TOOLCALL and self.sess.last_tc is not None:
            cmd = self.sess.last_tc[0].strip()   # the whole command: the game's block regex stops at a line of ```
        st = self.sess.last_call   # this turn's model call goes on its record; a queued call (no model call) has none
        if st is not None and cmd is not None:
            self._pending[1]["usage"] = st   # observe() spreads it into self.turns
        elif st is not None and self.turns:   # a format error or a cut turn appended its record already
            self.turns[-1]["usage"] = st
        if cmd is None:   # a format or length error added (assistant, error) now
            self._keep_reasoning()
        return cmd

    def observe(self, cmd, out, rnd, order):
        n = len(self.chat)
        super().observe(cmd, out, rnd, order)
        self._keep_reasoning()
        if self.sess.cont and len(self.chat) >= 4 and self.chat[-2].get("role") == "assistant":
            self.chat[-2] = {**self.chat[-2], "_cont": True}
        if TOOLCALL and self.sess.last_tc is not None and len(self.chat) == n + 2 \
                and self.chat[-2].get("role") == "assistant":   # to_api sends the call as Flint made it
            self.chat[-2] = {**self.chat[-2], "_cmds": [cmd], "_fn": self.sess.last_tc[1]}


L.Session = ChatSession
L._Agent = _Agent


def _clean(r: dict) -> bool:
    """A row whose score counts: S is set and no submission met a judge infrastructure error (its 0 is not the
    model's), so a rerun keeps it and skips the episode."""
    return r.get("S") is not None and not (r.get("info") or {}).get("fo_judge_infra_error")


def _read(path: Path) -> list:
    """results.jsonl's rows; a line cut short by a kill is skipped (its episode reruns)."""
    rows = []
    for line in (path.read_text().splitlines() if path.exists() else []):
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def outcome(want: list, recs: list, ran: list, max_tries: int, episodes: int) -> tuple[str, int]:
    """The summary line and the exit status. A key (pid, episode) of `want` is complete with a clean row, or given up
    after max_tries rows that are not clean (aborted, crashed or judge-infra); the others are retried by the next run
    (exit 99). The mean covers the clean rows only, and the line says how many keys it leaves out."""
    by: dict = {}
    for r in recs:
        by.setdefault((r["pid"], r["episode"]), []).append(r)
    ok, retry, given_up, infra = {}, [], [], 0
    for k in want:
        clean = [r for r in by.get(k, []) if _clean(r)]
        if clean:
            ok[k] = clean[-1]
            continue
        (given_up if len(by.get(k, [])) >= max_tries else retry).append(k)
        infra += any(r.get("S") is not None for r in by.get(k, []))

    def mean(xs):
        return 100 * sum(r["S"] for r in xs) / len(xs) if xs else float("nan")
    by_ep = " ".join(f"ep{e} {mean([r for (_, ep), r in ok.items() if ep == e]):.2f}" for e in range(episodes))
    line = (f"{'INCOMPLETE' if retry else 'DONE'} {len(ok)}/{len(want)} clean episodes (to rerun {len(retry)}, given "
            f"up after {max_tries} tries {len(given_up)}, judge-infra among them {infra}); mean final score "
            f"{mean(list(ok.values())):.2f} over clean episodes; {by_ep}")
    if ran and not any(_clean(r) for r in ran) and all(r.get("crash") for r in ran):
        return line + f"\nFAILED: all {len(ran)} episodes of this run crashed", 1
    return line, 99 if retry else 0


async def main():
    global _http
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    res_path = OUT / "results.jsonl"
    old = _read(res_path)
    done = {(r["pid"], r["episode"]) for r in old if _clean(r)}
    tries = Counter((r["pid"], r["episode"]) for r in old if not _clean(r))
    want = [(r["metadata"]["pid"], e) for e in range(args.episodes) for r in rows]
    todo = [(i, e, r) for e in range(args.episodes) for i, r in enumerate(rows)
            if (r["metadata"]["pid"], e) not in done and tries[(r["metadata"]["pid"], e)] < args.max_episode_tries]
    n_cap = sum(k not in done and tries[k] >= args.max_episode_tries for k in want)
    print(f"{len(rows)} problems x {args.episodes} episodes; {len(done)} done, {len(todo)} to run, {n_cap} given up "
          f"after {args.max_episode_tries} tries; model {args.model}; "
          f"N={L.N} tokens={L.TOKENS} ctx_chars={L.CTX_CHARS} turn={A.TURN} max_len={A.MAX_LEN}", flush=True)
    def ckpt_for(pid, e):
        return L.Checkpoint(OUT / "state" / f"{str(pid).replace('/', '_')}_e{e}")
    if not args.no_ckpt:
        for k in done:   # a kill between an episode's row and its clear left the state
            ckpt_for(*k).clear()
        n_res = sum(ckpt_for(r["metadata"]["pid"], e).exists() for _, e, r in todo)
        print(f"{n_res} of them resume from a checkpoint", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    _http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout))
    ran, cks, running, started, paused = [], [], {}, [0], [0]   # this run's rows and checkpoints; episodes in flight

    async def one(i, e, r):
        async with sem:
            started[0] += 1
            if DOWN or _stop[0]:   # a server died or the run stops: the requeue (exit 99) runs it on fresh servers
                return
            t0 = time.time()
            key = (r["metadata"]["pid"], e)
            ck = None if args.no_ckpt else ckpt_for(*key)
            if ck is not None:
                cks.append(ck)
            running[key] = ck
            url = min(URLS, key=_active.get)
            _active[url] += 1
            inp = SimpleNamespace(evaluation=True, sampling_params={}, base_url=url, moved_at=t0, moves=0,
                                  sample=SimpleNamespace(index=i, group_index=e, label=r["label"], metadata=r["metadata"]))
            rec = {"pid": r["metadata"]["pid"], "episode": e, "model": args.model}
            try:
                _, S, info = await L.episode(inp, r["metadata"]["messages"], r["label"],
                                             **({"ckpt": ck} if ck is not None else {}))
                rec.update(S=S, info=info)
            except L.Paused as ex:   # the run stops: the state is saved, no row; the rerun resumes it
                paused[0] += 1
                rec = None
                print(f"[{key[0]} ep{e}] {ex}", flush=True)
            except _HttpAborted as ex:
                rec.update(S=None, aborted=str(ex))
                if ex.url in DOWN:   # every server is down: not the episode's failure, no row; the rerun resumes it
                    rec = None
            except Aborted as ex:
                rec.update(S=None, aborted=str(ex))
            except Exception as ex:   # one episode's bug or bad reply; CancelledError (a BaseException) still stops all
                traceback.print_exc()
                rec.update(S=None, aborted=f"{type(ex).__name__}: {ex}"[:500], crash=True,
                           traceback=traceback.format_exc()[-4000:])
            finally:
                _active[inp.base_url] -= 1   # the server it ended on
                running.pop(key, None)
            if rec is None:
                return
            rec["moves"] = inp.moves
            rec["minutes"] = round(((ck.prior_s if ck is not None else 0.0) + time.time() - t0) / 60, 1)   # all runs
            with open(res_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            ran.append(rec)
            if ck is not None and (rec["S"] is not None or rec.get("crash")):   # a transport abort resumes
                ck.clear()
            print(f"[{rec['pid']} ep{e}] S={rec.get('S')} subs={(rec.get('info') or {}).get('fo_lg_submissions')} "
                  f"tokens={(rec.get('info') or {}).get('fo_lg_tokens')} {rec['minutes']} min moves={inp.moves}"
                  + (f" ABORTED {rec['aborted'][:200]}" if rec.get("S") is None else ""), flush=True)

    tasks = [asyncio.create_task(one(i, e, r)) for i, e, r in todo]

    async def watch():
        zero = {u: None for u in URLS}   # since when a live server has had no episode
        while True:
            await asyncio.sleep(WATCH_S)
            now, end = time.time(), os.environ.get("SLURM_JOB_END_TIME", "")
            for u in URLS:
                zero[u] = (zero[u] or now) if _active[u] == 0 and u not in DOWN else None
            if end.isdigit() and now > int(end) - STOP_MARGIN_S:
                _stop[0] = "time"
            elif DOWN:
                _stop[0] = "down"
            elif args.tail_exit_min > 0 and started[0] >= len(todo) and running and \
                    any(z is not None and now - z >= 60 * args.tail_exit_min for z in zero.values()):
                _stop[0] = "tail"
            if _stop[0]:
                break
        print(f"stopping ({_stop[0]}): {len(running)} episodes in flight pause at the end of their round", flush=True)
        for ck in running.values():
            if ck is not None:
                ck.stop = True
        t_end = time.time() + STOP_GRACE_S
        while any(ck is not None for ck in running.values()) and time.time() < t_end:
            await asyncio.sleep(min(WATCH_S, 5))
        for t in tasks:   # the rest lose the work since their last save
            t.cancel()

    watcher = asyncio.create_task(watch())
    try:
        if tasks:
            await asyncio.wait(tasks)
    finally:
        watcher.cancel()
        await _http.close()
    for t in tasks:
        if not t.cancelled() and t.exception() is not None:   # the driver's own bug (one() catches the episode's)
            raise t.exception()
    line, code = outcome(want, _read(res_path), ran, args.max_episode_tries, args.episodes)
    print(line, flush=True)
    if DOWN:
        print(f"servers down: {sorted(DOWN)}", flush=True)
    n_cut = sum(t.cancelled() for t in tasks)
    if _stop[0] or paused[0] or n_cut:
        print(f"stopped ({_stop[0]}): {paused[0]} episodes paused at a checkpoint, {n_cut} cancelled", flush=True)
    if code and _stop[0] == "tail":
        code = 3
    elif code == 99 and not ran and not sum(ck.saves for ck in cks):   # no row, no checkpoint: a rerun would repeat
        print("no progress in this run", flush=True)
        code = 1
    if code:
        print(f"exit {code}" + (": rerun the same command to resume" if code in (3, 99) else ""), flush=True)
        sys.exit(code)


asyncio.run(main())
