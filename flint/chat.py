"""The model side of the game: an OpenAI-compatible chat endpoint (SGLang serving Recursive Flint with its reasoning
parser and tool parser), one or more server replicas.

ChatSession is an agent's conversation with the model. turn(messages, max_tokens) -> (text, cut): the text is
'<think>reasoning</think>answer', with the turn's tool call written into the answer as a ```mswea_bash_command block
(ACTION_RE); the reasoning waits in last_reasoning for game.Agent to store in the history. Flint answers with
tool calls (user 2026-10-22: "Flint自己爱用什么格式我们就适配他的"): the kit's tool parser and grammar turn its native calls
into OpenAI tool calls, "strict" holds the arguments to the schema, and a response may make several calls, as
mini-swe-agent's mini.yaml and Flint's template allow. The game runs one action per round, so ChatSession hands the
extra calls out one per round without a model call, and to_api folds them back into the turn that made them.

Client routes the episodes over the servers: a new episode takes the least busy server, and an episode moves to
another server (one cold prefill) when its server has 2 or more episodes more than the least busy one, or when a
server without episodes has had no request for IDLE_S seconds (the cluster mails at 30 min of 0% GPU and cancels the
whole host at 90) and the episode has stayed DWELL_S where it is. A lone episode then visits every server at least
every IDLE_S + (servers - 1) * DWELL_S + one turn (36 min with 4 servers). A server that fails the retries and a health
probe is down for the rest of the run."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

import aiohttp

from . import Aborted

IDLE_S = float(os.environ.get("MA_CE_IDLE_S", 1200))
DWELL_S = 300
# a turn's command in the game's history: ChatSession writes the tool call into the reply as this block
ACTION_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
# "strict" holds the arguments to the schema through the kit's grammar (the eval game). miles records the sampler's
# candidates for score centering only from unconstrained sampling, so its rollouts set MA_AG_TOOL_STRICT=0.
STRICT = os.environ.get("MA_AG_TOOL_STRICT", "1") != "0"
RETRY_S = float(os.environ.get("MA_AG_RETRY_S", "60"))   # how long a failing request is retried (ChatSession)
BASH_TOOL = {"type": "function", "function": {"name": "bash", "strict": STRICT, "description": "Execute a bash command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string",
    "description": "The bash command to execute"}}, "required": ["command"]}}}


class BadReply(Exception):
    """A 200 reply without a usable body: retried as a transport error."""


class HttpAborted(Aborted):
    """The server did not answer after the retries; url is the server it was sent to."""

    def __init__(self, msg: str, url: str):
        super().__init__(msg)
        self.url = url


def block(cmd: str) -> str:
    return f"\n\n```mswea_bash_command\n{cmd}\n```"


def _calls(m: dict) -> tuple[str, list]:
    """(text, tool-call functions) of an assistant message. A message from a tool call keeps the command (_cmds) and
    the function Flint sent (_fn): the block re-parsed would end at the first line of the command that starts with ```
    (a heredoc of Markdown), so the model would see a command it did not write. Arguments that are not a JSON object
    are re-encoded, so the server's chat template always parses them. Other messages (format errors): the blocks."""
    if "_cmds" in m:
        fn = m.get("_fn") or {}
        try:
            good = isinstance(json.loads(fn.get("arguments")), dict)
        except (TypeError, ValueError):
            good = False
        return m["content"].split("```mswea_bash_command")[0].strip(), [
            {"name": fn.get("name") or "bash", "arguments": fn["arguments"]} if good
            else {"name": "bash", "arguments": json.dumps({"command": c})} for c in m["_cmds"]]
    return ACTION_RE.sub("", m["content"]).strip(), [
        {"name": "bash", "arguments": json.dumps({"command": c.strip()})} for c in ACTION_RE.findall(m["content"])]


def to_api(messages: list) -> list:
    """The game's history (commands as blocks in the assistant text, observations as user messages) as a tool-calling
    chat: each block becomes a bash tool call and the next user message its tool result. A message marked _cont holds a
    call that an earlier response made (ChatSession ran it a round later); it joins that response's tool_calls when
    only its tool results lie between them, so the model sees the turn it wrote."""
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


class Client:
    """The servers, the HTTP session and the sampling settings of a run, and the routing of its episodes."""

    def __init__(self, urls: list, model: str, temperature: float = 1.0, top_p: float = 0.95, top_k: int = 20,
                 route: bool = True, request_timeout: float = 3600):
        self.urls = urls
        self.model, self.temperature, self.top_p, self.top_k = model, temperature, top_p, top_k
        self.no_route, self.request_timeout = not route, request_timeout
        self.active = {u: 0 for u in urls}   # episodes running per server; a new episode takes the least busy one
        self.last = {u: time.time() for u in urls}   # the last request start or end on each server
        self.down: set = set()   # servers that failed the retries and a health probe
        self.http: aiohttp.ClientSession | None = None

    async def open(self) -> None:
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.request_timeout))

    async def close(self) -> None:
        await self.http.close()

    def route(self, inp) -> str:
        """The server for an episode's next request; moves the episode (all its agents share `inp`) as in the module
        docstring. No await between the check and the update, so two episodes never interleave a move."""
        cur = getattr(inp, "base_url", None)
        if cur not in self.active:   # a session made outside the driver (tests)
            return cur or self.urls[0]
        live = [u for u in self.urls if u not in self.down]
        now, new = time.time(), None
        if live:
            low = min(live, key=self.active.get)
            if cur in self.down or (not self.no_route and self.active[cur] - self.active[low] >= 2):
                new = low
            elif not self.no_route:
                idle = [u for u in live if u != cur and self.active[u] == 0 and now - self.last[u] > IDLE_S]
                if idle and now - getattr(inp, "moved_at", 0) > DWELL_S:
                    new = min(idle, key=self.last.get)
        if new is not None and new != cur:
            self.active[cur] -= 1
            self.active[new] += 1
            inp.base_url, inp.moved_at, inp.moves = new, now, getattr(inp, "moves", 0) + 1
            s = getattr(inp, "sample", None)
            print(f"[{(getattr(s, 'metadata', None) or {}).get('pid')} ep{getattr(s, 'group_index', '?')}] move {cur} -> "
                  f"{new} (active {self.active})", flush=True)
        self.last[inp.base_url] = now
        return inp.base_url

    async def healthy(self, url: str) -> bool:
        """SGLang's /health_generate runs a one-token generation, so a hung engine fails it; its plain /health answers
        200 from the HTTP server without touching the engine."""
        try:
            async with self.http.get(f"{url}/health_generate", timeout=aiohttp.ClientTimeout(total=30)) as r:
                return r.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False


class ChatSession:
    """An agent's conversation with the model (see the module docstring)."""

    def __init__(self, client: Client, input, name: str, max_len: int):
        self.client, self.input, self.name, self.max_len = client, input, name, max_len
        self.last_reasoning, self.fresh, self.used = "", False, 0   # used: prompt + completion tokens of the last call
        self.queue, self.cont, self.n_cont = [], False, 0   # calls a response made beyond its first; cont: this turn
        # handed one out; n_cont: how many since the last model call
        self.last_call = {}   # the last turn's numbers for the trace (see turn)
        self.last_tc = None   # the last turn's tool call (command, function as sent); None for a turn without one
        self.completion = None   # the last turn's completion tokens; None for a turn without a model call

    def forget(self) -> None:
        """The agent's conversation starts over: the calls and the size of the old one go."""
        self.queue, self.cont, self.n_cont, self.used, self.fresh = [], False, 0, 0, False

    async def turn(self, messages: list, max_tokens: int) -> tuple[str, bool]:
        """last_call: this turn's numbers for the trace (game.Agent merges them flat into the turn record): the model
        call's start, wall seconds (llm_s), server, requests sent (tries, retries and moves included) and token counts;
        or a flag for a turn without a reply (queued_call, no_room, context_400)."""
        c = self.client
        self.last_call, self.last_tc, self.completion = {"llm_s": 0.0}, None, None
        if self.queue:   # the next call of the last response: no model call, no tokens
            self.last_call["queued_call"] = True
            self.cont, self.fresh, self.n_cont = True, False, self.n_cont + 1
            self.last_tc = self.queue.pop(0)
            return f"<think></think>{block(self.last_tc[0])}", False
        # the messages since the last model call (its observation, then each handed-out call and its observation),
        # ~3 chars/token
        est = self.used + sum(len(m["content"]) for m in messages[-(1 + 2 * self.n_cont):]) // 3
        self.cont, self.n_cont = False, 0
        room = self.max_len - est
        if room <= 256:
            self.last_call["no_room"] = True
            return "", True
        body = {"model": c.model, "messages": to_api(messages), "max_tokens": min(max_tokens, room),
                "temperature": c.temperature, "top_p": c.top_p, "top_k": c.top_k, "tools": [BASH_TOOL]}
        out, attempt, tries, slept, t_req, t0 = None, 0, 0, 0, round(time.time(), 1), time.monotonic()
        while True:
            url = c.route(self.input)
            tries += 1
            try:
                async with c.http.post(f"{url}/v1/chat/completions", json=body) as r:
                    if r.status == 400:
                        err = await r.text()
                        if "context" in err.lower() or "length" in err.lower():
                            self.last_call.update(llm_s=round(time.monotonic() - t0, 3), base_url=url, tries=tries,
                                                  context_400=True)
                            return "", True
                        # e.g. a strict-grammar compile failure (the kit fails the request): this episode only
                        raise Aborted(f"{self.name}: 400: {err[:500]}")
                    r.raise_for_status()
                    try:
                        out = await r.json()
                    except ValueError as e:   # malformed JSON (a wrong content type is a ClientError already)
                        raise BadReply(f"bad JSON: {e!r}")
                    if not isinstance(out, dict) or not out.get("choices"):
                        raise BadReply(f"no choices: {str(out)[:300]}")
                break
            except (aiohttp.ClientError, asyncio.TimeoutError, BadReply) as e:
                attempt += 1
                # retries 10, 20, 30, 30, ... s apart for MA_AG_RETRY_S in all (default 60: three retries, as before;
                # elastic RL, rl/rl_inner.sh: 600, so a turn whose engine left reaches another one once miles has
                # withdrawn it, flint/engine_registry.py)
                if slept < RETRY_S:
                    await asyncio.sleep(min(10 * attempt, 30))
                    slept += min(10 * attempt, 30)
                    continue
                # RETRY_S of failures: a server that also fails /health_generate is down, and the episode goes on elsewhere
                if url in c.active and url not in c.down and not await c.healthy(url):
                    c.down.add(url)
                    print(f"server {url} is down after {e!r}: no new episodes start in this run", flush=True)
                if url in c.down and any(u not in c.down for u in c.urls):
                    attempt = slept = 0
                    continue
                raise HttpAborted(f"{self.name}: {e!r}", url)
            finally:
                if url in c.last:   # a long decode does not look idle
                    c.last[url] = time.time()
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
            content += block(calls[0][0])
            self.last_tc = calls[0]
            self.queue = [] if cut else calls[1:]
        n_out = int(usage.get("completion_tokens") or 0)
        # cached_tokens needs SGLang's --enable-cache-report; None means not reported, not zero
        self.last_call = {"t_req_start": t_req, "llm_s": round(time.monotonic() - t0, 3), "base_url": url,
                          "tries": tries, "prompt_tokens": usage.get("prompt_tokens"),
                          "completion_tokens": usage.get("completion_tokens"),
                          "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                          "finish_reason": ch.get("finish_reason")}
        self.used = int(usage.get("prompt_tokens") or 0) + n_out
        self.completion = n_out
        self.last_reasoning, self.fresh = reasoning, True
        text = f"<think>{reasoning}</think>{content}" if (content or not cut) else f"<think>{reasoning}"
        return text, cut
