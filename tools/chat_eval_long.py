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
MA_AG_TRACE_DIR (defaults to <out>/traces). Output: <out>/results.jsonl, one line per finished episode (resumable)."""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True)
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


class ChatSession:
    """team_rollout.Session's interface over the chat API: turn(messages, max_tokens) -> (text, cut). The text is
    '<think>reasoning</think>answer' so fcs_agent's answer() splits it as for the in-process engines; the reasoning
    of the last turn waits in last_reasoning for _Agent to store in the history."""

    def __init__(self, input, name: str, think: bool, max_len: int | None = None):
        self.input, self.name, self.think = input, name, think
        self.max_len = max_len or A.MAX_LEN
        self.segments, self.last_messages = [], None
        self.last_reasoning, self.fresh, self.used = "", False, 0   # used: prompt + completion tokens of the last call

    async def turn(self, messages: list, max_tokens: int) -> tuple[str, bool]:
        est = self.used + sum(len(m["content"]) for m in messages[-1:]) // 3   # the new observation, ~3 chars/token
        room = self.max_len - est
        if room <= 256:
            return "", True
        body = {"model": args.model, "messages": messages, "max_tokens": min(max_tokens, room),
                "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}
        out = None
        for attempt in range(4):
            try:
                async with _http.post(f"{args.base_url}/v1/chat/completions", json=body) as r:
                    if r.status == 400:
                        err = await r.text()
                        if "context" in err.lower() or "length" in err.lower():
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
        msg = ch["message"]
        reasoning, content = msg.get("reasoning_content") or "", msg.get("content") or ""
        n_out = int(usage.get("completion_tokens") or 0)
        self.used = int(usage.get("prompt_tokens") or 0) + n_out
        self.segments.append(SimpleNamespace(loss_mask=[1] * n_out))
        self.last_reasoning, self.fresh = reasoning, True
        cut = ch.get("finish_reason") == "length"
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
          f"N={L.N} tokens={L.TOKENS} ctx_chars={L.CTX_CHARS} turn={A.TURN} max_len={A.MAX_LEN}", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    _http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout))

    async def one(i, e, r):
        async with sem:
            t0 = time.time()
            inp = SimpleNamespace(evaluation=True, sampling_params={},
                                  sample=SimpleNamespace(index=i, group_index=e, label=r["label"], metadata=r["metadata"]))
            rec = {"pid": r["metadata"]["pid"], "episode": e, "model": args.model}
            try:
                _, S, info = await L.episode(inp, r["metadata"]["messages"], r["label"])
                rec.update(S=S, info=info)
            except Aborted as ex:
                rec.update(S=None, aborted=str(ex))
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


asyncio.run(main())
