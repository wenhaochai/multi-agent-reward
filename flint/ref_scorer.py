"""Reference log-probs for a KL loss to the original Flint, scored during rollout (rl/rl_job.sh kl=B; owner popups
2026-11-06 「方案A肯定也要做啊，这不是GRPO里的必备项吗」 and 「和 rollout 引擎共用 4 张卡（推荐）」).

The trainer has no room for a reference model (host memory peaks at 790-819 of the 860 GiB guard, the actor's GPUs at
72-75 of 80 GB), so a second SGLang server holds the original weights on the run's own engine GPUs (rl/rl_inner.sh
REF_*), and the producer (flint.elastic_rollout) sends every sample of a finished group to it before the group enters
the buffer. miles (flint-grpo-kl-rollref, --ref-log-probs-from-rollout) carries Sample.ref_log_probs to the KL loss,
which compares them with the trainer's full-vocabulary log-probs of the sampled tokens.

One request per sample: input_ids = the sample's tokens, max_new_tokens 0, temperature 1 (the server returns
log_softmax of the raw logits; miles refuses the flag unless --rollout-temperature is 1), return_logprob from
logprob_start_len = P - 1, where P = len(tokens) - response_length. SGLang returns input_token_logprobs for the tokens
at P - 1 .. n - 1 with a None first (its hidden state is not computed: sglang managers/scheduler_components/
logprob_result_processor.py, [None] + logprobs[:-1]), so entries 1.. are log p(token_i | tokens_<i) for the response
tokens i = P .. n - 1. A start at P - 1 also caps the radix-cache match there (schedule_batch.py), so a shared prompt
is reused and every response token is computed. The returned token ids must equal the sample's (SGLang returns 0 for
ids >= vocab_size - 1), or the sample fails. check() runs at the job's start: the server's own sampled tokens, scored
again, must come back with their sampling log-probs (decode against prefill numerics), and the same scores shifted by
one position must not.

A failure (after RETRIES tries over about 10 min) raises: the group's task fails and the run ends, as a KL with a
missing reference must not train. Records one line per group in MA_REF_LOG (json: samples, tokens, seconds).
"""

import asyncio
import json
import math
import os
import sys
import time

import aiohttp

RETRIES = 6
BACKOFF_S = (5, 15, 30, 60, 120, 300)
REQUEST_TIMEOUT_S = 600.0       # one sample of up to 70k tokens behind at most one other (audit 2026-11-06: a server that
                                # took requests and never answered held a group 7 x 30 min, past the GPU watch)
CONCURRENCY = int(os.environ.get("MA_REF_CONCURRENCY", "2"))


def _samples(group) -> list:
    """Every sample of a finished group: a group holds one entry per episode, a sample or a list of the episode's
    samples (one per session-tree leaf)."""
    out = []
    for ep in group if isinstance(group, list) else [group]:
        out.extend(ep if isinstance(ep, list) else [ep])
    return out


def payload(tokens: list[int], response_length: int) -> dict:
    p = len(tokens) - response_length
    assert p >= 1 and response_length >= 1, (len(tokens), response_length)
    return {"input_ids": list(tokens), "sampling_params": {"max_new_tokens": 0, "temperature": 1.0},
            "return_logprob": True, "logprob_start_len": p - 1, "stream": False}


def parse(reply: dict, tokens: list[int], response_length: int, vocab_size: int) -> list[float]:
    """The response tokens' reference log-probs from a /generate reply to payload(tokens, response_length)."""
    entries = reply["meta_info"]["input_token_logprobs"]
    if len(entries) != response_length + 1:
        raise ValueError(f"reference scores: {len(entries)} input log-probs for {response_length} response tokens + 1")
    if entries[0][0] is not None:
        raise ValueError("reference scores: the first input log-prob is not SGLang's placeholder (None)")
    want = [t if t < vocab_size - 1 else 0 for t in tokens[len(tokens) - response_length:]]
    got = [int(e[1]) for e in entries[1:]]
    if got != want:
        k = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
        raise ValueError(f"reference scores: token {k} of the response is {got[k]} on the server, {want[k]} in the sample")
    out = [e[0] for e in entries[1:]]
    if any(x is None or not math.isfinite(x) for x in out):
        raise ValueError("reference scores: a response token without a finite log-prob")
    return [float(x) for x in out]


class RefScorer:
    def __init__(self, url: str, vocab_size: int, log_path: str | None = None):
        self.url = url.rstrip("/") + "/generate"
        self.vocab_size = vocab_size
        self.log_path = log_path
        self._sem = asyncio.Semaphore(CONCURRENCY)
        self._session = None

    async def _post(self, body: dict) -> dict:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S))
        last = None
        for i in range(RETRIES + 1):
            try:
                async with self._sem, self._session.post(self.url, json=body) as r:
                    if r.status == 200:
                        return await r.json()
                    last = f"HTTP {r.status}: {(await r.text())[:300]}"
                    if r.status < 500:
                        break   # a request the server refuses will not pass on a retry
            except asyncio.TimeoutError as e:
                # a request that ran REQUEST_TIMEOUT_S without an answer is not retried (audit 2026-11-06: 7 tries of a
                # hung server took ~79 min, past the GPU watch's 60): the group fails now
                last = f"{type(e).__name__}: no answer in {REQUEST_TIMEOUT_S:.0f} s"
                break
            except aiohttp.ClientError as e:
                last = f"{type(e).__name__}: {e}"
            if i < RETRIES:
                await asyncio.sleep(BACKOFF_S[i])
        raise RuntimeError(f"reference server {self.url}: {last}")

    async def score_sample(self, s) -> None:
        if s.response_length == 0:
            s.ref_log_probs = []
            return
        reply = await self._post(payload(s.tokens, s.response_length))
        s.ref_log_probs = parse(reply, s.tokens, s.response_length, self.vocab_size)

    async def score_group(self, group) -> str:
        """Fill ref_log_probs on every sample of a finished group (the samples of one GRPO group's episodes). A group
        with an aborted sample is skipped: miles' buffer rejects it (apply_aborted_filter). A group the sample filter
        will take out of the loss (flint.miles_agent.group_leaves_the_loss: a failed episode or a sample over the
        actor's cap, which may not fit the server's KV pool either) gets zeros: its loss mask becomes 0, so the KL
        never reads them, and the train data still needs a reference per sample. Returns what it did."""
        from flint.miles_agent import group_leaves_the_loss

        every = _samples(group)
        if any(str(getattr(s.status, "value", s.status)).lower() == "aborted" for s in every):
            what = "skipped (aborted)"
            todo = []
        elif group_leaves_the_loss(every):
            for s in every:
                if getattr(s, "ref_log_probs", None) is None:
                    s.ref_log_probs = [0.0] * s.response_length
            what = "zeros (the group leaves the loss)"
            todo = []
        else:
            what = "scored"
            todo = [s for s in every if getattr(s, "ref_log_probs", None) is None]
        t0 = time.monotonic()
        await asyncio.gather(*(self.score_sample(s) for s in todo))
        if self.log_path:
            rec = {"time": time.time(), "what": what, "samples": len(todo), "tokens": sum(len(s.tokens) for s in todo),
                   "response_tokens": sum(s.response_length for s in todo), "seconds": round(time.monotonic() - t0, 1)}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return what


def vocab_size_of(model_dir: str) -> int:
    return int(json.load(open(os.path.join(model_dir, "config.json")))["vocab_size"])


CHECK_PROMPT_TOKENS = 3500   # with the 1,200 generated ones the sequence crosses the 4,096-token prefill chunk
CHECK_NEW_TOKENS = 1200
CHECK_JUMP = 0.1             # a position whose sampled log-prob differs this much from its neighbor's tests the shift
CHECK_MIN_JUMPS = 30
FCS_PROBLEMS = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems"


def _check_prompt(tok) -> list[int]:
    """Natural text, not a repeated paragraph: Frontier-CS problem statements in id order until 3,500 tokens (w1429's
    check, 2026-11-06, repeated one paragraph; the model copied it at log-probs near 0, and a shift by one changed
    almost nothing, so the shift test had no power)."""
    text, ids = "", []
    for d in sorted((x for x in os.listdir(FCS_PROBLEMS) if x.isdigit()), key=int):
        f = os.path.join(FCS_PROBLEMS, d, "statement.txt")
        if os.path.isfile(f):
            text += open(f, errors="replace").read() + "\n\n"
            ids = tok.encode(text)
            if len(ids) >= CHECK_PROMPT_TOKENS:
                break
    assert len(ids) >= CHECK_PROMPT_TOKENS, f"check prompt: {len(ids)} tokens from {FCS_PROBLEMS}"
    return ids[:CHECK_PROMPT_TOKENS]


async def _check(url: str, model_dir: str) -> dict:
    """The server's own sampled tokens, scored again, come back with their sampling log-probs, also across the prefill
    chunk boundary at 4,096 tokens: mean |diff| under 0.03 and its 99th percentile under 0.5 (decode against prefill
    numerics; a router flip moves a single token). A shift by one position must show: on the positions whose sampled
    log-prob differs from its neighbor's by over 0.1 (at least 30), the shifted mean |diff| is over 5x the aligned one."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    prompt = _check_prompt(tok)
    scorer = RefScorer(url, vocab_size_of(model_dir))
    try:
        gen = await scorer._post({"input_ids": prompt, "return_logprob": True, "stream": False,
                                  "sampling_params": {"max_new_tokens": CHECK_NEW_TOKENS, "temperature": 1.0, "ignore_eos": True}})
        out = gen["meta_info"]["output_token_logprobs"]
        ids, sampled = [int(e[1]) for e in out], [float(e[0]) for e in out]
        tokens = prompt + ids
        # an empty radix cache, so the scoring prefill runs the whole sequence and crosses the 4,096-token chunk (the
        # cached prompt would otherwise leave only the 1,200 new tokens, one chunk; audit 2026-11-06)
        # timeout: wait up to 30 s for the finished generation to leave the scheduler (at 0 SGLang answers 400 meanwhile;
        # audit 2026-11-06)
        async with scorer._session.post(url.rstrip("/") + "/flush_cache?timeout=30") as r:
            if r.status != 200:
                raise RuntimeError(f"reference server {url}: /flush_cache answered {r.status}")
        scored = parse(await scorer._post(payload(tokens, len(ids))), tokens, len(ids), scorer.vocab_size)
    finally:
        if scorer._session is not None:
            await scorer._session.close()
    diff = [abs(a - b) for a, b in zip(scored, sampled)]
    jumps = [i for i in range(1, len(sampled)) if abs(sampled[i] - sampled[i - 1]) > CHECK_JUMP]
    aligned_j = sum(abs(scored[i] - sampled[i]) for i in jumps) / max(len(jumps), 1)
    shifted_j = sum(abs(scored[i] - sampled[i - 1]) for i in jumps) / max(len(jumps), 1)
    srt = sorted(diff)
    res = {"tokens": len(ids), "crosses_chunk": len(tokens) > 4096, "mean_abs_diff": sum(diff) / len(diff),
           "p99_abs_diff": srt[int(0.99 * (len(srt) - 1))], "max_abs_diff": srt[-1], "jump_positions": len(jumps),
           "aligned_mean_at_jumps": aligned_j, "shifted_mean_at_jumps": shifted_j}
    res["ok"] = (res["crosses_chunk"] and res["mean_abs_diff"] < 0.03 and res["p99_abs_diff"] < 0.5
                 and len(jumps) >= CHECK_MIN_JUMPS and shifted_j > 5 * aligned_j)
    return res


def check(url: str, model_dir: str) -> int:
    res = asyncio.run(_check(url, model_dir))
    print(f"[ref] check {'OK' if res['ok'] else 'FAILED'}: {json.dumps(res)}", flush=True)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "check":
        sys.exit(check(sys.argv[2], sys.argv[3]))
    print("usage: python3 -m flint.ref_scorer check URL MODEL_DIR", file=sys.stderr)
    sys.exit(2)
