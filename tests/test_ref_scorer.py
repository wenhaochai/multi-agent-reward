"""The reference scoring of kl=B (flint/ref_scorer.py; owner popups 2026-11-06 「方案A肯定也要做啊」 and 「和 rollout 引擎共用 4 张卡
（推荐）」). (1) The request: the sample's tokens, max_new_tokens 0, temperature 1, log-probs from P - 1. (2) The reply:
SGLang's placeholder first, then one log-prob per response token, each for the sample's own token (ids >= vocab - 1
come back as 0); a short reply, a missing placeholder, another token or a missing log-prob fail the sample. (3) A
group, against a fake server that answers as SGLang's input log-probs do (each token's log-prob a function of the token
and its position, so a shift by one is caught): every sample of every episode gets its own response tokens' scores, a
sample without response tokens none, and one line in the log. (4) A server error is retried (a 503 once), a refused
request is not, and a server that keeps failing fails the group. (5) The producer's hook (flint.elastic_rollout):
_generate_group scores the group miles generated before returning it, and without a scorer returns it as it was.
python3 tests/test_ref_scorer.py in miles-v0.1.3.sif with the run's miles on PYTHONPATH (no GPU)."""
import asyncio
import json
import math
import os
import socket
from types import SimpleNamespace

from helpers import check, done, env

env()
from aiohttp import web  # noqa: E402

from flint import ref_scorer as RS  # noqa: E402

RS.BACKOFF_S = (0.01,) * 10
VOCAB = 1000


def lp(token, pos):
    """The fake model: log p(token at pos | prefix)."""
    return -0.001 * token - 0.01 * pos - 0.1


def sample(tokens, response_length, index=0):
    return SimpleNamespace(tokens=list(tokens), response_length=response_length, index=index, ref_log_probs=None)


# (1) the request
p = RS.payload([5, 6, 7, 8, 9], 2)
check("payload: the sample's tokens, no new tokens, temperature 1, log-probs from P - 1",
      p["input_ids"] == [5, 6, 7, 8, 9] and p["logprob_start_len"] == 2 and p["sampling_params"]["max_new_tokens"] == 0
      and p["sampling_params"]["temperature"] == 1.0 and p["return_logprob"] is True, p)


# (2) the reply
def reply_for(tokens, start, vocab=VOCAB):
    ent = [[None, tokens[start] if tokens[start] < vocab - 1 else 0, None]]
    ent += [[lp(t, i), t if t < vocab - 1 else 0, None] for i, t in enumerate(tokens) if i > start]
    return {"meta_info": {"input_token_logprobs": ent}}


toks = [11, 12, 13, 14, 15, 16]
got = RS.parse(reply_for(toks, 3), toks, 2, VOCAB)
check("parse: the response tokens' log-probs, in order", got == [lp(15, 4), lp(16, 5)], got)
edge = [11, 12, VOCAB - 1, 13]
check("parse: an id >= vocab - 1 comes back as 0 and passes", RS.parse(reply_for(edge, 1), edge, 2, VOCAB) == [lp(VOCAB - 1, 2), lp(13, 3)])


def fails(reply, tokens, r, words):
    try:
        RS.parse(reply, tokens, r, VOCAB)
    except ValueError as e:
        return words in str(e)
    return False


short = reply_for(toks, 3); short["meta_info"]["input_token_logprobs"].pop()
check("parse: a reply one log-prob short fails", fails(short, toks, 2, "input log-probs for 2 response tokens"))
noph = reply_for(toks, 3); noph["meta_info"]["input_token_logprobs"][0][0] = -1.0
check("parse: a reply without SGLang's placeholder first fails", fails(noph, toks, 2, "placeholder"))
shifted = reply_for(toks, 2)   # scored from P - 2: one entry too many and every id one place off
check("parse: a reply scored from the wrong position fails", fails(shifted, toks, 2, "input log-probs"))
other = reply_for(toks, 3); other["meta_info"]["input_token_logprobs"][2][1] = 99
check("parse: another token on the server fails, at its place", fails(other, toks, 2, "token 1 of the response is 99"))
nan = reply_for(toks, 3); nan["meta_info"]["input_token_logprobs"][1][0] = float("nan")
check("parse: a response token without a finite log-prob fails", fails(nan, toks, 2, "finite"))


# (3), (4) against a fake server
def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


class Fake:
    def __init__(self, fail_first=0, status=503):
        self.calls, self.fail_first, self.status = 0, fail_first, status

    async def generate(self, request):
        body = await request.json()
        self.calls += 1
        if self.calls <= self.fail_first:
            return web.Response(status=self.status, text="busy")
        assert body["sampling_params"]["max_new_tokens"] == 0 and body["return_logprob"]
        return web.json_response(reply_for(body["input_ids"], body["logprob_start_len"]))


async def with_server(fake, fn):
    app = web.Application(); app.router.add_post("/generate", fake.generate)
    runner = web.AppRunner(app); await runner.setup()
    port = free_port(); site = web.TCPSite(runner, "127.0.0.1", port); await site.start()
    try:
        return await fn(f"http://127.0.0.1:{port}")
    finally:
        await runner.cleanup()


log = os.path.join(os.environ["TMPDIR"], "ref_scoring.jsonl")
eps = [[sample(range(1, 40), 25, 0), sample(range(1, 30), 9, 1)], sample(range(100, 140), 31, 2), [sample([7, 8, 9], 0, 3)]]
fake = Fake(fail_first=1)


async def run_group(url):
    scorer = RS.RefScorer(url, VOCAB, log)
    try:
        await scorer.score_group(eps)
    finally:
        await scorer._session.close()


asyncio.run(with_server(fake, run_group))
flat = [eps[0][0], eps[0][1], eps[1], eps[2][0]]
want = [[lp(t, len(s.tokens) - s.response_length + i) for i, t in enumerate(s.tokens[len(s.tokens) - s.response_length:])]
        for s in flat]
check("group: every sample of every episode has its own response tokens' scores",
      all(s.ref_log_probs == w for s, w in zip(flat, want)), [s.ref_log_probs and s.ref_log_probs[:2] for s in flat])
check("group: a sample without response tokens gets an empty list and no request", flat[3].ref_log_probs == [] and fake.calls == 4,
      (flat[3].ref_log_probs, fake.calls))
rec = [json.loads(x) for x in open(log)]
check("group: one log line (samples, tokens, response tokens, seconds)",
      len(rec) == 1 and rec[0]["samples"] == 4 and rec[0]["response_tokens"] == 25 + 9 + 31 and rec[0]["tokens"] == 39 + 29 + 40 + 3, rec)
check("retry: a 503 once, then the answer", fake.calls == 4 and all(s.ref_log_probs is not None for s in flat))


def raises(fake):
    async def go(url):
        scorer = RS.RefScorer(url, VOCAB)
        try:
            await scorer.score_group([sample(range(1, 20), 5)])
        finally:
            await scorer._session.close()
    try:
        asyncio.run(with_server(fake, go))
    except RuntimeError as e:
        return str(e)
    return None


bad = Fake(fail_first=99, status=500)
msg = raises(bad)
check("retry: a server that keeps failing fails the group after the retries", msg is not None and "HTTP 500" in msg
      and bad.calls == RS.RETRIES + 1, (msg, bad.calls))
refused = Fake(fail_first=99, status=400)
msg = raises(refused)
check("retry: a refused request (4xx) fails at once", msg is not None and "HTTP 400" in msg and refused.calls == 1, (msg, refused.calls))

# (5) the producer's hook
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn  # noqa: E402

from flint import elastic_rollout as ER  # noqa: E402

entry = SimpleNamespace(group=[sample(range(1, 12), 4)])


async def fake_generate(self, prompt_group):
    return entry


FullyAsyncRolloutFn._generate_group = fake_generate
fn = object.__new__(ER.ElasticFullyAsyncRolloutFn)
fn._ref = None
out = asyncio.run(fn._generate_group([]))
check("hook: without a scorer the group comes back as miles made it", out is entry and entry.group[0].ref_log_probs is None)


async def hooked(url):
    fn._ref = RS.RefScorer(url, VOCAB)
    try:
        return await fn._generate_group([])
    finally:
        await fn._ref._session.close()


out = asyncio.run(with_server(Fake(), hooked))
s = out.group[0]
check("hook: with a scorer the group comes back scored", out is entry and s.ref_log_probs ==
      [lp(t, 7 + i) for i, t in enumerate(s.tokens[7:])], s.ref_log_probs)
check("the fake model's scores are finite and differ by position", all(math.isfinite(x) for x in s.ref_log_probs)
      and len(set(s.ref_log_probs)) == len(s.ref_log_probs))
done()
