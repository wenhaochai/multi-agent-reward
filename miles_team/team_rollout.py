"""miles generate function for labs-molt's team math game: --custom-generate-function-path miles_team.team_rollout.generate

The game is molt's TeamMathAgent.run (examples/python/agents/team_math.py) line for line: the same knobs (MA_TM_* env),
prompts, delegate parsing, mailbox, cut/limit/invalid notes, grading, critical-path latency and team reward, from
team_core.py (a verbatim copy). What changes is the transport. molt's agent talks to an OpenAI chat server that
records each session as token segments; here each agent is a Session that does the same recording itself against
SGLang's /generate:

* a turn renders the whole chat with the chat template (enable_thinking per role); when that render string-extends the
  previous turn's (the chat minus its last user message, no generation prompt), the difference - the new user turn and
  the generation prompt - is tokenized and appended as masked feedback; otherwise a new segment starts from the full
  render (molt/agents/_chat_server.py _run_turn). The engine's tokens are appended as generated (loss mask 1, engine
  logprobs), so the trained tokens are exactly the sampled ones.
* the text an agent sees is the engine tokens decoded with special tokens kept, as molt's server returns it.

Training returns every segment of every session of the episode (lead + teammates) as one compact miles rollout
(shared rollout_id): lead segments carry R = task + collab - latency, teammate segments R - collab (molt audit M2).
miles weighs a compact rollout as one token-weighted mean (molt: one mean per session) - a known difference.
An odd count is padded with one dummy sample (1 masked token, metadata tm_pad) so every step has an even sample count:
Qwen3.5's Gated DeltaNet needs --qkv-format bshd, which needs static micro-batches, which need a multiple of the DP size.
Evaluation returns only the lead's final segment with reward = task (0/1), so eval means are per-episode accuracy.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, update_sample_from_response
from miles.rollout.generate_utils.sampling_mask import append_forced_sampling_tokens
from miles.utils.http_utils import post
from miles.utils.types import Sample

from . import team_core as C

MAX_LEN = int(os.environ.get("MA_TM_MAX_LEN", "32768"))  # molt --data.max_len: a session's context cap
PAD_EVEN = os.environ.get("MA_TM_PAD_EVEN", "1") == "1"


class Aborted(Exception):
    """The engine aborted a request (weight update / shutdown): the episode is retried whole."""


class Session:
    """One agent's chat session, recorded as token segments exactly as molt's chat server records it."""

    def __init__(self, input: GenerateFnInput, name: str, think: bool, max_len: int | None = None):
        self.input, self.args, self.tok = input, input.args, input.state.tokenizer
        self.max_len = max_len or MAX_LEN  # context cap of this session (default: MA_TM_MAX_LEN)
        self.name, self.think = name, think
        self.kw = {} if think else {"enable_thinking": False}
        self.segments: list[Sample] = []
        self.last_messages: list | None = None
        self.url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/generate"

    def _render(self, messages: list, gen: bool) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen, **self.kw)

    def _new_segment(self, prompt_text: str, prompt_ids: list[int]) -> Sample:
        s = deepcopy(self.input.sample)
        s.prompt, s.tokens, s.response, s.response_length = prompt_text, list(prompt_ids), "", 0
        s.loss_mask, s.rollout_log_probs, s.reward, s.status = [], [], None, Sample.Status.PENDING
        s.weight_versions = []
        s.metadata = {**(s.metadata or {}), "tm_role": self.name, "tm_segment": len(self.segments)}
        return s

    async def turn(self, messages: list, max_tokens: int) -> tuple[str, bool]:
        """One chat turn: (text, cut). ``cut`` = the reply hit its token cap (or the context was full)."""
        full = self._render(messages, True)
        seg = self.segments[-1] if self.segments else None
        extends = (seg is not None and self.last_messages is not None
                   and messages[: len(self.last_messages)] == self.last_messages)
        if extends:
            prefix = self._render(messages[:-1], False)
            extends = full.startswith(prefix)
        if extends:
            delta_ids = self.tok.encode(full[len(prefix):], add_special_tokens=False)
            gen_tokens = seg.tokens + delta_ids
        else:
            delta_ids = None
            gen_tokens = self.tok.encode(full, add_special_tokens=False)
        remaining = self.max_len - len(gen_tokens)
        if remaining <= 0:  # the context is full: nothing generated, the session ends truncated
            if extends:
                seg.status = Sample.Status.TRUNCATED
            self.last_messages = messages
            return "", True
        sp = dict(self.input.sampling_params)
        sp["max_new_tokens"] = min(max_tokens, remaining)
        payload, halt = compute_request_payload(self.args, gen_tokens, sp, evaluation=self.input.evaluation)
        if payload is None:
            self.last_messages = messages
            return "", True
        output = await post(self.url, payload)
        finish = output["meta_info"]["finish_reason"]["type"]
        if finish == "abort":
            raise Aborted(self.name)
        # commit only after a successful generate (molt: a failed call leaves the session untouched)
        if not extends:
            seg = self._new_segment(full, gen_tokens)
            self.segments.append(seg)
        else:
            seg.tokens = seg.tokens + delta_ids
            seg.response += self.tok.decode(delta_ids)
            seg.response_length += len(delta_ids)
            seg.loss_mask += [0] * len(delta_ids)
            seg.rollout_log_probs += [0.0] * len(delta_ids)
            if seg.rollout_sampling_mask is not None:
                append_forced_sampling_tokens(seg, delta_ids)
        truncated_before = seg.status == Sample.Status.TRUNCATED
        await update_sample_from_response(self.args, seg, payload=payload, output=output, update_loss_mask=True)
        if truncated_before:  # molt: traj.truncated is sticky across a segment's turns
            seg.status = Sample.Status.TRUNCATED
        ids = [item[1] for item in (output["meta_info"].get("output_token_logprobs") or [])]
        text = self.tok.decode(ids, skip_special_tokens=False) if ids else ""
        self.last_messages = messages
        return text, finish == "length"


def _pad_sample(like: Sample) -> Sample:
    """A 1-token dummy sibling that trains nothing (see the module docstring)."""
    s = deepcopy(like)
    n_prompt = len(like.tokens) - like.response_length
    s.tokens = like.tokens[: n_prompt + 1]
    s.response_length, s.loss_mask, s.rollout_log_probs = 1, [0], [0.0]
    s.response, s.weight_versions, s.remove_sample = "", [], True
    s.rollout_sampling_mask = None
    s.metadata = {**(like.metadata or {}), "tm_pad": 1, "tm_role": "pad"}
    return s


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    try:
        return await _play(input)
    except Aborted:
        s = deepcopy(input.sample)
        s.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=s)


async def _play(input: GenerateFnInput) -> GenerateFnOutput:
    tok = input.state.tokenizer
    meta = input.sample.metadata or {}
    messages = meta.get("messages")
    assert messages, "team rollout needs the raw chat messages in sample.metadata['messages'] (tools/make_team_data.py)"
    prompt_text = messages[-1]["content"]  # molt's ctx.prompt: the grader only strips it from a reply that echoes it
    label = input.sample.label
    n_tokens = lambda text: len(tok.encode(text, add_special_tokens=False))  # noqa: E731  (molt: the policy tokenizer)

    solo = C.MATES == 0
    suffix = (C.SOLO_SUFFIX.format(t=C.LEAD_TURNS, b=C.LEAD_BUDGET) if solo
              else C.LEAD_SUFFIX.format(k=C.MATES, m=C.MATE_TURNS, t=C.LEAD_TURNS, b=C.LEAD_BUDGET))
    lead = Session(input, "lead", C.LEAD_THINK)
    lead_hist = C._with_suffix(messages, suffix)
    mates: dict[int, dict] = {}  # n -> {"session", "hist", "turns"}
    rounds: list[dict] = []
    log: list[dict] = []
    final_text, final_cut = "", True
    delegated = read_report = False
    n_msgs = clamped = 0
    finished_report_tokens: list[int] = []
    new_prompt = n_tokens(json.dumps(lead_hist, ensure_ascii=False))
    lead_ctx = new_prompt
    turns = C.LEAD_TURNS
    for t in range(1, turns + 1):
        budget = max(1024, min(C.LEAD_BUDGET, MAX_LEN - lead_ctx - C.LEAD_RESERVE))
        clamped += int(budget < C.LEAD_BUDGET)
        text, cut = await lead.turn(lead_hist, budget)
        lead_hist.append({"role": "assistant", "content": text})
        gen = n_tokens(text)
        lead_ctx += gen
        rnd = {"lead": (gen, new_prompt), "mates": []}
        rounds.append(rnd)
        asked = [] if (solo or cut) else C.parse_delegations(text, thinking=C.LEAD_THINK)
        dels = [(n, m) for n, m in asked if n not in mates or mates[n]["turns"] < C.MATE_TURNS]
        entry = {"turn": t, "cut": cut, "budget": budget, "chars": len(text), "head": text[:1500],
                 "tail": text[-2500:], "visible": C.visible(text, C.LEAD_THINK)[:4000], "delegations": dels,
                 "reports": []}
        log.append(entry)
        last = t == turns
        invalid = (not solo) and (not cut) and not asked and "<delegate" in C.visible(text, C.LEAD_THINK).lower()
        if invalid and not last:
            note = C.INVALID_NOTE.format(k=C.MATES)
            note = note + "\n\n" + (C.TAIL_LAST if t + 1 == turns else C.TAIL_MORE.format(left=turns - t))
            lead_hist.append({"role": "user", "content": note})
            new_prompt = n_tokens(note)
            lead_ctx += new_prompt
            continue
        if asked and last:
            final_text, final_cut = text, True
            break
        if not asked:
            if not cut or last:
                final_text, final_cut = text, cut
                break
            note = C.CUT_NOTE
        elif not dels:
            note = C.LIMIT_NOTE.format(who=", ".join(str(n) for n, _ in asked), m=C.MATE_TURNS)
        else:
            note = None
        if note is not None:
            tail = C.TAIL_LAST if t + 1 == turns else (C.TAIL_SOLO if solo else C.TAIL_MORE).format(left=turns - t)
            note = note + "\n\n" + tail
            lead_hist.append({"role": "user", "content": note})
            new_prompt = n_tokens(note)
            lead_ctx += new_prompt
            continue
        delegated = True
        n_msgs += len(dels)
        blocked = [n for n, _ in asked if n not in dict(dels)]

        async def mate_turn(n: int, msg: str):
            if n not in mates:
                suffix = C.MATE_FIRST.format(j=n, msg=msg)
                if C.FORK:
                    notes = "\n\n".join(e["visible"] for e in log if e["visible"])
                    suffix += C.MATE_FORK.format(notes=notes[:4000] or "(none)")
                hist = C._with_suffix(messages, suffix)
                mates[n] = {"session": Session(input, f"mate{n}", C.MATE_THINK), "hist": hist, "turns": 0}
                prompt_tok = n_tokens(json.dumps(hist, ensure_ascii=False))
            else:
                user = C.MATE_NEXT.format(msg=msg)
                mates[n]["hist"].append({"role": "user", "content": user})
                prompt_tok = n_tokens(user)
            m = mates[n]
            try:
                mtext, mcut = await m["session"].turn(m["hist"], C.MATE_BUDGET)
            except Aborted:
                raise
            except Exception as exc:
                print(f"[team_rollout] teammate {n} call failed: {exc!r}", flush=True)
                if m["turns"] > 0:
                    m["hist"].pop()
                return n, "", True, "", 0, prompt_tok, True
            m["hist"].append({"role": "assistant", "content": mtext})
            m["turns"] += 1
            done = (not mcut) and ("</think>" in mtext or not C.MATE_THINK)
            _, ans = await asyncio.to_thread(C._grade, mtext, prompt_text, label) if done else (0.0, "")
            return n, mtext, not done, ans, n_tokens(mtext), prompt_tok, False

        got = await asyncio.gather(*[mate_turn(n, m) for n, m in dels])
        lines = []
        for n, mtext, mcut, ans, g, p, failed in got:
            rnd["mates"].append((g, p))
            lines.append(C.report_line(n, mtext, ans, mcut, failed=failed, thinking=C.MATE_THINK))
            entry["reports"].append({"to": n, "cut": mcut, "failed": failed, "answer": ans, "chars": len(mtext),
                                     "tokens": g, "report": C.visible(mtext, C.MATE_THINK)[:C.REPORT_CHARS]})
            read_report = read_report or not mcut
            if not mcut:
                finished_report_tokens.append(g)
        lines += [f"- Teammate {n}: (already used all {C.MATE_TURNS} messages; nothing was sent)" for n in blocked]
        box = C.MAILBOX.format(reports="\n\n".join(lines),
                               tail=C.TAIL_LAST if t + 1 == turns else C.TAIL_MORE.format(left=turns - t))
        lead_hist.append({"role": "user", "content": box})
        new_prompt = n_tokens(box)
        lead_ctx += new_prompt

    task, pred = (0.0, "") if final_cut else await asyncio.to_thread(C._grade, final_text, prompt_text, label)
    lat_tokens = C.critical_path(rounds)
    reward, collab, lat = C.team_reward(task, delegated, read_report, lat_tokens)
    mate_answers = [r["answer"] for e in log for r in e["reports"] if r["answer"]]
    mate_right = [C._grade(f"\\boxed{{{a}}}", prompt_text, label)[0] for a in mate_answers]
    info = {
        "tm_task_correct": task, "tm_reward": reward, "tm_collab_bonus": collab, "tm_latency_penalty": lat,
        "tm_latency_tokens": lat_tokens, "tm_lead_turns": len(log), "tm_delegated": 1.0 if delegated else 0.0,
        "tm_teammates": len(mates), "tm_messages": n_msgs, "tm_final_cut": 1.0 if final_cut else 0.0,
        "tm_mate_answer_correct": sum(mate_right) / len(mate_right) if mate_right else 0.0,
        "tm_any_mate_correct": 1.0 if any(c > 0 for c in mate_right) else 0.0, "tm_solo": 1.0 if solo else 0.0,
        "tm_lead_clamped": clamped, "tm_sessions": 1 + len(mates),
        "tm_cheap_bonus": 1.0 if (collab > 0 and finished_report_tokens
                                  and max(finished_report_tokens) < C.CHEAP_REPORT) else 0.0,
    }
    if C._trace_due():
        await asyncio.to_thread(
            C._write_trace,
            {"time": round(time.time(), 1), "solo": solo, "eval": input.evaluation, "prompt": prompt_text[:4000],
             "label": label, "task": task, "pred": pred, "reward": reward, "collab": collab, "latency": lat,
             "latency_tokens": lat_tokens, "lead": log, "temperature": input.sampling_params.get("temperature")},
        )

    if input.evaluation:  # one sample per episode, scored by the task alone
        out = lead.segments[-1] if lead.segments else deepcopy(input.sample)
        if not lead.segments:
            out.status = Sample.Status.TRUNCATED
        out.reward = task
        out.metadata = {**(out.metadata or {}), **info}
        return GenerateFnOutput(samples=out)

    rollout_id = input.sample.rollout_id if input.sample.rollout_id is not None else input.sample.index
    samples: list[Sample] = []
    for sess, r in [(lead, reward)] + [(m["session"], reward - collab) for m in mates.values()]:
        for seg in sess.segments:
            if not any(seg.loss_mask):  # nothing generated in it (molt's stitch drops empty rollouts)
                continue
            seg.reward, seg.rollout_id = r, rollout_id
            samples.append(seg)
    if not samples:  # nothing trainable: hand back the prompt as a truncated, zero-reward sample
        s = deepcopy(input.sample)
        s.status, s.reward = Sample.Status.TRUNCATED, reward
        return GenerateFnOutput(samples=s)
    samples[0].metadata = {**(samples[0].metadata or {}), **info}  # episode stats ride on the lead's first segment
    if PAD_EVEN and len(samples) % 2 == 1:
        pad = _pad_sample(samples[0])
        pad.reward = samples[0].reward
        samples.append(pad)
    return GenerateFnOutput(samples=samples if len(samples) > 1 else samples[0])
