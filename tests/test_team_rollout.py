"""Offline check of miles_team.team_rollout against a scripted fake SGLang /generate (no GPU): the full team game,
molt-exact token stitching of multi-turn sessions, per-session rewards, eval output, padding, and the FlashREINFORCE
post-process. Runs in miles.sif with the real Qwen3.8 tokenizer:
  PYTHONPATH=/root/Megatron-LM:<miles src>:<this repo> python3 tests/test_team_rollout.py team|solo
"""
import asyncio
import os
import sys
from types import SimpleNamespace

MODE = sys.argv[1]
os.environ.update(MA_TM_MATES="0" if MODE == "solo" else "3", MA_TM_LEAD_TURNS="4", MA_TM_MATE_TURNS="2",
                  MA_TM_LEAD_THINK="0", MA_TM_MATE_THINK="0", MA_TM_LEAD_BUDGET="12288", MA_TM_MATE_BUDGET="12288",
                  MA_TM_COLLAB_BONUS="0.1", MA_TM_TRACE_DIR="")
MODEL = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/Qwen3.8-27B"

from transformers import AutoTokenizer  # noqa: E402

from miles.rollout.base_types import GenerateFnInput  # noqa: E402
from miles.utils.types import Sample  # noqa: E402

import miles_team.team_rollout as TR  # noqa: E402
from miles_team import reward_post  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
EOS = tok.eos_token_id
KW = {"enable_thinking": False}
PROBLEM = [{"role": "user", "content": "Solve the following math problem step by step. The last line of your "
            "response should be of the form Answer: $Answer (without quotes).\n\nWhat is 200 + 4?"}]

if MODE == "team":
    SCRIPT = {
        "lead": [("I will split the work.\n<delegate to=1>Compute 200+4.</delegate>\n<delegate to=2>Check it.</delegate>", "stop"),
                 ("Both agree. The answer is \\boxed{204}.", "stop")],
        "mate1": [("200 + 4 = 204. Report: \\boxed{204}", "stop")],
        "mate2": [("x" * 50, "length")],  # cut off: no report
    }
else:
    SCRIPT = {"lead": [("Working step by step 200 plus", "length"), ("... 4 gives \\boxed{204}.", "stop")]}
CALLS = []


def role_of(input_ids):
    text = tok.decode(input_ids)
    for j in (1, 2, 3):
        if f"You are teammate {j} of a team" in text:
            return f"mate{j}"
    return "lead"


async def fake_post(url, payload, headers=None):
    role = role_of(payload["input_ids"])
    reply, finish = SCRIPT[role].pop(0)
    ids = tok.encode(reply, add_special_tokens=False)
    if finish == "stop":
        ids = ids + [EOS]  # SGLang reports the matched EOS as an output token
    CALLS.append((role, list(payload["input_ids"]), ids, payload["sampling_params"]["max_new_tokens"]))
    return {"text": reply, "meta_info": {"finish_reason": {"type": finish}, "completion_tokens": len(ids),
                                         "output_token_logprobs": [[-0.25, i, None] for i in ids]}}


TR.post = fake_post
class Args(SimpleNamespace):
    def __getattr__(self, name):  # any flag the rollout helpers read that the test does not set: off
        return False


args = Args(sglang_router_ip="127.0.0.1", sglang_router_port=1, rollout_max_response_len=12288,
                       rollout_max_context_len=32768, use_rollout_routing_replay=False,
                       use_rollout_indexer_replay=False, sglang_speculative_algorithm=None, reward_key=None,
                       lora_rank=0, lora_adapter_path=None, return_sampling_mask=False)


def run(evaluation):
    CALLS.clear()
    for k, v in list(SCRIPT.items()):
        SCRIPT[k] = list(ORIG[k])
    sample = Sample(index=7, group_index=7, prompt="(templated)", label="204",
                    metadata={"messages": PROBLEM, "datasource": "test"})
    state = SimpleNamespace(args=args, tokenizer=tok)
    inp = GenerateFnInput(state=state, sample=sample, sampling_params={"temperature": 1.0, "top_p": 1.0,
                                                                        "max_new_tokens": 12288},
                          evaluation=evaluation)
    return asyncio.run(TR.generate(inp)).samples


ORIG = {k: list(v) for k, v in SCRIPT.items()}
out = run(evaluation=False)
out = out if isinstance(out, list) else [out]
real = [s for s in out if not s.metadata.get("tm_pad")]
lead = [s for s in real if s.metadata["tm_role"] == "lead"]
assert len(lead) == 1, f"lead should be ONE stitched segment, got {len(lead)}"
L = lead[0]

# 1) the lead's token stream is molt's: turn-1 prompt render, turn-1 tokens as generated (EOS kept), then the template
#    delta string (new user turn + generation prompt) tokenized, then turn-2 tokens
lead_calls = [c for c in CALLS if c[0] == "lead"]
p1 = tok.encode(tok.apply_chat_template(TR.C._with_suffix(PROBLEM, (TR.C.SOLO_SUFFIX if MODE == "solo" else TR.C.LEAD_SUFFIX).format(
    k=TR.C.MATES, m=TR.C.MATE_TURNS, t=TR.C.LEAD_TURNS, b=TR.C.LEAD_BUDGET)), tokenize=False, add_generation_prompt=True, **KW),
    add_special_tokens=False)
assert lead_calls[0][1] == p1, "turn-1 prompt ids differ from the template render"
g1, g2 = lead_calls[0][2], lead_calls[1][2]
turn2_in = lead_calls[1][1]
assert turn2_in[: len(p1) + len(g1)] == p1 + g1, "turn 2 must extend the turn-1 tokens as generated"
delta = turn2_in[len(p1) + len(g1):]
dtext = tok.decode(delta)
assert dtext.startswith("<|im_start|>user\n") and dtext.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), dtext
assert L.tokens == turn2_in + g2, "segment tokens = turn-2 input + turn-2 generation"
n_prompt = len(L.tokens) - L.response_length
assert n_prompt == len(p1)
assert L.loss_mask == [1] * len(g1) + [0] * len(delta) + [1] * len(g2), "loss mask: generated 1, feedback 0"
assert L.rollout_log_probs == [-0.25] * len(g1) + [0.0] * len(delta) + [-0.25] * len(g2)
assert len(L.loss_mask) == L.response_length == len(L.rollout_log_probs)
print(f"[ok] lead stitched: prompt {len(p1)} + gen {len(g1)} + feedback {len(delta)} + gen {len(g2)} tokens; "
      f"feedback = {dtext[:60]!r}...")

info = L.metadata
if MODE == "team":
    assert info["tm_delegated"] == 1.0 and info["tm_teammates"] == 2 and info["tm_task_correct"] == 1.0
    assert abs(info["tm_collab_bonus"] - 0.1) < 1e-9, info
    R = info["tm_reward"]
    mates = [s for s in real if s.metadata["tm_role"].startswith("mate")]
    assert len(mates) == 2 and all(abs(s.reward - (R - 0.1)) < 1e-9 for s in mates), [s.reward for s in mates]
    assert abs(L.reward - R) < 1e-9
    pads = [s for s in out if s.metadata.get("tm_pad")]
    assert len(out) % 2 == 0 and len(pads) == 1 and pads[0].loss_mask == [0] and pads[0].response_length == 1
    assert len({s.rollout_id for s in out}) == 1 and out[0].rollout_id == 7
    m2 = [s for s in mates if s.metadata["tm_role"] == "mate2"][0]
    assert m2.status == Sample.Status.TRUNCATED and m2.loss_mask and all(m2.loss_mask), m2.loss_mask
    box = tok.decode(delta)
    assert "Teammate 1 (answer: 204)" in box and "Teammate 2: (ran out of tokens" in box, box
    print(f"[ok] team: R={R:.4f} lead, {R - 0.1:.4f} x2 mates, 1 pad; mailbox reached the lead")
else:
    assert info["tm_task_correct"] == 1.0 and info["tm_teammates"] == 0 and info["tm_collab_bonus"] == 0
    assert "(Your previous turn ran out of tokens" in dtext, dtext
    assert L.status == Sample.Status.TRUNCATED, "molt: a cut turn marks the segment truncated (sticky)"
    print(f"[ok] solo: cut turn continued in the same segment, R={L.reward:.4f}")

raw, adv = reward_post.flash_reinforce(args, out)
mean = sum(r for r, s in zip(raw, out) if not s.metadata.get("tm_pad")) / len(real)
for r, a, s in zip(raw, adv, out):
    assert abs(a - (0.0 if s.metadata.get("tm_pad") else r - mean)) < 1e-9
print(f"[ok] flash_reinforce: adv {[round(a, 4) for a in adv]}")

ev = run(evaluation=True)
assert isinstance(ev, Sample) and ev.reward == 1.0 and ev.metadata["tm_role"] == "lead"
print(f"[ok] eval: one sample per episode, reward = task = {ev.reward}")
print("ALL OK", MODE)
