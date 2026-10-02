"""Offline check of miles_team.fcs_team (game G1) against a scripted fake SGLang and a fake judge (no GPU): every
reward arm on hand-made per-case vectors, adoption by tag and by identical code, one sample per agent with the
episode's rollout_id and role groups, the masked stand-in for an agent that generated nothing, solo reduction,
team and solo eval. Runs in miles.sif with the real Qwen3.5 tokenizer:
  PYTHONPATH=/root/Megatron-LM:<miles src>:<this repo> python3 tests/test_fcs_team.py
"""
import asyncio
import math
import os
from types import SimpleNamespace

os.environ.update(MA_FT_MATES="3", MA_FT_REWARD="shared", MA_FT_TRACE_DIR="", MA_FT_MATE_BUDGET="4096",
                  MA_FT_LEAD_BUDGET="4096")
MODEL = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/Qwen3.5-9B-FCS-SFT-lm"

from transformers import AutoTokenizer  # noqa: E402

from miles.rollout.base_types import GenerateFnInput  # noqa: E402
from miles.utils.types import Sample  # noqa: E402

import miles_team.team_rollout as TR  # noqa: E402
import miles_team.fcs_team as F  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
EOS = tok.eos_token_id
PROBLEM = [{"role": "user", "content": "You are a competitive programmer. Solve the following problem in C++. Output "
            "ONLY the C++ code wrapped in ```cpp and ```. No explanation.\n\nPrint 42.\n\nGenerate solution code:"}]


def cpp(body):
    return f"thinking...\n</think>\n\n```cpp\n{body}\n```"


# per-case ratio vectors the fake judge returns for each program body (3 cases)
CASES = {"int main(){A;}": [1.0, 0.0, 0.0], "int main(){B;}": [0.0, 1.0, 0.0], "int main(){C;}": [0.5, 0.5, 0.0],
         "int main(){D;}": [1.0, 1.0, 1.0]}
MATE_REPLIES, LEAD_REPLY = [], [""]
CALLS = []


async def fake_post(url, payload, headers=None):
    text = tok.decode(payload["input_ids"])
    if "You lead a team" in text:
        role, (reply, finish) = "lead", LEAD_REPLY[0]
    else:
        role, (reply, finish) = "mate", MATE_REPLIES.pop(0)
    ids = tok.encode(reply, add_special_tokens=False)
    if finish == "stop":
        ids = ids + [EOS]
    CALLS.append((role, len(payload["input_ids"]), ids))
    return {"text": reply, "meta_info": {"finish_reason": {"type": finish}, "completion_tokens": len(ids),
                                         "output_token_logprobs": [[-0.5, i, None] for i in ids]}}


async def fake_judge(problem_dir, code):
    cases = CASES.get(code.strip(), [0.0, 0.0, 0.0]) if code else [0.0, 0.0, 0.0]
    status = "done" if code.strip() in CASES else ("no code" if not code else "compile error")
    return {"score": sum(cases) / len(cases), "cases": cases, "status": status, "infra_error": False}


TR.post = fake_post
F.judge_code = fake_judge


class Args(SimpleNamespace):
    def __getattr__(self, name):
        return False


args = Args(sglang_router_ip="127.0.0.1", sglang_router_port=1, rollout_max_response_len=4096,
            rollout_max_context_len=65536, use_rollout_routing_replay=False, use_rollout_indexer_replay=False,
            sglang_speculative_algorithm=None, reward_key=None, lora_rank=0, lora_adapter_path=None,
            return_sampling_mask=False)


def play(mates, lead, evaluation=False, mode=None, group=5):
    MATE_REPLIES[:] = list(mates)
    LEAD_REPLY[0] = lead
    CALLS.clear()
    md = {"messages": PROBLEM, "pid": "t"}
    if mode:
        md["ft_mode"] = mode
    sample = Sample(index=11, group_index=group, prompt="(templated)", label="/nonexistent/problem", metadata=md)
    inp = GenerateFnInput(state=SimpleNamespace(args=args, tokenizer=tok), sample=sample,
                          sampling_params={"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 4096},
                          evaluation=evaluation)
    out = asyncio.run(F.generate(inp)).samples
    return out if isinstance(out, list) else [out]


close = lambda a, b: abs(a - b) < 1e-9  # noqa: E731
MATES3 = [(cpp("int main(){A;}"), "stop"), (cpp("int main(){B;}"), "stop"), (cpp("int main(){C;}"), "stop")]
sA, sB, sC, sD = 1 / 3, 1 / 3, 1 / 3, 1.0

# 1) reward arms on hand-made vectors (s_j, cases), S = 1.0, adopted = 2
s, cases = [sA, sB, sC], [CASES["int main(){A;}"], CASES["int main(){B;}"], CASES["int main(){C;}"]]
assert F.rewards("shared", 1.0, s, cases, None) == (1.0, [1.0] * 3)
assert F.rewards("indiv", 1.0, s, cases, None) == (1.0, s)
lr, mr = F.rewards("mix", 1.0, s, cases, None, alpha=0.25)
assert close(lr, 1.0) and all(close(m, 0.25 * x + 0.75) for m, x in zip(mr, s))
# unique: A owns case 0 by 0.5 over C, B owns case 1 by 0.5 over C, C owns nothing
lr, mr = F.rewards("unique", 1.0, s, cases, None, beta=0.0)
assert all(close(a, b) for a, b in zip(mr, [0.5 / 3, 0.5 / 3, 0.0])), mr
lr, mr = F.rewards("unique", 1.0, s, cases, None, beta=0.2)
assert all(close(a, b) for a, b in zip(mr, [0.5 / 3 + 0.2, 0.5 / 3 + 0.2, 0.2])), mr
assert F.rewards("adopt", 0.7, s, cases, 2) == (0.7, [0.0, 0.7, 0.0])
lr, mr = F.rewards("synth", 1.0, s, cases, None, alpha=0.5)
assert close(lr, 1.0 - 1 / 3) and all(close(m, 0.5 * x + 0.5) for m, x in zip(mr, s))
assert F.unique_credit([[0.3, 0.9]], 0) == 0.6  # a lone teammate owns everything it scores
print("[ok] reward arms")

# 2) adoption parsing
assert F.parse_adopt("t</think>\n<adopt 2/>", 3) == 2 and F.parse_adopt("<adopt j=3/>", 3) == 3
assert F.parse_adopt("t</think>\n<adopt 4/>", 3) is None and F.parse_adopt("<adopt 2/> in thinking</think>ok", 3) is None
print("[ok] adoption parsing")

# 3) full episode, shared arm, lead writes a new program D
out = play(MATES3, (cpp("int main(){D;}"), "stop"))
assert len(out) == 4, len(out)
roles = [o.metadata["ft_role"] for o in out]
assert roles == ["lead", "mate1", "mate2", "mate3"], roles
assert all(close(o.reward, sD) for o in out), [o.reward for o in out]
assert len({o.rollout_id for o in out}) == 1
assert [o.group_index for o in out] == [10, 11, 11, 11], [o.group_index for o in out]
assert all(any(o.loss_mask) for o in out)
info = out[0].metadata
assert close(info["ft_lead_score"], 1.0) and close(info["ft_best_mate"], 1 / 3) and info["ft_adopt"] == 0.0
assert close(info["ft_synth_gain"], 2 / 3) and info["ft_mate_bad"] == 0.0 and info["ft_tokens"] > 0
lead_call = [c for c in CALLS if c[0] == "lead"][0]
lead_text = tok.decode(out[0].tokens)
assert "### Teammate 2\n```cpp\nint main(){B;}\n```" in lead_text and "<adopt j/>" in lead_text
assert sum(1 for c in CALLS if c[0] == "mate") == 3
print(f"[ok] episode: 4 samples, rewards {[round(o.reward, 3) for o in out]}, groups {[o.group_index for o in out]}")

# 4) adoption by tag (indiv arm) and by identical code (adopt arm)
F.REWARD = "indiv"
out = play(MATES3, ("hmm\n</think>\n\n<adopt 2/>", "stop"))
assert out[0].metadata["ft_adopt"] == 1.0 and out[0].metadata["ft_adopt_tag"] == 1.0
assert close(out[0].reward, sB) and [round(o.reward, 4) for o in out[1:]] == [round(x, 4) for x in s]
F.REWARD = "adopt"
out = play(MATES3, (cpp("int  main(){C;}"), "stop"))  # whitespace differs: still teammate 3's program
assert out[0].metadata["ft_adopt"] == 1.0 and out[0].metadata["ft_adopt_tag"] == 0.0
assert [o.reward for o in out[1:]] == [0.0, 0.0, out[0].reward], [o.reward for o in out]
print("[ok] adoption by tag and by identical code")

# 5) a teammate cut off with no code; a lead that generates nothing gets a masked stand-in
F.REWARD = "shared"
out = play([MATES3[0], ("still thinking " * 20, "length"), MATES3[2]], (cpp("int main(){A;}"), "stop"))
assert out[0].metadata["ft_mate_cut"] == 1 / 3 and "no code" in tok.decode(out[0].tokens)
saved = F.MAX_LEN
mate_prompt = len(tok.encode(tok.apply_chat_template(PROBLEM, tokenize=False, add_generation_prompt=True),
                             add_special_tokens=False))
F.MAX_LEN = mate_prompt + 120  # mates fit, the lead's prompt (with three programs) does not
out = play(MATES3, (cpp("int main(){D;}"), "stop"))
F.MAX_LEN = saved
assert len(out) == 4 and out[0].metadata.get("tm_pad") == 1 and not any(out[0].loss_mask), out[0].metadata
assert out[0].metadata["ft_lead_cut"] == 1.0 and close(out[0].reward, 0.0) and out[0].group_index == 10
print("[ok] cut teammate, empty lead -> masked stand-in; still 4 samples")

# 6) team eval returns the lead's sample with reward S; solo eval plays one agent on the solo prompt
out = play(MATES3, (cpp("int main(){D;}"), "stop"), evaluation=True)
assert len(out) == 1 and close(out[0].reward, 1.0) and out[0].metadata["ft_solo"] == 0.0
out = play([(cpp("int main(){B;}"), "stop")], ("unused", "stop"), evaluation=True, mode="solo")
assert len(out) == 1 and close(out[0].reward, sB) and out[0].metadata["ft_solo"] == 1.0
assert all(c[0] == "mate" for c in CALLS) and len(CALLS) == 1
print("[ok] team eval and solo eval")

# 7) solo reduction (MATES = 0): one training sample on the solo prompt
F.MATES = 0
out = play([(cpp("int main(){C;}"), "stop")], ("unused", "stop"))
F.MATES = 3
assert len(out) == 1 and close(out[0].reward, sC) and out[0].metadata["ft_solo"] == 1.0
prompt_ids = out[0].tokens[: len(out[0].tokens) - out[0].response_length]
assert prompt_ids == tok.encode(tok.apply_chat_template(PROBLEM, tokenize=False, add_generation_prompt=True),
                                add_special_tokens=False), "solo must see exactly the solo prompt"
print("[ok] solo reduction")

# 8) rollout log helper
m = F._ft_means(play(MATES3, (cpp("int main(){D;}"), "stop")))
assert close(m["ft_lead_score"], 1.0) and not math.isnan(m.get("ft_mate_corr", 0.0))
print("[ok] metrics", {k: round(v, 3) for k, v in m.items()})
print("ALL OK")
