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

# 9) mixed team sizes (K_SET 0,1,3): K from sample.index, always 1 + MATES samples, groups split by K, fc_* metadata
import miles_team.fcs_cost as FC  # noqa: E402

F.K_SET = [0, 1, 3]
eps = []
for idx, mates, lead in [(12, [MATES3[2]], ("unused", "stop")),            # 12 % 3 = 0 -> K = 0
                         (13, MATES3[:1], (cpp("int main(){D;}"), "stop")),  # K = 1
                         (14, MATES3, (cpp("int main(){D;}"), "stop"))]:     # K = 3
    MATE_REPLIES[:] = list(mates)
    LEAD_REPLY[0] = lead
    sample = Sample(index=idx, group_index=5, prompt="(templated)", label="/nonexistent/problem",
                    metadata={"messages": PROBLEM, "pid": "t"})
    inp = GenerateFnInput(state=SimpleNamespace(args=args, tokenizer=tok), sample=sample,
                          sampling_params={"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 4096}, evaluation=False)
    eps.append(asyncio.run(F.generate(inp)).samples)
k0, k1, k3 = eps
assert [len(e) for e in eps] == [4, 4, 4]
assert [o.metadata["ft_role"] for o in k0] == ["lead", "pad", "pad", "pad"]
assert [o.metadata["ft_role"] for o in k1] == ["lead", "mate1", "pad", "pad"]
assert [o.group_index for o in k0] == [40, 47, 47, 47] and [o.group_index for o in k1] == [41, 44, 47, 47]
assert [o.group_index for o in k3] == [42, 45, 45, 45], [o.group_index for o in k3]
assert all(o.remove_sample and not any(o.loss_mask) and o.metadata.get("fc_pad") for e in eps for o in e
           if o.metadata["ft_role"] == "pad")
assert k0[0].metadata["ft_solo"] == 1.0 and close(k0[0].reward, sC) and k0[0].metadata["fc_k"] == 0
assert all(not any(key.startswith("fc_") and key != "fc_pad" for key in o.metadata) for o in k0[1:])
assert k3[0].metadata["fc_lead"] and not k3[1].metadata["fc_lead"] and k3[1].metadata["fc_cost"] == k3[0].metadata["ft_tokens"]
flat = [o for e in eps for o in e]
assert sorted(FC.episodes(flat)) == sorted([(0, sC, float(k0[0].metadata["ft_tokens"])),
                                            (1, 1.0, float(k1[0].metadata["ft_tokens"])),
                                            (3, 1.0, float(k3[0].metadata["ft_tokens"]))])
F.K_SET = []
print("[ok] mixed K: 4 samples each, groups", [[o.group_index for o in e] for e in eps])

# 10) lambda replay: pi_0 from critic-only rollouts; alpha = 1 raises lambda when the cost rises, lowers it when it falls
obs = {0: {3: [1.0, 1000.0, 10]}, 1: {3: [1.0, 1000.0, 10]},     # pi_0: s0 = 0.1, c0 = 100
       2: {3: [1.0, 2000.0, 10]}}                                # cost doubles
r = FC.replay(obs, 1, critic_only=2, alpha=1.0, eta=0.5, lam0=0.02, ema=0.0)
assert r["lam"][3] == 0.02 and close(r["s0"][3], 0.1) and close(r["c0"][3], 100.0) and not r["uv"]
r = FC.replay(obs, 2, critic_only=2, alpha=1.0, eta=0.5, lam0=0.02, ema=0.0)
assert close(r["uv"][3][1], -1.0) and close(r["lam"][3], 0.02 * math.exp(0.5)), r
r = FC.replay({**obs, 2: {3: [1.0, 500.0, 10]}}, 2, critic_only=2, alpha=1.0, eta=0.5, lam0=0.02, ema=0.0)
assert close(r["lam"][3], 0.02 * math.exp(-0.25))
# alpha = 0.5: a score gain of +50% with the cost unchanged raises lambda by exp(0.5 * 0.5 * 0.5)
r = FC.replay({**obs, 2: {3: [1.5, 1000.0, 10]}}, 2, critic_only=2, alpha=0.5, eta=0.5, lam0=0.02, ema=0.0)
assert close(r["lam"][3], 0.02 * math.exp(0.125)), r
r = FC.replay({0: {3: [1.0, 1000.0, 10]}, 1: {3: [1.0, 1e6, 10]}}, 1, critic_only=1, alpha=1.0, eta=50, lam0=0.02,
              ema=0.0)
assert r["lam"][3] == FC.LAM_MAX
print("[ok] lambda replay")

# 11) observe + post_process: cost term on real samples only; a resumed rollout overwrites its stale sums
FC._obs.clear()
FC.STATE = ""
a = Args(num_critic_only_steps=1, reward_key=None)
FC.observe(0, a, flat)
FC.observe(5, a, flat)
FC.observe(1, a, flat)  # resumed at rollout 1: rollout 5's sums are dropped
assert sorted(FC._obs) == [0, 1]
FC.ON = True
raw, rew = FC.post_process(a, flat)
for o, x in zip(flat, rew):
    if o.metadata.get("fc_pad"):
        assert x == o.reward
    else:
        k = o.metadata["fc_k"]
        assert close(x, o.reward - FC._cur["lam"][k] * o.metadata["fc_cost"] / FC._cur["c0"][k]), (x, o.reward)
assert raw == rew
FC.ON = False
assert FC.post_process(a, flat)[0] == [o.reward for o in flat]
print("[ok] observe / post_process")

# 12) relabel (value pretraining shared across arms): an episode generated under one arm, relabeled for every arm,
#     gets exactly the rewards that arm computes; post_process relabels real samples and leaves padding alone
F.REWARD = "shared"
ep = play(MATES3, ("hmm\n</think>\n\n<adopt 2/>", "stop"))  # adopted teammate 2
fr0 = ep[0].metadata["fr"]
assert [o.metadata["fr"]["role"] for o in ep] == [0, 1, 2, 3] and fr0["adopted"] == 2
for arm in F.REWARDS:
    lr, mr = F.rewards(arm, fr0["S"], fr0["s"], fr0["cases"], fr0["adopted"])
    assert [F.relabel(o.metadata["fr"], arm) for o in ep] == [lr] + mr, arm
F.REWARD = "adopt"
raw, rew = F.post_process(a, ep)
assert rew == [fr0["S"], 0.0, fr0["S"], 0.0] and raw == rew, rew
F.REWARD = "shared"
assert F.post_process(a, ep)[1] == [o.reward for o in ep]  # live rollout: reproduces generate()'s rewards
F.K_SET = [0, 1, 3]
FC.ON = False
mixed = [o for e in eps for o in e]
F.REWARD = "indiv"
raw, rew = F.post_process(a, mixed)
for o, x in zip(mixed, rew):
    if o.metadata.get("fc_pad"):
        assert x == o.reward and "fr" not in o.metadata
    else:
        assert x == F.relabel(o.metadata["fr"], "indiv")
F.K_SET = []
F.REWARD = "shared"
print("[ok] relabel across arms and post_process")

# 13) miles' dump / replay round trip keeps the relabel metadata (save_debug_rollout_data -> load_replay_rollout_data)
import tempfile  # noqa: E402

from miles.ray.rollout.debug_data import load_replay_rollout_data, save_debug_rollout_data  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    da = Args(save_debug_rollout_data=td + "/rollout_data/{rollout_id}.pt", save_debug_trajectory_data=None,
              replay_rollout_data=td + "/rollout_data/{rollout_id}.pt")
    save_debug_rollout_data(da, ep, rollout_id=3, evaluation=False, metadata={"x": 1})
    back, md = load_replay_rollout_data(da, rollout_id=3)
assert md == {"x": 1} and len(back) == len(ep)
assert [b.metadata["fr"] for b in back] == [o.metadata["fr"] for o in ep]
assert [b.tokens for b in back] == [o.tokens for o in ep] and [b.loss_mask for b in back] == [o.loss_mask for o in ep]
assert F.post_process(a, back) == F.post_process(a, ep)
print("[ok] dump / replay round trip")
print("ALL OK")
