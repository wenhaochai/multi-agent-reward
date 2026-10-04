"""Offline check of miles_team.fcs_oracle (games with an oracle: team = lead plans + 4 subagents + lead final, seq,
par) against a scripted fake SGLang and a fake judge (no GPU): reward tables, task parsing, the feedback the lead and
the sequential agent see (scores and per-case scores), one sample per turn with its group, padding, eval samples,
relabel and the dump/replay round trip. Runs in miles.sif with the real Qwen3.5 tokenizer:
  PYTHONPATH=/root/Megatron-LM:<miles src>:<this repo> python3 tests/test_fcs_oracle.py
"""
import asyncio
import os
import tempfile
from types import SimpleNamespace

os.environ.update(MA_FO_GAME="team", MA_FO_REWARD="shared", MA_FO_TRACE_DIR="", MA_FO_BUDGET="4096",
                  MA_FO_PLAN_BUDGET="1024")
MODEL = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/Qwen3.5-9B-FCS-SFT-lm"

from transformers import AutoTokenizer  # noqa: E402

from miles.rollout.base_types import GenerateFnInput  # noqa: E402
from miles.utils.types import Sample  # noqa: E402

import miles_team.team_rollout as TR  # noqa: E402
import miles_team.fcs_oracle as O  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
EOS = tok.eos_token_id
PROBLEM = [{"role": "user", "content": "You are a competitive programmer. Solve the following problem in C++. Output "
            "ONLY the C++ code wrapped in ```cpp and ```. No explanation.\n\nPrint 42.\n\nGenerate solution code:"}]


def cpp(body):
    return f"thinking...\n</think>\n\n```cpp\n{body}\n```"


CASES = {"int main(){A;}": [1.0, 0.0, 0.0, 0.5], "int main(){B;}": [0.0, 1.0, 0.0, 0.5],
         "int main(){C;}": [0.5, 0.5, 0.0, 0.0], "int main(){D;}": [1.0, 1.0, 1.0, 1.0],
         "int main(){E;}": [0.0, 0.0, 1.0, 0.0]}
SC = {k: sum(v) / len(v) for k, v in CASES.items()}
PLAN_TXT = ("plan\n</think>\n\n<task 1>Try greedy A.</task>\n<task 2>Try DP B.</task>\n<task 3>Try C.</task>\n"
            "<task 4>Try E.</task>")
Q = {"plan": [], "final": [], "sub": [], "solo": [], "revise": []}
CALLS = []


async def fake_post(url, payload, headers=None):
    text = tok.decode(payload["input_ids"])
    last_user = text.rsplit("<|im_start|>user", 1)[-1]
    if "submissions were judged" in last_user:
        kind = "final"
    elif "Write the 4 tasks now" in last_user:
        kind = "plan"
    elif "Your team lead gave you this task" in last_user:
        kind = "sub"
    elif "Write an improved solution" in last_user:
        kind = "revise"
    else:
        kind = "solo"
    reply, finish = Q[kind].pop(0)
    ids = tok.encode(reply, add_special_tokens=False)
    if finish == "stop":
        ids = ids + [EOS]
    CALLS.append((kind, text, len(payload["input_ids"])))
    return {"text": reply, "meta_info": {"finish_reason": {"type": finish}, "completion_tokens": len(ids),
                                         "output_token_logprobs": [[-0.5, i, None] for i in ids]}}


async def fake_judge(problem_dir, code):
    c = code.strip()
    cases = CASES.get(c, [0.0] * 4)
    status = "done" if c in CASES else ("no code" if not code else "compile error")
    return {"score": sum(cases) / len(cases), "cases": cases, "status": status, "infra_error": False}


TR.post = fake_post
O.judge_code = fake_judge


class Args(SimpleNamespace):
    def __getattr__(self, name):
        return False


args = Args(sglang_router_ip="127.0.0.1", sglang_router_port=1, rollout_max_response_len=4096,
            rollout_max_context_len=65536, use_rollout_routing_replay=False, use_rollout_indexer_replay=False,
            sglang_speculative_algorithm=None, reward_key=None, lora_rank=0, lora_adapter_path=None,
            return_sampling_mask=False)


def play(evaluation=False, game=None, group=5, **queues):
    for k in Q:
        Q[k][:] = list(queues.get(k, []))
    CALLS.clear()
    md = {"messages": PROBLEM}
    if game:
        md["fo_game"] = game
    sample = Sample(index=11, group_index=group, prompt="(templated)", label="/nonexistent/problem", metadata=md)
    inp = GenerateFnInput(state=SimpleNamespace(args=args, tokenizer=tok), sample=sample,
                          sampling_params={"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 4096},
                          evaluation=evaluation)
    out = asyncio.run(O.generate(inp)).samples
    assert all(not v for v in Q.values()), {k: len(v) for k, v in Q.items()}  # every scripted reply was used
    return out if isinstance(out, list) else [out]


close = lambda a, b: abs(a - b) < 1e-9  # noqa: E731
SUBS = [(cpp("int main(){A;}"), "stop"), (cpp("int main(){B;}"), "stop"), (cpp("int main(){C;}"), "stop"),
        (cpp("int main(){E;}"), "stop")]
sA, sB, sC, sD, sE = (SC[f"int main(){{{x};}}"] for x in "ABCDE")

# 1) reward tables
rec = {"game": "team", "s": [0.2, 0.5, 0.1, 0.3], "S": 0.6}
assert O.rewards("team", "shared", rec) == [0.6] * 6
assert O.rewards("team", "indiv", rec) == [0.5, 0.6, 0.2, 0.5, 0.1, 0.3]
assert all(close(a, b) for a, b in zip(O.rewards("team", "gain", rec), [0.6, 0.1, 0.6, 0.6, 0.6, 0.6]))
d = O.rewards("team", "diff", rec)  # S = 0.6 is the max: no subagent is pivotal
assert close(d[1], 0.1) and all(close(x, 0.0) for x in d[2:]) and d[0] == 0.6, d
rec2 = {"game": "team", "s": [0.2, 0.7, 0.1, 0.3], "S": 0.4}  # subagent 2 is pivotal by 0.3
d = O.rewards("team", "diff", rec2)
assert close(d[1], 0.0) and close(d[3], 0.3) and close(d[2], 0.0), d
assert O.rewards("seq", "diff", {"game": "seq", "s": [0.2, 0.1, 0.5, 0.5, 0.6]}) == [0.2, 0.0, 0.3, 0.0, 0.1] or \
    all(close(a, b) for a, b in zip(O.rewards("seq", "diff", {"game": "seq", "s": [0.2, 0.1, 0.5, 0.5, 0.6]}),
                                    [0.2, 0.0, 0.3, 0.0, 0.1]))
assert all(close(a, b) for a, b in zip(O.rewards("par", "diff", {"game": "par", "s": [0.2, 0.7, 0.1, 0.7, 0.3]}),
                                       [0.0, 0.0, 0.0, 0.0, 0.0]))
assert O.rewards("par", "indiv", {"game": "par", "s": [0.2, 0.7]}) == [0.2, 0.7]
print("[ok] reward tables")

# 2) task parsing: tags in the reasoning do not count; missing tasks get NO_TASK
t = O.parse_tasks("<task 1>in thinking</task></think>\n<task 2> B </task><task 9>x</task>", 4)
assert t == [O.NO_TASK, "B", O.NO_TASK, O.NO_TASK], t
print("[ok] task parsing")

# 3) team episode (shared): lead plan -> 4 subagents with their tasks -> lead final with scores and per-case scores
out = play(plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[(cpp("int main(){D;}"), "stop")])
assert len(out) == 6, len(out)
assert [o.metadata["fo_role"] for o in out] == [0, 1, 2, 3, 4, 5]
assert [o.group_index for o in out] == [40, 41, 42, 42, 42, 42], [o.group_index for o in out]
assert all(close(o.reward, 1.0) for o in out) and all(any(o.loss_mask) for o in out)
info = out[0].metadata
assert close(info["fo_V"], 1.0) and close(info["fo_S"], 1.0) and close(info["fo_best_sub"], max(sA, sB, sC, sE))
assert info["fo_beat"] == 1.0 and info["fo_tasks_ok"] == 1.0 and info["fo_tokens"] > info["fo_latency"] > 0
sub_texts = [c[1] for c in CALLS if c[0] == "sub"]
assert any("Try DP B." in x for x in sub_texts) and not any("Try greedy A." in x and "Try DP B." in x for x in sub_texts)
final_text = [c[1] for c in CALLS if c[0] == "final"][0]
assert "Result: done, score 37.50/100" in final_text and "Per-test-case scores: 0.00 1.00 0.00 0.50" in final_text
assert "Task: Try E." in final_text and "currently 37.50/100" in final_text and "int main(){B;}" in final_text
assert "<task 1>Try greedy A.</task>" in final_text  # the lead's own plan answer is in its context
print(f"[ok] team episode: groups {[o.group_index for o in out]}; lead turns in "
      f"{len({id(o) for o in out[:2]})} segments")

# 4) arms on the same episode: indiv / gain / diff, and adoption (S = s_j, no gain)
O.REWARD = "indiv"
out = play(plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[("hmm\n</think>\n\n<adopt 2/>", "stop")])
assert out[0].metadata["fo_adopt"] == 1.0 and close(out[0].metadata["fo_S"], sB) and out[0].metadata["fo_beat"] == 0.0
assert [round(o.reward, 4) for o in out] == [round(x, 4) for x in [max(sA, sB, sC, sE), sB, sA, sB, sC, sE]]
O.REWARD = "diff"
out = play(plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[(cpp("int main(){C;}"), "stop")])
best = max(sA, sB, sC, sE)  # A and B tie at 0.375: neither is pivotal
assert close(out[1].reward, 0.0) and all(close(o.reward, 0.0) for o in out[2:]), [o.reward for o in out]
O.REWARD = "shared"
print("[ok] indiv / diff arms and adoption")

# 5) a subagent cut off shows as no code; a plan cut off gives every subagent NO_TASK
out = play(plan=[("still thinking " * 30, "length")], sub=[SUBS[0], ("x " * 50, "length"), SUBS[2], SUBS[3]],
           final=[(cpp("int main(){A;}"), "stop")])
final_text = [c[1] for c in CALLS if c[0] == "final"][0]
assert "### Subagent 2\nTask: " + O.NO_TASK + "\nResult: no code" in final_text
assert out[0].metadata["fo_plan_cut"] == 1.0 and out[0].metadata["fo_tasks_ok"] == 0.0 and len(out) == 6
assert "(cut off: the answer hit the token limit)" in final_text
print("[ok] cut plan / cut subagent")

# 6) seq (shared): 5 rounds in one chat, each round sees its result, per-case scores and the best so far
O.GAME = "seq"
out = play(solo=[(cpp("int main(){A;}"), "stop")],
           revise=[(cpp("int main(){C;}"), "stop"), (cpp("int main(){B;}"), "stop"), (cpp("int main(){D;}"), "stop"),
                   (cpp("int main(){E;}"), "stop")])
assert len(out) == 5 and [o.group_index for o in out] == [40, 41, 42, 43, 44], [o.group_index for o in out]
assert all(close(o.reward, 1.0) for o in out)
m = out[0].metadata
assert [round(m[f"fo_V_at{t}"], 4) for t in range(1, 6)] == [sA, sA, sB, 1.0, 1.0] and close(m["fo_S"], sE)
rev = [c[1] for c in CALLS if c[0] == "revise"]
assert "Your submission was judged: done, score 37.50/100." in rev[0] and "0.50 0.50 0.00 0.00" in rev[1]
assert "best score so far is 37.50/100" in rev[1] and "int main(){A;}" in rev[0]
print(f"[ok] seq: 5 samples, groups {[o.group_index for o in out]}, V@t {[round(m[f'fo_V_at{t}'], 3) for t in range(1, 6)]}")

# 7) par (indiv): 5 independent attempts on the solo prompt, each its own score
O.GAME, O.REWARD = "par", "indiv"
out = play(solo=[(cpp(f"int main(){{{x};}}"), "stop") for x in "ABCDE"])
assert len(out) == 5 and all(o.group_index == 40 for o in out)
assert sorted(round(o.reward, 4) for o in out) == sorted(round(x, 4) for x in [sA, sB, sC, sD, sE])
assert close(out[0].metadata["fo_V"], 1.0) and all(c[0] == "solo" for c in CALLS)
O.GAME, O.REWARD = "team", "shared"
print("[ok] par")

# 8) eval: one sample per episode with reward V; an eval set picks its game with metadata.fo_game
out = play(evaluation=True, game="team", plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[(cpp("int main(){C;}"), "stop")])
assert len(out) == 1 and close(out[0].reward, max(sA, sB, sC, sE)) and "submissions were judged" in tok.decode(out[0].tokens)
out = play(evaluation=True, game="par", solo=[(cpp(f"int main(){{{x};}}"), "stop") for x in "ABCAB"])
assert len(out) == 1 and close(out[0].reward, max(sA, sB, sC)) and close(out[0].metadata["fo_V_at3"], max(sA, sB, sC))
print("[ok] eval samples")

# 9) relabel: an episode generated under shared, relabeled for every team arm, equals that arm's rewards; padding
#    is left alone; the dump / replay round trip keeps the record
ep = play(plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[(cpp("int main(){C;}"), "stop")])
for arm in O.ARMS["team"]:
    assert [O.relabel(o.metadata["fo"], arm) for o in ep] == O.rewards("team", arm, ep[0].metadata["fo"]), arm
a = Args(reward_key=None)
O.REWARD = "gain"
raw, rew = O.post_process(a, ep)
assert rew == O.rewards("team", "gain", ep[0].metadata["fo"]) and raw == rew
O.REWARD = "shared"
from miles.ray.rollout.debug_data import load_replay_rollout_data, save_debug_rollout_data  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    da = Args(save_debug_rollout_data=td + "/rd/{rollout_id}.pt", save_debug_trajectory_data=None,
              replay_rollout_data=td + "/rd/{rollout_id}.pt")
    save_debug_rollout_data(da, ep, rollout_id=0, evaluation=False, metadata={})
    back, _ = load_replay_rollout_data(da, rollout_id=0)
assert [b.metadata["fo"] for b in back] == [o.metadata["fo"] for o in ep] and O.post_process(a, back) == O.post_process(a, ep)
print("[ok] relabel and dump / replay")

# 10) metrics helper
m = O._means(ep)
assert close(m["fo_V"], max(sA, sB, sC, sE)) and "fo_role" not in m and "fo_pad" not in m
print("[ok] metrics", {k: round(v, 3) for k, v in sorted(m.items())})
print("ALL OK")
