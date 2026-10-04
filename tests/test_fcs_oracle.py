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
Q = {"plan": [], "final": [], "sub1": [], "sub2": [], "sub3": [], "sub4": [], "solo": [], "revise": []}
TASKS = {"Try greedy A.": "sub1", "Try DP B.": "sub2", "Try C.": "sub3", "Try E.": "sub4"}
CALLS = []


async def fake_post(url, payload, headers=None):
    text = tok.decode(payload["input_ids"])
    last_user = text.rsplit("<|im_start|>user", 1)[-1]
    if "Your subagents reported back" in last_user:
        kind = "final"
    elif "Write the 4 tasks now" in last_user:
        kind = "plan"
    elif "Your team lead gave you this task" in text:  # every turn of a subagent: which one, from its task
        kind = next((v for k, v in TASKS.items() if k in text), None) or \
            next(k for k in ("sub1", "sub2", "sub3", "sub4") if Q[k])  # identical NO_TASK prompts: in turn
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
    msg = ("/tmp/fcsj_x/sol.cpp: In function 'int main()':\n/tmp/fcsj_x/sol.cpp:1:12: error: 'X' was not declared in "
           "this scope\n    1 | int main(){X;}\n      |            ^\n/usr/include/c++/13/bits/stl_algo.h:9:1: note: "
           "candidate: template<...>\n" + "  required from here\n" * 50) if status == "compile error" else ""
    return {"score": sum(cases) / len(cases), "cases": cases, "status": status, "infra_error": False, "msg": msg}


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
    for j, replies in enumerate(queues.get("sub", [])):  # sub=[one reply per subagent] or [[replies], ...]
        Q[f"sub{j + 1}"][:] = replies if isinstance(replies, list) else [replies]
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


def rep_code(body, report="see my code"):
    """A subagent reply that tests a program and reports in the same turn."""
    return (f"thinking\n</think>\n\n<report>{report}</report>\n```cpp\n{body}\n```", "stop")


SUBS = [rep_code("int main(){A;}"), rep_code("int main(){B;}"), rep_code("int main(){C;}"), rep_code("int main(){E;}")]
sA, sB, sC, sD, sE = (SC[f"int main(){{{x};}}"] for x in "ABCDE")

# 1) reward tables: the team's outcome is the lead's final S
rec = {"game": "team", "S": 0.6, "tests": [[0.2], [0.1, 0.5], [], [0.3]], "S_minus": [0.6, 0.4, 0.6, 0.5]}
assert O.rewards("team", "shared", rec) == [0.6] * 6
assert all(close(a, b) for a, b in zip(O.rewards("team", "bonus", rec), [0.6, 0.6, 0.7, 0.85, 0.6, 0.75]))
assert all(close(a, b) for a, b in zip(O.rewards("team", "diff", rec), [0.6, 0.6, 0.0, 0.2, 0.0, 0.1]))
try:
    O.rewards("team", "diff", {**rec, "S_minus": None})
    raise SystemExit("diff without counterfactuals must fail")
except AssertionError:
    pass
seqr = {"game": "seq", "s": [0.2, 0.1, 0.5, 0.5, 0.3]}
assert O.rewards("seq", "shared", seqr) == [0.3] * 5  # the last submission, not the best
assert all(close(a, b) for a, b in zip(O.rewards("seq", "diff", seqr), [0.2, 0.0, 0.3, 0.0, 0.0]))
assert all(close(a, b) for a, b in zip(O.rewards("par", "diff", {"game": "par", "s": [0.2, 0.7, 0.1, 0.7, 0.3]}),
                                       [0.0, 0.0, 0.0, 0.0, 0.0]))
assert O.rewards("par", "shared", {"game": "par", "s": [0.2, 0.7]}) == [0.7, 0.7]
print("[ok] reward tables")

# 2) task parsing and the last-block rule
t = O.parse_tasks("<task 1>in thinking</task></think>\n<task 2> B </task><task 9>x</task>", 4)
assert t == [O.NO_TASK, "B", O.NO_TASK, O.NO_TASK], t
assert O.last_block("```cpp\nint a;\n``` then ```cpp\nint b;\n```") == "int b;"
assert O.last_block("x</think>no code here, just ideas") == ""
assert O.OUTPUT_RULE not in O._without_output_rule([{"role": "user", "content": "a " + O.OUTPUT_RULE + " b"}])[0]["content"]
print("[ok] task parsing, last block")

# 3) team episode (shared): plan -> subagents test and report -> lead final; outcome = the final only
out = play(plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[(cpp("int main(){D;}"), "stop")])
assert len(out) == O.n_samples("team") == 18, len(out)
real = [o for o in out if not o.metadata.get("fo_pad")]
assert [o.metadata["fo_role"] for o in real] == [0, 1, 2, 3, 4, 5], [o.metadata["fo_role"] for o in real]
assert [o.group_index for o in real] == [40, 41, 42, 42, 42, 42] and all(o.group_index == 47 for o in out[6:])
assert all(close(o.reward, 1.0) for o in real) and all(any(o.loss_mask) for o in real)
info = out[0].metadata
assert close(info["fo_V"], 1.0) and close(info["fo_S"], 1.0) and close(info["fo_best_test"], max(sA, sB, sC, sE))
assert info["fo_beat"] == 1.0 and info["fo_sub_tests"] == 1.0 and info["fo_sub_notest"] == 0.0
plan_text = [c[1] for c in CALLS if c[0] == "plan"][0]
assert O.OUTPUT_RULE not in plan_text and "only your final submission counts" in plan_text
sub_texts = [c[1] for c in CALLS if c[0].startswith("sub")]
assert all("You do not have to write a full solution" in x for x in sub_texts)
assert not any("Try greedy A." in x and "Try DP B." in x for x in sub_texts)
final_text = [c[1] for c in CALLS if c[0] == "final"][0]
assert "Programs tested: 1 (scores: 37.50)\nLast tested program: done, score 37.50/100" in final_text
assert "Per-test-case scores: 0.00 1.00 0.00 0.50" in final_text and "Report:\nsee my code" in final_text
assert "Task: Try E." in final_text and "<task 1>Try greedy A.</task>" in final_text
print(f"[ok] team episode: {len(real)} real samples, groups {[o.group_index for o in real]}")

# 3b) a subagent tests, sees its result, revises, then reports without code; another only gives ideas
out = play(plan=[(PLAN_TXT, "stop")],
           sub=[[(cpp("int main(){X;}"), "stop"), (cpp("int main(){A;}"), "stop"),
                 ("ok\n</think>\n\nA works: greedy by deadline. <report>use A</report>", "stop")],
                [("t\n</think>\n\nIdea: B is a DP over subsets; check n <= 20 first.", "stop")],
                [rep_code("int main(){C;}")], [rep_code("int main(){E;}")]],
           final=[("t\n</think>\n\n<adopt 1/>", "stop")])
s1 = [c[1] for c in CALLS if c[0] == "sub1"]
assert len(s1) == 3 and "Your program was judged: compile error" in s1[1] and "You can test 2 more" in s1[1]
assert "Your program was judged: done, score 37.50/100" in s1[2] and "You can test 1 more" in s1[2]
real = [o for o in out if not o.metadata.get("fo_pad")]
assert [o.metadata["fo_role"] for o in real] == [0, 1, 2, 2, 2, 3, 4, 5], [o.metadata["fo_role"] for o in real]
info = out[0].metadata
assert info["fo_adopt"] == 1.0 and close(info["fo_S"], sA) and close(info["fo_sub_tests"], 4 / 4)
assert close(info["fo_sub_notest"], 0.25) and out[0].metadata["fo"]["tests"] == [[0.0, sA], [], [sC], [sE]]
final_text = [c[1] for c in CALLS if c[0] == "final"][0]
assert "### Subagent 2\nTask: Try DP B.\nPrograms tested: 0\nReport:\nIdea: B is a DP" in final_text
assert "Programs tested: 2 (scores: 0.00, 37.50)" in final_text and "Report:\nuse A" in final_text
print("[ok] multi-turn subagent (test, see result, revise, report) and an ideas-only subagent")

# 3c) the test cap: after SUB_TESTS programs the subagent gets one report-only turn
out = play(plan=[(PLAN_TXT, "stop")],
           sub=[[(cpp("int main(){C;}"), "stop"), (cpp("int main(){B;}"), "stop"), (cpp("int main(){A;}"), "stop"),
                 ("r\n</think>\n\n<report>B is best so far</report>\n```cpp\nint main(){D;}\n```", "stop")],
                SUBS[1], SUBS[2], SUBS[3]],
           final=[(cpp("int main(){B;}"), "stop")])
s1 = [c[1] for c in CALLS if c[0] == "sub1"]
assert "You have no tests left" in s1[3] and out[0].metadata["fo"]["tests"][0] == [sC, sB, sA]  # D not judged
print("[ok] test cap and report-only turn")

# 4) arms: bonus, and diff with counterfactual finals (never trained on)
O.REWARD = "bonus"
out = play(plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[(cpp("int main(){C;}"), "stop")])
real = [o for o in out if not o.metadata.get("fo_pad")]
assert [round(o.reward, 4) for o in real] == [round(x, 4) for x in [sC, sC, sC + 0.5 * sA, sC + 0.5 * sB,
                                                                       sC + 0.5 * sC, sC + 0.5 * sE]]
O.REWARD = "diff"
out = play(plan=[(PLAN_TXT, "stop")], sub=SUBS,
           final=[(cpp("int main(){D;}"), "stop")] + [(cpp(f"int main(){{{x};}}"), "stop") for x in "DACD"])
real = [o for o in out if not o.metadata.get("fo_pad")]
assert out[0].metadata["fo"]["S_minus"] == [1.0, sA, sC, 1.0]
assert [round(o.reward, 4) for o in real] == [round(x, 4) for x in [1.0, 1.0, 0.0, 1 - sA, 1 - sC, 0.0]]
cf = [c[1] for c in CALLS if c[0] == "final"][1:]
assert len(cf) == 4 and "### Subagent 2\n(this subagent's work is not available)" in cf[1]
assert "Try DP B." not in cf[1].split("Your subagents reported back")[-1] and out[0].metadata["fo_cf_tokens"] > 0
assert len(real) == 6  # the counterfactual turns are not samples
O.REWARD = "shared"
print("[ok] bonus / diff arms; counterfactual finals not trained")

# 4b) adoption reuses the tested result (no second judge call); a fence-less final falls back to the raw answer
JUDGED = []
_judge = O.judge_code


async def counting_judge(problem_dir, code):
    JUDGED.append(code)
    return await fake_judge(problem_dir, code)


O.judge_code = counting_judge
out = play(plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[("ok\n</think>\n\n<adopt 1/>", "stop")])
assert len(JUDGED) == 4 and close(out[0].metadata["fo_S"], sA), JUDGED
JUDGED.clear()
out = play(plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[("ok\n</think>\n\nint main(){D;}", "stop")])
assert JUDGED[-1] == "int main(){D;}" and close(out[0].metadata["fo_S"], 1.0), JUDGED[-1]
O.judge_code = _judge
print("[ok] adopted result reused; fence-less fallback")

# 4c) a judge infrastructure error aborts a training episode (miles resubmits the group); eval keeps the flagged 0
async def infra_judge(problem_dir, code):
    r = await fake_judge(problem_dir, code)
    return {**r, "infra_error": "D;" in code}


O.judge_code = infra_judge
out = play(plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[(cpp("int main(){D;}"), "stop")])
assert len(out) == 1 and out[0].status == Sample.Status.ABORTED, [o.status for o in out]
out = play(evaluation=True, game="team", plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[(cpp("int main(){D;}"), "stop")])
assert len(out) == 1 and out[0].metadata["fo_judge_infra_error"] == 1.0 and out[0].status != Sample.Status.ABORTED
O.judge_code = _judge
print("[ok] judge infra error aborts training episodes, flags eval")

# 5) a cut subagent reports "cut off"; a cut plan gives every subagent NO_TASK
out = play(plan=[("still thinking " * 30, "length")], sub=[SUBS[0], ("x " * 50, "length"), SUBS[2], SUBS[3]],
           final=[(cpp("int main(){A;}"), "stop")])
final_text = [c[1] for c in CALLS if c[0] == "final"][0]
assert "### Subagent 2\nTask: " + O.NO_TASK + "\nPrograms tested: 0\nReport:\n(cut off" in final_text
assert out[0].metadata["fo_plan_cut"] == 1.0 and out[0].metadata["fo_tasks_ok"] == 0.0
assert "(cut off: the answer hit the token limit)" in final_text and close(out[0].metadata["fo_sub_cut"], 0.25)
print("[ok] cut plan / cut subagent")

# 5b) the trace file keeps every turn's full text, cut flag and judged status
import glob, json  # noqa: E402
_td = tempfile.mkdtemp()
O.TRACE_DIR, O.TRACE_EVERY, O._trace_count = _td, 1, 0
play(plan=[("still thinking " * 30, "length")], sub=[SUBS[0], ("x " * 50, "length"), SUBS[2], SUBS[3]],
     final=[(cpp("int main(){A;}"), "stop")])
O.TRACE_DIR = ""
tr = [json.loads(l) for f in glob.glob(_td + "/*.jsonl") for l in open(f)][-1]
roles = [t["role"] for t in tr["turns"]]
assert roles == ["plan", "sub1", "sub2", "sub3", "sub4", "final"], roles
assert tr["turns"][0]["cut"] and not tr["turns"][0]["closed_think"] and tr["turns"][0]["text"].count("still thinking") == 30
assert tr["turns"][2]["cut"] and tr["turns"][2]["status"] is None and tr["turns"][1]["status"] == "done"
assert tr["turns"][5]["closed_think"] and "int main(){A;}" in tr["turns"][5]["text"] and tr["turns"][0]["tasks"]
print("[ok] trace keeps full texts, cut flags and statuses")

# 5c) a compile error shows every compiler error (path shortened, source line and caret), without notes
out = play(plan=[(PLAN_TXT, "stop")], sub=[rep_code("int main(){X;}"), SUBS[1], SUBS[2], SUBS[3]],
           final=[(cpp("int main(){A;}"), "stop")])
final_text = [c[1] for c in CALLS if c[0] == "final"][0]
assert ("Last tested program: compile error, score 0.00/100\nCompiler errors:\nsol.cpp:1:12: error: 'X' was not "
        "declared in this scope\n    1 | int main(){X;}\n      |            ^\nPer-test-case") in final_text, final_text[-900:]
assert "note:" not in final_text and "required from here" not in final_text and "/tmp/fcsj" not in final_text
from miles_team.fcs_rm import judge_full  # noqa: E402
from miles_team.fcs_judge import FCS_ROOT  # noqa: E402
r = judge_full(str(FCS_ROOT / "problems" / "0"), "#include <vector>\nint main(){ std::vector<int> v; v.push(1); "
               "return y; }\n")
shown = O.fmt_result(r)
assert r["status"] == "compile error" and shown.count(" error: ") == 2 and "sol.cpp:2:" in shown and "^" in shown, shown
print("[ok] compile errors in the feedback")

# 6) seq (shared): 5 rounds in one chat, each round sees its result, per-case scores and the best so far
O.GAME = "seq"
out = play(solo=[(cpp("int main(){A;}"), "stop")],
           revise=[(cpp("int main(){C;}"), "stop"), (cpp("int main(){B;}"), "stop"), (cpp("int main(){D;}"), "stop"),
                   (cpp("int main(){E;}"), "stop")])
assert len(out) == 5 and [o.group_index for o in out] == [40, 41, 42, 43, 44], [o.group_index for o in out]
assert all(close(o.reward, sE) for o in out)  # shared on seq: every round gets the LAST submission's score
m = out[0].metadata
assert [round(m[f"fo_V_at{t}"], 4) for t in range(1, 6)] == [sA, sA, sB, 1.0, 1.0] and close(m["fo_S"], sE)
assert close(m["fo_V"], sE)
rev = [c[1] for c in CALLS if c[0] == "revise"]
assert "Your submission was judged: done, score 37.50/100\n" in rev[0] and "0.50 0.50 0.00 0.00" in rev[1]
assert "best score so far is 37.50/100" in rev[1] and "```cpp\nint main(){A;}\n```" in rev[0]
assert "only the last ```cpp block is submitted" in rev[0]
assert "thinking..." not in rev[0].split("Your submission was judged")[0].split("<|im_start|>assistant")[-1]
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
assert len(out) == 1 and close(out[0].reward, sC) and "Your subagents reported back" in tok.decode(out[0].tokens)
out = play(evaluation=True, game="par", solo=[(cpp(f"int main(){{{x};}}"), "stop") for x in "ABCAB"])
assert len(out) == 1 and close(out[0].reward, max(sA, sB, sC)) and close(out[0].metadata["fo_V_at3"], max(sA, sB, sC))
print("[ok] eval samples")

# 9) relabel: an episode generated under shared, relabeled for every team arm, equals that arm's rewards; padding
#    is left alone; the dump / replay round trip keeps the record
O.CF = True
ep = play(plan=[(PLAN_TXT, "stop")], sub=SUBS,
          final=[(cpp("int main(){C;}"), "stop")] + [(cpp(f"int main(){{{x};}}"), "stop") for x in "CABC"])
O.CF = False
real = [o for o in ep if not o.metadata.get("fo_pad")]
for arm in O.ARMS["team"]:
    want = O.rewards("team", arm, ep[0].metadata["fo"])
    assert [O.relabel(o.metadata["fo"], arm) for o in real] == [want[o.metadata["fo_role"]] for o in real], arm
a = Args(reward_key=None)
O.REWARD = "bonus"
raw, rew = O.post_process(a, real)
want = O.rewards("team", "bonus", ep[0].metadata["fo"])
assert rew == [want[o.metadata["fo_role"]] for o in real] and raw == rew
O.REWARD = "shared"
from miles.ray.rollout.debug_data import load_replay_rollout_data, save_debug_rollout_data  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    da = Args(save_debug_rollout_data=td + "/rd/{rollout_id}.pt", save_debug_trajectory_data=None,
              replay_rollout_data=td + "/rd/{rollout_id}.pt")
    save_debug_rollout_data(da, ep, rollout_id=0, evaluation=False, metadata={})
    back, _ = load_replay_rollout_data(da, rollout_id=0)
assert [b.metadata.get("fo") for b in back] == [o.metadata.get("fo") for o in ep] and O.post_process(a, back) == O.post_process(a, ep)
print("[ok] relabel and dump / replay")

# 10) metrics helper
m = O._means(ep)
assert close(m["fo_V"], sC) and "fo_role" not in m and "fo_pad" not in m
print("[ok] metrics", {k: round(v, 3) for k, v in sorted(m.items())})
print("ALL OK")
