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
    elif "Write your solution for this round" in last_user:
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


def two(body, report="see my code"):
    """A subagent that tests one program, sees its result, and then reports (a report reply is never judged)."""
    return [(cpp(body), "stop"), (f"r\n</think>\n\n<report>{report}</report>", "stop")]


SUBS = [two("int main(){A;}"), two("int main(){B;}"), two("int main(){C;}"), two("int main(){E;}")]
sA, sB, sC, sD, sE = (SC[f"int main(){{{x};}}"] for x in "ABCDE")

# 1) reward tables: the team's outcome is the lead's final S; bonus pays the best test
rec = {"game": "team", "S": 0.6, "tests": [[0.2], [0.5, 0.1], [], [0.3]], "S_minus": [0.6, 0.4, 0.6, 0.5]}
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
assert O.rewards("par", "shared", {"game": "par", "s": [0.2, 0.7]}) == [0.7, 0.7]
print("[ok] reward tables")

# 2) parsing: tasks (also </task N>), code blocks (line-anchored, cpp-tagged for tests), output rule
t = O.parse_tasks("<task 1>in thinking</task></think>\n<task 2> B </task 2><task 3>C</task><task 9>x</task>", 4)
assert t == [O.NO_TASK, "B", "C", O.NO_TASK], t
assert O.last_block("```cpp\nint a;\n```\nthen\n```cpp\nint b;\n```") == "int b;"
assert O.last_block("x</think>no code here, just ideas") == ""
assert O.last_block("x</think>```text\n3\n```\nThen:\n```cpp\nint main(){}\n```") == "int main(){}"
assert O.last_block("x</think>```cpp\nint main(){}\n```\nTest:\n```\n5\n1 2\n```") == "int main(){}"
assert O.last_block("x</think>Test:\n```\n5\n1 2\n```") == ""  # an untagged block is not a program to test
assert O.extract_cpp("x</think>```cpp\nint main(){A;}\n```\n```\n5\n```") == "int main(){A;}"
assert O.extract_cpp("x</think>```\nint main(){A;}\n```") == "int main(){A;}"  # a final may be untagged
assert O.extract_cpp("x</think>```cpp\nint main(){\n}```") == "int main(){\n}"
assert O.extract_cpp("x</think>```cpp\nint main(){A;}\n```<|im_end|>") == "int main(){A;}"
assert O.last_block("x</think>```cpp\na\n```python\nb\n```") == "a"  # a tagged fence opens the next block
assert O.last_block("x</think>```cpp``` blocks follow:\n```cpp\nint main(){}\n```") == "int main(){}"  # inline span
assert O.last_block("x</think>```cpp17\nA\n```\n```c++17\nB\n```\n```{.cpp}\nC\n```") == "C"
assert O.last_block("x</think>```c\nA\n```") == ""  # C is not C++
assert O.last_block("x</think>```cpp\nfirst\n```cpp\nsecond\n```") == "second"  # an unclosed block, then the next
assert O.extract_cpp("x</think>```cpp\nint main(){A;}\n```\nWait, rewrite:\n```cpp\nint mai") == "int main(){A;}"
assert O.extract_cpp("x</think>```cpp\nint mai") == "int mai"  # an unclosed block counts when it is the only one
assert O.extract_cpp("x</think>```cpp\r\nint main(){}\r\n```") == "int main(){}"
# "longest" is FrontierSmith's extract_cpp exactly (the blind arms run it): compare on hostile inputs
import importlib.util as _iu  # noqa: E402
_spec = _iu.spec_from_file_location("fs_frontiercs", "/scratch/gpfs/GROUP/USER/project/FrontierSmith/verl/verl/utils/reward_score/frontiercs.py")
_fs = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_fs)
HOSTILE = ["```text\nhi\n```\n```cpp\nnew\n```", "x</think>```cpp\na\n```\n```\n5\n```", "no code", "```cpp\nint a;",
           "t</think>```c++\nlong one\n```\n```cpp\nb\n```", "```cpp a```", "", "```\n```", "<think>x</think>\n```cpp\nz\n```"]
assert all(O.extract_cpp(h, "longest") == _fs.extract_cpp(h) for h in HOSTILE), [h for h in HOSTILE if O.extract_cpp(h, "longest") != _fs.extract_cpp(h)]
assert O.OUTPUT_RULE not in O._without_output_rule([{"role": "user", "content": "a " + O.OUTPUT_RULE + " b"}])[0]["content"]
assert O._clip("```cpp\n" + "x" * 50, 20, "code").endswith("(code truncated)\n```")
print("[ok] parsing: tasks, code blocks, clipping")

# 3) team episode (shared): plan -> subagents test, see the result, report -> lead final; outcome = the final only
out = play(plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[(cpp("int main(){D;}"), "stop")])
assert len(out) == O.n_samples("team") == 18, len(out)
real = [o for o in out if not o.metadata.get("fo_pad")]
assert [o.metadata["fo_role"] for o in real] == [0, 1, 2, 2, 3, 3, 4, 4, 5, 5], [o.metadata["fo_role"] for o in real]
assert [o.group_index for o in real] == [40, 41] + [42] * 8 and all(o.group_index == 47 for o in out[10:])
assert all(len(o.tokens) == 2 and o.loss_mask == [0] for o in out[10:])  # 2-token pads
assert all(close(o.reward, 1.0) for o in real) and all(any(o.loss_mask) for o in real)
info = out[0].metadata
assert close(info["fo_V"], 1.0) and close(info["fo_S"], 1.0) and close(info["fo_best_test"], max(sA, sB, sC, sE))
assert info["fo_beat"] == 1.0 and info["fo_sub_tests"] == 1.0 and info["fo_sub_notest"] == 0.0
assert info["fo_final_clamped"] == 0.0 and info["fo_final_prompt_tokens"] > 0
plan_text = [c[1] for c in CALLS if c[0] == "plan"][0]
assert O.OUTPUT_RULE not in plan_text and "only your final submission counts" in plan_text
sub_texts = [c[1] for c in CALLS if c[0].startswith("sub")]
assert all("You do not have to write a full solution" in x for x in sub_texts)
assert not any("Try greedy A." in x and "Try DP B." in x for x in sub_texts)
assert "Your program was judged: done, score 37.50/100" in [c[1] for c in CALLS if c[0] == "sub2"][1]
final_text = [c[1] for c in CALLS if c[0] == "final"][0]
assert "Programs tested: 1 (scores in order: 37.50)\nBest tested program (test 1): done, score 37.50/100" in final_text
assert "Per-test-case scores: 0.00 1.00 0.00 0.50" in final_text and "Report:\nsee my code" in final_text
assert "Task: Try E." in final_text and "<task 1>Try greedy A.</task>" in final_text
print(f"[ok] team episode: {len(real)} real samples, groups {[o.group_index for o in real]}")

# 3b) best test shown and adopted; an ideas-only subagent; a report reply with code is not judged
out = play(plan=[(PLAN_TXT, "stop")],
           sub=[[(cpp("int main(){X;}"), "stop"), (cpp("int main(){A;}"), "stop"), (cpp("int main(){C;}"), "stop"),
                 ("ok\n</think>\n\n<report>A was best; partial idea:</report>\n```cpp\nint main(){E;}\n```", "stop")],
                [("t\n</think>\n\nIdea: B is a DP over subsets; check n <= 20 first.", "stop")],
                two("int main(){C;}"), two("int main(){E;}")],
           final=[("t\n</think>\n\n<adopt 1/>", "stop")])
s1 = [c[1] for c in CALLS if c[0] == "sub1"]
assert len(s1) == 4 and "Your program was judged: compile error" in s1[1] and "You can test 2 more" in s1[1]
assert "You have no tests left" in s1[3]
info = out[0].metadata
assert out[0].metadata["fo"]["tests"] == [[0.0, sA, sC], [], [sC], [sE]]  # the report's code was not judged
assert info["fo_adopt"] == 1.0 and close(info["fo_S"], sA)  # adopt = the BEST test (A), not the last (C)
final_text = [c[1] for c in CALLS if c[0] == "final"][0]
assert "Programs tested: 3 (scores in order: 0.00, 37.50, 25.00)\nBest tested program (test 2)" in final_text
assert "### Subagent 2\nTask: Try DP B.\nPrograms tested: 0\nReport:\nIdea: B is a DP" in final_text
O.REWARD = "bonus"
out = play(plan=[(PLAN_TXT, "stop")], sub=[[(cpp("int main(){A;}"), "stop"), (cpp("int main(){X;}"), "stop"),
                                           ("r\n</think>\n\n<report>r</report>", "stop")], SUBS[1], SUBS[2], SUBS[3]],
           final=[(cpp("int main(){C;}"), "stop")])
assert close(out[2].reward, sC + 0.5 * sA)  # the best test, though the last one failed to compile
O.REWARD = "shared"
out = play(plan=[(PLAN_TXT, "stop")],
           sub=[[("t\n</think>\n\nTesting before my <report>:\n```cpp\nint main(){A;}\n```", "stop"),
                 ("r\n</think>\n\n<report>done</report>", "stop")], SUBS[1], SUBS[2], SUBS[3]],
           final=[(cpp("int main(){C;}"), "stop")])
assert out[0].metadata["fo"]["tests"][0] == [sA]
print("[ok] best test shown, adopted and paid; ideas-only subagent; report replies not judged")

# 4) diff: counterfactual finals are not trained on and cannot adopt the removed subagent
O.REWARD = "diff"
out = play(plan=[(PLAN_TXT, "stop")], sub=SUBS,
           final=[(cpp("int main(){D;}"), "stop")] + [("t\n</think>\n\n<adopt 1/>", "stop"),
                                                     (cpp("int main(){A;}"), "stop"), (cpp("int main(){C;}"), "stop"),
                                                     (cpp("int main(){D;}"), "stop")])
real = [o for o in out if not o.metadata.get("fo_pad")]
assert out[0].metadata["fo"]["S_minus"] == [0.0, sA, sC, 1.0], out[0].metadata["fo"]["S_minus"]
cf = [c[1] for c in CALLS if c[0] == "final"][1:]
assert len(cf) == 4 and "### Subagent 2\n(this subagent's work is not available)" in cf[1]
assert "Try DP B." not in cf[1].split("Your subagents reported back")[-1] and out[0].metadata["fo_cf_tokens"] > 0
assert len(real) == 10 and [round(o.reward, 4) for o in real[:2]] == [1.0, 1.0]
O.REWARD = "shared"
print("[ok] diff: counterfactuals not trained, removed subagent not adoptable")

# 4b) adoption reuses the judged result; an adopt of a subagent with no tests submits nothing (no judge call)
JUDGED = []
_judge = O.judge_code


async def counting_judge(problem_dir, code):
    JUDGED.append(code)
    return await fake_judge(problem_dir, code)


O.judge_code = counting_judge
out = play(plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[("ok\n</think>\n\n<adopt 1/>", "stop")])
assert len(JUDGED) == 4 and close(out[0].metadata["fo_S"], sA), JUDGED
JUDGED.clear()
out = play(plan=[(PLAN_TXT, "stop")], sub=[SUBS[0], [("ideas only", "stop")], SUBS[2], SUBS[3]],
           final=[("ok\n</think>\n\n<adopt 2/>", "stop")])
# the final passes empty code (judge_full answers "no code" without compiling), never the adopt text itself
assert JUDGED[-1] == "" and out[0].metadata["fo_S"] == 0.0 and out[0].metadata["fo_final_bad"] == 1.0, JUDGED
JUDGED.clear()
out = play(plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[("ok\n</think>\n\nint main(){D;}", "stop")])
assert JUDGED[-1] == "int main(){D;}" and close(out[0].metadata["fo_S"], 1.0), JUDGED[-1]
O.judge_code = _judge
print("[ok] adoption reuse; empty adopt not judged; fence-less fallback")

# 4c) a judge infrastructure error aborts a training episode at once (also inside a subagent); eval keeps the 0
async def infra_judge(problem_dir, code):
    r = await fake_judge(problem_dir, code)
    return {**r, "infra_error": "D;" in code or "B;" in code}


O.judge_code = infra_judge
out = play(plan=[(PLAN_TXT, "stop")], sub=[SUBS[0], [(cpp("int main(){B;}"), "stop")], SUBS[2], SUBS[3]])
assert len(out) == 1 and out[0].status == Sample.Status.ABORTED, [o.status for o in out]
for k in Q:
    Q[k].clear()  # cancelled siblings may leave scripted replies unused
O.judge_code = lambda d, c: infra_judge(d, c.replace("B;", "A;"))
out = play(evaluation=True, game="team", plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[(cpp("int main(){D;}"), "stop")])
assert len(out) == 1 and out[0].metadata["fo_judge_infra_error"] == 1.0 and out[0].status != Sample.Status.ABORTED
O.judge_code = _judge
print("[ok] judge infra error aborts training episodes at once, flags eval")

# 4d) eval when the final generated nothing (cut at once): a TRUNCATED stand-in with reward 0
out = play(evaluation=True, game="team", plan=[(PLAN_TXT, "stop")], sub=SUBS, final=[("", "length")])
assert len(out) == 1 and out[0].reward == 0.0 and out[0].metadata["fo_final_clamped"] == 1.0
print("[ok] eval with an empty final")

# 5) a cut subagent reports "cut off"; a cut plan gives every subagent NO_TASK
out = play(plan=[("still thinking " * 30, "length")], sub=[SUBS[0], ("x " * 50, "length"), SUBS[2], SUBS[3]],
           final=[(cpp("int main(){A;}"), "stop")])
final_text = [c[1] for c in CALLS if c[0] == "final"][0]
assert "Task: " + O.NO_TASK + "\nPrograms tested: 0\nReport:\n(cut off" in final_text
assert out[0].metadata["fo_plan_cut"] == 1.0 and out[0].metadata["fo_tasks_ok"] == 0.0
assert "(cut off: the answer hit the token limit)" in final_text and close(out[0].metadata["fo_sub_cut"], 0.25)
print("[ok] cut plan / cut subagent")

# 5b) the trace file keeps every turn's full text, cut flag and judged status
import glob, json  # noqa: E402
_td = tempfile.mkdtemp()
O.TRACE_DIR, O.TRACE_EVERY, O._trace_count = _td, 1, 0
play(plan=[(PLAN_TXT, "stop")], sub=[SUBS[0], ("x " * 50, "length"), SUBS[2], SUBS[3]],
     final=[(cpp("int main(){A;}"), "stop")])
O.TRACE_DIR = ""
tr = [json.loads(l) for f in glob.glob(_td + "/*.jsonl") for l in open(f)][-1]
roles = [t["role"] for t in tr["turns"]]
assert roles == ["plan", "sub1", "sub1", "sub2", "sub3", "sub3", "sub4", "sub4", "final"], roles
assert tr["turns"][1]["status"] == "done" and tr["turns"][2]["status"] is None and tr["turns"][3]["cut"]
assert tr["turns"][-1]["closed_think"] and "int main(){A;}" in tr["turns"][-1]["text"] and tr["turns"][0]["tasks"]
print("[ok] trace keeps full texts, cut flags and statuses")

# 5c) a compile error shows every compiler error (path shortened, source line and caret), without notes
out = play(plan=[(PLAN_TXT, "stop")], sub=[two("int main(){X;}"), SUBS[1], SUBS[2], SUBS[3]],
           final=[(cpp("int main(){A;}"), "stop")])
final_text = [c[1] for c in CALLS if c[0] == "final"][0]
assert ("Best tested program (test 1): compile error, score 0.00/100\nCompiler errors:\nsol.cpp:1:12: error: 'X' was "
        "not declared in this scope\n    1 | int main(){X;}\n      |            ^\nPer-test-case") in final_text
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
assert [round(m[f"fo_best_at{t}"], 4) for t in range(1, 6)] == [sA, sA, sB, 1.0, 1.0] and close(m["fo_S"], sE)
assert close(m["fo_V"], sE)
rev = [c[1] for c in CALLS if c[0] == "revise"]
assert "Your submission was judged: done, score 37.50/100\n" in rev[0] and "0.50 0.50 0.00 0.00" in rev[1]
assert "best score so far is 37.50/100" in rev[1] and "```cpp\nint main(){A;}\n```" in rev[0]
assert "only the last ```cpp block is submitted" in rev[0] and "This is round 2 of 5" in rev[0]
assert "This is round 5 of 5; only your round-5 submission counts" in rev[3]
assert "You have 5 rounds" in [c[1] for c in CALLS if c[0] == "solo"][0]
assert "thinking..." not in rev[0].split("Your submission was judged")[0].split("<|im_start|>assistant")[-1]
print(f"[ok] seq: 5 samples, groups {[o.group_index for o in out]}, best@t {[round(m[f'fo_best_at{t}'], 3) for t in range(1, 6)]}")

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
assert len(out) == 1 and close(out[0].reward, max(sA, sB, sC)) and close(out[0].metadata["fo_best_at3"], max(sA, sB, sC))
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
