"""flint.game without GPUs: the prompt, the provisional/final case split, the chat trimming and the harness commands.
tests/run_all.sh runs it with MA_LG_N=4 and again with MA_LG_N=1."""
import hashlib
import os

from helpers import check, done, env

os.environ.pop("MA_LG_CTX_LOW", None)   # the defaults below are the code's own
env()
from flint import game as G  # noqa: E402
from flint import prompts as P  # noqa: E402

pr = G.instance("Read n {and} print it.", "agent2")
team = G.N > 1
check("instance: fields filled, braces kept", "Read n {and} print it." in pr and "{" + "name}" not in pr
      and "{budget}" not in pr and '{"command": "ls -la"}' in pr)
check("instance: team or alone", ("You are agent2, one of 4 agents" in pr) == team and ("You work alone" in pr) == (not team))
check("instance: tools described", "`submit` or `submit <file>`" in pr and "`post <message>`" in pr and "PROVISIONAL" in pr)
check("instance: the post rule only in a team", ("your next command must be `post <message>`" in pr) == team)
check("instance: only the budget ends the work", "ends your work" not in pr and "used up" in pr
      and "7. `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` also submits" in pr and "your work then goes on" in pr)
check("instance: tools before the rules", pr.index("## Long work") < pr.index("## Important Rules"))
# the prompt of the runs before the cleanup (eval2 09cb863 in tool-call mode, budget end, fresh, ctx_low 0.75; a team:
# own repositories, must post), byte for byte at the defaults (A/B over every val40 and train67 prompt, 2026-10-28)
if not any(k in os.environ for k in ("MA_AG_TIMEOUT", "MA_LG_TOKENS")) and G.N in (1, 4):
    h = hashlib.sha1(pr.encode()).hexdigest()[:10]
    check("instance: the frozen prompt", h == {1: "c4ef55408f", 4: "d7bda03d2d"}[G.N], h)
check("harness tag", G.harness_tag() == "long-v1-budget-fresh-ctxlow0.75" + ("-own-mustpost" if team else ""))
check("split: even provisional, odd final, one case shared", G.split_cases(1) == ([0], [0], True)
      and G.split_cases(4) == ([0, 2], [1, 3], False) and G.split_cases(5) == ([0, 2, 4], [1, 3], False))
check("subset mean", abs(G.subset_mean([1, 0, 0.5, 0], [0, 2]) - 0.75) < 1e-9 and G.subset_mean([], [0]) == 0.0)
check("answer: after the last </think>, none without one", G.answer("<think>a</think>b</think>c") == "c"
      and G.answer("<think>never closed") == "" and G.answer("") == "")
chat = [{"role": "system", "content": "s"}, {"role": "user", "content": "inst"}]
for i in range(10):
    chat += [{"role": "assistant", "content": "a" * 100}, {"role": "user", "content": "o" * 100}]
c2, d = G.trim_chat(chat, 1000, 0)
hist = c2[2:]
check("trim: oldest pairs dropped, alternation and note kept", sum(len(m["content"]) for m in hist) <= 1000 and d == 5
      and len(hist) == 10 and hist[0]["role"] == "assistant" and "5 earliest steps were removed" in c2[1]["content"])
c3, d3 = G.trim_chat(c2 + [{"role": "assistant", "content": "a" * 400}, {"role": "user", "content": "o" * 400}], 1000, d)
check("trim: one note, count grows", d3 > 5 and c3[1]["content"].count("[Note:") == 1 and "inst" in c3[1]["content"])
check("trim: the last step is never dropped", G.trim_chat(chat[:4], 10, 0)[1] == 0)
c4, d4 = G.trim_chat(chat, 1000, 0, 600)
check("trim with hysteresis: cut to low_chars", sum(len(m["content"]) for m in c4[2:]) <= 600 and d4 == 7
      and c4[2]["role"] == "assistant" and "7 earliest steps were removed" in c4[1]["content"])
c5 = c4 + [{"role": "assistant", "content": "a" * 100}, {"role": "user", "content": "o" * 100}]
c6, d6 = G.trim_chat(c5, 1000, d4, 600)
check("trim with hysteresis: under keep_chars again, chat and prefix unchanged", c6 is c5 and d6 == d4)
c7, d7 = G.trim_chat(c4 + [{"role": "assistant", "content": "a" * 300}, {"role": "user", "content": "o" * 300}] * 2,
                     1000, d4, 600)
check("trim with hysteresis: over keep_chars again, one more cut and one note",
      sum(len(m["content"]) for m in c7[2:]) <= 600 and d7 > d4 and c7[1]["content"].count("[Note:") == 1)
check("trim counts the reasoning", G.trim_chat(chat[:2] + [{"role": "assistant", "content": "a", "reasoning_content":
      "r" * 2000}, {"role": "user", "content": "o"}] * 2, 1000, 0)[1] == 1)
check("CTX_LOW default 0.75", G.CTX_LOW == 0.75, G.CTX_LOW)
m = G.SUBMIT_RE.fullmatch("submit")
check("submit: alone, optional file", m and m.group(1) is None and G.SUBMIT_RE.fullmatch(" submit sol2.cpp ").group(1)
      == "sol2.cpp" and not G.SUBMIT_RE.fullmatch("g++ a.cpp && submit"))
m = G.POST_RE.fullmatch("post I take the greedy part\nand the tests")
check("post: message to the end, multi-line", m and m.group(1).startswith("I take") and "tests" in m.group(1)
      and not G.POST_RE.fullmatch("posting"))
obs = P.observation({"output": "x" * 10000, "returncode": 0, "exception_info": ""})
check("observation: mini.yaml's JSON, elided past 10000 characters", '"elided_chars": 0' in obs and '"output_head"' in obs
      and '"output": "ab"' in P.observation({"output": "ab", "returncode": 0, "exception_info": ""}))
done()
