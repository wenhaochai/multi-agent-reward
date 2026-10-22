"""fcs_agent_long without GPUs: the prompt, the provisional/final case split, the chat trimming and the harness
commands (run in miles.sif with MA_LG_N=4 and again with MA_LG_N=1)."""
import os
import sys

from miles_team import fcs_agent_long as L

ok = True


def check(c, what):
    global ok
    print(("PASS " if c else "FAIL ") + what)
    ok &= bool(c)


pr = L.instance("Read n {and} print it.", "agent2")
team = L.N > 1
check("Read n {and} print it." in pr and "{" + "name}" not in pr and "{budget}" not in pr, "instance: fields filled, braces kept")
check(("You are agent2, one of 4 agents" in pr) == team and ("You work alone" in pr) == (not team), "instance: team or alone")
check("`submit` or `submit <file>`" in pr and "`post <message>`" in pr and "PROVISIONAL" in pr, "instance: tools described")
check("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat solution.cpp" not in pr and "7. When you cannot improve" in pr,
      "instance: the one-shot submit step is replaced by the long workflow")
check(pr.index("## Long work") < pr.index("## Important Rules"), "instance: tools before the rules")
check(L.split_cases(1) == ([0], [0], True) and L.split_cases(4) == ([0, 2], [1, 3], False)
      and L.split_cases(5) == ([0, 2, 4], [1, 3], False), "split: even provisional, odd final, one case shared")
check(abs(L.subset_mean([1, 0, 0.5, 0], [0, 2]) - 0.75) < 1e-9 and L.subset_mean([], [0]) == 0.0, "subset mean")
chat = [{"role": "system", "content": "s"}, {"role": "user", "content": "inst"}]
for i in range(10):
    chat += [{"role": "assistant", "content": "a" * 100}, {"role": "user", "content": "o" * 100}]
c2, d = L.trim_chat(chat, 1000, 0)
hist = c2[2:]
check(sum(len(m["content"]) for m in hist) <= 1000 and d == 5 and len(hist) == 10 and hist[0]["role"] == "assistant"
      and "5 earliest steps were removed" in c2[1]["content"], "trim: oldest pairs dropped, alternation and note kept")
c3, d3 = L.trim_chat(c2 + [{"role": "assistant", "content": "a" * 400}, {"role": "user", "content": "o" * 400}], 1000, d)
check(d3 > 5 and c3[1]["content"].count("[Note:") == 1 and "inst" in c3[1]["content"], "trim: one note, count grows")
check(L.trim_chat(chat[:4], 10, 0)[1] == 0, "trim: the last step is never dropped")
m = L.SUBMIT_RE.fullmatch("submit")
check(m and m.group(1) is None and L.SUBMIT_RE.fullmatch(" submit sol2.cpp ").group(1) == "sol2.cpp"
      and not L.SUBMIT_RE.fullmatch("g++ a.cpp && submit"), "submit: alone, optional file")
m = L.POST_RE.fullmatch("post I take the greedy part\nand the tests")
check(m and m.group(1).startswith("I take") and "tests" in m.group(1) and not L.POST_RE.fullmatch("posting"),
      "post: message to the end, multi-line")
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
