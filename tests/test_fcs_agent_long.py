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
check("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat solution.cpp" not in pr and "7. `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` also submits" in pr,
      "instance: the one-shot submit step is replaced by the long workflow")
check(pr.index("## Long work") < pr.index("## Important Rules"), "instance: tools before the rules")
# harness v1 is frozen: the prompt of the smokes and of w100-w105 (at their defaults), byte for byte
if L.HARNESS == "v1" and not any(k in os.environ for k in ("MA_AG_STEPS", "MA_AG_TIMEOUT", "MA_LG_TOKENS")) \
        and L.N in (1, 4):
    import hashlib
    check(hashlib.sha1(pr.encode()).hexdigest()[:10] == {1: "20d1d16910", 4: "97b687a74e"}[L.N]
          and L.instance("Read n {and} print it.", "agent2", shared=True) == pr, "instance: v1 prompt unchanged")
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
check(L.trim_chat(chat, 1000, 0, 1000) == (c2, 5), "trim: low_chars = keep_chars is the old trim")
c4, d4 = L.trim_chat(chat, 1000, 0, 600)
check(sum(len(m["content"]) for m in c4[2:]) <= 600 and d4 == 7 and c4[2]["role"] == "assistant"
      and "7 earliest steps were removed" in c4[1]["content"], "trim with hysteresis: cut to low_chars")
c5 = c4 + [{"role": "assistant", "content": "a" * 100}, {"role": "user", "content": "o" * 100}]
c6, d6 = L.trim_chat(c5, 1000, d4, 600)
check(c6 is c5 and d6 == d4, "trim with hysteresis: under keep_chars again, chat and prefix unchanged")
c7, d7 = L.trim_chat(c4 + [{"role": "assistant", "content": "a" * 300}, {"role": "user", "content": "o" * 300}] * 2,
                     1000, d4, 600)
check(sum(len(m["content"]) for m in c7[2:]) <= 600 and d7 > d4 and c7[1]["content"].count("[Note:") == 1,
      "trim with hysteresis: over keep_chars again, one more cut and one note")
check(L.trim_chat(chat[:4], 10, 0, 5)[1] == 0, "trim with hysteresis: the last step is never dropped")
check(L.CTX_LOW == float(os.environ.get("MA_LG_CTX_LOW", "1.0")), "CTX_LOW from the environment")
m = L.SUBMIT_RE.fullmatch("submit")
check(m and m.group(1) is None and L.SUBMIT_RE.fullmatch(" submit sol2.cpp ").group(1) == "sol2.cpp"
      and not L.SUBMIT_RE.fullmatch("g++ a.cpp && submit"), "submit: alone, optional file")
m = L.POST_RE.fullmatch("post I take the greedy part\nand the tests")
check(m and m.group(1).startswith("I take") and "tests" in m.group(1) and not L.POST_RE.fullmatch("posting"),
      "post: message to the end, multi-line")
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
