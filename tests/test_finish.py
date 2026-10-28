"""The finish command and the judge in the game: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` submits solution.cpp and
the agent goes on; with no solution.cpp it says so and goes on; a submission's judge time, cases and the judge's
per-case lines reach the trace, never the agent; a judge infrastructure failure is judged again, and one that stays is
never picked as the best; the unsubmitted solution.cpp left at the budget's end is judged apart (fo_lg_S_auto).
Scripted agent, stubbed judge, real problem 0 for the case count."""
import asyncio

from helpers import FIN, LABEL, MSG, check, done, env, inp, scripted, sessions, traces

env(MA_LG_N=1, MA_LG_TOKENS=100000)
from flint import game as G  # noqa: E402

judged = []
SECRET = "wrong answer expected 4242, found 7"   # a checker line that quotes the expected output
RESULT = None   # RESULT(code, call number) -> a judge result, to stub infra failures


async def fake_judge(label, code):
    judged.append(code)
    await asyncio.sleep(0.05)
    if RESULT is not None:
        return RESULT(code, len(judged))
    return {"cases": [0.9 if "return 1" in code else 0.5] * 10, "status": "ok", "infra_error": "", "msg": "",
            "case_info": [SECRET] * 10}


def infra(cases):
    return {"cases": cases, "status": "done", "infra_error": True, "msg": "", "case_info": ["INFRA exception: x"] * 10}


def fake_ok(code):
    return {"cases": [0.5] * 10, "status": "done", "infra_error": False, "msg": "", "case_info": []}


Scripted = scripted(G)
G.judge_code, G.INFRA_BACKOFF, G.Agent = fake_judge, 0.01, Scripted


def run(script, full=False):
    Scripted.script = list(script)
    judged.clear()
    agents, S, info = asyncio.run(G.episode(inp(), MSG, LABEL, sessions(None)))
    run.info = info
    return (agents[0], S, info) if full else agents[0]


a = run([FIN, "printf 'int main(){}\\n' > solution.cpp", FIN, FIN])
check("no solution.cpp: told, goes on", "no solution.cpp" in a.outs[0] and a.steps == 4, a.outs)
check("finish submits solution.cpp", len(judged) == 1 and a.n_submit == 1 and "Your work goes on" in a.outs[2], a.outs)
check("finish on an unchanged file: not judged, a restart", len(judged) == 1 and a.restarts == 1
      and a.exit == "TokensExceeded", (a.exit, a.restarts))
a = run(["printf 'int main(){}\\n' > solution.cpp", "submit", FIN])
check("an explicit submit counts as the last submission", len(judged) == 1 and a.restarts == 1, (judged, a.restarts))
a = run(["printf 'int main(){}\\n' > solution.cpp", "submit sol2.cpp", FIN])
check("a submit of another file does not", len(judged) == 1 and a.restarts == 0 and "Your work goes on" in a.outs[2],
      (judged, a.outs))
# judge time, every case and the judge's per-case lines reach the trace, never the agent (they would leak final tests)
a = run(["printf 'int main(){}\\n' > solution.cpp", "submit"])
sub = traces()[-1]["submissions"][0]
check("trace has judge_s, cases, case_info", sub.get("judge_s", 0) >= 0.05 and sub.get("cases") == [0.5] * 10
      and sub.get("case_info") == [SECRET] * 10, sub)
check("agent never sees case_info", not any("4242" in o for o in a.outs), a.outs)
check("scores per provisional test", "(per test: 50, 50, 50, 50, 50). Best" in a.outs[1], a.outs[1])
check("fo_lg_judge_min in info", abs(run.info["fo_lg_judge_min"] - sub["judge_s"] / 60) < 1e-9, run.info)
check("judge time is not command time", run.info["fo_ag_run_min"] * 60 < sub["judge_s"] + 5, run.info)
# an infra failure is judged again: once infra, then a real result
RESULT = lambda code, k: infra([0.0] * 10) if k == 1 else fake_ok(code)  # noqa: E731
a = run(["printf 'int main(){}\\n' > solution.cpp", "submit"])
sub = traces()[-1]["submissions"][0]
check("infra once: one retry, then a real result", len(judged) == 2 and sub["infra_retries"] == 1 and not sub["infra"]
      and sub["prov"] == 0.5 and "could not finish" not in a.outs[1], (judged, sub, a.outs))
check("infra info keys: a resolved retry is no infra error", run.info["fo_judge_infra_retries"] == 1
      and run.info["fo_judge_infra_unresolved"] == 0.0 and run.info["fo_judge_infra_error"] == 0.0
      and run.info["fo_judge_infra_any"] == 1.0, run.info)
# an infra failure that stays: told, and never picked over a real result even with a higher provisional score
RESULT = lambda code, k: infra([1.0] * 10) if "A" in code else fake_ok(code)  # noqa: E731
a = run(["printf 'int main(){A;}\\n' > solution.cpp", "submit", "printf 'int main(){}\\n' > solution.cpp", "submit"])
tr = traces()[-1]
subs = tr["submissions"]
check("persistent infra: 3 tries, then told", len(judged) == 4 and subs[0]["infra"] and subs[0]["infra_retries"] == 2
      and "could not finish this submission after 3 tries (status: done)" in a.outs[1], (judged, subs, a.outs))
check("an infra entry with a higher provisional score is not the best", tr["best"]["sha"] == subs[1]["sha"]
      and tr["S"] == 0.5 and run.info["fo_lg_best_prov"] == 0.5, tr["best"])
check("the infra submission is not shown as the best: none yet, then the real one", "Best provisional so far: none yet."
      in a.outs[1] and "Best provisional so far: 50.00 (#2, agent1)." in a.outs[3], a.outs)
check("unresolved infra flagged", run.info["fo_judge_infra_unresolved"] == 1.0 and run.info["fo_judge_infra_error"] == 1.0
      and run.info["fo_judge_infra_any"] == 1.0 and run.info["fo_judge_infra_retries"] == 2, run.info)
a = run(["printf 'int main(){A;}\\n' > solution.cpp", "submit"])
check("all infra: the infra one is still the answer", traces()[-1]["best"]["infra"]
      and run.info["fo_lg_submissions"] == 1 and "Best provisional so far: none yet." in a.outs[1], a.outs)
RESULT = None
# the budget runs out with a better solution.cpp unsubmitted: S keeps the rule, fo_lg_S_auto judges it once apart
a, S, info = run(["printf 'int main(){}\\n' > solution.cpp", "submit", "printf 'int main(){return 1;}\\n' > solution.cpp"],
                 True)
check("budget end: S counts only the submission", abs(S - 0.5) < 1e-9 and len(judged) == 2, (S, judged))
check("budget end: the unsubmitted file is judged apart", abs(info["fo_lg_S_auto"] - 0.9) < 1e-9
      and info["fo_lg_auto_judged"] == 1.0 and info["fo_lg_submissions"] == 1, info)
a, S, info = run(["printf 'int main(){return 1;}\\n' > solution.cpp"], True)
check("budget end, no submission: S 0, S_auto > 0", S == 0.0 and info["fo_lg_S_auto"] > 0, (S, info))
a, S, info = run(["printf 'int main(){}\\n' > solution.cpp", "submit"], True)
check("budget end, file already submitted: no extra judge", len(judged) == 1 and info["fo_lg_auto_judged"] == 0.0
      and info["fo_lg_S_auto"] == S and info["fo_lg_shared_case"] == 0.0, (judged, info))
done()
