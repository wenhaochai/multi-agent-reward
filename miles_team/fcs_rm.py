"""Frontier-CS reward for miles: FrontierSmith's code extraction (verl/verl/utils/reward_score/frontiercs.py,
strip_think + extract_cpp) scored by the local judge (fcs_judge.py, the official engine's rules).
sample.label is the problem directory; the reward is the judge score / 100 (EasyPPO's continuous 0-100 score, rescaled).
Judges run in threads (each a few subprocesses), at most FCS_RM_CONCURRENCY at once (default: cpus / FCS_CASE_WORKERS).
Use: --custom-rm-path miles_team.fcs_rm.fcs_rm
Code block rule (2026-10-04, user decision): the LAST ```cpp block is judged (FCS_EXTRACT=last, the default), as in
LiveCodeBench, open-r1, rllm/DeepCoder and verl. Frontier-CS's official harness and FrontierSmith take the LONGEST
block (FCS_EXTRACT=longest; Frontier-CS d5185d23, 2025-12-10, "likely the main solution"); on the SFT init's
multi-block answers the longest is usually the first draft (149 of 203) and the last block scores 4.88 vs 1.29.
Every comparison is against our own single-agent baseline under the same rule, so no second (official) score is kept.
Every FCS_RM_TRACE_EVERY-th judged sample (default 32; 0 = off) is written in full to <run>/traces/solo_<pid>.jsonl
(<run> = the parent of --save): response, finish reason, extracted code, status, score; tools/read_traces.py reads it.
No answer, no score (owner decision 2026-10-07): a reply cut off at its token limit, or a thinking reply that never
closed its reasoning (no </think>), scores 0 without being judged, in this solo reward and in every fcs_oracle game
(answer()). FrontierSmith's strip_think keeps the whole text of an unclosed reply, so the code written inside the
reasoning was judged; this reward did that until 2026-10-07 (runs switch at their first segment after that).
FCS_THINK=0 (a non-thinking chat template, whose replies have no </think>) zeroes only the cut-off ones.
Eval scoring after training resumes (owner 2026-10-07): an eval sample whose metadata has fcs_defer_score
(tools/make_defer_eval_data.py) gets a placeholder reward of 0, so generation ends without waiting for the judge. The
eval log function (--custom-eval-rollout-log-function-path miles_team.fcs_rm.log_eval) saves the eval's samples to
<run>/eval_pending/eval_<K>.pt and judges them in a background thread on the same node, with the same rules, while
training goes on; the result is logged by miles' own log_eval_rollout_data (the same "metrics.py - eval K:" line and
W&B row) at the next rollout (--custom-rollout-log-function-path miles_team.fcs_rm.log_rollout), and the eval's
replies are kept in <run>/eval_pending/eval_<K>.jsonl.gz. The GPUs no longer sit idle while the judge waits out
interactive cases. A segment that ends before a pending eval is logged leaves its .pt; the next segment judges it at
its first rollout (python3 -m miles_team.fcs_rm --score-pending <run dir> does it without training). The last eval
of a run is judged at once (nothing to overlap with).
"""
import asyncio
import functools
import glob
import gzip
import json
import os
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from miles_team.fcs_judge import CASE_WORKERS, job_cpus, judge

_SEM = None


def strip_think(response: str) -> str:
    """Remove everything through the last closing think tag."""
    if not response:
        return response
    _, sep, suffix = response.rpartition("</think>")
    return suffix if sep else response


def answer(text: str, thinking: bool = True) -> str:
    """The answer part of a reply: the text after the last </think>; a thinking reply that never closed its reasoning
    has none."""
    if thinking and "</think>" not in (text or ""):
        return ""
    return strip_think(text or "") or ""


THINK = os.environ.get("FCS_THINK", "1") == "1"


def _cut(sample) -> bool:
    """The sample hit its token limit (miles sets Sample.Status.TRUNCATED)."""
    st = getattr(sample, "status", None)
    return getattr(st, "name", str(st)) == "TRUNCATED"


def answered(text: str, cut: bool, thinking: bool = THINK) -> bool:
    """A reply that may be judged: not cut off, and (thinking on) its reasoning closed."""
    return not cut and (not thinking or "</think>" in (text or ""))


EXTRACT = os.environ.get("FCS_EXTRACT", "last")
assert EXTRACT in ("last", "longest"), f"FCS_EXTRACT must be last or longest, not {EXTRACT}"


_SPECIAL = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


PARSER = "lines-v2"  # recorded with re-judged scores (tools/rescore_probe.py)
_TAG_RE = re.compile(r"^(?:cpp|c\+\+|cc|cxx)\d*$")


def _cpp_tag(info: str) -> bool | None:
    """True for a C++ fence tag (cpp, c++, cc, cxx, cpp17, c++17, {.cpp}, "cpp title=..."), None for no tag."""
    word = (info.split() or [""])[0].strip("{}.").lower()
    return None if not word else bool(_TAG_RE.match(word))


def code_blocks(text: str, untagged: bool = True) -> list[str]:
    """Fenced blocks of a reply, in order. A fence opens on a line starting with ``` (its tag is the rest of the line;
    a line holding a second ``` is inline code, not a fence) and closes on a line that starts or ends with ```; a
    tagged fence line inside an open block (the model forgot to close it) closes that block and opens the next.
    Only C++-tagged blocks count, plus untagged ones when `untagged`. An unclosed block at the end counts only when
    the reply has no closed block (a cut-off rewrite must not replace a complete program). Audit 2026-10-04:
    FrontierSmith's regex let a closing fence open a "block" and took every untagged fence (a test case) as C++."""
    for tok in _SPECIAL:  # a decoded reply can end in "```<|im_end|>"
        text = (text or "").replace(tok, "")
    closed, open_block, cur, tag = [], None, None, None

    def keep(t):
        return t is True or (untagged and t is None)

    for line in text.replace("\r\n", "\n").split("\n"):
        st = line.strip()
        is_fence = st.startswith("```") and "```" not in st[3:]
        if cur is None:
            if is_fence:
                cur, tag = [], _cpp_tag(st[3:])
            continue
        if is_fence and st[3:].strip():  # a tagged fence inside an open block: close it, open the next
            if keep(tag):
                closed.append("\n".join(cur).strip())
            cur, tag = [], _cpp_tag(st[3:])
            continue
        if st.startswith("```") or st.endswith("```"):  # a fence line closes the block (code may precede it)
            if not st.startswith("```"):
                cur.append(line.rstrip()[:-3])
            if keep(tag):
                closed.append("\n".join(cur).strip())
            cur = None
            continue
        cur.append(line)
    if cur is not None and keep(tag):
        open_block = "\n".join(cur).strip()
    out = [c for c in closed if c]
    return out if out else ([open_block] if open_block else [])


def extract_cpp(response_text: str, rule: str | None = None) -> str:
    """Extract C++ code from model response (markdown or raw), ignoring <think> blocks: the last cpp block (rule
    "last", the default), or with rule "longest" FrontierSmith's original regex and longest block, verbatim (the
    blind-game arms keep it)."""
    if not response_text:
        return ""

    response_text = strip_think(response_text)
    if not response_text:
        return ""

    code = response_text.strip()

    if (rule or EXTRACT) == "longest":
        cpp_pattern = r'```(?:cpp|c\+\+)?\s*\n(.*?)```'
        matches = re.findall(cpp_pattern, code, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
    else:  # the last cpp-tagged block; an untagged one only when the reply has no tagged block
        blocks = code_blocks(code, untagged=False) or code_blocks(code)
        if blocks:
            return blocks[-1]

    # Fallback: strip markdown if present
    if code.startswith("```cpp"):
        code = code[6:].strip()
    elif code.startswith("```c++"):
        code = code[6:].strip()
    elif code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    return code


def score(response: str, problem_dir: str, pool=None) -> float:
    code = extract_cpp(response or "")
    if not code:
        return 0.0
    try:
        r = judge(problem_dir, code, pool=pool)
        if r.get("infra"):  # a case or the compile could not run (fork / disk / sandbox): scored 0, made visible
            print(f"[fcs_rm] JUDGE-INFRA {problem_dir}: {r.get('status')} {(r.get('msg') or '')[-200:]}"[:500],
                  file=sys.stderr, flush=True)
        return r["score"] / 100.0
    except Exception as e:  # an infrastructure failure (helper compile, disk), not the policy's fault: make it visible
        print(f"[fcs_rm] JUDGE-ERROR {problem_dir}: {e}"[:500], file=sys.stderr, flush=True)
        return 0.0


def _sem():
    global _SEM
    if _SEM is None:
        # default: the job's own cores (sched_getaffinity, not the node's cpu_count), 8 kept for SGLang
        n = int(os.environ.get("FCS_RM_CONCURRENCY", max(1, (job_cpus() - 8) // CASE_WORKERS)))
        _SEM = asyncio.Semaphore(n)
        _SEM.size = n
    return _SEM


_POOL = None


def _thread(fn, *a):
    """Run fn in this module's own thread pool. asyncio.to_thread uses the loop's default executor, which Python sizes
    at min(32, os.cpu_count() + 4) = 32 on every node, so FCS_RM_CONCURRENCY=56 judged at most 32 programs at once
    (audit 2026-10-07). The pool has a thread per semaphore slot plus 8 for traces."""
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=_sem().size + 8, thread_name_prefix="fcs-rm")
    return asyncio.get_running_loop().run_in_executor(_POOL, functools.partial(fn, *a))


_TRACE_EVERY = int(os.environ.get("FCS_RM_TRACE_EVERY", "32"))
_trace_n = 0


def _trace(args, sample, reward: float) -> None:
    """Logging only: the full sample of every _TRACE_EVERY-th call, with the status the reward came from."""
    global _trace_n
    _trace_n += 1
    if not _TRACE_EVERY or (_trace_n - 1) % _TRACE_EVERY or not getattr(args, "save", None):
        return
    try:
        resp = sample.response or ""
        code = extract_cpp(resp)
        r = judge_full(sample.label, code) if code else {"status": "no code", "score": 0.0}
        d = os.path.join(os.path.dirname(os.path.abspath(args.save)), "traces")
        os.makedirs(d, exist_ok=True)
        rec = {"time": round(time.time(), 1), "label": sample.label, "reward": reward, "status": r["status"], "rejudged_score": r["score"],
               "answered": answered(resp, _cut(sample)),
               "truncated": str(getattr(sample, "status", "")).endswith("TRUNCATED"),
               "response_length": getattr(sample, "response_length", None), "closed_think": "</think>" in resp,
               "text": resp}
        with open(os.path.join(d, f"solo_{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:  # tracing must never affect training
        print(f"[fcs_rm] trace failed: {e}"[:300], file=sys.stderr, flush=True)


DEFER_KEY = "fcs_defer_score"  # eval data metadata: judge this sample after training resumes


async def _one(sample, args=None) -> float:
    meta = getattr(sample, "metadata", None)
    if isinstance(meta, dict) and meta.get(DEFER_KEY) and _deferring(args):
        meta["fcs_deferred"] = True  # placeholder; log_eval judges it
        return 0.0
    async with _sem():
        resp = sample.response or ""
        reward = await _thread(score, resp, sample.label) if answered(resp, _cut(sample)) else 0.0
        if args is not None and _TRACE_EVERY:
            await _thread(_trace, args, sample, reward)
        return reward


def judge_full(problem_dir: str, code: str) -> dict:
    """Judge already-extracted code: {"score": [0, 1], "cases": per-case ratios (zeros when nothing ran),
    "status": ..., "infra_error": bool}. Infrastructure failures are logged and flagged, never a silent 0."""
    from miles_team.fcs_judge import load_problem
    from pathlib import Path
    n = len(load_problem(Path(problem_dir))["cases"])
    if not code:
        return {"score": 0.0, "cases": [0.0] * n, "status": "no code", "infra_error": False, "msg": ""}
    try:
        r = judge(problem_dir, code)
    except Exception as e:
        print(f"[fcs_rm] JUDGE-ERROR {problem_dir}: {e}"[:500], file=sys.stderr, flush=True)
        return {"score": 0.0, "cases": [0.0] * n, "status": "judge error", "infra_error": True, "msg": ""}
    cases = r.get("cases") or [0.0] * n
    infra = bool(r.get("infra")) or r.get("status") == "infra error"
    if infra:
        print(f"[fcs_rm] JUDGE-INFRA {problem_dir}: {r.get('status')} {(r.get('msg') or '')[-200:]}"[:500],
              file=sys.stderr, flush=True)
    return {"score": r["score"] / 100.0, "cases": [float(c) for c in cases], "status": r["status"],
            "infra_error": infra, "msg": r.get("msg", "") if r["status"] == "compile error" else ""}


async def judge_code(problem_dir: str, code: str) -> dict:
    """judge_full in a thread under the shared FCS_RM_CONCURRENCY semaphore. The slot is held until the thread ends,
    also when the caller is cancelled (an aborted episode): a cancelled await would otherwise free the slot while the
    judge keeps running, and the extra judges oversubscribe the node (wall-limit kills elsewhere)."""
    async with _sem():
        fut = asyncio.ensure_future(_thread(judge_full, problem_dir, code))
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            await asyncio.wait([fut])
            raise


async def fcs_rm(args, sample, **kwargs):
    if isinstance(sample, list):
        return list(await asyncio.gather(*(_one(s, args) for s in sample)))
    return await _one(sample, args)


# ---- eval scoring after training resumes --------------------------------------------------------------------------
# Background judging has its own slots (FCS_EVAL_BG_SLOTS compiles and cases at once, FCS_EVAL_BG_PROGRAMS programs),
# so its cases never queue ahead of training's in fcs_judge's shared pool; the judge limits CPU time (wall 2x), so the
# extra slots beside training's leave verdicts as they were (review 2026-10-07).
_BG_SLOTS = int(os.environ.get("FCS_EVAL_BG_SLOTS", "16"))
_BG_PROGRAMS = int(os.environ.get("FCS_EVAL_BG_PROGRAMS", "8"))
_BG_POOL = None
_bg: list[threading.Thread] = []
_ready: "queue.Queue[tuple]" = queue.Queue()  # (rollout_id, args, data, extra_metrics, path): judged, to log on miles' loop
_resumed = False
_final_next = False  # set by log_rollout: the eval after this rollout is the run's last, judged inline as before
_scoring: set[str] = set()
_ME = "miles_team.fcs_rm."


def _deferring(args) -> bool:
    """Defer only when both log functions are ours (else miles would log the placeholder zeros) and not for the run's
    last eval."""
    return (getattr(args, "save", None) is not None and not _final_next
            and getattr(args, "custom_eval_rollout_log_function_path", None) == _ME + "log_eval"
            and getattr(args, "custom_rollout_log_function_path", None) == _ME + "log_rollout")


def _bg_pool() -> ThreadPoolExecutor:
    global _BG_POOL
    if _BG_POOL is None:
        _BG_POOL = ThreadPoolExecutor(max_workers=max(1, _BG_SLOTS), thread_name_prefix="fcs-eval-slot")
    return _BG_POOL


def _say(msg: str) -> None:
    print(f"[fcs_rm] {msg}"[:800], file=sys.stderr, flush=True)


def pending_dir(save: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(save)), "eval_pending")


def _deferred(data: dict) -> list[tuple[str, int]]:
    return [(k, i) for k, v in data.items() for i, s in enumerate(v.get("samples") or [])
            if isinstance(getattr(s, "metadata", None), dict) and s.metadata.get("fcs_deferred")]


def _plain_args(args) -> dict:
    """The args a later process needs to log this eval: every value that pickles."""
    import pickle
    out = {}
    for k, v in vars(args).items():
        try:
            pickle.dumps(v)
            out[k] = v
        except Exception:
            pass
    return out


def _save_pending(path: str, rollout_id: int, args, data: dict, extra: dict) -> None:
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = {"rollout_id": rollout_id, "extra_metrics": dict(extra), "args": _plain_args(args),
            "data": {k: {**{f: v[f] for f in v if f != "samples"}, "samples": [s.to_dict() for s in v["samples"]]}
                     for k, v in data.items()}}
    torch.save(blob, path + ".tmp")
    os.replace(path + ".tmp", path)  # a kill mid-write leaves no half file


def _load_pending(path: str):
    import torch
    from argparse import Namespace
    from miles.utils.types import Sample
    blob = torch.load(path, weights_only=False)
    data = {k: {**v, "samples": [Sample.from_dict(d) for d in v["samples"]]} for k, v in blob["data"].items()}
    return blob["rollout_id"], Namespace(**blob["args"]), data, blob["extra_metrics"]


def _fail(path: str, why) -> None:
    """A pending eval that cannot be judged or logged: set aside (the launcher then counts the eval as lost)."""
    _scoring.discard(path)
    try:
        os.replace(path, path + ".failed")
    except OSError:
        pass
    _say(f"pending eval {path} set aside as .failed: {why!r}")


def _score(data: dict, programs: int, pool=None) -> None:
    """Judge every deferred sample of an eval in place, exactly as _one would have."""
    todo = _deferred(data)

    def one(ki):
        s = data[ki[0]]["samples"][ki[1]]
        resp = s.response or ""
        return score(resp, s.label, pool) if answered(resp, _cut(s)) else 0.0

    with ThreadPoolExecutor(max_workers=max(1, programs), thread_name_prefix="fcs-eval") as ex:
        rewards = list(ex.map(one, todo))
    for (k, i), r in zip(todo, rewards):
        s = data[k]["samples"][i]
        s.reward = r
        s.metadata.pop("fcs_deferred", None)
        data[k]["rewards"][i] = r


def _finish(rollout_id: int, args, data: dict, extra: dict, path: str) -> None:
    """Keep a judged eval's replies, log it through miles (its line, its W&B row), then drop the pending file: the
    slow write comes first, so a kill can repeat the log only in the instant between logging and the remove."""
    from miles.ray.rollout.metrics import log_eval_rollout_data
    keep = path[:-3] + ".jsonl.gz"
    with gzip.open(keep + ".tmp", "wt") as f:
        for k, v in data.items():
            for s in v["samples"]:
                f.write(json.dumps({"dataset": k, "index": s.index, "label": s.label, "status": s.status.value,
                                    "response_length": s.response_length, "reward": s.reward,
                                    "response": s.response}) + "\n")
    os.replace(keep + ".tmp", keep)
    log_eval_rollout_data(rollout_id, args, data, dict(extra))
    os.remove(path)
    _scoring.discard(path)


def _background(rollout_id: int, args, data: dict, extra: dict, path: str) -> None:
    started = time.time()
    try:
        _score(data, _BG_PROGRAMS, _bg_pool())
        _ready.put((rollout_id, args, data, extra, path))
        _say(f"eval {rollout_id}: judged in {(time.time() - started) / 60:.1f} min in the background; logged at the next rollout")
    except Exception as e:  # the pending file stays; the next segment judges it again
        _scoring.discard(path)
        _say(f"eval {rollout_id}: background judging failed: {e!r}")


def _start(rollout_id: int, args, data: dict, extra: dict, path: str) -> None:
    _scoring.add(path)
    t = threading.Thread(target=_background, args=(rollout_id, args, data, extra, path), daemon=True,
                         name=f"fcs-eval-{rollout_id}")
    t.start()
    _bg.append(t)


def _drain() -> None:
    while True:
        try:
            item = _ready.get_nowait()
        except queue.Empty:
            return
        try:
            _finish(*item)
        except Exception as e:
            _fail(item[4], e)


def _leftovers(args) -> list[str]:
    return [p for p in sorted(glob.glob(os.path.join(pending_dir(args.save), "eval_*.pt"))) if p not in _scoring]


def _resume(args) -> None:
    """Pending evals a previous segment left (it ended before logging them): judge them in the background."""
    global _resumed
    if _resumed or not getattr(args, "save", None):
        return
    _resumed = True
    for path in _leftovers(args):
        try:
            rid, _, data, extra = _load_pending(path)
        except Exception as e:
            _fail(path, e)
            continue
        _say(f"eval {rid}: left pending by an earlier segment; judging it in the background")
        _start(rid, args, data, extra, path)


def _settle_all(args) -> None:
    """The run ends after this eval: wait for background evals, log them, and judge any other pending eval here."""
    for t in _bg:
        t.join()
    _drain()
    for path in _leftovers(args):
        try:
            rid, _, data, extra = _load_pending(path)
            _scoring.add(path)
            _score(data, _sem().size)
            _finish(rid, args, data, extra, path)
        except Exception as e:
            _fail(path, e)


def log_eval(rollout_id, args, data, extra_metrics) -> bool:
    """--custom-eval-rollout-log-function-path: defer the judging of an eval whose samples carry fcs_defer_score.
    Returns True when miles must not log now (the eval is logged once judged)."""
    _drain()
    todo = _deferred(data)
    if not todo:
        if rollout_id + 1 >= args.num_rollout and getattr(args, "save", None):
            _settle_all(args)
        return False  # judged inline (or never deferred): miles logs it
    path = os.path.join(pending_dir(args.save), f"eval_{rollout_id}.pt")
    try:
        _save_pending(path, rollout_id, args, data, extra_metrics or {})
    except Exception as e:  # cannot keep it for later: judge it here, as inline judging would have, and let miles log
        _say(f"eval {rollout_id}: could not save for background judging ({e!r}); judging it now")
        _score(data, _sem().size)
        return False
    _start(rollout_id, args, data, extra_metrics or {}, path)
    _say(f"eval {rollout_id}: {len(todo)} replies saved to {path}; judging them while training goes on")
    return True


def log_rollout(rollout_id, args, samples, extra_metrics, rollout_time) -> bool:
    """--custom-rollout-log-function-path: log the evals judged since the last rollout; at the first rollout of a
    segment, resume the evals an earlier segment left pending; mark the run's last eval to be judged inline. miles then
    logs the rollout as usual. Never raises: a pending eval that fails is set aside, training goes on."""
    global _final_next
    try:
        _final_next = rollout_id + 1 >= args.num_rollout
        _resume(args)
        _drain()
    except Exception as e:
        _say(f"log_rollout {rollout_id}: {e!r}")
    return False


def score_pending(run_dir: str) -> None:
    """Judge and log a run's pending evals without training (the segment that finds the run complete)."""
    from miles.utils import logging_utils
    from miles.utils.tracking_utils import tracking
    logging_utils.configure_logger_raw("rollout_executor")
    tracking.log = lambda *a, **k: None  # no W&B here: relog_wandb takes these evals from the log lines
    for path in sorted(glob.glob(os.path.join(run_dir, "eval_pending", "eval_*.pt"))):
        try:
            rid, a, data, extra = _load_pending(path)
            a.custom_eval_rollout_log_function_path = None
            _score(data, _sem().size)
            _finish(rid, a, data, extra, path)
        except Exception as e:
            _fail(path, e)


if __name__ == "__main__":
    assert sys.argv[1] == "--score-pending", "python3 -m miles_team.fcs_rm --score-pending <run dir>"
    score_pending(sys.argv[2])
