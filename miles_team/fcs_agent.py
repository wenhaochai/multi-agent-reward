"""Frontier-CS in mini-swe-agent's bash-only scaffold (owner 2026-10-13: an agentic baseline; public tools only).

One agent works in a sandboxed shell. Each step, the model writes a THOUGHT and exactly one ```mswea_bash_command
block; the command runs in the episode's own work dir and its output comes back as the next user message. The episode
ends when the agent runs `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat solution.cpp` (the output after that first
line is the submission, judged once on the hidden tests), or at a limit (no submission, score 0), as in mini-swe-agent.

Copied verbatim from mini-swe-agent 2.4.6 (src/minisweagent/config/mini_textbased.yaml; models/litellm_textbased_model,
models/utils/actions_text; environments/local): the system template, the format rules and command examples of the
instance template, the action regex, the observation template (10000-character elision), the format-error template,
the submit marker, and the agent loop's limits (step limit, max 3 consecutive format errors). Changed: the task text
(the Frontier-CS statement, the workflow for a C++ program, the submit command); every command runs under the judge's
Landlock sandbox (system dirs read/execute, the work dir read/write, nothing else; TCP blocked; CPU, memory, file size
and process limits; a wall timeout that kills the process group); the hidden tests never enter the work dir. The model
is our policy through miles' sglang router: every step is one new segment rendered from the visible history (THOUGHT
and command, reasoning dropped), so a 40-step episode does not carry 40 steps of reasoning tokens.

--custom-generate-function-path miles_team.fcs_agent.generate (eval sets: metadata.messages = the problem prompt)
Env: MA_AG_STEPS (40), MA_AG_TURN (16384 tokens per step), MA_AG_TOKENS (131072 generated per episode), MA_AG_MAX_LEN
(65536 context), MA_AG_TIMEOUT (60 s per command), MA_AG_TRACE_DIR, MA_AG_THINK (1).
"""
from __future__ import annotations

import asyncio
import ctypes
import json
import math
import os
import re
import shutil
import struct
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput

from . import fcs_judge as J
from .fcs_rm import answer, judge_code
from .team_rollout import Aborted, Session

STEPS = int(os.environ.get("MA_AG_STEPS", "40"))
TURN = int(os.environ.get("MA_AG_TURN", "16384"))
TOKENS = int(os.environ.get("MA_AG_TOKENS", "131072"))
MAX_LEN = int(os.environ.get("MA_AG_MAX_LEN", "65536"))
TIMEOUT = int(os.environ.get("MA_AG_TIMEOUT", "60"))
TRACE_DIR = os.environ.get("MA_AG_TRACE_DIR", "")
THINK = os.environ.get("MA_AG_THINK", "1") == "1"
MU = float(os.environ.get("MA_AG_MU", "0"))   # reward cost per minute of command run time (owner 2026-10-13: R = S - mu c)
MAX_FORMAT_ERRORS = 3
CMD_MEM = 4 << 30            # address space of a command and its children (g++ -O2 of a large file fits)
OUT_CHARS = 10000            # mini-swe-agent's elision threshold
MARK = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
ACTION_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
_SPECIAL = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")
_POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("MA_AG_CMD_WORKERS", "32")))

# ---- mini-swe-agent 2.4.6 mini_textbased.yaml (system template verbatim) ----------------------------------------------
SYSTEM = """You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.

```mswea_bash_command
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected.
"""
# the instance template: the task and workflow are ours (a C++ program for a Frontier-CS problem); the rules, the
# formatting example and the command examples are mini-swe-agent's (the macOS note dropped: the sandbox is Linux)
INSTANCE = """Please solve this competitive programming problem: write a C++17 program that reads the input from \
stdin and writes the output to stdout.

<problem>
{problem}
</problem>

You work in an empty directory with a Linux shell (g++, python3 and the usual command-line tools; no internet). \
Your program is scored on hidden tests with the time and memory limits of the problem; you cannot see the hidden \
tests. A program that does not compile, crashes, exceeds a limit or prints an invalid answer scores 0 on a test.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your solution and any possible problems.

1. Analyze the problem and plan an algorithm
2. Write your solution to solution.cpp
3. Compile it: g++ -O2 -std=gnu++17 -o solution solution.cpp
4. Run it on the example(s) from the problem statement and check the output
5. Test edge cases and larger inputs (write a generator, a brute-force or a checker if it helps), and improve your solution
6. Submit your solution and finish your work by issuing the following command: \
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat solution.cpp`.
   The output of that command after its first line is your submission. Do not combine it with any other command. \
<important>After this command, you cannot continue working on this task.</important>

## Important Rules

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks
3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
   However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load \
environment variables from files
4. Each command may run for at most {timeout} seconds. You have at most {steps} steps.

## Formatting your response

Here is an example of a correct response:

<example_response>
THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current \
directory to get a better understanding of the codebase.

```mswea_bash_command
ls -la
```
</example_response>

## Useful command examples

### Create a new file:

```mswea_bash_command
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

```mswea_bash_command
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```mswea_bash_command
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```mswea_bash_command
anything
```
"""
FORMAT_ERROR = """Format error:

<error>
{error}
</error>

Here is general guidance on how to format your response:

Please always provide EXACTLY ONE action in triple backticks, found {n} actions.

Please format your action in triple backticks as shown in <response_example>.

<response_example>
Here are some thoughts about why you want to perform the action.

```mswea_bash_command
<action>
```
</response_example>

If you have completed your assignment, please consult the first message about how to
submit your solution (you will not be able to continue working on this task after that)."""
LENGTH_ERROR = ("Your previous response reached the output token limit (finish_reason=length) before you produced a "
                "complete action, so it was cut off. Respond more concisely and provide exactly one action in the "
                "required format. If you need to think more, do so briefly.")
# the problem prompt of the data files (one-answer rule + "Generate solution code:"): the statement is what lies between
_PREFIX_RE = re.compile(r"^You are a competitive programmer\. Solve the following problem in C\+\+\..*?\n\n", re.DOTALL)
_SUFFIX = "Generate solution code:"


def problem_text(messages: list) -> str:
    text = messages[-1]["content"]
    text = _PREFIX_RE.sub("", text, count=1).rstrip()
    if text.endswith(_SUFFIX):
        text = text[: -len(_SUFFIX)].rstrip()
    return text


def observation(out: dict) -> str:
    """mini-swe-agent's observation template, rendered by hand."""
    s = ""
    if out.get("exception_info"):
        s += f"<exception>{out['exception_info']}</exception>\n"
    s += f"<returncode>{out['returncode']}</returncode>\n"
    o = out["output"]
    if len(o) < OUT_CHARS:
        s += f"<output>\n{o}</output>"
    else:
        s += ("<warning>\nThe output of your last command was too long.\nPlease try a different command that produces "
              "less output.\nIf you're looking at a file you can try use head, tail or sed to view a smaller number of "
              "lines selectively.\nIf you're using grep or find and it produced too much output, you can use a more "
              "selective search pattern.\nIf you really need to see something from the full command's output, you can "
              "redirect output to a file and then search in that file.\n</warning>"
              f"<output_head>\n{o[:5000]}\n</output_head>\n<elided_chars>\n{len(o) - OUT_CHARS} characters elided\n"
              f"</elided_chars>\n<output_tail>\n{o[-5000:]}\n</output_tail>")
    return s


# ---- the sandbox: the judge's Landlock rules for the compiler, plus no TCP --------------------------------------------
_NET_TCP = (1 << 0) | (1 << 1)  # LANDLOCK_ACCESS_NET_BIND_TCP | CONNECT_TCP (ABI >= 4)


def _restrict_net(rules):
    """fcs_judge._restrict with TCP handled (and no TCP rule, so no bind or connect) when the kernel has it."""
    if J.LANDLOCK_ABI < 4:
        return J._restrict(rules)
    scope = J._SCOPE_SIGNAL if J.LANDLOCK_ABI >= 6 else 0
    attr = ctypes.create_string_buffer(struct.pack("<QQQ", J._HANDLED, _NET_TCP, scope) if J.LANDLOCK_ABI >= 6
                                       else struct.pack("<QQ", J._HANDLED, _NET_TCP))
    rs = J._libc.syscall(J._LL_CREATE, attr, ctypes.c_size_t(len(attr.raw)), ctypes.c_uint32(0))
    if rs < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        for path, access in rules:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = ctypes.create_string_buffer(struct.pack("<Qi", access, fd))
                if J._libc.syscall(J._LL_ADD, ctypes.c_int(rs), ctypes.c_int(1), rule, ctypes.c_uint32(0)) < 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule {path}")
            finally:
                os.close(fd)
        if J._libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(NO_NEW_PRIVS)")
        if J._libc.syscall(J._LL_RESTRICT, ctypes.c_int(rs), ctypes.c_uint32(0)) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(rs)


def _preexec(work: str, nproc: int):
    rules = J._compile_sandbox(J.Path(work))

    def f():
        import resource
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT + 1,) * 2)
        resource.setrlimit(resource.RLIMIT_AS, (CMD_MEM, CMD_MEM))
        resource.setrlimit(resource.RLIMIT_FSIZE, (J.STDOUT_MAX, J.STDOUT_MAX))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        if rules is not None:
            _restrict_net(rules)
    return f


def run_command(command: str, work: str) -> dict:
    """One action in the episode's work dir (blocking; run in _POOL). mini-swe-agent's output dict, plus "seconds",
    the command's wall time (the run time that the reward charges)."""
    t0 = time.monotonic()
    out = _run_command(command, work)
    out["seconds"] = time.monotonic() - t0
    return out


def _run_command(command: str, work: str) -> dict:
    env = {"PATH": "/usr/bin:/bin", "LANG": "C", "HOME": work, "TMPDIR": work, "PAGER": "cat", "MANPAGER": "cat",
           "LESS": "-R", "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"}
    try:
        p = subprocess.Popen(["bash", "-c", command], cwd=work, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             preexec_fn=_preexec(work, J._nproc_limit()))
    except Exception as e:
        return {"output": "", "returncode": -1, "exception_info": f"An error occurred while executing the command: {e}"}
    try:
        out, _ = p.communicate(timeout=TIMEOUT)
        return {"output": out.decode("utf-8", errors="replace"), "returncode": p.returncode, "exception_info": ""}
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, 9)
        except ProcessLookupError:
            pass
        out, _ = p.communicate()
        return {"output": (out or b"").decode("utf-8", errors="replace"), "returncode": -1,
                "exception_info": f"An error occurred while executing the command: Command '{command}' timed out "
                                  f"after {TIMEOUT} seconds"}
    finally:
        try:  # background children of the command die with it: each action is a fresh subshell
            os.killpg(p.pid, 9)
        except (ProcessLookupError, PermissionError):
            pass


def _gen(sess: Session) -> int:
    return sum(sum(sg.loss_mask) for sg in sess.segments)


_trace_n = 0


def _trace(rec: dict) -> None:
    if not TRACE_DIR:
        return
    os.makedirs(TRACE_DIR, exist_ok=True)
    with open(os.path.join(TRACE_DIR, f"agent_{os.getpid()}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


async def episode(input: GenerateFnInput, messages: list, label: str):
    sess = Session(input, "agent", THINK, max_len=MAX_LEN)
    chat = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTANCE.format(problem=problem_text(messages), timeout=TIMEOUT, steps=STEPS)}]
    base = os.environ.get("TMPDIR", "/tmp")
    work = os.path.join(base, "agent", uuid.uuid4().hex)
    os.makedirs(work)
    loop = asyncio.get_running_loop()
    steps, fmt_run, fmt_total, timeouts, gpp, cut_n, run_s = 0, 0, 0, 0, 0, 0, 0.0
    exit_status, submission, turns, segs = "", None, [], []
    try:
        while True:
            if steps >= STEPS:
                exit_status = "LimitsExceeded"; break
            if _gen(sess) >= TOKENS:
                exit_status = "TokensExceeded"; break
            steps += 1
            sess.last_messages = None  # a new segment rendered from the visible history (see the module docstring)
            n_seg = len(sess.segments)
            text, cut = await sess.turn(chat, min(TURN, TOKENS - _gen(sess)))
            for tok in _SPECIAL:
                text = text.replace(tok, "")
            if len(sess.segments) > n_seg:
                segs.append(len(sess.segments) - 1)
            vis = "" if cut else answer(text, THINK).strip()
            if not text and cut:  # the context is full: nothing was generated
                exit_status = "ContextExceeded"
                turns.append({"step": steps, "text": "", "cut": True})
                break
            actions = [] if cut else [a.strip() for a in ACTION_RE.findall(vis)]
            rec = {"step": steps, "text": text, "cut": bool(cut)}
            if len(actions) != 1:
                fmt_run, fmt_total, cut_n = fmt_run + 1, fmt_total + 1, cut_n + bool(cut)
                msg = LENGTH_ERROR if cut else FORMAT_ERROR.format(
                    error=f"Expected exactly one action, found {len(actions)}.", n=len(actions))
                turns.append({**rec, "format_error": True})
                chat = chat + [{"role": "assistant", "content": vis or "(cut off)"}, {"role": "user", "content": msg}]
                if fmt_run >= MAX_FORMAT_ERRORS:
                    exit_status = "RepeatedFormatError"; break
                continue
            fmt_run = 0
            cmd = actions[0]
            gpp += bool(re.search(r"\bg\+\+\b", cmd))
            out = await loop.run_in_executor(_POOL, run_command, cmd, work)
            timeouts += "timed out" in out["exception_info"]
            run_s += out["seconds"]
            lines = out["output"].lstrip().splitlines(keepends=True)
            turns.append({**rec, "command": cmd, "returncode": out["returncode"], "output": out["output"][:20000],
                          "exception": out["exception_info"]})
            if lines and lines[0].strip() == MARK and out["returncode"] == 0:
                submission, exit_status = "".join(lines[1:]), "Submitted"
                break
            chat = chat + [{"role": "assistant", "content": vis}, {"role": "user", "content": observation(out)}]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    res = await judge_code(label, submission) if submission and submission.strip() else None
    S = res["score"] if res else 0.0
    score = S - MU * run_s / 60
    gen = _gen(sess)
    info = {"fo_S": S, "fo_R": score, "fo_ag_run_min": run_s / 60, "fo_ag_submitted": float(exit_status == "Submitted"), "fo_ag_steps": steps,
            "fo_ag_format_errors": fmt_total, "fo_ag_cut": cut_n, "fo_ag_timeouts": timeouts, "fo_ag_gpp": gpp,
            "fo_ag_compile_fail": float(res is not None and res["status"] == "compile error"),
            "fo_ag_no_code": float(res is None), "fo_tokens": gen, "fo_latency": gen,
            "fo_ag_context_tokens": max((len(sess.segments[i].tokens) for i in segs), default=0),
            **{f"fo_ag_exit_{e}": float(exit_status == e) for e in
               ("Submitted", "LimitsExceeded", "TokensExceeded", "ContextExceeded", "RepeatedFormatError")},
            "fo_judge_infra_error": float(bool(res and res["infra_error"]))}
    _trace({"time": round(time.time(), 1), "eval": input.evaluation, "game": "agent", "label": label, "S": S, "R": score, "run_min": run_s / 60,
            "status": res["status"] if res else "no submission", "exit": exit_status, "steps": steps,
            "submission": submission, "turns": turns})
    return sess, segs, score, info


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    meta = input.sample.metadata or {}
    messages = meta.get("messages")
    assert messages, "fcs_agent needs the raw chat in sample.metadata['messages']"
    sess, segs, score, info = await episode(input, messages, input.sample.label)
    if not segs:
        raise Aborted("empty episode")
    if input.evaluation:  # the last step stands for the episode, scored by the submission
        out = sess.segments[segs[-1]]
        out.reward = float(score)
        out.metadata = {**(out.metadata or {}), **info}
        return GenerateFnOutput(samples=out)
    # training: every step is a sample with the episode's score (one agent, one group per problem)
    rollout_id = input.sample.rollout_id if input.sample.rollout_id is not None else input.sample.index
    g = input.sample.group_index if input.sample.group_index is not None else input.sample.index
    samples = []
    for i in segs:
        s = sess.segments[i]
        s.reward, s.rollout_id, s.group_index = float(score), rollout_id, g
        samples.append(s)
    samples[0].metadata = {**samples[0].metadata, **info}
    return GenerateFnOutput(samples=samples)


def _means(samples) -> dict:
    acc: dict[str, list[float]] = {}
    for s in samples:
        for k, v in (s.metadata or {}).items():
            if k.startswith("fo_") and isinstance(v, (int, float)) and not isinstance(v, bool) \
                    and not (isinstance(v, float) and math.isnan(v)):
                acc.setdefault(k, []).append(float(v))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    m = {f"rollout/{k}": v for k, v in _means(samples).items()}
    print(f"[fcs_agent] rollout {rollout_id}: {json.dumps({k: round(v, 4) for k, v in m.items()})}", flush=True)
    if isinstance(rollout_extra_metrics, dict):
        rollout_extra_metrics.update(m)
    return False


def log_eval(rollout_id, args, data, extra_metrics) -> bool:
    for key, d in data.items():
        if d.get("samples"):
            m = {f"eval/{key}/{k}": v for k, v in _means(d["samples"]).items()}
            print(f"[fcs_agent] eval {rollout_id} {key}: {json.dumps({k: round(v, 4) for k, v in m.items()})}",
                  flush=True)
            if isinstance(extra_metrics, dict):
                extra_metrics.update(m)
    return False
