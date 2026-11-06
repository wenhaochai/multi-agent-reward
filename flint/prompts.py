"""Everything an agent reads that does not depend on the episode's state: mini-swe-agent 2.4.6's tool-calling config
(mini.yaml; Flint's SWE-Bench evals ran mini-swe-agent, and the text-based format cost it turns to format errors in
smoke 15968124) with our task, workflow and the long game's commands.

Verbatim from mini.yaml: the system message, the command execution rules and example response, the ```bash command
examples, the JSON observation template, and the tool-call error and length templates, except their last line about
ending the task (here the finish command submits and the work goes on). Ours: the task and workflow, the team's
situation, the long game's commands and rules, and the replies that keep an agent working."""
from __future__ import annotations

import json

SYSTEM = "You are a helpful assistant that can interact with a computer.\n"

# {env} is SOLO_ENV or TEAM_ENV, {tools} the long game's commands (TOOLS)
INSTANCE = """Please solve this competitive programming problem: write a C++17 program that reads the input from \
stdin and writes the output to stdout.

<problem>
{problem}
</problem>

{env} \
Your program is scored on hidden tests with the time and memory limits of the problem; you cannot see the hidden \
tests. A program that does not compile, crashes, exceeds a limit or prints an invalid answer scores 0 on a test.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your solution and any possible problems.

1. Analyze the problem and plan an algorithm
2. Write your solution to solution.cpp
3. Compile it: g++ -O2 -std=gnu++17 -o solution solution.cpp
4. Run it on the example(s) from the problem statement and check the output
5. Test edge cases and larger inputs (write a generator, a brute-force or a checker if it helps), and improve your solution
6. Submit with `submit` whenever you have something better, read the provisional score, and keep improving.
7. `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` also submits solution.cpp and shows its provisional score; your work \
then goes on, so keep improving and run it again.

{tools}

## Important Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load \
environment variables from files
- Each command may run for at most {timeout} seconds.

Example of a CORRECT response:
<example_response>
I need to understand the structure of the repository first. Let me check what files are in the current directory to \
get a better understanding of the codebase.

[Makes bash tool call with {{"command": "ls -la"}} as arguments]
</example_response>

## Useful command examples

### Create a new file:

```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

```bash
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

```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```bash
anything
```
"""
SOLO_ENV = ("You work alone in a git repository with a Linux shell (g++, python3, git and the usual command-line "
            "tools; no internet).")
TEAM_ENV = ("You are {name}, one of {n} agents working at the same time on this problem, with a Linux shell (g++, "
            "python3, git and the usual command-line tools; no internet). The agents act in rounds: in each round "
            "every agent runs one command, in a random order. Each agent works in a git repository of its own, "
            "which no one else can change. After every command your repository's source files are committed and "
            "published, and before every command the other agents' latest commits are fetched: team/agent1 to "
            "team/agent{n} are the agents' branches (yours too). To use a teammate's work, run for example `git "
            "log team/agent2`, `git diff team/agent2 -- solution.cpp` or `git checkout team/agent2 -- "
            "solution.cpp` (which replaces your solution.cpp with agent2's).")
TOOLS = """## Long work, submissions and messages

You have a budget of about {budget} generated tokens, and you should use it: keep improving your program for as long \
as you can make it better. Three commands are provided by the harness; each must be the whole action, alone:

- `submit` or `submit <file>`: judges the file (default solution.cpp) right away and shows its PROVISIONAL score \
(0-100): the mean over half of the hidden tests, with the score of each of those tests and any compiler errors. You \
may submit as often as you like.{team_submit}{dup_rule}
- `post <message>`: {post_text}
- `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`: submits solution.cpp, as `submit` does, and your work goes on.

When {when_end}, the submission with the highest provisional score is judged on the OTHER half of the hidden tests; \
that final score is the result. Older steps of this conversation are removed when it gets long, so keep what you need \
to remember in a notes file and re-read it; SCORES.md lists every submission and its provisional score."""
TEAM_SUBMIT = " Every submission of every agent and its provisional score is shown to all agents."
# MA_LG_DUP_END=K (flint/game.py; owner 2026-11-06 「首先prompt里写」): the rule, stated where the submit command is
DUP_RULE = (" Do not submit an unchanged program again: a file unchanged since one of your judged submissions (apart "
            "from comments and layout) is not judged again, and after {k} such submissions in a row your game ends at "
            "once and that submission scores 0.")
TEAM_POST = ("sends the message to every other agent; their messages appear at the top of your next observations, and "
             "BOARD.md keeps them all. Use it to split the work, report what you found and agree on the next step. "
             "After each of your submissions, your next command must be `post <message>`: tell the team what you "
             "changed, its provisional score and what you will try next. Until you post, no other command of yours is "
             "run.")
SOLO_POST = "there are no other agents, so this only writes BOARD.md."
TEAM_END = "every agent has used up its token budget"
SOLO_END = "your token budget is used up"

# mini.yaml's tool-call error, its last line replaced: here the finish command submits and the work goes on
FORMAT_ERROR = """Tool call error:

<error>
{error}
</error>

Here is general guidance on how to submit correct toolcalls:

Every response needs to use the 'bash' tool at least once to execute commands.

Call the bash tool with your command as the argument:
- Tool: bash
- Arguments: {{"command": "your_command_here"}}

To submit, run `submit` alone."""
LENGTH_ERROR = ("Your previous response reached the output token limit (finish_reason=length) before you produced "
                "a tool call, so it was cut off. Respond more concisely and finish with exactly one bash tool call. "
                "If you need to think more, do so briefly.")
OUT_CHARS = 10000   # mini-swe-agent's elision threshold


def observation(out: dict) -> str:
    """mini.yaml's observation template: JSON with the return code and the output (head and tail when long)."""
    o = out["output"]
    if len(o) < OUT_CHARS:
        d = {"returncode": out["returncode"], "output": o}
    else:
        d = {"returncode": out["returncode"], "output_head": o[:5000], "output_tail": o[-5000:],
             "elided_chars": len(o) - 10000, "warning": "Output too long."}
    if out.get("exception_info"):
        d["exception_info"] = out["exception_info"]
    return json.dumps(d, indent=2, ensure_ascii=False)


# what a stop attempt gets: the work goes on until the budget is used up (owner 2026-10-25: "只按预算结束，如果他们提前
# 结束请注入prompt强迫继续"). An unchanged finish command gets KEEP_GOING as its reply; then, and after a run of
# format errors, the agent's conversation starts over from the system message and its instance message with RESTART
# (its files, SCORES.md and BOARD.md stay), so no earlier "the task is complete" is left in its context (smoke w251,
# 2026-10-26: after the reply alone, none of 4 Flints changed solution.cpp again)
KEEP_GOING = ("Your work goes on until your token budget is used up: about {left:,} of your {total:,} tokens are left. "
              "Keep improving solution.cpp: for example, test it on new and larger inputs, compare it with a "
              "brute-force solution on small inputs, measure its running time on the largest inputs, or try a "
              "different algorithm.")
RESTART_SEP = "\n\n[Continued: "
RESTART = (RESTART_SEP + "your earlier conversation on this problem has ended, and this one goes on with the same "
           "work. Your files are still here: solution.cpp, any notes and tests you wrote, SCORES.md (every submission "
           "and its provisional score) and BOARD.md. {best} The task is not finished: your work goes on until your "
           "token budget is used up, and about {left:,} of your {total:,} tokens are left. Read SCORES.md and your "
           "notes first, then keep improving solution.cpp: for example, find out why it scores low on some tests, "
           "test it on new and larger inputs, compare it with a brute-force solution on small inputs, or try a "
           "different algorithm.]")
TRIM_SEP = "\n\n[Note: "
TRIM_NOTE = (TRIM_SEP + "your {dropped} earliest steps were removed from this conversation to fit the context. Your "
             "files, SCORES.md and BOARD.md still hold everything.]")
# the post rule (owner 2026-10-26 popup; four team smokes, 32 agents and 3,196 commands, had no post)
POST_NEXT = ("Your next command must be `post <message>`, alone: tell the team what you changed, this provisional "
             "score and what you will try next.\n")
POST_REFUSED = ("Nothing was run: after each of your submissions, your next command must be `post <message>` with a "
                "message, alone. Tell the team what you changed, its provisional score and what you will try next, "
                "then send your command again.\n")


def instance(problem: str, name: str, n: int, tokens: int, timeout: int, dup_end: int = 0) -> str:
    """The instance message of agent `name` in a game of n agents with `tokens` generated tokens each; dup_end > 0
    states the duplicate-submit rule of MA_LG_DUP_END (0: the message as before, byte for byte)."""
    team = n > 1
    tools = TOOLS.format(budget=f"{tokens:,}", team_submit=TEAM_SUBMIT if team else "",
                         post_text=TEAM_POST if team else SOLO_POST, when_end=TEAM_END if team else SOLO_END,
                         dup_rule=DUP_RULE.format(k=dup_end) if dup_end else "")
    env = TEAM_ENV.format(name=name, n=n) if team else SOLO_ENV
    return INSTANCE.format(problem=problem, env=env, tools=tools, timeout=timeout)
