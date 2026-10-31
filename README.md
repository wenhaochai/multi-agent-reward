# mas-rl

mas-rl runs teams of Recursive Flint agents and single Flint agents in a long shell game on Frontier-CS, with the
same number of generated tokens per problem, to find out if a team scores higher than one agent.

The older work (the Qwen3.5-9B and molt team studies, EasyPPO training on miles, the one-turn team games) is on the
branches `archive/eval2-2026-10-28` (this code before the cleanup) and `archive/main-2026-10-28` (the old main).

## The game

1. The model is Recursive Flint, a 40B mixture-of-experts model with 2.5B active parameters, served by SGLang.
2. The problems are the classic problems of Frontier-CS v1.1: 40 validation problems (`data/fcs11_val40_agent.jsonl`)
   and 67 training problems (`data/fcs11_train67_agent.jsonl`).
3. Each agent plays mini-swe-agent's tool-calling game: one bash command per step, in a sandbox (Landlock, no
   network, a private /tmp). It writes and compiles files and runs its own tests.
4. `submit` judges a file at once on half of the hidden tests. The agent sees this provisional score. The episode's
   answer is the submission with the best provisional score. Its final score on the other half of the tests, which no
   agent sees, is the result (0 to 100). The judge (`flint/judge.py`) scores like the official Frontier-CS v1.1
   engine (`algorithmic/judge/src/judge_engine.js`): a test's ratio is the first `Ratio:` value in the checker's
   message, not clamped, and a problem's score is 100 x the mean ratio. One deviation: a `Ratio:` label whose value
   the engine cannot read (a negative, nan, inf) gives 0, where the engine gives 1 to a checker that accepted. No v1.1
   checker prints such a value today. `tests/test_judge_ratio.py` checks the rule against the engine's own code. Where
   the judge runs programs differently (a file sandbox, memory limits), the header of `flint/judge.py` says so.
5. A team is 4 agents in lockstep rounds. Each agent works in a git repository of its own and can read its teammates'
   branches (`team/agent1` to `team/agent4`). `post <message>` sends a message to the others.

Two rules make the game work for Flint. Both come from test runs:

- **Forced restart.** Only the token budget ends an agent's work. When an agent tries to stop, its conversation starts
  again from the problem, with its files kept. Reason: one Flint at 500k tokens stopped after 107k tokens on average
  when it could stop. Told to go on in the same conversation, it only submitted the same program again.
- **Forced posts.** In a team, after each submission an agent must post a message before it can run another command.
  Reason: in 4 team test runs (32 agents, 3,196 commands) no Flint posted a message when posting was free.

## Code

| Path | What it does |
|---|---|
| `flint/game.py` | The game: rounds, submit, post, the finish command, restarts, context trimming, checkpoints, traces |
| `flint/prompts.py` | Every fixed text an agent reads |
| `flint/chat.py` | The model calls (OpenAI-compatible chat API, tool calls) and the routing of episodes over servers |
| `flint/repo.py` | The team's git repositories |
| `flint/sandbox.py` | The command sandbox |
| `flint/judge.py` | The local Frontier-CS judge |
| `run_eval.py` | The driver: runs the episodes of a data file against served Flint, resumable |
| `jobs/` | wq jobs on Della: `arm.sh`, `eval.sh`, `smoke.sh`, `smoke_wq.sh` |
| `flint/miles_agent.py` | One episode as a miles agent against a TITO session (RL rollouts), and its session message matcher |
| `serving/` | The Flint SGLang plugin ported to miles' SGLang 0.5.23, and its checks |
| `rl/` | RL through miles v0.1.3 (rollouts colocated with the FSDP actor of `flint_hf/`), the rollout-only runs, miles' GPU TITO check, the samples' checks |
| `tests/` | The tests: `bash tests/run_all.sh` |

## Run

The jobs run in this worktree on Della (`/scratch/gpfs/GROUP/USER/project/miles-q38-build-clean`), on 8 H100s,
through wq. Results go to `flint-eval/runs/`, one directory per arm.

An arm of the main comparison (budget = generated tokens per problem):

    wq submit --partial -J flint-team-1m --project flint-eval jobs/arm.sh team 1m    # 4 Flints, 250k each
    wq submit --partial -J flint-solo-1m --project flint-eval jobs/arm.sh solo 1m    # 1 Flint, 1M
    wq submit --partial -J flint-apart-1m --project flint-eval jobs/arm.sh apart 1m  # 4 independent Flints, 250k each

Each arm first runs a smoke of its setup (2 episodes, 150k tokens per agent). To run the smoke as its own 4-GPU job
before the arms, so the arms skip theirs:

    wq submit --prio 10 --project flint-eval jobs/smoke_wq.sh team w<arm> [w<arm> ...]

`jobs/eval.sh SPLIT TOKENS EPISODES agents=N` runs any other eval.

Rollouts through miles (one Flint per episode on the RL test problem, 4 H100s, two engines at TP2; miles from our fork
worktree `miles-rollout`, branch flint-grpo of wenhaochai/miles-private):

    wq submit --prio 10 --time 3:00:00 --project flint-eval rl/rollout_job.sh full   # 8 x 128k with and without top-32

RL on the same problem, one 8-GPU block (4 engines at TP2 colocated with the FSDP actor; GRPO without std
normalization, score centering over the top-32, routing replay, one step per 8 episodes of 128k, lr 2e-7). The game
runs with RL's context cap (110k characters, 16k-token turns; the eval's are 400k and 32k), so that a training sample
stays near the 64k tokens the actor step was measured at:

    wq submit --prio 10 --project flint-eval rl/rl_job.sh test2      # 2 RL steps
    wq submit --partial --prio 10 --time 23:00:00 --project flint-eval rl/rl_job.sh run20   # 20 steps, runs/run20_p0

Updates are judged by KL(rollout || train) and S. The served-weights probe (log-probs of 1,105 prompt tokens) only
shows that new weights reached the engines: the same weights served at TP8 instead of TP2 move it by 0.517 nats on
average (w662 vs w632), as much as one training step does, so it measures MoE routing sensitivity, not update size.
    rl/steps.py <run dir>                                             # the per-step table

RL on the 67 training problems (`rl/rl_job.sh train67*`, branch `rl-train67b`, worktree `mas-rl-train67b`: the job
scripts run the code of this worktree). Each step uses P problems x G
episodes (default 4 x 8 = 32 episodes). The episodes run on one TP8 engine colocated with the actor. Each problem is
one GRPO group. The problems come from `data/fcs11_train67_agent.jsonl` in a fixed order (`rl/train67_data.py`, seed
67). Problem 112 stays in because its fixed checker is in `flint/checkers/112`. The game settings are run20's, except
the wall cap: it is 60 min, not 45 min, because 32 episodes share one engine. The stop rules use fractions of the
episodes in a rollout:

1. The run stops if more than 25% of the episodes end at the wall cap in 2 rollouts in a row.
2. The run stops if more than 25% of the episodes leave the loss (each failed episode removes its group).

Every 5th rollout writes its samples (`r<k>/samples/`). `steps.jsonl` records the S mean of each problem and the
fraction of episodes that submitted.

Two settings come from the trace analysis of 2026-10-31 (`tmp/rl_setup_traces/T1_report.md`, `T2_report.md`; owner
popups of 2026-10-31):

1. `set=NAME` trains on the problems of `data/train67_sets/NAME.txt` only. The default is all 67. `signal34` keeps 34
   problems: 31 'mixed' and 3 'saturated'. It drops 20 'dead' problems (S < 0.01 in both 128k episodes), 7 'flat'
   problems (the same S in 3 runs) and 6 'mixed' problems whose checker scores against a hidden baseline (150, 152,
   158, 161, 165, 167). A set run has its own directories (`runs/train67[_test]_NAME_p<P>g<G>`), so it never resumes a
   67-problem run.
2. The train67 modes do not judge a resubmission of unchanged code (`MA_LG_DEDUP_SUBMIT=1`, in `rl/rl_inner.sh`). When
   the file equals one of the agent's own judged submissions, the judge is not called and no submission is recorded.
   The reply names that submission and its provisional score. S and the best provisional do not change.
   `steps.jsonl` records `dup_submits` per episode. Reason: in run20 rollout 17 one episode submitted the same
   solution.cpp 124 times in a row and got a positive advantage. run20, base_tp8 and the evals judge every submit.

    wq submit --prio 10 --time 1:45:00 --project flint-eval rl/rl_job.sh train67_smoke   # 2 steps x 32 x 4k
    wq submit --partial --prio 10 --time 6:00:00 --project flint-eval rl/rl_job.sh train67_test   # 2 steps x 32 x 128k
    wq submit --partial --time 23:00:00 --project flint-eval rl/rl_job.sh train67 4 8 40   # runs/train67_p4g8
    wq submit --partial --time 23:00:00 --project flint-eval rl/rl_job.sh train67 4 8 40 set=signal34   # runs/train67_signal34_p4g8
    ssh della-vis1 'cd <this worktree> && nice -n 19 bash rl/args_all.sh [P G] [set=NAME]'   # every mode's argv, the B=32 schedule

## Tests

    ssh della-vis1 'cd <this worktree> && nice -n 19 bash tests/run_all.sh'

The tests need Landlock (della-vis1 or a compute node) and run in `miles-q38-build/miles.sif`, which has Python with
aiohttp and yaml.
