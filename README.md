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
   message, not clamped, and a problem's score is 100 x the mean ratio. `tests/test_judge_ratio.py` checks the rule
   against the engine's own code. Where the judge runs programs differently (a file sandbox, memory limits), the
   header of `flint/judge.py` says so.
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

## Tests

    ssh della-vis1 'cd <this worktree> && nice -n 19 bash tests/run_all.sh'

The tests need Landlock (della-vis1 or a compute node) and run in `miles-q38-build/miles.sif`, which has Python with
aiohttp and yaml.
