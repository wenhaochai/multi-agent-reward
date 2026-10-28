"""One episode of the long game (flint.game) as a miles agent: --custom-agent-function-path flint.miles_agent.run.

miles' agentic generate (miles.rollout.generate_hub.agentic_tool_call) opens a TITO session per episode and passes its
session-scoped base_url; the game runs unchanged against it (flint.chat posts to <base_url>/v1/chat/completions), and
the session server records every generation with its tokens, log-probs and score-centering candidates. The game's
settings come from the environment as for run_eval.py (MA_LG_N=1: one Flint; MA_LG_TOKENS; MA_AG_TOOL_STRICT=0, since
miles records candidates only from unconstrained sampling). The episode's S is the reward of every sample it yields.

How the game's history maps onto the session tree (--use-session-server v2, an append-only tree; a request attaches
to the deepest recorded generation whose message path prefixes it):
- a model turn after a command extends the path, so one sample grows by the observation and the completion;
- a trim (the instance message changes and the oldest steps go) and a restart (a new conversation, files kept) match
  no recorded path, so they start a new root: a new sample;
- a turn cut at its token limit comes back as "(cut off)" with the length error, so the request attaches at the cut
  turn's parent: the cut turn stays a leaf (a sample of its own) and the conversation goes on in a branch.
Each leaf becomes one Sample; a completion shared by several leaves trains once (miles' default post-processing).

message_matches (--session-message-matcher flint.miles_agent.message_matches): the game replays an assistant turn
with tool-call ids of its own and with the text around the tool call stripped. Flint's chat template renders neither,
so that is the recorded turn: the matcher accepts these two differences and otherwise applies miles' strict matcher.
On a match the recorded tokens stay authoritative, so a sample holds exactly the tokens Flint generated.

drop_failed_groups (--rollout-sample-filter-path flint.miles_agent.drop_failed_groups): an episode whose score cannot be
trusted (a submission the judge could not judge after its retries, or an episode that could not go on) has no reward;
its whole group leaves the loss, so it biases no group mean either."""
import json
import os
import random
from types import SimpleNamespace

# what the agent returns besides the reward (merged into every sample's metadata)
INFO_KEYS = ("fo_lg_submissions", "fo_lg_restarts", "fo_lg_tokens", "fo_lg_rounds", "fo_lg_S_auto", "fo_lg_best_prov",
             "fo_lg_first_submit_round", "fo_ag_steps", "fo_ag_format_errors", "fo_lg_wall_min", "fo_lg_llm_min")


def _call(c: dict) -> tuple:
    """A tool call as the template renders it: name and arguments (a JSON object when they parse)."""
    f = c.get("function") or {}
    args = f.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            pass
    return c.get("type"), f.get("name"), args


def message_matches(stored: dict, new: dict) -> bool:
    from miles.utils.chat_template_utils.message_matcher_hub.funcs import strict_message_matches

    if strict_message_matches(stored, new):
        return True
    if stored.get("role") != "assistant" or new.get("role") != "assistant":
        return False
    calls_a, calls_b = stored.get("tool_calls") or [], new.get("tool_calls") or []
    return ((stored.get("reasoning_content") or None) == (new.get("reasoning_content") or None)
            and (stored.get("content") or "").strip() == (new.get("content") or "").strip()
            and len(calls_a) == len(calls_b) and all(_call(a) == _call(b) for a, b in zip(calls_a, calls_b)))


async def run(base_url: str, prompt, request_kwargs: dict, metadata: dict, **kwargs) -> dict:
    """Play one episode on metadata["label"] (the problem directory; metadata["messages"] holds the problem prompt)
    against the session; returns the reward (S) and the episode's numbers. An episode whose score cannot be trusted
    (flint.Aborted: no answer from the server after the retries; or a submission still a judge infra error after its
    retries) carries "flint_failed" (reward 0 as a placeholder only), and drop_failed_groups takes its group out of the
    loss."""
    from flint import Aborted
    from flint import game as G
    from flint.chat import ChatSession, Client

    assert G.N == 1, f"one Flint per episode, got MA_LG_N={G.N}"
    client = Client([base_url], "default", float(request_kwargs.get("temperature", 1.0)),
                    float(request_kwargs.get("top_p", 0.95)), int(request_kwargs.get("top_k", 20)), route=False)
    await client.open()
    inp = SimpleNamespace(evaluation=False, sampling_params={}, base_url=base_url, moved_at=0.0, moves=0,
                          sample=SimpleNamespace(index=0, group_index=random.getrandbits(32), label=metadata["label"],
                                                 metadata=metadata))
    out = {"flint_session": base_url.rsplit("/", 1)[-1], "flint_trace_pid": os.getpid()}
    try:
        _, S, info = await G.episode(inp, metadata["messages"], metadata["label"],
                                     lambda i, name: ChatSession(client, i, name, G.MAX_LEN))
    except Aborted as e:
        return {**out, "reward": 0.0, "flint_failed": f"aborted: {e}"[:500]}
    finally:
        await client.close()
    out = {**out, "reward": S, **{k: info[k] for k in INFO_KEYS}}
    if info["fo_judge_infra_error"]:
        out["flint_failed"] = "judge infra error"
    return out


def drop_failed_groups(args, data) -> None:
    """miles' sample filter: every sample of a group with a failed episode leaves the loss."""
    for group in data:
        if any((s.metadata or {}).get("flint_failed") for s in group):
            for s in group:
                s.remove_sample = True


def _probe(args, ids: list[int]) -> list[float]:
    """The served log-probs of ``ids`` (each token given the ones before it), from an engine behind miles' router."""
    import urllib.request

    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}, "return_logprob": True,
            "logprob_start_len": 0}
    req = urllib.request.Request(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return [x[0] for x in out["meta_info"]["input_token_logprobs"][1:]]


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """miles' rollout log hook (--custom-rollout-log-function-path flint.miles_agent.log_rollout) for the RL runs
    (rl/rl_inner.sh). After each rollout, in $MA_RL_OUT/r<rollout_id>/, where miles saved the rollout (rollout.pt):
    - the episodes' traces move in from $MA_AG_TRACE_DIR (the agents append there; the next rollout starts it anew);
    - rl/inspect_rollout.py checks the samples against the traces; a failed check raises, which ends the run, as does
      a sample over $MA_RL_MAX_SAMPLE_TOKENS (the actor's memory bound), before any training on it;
    - the served log-probs of a fixed sequence (the first 4096 tokens of rollout 0's first sample), twice, show which
      weights the engines serve: they change after a weight update, and the repeat gives the noise floor;
    - one line per rollout goes to $MA_RL_OUT/steps.jsonl: the episodes' rewards (S), submissions and failures, and
      the probe's change since the previous rollout.
    Returns False: miles' own rollout logging follows."""
    import statistics
    import subprocess
    import sys

    out = os.environ["MA_RL_OUT"]
    d = os.path.join(out, f"r{rollout_id}")
    os.makedirs(d, exist_ok=True)
    live = os.environ.get("MA_AG_TRACE_DIR")
    if live and os.path.isdir(live):
        os.rename(live, os.path.join(d, "traces"))

    ids_path = os.path.join(out, "probe_ids.json")
    if not os.path.exists(ids_path):
        json.dump([int(t) for t in samples[0].tokens[:4096]], open(ids_path, "w"))
    ids = json.load(open(ids_path))
    probe = [_probe(args, ids), _probe(args, ids)]
    json.dump(probe, open(os.path.join(d, "probe.json"), "w"))
    rec = {"rollout_id": rollout_id, "rollout_s": round(rollout_time, 1),
           "probe_repeat_max_abs": max(abs(a - b) for a, b in zip(*probe))}
    prev = os.path.join(out, f"r{rollout_id - 1}", "probe.json")
    if os.path.exists(prev):
        p0 = json.load(open(prev))[0]
        diffs = [abs(a - b) for a, b in zip(p0, probe[0])]
        rec |= {"probe_change_mean_abs": sum(diffs) / len(diffs), "probe_change_max_abs": max(diffs)}

    first = {}   # one sample per episode: its reward is the episode's S, its metadata the episode's numbers
    for s in samples:
        first.setdefault((s.metadata or {}).get("flint_session"), s)
    episodes = {k: s.metadata or {} for k, s in first.items()}
    S = [s.reward for s in first.values()]
    rec |= {"episodes": len(episodes), "samples": len(samples), "failed": sum(bool(m.get("flint_failed")) for m in episodes.values()),
            "reward_mean": statistics.fmean(S), "reward_var": statistics.pvariance(S), "S": sorted(S),
            "submissions": [m.get("fo_lg_submissions") for m in episodes.values()],
            "restarts": [m.get("fo_lg_restarts") for m in episodes.values()],
            "wall_min_max": max((m.get("fo_lg_wall_min") or 0) for m in episodes.values()),
            "tokens": sum(len(s.tokens) for s in samples), "max_sample_tokens": max(len(s.tokens) for s in samples)}

    # from the episodes' traces, per episode: cut turns (no command; the last one usually at the token budget's end),
    # those cut at the turn cap (MA_AG_TURN), trims, submissions
    eps = []
    for f in sorted(os.listdir(os.path.join(d, "traces"))) if os.path.isdir(os.path.join(d, "traces")) else []:
        for line in open(os.path.join(d, "traces", f)):
            t = json.loads(line)
            if not t.get("agents"):
                continue
            turns = t["agents"][0].get("turns", [])
            dropped = [x.get("dropped", 0) for x in turns]
            cap = int(os.environ.get("MA_AG_TURN", "32768"))
            eps.append({"S": t.get("S"), "cut_turns": sum(1 for x in turns if x.get("cut")),
                        "turn_cap_hits": sum(1 for x in turns if x.get("cut") and (x.get("completion_tokens") or 0) >= cap),
                        "turns": len(turns), "trims": sum(1 for x, y in zip(dropped, dropped[1:]) if y > x),
                        "submissions": len(t.get("submissions", [])), "restarts": t["agents"][0].get("restarts"),
                        "max_prompt_tokens": max((x.get("prompt_tokens") or 0 for x in turns), default=0)})
    rec |= {"trace_episodes": eps, "cut_turns": sum(e["cut_turns"] for e in eps),
            "turn_cap_hits": sum(e["turn_cap_hits"] for e in eps), "turns": sum(e["turns"] for e in eps)}

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, os.path.join(repo, "rl", "inspect_rollout.py"), d, "rollout.pt"],
                       capture_output=True, text=True)
    open(os.path.join(d, "inspect.txt"), "w").write(r.stdout + r.stderr)
    rec["inspect"] = (r.stdout.strip().splitlines() or ["no output"])[-1]
    with open(os.path.join(out, "steps.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[flint rollout {rollout_id}] {json.dumps(rec)}", flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"rollout {rollout_id}: the sample check failed ({rec['inspect']}); see {d}/inspect.txt")
    cap = int(os.environ.get("MA_RL_MAX_SAMPLE_TOKENS", "0"))
    if cap and rec["max_sample_tokens"] > cap:
        raise RuntimeError(f"rollout {rollout_id}: a sample of {rec['max_sample_tokens']} tokens is over {cap}")
    return False
