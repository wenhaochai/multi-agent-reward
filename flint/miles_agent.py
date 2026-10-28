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
