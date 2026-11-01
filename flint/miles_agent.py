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
    settings = None
    if os.environ.get("MA_RL_OUT"):   # the RL job's settings (rl/rl_inner.sh) must be the game's in this process
        out_dir = os.environ["MA_RL_OUT"]
        got = {"MA_LG_N": G.N, "MA_LG_TOKENS": G.TOKENS, "MA_LG_CTX_CHARS": G.CTX_CHARS, "MA_AG_TURN": G.TURN,
               "MA_LG_WALL_MIN": G.WALL_MIN, **({"MA_LG_DEDUP_SUBMIT": 1} if G.DEDUP_SUBMIT else {})}
        os.makedirs(os.path.join(out_dir, "settings_seen"), exist_ok=True)
        json.dump(got, open(os.path.join(out_dir, "settings_seen", f"{os.getpid()}.json"), "w"))   # startup_probe reads it
        expected = {k: float(v) for k, v in json.load(open(os.path.join(out_dir, "game_settings.json"))).items()}
        if got != expected:   # miles only logs an agent's exception, so the episode fails its group instead
            settings = f"the game's settings {got} are not the job's {expected}"
    client = Client([base_url], "default", float(request_kwargs.get("temperature", 1.0)),
                    float(request_kwargs.get("top_p", 0.95)), int(request_kwargs.get("top_k", 20)), route=False)
    await client.open()
    inp = SimpleNamespace(evaluation=False, sampling_params={}, base_url=base_url, moved_at=0.0, moves=0,
                          sample=SimpleNamespace(index=0, group_index=random.getrandbits(32), label=metadata["label"],
                                                 metadata=metadata))
    out = {"flint_session": base_url.rsplit("/", 1)[-1], "flint_trace_pid": os.getpid()}
    if settings:
        await client.close()
        return {**out, "reward": 0.0, "flint_failed": settings}
    try:
        _, S, info = await G.episode(inp, metadata["messages"], metadata["label"],
                                     lambda i, name: ChatSession(client, i, name, G.MAX_LEN))
    except Aborted as e:
        return {**out, "reward": 0.0, "flint_failed": f"aborted: {e}"[:500]}
    finally:
        await client.close()
    out = {**out, "reward": S, **{k: info[k] for k in INFO_KEYS}}
    if "fo_lg_dup_submits" in info:   # MA_LG_DEDUP_SUBMIT=1 (train67)
        out["fo_lg_dup_submits"] = info["fo_lg_dup_submits"]
    if info["fo_judge_infra_error"]:
        out["flint_failed"] = "judge infra error"
    return out


def _samples(node) -> list:
    return [s for x in node for s in _samples(x)] if isinstance(node, list) else [node]


def _train_steps(train_log: str) -> dict[int, dict]:
    """miles' per-step training metrics ("step N: {...}" in train.log; a resumed step's last record wins)."""
    import ast
    import re

    steps = {}
    for line in open(train_log, errors="replace") if os.path.exists(train_log) else []:
        m = re.search(r"\bstep (\d+): (\{.*\})\s*$", line)
        if m and "train/grad_norm" in m.group(2):
            try:
                steps[int(m.group(1))] = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                pass
    return steps


def _stop_rule(out: str, rollout_id: int) -> str | None:
    """The stop rules, checked after each rollout: the last training step's KL(rollout || train) over 0.01; the grad
    norm (before clipping) over 10x step 0's in the last 2 steps; every episode at S = 0 in the last 3 rollouts; the
    wall cap (MA_LG_WALL_MIN) binding often in each of the last 2 rollouts (it would teach the policy not to submit):
    run20 (coordinator 2026-10-29), 8 episodes per rollout: more than 1 episode; with MA_RL_WALL_STOP_FRAC (train67):
    more than that fraction of the rollout's episodes. With MA_RL_MAX_DROPPED_FRAC (train67) also: more than that
    fraction of the rollout's episodes out of the loss (failed episodes take their whole group out). Returns the
    reason, or None."""
    steps = _train_steps(os.path.join(out, "train.log"))
    if steps:
        last = steps[max(steps)]
        if last.get("train/train_rollout_kl", 0) > 0.01:
            return f"step {max(steps)}: KL(rollout || train) {last['train/train_rollout_kl']:.4f} > 0.01"
        g0 = steps.get(0, {}).get("train/grad_norm")
        recent = [steps[k]["train/grad_norm"] for k in sorted(steps)[-2:]]
        if g0 and len(steps) >= 3 and len(recent) == 2 and all(g > 10 * g0 for g in recent):
            return f"grad norm {recent} > 10 x step 0's {g0:.2f} in 2 steps"
    rollouts = {}
    for line in open(os.path.join(out, "steps.jsonl")):
        r = json.loads(line)
        rollouts[r["rollout_id"]] = r
    last2 = [rollouts.get(i) for i in (rollout_id - 1, rollout_id)]
    frac = os.environ.get("MA_RL_WALL_STOP_FRAC")
    if frac:
        if all(r is not None and r.get("wall_exceeded", 0) > float(frac) * max(len(r["S"]), 1) for r in last2):
            return (f"more than {float(frac):.0%} of the episodes ended at the wall cap in rollouts {rollout_id - 1}-"
                    f"{rollout_id} ({[r['wall_exceeded'] for r in last2]} of {[len(r['S']) for r in last2]})")
    elif all(r is not None and r.get("wall_exceeded", 0) > 1 for r in last2):
        return f"more than 1 episode ended at the wall cap in rollouts {rollout_id - 1}-{rollout_id}"
    last3 = [rollouts.get(i) for i in range(rollout_id - 2, rollout_id + 1)]
    if all(r is not None and max(r["S"]) == 0 for r in last3):
        return f"every episode at S = 0 in rollouts {rollout_id - 2}-{rollout_id}"
    drop = os.environ.get("MA_RL_MAX_DROPPED_FRAC")
    r = rollouts.get(rollout_id)
    if drop and r is not None and r.get("dropped_episodes", 0) > float(drop) * max(len(r["S"]), 1):
        return (f"rollout {rollout_id}: {r['dropped_episodes']} of {len(r['S'])} episodes out of the loss (failed "
                f"{r.get('failed')}), more than {float(drop):.0%}")
    return None


# the trainer's train/* consistency numbers of the last step against run20's maximum over its 20 steps
# (tmp/rollout_eff/consistency: the gate's tier 4); over a limit, log_rollout warns, it does not stop the run
CONSISTENCY_LIMITS = {"train/train_rollout_logprob_abs_diff": 0.0099, "train/train_rollout_kl": 3.4e-4,
                      "train/r3_token_mismatch": 0.146, "train/r3_pick_mismatch": 0.031}


def _consistency(out: str) -> dict:
    """The last training step's consistency numbers and the ones over CONSISTENCY_LIMITS (empty before step 0)."""
    steps = _train_steps(os.path.join(out, "train.log"))
    if not steps:
        return {}
    last = steps[max(steps)]
    return {"consistency_step": max(steps),
            "consistency": {k.split("/")[1]: last[k] for k in CONSISTENCY_LIMITS if k in last},
            "consistency_over": [k.split("/")[1] for k, lim in CONSISTENCY_LIMITS.items() if last.get(k, 0) > lim]}


def _keep_milestone(root: str, it: int, milestones: str) -> None:
    """The model part of root/iter_NNNNNNN (model/ and meta.json; no optimizer, scheduler or RNG) as
    milestones/iter_NNNNNNN: hard links (the same disk blocks, no copy; the checkpoint's own files stay as they are),
    built in a .partial directory and renamed when whole, so a milestone directory is always complete. An existing one
    is kept as it is."""
    import shutil

    name = f"iter_{it:07d}"
    dst = os.path.join(milestones, name)
    if os.path.isdir(dst):
        return
    src = os.path.join(root, name)
    part = dst + ".partial"
    shutil.rmtree(part, ignore_errors=True)
    os.makedirs(part)
    shutil.copytree(os.path.join(src, "model"), os.path.join(part, "model"), copy_function=os.link)
    if os.path.exists(os.path.join(src, "meta.json")):
        os.link(os.path.join(src, "meta.json"), os.path.join(part, "meta.json"))
    os.rename(part, dst)
    print(f"[flint] milestone {dst} (model only, hard links)", flush=True)


def _keep_newest_checkpoints(root: str, keep: int, every: int = 0) -> None:
    """Delete miles' checkpoints (root/iter_NNNNNNN) but the newest ``keep`` and the one the tracker names. With
    ``every`` > 0, each checkpoint whose iteration is a multiple of it first leaves its model part in
    <root>/../milestones/iter_NNNNNNN (_keep_milestone), when it is whole: not the newest one, which miles may still be
    writing (it writes model/ first, then the optimizer, then the tracker), unless the tracker names it."""
    import re
    import shutil

    if not os.path.isdir(root):
        return
    tracker = os.path.join(root, "latest_checkpointed_iteration.txt")
    named = int(open(tracker).read().strip()) if os.path.exists(tracker) else None
    iters = sorted(int(m.group(1)) for x in os.listdir(root) if (m := re.fullmatch(r"iter_(\d{7})", x)))
    if every > 0:
        milestones = os.path.join(os.path.dirname(os.path.abspath(root)), "milestones")
        for it in iters:
            if it % every == 0 and named is not None and it <= named and \
                    os.path.isdir(os.path.join(root, f"iter_{it:07d}", "model")):
                _keep_milestone(root, it, milestones)
    for it in iters[:-keep]:
        if it != named and os.path.isdir(os.path.join(root, f"iter_{it:07d}")):
            shutil.rmtree(os.path.join(root, f"iter_{it:07d}"))


def drop_failed_groups(args, data) -> None:
    """miles' sample filter: every sample of a group with a failed episode leaves the loss. A group holds one entry per
    episode, a list of the episode's samples (one per session-tree leaf)."""
    for group in data:
        samples = _samples(group)
        if any((s.metadata or {}).get("flint_failed") for s in samples):
            for s in samples:
                s.remove_sample = True


def _post(args, path: str, body: dict) -> tuple[int, object]:
    """POST JSON to miles' router (sgl-router): (status, the parsed reply or the error body as text). Its consistent
    hashing policy routes by the X-SMG-Routing-Key header and refuses a request without one (w612: 400
    missing_routing_key); a fixed key sends every probe to the same engine."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(f"http://{args.sglang_router_ip}:{args.sglang_router_port}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-SMG-Routing-Key": "flint-probe"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:2000]


def _probe(args, ids: list[int], messages: list) -> dict:
    """What the engines serve now: the log-probs of ``ids`` (each token given the ones before it, /generate), or, if
    /generate refuses the request, the top-20 log-probs of the first greedy token after ``messages``
    (/v1/chat/completions, the agents' endpoint). Both are deterministic for fixed weights."""
    status, out = _post(args, "/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                                            "return_logprob": True, "logprob_start_len": 0})
    if status == 200:
        return {"kind": "generate", "lp": [x[0] for x in out["meta_info"]["input_token_logprobs"][1:]]}
    refused = f"/generate {status}: {out}"
    status, out = _post(args, "/v1/chat/completions", {"model": "default", "messages": messages, "max_tokens": 1,
                                                       "temperature": 0.0, "logprobs": True, "top_logprobs": 20})
    if status != 200:
        raise RuntimeError(f"probe: {refused}; /v1/chat/completions {status}: {out}")
    top = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return {"kind": "chat", "top": {x["token"]: x["logprob"] for x in top}, "generate_refused": refused}


def _probe_engines(args, ids: list[int]) -> dict[str, list[float]]:
    """Elastic runs (MA_ENGINE_REGISTRY): the probe's /generate on every engine the router routes to, each by its own
    URL (the router's probe reaches one engine). After a rollout no request runs, so engines of one serving shape that
    hold the same weights return the same log-probs; an engine that missed a weight push differs by the size of a step."""
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers", timeout=60) as r:
        urls = sorted({w["url"].rstrip("/") for w in json.loads(r.read())["workers"]})
    out = {}
    body = json.dumps({"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                       "return_logprob": True, "logprob_start_len": 0}).encode()
    for url in urls:
        req = urllib.request.Request(f"{url}/generate", data=body, headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=600) as r:
            out[url] = [x[0] for x in json.loads(r.read())["meta_info"]["input_token_logprobs"][1:]]
    return out


def startup_probe(out: str, train_log: str, model: str) -> int:
    """Before rollout 0 (rl/rl_inner.sh, once the engines serve): the agents' game settings (each episode writes them)
    against the job's, and the probe on the initial weights, so that either problem ends the run in minutes. The probe sequence is the problem prompt as Flint's chat template renders it, fixed
    for the run ($MA_RL_OUT/probe_ids.json, which log_rollout reuses). Returns 0 when the probe works."""
    import re
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    try:
        m = re.findall(r"Router ready at host='([^']+)' port=(\d+)", open(train_log, errors="replace").read())
        args = SimpleNamespace(sglang_router_ip=m[0][0], sglang_router_port=int(m[0][1]))
        messages = json.loads(open(_data_file(out)).readline())["prompt"]
        ids = AutoTokenizer.from_pretrained(model).apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        ids = [int(t) for t in (ids["input_ids"] if hasattr(ids, "keys") else ids)]   # transformers 5: a BatchEncoding
        json.dump(ids, open(os.path.join(out, "probe_ids.json"), "w"))
        import time

        expected = {k: float(v) for k, v in json.load(open(os.path.join(out, "game_settings.json"))).items()}
        seen_dir = os.path.join(out, "settings_seen")
        for _ in range(60):   # the agents that made the first requests write their game's settings at episode start
            seen = [json.load(open(os.path.join(seen_dir, f))) for f in os.listdir(seen_dir)] if os.path.isdir(seen_dir) else []
            if seen:
                break
            time.sleep(5)
        if not seen or any(x != expected for x in seen):
            raise RuntimeError(f"agents' game settings {seen[:2]} are not the job's {expected}")
        probe = [_probe(args, ids, messages), _probe(args, ids, messages)]
        json.dump(probe, open(os.path.join(out, "probe_start.json"), "w"))
        print(f"[flint probe] start: {probe[0]['kind']}, {len(ids)} tokens, repeat max |diff| "
              f"{max(_probe_diffs(*probe), default=None)}", flush=True)
        return 0 if probe[0]["kind"] == "generate" else 1
    except Exception as e:
        print(f"[flint probe] start FAILED: {type(e).__name__}: {e}", flush=True)
        return 1


def _data_file(out: str) -> str:
    """The run's prompt file (rl/rl_inner.sh: MA_RL_DATA; run20's data_p0.jsonl by default): the probe's prompt is
    its first row, train67's planned problems its rows in order."""
    return os.environ.get("MA_RL_DATA") or os.path.join(out, "data_p0.jsonl")


def _pid(meta: dict) -> str:
    return str(meta.get("pid") or os.path.basename(str(meta.get("label") or "")) or "?")


def _per_problem(episodes: dict) -> dict:
    """Per problem (a GRPO group, train67): its episodes' S (sorted), mean, the fraction that submitted, failures."""
    import statistics

    by = {}
    for m, S in episodes.values():
        by.setdefault(_pid(m), []).append((m, S))
    return {pid: {"S_mean": statistics.fmean(S for _, S in xs), "S": sorted(S for _, S in xs), "episodes": len(xs),
                  "submitted_frac": sum((m.get("fo_lg_submissions") or 0) > 0 for m, _ in xs) / len(xs),
                  "failed": sum(bool(m.get("flint_failed")) for m, _ in xs)}
            for pid, xs in sorted(by.items(), key=lambda kv: (len(kv[0]), kv[0]))}


def _planned(out: str, rollout_id: int, p: int) -> list[str] | None:
    """The problems the data file plans for this rollout (rows rollout_id*p ... +p-1; miles reads it in order and saves
    its position with each checkpoint), or None without a file."""
    f = _data_file(out)
    if not os.path.exists(f):
        return None
    rows = []
    with open(f) as fh:
        for i, line in enumerate(fh):
            if i >= rollout_id * p + p:
                break
            if i >= rollout_id * p:
                rows.append(_pid(json.loads(line)["metadata"]))
    return rows


def _inspect(d: str, samples: list) -> tuple[dict, dict, list[str]]:
    """rl/inspect_rollout.py's check() on this rollout's samples, in this process (tests replace it)."""
    import importlib.util

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("flint_inspect_rollout", os.path.join(repo, "rl", "inspect_rollout.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check(d, samples)


def _dump_samples(d: str, samples: list) -> None:
    """The rollout's samples as miles' debug dump would hold them, one file per episode (r<k>/samples/<i>.pt), so the
    copy that pickling makes stays one episode's size (~0.7 GB) instead of the whole batch's."""
    import torch

    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    by = {}
    for s in samples:
        by.setdefault((s.metadata or {}).get("flint_session"), []).append(s)
    for i, (_, ss) in enumerate(sorted(by.items(), key=lambda kv: str(kv[0]))):
        torch.save({"samples": [s.to_dict() for s in ss]}, os.path.join(d, "samples", f"{i:03d}.pt"))


def _probe_diffs(a: dict, b: dict) -> list[float]:
    if a["kind"] == "generate":
        return [abs(x - y) for x, y in zip(a["lp"], b["lp"])]
    return [abs(a["top"][k] - b["top"][k]) for k in a["top"] if k in b["top"]]


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """miles' rollout log hook; any failure ends the run as a RuntimeError with a plain message (Ray pickles it to the
    driver; w606's urllib HTTPError held an open response and could not be pickled)."""
    try:
        return _log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time)
    except Exception as e:
        raise RuntimeError(f"log_rollout, rollout {rollout_id}: {type(e).__name__}: {e}") from None


def _log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """The rollout log hook (--custom-rollout-log-function-path flint.miles_agent.log_rollout) for the RL runs
    (rl/rl_inner.sh). After each rollout, in $MA_RL_OUT/r<rollout_id>/, where miles saved the rollout (rollout.pt):
    - the episodes' traces move in from $MA_AG_TRACE_DIR (the agents append there; the next rollout starts it anew);
    - rl/inspect_rollout.py checks the samples against the traces; a failed check raises, which ends the run, as does
      a sample over $MA_RL_MAX_SAMPLE_TOKENS (the actor's memory bound), before any training on it;
    - the served log-probs of a fixed sequence (the first 4096 tokens of rollout 0's first sample), twice, show which
      weights the engines serve: they change after a weight update, and the repeat gives the noise floor;
    - one line per rollout goes to $MA_RL_OUT/steps.jsonl: the episodes' rewards (S), submissions and failures, and
      the probe's change since the previous rollout.
    With MA_RL_P (train67: P problems per rollout, one GRPO group each) the hook also logs per problem (S mean, the
    fraction of episodes that submitted, failures) and the problems the data file planned; it checks the samples in
    this process (no rollout.pt is read back: rl/rl_inner.sh drops miles' per-step dump) and writes them only every
    MA_RL_DUMP_EVERY rollouts (r<k>/samples/); failed episodes take their groups out of the loss and fail the run only
    through the stop rule's MA_RL_MAX_DROPPED_FRAC (a checkpoint first), every other sample problem still ends it.
    Returns False: miles' own rollout logging follows."""
    import statistics
    import subprocess
    import sys

    P = int(os.environ.get("MA_RL_P", "0"))

    out = os.environ["MA_RL_OUT"]
    d = os.path.join(out, f"r{rollout_id}")
    os.makedirs(d, exist_ok=True)
    live = os.environ.get("MA_AG_TRACE_DIR")
    if live and os.path.isdir(live):
        if os.path.isdir(os.path.join(d, "traces")):   # this rollout ran before, in a segment that stopped
            os.rename(os.path.join(d, "traces"), os.path.join(d, f"traces.before_{int(os.path.getmtime(os.path.join(d, 'traces')))}"))
        os.rename(live, os.path.join(d, "traces"))

    ids_path = os.path.join(out, "probe_ids.json")
    if not os.path.exists(ids_path):
        json.dump([int(t) for t in samples[0].tokens[:4096]], open(ids_path, "w"))
    ids = json.load(open(ids_path))
    messages = json.loads(open(_data_file(out)).readline())["prompt"]
    probe = [_probe(args, ids, messages), _probe(args, ids, messages)]
    json.dump(probe, open(os.path.join(d, "probe.json"), "w"))
    rec = {"rollout_id": rollout_id, "rollout_s": round(rollout_time, 1), "probe_kind": probe[0]["kind"],
           "probe_repeat_max_abs": max(_probe_diffs(*probe), default=None)}
    if "generate_refused" in probe[0]:
        rec["probe_generate_refused"] = probe[0]["generate_refused"]
    if os.environ.get("MA_ENGINE_REGISTRY") and probe[0]["kind"] == "generate":
        # elastic: every engine against the router's probe engine; mean |diff| over 0.05 (a step moves the probe by
        # ~0.5) is an engine on other weights, and the run stops before it trains on that engine's samples
        per = _probe_engines(args, ids)
        json.dump(per, open(os.path.join(d, "probe_engines.json"), "w"))
        means = {u: sum(abs(x - y) for x, y in zip(lp, probe[0]["lp"])) / max(len(lp), 1) for u, lp in per.items()}
        rec |= {"probe_engines": len(per), "probe_engines_mean_abs": means,
                "probe_engines_max_abs": max((max((abs(x - y) for x, y in zip(lp, probe[0]["lp"])), default=0.0)
                                              for lp in per.values()), default=None)}
        stale = {u: m for u, m in means.items() if m > 0.05}
        if stale:
            raise RuntimeError(f"engines serve other weights than the probe engine (mean |diff| > 0.05): {stale}")
    prev = os.path.join(out, f"r{rollout_id - 1}", "probe.json")
    if os.path.exists(prev) and json.load(open(prev))[0]["kind"] == probe[0]["kind"]:
        diffs = _probe_diffs(json.load(open(prev))[0], probe[0])
        rec |= {"probe_change_mean_abs": sum(diffs) / max(len(diffs), 1), "probe_change_max_abs": max(diffs, default=None)}

    first = {}   # one sample per episode: its reward is the episode's S, its metadata the episode's numbers
    for s in samples:
        first.setdefault((s.metadata or {}).get("flint_session"), s)
    episodes = {k: s.metadata or {} for k, s in first.items()}
    S = [s.reward for s in first.values()]
    rec |= {"episodes": len(episodes), "samples": len(samples), "failed": sum(bool(m.get("flint_failed")) for m in episodes.values()),
            "reward_mean": statistics.fmean(S), "reward_var": statistics.pvariance(S), "S": sorted(S),
            "submissions": [m.get("fo_lg_submissions") for m in episodes.values()],
            "submitted_frac": sum((m.get("fo_lg_submissions") or 0) > 0 for m in episodes.values()) / len(episodes),
            "restarts": [m.get("fo_lg_restarts") for m in episodes.values()],
            **({"dup_submits": [m.get("fo_lg_dup_submits") for m in episodes.values()]}
               if any("fo_lg_dup_submits" in m for m in episodes.values()) else {}),
            "wall_min_max": max((m.get("fo_lg_wall_min") or 0) for m in episodes.values()),
            "tokens": sum(len(s.tokens) for s in samples), "max_sample_tokens": max(len(s.tokens) for s in samples)}
    if P:   # train67: one GRPO group per problem; a failed episode takes its whole group out of the loss
        groups = {}
        for k, s in first.items():
            groups.setdefault(s.group_index if getattr(s, "group_index", None) is not None else _pid(s.metadata or {}), []).append(k)
        dropped = [g for g, ks in groups.items() if any(episodes[k].get("flint_failed") for k in ks)]
        per = _per_problem({k: (episodes[k], first[k].reward) for k in first})
        rec |= {"problems": list(per), "planned": _planned(out, rollout_id, P), "per_problem": per,
                "groups": len(groups), "dropped_groups": len(dropped),
                "dropped_episodes": sum(len(groups[g]) for g in dropped),
                "S_mean_by_problem": {pid: round(v["S_mean"], 4) for pid, v in per.items()},
                "submitted_frac_by_problem": {pid: v["submitted_frac"] for pid, v in per.items()},
                "groups_with_signal": sum(1 for v in per.values() if len(set(v["S"])) > 1)}

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
                        "exit": t["agents"][0].get("exit"),
                        "max_prompt_tokens": max((x.get("prompt_tokens") or 0 for x in turns), default=0)})
    rec |= {"trace_episodes": eps, "cut_turns": sum(e["cut_turns"] for e in eps),
            "turn_cap_hits": sum(e["turn_cap_hits"] for e in eps), "turns": sum(e["turns"] for e in eps),
            "wall_exceeded": sum(e["exit"] == "WallExceeded" for e in eps),
            "wall_exceeded_S": [e["S"] for e in eps if e["exit"] == "WallExceeded"]}

    # the job's memory now and its peak so far (rl/rl_job.sh samples its cgroup every 10 s into host_mem.txt: time,
    # node used MiB, cgroup MiB and, since 2026-11-01, the cgroup's non-reclaimable MiB, the part its guard reads)
    rows = [x.split() for x in open(os.path.join(out, "host_mem.txt"))] \
        if os.path.exists(os.path.join(out, "host_mem.txt")) else []
    mem = [int(r[2]) for r in rows if len(r) in (3, 4)]
    nonrec = [int(r[3]) for r in rows if len(r) == 4]
    rec |= {"job_mem_gb": round(mem[-1] / 1024, 1) if mem else None, "job_mem_peak_gb": round(max(mem) / 1024, 1) if mem else None,
            "job_nonrec_peak_gb": round(max(nonrec) / 1024, 1) if nonrec else None}

    if P:   # in this process, on the samples miles holds anyway; failed episodes are the stop rule's to judge
        summary, problems, lines = _inspect(d, samples)
        bad = {k: v for k, v in problems.items() if v and k not in ("failed episodes", "judge infra submissions")}
        rec["inspect"] = f"INSPECT_FAIL {bad}" if bad else "INSPECT_OK" + (
            f" (failed episodes {problems['failed episodes']}, out of the loss)" if problems["failed episodes"] else "")
        open(os.path.join(d, "inspect.txt"), "w").write("\n".join(lines + [rec["inspect"]]) + "\n")
        r = subprocess.CompletedProcess([], 1 if bad else 0)
        every = int(os.environ.get("MA_RL_DUMP_EVERY", "0"))
        if every and rollout_id % every == 0:
            _dump_samples(d, samples)
            rec["dumped"] = True
    else:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run([sys.executable, os.path.join(repo, "rl", "inspect_rollout.py"), d, "rollout.pt"],
                           capture_output=True, text=True)
        open(os.path.join(d, "inspect.txt"), "w").write(r.stdout + r.stderr)
        rec["inspect"] = (r.stdout.strip().splitlines() or ["no output"])[-1]
    rec |= _consistency(out)
    if rec.get("consistency_over"):
        print(f"[flint rollout {rollout_id}] CONSISTENCY WARNING: step {rec['consistency_step']} "
              f"{ {k: rec['consistency'][k] for k in rec['consistency_over']} } over run20's maximum", flush=True)
    with open(os.path.join(out, "steps.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[flint rollout {rollout_id}] {json.dumps(rec)}", flush=True)
    _keep_newest_checkpoints(os.path.join(out, "actor"), 2, int(os.environ.get("MA_RL_KEEP_EVERY", "0") or 0))
    stop = _stop_rule(out, rollout_id)
    if stop and not os.path.exists(os.path.join(out, "stop_request")):
        open(os.path.join(out, "stop_request"), "w").write(f"3 rule: {stop}\n")   # rl/rl_job.sh: checkpoint, then stop
        print(f"[flint rollout {rollout_id}] STOP RULE: {stop}", flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"rollout {rollout_id}: the sample check failed ({rec['inspect']}); see {d}/inspect.txt")
    cap = int(os.environ.get("MA_RL_MAX_SAMPLE_TOKENS", "0"))
    if cap and rec["max_sample_tokens"] > cap:
        raise RuntimeError(f"rollout {rollout_id}: a sample of {rec['max_sample_tokens']} tokens is over {cap}")
    return False
