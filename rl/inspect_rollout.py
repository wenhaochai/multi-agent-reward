"""Checks of one miles rollout of the long game, per episode (one TITO session) against the episode's game trace: its
samples (one per session-tree leaf) and their tokens, the reward, the loss mask (only Flint's completions trained, each
once), the score-centering candidates (miles' own validator) and the routed experts recorded for routing replay (rows
per token, Flint's 49 layers x top-6; the 2 dense layers route nothing).
check(d, samples) runs in the rollout process on the rollout's own samples (flint.miles_agent.log_rollout): no dump is
read back, so a batch of 32 episodes (~23 GB of samples) costs no second copy in host memory. Failed episodes are
counted, not judged here: log_rollout decides how many a rollout may have (MA_RL_MAX_DROPPED_FRAC). The CLI reads a
dump and fails on any failed episode. Usage: inspect_rollout.py <run dir> [dump] (the dump, rollout_0.pt by default,
and traces/); prints a table and writes <run dir>/summary.json."""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

FLINT = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint"
ROLES = ("<|system|>", "<|user|>", "<|assistant|>", "<|observation|>")
_ROLE_IDS: list = []


def _fmt(*a) -> str:
    return " ".join(str(x) for x in a)


def _role_ids() -> set:
    if not _ROLE_IDS:   # once per process: log_rollout calls check() after every rollout
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(FLINT)
        _ROLE_IDS.append({tok.convert_tokens_to_ids(r) for r in ROLES})
    return _ROLE_IDS[0]


def check(d: str, samples: list) -> tuple[dict, dict, list[str]]:
    """(summary, problems, the printed lines) of the rollout of run dir ``d`` (its traces/), from its samples; writes
    d/summary.json. ``problems``: every count that makes the samples wrong to train on (any nonzero one fails the
    rollout); the failed episodes and the judge's unresolved infra submissions are in it as counts too, for the
    caller's threshold."""
    from miles.rollout.generate_utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sample

    role_ids = _role_ids()
    lines = []
    traces = {}
    for f in glob.glob(os.path.join(d, "traces", "agent_*.jsonl")):
        recs = [json.loads(x) for x in open(f)]
        traces[int(os.path.basename(f)[6:-6])] = recs[-1]
    by = defaultdict(list)
    for s in samples:
        by[(s.metadata or {}).get("flint_session")].append(s)

    rows = []
    for sess, ss in sorted(by.items(), key=lambda kv: str(kv[0])):
        meta = ss[0].metadata or {}
        tr = traces.get(meta.get("flint_trace_pid"))
        a = tr["agents"][0] if tr else {}
        turns = a.get("turns", [])
        dropped = [t.get("dropped", 0) for t in turns]
        k = (ss[0].rollout_topk_token_ids.shape[1] if ss[0].rollout_topk_token_ids is not None else 0)
        topk_errors = []
        if k:
            for s in ss:
                try:
                    validate_rollout_topk_logprobs_sample(s, k)
                except ValueError as e:
                    topk_errors.append(str(e))
        trained_roles = 0   # role tokens trained anywhere but at a sample's end (its last turn keeps its stop token)
        for s in ss:
            resp = np.asarray(s.tokens[len(s.tokens) - s.response_length:])
            mask = np.asarray(s.loss_mask, dtype=bool)
            mask[-1:] = False
            trained_roles += int(np.isin(resp[mask], list(role_ids)).sum())
        lp_bad = sum(len(s.rollout_log_probs or []) != s.response_length for s in ss)
        r3 = [s.rollout_routed_experts for s in ss]
        have = [x for x in r3 if x is not None]
        moe = [x[:, 2:, :] for x in have]   # mlp_only_layers = [0, 1]
        r3_info = {
            "r3_samples": len(have), "r3_shapes": [list(x.shape) for x in have][:4], "r3_dtype": str(have[0].dtype) if have else None,
            "r3_rows_minus_tokens": sorted({x.shape[0] - len(s.tokens) for x, s in zip(r3, ss) if x is not None}),
            "r3_mb": round(sum(x.nbytes for x in have) / 2**20, 1),
            "r3_moe_out_of_range": int(sum(((m < 0) | (m >= 256)).sum() for m in moe)),
            "r3_dense_values": sorted({int(v) for x in have for v in np.unique(x[:, :2, :])})[:5]}
        roots = {tuple((s.metadata.get("leaf") or {}).get("path_node_ids", [None]))[:1] for s in ss}
        row = {
            "session": sess, "trace": bool(tr), "failed": meta.get("flint_failed"),
            "judge_infra_submissions": sum(bool(x.get("infra")) for x in tr.get("submissions", [])) if tr else None,
            "S": tr.get("S") if tr else None, "rewards": sorted({s.reward for s in ss}),
            "samples": len(ss), "roots": len(roots), "status": sorted({str(s.status) for s in ss}),
            "model_calls": sum(1 for t in turns if t.get("completion_tokens") is not None),
            "cut_turns": sum(1 for t in turns if t.get("cut") and t.get("text")),
            "trims": sum(1 for x, y in zip(dropped, dropped[1:]) if y > x), "restarts": a.get("restarts"),
            "submissions": len(tr.get("submissions", [])) if tr else None, "format_errors": sum(1 for t in turns if t.get("format_error")),
            "generated": a.get("generated"), "trained_tokens": sum(int(sum(s.loss_mask)) for s in ss),
            "response_tokens": sum(s.response_length for s in ss), "tokens": sum(len(s.tokens) for s in ss),
            "trained_role_tokens": trained_roles, "logprob_len_mismatch": lp_bad, "topk": k,
            "topk_errors": topk_errors[:3], "topk_error_count": len(topk_errors),
            "tito_mismatch": sum(bool(s.metadata.get("tito_session_mismatch")) for s in ss),
            "wall_min": meta.get("fo_lg_wall_min"), **r3_info,
        }
        row["expected_leaves"] = 1 + row["trims"] + (row["restarts"] or 0) + row["cut_turns"]
        rows.append(row)

    gen = sum(r["generated"] or 0 for r in rows)
    ok_rows = [r for r in rows if not r["failed"]]   # a failed episode's reward is a placeholder, its group leaves the loss
    wall = max((r["wall_min"] or 0) for r in rows) if rows else 0
    summary = {
        "run": d, "episodes": len(rows), "samples": len(samples), "generated_tokens": gen, "trained_tokens":
        sum(r["trained_tokens"] for r in rows), "max_episode_wall_min": wall,
        "generated_tok_per_s": round(gen / (60 * wall), 1) if wall else None,
        "all_rewards_equal_S": all(r["trace"] and r["rewards"] == [r["S"]] for r in ok_rows),
        "ok_generated_tokens": sum(r["generated"] or 0 for r in ok_rows),
        "ok_trained_tokens": sum(r["trained_tokens"] for r in ok_rows),
        "trained_role_tokens": sum(r["trained_role_tokens"] for r in rows),
        "topk_error_count": sum(r["topk_error_count"] for r in rows),
        "r3_mb_total": round(sum(r["r3_mb"] for r in rows), 1), "r3_samples": sum(r["r3_samples"] for r in rows),
        "r3_shapes_first": rows[0]["r3_shapes"] if rows else None, "r3_dtype": rows[0]["r3_dtype"] if rows else None,
        "r3_rows_minus_tokens": sorted({v for r in rows for v in r["r3_rows_minus_tokens"]}),
        "r3_moe_out_of_range": sum(r["r3_moe_out_of_range"] for r in rows),
        "r3_dense_values": sorted({v for r in rows for v in r["r3_dense_values"]}), "rows": rows}
    json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=1, default=str)
    lines.append(_fmt(f"{'session':10} {'S':>6} {'samp':>4} {'exp':>4} {'roots':>5} {'calls':>5} {'cut':>3} {'trim':>4} {'rst':>3} "
          f"{'subs':>4} {'fmt':>3} {'generated':>9} {'trained':>8} {'resp':>8} {'roleTok':>7} {'topkErr':>7} {'wall':>5}"))
    for r in rows:
        lines.append(_fmt(f"{str(r['session'])[:10]:10} {r['S'] if r['S'] is None else round(100 * r['S'], 2)!s:>6} {r['samples']:>4} "
              f"{r['expected_leaves']:>4} {r['roots']:>5} {r['model_calls']:>5} {r['cut_turns']:>3} {r['trims']:>4} "
              f"{r['restarts']!s:>3} {r['submissions']!s:>4} {r['format_errors']:>3} {r['generated']!s:>9} "
              f"{r['trained_tokens']:>8} {r['response_tokens']:>8} {r['trained_role_tokens']:>7} {r['topk_error_count']:>7} "
              f"{r['wall_min'] if r['wall_min'] is None else round(r['wall_min'], 1)!s:>5}"
              + (f" FAILED {r['failed'][:80]}" if r["failed"] else "")))
    lines.append(_fmt(json.dumps({k: v for k, v in summary.items() if k != "rows"})))
    for r in rows:
        if r["topk_errors"]:
            lines.append(_fmt("topk errors", r["session"], r["topk_errors"]))
    problems = {"failed episodes": sum(bool(r["failed"]) for r in rows),
                "judge infra submissions": sum(r["judge_infra_submissions"] or 0 for r in rows),
                "rewards not S": 0 if summary["all_rewards_equal_S"] else 1,
                "trained != generated": int(summary["ok_trained_tokens"] != summary["ok_generated_tokens"]),
                "top-k errors": summary["topk_error_count"], "R3 ids out of range": summary["r3_moe_out_of_range"]}
    return summary, problems, lines


if __name__ == "__main__":
    import torch

    from miles.utils.types import Sample

    d = sys.argv[1]
    dump = torch.load(os.path.join(d, sys.argv[2] if len(sys.argv) > 2 else "rollout_0.pt"), weights_only=False)
    summary, problems, lines = check(d, [Sample.from_dict(s) for s in dump["samples"]])
    print("\n".join(lines))
    bad = {k: v for k, v in problems.items() if v}
    print("INSPECT_FAIL", bad) if bad else print("INSPECT_OK")
    sys.exit(1 if bad else 0)
