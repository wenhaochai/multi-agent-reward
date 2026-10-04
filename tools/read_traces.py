"""Read rollout traces, not only metrics: per run, what every agent turn ended as (cut off, never closed its
reasoning, no code, compile error, judged), early vs late in training, plus raw text samples to read.
  python3 tools/read_traces.py <run dir or name> [--show N] [--kind cut|compile|nocode|done]
Handles fcs_team traces (lead_text / mate_texts / lead_cut since 2026-10-04; older ones keep only 3000-char windows)
and fcs_oracle traces ("turns": role, cut, closed_think, status, score, text).
"""
import argparse, collections, glob, json, re, sys
from pathlib import Path

B = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--show", type=int, default=0)
ap.add_argument("--kind", default="")
a = ap.parse_args()
d = Path(a.run) if Path(a.run).exists() else B / "runs" / a.run
rows = sorted((json.loads(l) for f in glob.glob(str(d / "traces" / "*.jsonl")) for l in open(f)), key=lambda r: r["time"])


def turns(r):
    """(role, text, cut, status) per agent turn; text None when the trace predates full texts."""
    if "reward" in r and "text" in r:  # solo EasyPPO (fcs_rm)
        return [("solo", r["text"], r["truncated"], r["status"])]
    if "turns" in r:
        return [(t["role"], t["text"], t["cut"], t["status"]) for t in r["turns"]]
    if "lead_text" in r:
        return ([("lead", r["lead_text"], r["lead_cut"], r["lead_status"])]
                + [(f"mate{j + 1}", t, c, s) for j, (t, c, s) in enumerate(zip(r["mate_texts"], r["mate_cut"], r["mate_status"]))])
    return [("lead", None, None, r["lead_status"])] + [(f"mate{j + 1}", None, None, s) for j, s in enumerate(r["mate_status"])]


def kind(text, cut, status):
    if status is None:
        return "plan"
    if text is None:
        return f"{status} (old trace)"
    if cut:
        return "cut off"
    if "</think>" not in text:
        return "stopped inside reasoning"
    vis = text.rpartition("</think>")[2]
    if "```" not in vis and "#include" not in vis:
        return "no code after reasoning"
    return {"compile error": "compile error", "done": "judged"}.get(status, status)


tr = [r for r in rows if not r.get("eval")]
print(f"{d.name}: {len(tr)} train traces, {len(rows) - len(tr)} eval traces")
for name, part in (("first half", tr[: len(tr) // 2]), ("second half", tr[len(tr) // 2:]), ("eval", [r for r in rows if r.get("eval")])):
    if not part:
        continue
    by = collections.defaultdict(collections.Counter)
    for r in part:
        for role, text, cut, st in turns(r):
            by[re.sub(r"\d+$", "", role)][kind(text, cut, st)] += 1
    print(f"  {name} ({len(part)} episodes)")
    for role, c in by.items():
        n = sum(c.values())
        print(f"    {role:<6} " + ", ".join(f"{k} {v / n:.0%}" for k, v in c.most_common()))
if a.show:
    shown = 0
    for r in reversed(tr):
        for role, text, cut, st in turns(r):
            if text is not None and (not a.kind or a.kind in kind(text, cut, st)) and shown < a.show:
                shown += 1
                print(f"\n==== {r['label'].split('/')[-1]} {role} [{kind(text, cut, st)}] {len(text)} chars")
                print(text[:600] + "\n  [...]\n" + text[-900:])
