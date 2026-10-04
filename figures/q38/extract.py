"""Write the data behind the Q38 curves (logs/peer_team_lead.yaml) as CSVs next to this file; stdlib only.

eval_points.csv  one row per finished offline eval-only job (runs fr_peer_team_q38_<arm>_sc_ev<step>, logs
                 logs/molt-fr-fr_peer_team_q38_<arm>_sc_ev<step>-<jid>.out, the "[eval-only] {...}" line): arm, step,
                 pooled pass@1 (mean over the 5 competitions of each competition's pass1), the five pass1 values and
                 the five delegation rates. Re-runs (ev<step>r) count only when the first eval of that step has no
                 result. Step 0 is the untrained base model under that arm's protocol (lead's protocol is nob's).
training.csv     one row per training step: arm, step, per-episode training accuracy (tm_task_w / tm_inv_sessions)
                 and delegation rate (tm_delegated_w / tm_inv_sessions), from the "Global step N: {...}" lines of
                 every training segment's log; a step logged by two segments (a resume re-runs the steps after its
                 checkpoint) takes the later segment.
"""
import ast
import csv
import glob
import os
import re

W = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace"
HERE = os.path.dirname(os.path.abspath(__file__))
COMPS = ["aime_2024", "aime_2025", "aime_2026", "hmmt_feb_2025", "hmmt_feb_2026"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def eval_points():
    rows = {}
    logs = sorted(glob.glob(f"{W}/logs/molt-fr-fr_peer_team_q38_*_sc_ev*-*.out"),
                  key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]))
    for p in logs:
        m = re.search(r"q38_(lead|nob|solo)_sc_ev(\d+)(r?)-(\d+)\.out$", p)
        if not m:
            continue
        arm, step, rerun = m.group(1), int(m.group(2)), m.group(3)
        line = next((ANSI.sub("", l) for l in open(p, errors="ignore") if l.startswith("[eval-only] {")), None)
        if line is None:
            continue
        if rerun and (arm, step) in rows:
            continue
        d = ast.literal_eval(line[line.index("{"):].strip())
        p1 = [d[f"eval_{c}_pass1"] for c in COMPS]
        dl = [d.get(f"eval_{c}_tm_delegated", 0.0) for c in COMPS]
        rows[(arm, step)] = [arm, step, sum(p1) / len(p1)] + p1 + dl + [os.path.basename(p)]
    with open(os.path.join(HERE, "eval_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "step", "pooled"] + [f"pass1_{c}" for c in COMPS] + [f"deleg_{c}" for c in COMPS] + ["log"])
        for k in sorted(rows):
            w.writerow(rows[k])
    return rows


def training():
    out = {}
    for arm in ("lead", "nob", "solo"):
        logs = sorted(glob.glob(f"{W}/logs/molt-fr-fr_peer_team_q38_{arm}_sc-*.out"),
                      key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]))   # by job id: later segments win
        for p in logs:
            for l in open(p, errors="ignore"):
                if "Global step" not in l:
                    continue
                m = re.search(r"Global step (\d+): (\{.*\})", ANSI.sub("", l))
                if not m:
                    continue
                d = ast.literal_eval(m.group(2))
                inv = d.get("tm_inv_sessions") or 1.0
                out[(arm, int(m.group(1)))] = (d.get("tm_task_w", 0.0) / inv, d.get("tm_delegated_w", 0.0) / inv)
    with open(os.path.join(HERE, "training.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "step", "correct", "delegating"])
        for (arm, step), (c, dl) in sorted(out.items()):
            w.writerow([arm, step, f"{c:.4f}", f"{dl:.4f}"])
    return out


if __name__ == "__main__":
    ev = eval_points()
    tr = training()
    for k in sorted(ev):
        print("eval", k, f"{ev[k][2]:.3f}")
    for arm in ("lead", "nob", "solo"):
        steps = sorted(s for a, s in tr if a == arm)
        print("train", arm, f"steps {steps[0]}-{steps[-1]} ({len(steps)})" if steps else "none")
