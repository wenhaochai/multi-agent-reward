"""Generation sanity of real long-game episodes against one server (run_eval.py traces): per episode the exit, steps,
format-error turns (a turn whose tool call or command did not parse), submissions and S; then the format-error rate
over all turns. Usage: gen_summary.py <run_eval out dir> [...]"""
import glob
import json
import sys

for out in sys.argv[1:]:
    turns = errors = 0
    print("==", out)
    for f in sorted(glob.glob(f"{out}/traces/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            for a in r["agents"]:
                n = sum(1 for t in a["turns"] if "completion_tokens" in t)
                e = sum(1 for t in a["turns"] if t.get("format_error"))
                turns, errors = turns + n, errors + e
                print(r["label"].split("/")[-1], a["exit"], "steps", a["steps"], "turns", n, "format errors", e,
                      "submits", a["submits"], "restarts", a["restarts"], "generated", a["generated"], "S", r["S"])
    print(f"FORMAT_ERRORS {errors} of {turns} turns ({100 * errors / max(turns, 1):.1f}%)")
