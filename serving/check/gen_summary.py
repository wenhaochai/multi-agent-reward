"""Generation sanity of real long-game episodes against one server (run_eval.py traces): per episode the exit, steps,
turns the game flagged as format errors, split into turns cut at the token limit (finish_reason length: the budget or
the turn limit ran out) and turns whose tool call or command did not parse, then submissions and S; then both rates
over all turns. Usage: gen_summary.py <run_eval out dir> [...]"""
import glob
import json
import sys

for out in sys.argv[1:]:
    turns = errors = cuts = 0
    print("==", out)
    for f in sorted(glob.glob(f"{out}/traces/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            for a in r["agents"]:
                n = sum(1 for t in a["turns"] if "completion_tokens" in t)
                flagged = [t for t in a["turns"] if t.get("format_error")]
                c = sum(1 for t in flagged if t.get("finish_reason") == "length")
                e = len(flagged) - c
                turns, errors, cuts = turns + n, errors + e, cuts + c
                print(r["label"].split("/")[-1], a["exit"], "steps", a["steps"], "turns", n, "unparsed", e, "cut", c,
                      "submits", a["submits"], "restarts", a["restarts"], "generated", a["generated"], "S", r["S"])
    print(f"UNPARSED {errors} of {turns} turns ({100 * errors / max(turns, 1):.1f}%), CUT {cuts}")
