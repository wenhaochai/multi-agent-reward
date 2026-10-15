"""ev4 fill (owner 2026-10-14): finish an ev4 arm on the classic val problems only. Interactive problems stay skipped
(owner 2026-10-08), and the classic episodes the timed-out ev4 job traced are kept. Writes
  data/fcs_ev4fill_<arm>.jsonl: each classic problem of data/fcs_val98_<game>.jsonl repeated 4 - (its traced episodes)
  times, for --n-samples-per-eval-prompt 1;
  <emit>/<arm>_fill/{smoke,main}: copies of the arm's EMIT dirs with run names ev4 -> ev4fill (own run dir, traces, W&B
  id), the fill file as eval data, 1 sample per prompt, and --rollout-seed 43, so a refilled problem draws new episodes
  instead of repeating the seed-42 ones.
usage: python3 tools/make_ev4_fill.py ARM GAME   (ARM e.g. team_bonus, GAME fo_team | fo_seq | fo_par)"""
import collections
import glob
import json
import sys
from pathlib import Path

B = Path("/scratch/gpfs/GROUP/USER/project/miles-q38-build")
EMIT = Path("/scratch/gpfs/GROUP/USER/tmp/evalonly")
N = 4

arm, game = sys.argv[1], sys.argv[2]
done = collections.Counter()
for f in glob.glob(str(B / f"runs/fcs_evalonly_ev4_{arm}_q9b-fcs-sft_s42/traces/*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r.get("eval"):
            done[r["label"]] += 1
rows = (B / f"data/fcs_val98_{game}.jsonl").read_text().splitlines()
fill = [l for l in rows for _ in range(max(0, N - done[json.loads(l)["label"]]))]
classic = {json.loads(l)["label"] for l in rows}
out = B / f"data/fcs_ev4fill_{arm}.jsonl"
out.write_text("\n".join(fill) + "\n")
print(f"{arm}: traced classic episodes {sum(done[k] for k in classic)}, fill {len(fill)} -> {out}")

src, dst = EMIT / arm, EMIT / f"{arm}_fill"
if dst.exists():
    sys.exit(f"{dst} exists: remove it by hand first")
for part in ("smoke", "main"):
    (dst / part).mkdir(parents=True)
    for name in ("args", "env", "run"):
        text = (src / part / name).read_text().replace("_ev4_", "_ev4fill_")
        if name == "args":
            a = text.split("\n")
            a[a.index("--rollout-seed") + 1] = "43"
            if part == "main":
                i = a.index("--eval-prompt-data")
                a[i + 1], a[i + 2] = f"{game}_classic", str(out)
                a[a.index("--n-samples-per-eval-prompt") + 1] = "1"
            text = "\n".join(a)
        (dst / part / name).write_text(text)
print(f"EMIT {dst}: run {(dst / 'main' / 'run').read_text().strip()}")
