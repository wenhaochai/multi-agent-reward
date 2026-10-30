"""Fake miles train.py for tests/dryrun_ext.sh: records its argv, prints the actor's timer lines, writes an agent's
game settings (the startup probe's trigger), sleeps FAKE_TRAIN_S, writes a checkpoint tracker, exits FAKE_TRAIN_RC."""
import json
import os
import sys
import time

out = os.environ["OUT"]
json.dump(sys.argv[1:], open(os.path.join(out, "fake_train_args.json"), "w"))
pre = "(FSDPTrainRayActor pid=1) [2026-10-30 00:00:00.000 actor_cell00000_rank00000] timer.py:31 - "
print(pre + "Timer train_wait start", flush=True)
os.makedirs(os.path.join(out, "settings_seen"), exist_ok=True)
json.dump(json.load(open(os.path.join(out, "game_settings.json"))), open(os.path.join(out, "settings_seen", "1.json"), "w"))
for name in ("train", "log_probs", "actor_train"):
    print(pre + f"Timer {name} start", flush=True)
time.sleep(float(os.environ.get("FAKE_TRAIN_S", "3")))
print(pre + "Timer train_wait start", flush=True)
a = sys.argv[1:]
save = a[a.index("--save") + 1]
os.makedirs(save, exist_ok=True)
open(os.path.join(save, "latest_checkpointed_iteration.txt"), "w").write("1")
sys.exit(int(os.environ.get("FAKE_TRAIN_RC", "0")))
