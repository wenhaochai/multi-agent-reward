"""Token sequences for the HF-vs-SGLang parity check: real Flint long-game conversations (train67, one Flint, 500k:
runs/train67_t500000_e1) rendered with Flint's chat template: the problem as the user message, then each turn's text
as the assistant message and its command output as the next user message, cut at the target lengths. Usage:
build_seqs.py <out.json>"""
import json
import sys

from transformers import AutoTokenizer

MODEL = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint"
TRACES = "/scratch/gpfs/GROUP/USER/project/flint-eval/runs/train67_t500000_e1/traces/agent_1072060.jsonl"
DATA = "/scratch/gpfs/GROUP/USER/project/mas-rl-flintrl/data_v11/fcs11_train67_agent.jsonl"
TARGETS = [2000, 4000, 8000, 16000, 24000, 32000, 48000, 60000]

tok = AutoTokenizer.from_pretrained(MODEL)
probs = {}
for line in open(DATA):
    r = json.loads(line)
    probs[r["label"]] = r["metadata"]["messages"]
eps = [json.loads(x) for x in open(TRACES)]
eps.sort(key=lambda r: -sum(len(t.get("text") or "") + len(t.get("output") or "") for t in r["agents"][0]["turns"]))
out = []
for i, target in enumerate(TARGETS):
    r = eps[i % len(eps)]
    msgs = [dict(m) for m in probs[r["label"]]]
    for t in r["agents"][0]["turns"]:
        msgs.append({"role": "assistant", "content": (t.get("text") or "").strip() or "ls"})
        msgs.append({"role": "user", "content": (t.get("output") or "")[:20000] or "(no output)"})
    text = tok.apply_chat_template(msgs, tokenize=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < target:
        raise SystemExit(f"episode {r['label']} has only {len(ids)} tokens < {target}")
    ids = ids[:target]
    out.append({"label": r["label"], "target": target, "ids": ids})
    print(target, len(ids), r["label"].split("/")[-1], flush=True)
json.dump(out, open(sys.argv[1], "w"))
