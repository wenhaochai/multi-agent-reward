"""A fixed GRPO batch for the S2 training-step test: one Flint's long-game episodes on Pack the Polyominoes (val40
problem 0), rendered with Flint's chat template as S1's parity sequences are (the problem as the user message, each
turn's text as an assistant message and its command output as the next user message), and cut at the last turn that
ends within MAX_TOKENS: the longest context one segment reaches. Generated tokens (the assistant content and its end
token, not the role header the template adds) carry the loss. Usage: build_p0.py out.json
Episodes: the 8 single-Flint episodes of problem 0 with stored turns; reward = the episode's S."""
import glob
import json
import sys

from transformers import AutoTokenizer

MODEL = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint"
RUNS = "/scratch/gpfs/GROUP/USER/project/flint-eval/runs"
DATA = "/scratch/gpfs/GROUP/USER/project/mas-rl-flintrl/data_v11/fcs11_val40_agent.jsonl"
EPISODES = ["smoke_v2_w1531", "smoke_v2_w2339", "smoke_v2_w2349", "smoke_v2_w1532", "smoke_v2_w1533",
            "smoke_v2_w1534", "smoke_v2_w251", "smoke_v2_w259"]
MAX_TOKENS = 65536

tok = AutoTokenizer.from_pretrained(MODEL)
problem = next(json.loads(x)["metadata"]["messages"] for x in open(DATA) if json.loads(x)["metadata"]["pid"] == "0")


def render(msgs, gen=False):
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen, return_dict=False)


out = []
for run in EPISODES:
    recs = [json.loads(x) for f in glob.glob(f"{RUNS}/{run}/traces/*.jsonl") for x in open(f)]
    rec = next(r for r in recs if r["label"].split("/")[-1] == "0" and len(r["agents"]) == 1)
    msgs = [dict(m) for m in problem]
    prompt = render(msgs, gen=True)
    ids, spans = list(prompt), []
    for t in rec["agents"][0]["turns"]:
        if t.get("text") is None:
            continue
        nxt = msgs + [{"role": "assistant", "content": t["text"].strip() or "ls"}]
        head = render(msgs, gen=True)
        full = render(nxt)
        user = nxt + [{"role": "user", "content": (t.get("output") or "")[:20000] or "(no output)"}]
        after = render(user)
        if len(after) > MAX_TOKENS:
            break
        assert full[:len(head)] == head and after[:len(full)] == full, "the template must only append"
        spans.append((len(head), len(full)))
        msgs, ids = user, after
    out.append({"run": run, "S": rec["S"], "ids": ids, "prompt_len": len(prompt), "spans": spans})
    n_gen = sum(b - a for a, b in spans)
    print(run, "S", round(rec["S"], 3), "tokens", len(ids), "prompt", len(prompt), "turns", len(spans),
          "generated", n_gen, flush=True)
json.dump(out, open(sys.argv[1], "w"))
