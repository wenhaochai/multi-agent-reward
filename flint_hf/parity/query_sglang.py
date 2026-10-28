"""SGLang side of the parity check. For every sequence: the log-probability of each next token and the top-32 head,
from a prefill with temperature 1 and no top-p/top-k. With --floor, also every sequence again with all of them sent
at once (a changed batch: the noise floor) and 1024 tokens sampled after three prefixes, with their decode-time
logprobs and heads (what an RL rollout records) and a prefill of the same tokens.
Usage: query_sglang.py seqs.json url out.pt [--floor]"""
import concurrent.futures as cf
import json
import sys

import requests
import torch

seqs = json.load(open(sys.argv[1]))
URL, OUT, FLOOR = sys.argv[2], sys.argv[3], "--floor" in sys.argv
K = 32
SP = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0}


def rows(entries):
    return (torch.tensor([[x[0] for x in r] for r in entries], dtype=torch.float64),
            torch.tensor([[x[1] for x in r] for r in entries], dtype=torch.long))


def prefill(ids):
    body = {"input_ids": ids, "sampling_params": dict(SP, max_new_tokens=1), "return_logprob": True,
            "logprob_start_len": 0, "top_logprobs_num": K}
    m = requests.post(f"{URL}/generate", json=body, timeout=3600).json()["meta_info"]
    assert len(m["input_token_logprobs"]) == len(ids), (len(m["input_token_logprobs"]), len(ids))
    tv, ti = rows(m["input_top_logprobs"][1:])   # entry i + 1 holds the head of the prediction made at position i
    lp = torch.tensor([x[0] for x in m["input_token_logprobs"][1:]], dtype=torch.float64)
    return {"lp": lp, "top_v": tv, "top_i": ti}


def sample(ids, n=1024):
    body = {"input_ids": ids, "sampling_params": dict(SP, max_new_tokens=n, ignore_eos=True), "return_logprob": True,
            "logprob_start_len": len(ids) - 1, "top_logprobs_num": K}
    m = requests.post(f"{URL}/generate", json=body, timeout=3600).json()["meta_info"]
    tv, ti = rows(m["output_top_logprobs"])
    return {"ids": [x[1] for x in m["output_token_logprobs"]], "top_v": tv, "top_i": ti,
            "lp": torch.tensor([x[0] for x in m["output_token_logprobs"]], dtype=torch.float64)}


res = {"solo": []}
for s in seqs:
    res["solo"].append(prefill(s["ids"]))
    print("prefill", len(s["ids"]), flush=True)
if FLOOR:
    res["batch"] = [None] * len(seqs)
    with cf.ThreadPoolExecutor(len(seqs)) as ex:
        futs = {ex.submit(prefill, s["ids"]): j for j, s in enumerate(seqs)}
        for f in cf.as_completed(futs):
            res["batch"][futs[f]] = f.result()
    print("batched prefill done", flush=True)
    res["samples"] = []
    for j in (1, 3, 6):   # after the first 4k, 16k and 48k tokens
        prefix = seqs[j]["ids"]
        smp = sample(prefix)
        smp["prefix"] = prefix
        smp["prefill"] = prefill(prefix + smp["ids"])
        res["samples"].append(smp)
        print("sampled after", len(prefix), flush=True)
torch.save(res, OUT)
print("SGLANG_DONE", OUT, flush=True)
