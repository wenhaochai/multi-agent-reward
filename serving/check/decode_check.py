"""Decode-time logprobs against a prefill of the same tokens, on one Flint server: what an RL rollout records against
what a prefill scores. From each S1 sequence's first 4,000 and (when long enough) 16,000 tokens, 512 tokens are
sampled (temperature 1, one request at a time), then the prompt plus those tokens is prefilled; per sampled token the
decode and the prefill logprob of that token are saved. Usage: decode_check.py seqs.json url out.pt"""
import json
import sys

import requests
import torch

seqs = json.load(open(sys.argv[1]))
URL, OUT = sys.argv[2], sys.argv[3]
SP = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0}


def post(body):
    return requests.post(f"{URL}/generate", json=body, timeout=1800).json()["meta_info"]


res = []
for s in seqs:
    for n in (4000, 16000):
        if len(s["ids"]) < n:
            continue
        prefix = s["ids"][:n]
        m = post({"input_ids": prefix, "sampling_params": dict(SP, max_new_tokens=512, ignore_eos=True),
                  "return_logprob": True, "logprob_start_len": n - 1})
        ids = [x[1] for x in m["output_token_logprobs"]]
        dec = torch.tensor([x[0] for x in m["output_token_logprobs"]], dtype=torch.float64)
        p = post({"input_ids": prefix + ids, "sampling_params": dict(SP, max_new_tokens=1), "return_logprob": True,
                  "logprob_start_len": n - 1})
        pre = torch.tensor([x[0] for x in p["input_token_logprobs"][1:]], dtype=torch.float64)
        assert len(pre) == len(dec), (len(pre), len(dec))
        res.append({"prefix": n, "decode": dec, "prefill": pre})
        print("prefix", n, "decode vs prefill mean |d|", round(float((dec - pre).abs().mean()), 4), flush=True)
torch.save(res, OUT)
print("DECODE_DONE", OUT, flush=True)
