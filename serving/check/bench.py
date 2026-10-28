"""Decode throughput of one Flint server: 1024 tokens generated after a 2k-token prompt (the first 2000 tokens of each
parity sequence), at 1 request and at 8 concurrent requests, sampled as the eval samples (temperature 1, top-p 0.95,
top-k 20) with ignore_eos. Each setting runs twice and reports the second (warm) run.
Usage: bench.py seqs.json url out.json"""
import concurrent.futures as cf
import json
import sys
import time

import requests

seqs = json.load(open(sys.argv[1]))
URL, OUT = sys.argv[2], sys.argv[3]
SP = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_new_tokens": 1024, "ignore_eos": True}


def generate(ids):
    m = requests.post(f"{URL}/generate", json={"input_ids": ids, "sampling_params": SP}, timeout=3600).json()["meta_info"]
    return m["completion_tokens"]


res = {}
for conc in (1, 8):
    for _ in range(2):
        prompts = [s["ids"][:2000] for s in seqs[:conc]]
        t0 = time.time()
        with cf.ThreadPoolExecutor(conc) as ex:
            toks = list(ex.map(generate, prompts))
        wall = time.time() - t0
    res[f"c{conc}"] = {"tokens": sum(toks), "seconds": round(wall, 2), "tok_per_s": round(sum(toks) / wall, 1)}
    print(conc, res[f"c{conc}"], flush=True)
json.dump(res, open(OUT, "w"))
