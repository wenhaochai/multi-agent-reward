"""The sampler's view of the S2 batch: for each sequence of build_p0.py's output (the indices given), SGLang's prefill
log-probability of every next token and its top-32 head at temperature 1 (what a rollout with score centering
records). Usage: rescore.py p0_seqs.json url out_dir i [i ...]; writes out_dir/seq<i>.pt {lp, top_v, top_i}."""
import json
import sys

import requests
import torch

seqs = json.load(open(sys.argv[1]))
URL, OUT = sys.argv[2], sys.argv[3]
for i in map(int, sys.argv[4:]):
    ids = seqs[i]["ids"]
    body = {"input_ids": ids, "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                                                  "max_new_tokens": 1},
            "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": 32}
    m = requests.post(f"{URL}/generate", json=body, timeout=7200).json()["meta_info"]
    assert len(m["input_token_logprobs"]) == len(ids), (len(m["input_token_logprobs"]), len(ids))
    heads = m["input_top_logprobs"][1:]          # entry j + 1: the head of the prediction made at position j
    torch.save({"lp": torch.tensor([x[0] for x in m["input_token_logprobs"][1:]], dtype=torch.float32),
                "top_v": torch.tensor([[x[0] for x in r] for r in heads], dtype=torch.float32),
                "top_i": torch.tensor([[x[1] for x in r] for r in heads], dtype=torch.int32)}, f"{OUT}/seq{i}.pt")
    print("seq", i, len(ids), flush=True)
