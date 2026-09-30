from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/Qwen3.8-27B')
kw = {"enable_thinking": False}
m1 = [{"role": "user", "content": "Problem P. Lead suffix."}]
g1 = tok.apply_chat_template(m1, tokenize=False, add_generation_prompt=True, **kw)
print("GEN1:", repr(g1[-80:]))
text = "Let me delegate.\n<delegate to=1>solve it</delegate><|im_end|>"
m2 = m1 + [{"role": "assistant", "content": text}, {"role": "user", "content": "Reports: ..."}]
full = tok.apply_chat_template(m2, tokenize=False, add_generation_prompt=True, **kw)
prefix = tok.apply_chat_template(m2[:-1], tokenize=False, add_generation_prompt=False, **kw)
print("EXTENDS:", full.startswith(prefix))
print("PREFIX tail:", repr(prefix[-120:]))
print("FULL  tail:", repr(full[-200:]))
if full.startswith(prefix): print("DELTA:", repr(full[len(prefix):]))
print("eos", tok.eos_token, tok.eos_token_id, "pad", tok.pad_token)
