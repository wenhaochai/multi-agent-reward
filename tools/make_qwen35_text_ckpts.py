"""Drop the vision tower of a Qwen3.5 (Qwen3_5ForConditionalGeneration) checkpoint for text-only RL:
  <src>-text: Qwen3_5ForCausalLM (model_type qwen3_5_text), weights model.language_model.* -> model.*, lm_head and
              mtp.* kept, model.visual.* dropped. For Megatron through the bridge (Qwen35Bridge -> GPTModel).
  <src>-lm:   the original architecture and weight names with model.visual.* dropped and language_model_only=true in
              config.json, so SGLang never builds or loads the vision tower. For SGLang (it registers only the
              ConditionalGeneration classes; the bridge's model.layers.* names load into it unchanged).
Single-file checkpoints (model.safetensors, no index) are handled; tensors are cast to the config's dtype
(bfloat16: the EasyPPO SFT release stores fp32) and a text_config without vocab_size gets it from lm_head.
usage: python make_qwen35_text_ckpts.py SRC_DIR"""
import json, shutil, sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    src = Path(sys.argv[1])
    cfg = json.load(open(src / 'config.json'))
    assert cfg['architectures'] == ['Qwen3_5ForConditionalGeneration'], cfg['architectures']
    text_dir, lm_dir = Path(f'{src}-text'), Path(f'{src}-lm')
    for d in (text_dir, lm_dir):
        d.mkdir(exist_ok=True)
        for f in src.iterdir():
            if f.suffix not in ('.safetensors',) and f.name not in ('config.json', 'model.safetensors.index.json') and f.is_file():
                shutil.copy2(f, d / f.name)
    idx_path = src / 'model.safetensors.index.json'
    idx = json.load(open(idx_path)) if idx_path.exists() else None
    shards = sorted(set(idx['weight_map'].values())) if idx else ['model.safetensors']
    dtype = getattr(torch, cfg.get('dtype') or cfg.get('torch_dtype') or cfg['text_config'].get('dtype') or 'bfloat16')
    tcfg = dict(cfg['text_config'])
    if 'vocab_size' not in tcfg:
        with safe_open(src / shards[-1] if idx is None else src / idx['weight_map']['lm_head.weight'], 'pt') as f:
            tcfg['vocab_size'] = f.get_slice('lm_head.weight').get_shape()[0]
        cfg['text_config'] = dict(cfg['text_config'], vocab_size=tcfg['vocab_size'])
    tcfg.update(architectures=['Qwen3_5ForCausalLM'], model_type='qwen3_5_text',
                tie_word_embeddings=cfg.get('tie_word_embeddings', False))
    tcfg.setdefault('torch_dtype', cfg.get('torch_dtype', 'bfloat16'))
    json.dump(tcfg, open(text_dir / 'config.json', 'w'), indent=2)
    lcfg = dict(cfg, language_model_only=True)  # (cfg carries the filled-in vocab_size)
    json.dump(lcfg, open(lm_dir / 'config.json', 'w'), indent=2)
    t_map, l_map, n_vis = {}, {}, 0
    for shard in shards:
        t_sd, l_sd = {}, {}
        with safe_open(src / shard, 'pt') as f:
            for k in f.keys():
                if k.startswith('model.visual.'):
                    n_vis += 1
                    continue
                x = f.get_tensor(k).to(dtype)
                l_sd[k] = x
                tk = k.replace('model.language_model.', 'model.', 1)
                t_sd[tk] = x
        save_file(t_sd, text_dir / shard, metadata={'format': 'pt'})
        save_file(l_sd, lm_dir / shard, metadata={'format': 'pt'})
        t_map.update({k: shard for k in t_sd})
        l_map.update({k: shard for k in l_sd})
        print(shard, len(t_sd), flush=True)
    for d, m in ((text_dir, t_map), (lm_dir, l_map)):
        json.dump({'metadata': {}, 'weight_map': m}, open(d / 'model.safetensors.index.json', 'w'), indent=2)
    print(f'dropped {n_vis} vision tensors; text keys e.g. {sorted(t_map)[:3]}; lm_head in text: {"lm_head.weight" in t_map}; '
          f'mtp keys: {sum(k.startswith("mtp.") for k in t_map)}')


if __name__ == '__main__':
    main()
