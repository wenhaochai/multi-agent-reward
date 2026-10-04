"""Difficulty probe of a model on the Frontier-CS study data: sample with SGLang (EasyPPO's eval settings: T 1.0,
top-p 1.0, 32768 new tokens, default chat template; THINKING=0 turns thinking off), score each response with miles_team.fcs_judge, write every record
incrementally (gen.jsonl, scores.jsonl; a rerun skips what is done) and a summary.
usage: python fcs_probe.py MODEL OUTDIR SPEC... where SPEC = name:path.jsonl:n_samples
Prompt variants: PROMPT_FILE=<json {"system": str, "user_suffix": str}> adds a system message and/or appends text to
the user message; TEMP sets the temperature (default 1.0)."""
import json, os, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miles_team.fcs_judge import CASE_WORKERS, job_cpus, judge
from miles_team.fcs_rm import extract_cpp

MAX_NEW, CHUNK = int(os.environ.get('MAX_NEW', 32768)), int(os.environ.get('CHUNK', 256))


_PV = json.load(open(os.environ['PROMPT_FILE'])) if os.environ.get('PROMPT_FILE') else {}


def variant(messages):
    """The prompt variant of PROMPT_FILE applied to a chat (system message first, suffix on the last user turn)."""
    msgs = [dict(m) for m in messages]
    if _PV.get('user_suffix'):
        msgs[-1]['content'] = msgs[-1]['content'] + _PV['user_suffix']
    if _PV.get('system'):
        msgs = [{'role': 'system', 'content': _PV['system']}] + [m for m in msgs if m['role'] != 'system']
    return msgs


def load(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def score_one(rec):
    code = extract_cpp(rec['response'])
    if not code:
        return {**{k: rec[k] for k in ('key', 'split', 'pid', 'i')}, 'score': 0.0, 'status': 'no code'}
    try:
        r = judge(rec['label'], code)
        out = {'score': r['score'], 'status': r['status'], 'infra': bool(r.get('infra'))}
    except Exception as e:
        out = {'score': 0.0, 'status': f'judge error: {e}'[:200]}
    return {**{k: rec[k] for k in ('key', 'split', 'pid', 'i')}, **out}


def main():
    model, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for spec in sys.argv[3:]:
        name, path, n = spec.split(':')
        for row in load(path):
            for i in range(int(n)):
                todo.append({'key': f"{name}/{row['metadata']['pid']}/{i}", 'split': name, 'pid': row['metadata']['pid'],
                             'i': i, 'label': row['label'], 'prompt': row['prompt']})
    gen_f, sc_f = out / 'gen.jsonl', out / 'scores.jsonl'
    done_gen = {r['key']: r for r in load(gen_f)}
    done_sc = {r['key'] for r in load(sc_f)}
    # judges at once: the job's own cores (os.cpu_count() is the whole 96-core node, the job has 64), 8 kept for the
    # SGLang schedulers; oversubscription trips the judge's wall limit (2 x the time limit)
    pool = ThreadPoolExecutor(max(1, (job_cpus() - 8) // CASE_WORKERS))
    futs = [pool.submit(score_one, r) for k, r in done_gen.items() if k not in done_sc]
    rest = [t for t in todo if t['key'] not in done_gen]
    print(f'{len(todo)} samples; {len(done_gen)} generated already; {len(rest)} to generate', flush=True)
    if rest:
        import sglang as sgl
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        eng = sgl.Engine(model_path=model, dp_size=int(os.environ.get('DP', 8)), tp_size=1, mem_fraction_static=0.85,
                         log_level='warning')
        sp = {'temperature': float(os.environ.get('TEMP', 1.0)), 'top_p': 1.0, 'max_new_tokens': MAX_NEW}
        with open(gen_f, 'a') as gf:
            for c in range(0, len(rest), CHUNK):
                part, t0 = rest[c:c + CHUNK], time.time()
                texts = [tok.apply_chat_template(variant(t['prompt']), tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=os.environ.get('THINKING', '1') == '1') for t in part]
                res = eng.generate(prompt=texts, sampling_params=sp)
                ntok = 0
                for t, r in zip(part, res):
                    mi = r['meta_info']
                    rec = {**{k: t[k] for k in ('key', 'split', 'pid', 'i', 'label')}, 'response': r['text'],
                           'finish': mi['finish_reason'].get('type') if isinstance(mi.get('finish_reason'), dict) else mi.get('finish_reason'),
                           'prompt_tokens': mi['prompt_tokens'], 'completion_tokens': mi['completion_tokens']}
                    ntok += mi['completion_tokens']
                    gf.write(json.dumps(rec) + '\n'); gf.flush()
                    futs.append(pool.submit(score_one, rec))
                print(f'gen {c + len(part)}/{len(rest)}: {ntok / (time.time() - t0):.0f} tok/s, '
                      f'{sum(f.done() for f in futs)}/{len(futs)} judged', flush=True)
        # finish every judge first: sglang's Engine.shutdown kills this process's whole child tree, including in-flight
        # g++ / solutions / checkers, which then score as compile errors (3 SFT train samples did; audit 2026-10-04)
        for f in futs:
            f.exception()
        eng.shutdown()
    with open(sc_f, 'a') as sf:
        for f in futs:
            sf.write(json.dumps(f.result()) + '\n'); sf.flush()
    summarize(out)


def summarize(out):
    gen = {r['key']: r for r in load(out / 'gen.jsonl')}
    sc = {r['key']: r for r in load(out / 'scores.jsonl')}
    lines = []
    for split in sorted({r['split'] for r in sc.values()}):
        rs = [r for r in sc.values() if r['split'] == split and not r.get('infra')]  # infra zeros are not scores
        g = [gen[r['key']] for r in rs if r['key'] in gen]
        per = defaultdict(list)
        for r in rs:
            per[r['pid']].append(r['score'])
        pm = sorted(sum(v) / len(v) for v in per.values())
        stat = defaultdict(int)
        for r in rs:
            stat[r['status']] += 1
        lines += [f'== {split}: {len(rs)} samples over {len(per)} problems',
                  f'mean score {sum(r["score"] for r in rs) / len(rs):.2f} / 100; zero {sum(r["score"] == 0 for r in rs) / len(rs):.1%}; '
                  f'full {sum(r["score"] >= 100 for r in rs) / len(rs):.1%}',
                  f'truncated (finish=length) {sum(x["finish"] == "length" for x in g) / max(1, len(g)):.1%}; '
                  f'mean completion tokens {sum(x["completion_tokens"] for x in g) / max(1, len(g)):.0f}',
                  f'status {dict(stat)}',
                  f'problems with mean 0: {sum(m == 0 for m in pm)}/{len(pm)}; mean >= 50: {sum(m >= 50 for m in pm)}; '
                  f'mean = 100: {sum(m >= 100 for m in pm)}',
                  f'per-problem mean quartiles: {[round(pm[int(q * (len(pm) - 1))], 1) for q in (0, .25, .5, .75, 1)]}']
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
