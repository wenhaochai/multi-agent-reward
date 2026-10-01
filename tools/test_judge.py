import sys, time, json
sys.path.insert(0, '/scratch/gpfs/GROUP/USER/project/miles-q38-build')
from miles_team.fcs_judge import judge
A = '/scratch/gpfs/GROUP/USER/project/Frontier-CS/algorithmic/problems'
for p in sys.argv[1:]:
    for name in ('reference.cpp', 'gpt5.cpp'):
        try:
            src = open(f'{A}/{p}/examples/{name}').read()
        except FileNotFoundError:
            continue
        t = time.time(); r = judge(f'{A}/{p}', src, case_workers=6)
        print(json.dumps({'p': p, 'sol': name, 'score': round(r['score'], 2), 'status': r['status'], 'n': len(r['cases']), 'sec': round(time.time() - t, 1)}), flush=True)
