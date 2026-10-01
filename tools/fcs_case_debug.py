"""Per-case verdicts of the local judge for given solution files (paths under Frontier-CS/algorithmic/solutions)."""
import sys, time
sys.path.insert(0, '/scratch/gpfs/GROUP/USER/project/miles-q38-build')
from miles_team.fcs_judge import judge, load_problem
A = '/scratch/gpfs/GROUP/USER/project/Frontier-CS/algorithmic'
for sol in sys.argv[1:]:
    p = sol.split('/')[0]
    t = time.time()
    r = judge(f'{A}/problems/{p}', open(f'{A}/solutions/{sol}').read(), case_workers=int(__import__('os').environ.get('CW', 2)))
    print(f'== {sol} score {r["score"]:.1f} {r["status"]} {time.time()-t:.0f}s n={len(r["cases"])}', r.get('msg', '')[:300])
    for i, (c, ratio, info) in enumerate(zip(load_problem(__import__('pathlib').Path(f'{A}/problems/{p}'))['cases'], r['cases'], r.get('case_info', []))):
        if ratio < 1:
            print(f'  case {c[0]} {c[2]} {c[3]} ratio={ratio:.3f} {info}')
