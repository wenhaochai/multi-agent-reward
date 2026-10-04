"""Checker verdicts on val problem 23 under the old 256 MiB address-space limit, the new TRUSTED_AS limit and no
limit (fcs_judge audit 2026-10-04). Run in miles.sif: python3 tools/check_trusted_limits.py"""
import json, sys, subprocess
from pathlib import Path
sys.path.insert(0, '/scratch/gpfs/GROUP/USER/project/miles-q38-build')
from miles_team.fcs_judge import load_problem, _compile_helper, _limits, TRUSTED_AS
D = '/scratch/gpfs/GROUP/USER/project/miles-q38-build/data/'
label = [json.loads(l)['label'] for l in open(D + 'fcs_val172.jsonl') if json.loads(l)['label'].rstrip('/').endswith('/23')][0]
pr = load_problem(Path(label)); exe = _compile_helper(pr['dir'], pr['checker']); td = pr['dir'] / 'testdata'
def run(cmd, lim):
    r = subprocess.run(cmd, capture_output=True, preexec_fn=lim, timeout=60)
    return r.returncode, (r.stdout or r.stderr).decode(errors='replace')[:60].replace('\n', ' ')
same = 0
for c in pr['cases']:
    ans = td / c[1] if (td / c[1]).exists() else td / c[1].replace('.ans', '.out')
    cmd = [str(exe), str(td / c[0]), str(ans), str(ans)]
    old, new, none = run(cmd, _limits(10, 256 << 20)), run(cmd, _limits(10, TRUSTED_AS)), run(cmd, _limits(10, None))
    same += new == none
    if c is pr['cases'][6]: print('case 7 old', old, '| new', new)
print(label, 'cases', len(pr['cases']), 'new == unlimited:', same, '| interactive:', pr['interactive'])
