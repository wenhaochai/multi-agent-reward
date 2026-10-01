"""Same-machine parity: the official Frontier-CS judge (go-judge + node orchestrator, served by tools/gojudge on this
node) and miles_team/fcs_judge.py score the same solutions, one judge after the other under the same load.
Input: the stratified sample of fcs_parity.jsonl (solution, problem, official January score).
usage: python fcs_parity_same_env.py SAMPLE.jsonl OUT.jsonl API_PORT [CONC]"""
import json, sys, time, urllib.request, uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miles_team.fcs_judge import judge

ALG = Path('/scratch/gpfs/GROUP/USER/project/Frontier-CS/algorithmic')


def official(port, pid, code):
    b = uuid.uuid4().hex
    body = b''.join([f'--{b}\r\nContent-Disposition: form-data; name="pid"\r\n\r\n{pid}\r\n'.encode(),
                     f'--{b}\r\nContent-Disposition: form-data; name="lang"\r\n\r\ncpp\r\n'.encode(),
                     f'--{b}\r\nContent-Disposition: form-data; name="code"; filename="sol.cpp"\r\nContent-Type: text/plain\r\n\r\n'.encode(),
                     code.encode(), f'\r\n--{b}--\r\n'.encode()])
    req = urllib.request.Request(f'http://127.0.0.1:{port}/submit', data=body,
                                 headers={'Content-Type': f'multipart/form-data; boundary={b}'})
    sid = json.load(urllib.request.urlopen(req, timeout=60))['sid']
    t = time.time()
    while time.time() - t < 1800:
        try:
            r = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/result/{sid}', timeout=30))
        except Exception:
            r = {}
        if r.get('status') in ('done', 'error'):
            return r.get('score') if r.get('status') == 'done' else None, r.get('status'), r.get('error')
        time.sleep(1)
    return None, 'timeout', None


def ours(row):
    r = judge(ALG / 'problems' / row['problem'], open(ALG / 'solutions' / row['solution']).read(), case_workers=4)
    return r['score']


def main():
    sample, out, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
    conc = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    rows = [json.loads(l) for l in open(sample)]
    t = time.time()
    with ThreadPoolExecutor(conc) as ex:
        offs = list(ex.map(lambda r: official(port, r['problem'], open(ALG / 'solutions' / r['solution']).read()), rows))
    print(f'official judge: {time.time() - t:.0f}s', flush=True)
    t = time.time()
    with ProcessPoolExecutor(conc) as ex:
        mine = list(ex.map(ours, rows))
    print(f'local judge: {time.time() - t:.0f}s', flush=True)
    with open(out, 'w') as f:
        for r, (o, st, err), m in zip(rows, offs, mine):
            f.write(json.dumps({'solution': r['solution'], 'problem': r['problem'], 'interactive': r['interactive'],
                                'jan': r['official'], 'official_here': o, 'off_status': st, 'off_error': err,
                                'local': m}) + '\n')
    res = [json.loads(l) for l in open(out)]
    ok = [r for r in res if r['official_here'] is not None]
    print(f'official errors/timeouts: {len(res) - len(ok)}')
    for inter in (False, True):
        d = [abs(r['local'] - r['official_here']) for r in ok if r['interactive'] == inter]
        j = [abs(r['jan'] - r['official_here']) for r in ok if r['interactive'] == inter]
        print(f'interactive={inter}: n={len(d)} local-vs-official mean|diff|={sum(d)/max(1,len(d)):.2f} exact={sum(x < 0.5 for x in d)}; '
              f'jan-vs-official-here mean|diff|={sum(j)/max(1,len(j)):.2f} exact={sum(x < 0.5 for x in j)}')


if __name__ == '__main__':
    main()
