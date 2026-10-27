"""Copy the small files of temp/memory-chain-v3 (REPORT, bench/run_tests, host, scripts, results without big logs)
into temp/rl-stage-memory-chain-v3/memory-chain-v3 for the rl snapshot commit, and check that no literal password
assignment is in the stage. usage: python -X utf8 stage_backup.py"""
import re, shutil
from pathlib import Path

src = Path(r'D:\Projects\fortune\Isaac\temp\memory-chain-v3')
dst = Path(r'D:\Projects\fortune\Isaac\temp\rl-stage-memory-chain-v3\memory-chain-v3')
if dst.exists(): shutil.rmtree(dst)
LIM = 5 * 2**20
skip_ext = ('.zip', '.pt', '.pstats', '.gz', '.so', '.pyc')
skip_names = {'train.log', 'telemetry.jsonl', 'memory.jsonl', 'episodes.jsonl', 'xvfb.log'}
n = total = 0


def put(p, rel):
    global n, total
    if p.suffix in skip_ext or p.name in skip_names or p.stat().st_size >= LIM: return
    q = dst / rel; q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q); n += 1; total += p.stat().st_size


put(src / 'REPORT.md', 'REPORT.md')
for f in ['bench_v3.py', 'run_tests.py']: put(src / f, f)
for d in ['host', 'scripts']:
    for p in (src / d).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts: put(p, p.relative_to(src))
for p in (src / 'results').rglob('*'):
    if not p.is_file(): continue
    rel = p.relative_to(src); parts = rel.parts
    if any(x.startswith('tests-') for x in parts) and p.suffix == '.log': continue   # (per-test logs: summary.json kept)
    if 'actorprof' in parts and not p.name.endswith('.json'): continue
    if 'code-' in p.name and not p.name.endswith('.commit.txt'): continue
    put(p, rel)
print(n, 'files', round(total / 2**20, 1), 'MiB')
pat = re.compile(rb"password\s*=\s*['\"][^'\"]+['\"]|sshpass")
# (bench_v3.py's runtime regex that reads the password from the v2 script contains the pattern itself)
hits = [str(p) for p in dst.rglob('*') if p.is_file() and p.name != 'stage_backup.py'   # (this checker names the patterns)
        and any(pat.search(l) and b're.search' not in l for l in p.read_bytes().splitlines())]
print('literal password assignments:', hits)
assert not hits, hits
