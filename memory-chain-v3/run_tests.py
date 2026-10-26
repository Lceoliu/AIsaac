"""Run every bridge/python/tests/test_*.py of one checkout (CPU), one process each.

usage: python -X utf8 run_tests.py <checkout> <out_dir> [timeout_s] [name ...]
Writes <out_dir>/<test>.log (stdout+stderr) and <out_dir>/summary.json
(name, exit code, seconds). Exit code: 0 always; the table is the result.
"""
import json, os, subprocess, sys, time
from pathlib import Path

checkout, out = Path(sys.argv[1]), Path(sys.argv[2])
timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 900
only = set(sys.argv[4:])
root = checkout / 'bridge' / 'python'
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONPATH='.', PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2')
rows = []
for test in sorted((root / 'tests').glob('test_*.py')):
    if only and test.stem not in only: continue
    start = time.time()
    try:
        p = subprocess.run([sys.executable, '-X', 'utf8', str(test.relative_to(root))], cwd=root, env=env,
                           capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
        code, text = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        code, text = 'timeout', (e.stdout or '') + (e.stderr or '') if isinstance(e.stdout, str) else 'timeout'
    (out / f'{test.stem}.log').write_text(str(text), encoding='utf-8')
    rows.append(dict(name=test.stem, exit=code, seconds=round(time.time() - start, 1)))
    print(rows[-1], flush=True)
(out / 'summary.json').write_text(json.dumps(rows, indent=1))
