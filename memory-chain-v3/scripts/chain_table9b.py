"""Table 9 waiter (2026-10-27): every 10 min for up to 8 h, look at which of GPUs 1-5 others use; all free -> the
5-card run t9t; only GPU2 taken -> the 4-card run t9t4 (GPUs 3,1,4,5); anything else -> keep waiting.
usage: python -X utf8 chain_table9b.py"""
import importlib.util, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('r', HERE / 'scripts' / 'run_remote_all.py')
sys.argv = ['x']
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
case = None
for attempt in range(48):
    busy, _ = m.status('wait-t9b-status.log')
    print(time.ctime(), 'busy', busy, flush=True)
    if not busy: case = 't9t'; break
    if set(busy) <= {2}: case = 't9t4'; break
    time.sleep(600)
print('CASE', case, flush=True)
if case:
    import os
    env = dict(os.environ, V3_REF='1c4fc72')
    rc = subprocess.call([sys.executable, '-X', 'utf8', str(HERE / 'scripts' / 'run_remote_all.py'), case, 'final'], env=env, cwd=HERE)
    print('EXIT9', rc, flush=True)
