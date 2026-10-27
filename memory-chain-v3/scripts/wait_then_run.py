"""Wait (every 10 min, at most --hours) until the GPUs others use are within --allow (e.g. 2), then run
run_remote_all.py with V3_REF=<ref> and the given steps. usage: python -X utf8 wait_then_run.py --allow 2 --hours 8 <ref> step ..."""
import importlib.util, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
args = sys.argv[1:]
allow = {int(x) for x in args[args.index('--allow') + 1].split(',') if x} if '--allow' in args else set()
hours = float(args[args.index('--hours') + 1]) if '--hours' in args else 8.
rest = [a for i, a in enumerate(args) if a not in ('--allow', '--hours') and (i == 0 or args[i - 1] not in ('--allow', '--hours'))]
ref, steps = rest[0], rest[1:]
spec = importlib.util.spec_from_file_location('r', HERE / 'scripts' / 'run_remote_all.py')
sys.argv = ['x']
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
deadline = time.time() + hours * 3600
while True:
    busy, _ = m.status('wait-then-run-status.log')
    print(time.ctime(), 'busy', busy, flush=True)
    if set(busy) <= allow: break
    if time.time() > deadline: print('GAVE UP', flush=True); sys.exit(1)
    time.sleep(600)
rc = subprocess.call([sys.executable, '-X', 'utf8', str(HERE / 'scripts' / 'run_remote_all.py')] + steps,
                     env=dict(os.environ, V3_REF=ref), cwd=HERE)
print('EXIT', rc, flush=True)
sys.exit(rc)
