"""Like wait_then_run.py, and also wait until the host's other CPU work is low enough for the guard
(other_workload_cpu_exceeds_9_cores_sustained): 1- and 5-minute load averages below --load (default 7), checked every
10 minutes, at most --hours. usage: python -X utf8 wait_quiet_then_run.py --allow 2,5 [--load 7] [--hours 12] <ref> step ..."""
import importlib.util, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
args = sys.argv[1:]
opts = {}
for key in ('--allow', '--hours', '--load'):
    if key in args:
        i = args.index(key); opts[key] = args[i + 1]; del args[i:i + 2]
allow = {int(x) for x in opts.get('--allow', '').split(',') if x}
hours, load = float(opts.get('--hours', 12)), float(opts.get('--load', 7))
ref, steps = args[0], args[1:]
spec = importlib.util.spec_from_file_location('r', HERE / 'scripts' / 'run_remote_all.py')
sys.argv = ['x']
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
deadline = time.time() + hours * 3600
while True:
    busy, text = m.status('wait-quiet-status.log')
    avg = [float(x) for x in re.search(r'load average: ([\d.]+), ([\d.]+), ([\d.]+)', text).groups()]
    print(time.ctime(), 'busy', busy, 'load', avg, flush=True)
    if set(busy) <= allow and avg[0] < load and avg[1] < load: break
    if time.time() > deadline: print('GAVE UP', flush=True); sys.exit(1)
    time.sleep(600)
rc = subprocess.call([sys.executable, '-X', 'utf8', str(HERE / 'scripts' / 'run_remote_all.py')] + steps,
                     env=dict(os.environ, V3_REF=ref), cwd=HERE)
print('EXIT', rc, flush=True)
sys.exit(rc)
