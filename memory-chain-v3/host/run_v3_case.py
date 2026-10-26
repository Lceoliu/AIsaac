"""Host-side runner for one memory-chain-v3 benchmark case (python3, stdlib only).

The docker command is the v2 long-59 command (command.json) with only these
substitutions: case name / run directory, code mount, container memory limit,
the in-container timeout, the selected free GPU UUIDs and --memory-horizon.
Telemetry and result.json follow temp/memory-chain-v2/run_memory_pipeline_full.py
(steady = rows[10:], x_real_time weighted by wall_s).

usage: python3 run_v3_case.py <spec.json>
"""
import csv, json, os, pathlib, statistics, subprocess, sys, time

spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(spec['root'])
case = spec['case']
out = root / 'runs' / case
out.mkdir()
gpus = dict(line.strip().split(', ') for line in subprocess.check_output(
    ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines())
selected = [gpus[str(i)] for i in spec['gpu_indices']]
occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
if any(line.split(',')[0].strip() in selected for line in occupied.splitlines()):
    (out / 'result.json').write_text(json.dumps(dict(case=case, exit_status=3, reason='selected GPU occupied',
                                                     occupied=occupied, selected=selected), indent=2))
    raise SystemExit('Selected GPU is occupied; no benchmark launched.')
template = json.loads(pathlib.Path(spec['template']).read_text())
old_case = template['args']['case']
cmd = []
for x in template['command']:
    x = x.replace(old_case, case)
    if spec['code_dir']: x = x.replace('memory-v2-code-' + old_case.removeprefix('memory-v2-'), spec['code_dir'])
    if x.startswith(('CUDA_VISIBLE_DEVICES=', 'NVIDIA_VISIBLE_DEVICES=')):
        x = x.split('=')[0] + '=' + ','.join(selected)
    cmd.append(x)
if spec['code_dir'] and spec['code_dir'] not in ' '.join(cmd):   # the old architecture's template: no code mount of its own
    at = next(i for i, x in enumerate(cmd) if x.endswith('/workspace:/bench/workspace:ro')) + 1
    cmd[at:at] = ['-v', f"{root}/{spec['code_dir']}/python:/bench/workspace/python:ro"]
for flag in ('--memory', '--memory-swap'):
    cmd[cmd.index(flag) + 1] = spec['memory']
cmd[cmd.index('--name') + 1] = 'aisaac-bench-' + case
if (spec.get('env') or {}).get('ISAAC_RL_MEM_LOG'): (out / 'memlog').mkdir()   # the workers' role dumps
for key, value in (spec.get('env') or {}).items():   # opt-in diagnostics, e.g. ISAAC_MEMORY_ACCOUNT=1
    have = [i for i, x in enumerate(cmd) if x.startswith(key + '=') and cmd[i - 1] == '-e']
    if have: cmd[have[0]] = f'{key}={value}'   # a template variable: replaced in place (e.g. TORCHINDUCTOR_COMPILE_THREADS)
    else: at = cmd.index('--entrypoint'); cmd[at:at] = ['-e', f'{key}={value}']
script = cmd[-1]
assert script.count('timeout --signal=INT --kill-after=45s 1500 ') == 1
script = script.replace('timeout --signal=INT --kill-after=45s 1500 ', f"timeout --signal=INT --kill-after=45s {spec['timeout']} ")
if spec.get('horizon') is not None:   # None: the old architecture (stock workspace code, no source pipeline)
    assert script.count(' --source-amp bf16 ') == 1
    script = script.replace(' --source-amp bf16 ', f" --source-amp bf16 --memory-horizon {spec['horizon']} ")
if spec.get('hours') is not None:
    assert script.count(' --game-hours 60.0 ') == 1
    script = script.replace(' --game-hours 60.0 ', f" --game-hours {spec['hours']} ")
if spec.get('source_devices'):   # fewer-GPU in-group A/B: only the source device list changes
    assert script.count(' --source-devices cuda:0,cuda:2,cuda:3,cuda:4 ') == 1
    script = script.replace(' --source-devices cuda:0,cuda:2,cuda:3,cuda:4 ', f" --source-devices {spec['source_devices']} ")
for old, new in spec.get('train_edits') or []:   # ablations: exact, single-occurrence edits of train_tok's arguments
    assert script.count(old) == 1, (old, script.count(old))
    script = script.replace(old, new)
cmd[-1] = script
assert (spec['code_dir'] is None or spec['code_dir'] in ' '.join(cmd)) and case in ' '.join(cmd) and old_case not in ' '.join(cmd)
train = template['train'] + (['--memory-horizon', str(spec['horizon'])] if spec.get('horizon') is not None else [])
if spec.get('source_devices'): train[train.index('--source-devices') + 1] = spec['source_devices']
if spec.get('hours') is not None: train[train.index('--game-hours') + 1] = str(spec['hours'])
if spec.get('train_edits'): train = script.split('train_tok.py', 1)[1].split()   # the edited argument list as run
(out / 'command.json').write_text(json.dumps({'command': cmd, 'train': train, 'spec': spec}, indent=2))
t0 = time.time()
with (out / 'train.log').open('w') as log, (out / 'telemetry.jsonl').open('w') as tele:
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    print('START', case, 'pid', proc.pid, flush=True)
    while proc.poll() is None:
        snap = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw',
                                        '--format=csv,noheader,nounits'], text=True)
        tele.write(json.dumps({'time': time.time(), 'gpus': snap, 'load': os.getloadavg()}) + '\n'); tele.flush()
        time.sleep(10)
    rc = proc.wait()
result = {'case': case, 'exit_status': rc, 'elapsed_seconds': time.time() - t0, 'spec': spec}
progress = out / 'progress.csv'
if progress.exists():
    rows = list(csv.DictReader(progress.open()))
    steady = rows[spec.get('discard', 10):]
    result['updates'] = len(rows)
    if rows: result['last'] = {k: rows[-1].get(k) for k in ['update', 'game_hours', 'errors', 'stalls', 'cf_done', 'cf_errors', 'searches', 'teach_held']}
    if steady:
        result['steady_updates'] = len(steady)
        result['steady_x_real_time'] = sum(float(r['x_real_time']) * float(r['wall_s']) for r in steady) / sum(float(r['wall_s']) for r in steady)
        keys = ['x_real_time', 'collect_s', 'update_s', 'wall_s', 'gpu_reserved_gb', 'wait_learner_s', 'c_learner_cores',
                'source_graph_frames', 'source_graph_frames_uid_max', 'source_alloc_gib_max', 'source_detached_frames']
        result['medians'] = {k: statistics.median(float(r[k]) for r in steady if r.get(k) not in ('', None))
                             for k in keys if any(r.get(k) not in ('', None) for r in steady)}
(out / 'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, ensure_ascii=False), flush=True)
raise SystemExit(rc)
