"""The predictive-memory experiments on the 8x3090 host, driven from the PC (rl/docs/PREDICTIVE_MEMORY_DESIGN.md, milestone M4). The
login is read at run time from temp/memory-chain-v2/host_status.py (nothing of it is written anywhere).

  python -X utf8 pred_host.py deploy               committed HEAD -> cpu-wide/pred-code-<short>; runner, guard; zstandard
  python -X utf8 pred_host.py chain <file.json>    start a chain of cases on the host (setsid; survives this script)
  python -X utf8 pred_host.py status               the chain's state, the GPUs, the running case's last line
  python -X utf8 pred_host.py table <prefix>       per case: x_real_time, medians, memory, the switches' columns
  python -X utf8 pred_host.py get <remote> <local> a file

Chain file: {"cases": [{"case": ..., "arm": a0|a1|a2|a3|l0|l3 ..., "hours": 60, "env": {...}, ...}], "gpus": [3, 1, 4, 5, 2]}. A case runs when
no aisaac-* container (ours or the other line's: they are never touched) is up and three of the allowed GPUs have no compute process (first come first served with the other
sessions on the host, nothing of theirs is touched; the third of the three is the second actor's); inside: the 843x baseline's docker command (memory-chain-v2 baseline-command.json) with
this code, minibatch / micro 4096, the arm's switches, and guard_v3's rules (they stop our own container only).
"""
import json
import posixpath
import re
import subprocess
import sys
import tarfile
import io
from pathlib import Path

import paramiko

ISAAC = Path(r'D:\Projects\fortune\Isaac')
ROOT = '/home/runaiwusheng/aisaac-bench-20261019/cpu-wide'
SWITCHES = dict(a0='', a1=' --aux-predict 1', a2=' --priv-critic 1', a3=' --aux-predict 1 --priv-critic 1',
                l0=' --log-stream 1', l1=' --aux-predict 1 --log-stream 1', l2=' --priv-critic 1 --log-stream 1',
                l3=' --aux-predict 1 --priv-critic 1 --log-stream 1')

CHAIN = r'''#!/usr/bin/env python3
"""On the host: run a chain's cases one after another under the guard. usage: run_chain.py <chain.json>"""
import json, os, pathlib, subprocess, sys, time
root = pathlib.Path('/home/runaiwusheng/aisaac-bench-20261019/cpu-wide')
chain = json.loads(pathlib.Path(sys.argv[1]).read_text())
state = root / 'pred' / (pathlib.Path(sys.argv[1]).stem + '.state')
def note(**k):
    with state.open('a') as f: f.write(json.dumps(dict(k, time=time.strftime('%F %T'))) + '\n')
def host_cores(seconds=30):
    """Busy CPUs of the whole host over a little while."""
    def read():
        v = [int(x) for x in pathlib.Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
        return sum(v) - v[3] - v[4], time.time()
    a, t0 = read()
    time.sleep(seconds)
    b, t1 = read()
    return (b - a) / os.sysconf('SC_CLK_TCK') / (t1 - t0)
def pick(allowed, n=3):
    """The first n of the allowed GPUs without a compute process, or (None, why). (In order: actor 0, learner, actor 1.)"""
    if subprocess.run("docker ps --format '{{.Names}}' | grep -q '^aisaac-'", shell=True).returncode == 0: return None, 'an aisaac container is up'
    idx = dict(l.strip().split(', ') for l in subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines())
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).split()
    free = [g for g in allowed if idx[str(g)] not in apps]
    return (free[:n], None) if len(free) >= n else (None, f'free {free} of {allowed}')
queue = list(chain['cases'])
settled = False   # (the chain holds the host: within it no second wait)
def ready(c):
    gpus, why = pick(c.get('gpus') or chain['gpus'])
    if why is None and chain.get('max_busy_cores'):   # (a measurement: only on a quiet host, the guard's own rule)
        if host_cores() > chain['max_busy_cores']:
            why = f'host busy, more than {chain["max_busy_cores"]} cores'
    return gpus, why
while queue:
    c = queue.pop(0)
    if (root / 'pred' / 'STOP').exists(): note(event='stop file'); break
    last = None
    while True:
        gpus, why = ready(c)
        if why is None and not settled and chain.get('settle_s'):
            # quiet: the other line on the host may be seeing the same window; wait and look again before taking it
            note(event='quiet, settling', case=c['case'], seconds=chain['settle_s'])
            time.sleep(chain['settle_s'])
            gpus, why = ready(c)
            if why is not None: why = 'after settling: ' + why
            else: settled = True
        if why is None: break
        if why != last:   # (only a change is noted)
            note(event='wait', case=c['case'], why=why)
            last = why
        time.sleep(60)
    free = int(subprocess.check_output(['df', '--output=avail', '-B1G', str(root)], text=True).split()[1])
    if free < chain.get('min_disk_gb', 200):
        note(event='disk', free=free); break
    spec = dict(root=str(root), case=c['case'], template=str(root / 'pred' / 'template.json'), code_dir=chain['code_dir'], gpu_indices=gpus,
                memory=chain.get('memory', '72g'), timeout=c.get('timeout', 2400), hours=c['hours'], train_edits=c['edits'], env=c.get('env') or {},
                discard=c.get('discard', 10), horizon=None, source_devices=None)
    p = root / 'pred' / f"{c['case']}.spec.json"
    p.write_text(json.dumps(spec, indent=1))
    note(event='start', case=c['case'], gpus=gpus)
    with (root / 'logs' / f"{c['case']}.dispatch.log").open('w') as log:
        rc = subprocess.call(['python3', str(root / 'pred' / 'guard_pred.py'), str(p)], stdout=log, stderr=subprocess.STDOUT)
    note(event='done', case=c['case'], rc=rc)
    if rc != 0:
        text = (root / 'logs' / f"{c['case']}.dispatch.log").read_text(errors='replace')
        if 'other_workload' in text and c.get('retry', 0) < chain.get('retries', 6):   # (someone else's burst: the case again)
            again = dict(c, case=c.get('base', c['case']) + f"-r{c.get('retry', 0) + 1}", base=c.get('base', c['case']), retry=c.get('retry', 0) + 1)
            again['env'] = {k: v.replace(c['case'], again['case']) for k, v in (c.get('env') or {}).items()}
            queue.insert(0, again)
            settled = False
            note(event='retry', case=c['case'], next=again['case'])
            continue
        note(event='chain stopped', case=c['case'], rc=rc)
        break
note(event='chain done')
'''


def client():
    src = (ISAAC / 'temp' / 'memory-chain-v2' / 'host_status.py').read_text(encoding='utf-8')
    m = re.search(r"c\.connect\('([^']+)',username='([^']+)',password=('[^']*'|\"[^\"]*\")", src)
    c = paramiko.SSHClient()
    c.load_system_host_keys()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(m.group(1), username=m.group(2), password=m.group(3)[1:-1], look_for_keys=False, allow_agent=False, timeout=15)
    return c


def run(c, cmd, timeout=120):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', 'replace') + e.read().decode('utf-8', 'replace')
    return o.channel.recv_exit_status(), out


def repo():
    return subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True, cwd=Path(__file__).parent).strip()


def deploy():
    r = repo()
    short = subprocess.check_output(['git', '-C', r, 'rev-parse', '--short', 'HEAD'], text=True).strip()
    if subprocess.run(['git', '-C', r, 'diff', '--quiet', 'HEAD', '--', 'bridge']).returncode:
        print('WARNING: uncommitted changes under bridge/ are not deployed')
    tar = subprocess.check_output(['git', '-C', r, 'archive', 'HEAD', 'bridge/python', 'bridge/abplus/abp_bridge.lua'])
    c = client()
    s = c.open_sftp()
    code = f'pred-code-{short}'
    d = f'{ROOT}/{code}'
    print(run(c, f'mkdir -p {d} {ROOT}/pred {ROOT}/runs {ROOT}/logs; rm -f {d}.tar')[1], end='')
    with s.open(f'{d}.tar', 'wb') as f:
        f.write(tar)
    print(run(c, f'cd {d} && tar -xf ../{code}.tar && mkdir -p python && cp -a bridge/python/. python/ && cp bridge/abplus/abp_bridge.lua . && '
                 f'mv ../{code}.tar source.tar && echo {short} > COMMIT && grep -n "^local VERSION" abp_bridge.lua && ls | head', 300)[1], end='')
    # the runner and the guard: memory-chain-v3's, the bridge mounted too
    run_src = (ISAAC / 'temp' / 'memory-chain-v3' / 'host' / 'run_v3_case.py').read_text(encoding='utf-8')
    old = """    cmd[at:at] = ['-v', f"{root}/{spec['code_dir']}/python:/bench/workspace/python:ro"]\n"""
    assert run_src.count(old) == 1
    run_src = run_src.replace(old, """    cmd[at:at] = ['-v', f"{root}/{spec['code_dir']}/python:/bench/workspace/python:ro",
                  '-v', f"{root}/{spec['code_dir']}/abp_bridge.lua:/bench/workspace/abp_bridge.lua:ro"]\n""")
    guard_src = (ISAAC / 'temp' / 'memory-chain-v3' / 'host' / 'guard_v3.py').read_text(encoding='utf-8')
    old = "runner = ['python3', str(root/'v3'/'run_v3_case.py'), spec_path]"
    assert guard_src.count(old) == 1
    guard_src = guard_src.replace(old, "runner = ['python3', str(root/'pred'/'run_pred_case.py'), spec_path]")
    tpl = (ISAAC / 'temp' / 'memory-chain-v2' / 'baseline-command.json').read_text(encoding='utf-8')
    for name, text in (('run_pred_case.py', run_src), ('guard_pred.py', guard_src), ('template.json', tpl), ('run_chain.py', CHAIN)):
        with s.open(f'{ROOT}/pred/{name}', 'w') as f:
            f.write(text)
    # zstandard: the wheel unpacked into the container's venv (one new package, nothing else touched)
    wheel = next(Path(sys.argv[2]).glob('zstandard-*cp310*.whl')) if len(sys.argv) > 2 else None
    site = f'{ROOT}/home/isaac-abplus/.venv/lib/python3.10/site-packages'
    if wheel is not None:
        with s.open(f'{ROOT}/pred/{wheel.name}', 'wb') as f:
            f.write(wheel.read_bytes())
        print(run(c, f"[ -d {site}/zstandard ] && echo 'zstandard already there' || python3 -c \"import zipfile; zipfile.ZipFile('{ROOT}/pred/{wheel.name}').extractall('{site}')\"; "
                     f"ls -d {site}/zstandard* | head -3")[1], end='')
    print(run(c, f"docker run --rm --entrypoint sh -v {ROOT}/home/isaac-abplus/.venv:/v:ro aisaac-bench:20261019 -c "
                 "\"/v/bin/python -c 'import zstandard; print(zstandard.__version__, zstandard.backend)'\"", 60)[1], end='')
    (Path(__file__).parent / '.pred_code').write_text(code)
    print('deployed', code)
    c.close()


def edits_for(arm, base, extra=()):
    return [[' --minibatch 8192 --micro 8192 ', ' --minibatch 4096 --micro 4096 '],
            [' --learner-fast 2 ', ' --learner-fast 2' + SWITCHES[arm] + ' ']] + list(base) + list(extra)


def chain(path):
    ch = json.loads(Path(path).read_text(encoding='utf-8'))
    ch.setdefault('code_dir', (Path(__file__).parent / '.pred_code').read_text().strip())
    ch.setdefault('gpus', [3, 1, 4, 5, 2])
    for k in ch['cases']:
        k['edits'] = edits_for(k['arm'], ch.get('edits') or [], k.get('edits') or [])
    c = client()
    s = c.open_sftp()
    name = Path(path).name
    with s.open(f'{ROOT}/pred/{name}', 'w') as f:
        f.write(json.dumps(ch, indent=1))
    # (the channel closes at once: nothing of the chain's holds it)
    c.exec_command(f'cd {ROOT}/pred && setsid nohup python3 run_chain.py {ROOT}/pred/{name} > {name}.out 2>&1 < /dev/null &')
    import time
    time.sleep(3)
    print(run(c, "pgrep -af '[r]un_chain' | cut -c1-120")[1], end='')
    c.close()


def status():
    c = client()
    print(run(c, f"date; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\\n' ';'; echo; docker ps --format '{{{{.Names}}}} {{{{.Status}}}}' | grep aisaac; "
                 f"free -g | sed -n 2p; df -h {ROOT} | tail -1; cd {ROOT}/pred; for f in *.state; do echo $f; tail -n 3 $f; done; pgrep -af run_chain | cut -c1-90")[1])
    c.close()


TABLE = r'''
import csv, glob, json, os, statistics as st, sys
root = '/home/runaiwusheng/aisaac-bench-20261019/cpu-wide'
keys = ('collect_s', 'update_s', 'wall_s', 'u_ppo', 'u_recompute', 'u_batch', 'u_opt', 'u_aux', 'u_priv', 'u_log', 'handoff_s', 'publish_s', 'c_learner_cores',
        'gpu_reserved_gb', 'gpu_reserved_gb_actor', 'grad_norm', 'log_raw_gb', 'log_dropped', 'log_write_s', 'priv_archive_share', 'aux_me_256', 'aux_ms_256', 'aux_me_64', 'aux_ms_64')
for d in sorted(glob.glob(root + '/runs/' + sys.argv[1] + '*')):
    case = os.path.basename(d)
    if not os.path.exists(d + '/progress.csv'):
        print(case, 'no progress'); continue
    rows = list(csv.DictReader(open(d + '/progress.csv')))
    steady = rows[int(sys.argv[2]) if len(sys.argv) > 2 else 10:]
    out = dict(updates=len(rows), hours=round(float(rows[-1]['game_hours']), 1) if rows else 0)
    if steady:
        w = [float(r['wall_s']) for r in steady]
        out['x'] = round(sum(float(r['x_real_time']) * a for r, a in zip(steady, w)) / sum(w), 1)
        for k in keys:
            v = [float(r[k]) for r in steady if r.get(k) not in ('', None)]
            if v: out[k] = round(st.median(v), 3)
    mem = root + '/logs/' + case + '.memory.jsonl'
    if os.path.exists(mem):
        g = [json.loads(l).get('memory_gib') for l in open(mem)]
        g = [x for x in g if x is not None]
        if g: out['mem_gib_max'] = round(max(g), 1); out['mem_gib_last'] = round(g[-1], 1)
    res = d + '/result.json'
    if os.path.exists(res):
        r = json.load(open(res)); out['exit'] = r.get('exit_status')
    log = root + '/logs/' + case + '.dispatch.log'
    if os.path.exists(log):
        stop = [l.strip() for l in open(log, errors='replace') if 'STOP_OWN' in l]
        if stop: out['stop'] = stop[-1][:80]
    tl = d + '/train.log'
    if os.path.exists(tl):
        out['tracebacks'] = sum('Traceback' in l for l in open(tl, errors='replace'))
    s = d + '/stream'
    if os.path.isdir(s):
        out['stream_gb'] = round(sum(os.path.getsize(os.path.join(s, f)) for f in os.listdir(s)) / 2**30, 3)
    print(case, json.dumps(out))
'''


def table(prefix, discard='10'):
    c = client()
    s = c.open_sftp()
    with s.open(f'{ROOT}/pred/table.py', 'w') as f:
        f.write(TABLE)
    print(run(c, f'python3 {ROOT}/pred/table.py {prefix} {discard}', 120)[1])
    c.close()


def get(remote, local):
    c = client()
    s = c.open_sftp()
    s.get(remote if remote.startswith('/') else posixpath.join(ROOT, remote), local)
    c.close()


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'deploy':
        deploy()
    elif cmd == 'chain':
        chain(sys.argv[2])
    elif cmd == 'status':
        status()
    elif cmd == 'table':
        table(*sys.argv[2:])
    elif cmd == 'get':
        get(sys.argv[2], sys.argv[3])
    elif cmd == 'run':
        c = client()
        print(run(c, sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 120)[1])
        c.close()
