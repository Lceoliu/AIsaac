"""Local controller for memory-chain-v3 runs on the shared 8x3090 Docker host.

Connection details are read from temp/memory-chain-v2/host_status.py at run
time (not copied into this file or any result). Every command prints its
remote exit status; logs land in temp/memory-chain-v3/results/.

  status                                   uptime, memory, GPUs, compute apps, containers
  deploy <tag>                             git archive HEAD:bridge/python -> cpu-wide/memory-v3-code-<tag>/python
  launch <case> <tag> <horizon> [hours] [gpu-indices]  detached guard+runner (v2 long-59 config, 72g)
  collect <case>                           download runs/<case>/* and logs/<case>.guard.*
  stop <case> <reason>                     docker stop (own container only)
  gate <name> <tag> <gpu-indices> <shell>  one container running <shell> in /bench/workspace/python
"""
import io, json, re, shlex, subprocess, sys, time, zipfile
from pathlib import Path
import paramiko

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / 'memory-chain-v2'
ROOT = '/home/runaiwusheng/aisaac-bench-20261019/cpu-wide'
TEMPLATE = 'memory-v2-actor2-shared4-bf16-rollout-sourceflash-sourcecompiled-overlaproll-hosttrim1-mem88-clock2-prefetch8192-sourceproc-pointwise-components-mps-edgeexport-staticcomponent-componentgraph-preemptvjp-actorpool-reserve7-latespill-long-59'
RESULTS = HERE / 'results'


def connect():
    src = (V2 / 'host_status.py').read_text()
    host, user, password = re.search(r"connect\('([^']+)',username='([^']+)',password='([^']*)'", src).groups()
    c = paramiko.SSHClient(); c.load_system_host_keys(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=password, look_for_keys=False, allow_agent=False, timeout=20, banner_timeout=60)
    return c


def run(c, command, timeout=120):
    _, o, e = c.exec_command(command, timeout=timeout)
    out, err = o.read().decode(errors='replace'), e.read().decode(errors='replace')
    return o.channel.recv_exit_status(), out, err


def main():
    mode = sys.argv[1]
    c = connect(); s = c.open_sftp()
    if mode == 'status':
        rc, out, err = run(c, "uptime; free -g; nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader;"
                              " echo APPS; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader;"
                              " echo CONTAINERS; docker ps --format '{{.Names}} {{.Status}}'")
        print(out + err); print('EXIT', rc)
    elif mode == 'deploy':
        tag = sys.argv[2]
        repo = HERE / 'candidate'
        ref = sys.argv[3] if len(sys.argv) > 3 else 'HEAD'   # e.g. 64ae451 = v2 as received
        head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', ref], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', 'bridge/python'], text=True).strip()
        assert not dirty, 'uncommitted bridge/python changes: deploy commits only\n' + dirty
        archive = subprocess.check_output(['git', '-C', str(repo), 'archive', '--format=zip', head, 'bridge/python'])
        # Same layout as v2's actor_bench.py: <dir>/python/... (prefix bridge/ removed)
        zin = zipfile.ZipFile(io.BytesIO(archive)); buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
            for info in zin.infolist():
                if info.is_dir(): continue
                z.writestr(info.filename.removeprefix('bridge/'), zin.read(info))
        local = RESULTS / f'code-{tag}.zip'; local.write_bytes(buf.getvalue())
        (RESULTS / f'code-{tag}.commit.txt').write_text(head + '\n')
        s.put(str(local), f'{ROOT}/memory-v3-code-{tag}.zip')
        rc, out, err = run(c, f'test ! -e {ROOT}/memory-v3-code-{tag} && python3 -m zipfile -e {ROOT}/memory-v3-code-{tag}.zip {ROOT}/memory-v3-code-{tag}'
                              f' && echo {head} > {ROOT}/memory-v3-code-{tag}/COMMIT && ls {ROOT}/memory-v3-code-{tag}/python | wc -l')
        print(out + err); print('DEPLOY', tag, head, 'EXIT', rc); sys.exit(rc)
    elif mode == 'launch':
        case, tag, horizon = sys.argv[2], sys.argv[3], int(sys.argv[4])
        hours = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != '-' else None
        indices = [int(i) for i in sys.argv[6].split(',')] if len(sys.argv) > 6 else [3, 2, 1, 4, 5]
        source_devices = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] != '-' else None
        env = dict(kv.split('=', 1) for kv in sys.argv[8:] if not kv.startswith('edits='))   # extra container env
        edits = next((json.loads(kv[6:]) for kv in sys.argv[8:] if kv.startswith('edits=')), None)   # [[old, new], ...]
        run(c, f'mkdir -p {ROOT}/v3 {ROOT}/v3/specs')
        for name in ('run_v3_case.py', 'guard_v3.py', 'audit_inside.py'):
            s.put(str(HERE / 'host' / name), f'{ROOT}/v3/{name}')
        s.put(str(V2 / (TEMPLATE.removeprefix('memory-v2-') + '-command.json')), f'{ROOT}/v3/template-long-59-command.json')
        s.put(str(V2 / 'baseline-command.json'), f'{ROOT}/v3/template-oldarch-baseline-command.json')
        old = tag == 'oldarch' or tag.startswith('oldarch:')   # the 843x old architecture: its own 3-GPU command;
        #   stock workspace code, or 'oldarch:<code tag>' = cpu-wide/memory-v3-code-<code tag> mounted over it
        spec = dict(root=ROOT, case='memory-v3-' + case,
                    template=f'{ROOT}/v3/template-{"oldarch-baseline" if old else "long-59"}-command.json',
                    code_dir=(f'memory-v3-code-{tag[8:]}' if tag.startswith('oldarch:') else None) if old else f'memory-v3-code-{tag}',
                    memory='72g', timeout=2400,
                    horizon=None if old else horizon, hours=hours,
                    gpu_indices=indices, discard=10, source_devices=source_devices, env=env, train_edits=edits)
        (RESULTS / f'{case}-spec.json').write_text(json.dumps(spec, indent=2))
        with s.open(f'{ROOT}/v3/specs/{case}.json', 'w') as f: f.write(json.dumps(spec, indent=2))
        rc, out, err = run(c, (f'test -d {ROOT}/{spec["code_dir"]}/python' if spec['code_dir'] else 'true') + f' && test ! -e {ROOT}/runs/memory-v3-{case}')
        assert rc == 0, ('code missing or case exists', out, err)
        launcher = (f"import subprocess; f=open({ROOT + '/logs/memory-v3-' + case + '.guard.log'!r},'w');"
                    f" p=subprocess.Popen(['python3',{ROOT + '/v3/guard_v3.py'!r},{ROOT + '/v3/specs/' + case + '.json'!r}],"
                    f"stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print('STARTED',p.pid)")
        rc, out, err = run(c, 'cd ' + ROOT + ' && python3 -c ' + shlex.quote(launcher))
        print(out + err); print('LAUNCH', case, 'EXIT', rc)
        (RESULTS / f'{case}-launch.log').write_text(f'{time.ctime()} {out}{err}EXIT {rc}\n')
    elif mode == 'collect':
        case = sys.argv[2]; full = 'memory-v3-' + case
        dst = RESULTS / case; dst.mkdir(exist_ok=True)
        for folder, names in [(ROOT + '/logs', [full + '.guard.log', full + '.guard.exit.json', full + '.guard.jsonl', full + '.dispatch.json', full + '.memory.jsonl']),
                              (ROOT + '/runs/' + full, None)]:
            try: listing = s.listdir(folder)
            except FileNotFoundError: continue
            for n in (names or listing):
                if n in listing and (n.endswith(('.json', '.jsonl', '.log', '.csv', '.pstats', '.txt'))):
                    s.get(folder + '/' + n, str(dst / n.removeprefix(full + '.')))
        try: prof = s.listdir(ROOT + '/runs/' + full + '/actorprof')   # (ISAAC_ACTOR_TORCH_PROFILE)
        except FileNotFoundError: prof = []
        if prof: (dst / 'actorprof').mkdir(exist_ok=True)
        for n in prof: s.get(ROOT + '/runs/' + full + '/actorprof/' + n, str(dst / 'actorprof' / n))
        for n in sorted(p.name for p in dst.iterdir()): print(n, (dst / n).stat().st_size if (dst / n).is_file() else 'dir')
        if (dst / 'result.json').exists(): print((dst / 'result.json').read_text()[:3000])
    elif mode == 'stop':
        case, reason = sys.argv[2], sys.argv[3]
        command = ['docker', 'stop', '--timeout', '30', 'aisaac-bench-memory-v3-' + case]
        rc, out, err = run(c, shlex.join(command), timeout=90)
        record = dict(command=command, stdout=out, stderr=err, exit_status=rc, reason=reason, time=time.ctime())
        (RESULTS / f'{case}-stop.json').write_text(json.dumps(record, indent=2)); print(json.dumps(record))
    elif mode == 'sha':   # sha <path under cpu-wide> ...: sha256 of host files (read-only)
        rc, out, err = run(c, 'cd ' + ROOT + ' && sha256sum ' + ' '.join(shlex.quote(p) for p in sys.argv[2:]))
        print(out + err); print('EXIT', rc)
    elif mode == 'fetchpy':   # fetchpy <dir under cpu-wide> <local.zip>: its *.py files (read-only on the host)
        src, local = sys.argv[2], sys.argv[3]
        listing = run(c, f'cd {ROOT}/{shlex.quote(src)} && find . -name "*.py" -not -path "*/__pycache__/*"')[1].split()
        with zipfile.ZipFile(local, 'w', zipfile.ZIP_DEFLATED) as z:
            for rel in listing:
                with s.open(f'{ROOT}/{src}/{rel[2:]}') as f: z.writestr(rel[2:], f.read())
        print('FETCHED', len(listing), 'files'); print('EXIT 0')
    elif mode == 'deploydir':   # deploydir <tag> <local dir>: <dir> (a python tree) -> cpu-wide/memory-v3-code-<tag>/python
        tag, local = sys.argv[2], Path(sys.argv[3])
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(local.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts: z.write(p, 'python/' + p.relative_to(local).as_posix())
        zpath = RESULTS / f'code-{tag}.zip'; zpath.write_bytes(buf.getvalue())
        s.put(str(zpath), f'{ROOT}/memory-v3-code-{tag}.zip')
        rc, out, err = run(c, f'test ! -e {ROOT}/memory-v3-code-{tag} && python3 -m zipfile -e {ROOT}/memory-v3-code-{tag}.zip {ROOT}/memory-v3-code-{tag}'
                              f' && ls {ROOT}/memory-v3-code-{tag}/python | wc -l')
        print(out + err); print('DEPLOY', tag, 'EXIT', rc); sys.exit(rc)
    elif mode == 'overlay':   # overlay <tag> <dir under cpu-wide> <local dir> <rel files...>: a copy of the host dir
        #                         as cpu-wide/memory-v3-code-<tag>/python with the listed files replaced from <local dir>
        tag, src, local, files = sys.argv[2], sys.argv[3], Path(sys.argv[4]), sys.argv[5:]
        dst = f'{ROOT}/memory-v3-code-{tag}'
        rc, out, err = run(c, f'test ! -e {dst} && mkdir {dst} && cp -a {ROOT}/{shlex.quote(src)} {dst}/python')
        assert rc == 0, (out, err)
        for rel in files:
            run(c, f'mkdir -p {dst}/python/{shlex.quote(str(Path(rel).parent.as_posix()))}')
            s.put(str(local / rel), f'{dst}/python/{rel}')
        rc, out, err = run(c, f'cd {dst}/python && sha256sum ' + ' '.join(shlex.quote(f) for f in files))
        print(out + err); print('OVERLAY', tag, 'EXIT', rc); sys.exit(rc)
    elif mode == 'io':   # io <container> <seconds>: block IO of another workload (read-only)
        name, seconds = sys.argv[2], int(sys.argv[3])
        rc, cid, err = run(c, 'docker inspect -f {{.Id}} ' + shlex.quote(name))
        stat = f'/sys/fs/cgroup/docker/{cid.strip()}/io.stat'
        rc, out, err = run(c, f'cat {stat}; sleep {seconds}; echo ---; cat {stat}', timeout=seconds + 60)
        def total(text, key):
            return sum(int(f.split('=')[1]) for line in text.strip().splitlines() for f in line.split() if f.startswith(key + '='))
        before, after = out.split('---')
        record = dict(container=name, seconds=seconds, time=time.ctime(),
                      read_mb_s=(total(after, 'rbytes') - total(before, 'rbytes')) / 2**20 / seconds,
                      write_mb_s=(total(after, 'wbytes') - total(before, 'wbytes')) / 2**20 / seconds)
        print(json.dumps(record)); print('EXIT', rc)
    elif mode == 'gate':
        name, tag, indices, shell = sys.argv[2], sys.argv[3], [int(i) for i in sys.argv[4].split(',')], sys.argv[5]
        rc, out, err = run(c, 'nvidia-smi --query-gpu=index,uuid --format=csv,noheader; echo APPS; nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader')
        uuids = dict(line.split(', ') for line in out.split('APPS')[0].strip().splitlines())
        busy = set(out.split('APPS')[1].split())
        selected = [uuids[str(i)].strip() for i in indices]
        assert not busy & set(selected), ('selected GPU busy', busy & set(selected))
        cmd = json.loads((V2 / 'pipeline-cuda-command.json').read_text())
        cmd = [x.replace('memory-v2-code-pipeline/python', f'memory-v3-code-{tag}/python') for x in cmd]
        cmd[cmd.index('--name') + 1] = 'aisaac-memory-v3-gate-' + name
        cmd = [x.split('=')[0] + '=' + ','.join(selected) if x.startswith(('CUDA_VISIBLE_DEVICES=', 'NVIDIA_VISIBLE_DEVICES=')) else x for x in cmd]
        cmd[-1] = 'cd /bench/workspace/python && ' + shell
        (RESULTS / f'gate-{name}-command.json').write_text(json.dumps(cmd, indent=2))
        log = f'{ROOT}/v3/gate-{name}.log'
        run(c, f'mkdir -p {ROOT}/v3')
        rc, out, err = run(c, shlex.join(cmd) + f' > {log} 2>&1; echo $? > {log}.exit', timeout=3600)
        s.get(log, str(RESULTS / f'gate-{name}.log')); s.get(log + '.exit', str(RESULTS / f'gate-{name}.exit'))
        text = (RESULTS / f'gate-{name}.log').read_text(errors='replace')
        print(text[-4000:]); code = int((RESULTS / f'gate-{name}.exit').read_text().strip())
        print('GATE', name, 'EXIT', code); sys.exit(code)
    s.close(); c.close()


if __name__ == '__main__':
    main()
