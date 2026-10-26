"""memory-chain-v3 guard: temp/memory-chain-v2/guard_memory_pipeline_full.py with the v3 runner.

usage: python3 guard_v3.py <spec.json>; stops only its own container (memory, CPU, progress rules unchanged).
"""
import sys, os, json, pathlib, subprocess, time
root = pathlib.Path('/home/runaiwusheng/aisaac-bench-20261019/cpu-wide')
spec_path = sys.argv[1]
spec = json.loads(pathlib.Path(spec_path).read_text())
case = spec['case']
name = 'aisaac-bench-' + case
runner = ['python3', str(root/'v3'/'run_v3_case.py'), spec_path]
(root/'logs'/f'{case}.dispatch.json').write_text(json.dumps({'command':runner,'case':case,'guard':{'available_gib_min':24,'memory_early_stop_gib':26,'memory_poll_seconds':0.25,'external_cpu_cores_max_sustained':9,'sustained_samples':3,'memory_pressure_rule':'external cgroup full avg10 >10% for three samples (v3; v2 used 1%); host pswpin >1000 pages/s for three samples; external swap growth since start >1 GiB; external memory.events oom/oom_kill increase stops at once; own host PSI logged separately','interval_seconds':20,'no_first_progress_seconds':900,'no_progress_seconds':180}},indent=2))
start = time.time()
previous = None
memory_cg = None
busy_samples = 0
pressure_samples = 0
external_cgroups = json.loads((root/"external_cgroups_r3.json").read_text())
def others(own=None):
    # v2's recorded list plus every docker cgroup present now, never our own.
    found = ['/docker/'+d.name for d in pathlib.Path('/sys/fs/cgroup/docker').iterdir() if d.is_dir()]
    return [p for p in dict.fromkeys(external_cgroups + found) if own is None or p.lstrip('/') != own]
def external_harm(own=None):
    # Real harm to other workloads: their swap and their OOM counters.
    out = {}
    for p in others(own):
        d = pathlib.Path('/sys/fs/cgroup')/p.lstrip('/')
        if not d.exists(): continue
        events = {k:int(v) for k,v in (line.split() for line in (d/'memory.events').read_text().splitlines())}
        swap = int((d/'memory.swap.current').read_text()) if (d/'memory.swap.current').exists() else 0
        out[p] = dict(swap=swap, oom=events.get('oom',0), oom_kill=events.get('oom_kill',0))
    return out
harm_previous = external_harm()
harm_start = harm_previous
def pswpin():
    return int(next(x for x in pathlib.Path('/proc/vmstat').read_text().splitlines() if x.startswith('pswpin ')).split()[1])
swapin_previous = (time.time(), pswpin())
swapin_samples = 0
def roles(cid):
    # RSS by role (RSS double-counts shared pages). trainer = the python
    # running train_tok.py (not its nice/timeout wrappers); learner = the
    # trainer's spawned child with the most >=500 MB spawned children (the
    # source workers); actor = other >=500 MB spawned processes; worker =
    # smaller spawned processes (game bridges); game = isaac.x64; compile =
    # torch inductor workers; other = MPS, Xvfb, shells.
    rows = [l.split(None, 3) for l in subprocess.check_output(['docker', 'top', cid, '-eo', 'pid,ppid,rss,args'], text=True).splitlines()[1:]]
    rows = [(int(r[0]), int(r[1]), int(r[2]), r[3]) for r in rows if len(r) == 4]
    trainer = {r[0] for r in rows if 'train_tok.py' in r[3] and r[3].split()[0].endswith('python')}
    spawned = {r[0]: r for r in rows if 'spawn_main' in r[3]}
    big = lambda pid: [r for r in spawned.values() if r[1] == pid and r[2] >= 500 * 1024]
    first = [r for r in spawned.values() if r[1] in trainer]
    learner = max(first, key=lambda r: len(big(r[0])))[0] if first else None
    names = ('trainer', 'learner', 'source', 'actor', 'worker', 'game', 'compile', 'other')
    out = {k: 0 for k in names}; out['count'] = {k: 0 for k in names}
    for pid, ppid, rss, args in rows:
        role = ('game' if 'isaac.x64' in args else 'compile' if 'torch/_inductor' in args else 'trainer' if pid in trainer
                else 'learner' if pid == learner else 'source' if ppid == learner and pid in spawned
                else 'actor' if pid in spawned and rss >= 500 * 1024 else 'worker' if pid in spawned else 'other')
        out[role] += rss * 1024; out['count'][role] += 1
    return out
memory_rows = 0
audit_thread = [None]
memory_log = (root/'logs'/f'{case}.memory.jsonl').open('w')
def memory_snapshot(cg, progress):
    global memory_rows
    if not progress.exists(): return
    rows = len(progress.read_text().splitlines()) - 1
    if rows <= memory_rows: return
    memory_rows = rows
    stats = dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
    snap = dict(time=time.time(), progress_rows=rows, memory_gib=int((cg/'memory.current').read_text())/1024**3,
                stat={k: int(v) for k, v in stats.items()}, rss=roles(name))
    memlog = (spec.get('env') or {}).get('ISAAC_RL_MEM_LOG')
    if memlog and not (audit_thread[0] and audit_thread[0].is_alive()):
        # smaps_rollup of every process (host/audit_inside.py), off the guard's polling thread
        def audit(rows=rows):
            try:
                p = subprocess.run(['docker', 'exec', name, '/bench/home/isaac-abplus/.venv/bin/python', '/bench/v3/audit_inside.py', memlog],
                                   capture_output=True, text=True, timeout=120)
                record = json.loads(p.stdout) if p.returncode == 0 else dict(error=p.stderr[-2000:])
            except Exception as error:
                record = dict(error=repr(error))
            print(json.dumps(dict(time=time.time(), progress_rows=rows, audit=record)), file=memory_log, flush=True)
        import threading
        audit_thread[0] = threading.Thread(target=audit, daemon=True); audit_thread[0].start()
    print(json.dumps(snap), file=memory_log, flush=True)
proc = subprocess.Popen(runner)
print('DISPATCH', json.dumps(runner), 'pid', proc.pid, flush=True)
try:
    with (root/'logs'/f'{case}.guard.jsonl').open('w') as log:
        while proc.poll() is None:
            memory_stop=False
            for tick in range(80):
                time.sleep(.25)
                if proc.poll() is not None:break
                if tick % 8 == 0 and memory_cg is not None:
                    try: memory_snapshot(memory_cg, root/'runs'/case/'progress.csv')
                    except Exception as error: print('MEMORY_SNAPSHOT_ERROR', repr(error), flush=True)
                available=int(next(x for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])/1024**2
                if available<26:
                    log.write(json.dumps(dict(time=time.time(),mem_available_gib=available,stop_reason='host_memory_fast_guard'))+'\n');log.flush()
                    print('STOP_OWN_ONLY',name,'host_memory_fast_guard',available,flush=True)
                    subprocess.run(['docker','kill',name],check=True)
                    memory_stop=True
                    break
            if memory_stop or proc.poll() is not None:
                break
            inspect = subprocess.run(['docker','inspect',name], text=True, capture_output=True)
            if inspect.returncode:
                # --rm may remove the finished container before the runner's
                # ten-second telemetry poll publishes its exit/result.
                code = proc.wait(timeout=15)
                # (a failed run whose container is gone still ends with RUNNER_EXIT: the orchestrator waits for it)
                print('CONTAINER_REMOVED_AFTER_SUCCESS' if code == 0 else f'CONTAINER_GONE runner exit {code}: {inspect.stderr.strip()}', name, flush=True)
                break
            current = json.loads(inspect.stdout)[0]
            if not current['State']['Running']:
                break
            pid = current['State']['Pid']
            cgline = next(x for x in pathlib.Path(f'/proc/{pid}/cgroup').read_text().splitlines() if x.startswith('0::'))
            cg = pathlib.Path('/sys/fs/cgroup') / cgline.split('::',1)[1].lstrip('/')
            memory_cg = cg
            cpu = {x.split()[0]:int(x.split()[1]) for x in (cg/'cpu.stat').read_text().splitlines()}
            host = [int(x) for x in pathlib.Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
            host_busy = sum(host)-host[3]-host[4]
            now = time.time()
            available = int(next(x for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])/1024**2
            pressure = pathlib.Path('/proc/pressure/memory').read_text().strip()
            full_avg10 = float(pressure.splitlines()[1].split()[1].split('=')[1])
            external = None
            if previous is not None:
                dt = now-previous[0]
                external = max(0,(host_busy-previous[1])/os.sysconf('SC_CLK_TCK')/dt-(cpu['usage_usec']-previous[2])/1e6/dt)
                busy_samples = busy_samples+1 if external>9 else 0
            own_rel = str(cg.relative_to('/sys/fs/cgroup'))
            external_pressure = {p: (pathlib.Path('/sys/fs/cgroup')/p.lstrip('/')/'memory.pressure').read_text().strip() for p in others(own_rel) if (pathlib.Path('/sys/fs/cgroup')/p.lstrip('/')/'memory.pressure').exists()}
            external_full_max = max(float(v.splitlines()[1].split()[1].split('=')[1]) for v in external_pressure.values())
            pressure_samples = pressure_samples+1 if external_full_max>10 else 0
            harm_now = external_harm(own_rel)
            harm = {p: dict(now=v, before=harm_previous.get(p)) for p, v in harm_now.items()
                    if p in harm_previous and (v['oom'] > harm_previous[p]['oom'] or v['oom_kill'] > harm_previous[p]['oom_kill'])}
            swap_growth = sum(max(0, v['swap'] - harm_start[p]['swap']) for p, v in harm_now.items() if p in harm_start)
            harm_previous = harm_now
            swapin_now = (time.time(), pswpin())
            swapin_rate = (swapin_now[1] - swapin_previous[1]) / max(1e-6, swapin_now[0] - swapin_previous[0])
            swapin_previous = swapin_now
            swapin_samples = swapin_samples + 1 if swapin_rate > 1000 else 0
            previous = (now,host_busy,cpu['usage_usec'])
            progress = root/'runs'/case/'progress.csv'
            reason = None
            own_events={k:int(v) for k,v in (line.split() for line in (cg/'memory.events').read_text().splitlines())}
            if own_events['oom']:
                reason='benchmark_cgroup_oom'
            elif harm:
                reason='external_oom_increase'
            elif swapin_samples >= 3:
                reason='host_swapin_over_1000_pages_per_second_sustained'
            elif swap_growth > 2**30:
                reason='external_swap_growth_over_1_gib'
            elif available<24 or pressure_samples>=3:
                reason='host_memory_pressure'
            elif busy_samples>=3:
                reason='other_workload_cpu_exceeds_9_cores_sustained'
            elif not progress.exists() and now-start>900:
                reason='no_first_update_in_900_seconds'
            elif progress.exists() and now-progress.stat().st_mtime>180:
                reason='no_progress_in_180_seconds'
            snap={'time':now,'external_cpu_cores':external,'mem_available_gib':available,'memory_pressure':pressure,'own_memory_pressure':(cg/'memory.pressure').read_text().strip(),'own_memory_events':(cg/'memory.events').read_text().strip(),'own_pids_events':(cg/'pids.events').read_text().strip(),'external_memory_pressure':external_pressure,'external_full_max':external_full_max,'external_harm':harm,'external_swap_total':sum(v['swap'] for v in harm_now.values()),'external_swap_growth':swap_growth,'host_pswpin_rate':swapin_rate,'pids':int((cg/'pids.current').read_text()),'memory_gib':int((cg/'memory.current').read_text())/1024**3,'stop_reason':reason}
            snap['top_rss']=subprocess.check_output(['docker','top',name,'-eo','pid,ppid,rss,comm','--sort=-rss'],text=True).splitlines()[:13]
            stats=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
            snap['memory_parts']={k:int(stats[k]) for k in ('anon','file','kernel','shmem')}
            log.write(json.dumps(snap)+'\n');log.flush()
            if reason:
                print('STOP_OWN_ONLY',name,reason,flush=True)
                subprocess.run(['docker','stop','--timeout','20',name],check=True)
                break
    code=proc.wait(timeout=60)
    print('RUNNER_EXIT',code,flush=True)
finally:
    if proc.poll() is None:
        subprocess.run(['docker','stop','--timeout','20',name],check=True)
        proc.wait(timeout=60)
sys.exit(code)
