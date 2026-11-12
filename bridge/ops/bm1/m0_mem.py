"""bm1 M0: memory of a running train_tok and its container, sampled from the host's /proc and the container's cgroup.

  python3 m0_mem.py <out.csv> [period s=5] [match] [threads.csv]      (runs until the matched process is gone)

Every period, for the newest process whose command line contains `match` (default the shared venv's
'python -u train_tok.py', the trainer itself) and all its descendants: RSS and PSS summed by kind
  trainer   the trainer process
  game      isaac.x64 instances (the AB+ engines the workers drive)
  workers   other python descendants (worker / sampler processes)
  other     everything else under the trainer (xvfb, shells)
with process counts and the cores each kind used since the previous sample (utime + stime deltas of the processes
alive at both samples),
and the trainer's cgroup (the container): memory.current, memory.peak, anon, file, shmem from
memory.stat, and the host's MemAvailable (every tenant's). One CSV row per sample. threads.csv: the trainer's threads'
ticks per sample and their schedstat run / wait ns; per kind also the system part of its cores, the cores of the
children it reaped (cutime + cstime: short-lived processes counted on their parent), its common nice value, and the
container's user / system ticks; <tag>.sysc.csv: the syscall every task sits in at each sample, per kind (the
trainer's threads by name) (b57_threads.py's format plus two columns; replaces it here: b57 matched its own command line when given a match). Standard library only (host python3); pid-free names (the trainer is found by
its command line), so K trainers on one host each get their own sampler by `match` / out path."""
import os
import re
import sys
import time

KINDS = ('trainer', 'game', 'workers', 'other')


def kind_of(name, top):
    """A process's kind by its /proc name: the trainer (the top process), the game ('isaac*'), the sampler workers
    ('python*', and with ISAAC_RL_WORKER_CPU=1 the names tok_floor.name_thread gives a worker and its forked children:
    'w<index>-main', 'w<index>-cfchild', 'w<index>-schild'; before 2026-11-12 these fell into 'other'), the rest."""
    if top:
        return 'trainer'
    if name.startswith('isaac'):
        return 'game'
    if 'python' in name or re.match(r'w\d+-', name):
        return 'workers'
    return 'other'


def cmdline(p):
    try:
        with open(f'/proc/{p}/cmdline', 'rb') as f:
            return f.read().replace(b'\0', b' ').decode('utf-8', 'replace')
    except OSError:
        return ''


def status(p):
    out = {}
    try:
        with open(f'/proc/{p}/status') as f:
            for line in f:
                k, _, v = line.partition(':')
                out[k] = v.strip()
    except OSError:
        pass
    return out


def pss_kb(p):
    """Proportional set size (shared pages split among their users: sums add up, unlike RSS)."""
    try:
        with open(f'/proc/{p}/smaps_rollup') as f:
            for line in f:
                if line.startswith('Pss:'):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def stat4(p):
    """(utime, stime, cutime + cstime, nice) of /proc/<p>/stat (ticks; cutime / cstime: the children it has reaped, so
    short-lived processes that died between two samples still count, on their parent); None when gone."""
    try:
        with open(f'/proc/{p}/stat') as f:
            s = f.read()
        rest = s[s.rindex(')')+2:].split()
        return int(rest[11]), int(rest[12]), int(rest[13])+int(rest[14]), int(rest[16])
    except (OSError, ValueError, IndexError):
        return None


def syscall_of(path):
    """The syscall a task is in now (/proc/.../syscall's first field: a number, 'running', or '' when unreadable)."""
    try:
        with open(path+'/syscall') as f:
            return f.read().split(' ', 1)[0].strip()
    except OSError:
        return ''


def ticks(p):
    try:
        with open(f'/proc/{p}/stat') as f:
            s = f.read()
        rest = s[s.rindex(')')+2:].split()
        return int(rest[11])+int(rest[12])
    except (OSError, ValueError, IndexError):
        return 0


def find(match):
    best = None
    for p in os.listdir('/proc'):
        c = cmdline(p) if p.isdigit() else ''
        if match in c and c.split(' ', 1)[0].endswith('python'):   # (the interpreter, not timeout / xvfb-run)
            if best is None or int(p) > best:
                best = int(p)
    return best


def children():
    kids = {}
    for p in os.listdir('/proc'):
        if p.isdigit():
            s = status(p)
            if 'PPid' in s:
                kids.setdefault(int(s['PPid']), []).append(int(p))
    return kids


def cgroup_dir(pid):
    """The container's cgroup: ('v2', dir) or ('v1', {controller: dir}) (cgroup v1 hosts: memory, cpuacct)."""
    try:
        with open(f'/proc/{pid}/cgroup') as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    v1 = {}
    for line in lines:
        hid, ctl, path = line.split(':', 2)
        if hid == '0' and ctl == '':
            if os.path.exists('/sys/fs/cgroup/cgroup.controllers'):
                return ('v2', '/sys/fs/cgroup'+path)
        for c in ctl.split(','):
            if c in ('memory', 'cpuacct'):
                base = '/sys/fs/cgroup/'+ctl
                v1[c] = (base if os.path.isdir(base) else '/sys/fs/cgroup/'+c)+path
    return ('v1', v1) if v1 else None


def cg_sample(cg):
    """(current bytes, peak bytes, anon, file, shmem, cpu usec) of the cgroup, '' where unknown; v1 equivalents:
    memory.usage_in_bytes, memory.max_usage_in_bytes, memory.stat total_rss / total_cache / total_shmem,
    cpuacct.usage (ns)."""
    if cg is None:
        return ('',)*6
    kind, d = cg
    if kind == 'v2':
        st = dict(x.split() for x in read(d+'/memory.stat').splitlines() if x.count(' ') == 1)
        cpu = dict(x.split() for x in read(d+'/cpu.stat').splitlines() if x.count(' ') == 1).get('usage_usec', '')
        return (read(d+'/memory.current').strip(), read(d+'/memory.peak').strip(), st.get('anon', ''),
                st.get('file', ''), st.get('shmem', ''), cpu)
    m, c = d.get('memory', ''), d.get('cpuacct', '')
    st = dict(x.split() for x in read(m+'/memory.stat').splitlines() if x.count(' ') == 1) if m else {}
    ns = read(c+'/cpuacct.usage').strip() if c else ''
    return (read(m+'/memory.usage_in_bytes').strip() if m else '', read(m+'/memory.max_usage_in_bytes').strip() if m else '',
            st.get('total_rss', ''), st.get('total_cache', ''), st.get('total_shmem', ''),
            str(int(ns)//1000) if ns.isdigit() else '')


def cg_ticks(cg):
    """(user, system) CPU of the container's cgroup: v1 cpuacct.stat (USER_HZ ticks), v2 cpu.stat (usec / 1e4: the same
    10 ms units); '' where unknown."""
    if cg is None:
        return ('', '')
    kind, d = cg
    if kind == 'v2':
        st = dict(x.split() for x in read(d+'/cpu.stat').splitlines() if x.count(' ') == 1)
        return tuple(str(int(st[k])//10000) if k in st else '' for k in ('user_usec', 'system_usec'))
    c = d.get('cpuacct', '')
    st = dict(x.split() for x in read(c+'/cpuacct.stat').splitlines() if x.count(' ') == 1) if c else {}
    return (st.get('user', ''), st.get('system', ''))


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ''


def main():
    out = sys.argv[1]
    period = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    match = sys.argv[3] if len(sys.argv) > 3 else '/.venv/bin/python -u train_tok.py'
    tout = sys.argv[4] if len(sys.argv) > 4 else None   # the trainer's threads: t,kind,name,ticks (b57 format)
    # (the ledger, BM1 2026-11-11: <out minus .mem.csv>.sysc.csv: per sample and process kind, how many tasks sit in
    #  which syscall; the trainer's threads by name)
    sc = open(out[:-len('.mem.csv')]+'.sysc.csv' if out.endswith('.mem.csv') else out+'.sysc.csv', 'w')
    sc.write('t,kind,syscall,tasks\n')
    pid = None
    while pid is None:
        pid = find(match)
        if pid is None:
            time.sleep(1)
    cg = cgroup_dir(pid)
    prev, t_prev = {}, None
    prev4 = {}
    prev_sch = {}
    tf = open(tout, 'w') if tout else None
    if tf:
        tf.write('t,kind,name,ticks,run_ns,wait_ns\n')
    with open(out, 'w') as f:
        f.write('t,'+','.join(f'{k}_rss_mb,{k}_pss_mb,{k}_n,{k}_cores' for k in KINDS)
                + ',cg_current_mb,cg_peak_mb,cg_anon_mb,cg_file_mb,cg_shmem_mb,cg_cpu_usec,host_avail_mb,'
                + ','.join(f'{k}_sys_cores,{k}_child_cores,{k}_nice' for k in KINDS) + ',cg_user_ticks,cg_sys_ticks,'
                + ','.join(f'{k}_sched_run,{k}_sched_wait' for k in KINDS) + '\n')
        while os.path.exists(f'/proc/{pid}'):
            now = time.time()
            kids = children()
            rss = dict.fromkeys(KINDS, 0)
            n = dict.fromkeys(KINDS, 0)
            dtk = dict.fromkeys(KINDS, 0)
            pss = dict.fromkeys(KINDS, 0)
            cur = {}
            cur4 = {}
            dsys = dict.fromkeys(KINDS, 0)
            dchild = dict.fromkeys(KINDS, 0)
            nices = {k: {} for k in KINDS}
            calls = {}
            sch = {}   # (per kind: schedstat run / wait ns summed over every task of its processes)
            schk = dict.fromkeys(KINDS, (0, 0))
            todo = [(pid, True)]
            while todo:
                p, top = todo.pop()
                s = status(p)
                kb = int(s.get('VmRSS', '0 kB').split()[0]) if s.get('VmRSS') else 0
                name = s.get('Name', '')
                kind = kind_of(name, top)
                rss[kind] += kb
                cur[p] = ticks(p)
                if p in prev:   # (cores: per-process tick deltas of the processes alive at both samples)
                    dtk[kind] += max(0, cur[p]-prev[p])
                pss[kind] += pss_kb(p)
                n[kind] += 1
                st4 = stat4(p)
                if st4 is not None:
                    cur4[p] = st4
                    nices[kind][st4[3]] = nices[kind].get(st4[3], 0)+1
                    if p in prev4:
                        dsys[kind] += max(0, st4[1]-prev4[p][1])
                        dchild[kind] += max(0, st4[2]-prev4[p][2])
                try:
                    for tid in os.listdir(f'/proc/{p}/task'):
                        sy = syscall_of(f'/proc/{p}/task/{tid}')
                        ss_ = read(f'/proc/{p}/task/{tid}/schedstat').split()
                        if len(ss_) > 1:
                            sch[(p, tid)] = (kind, int(ss_[0]), int(ss_[1]))
                        key_ = kind if not top else 'trainer:'+(read(f'/proc/{p}/task/{tid}/comm').strip()
                                                                .rstrip('0123456789_') or '?')
                        calls[(key_, sy)] = calls.get((key_, sy), 0)+1
                except OSError:
                    pass
                todo += [(c, False) for c in kids.get(p, [])]
            dt = now-t_prev if t_prev else 0.0
            prev, t_prev = cur, now
            prev4 = cur4
            for key_, (k_, r_, w_) in sch.items():
                if key_ in prev_sch:
                    a_, b_ = schk[k_]
                    schk[k_] = (a_+max(0, r_-prev_sch[key_][1]), b_+max(0, w_-prev_sch[key_][2]))
            prev_sch = sch
            for (k_, sy_), c_ in sorted(calls.items()):
                sc.write(f'{now:.1f},{k_},{sy_},{c_}\n')
            sc.flush()
            if tf:
                try:
                    for tid in os.listdir(f'/proc/{pid}/task'):
                        try:
                            nm = read(f'/proc/{pid}/task/{tid}/comm').strip()
                            nm = 'main' if tid == str(pid) else nm   # (the process's first thread: the actor's)
                            # (schedstat: ns on a CPU, ns runnable but waiting for one: the scheduler's view)
                            ss = read(f'/proc/{pid}/task/{tid}/schedstat').split()
                            tf.write(f'{now:.3f},thread,{nm}:{tid},{ticks(f"{pid}/task/{tid}")},'
                                     f'{ss[0] if ss else ""},{ss[1] if len(ss) > 1 else ""}\n')
                        except OSError:
                            pass
                    tf.flush()
                except OSError:
                    pass
            cur_, peak_, anon_, file_, shm_, cpu_ = cg_sample(cg)

            def mb(x):
                try:
                    return f'{int(x)/2**20:.0f}'
                except ValueError:
                    return ''
            f.write(f'{now:.1f},'+','.join(f'{rss[k]/1024:.0f},{pss[k]/1024:.0f},{n[k]},'
                                           + (f'{dtk[k]/100/dt:.2f}' if dt else '') for k in KINDS)+','
                    + ','.join(mb(v) for v in (cur_, peak_, anon_, file_, shm_))
                    + ',' + cpu_
                    + ',' + next((f'{int(x.split()[1])/1024:.0f}' for x in read('/proc/meminfo').splitlines()
                                  if x.startswith('MemAvailable:')), '') + ','
                    + ','.join((f'{dsys[k]/100/dt:.2f},{dchild[k]/100/dt:.2f}' if dt else ',')
                               + ',' + (str(max(nices[k], key=nices[k].get)) if nices[k] else '') for k in KINDS)
                    + ',' + ','.join(cg_ticks(cg)) + ','
                    + ','.join((f'{schk[k][0]/1e9/dt:.3f},{schk[k][1]/1e9/dt:.3f}' if dt else ',') for k in KINDS)
                    + '\n')
            f.flush()
            time.sleep(period)

if __name__ == '__main__':
    main()
