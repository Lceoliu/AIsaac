"""bm1 M0: memory of a running train_tok and its container, sampled from the host's /proc and the container's cgroup.

  python3 m0_mem.py <out.csv> [period s=5] [match]      (runs until the matched process is gone)

Every period, for the newest process whose command line contains `match` (default the shared venv's
'python -u train_tok.py', the trainer itself) and all its descendants: RSS and PSS summed by kind
  trainer   the trainer process
  game      isaac.x64 instances (the AB+ engines the workers drive)
  workers   other python descendants (worker / sampler processes)
  other     everything else under the trainer (xvfb, shells)
with process counts and CPU ticks (utime + stime, 1/100 s; deltas between rows = cores used by that kind),
and the trainer's cgroup (the container): memory.current, memory.peak, anon, file, shmem from
memory.stat, and the host's MemAvailable (every tenant's). One CSV row per sample. Standard library only (host python3); pid-free names (the trainer is found by
its command line), so K trainers on one host each get their own sampler by `match` / out path."""
import os
import sys
import time

KINDS = ('trainer', 'game', 'workers', 'other')


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
    try:
        with open(f'/proc/{pid}/cgroup') as f:
            for line in f:
                if line.startswith('0::'):
                    return '/sys/fs/cgroup'+line.strip()[3:]
    except OSError:
        pass
    return None


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
    pid = None
    while pid is None:
        pid = find(match)
        if pid is None:
            time.sleep(1)
    cg = cgroup_dir(pid)
    with open(out, 'w') as f:
        f.write('t,'+','.join(f'{k}_rss_mb,{k}_pss_mb,{k}_n,{k}_ticks' for k in KINDS)
                + ',cg_current_mb,cg_peak_mb,cg_anon_mb,cg_file_mb,cg_shmem_mb,cg_cpu_usec,host_avail_mb\n')
        while os.path.exists(f'/proc/{pid}'):
            kids = children()
            rss = dict.fromkeys(KINDS, 0)
            n = dict.fromkeys(KINDS, 0)
            tk = dict.fromkeys(KINDS, 0)
            pss = dict.fromkeys(KINDS, 0)
            todo = [(pid, True)]
            while todo:
                p, top = todo.pop()
                s = status(p)
                kb = int(s.get('VmRSS', '0 kB').split()[0]) if s.get('VmRSS') else 0
                name = s.get('Name', '')
                kind = 'trainer' if top else 'game' if name.startswith('isaac') else \
                    'workers' if 'python' in name else 'other'
                rss[kind] += kb
                tk[kind] += ticks(p)
                pss[kind] += pss_kb(p)
                n[kind] += 1
                todo += [(c, False) for c in kids.get(p, [])]
            st = dict(line.split() for line in read(cg+'/memory.stat').splitlines() if line.count(' ') == 1) if cg else {}

            def mb(x):
                try:
                    return f'{int(x)/2**20:.0f}'
                except ValueError:
                    return ''
            f.write(f'{time.time():.1f},'+','.join(f'{rss[k]/1024:.0f},{pss[k]/1024:.0f},{n[k]},{tk[k]}' for k in KINDS)+','
                    + ','.join(mb(v) for v in (read(cg+'/memory.current').strip() if cg else '',
                                               read(cg+'/memory.peak').strip() if cg else '',
                                               st.get('anon', ''), st.get('file', ''), st.get('shmem', '')))
                    + ',' + (dict(x.split() for x in read(cg+'/cpu.stat').splitlines() if x.count(' ') == 1)
                             .get('usage_usec', '') if cg else '')
                    + ',' + next((f'{int(x.split()[1])/1024:.0f}' for x in read('/proc/meminfo').splitlines()
                                  if x.startswith('MemAvailable:')), '') + '\n')
            f.flush()
            time.sleep(period)


if __name__ == '__main__':
    main()
