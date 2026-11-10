"""B57: per-thread CPU of a running train_tok (and the machine's cores), sampled from /proc, for the cycle budget.

usage: python3 b57_threads.py <out.csv> [period s] [match]   (runs until the matched process is gone)

Every period: for the newest process whose command line contains `match` (default 'train_tok.py --groups-file', the
trainer, not its workers), each thread's name (comm) and utime + stime ticks; the workers' processes (command line
contains 'floor_worker' or a child of the trainer) summed; /proc/stat's per-core busy ticks. Writes one row per sample
and thread: t, kind (thread / workers / core), name, ticks. b57_nsys.py turns the deltas into CPU shares per window.
Standard library only."""
import os
import sys
import time


def procs():
    out = []
    for p in os.listdir('/proc'):
        if p.isdigit():
            try:
                with open(f'/proc/{p}/cmdline', 'rb') as f:
                    out.append((int(p), f.read().replace(b'\0', b' ').decode('utf-8', 'replace')))
            except OSError:
                pass
    return out


def stat_ticks(path):
    with open(path) as f:
        s = f.read()
    rest = s[s.rindex(')')+2:].split()
    return int(rest[11])+int(rest[12]), int(rest[1])   # utime + stime, ppid


def main():
    out, period = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    match = sys.argv[3] if len(sys.argv) > 3 else 'train_tok.py --groups-file'
    pid = None
    t0 = time.time()
    while pid is None and time.time()-t0 < 600:
        cands = [p for p, c in procs() if match in c and 'nsys' not in c.split()[0]]
        pid = max(cands) if cands else None
        if pid is None:
            time.sleep(1)
    if pid is None:
        print('no trainer process'); return
    with open(out, 'w') as f:
        f.write('t,kind,name,ticks\n')
        while os.path.exists(f'/proc/{pid}'):
            t = time.time()
            rows = []
            try:
                for tid in os.listdir(f'/proc/{pid}/task'):
                    try:
                        with open(f'/proc/{pid}/task/{tid}/comm') as g:
                            name = g.read().strip()
                        ticks, _ = stat_ticks(f'/proc/{pid}/task/{tid}/stat')
                        rows.append(('thread', f'{name}:{tid}', ticks))
                    except OSError:
                        pass
            except OSError:
                break
            wt = 0
            for p, c in procs():
                if p == pid:
                    continue
                try:
                    ticks, ppid = stat_ticks(f'/proc/{p}/stat')
                except OSError:
                    continue
                if ppid == pid or 'isaac' in c.lower():
                    wt += ticks
            rows.append(('workers', 'all', wt))
            with open('/proc/stat') as g:
                for line in g:
                    if line.startswith('cpu') and line[3].isdigit():
                        v = [int(x) for x in line.split()[1:]]
                        rows.append(('core', line.split()[0], sum(v)-v[3]-v[4]))   # busy = all - idle - iowait
            for kind, name, ticks in rows:
                f.write(f'{t:.3f},{kind},{name},{ticks}\n')
            f.flush()
            time.sleep(max(0.0, period-(time.time()-t)))


if __name__ == '__main__':
    main()
