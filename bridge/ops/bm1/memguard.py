"""bm1 direct mode: the memory limit of one run (run.sh), standard library only.

  python3 memguard.py <pid> <limit, e.g. 64g / 50g / 48000m> <bm1.log> <purpose>

First choice: a cgroup v2 child of this process's cgroup with memory.max = limit and the run's pid moved into it (its
later children follow); the kernel then enforces the limit (an OOM kill inside it shows in memory.events, read and
logged when the run ends). Where the cgroup cannot be written (a container without the rights): every 2 s the RSS of
the run's whole process tree (the pid's session) is summed; over the limit, a line goes to bm1.log and the tree gets
SIGTERM, then SIGKILL after 10 s. Never silent: every action is logged."""
import os
import signal
import sys
import time


def parse(s):
    s = s.strip().lower()
    mult = {'k': 2**10, 'm': 2**20, 'g': 2**30, 't': 2**40}
    return int(float(s[:-1])*mult[s[-1]]) if s[-1] in mult else int(s)


def log(path, text):
    with open(path, 'a') as f:
        f.write(time.strftime('%F %T ')+text+'\n')


def tree(root):
    kids = {}
    for p in os.listdir('/proc'):
        if p.isdigit():
            try:
                with open(f'/proc/{p}/stat') as f:
                    s = f.read()
                ppid = int(s[s.rindex(')')+2:].split()[1])
                kids.setdefault(ppid, []).append(int(p))
            except (OSError, ValueError):
                pass
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += kids.get(p, [])
    return out


def rss(pids):
    t = 0
    for p in pids:
        try:
            with open(f'/proc/{p}/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        t += int(line.split()[1])*1024
                        break
        except OSError:
            pass
    return t


def cgroup(pid, limit, name):
    """A cgroup v2 child with memory.max; None when not possible."""
    try:
        with open('/proc/self/cgroup') as f:
            rel = next(x[3:].strip() for x in f if x.startswith('0::'))
        base = '/sys/fs/cgroup'+rel
        d = os.path.join(base, f'bm1-{name}-{pid}')
        os.mkdir(d)
        with open(os.path.join(d, 'memory.max'), 'w') as f:
            f.write(str(limit))
        with open(os.path.join(d, 'cgroup.procs'), 'w') as f:
            f.write(str(pid))
        return d
    except (OSError, StopIteration):
        return None


def main():
    pid, limit, logp, name = int(sys.argv[1]), parse(sys.argv[2]), sys.argv[3], sys.argv[4]
    cg = cgroup(pid, limit, name)
    if cg:
        log(logp, f'memguard {name}: cgroup {cg} memory.max {limit} bytes')
    else:
        log(logp, f'memguard {name}: no writable cgroup; watching the process tree of {pid} (RSS sum, 2 s, limit {limit} bytes)')
    peak = seen = 0
    while os.path.exists(f'/proc/{pid}'):
        if cg:
            try:
                with open(os.path.join(cg, 'memory.events')) as f:
                    ev = dict(x.split() for x in f)
                k = int(ev.get('oom_kill', 0))
                if k > seen:
                    log(logp, f'memguard {name}: {k-seen} OOM kill(s) inside the cgroup (memory.events {ev})')
                    seen = k
            except OSError:
                pass
        else:
            m = rss(tree(pid))
            peak = max(peak, m)
            if m > limit:
                log(logp, f'memguard {name}: RSS {m} bytes over the limit {limit}: SIGTERM to the process tree of {pid}')
                for p in tree(pid):
                    try:
                        os.kill(p, signal.SIGTERM)
                    except OSError:
                        pass
                time.sleep(10)
                left = [p for p in tree(pid) if os.path.exists(f'/proc/{p}')]
                if left:
                    log(logp, f'memguard {name}: SIGKILL to {len(left)} process(es) still alive')
                    for p in left:
                        try:
                            os.kill(p, signal.SIGKILL)
                        except OSError:
                            pass
                return
        time.sleep(2)
    if not cg:
        log(logp, f'memguard {name}: run ended, RSS peak {peak} bytes (2 s samples)')


if __name__ == '__main__':
    main()
