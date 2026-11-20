"""P package (2026-11-20, the coordinator's CPU ledger): where a training run's CPU goes, by process and thread, over a
window of updates -- run on the host next to the run (standard library only: the host's python3).

The trainer is found by its command line ('train_tok.py' and --match, e.g. its --out directory); every descendant is
read from /proc each --interval seconds (utime + stime per thread, its state and the syscall it is in), and the ticks
between consecutive samples are summed per role while the window is open: from the moment progress.csv holds
--from-update - 1 rows (update --from-update under way) to the moment it holds --to-update rows.
Roles:
  actor main thread   the trainer's first thread (the Python loop: poll / book / act / store; with the C++ driver:
                      the consumer -- book, store -- and the lanes' rounds)
  driver thread       the trainer's thread named tok-driver (the C++ driver)
  trainer:<name>      the trainer's other threads by name (CUDA's, torch's, the lanes', ...)
  learner process     the spawned learner (its threads summed; per name below)
  python workers      the worker processes (the forked Python side: records, encoding, the game's I/O)
  game engines        isaac* processes (the engines the workers drive: they advance the game frames)
  other:<comm>        everything else under the trainer
Per role: cores (CPU seconds / window seconds), processes / threads seen; for the trainer's and the learner's threads,
the share of samples on a CPU outside any syscall ('running': user code -- a spin wait counts here), in sched_yield
(a yielding wait) and blocked in a syscall (futex / poll / ioctl ...: waiting without the CPU). The capacity is the
trainer's Cpus_allowed_list; idle = capacity - all roles.

   usage: python3 p_cpuacct.py --match <str> --progress <progress.csv> --out <f.json> [--from-update 2]
          [--to-update 6] [--interval 1] [--timeout 1800]"""
import argparse
import json
import os
import time

CLK = os.sysconf('SC_CLK_TCK')
SYS_NAMES = {'0': 'read', '1': 'write', '7': 'poll', '16': 'ioctl', '23': 'select', '24': 'sched_yield',
             '35': 'nanosleep', '202': 'futex', '232': 'epoll_wait', '228': 'clock_gettime', '230': 'clock_nanosleep',
             '271': 'ppoll', '281': 'epoll_pwait', '45': 'recvfrom', '47': 'recvmsg', '44': 'sendto'}


def rd(path):
    try:
        with open(path, 'rb') as f:
            return f.read()
    except OSError:
        return None


def stat_of(path):
    s = rd(path)
    if s is None:
        return None
    s = s.decode('latin-1')
    r = s.rfind(')')
    f = s[r + 2:].split()
    return dict(comm=s[s.find('(') + 1:r], state=f[0], ppid=int(f[1]), ticks=int(f[11]) + int(f[12]),
                cpu=int(f[36]) if len(f) > 36 else -1)


def cpu_stat():
    """/proc/stat per logical CPU: (busy ticks, all ticks)."""
    out = {}
    for line in (rd('/proc/stat') or b'').decode('latin-1').splitlines():
        if line.startswith('cpu') and line[3:4].isdigit():
            f = line.split()
            v = [int(x) for x in f[1:]]
            idle = v[3] + (v[4] if len(v) > 4 else 0)
            out[int(f[0][3:])] = (sum(v[:8]) - idle, sum(v[:8]))
    return out


def siblings(c):
    s = rd(f'/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list')
    return s.decode().strip() if s else str(c)


def allowed(pid):
    for line in (rd(f'/proc/{pid}/status') or b'').decode('latin-1').splitlines():
        if line.startswith('Cpus_allowed_list'):
            return line.split(':', 1)[1].strip()
    return '?'


def cmdline(pid):
    s = rd(f'/proc/{pid}/cmdline')
    return s.replace(b'\0', b' ').decode('latin-1') if s else ''


def find_trainer(match):
    """The trainer: the python process (not a wrapping shell / timeout) whose command line runs train_tok.py with
    `match`, its parent not one of them."""
    cand = []
    for p in os.listdir('/proc'):
        if not p.isdigit():
            continue
        c = cmdline(p)
        st = stat_of(f'/proc/{p}/stat')
        if st and 'train_tok.py' in c and match in c and st['comm'].startswith('python'):
            cand.append((int(p), st['ppid']))
    pids = {p for p, _ in cand}
    top = [p for p, pp in cand if pp not in pids]
    return min(top) if top else None


def uses_cuda(pid):
    """A process with CUDA's event-handler thread (the trainer, the learner process: the workers never use CUDA)."""
    try:
        for t in os.listdir(f'/proc/{pid}/task'):
            s = rd(f'/proc/{pid}/task/{t}/comm')
            if s and s.startswith(b'cuda-EvtHandlr'):
                return True
    except OSError:
        pass
    return False


def descendants(root):
    kids = {}
    for p in os.listdir('/proc'):
        if p.isdigit():
            st = stat_of(f'/proc/{p}/stat')
            if st:
                kids.setdefault(st['ppid'], []).append(int(p))
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += kids.get(p, [])
    return out


def _under(q, root, per_proc):
    """Is process q a descendant of root (by the parents seen)?"""
    seen = set()
    while q in per_proc and q not in seen:
        seen.add(q)
        q = per_proc[q][3]
        if q == root:
            return True
    return False


def rows_of(path):
    s = rd(path)
    return max(0, len([x for x in s.decode('latin-1').splitlines() if x.strip()]) - 1) if s else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--match', required=True)
    ap.add_argument('--progress', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--from-update', type=int, default=2)
    ap.add_argument('--to-update', type=int, default=6)
    ap.add_argument('--interval', type=float, default=1.0)
    ap.add_argument('--timeout', type=float, default=1800)
    a = ap.parse_args()
    t_end = time.time() + a.timeout
    root = None
    while root is None and time.time() < t_end:
        root = find_trainer(a.match)
        if root is None:
            time.sleep(2)
    if root is None:
        json.dump(dict(error='no trainer'), open(a.out, 'w'))
        return
    cpus = ''
    for line in (rd(f'/proc/{root}/status') or b'').decode('latin-1').splitlines():
        if line.startswith('Cpus_allowed_list'):
            cpus = line.split(':', 1)[1].strip()

    def ncpu(spec):
        n = 0
        for part in spec.split(','):
            if '-' in part:
                lo, hi = part.split('-')
                n += int(hi) - int(lo) + 1
            elif part:
                n += 1
        return n
    kinds = {}           # pid -> its kind once known
    per_tid = {}         # (pid, tid) -> [ticks in the window, kind, comm, last cpu]
    per_proc = {}        # pid -> [ticks in the window, kind, comm, parent]
    cs0 = cs1 = None
    procs_seen = {}      # pid -> (kind, comm)
    last = {}            # (pid, tid) -> ticks
    acc = {}             # role -> ticks within the window
    seen = {}            # role -> set of tids
    states = {}          # role -> {state: samples} (trainer / learner threads)
    open_ = False
    t0 = t1 = None
    while time.time() < t_end:
        n_rows = rows_of(a.progress)
        if not open_ and n_rows >= a.from_update - 1:
            open_, t0 = True, time.time()
            cs0 = cpu_stat()
        if open_ and n_rows >= a.to_update:
            t1 = time.time()
            cs1 = cpu_stat()
            break
        if not os.path.exists(f'/proc/{root}'):
            break
        procs = descendants(root)
        for pid in procs:
            st = stat_of(f'/proc/{pid}/stat')
            if st is None:
                continue
            if pid == root:
                kind = 'trainer'
            elif pid in kinds:
                kind = kinds[pid]
            elif st['comm'].startswith('isaac'):
                kind = 'game'
            elif st['comm'].startswith('python'):
                kind = 'learner' if uses_cuda(pid) else 'workers'
            else:
                kind = 'other'
            if kind == 'learner':
                kinds[pid] = kind   # (fixed once its CUDA threads exist; a python process without them re-checked)
            try:
                tids = os.listdir(f'/proc/{pid}/task')
            except OSError:
                continue
            for t in tids:
                ts = stat_of(f'/proc/{pid}/task/{t}/stat')
                if ts is None:
                    continue
                if kind == 'trainer':
                    role = 'actor main thread' if int(t) == pid else (
                        'driver thread' if ts['comm'] == 'tok-driver' else f"trainer:{ts['comm']}")
                elif kind == 'learner':
                    role = 'learner process' if int(t) == pid else f"learner:{ts['comm']}"
                elif kind == 'game':
                    role = 'game engines'
                elif kind == 'workers':
                    role = 'python workers'
                else:
                    role = f"other:{st['comm']}"
                key = (pid, int(t))
                procs_seen[pid] = (kind, st['comm'])
                prev = last.get(key)
                last[key] = ts['ticks']
                if open_ and prev is not None:
                    acc[role] = acc.get(role, 0) + ts['ticks'] - prev
                    e_ = per_tid.setdefault(key, [0, kind, ts['comm'], -1])
                    e_[0] += ts['ticks'] - prev
                    e_[3] = ts['cpu']
                    q_ = per_proc.setdefault(pid, [0, kind, st['comm'], st['ppid']])
                    q_[0] += ts['ticks'] - prev
                    seen.setdefault(role, set()).add(key)
                    if kind in ('trainer', 'learner'):
                        sc = rd(f'/proc/{pid}/task/{t}/syscall')
                        if sc is None:   # (not readable here: the scheduler state only)
                            lab = 'state R (on a CPU / runnable)' if ts['state'] == 'R' else f"state {ts['state']}"
                        else:
                            sc = sc.decode('latin-1').split()[0]
                            lab = 'running' if sc == 'running' or (ts['state'] == 'R' and sc == '-1') else (
                                'sched_yield' if sc == '24' else SYS_NAMES.get(sc, f'syscall {sc}'))
                        d = states.setdefault(role, {})
                        d[lab] = d.get(lab, 0) + 1
        time.sleep(a.interval)
    if t1 is None:
        t1 = time.time()
    win = (t1 - t0) if t0 else 0.0
    roles = {}
    for r, tk in acc.items():
        roles[r] = dict(cores=round(tk / CLK / win, 3) if win else None, threads=len(seen.get(r, ())))
        if r in states:
            n = sum(states[r].values())
            roles[r]['states'] = {k: round(v / n, 3) for k, v in sorted(states[r].items(), key=lambda kv: -kv[1])}
    total = sum(v['cores'] or 0 for v in roles.values())
    cap = ncpu(cpus)
    groups = {'game engines': 0.0, 'python workers': 0.0, 'actor main thread': 0.0, 'driver thread': 0.0,
              'trainer: other threads': 0.0, 'learner process': 0.0, 'other': 0.0}
    for r, v in roles.items():
        c_ = v['cores'] or 0
        if r in groups:
            groups[r] += c_
        elif r.startswith('trainer:'):
            groups['trainer: other threads'] += c_
        elif r.startswith('learner:'):
            groups['learner process'] += c_
        else:
            groups['other'] += c_
    nk = {}
    for k_, _ in procs_seen.values():
        nk[k_] = nk.get(k_, 0) + 1
    if cs1 is None:
        cs1 = cpu_stat()
    cpu_list = []
    for part in cpus.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            cpu_list += list(range(int(lo), int(hi) + 1))
        elif part:
            cpu_list.append(int(part))
    per_cpu = {}
    if cs0:
        for c in cpu_list:
            b0, a0 = cs0.get(c, (0, 0))
            b1, a1 = cs1.get(c, (0, 0))
            per_cpu[c] = round((b1 - b0) / max(a1 - a0, 1), 3)
    cores_ = {}
    for c in cpu_list:
        cores_.setdefault(siblings(c), []).append(per_cpu.get(c))
    top = sorted(per_tid.items(), key=lambda kv: -kv[1][0])[:20]
    top_threads = [dict(pid=k[0], tid=k[1], kind=v[1], comm=v[2], cores=round(v[0] / CLK / win, 3) if win else None,
                        last_cpu=v[3]) for k, v in top]
    # each worker: its own process and its game engines (their descendants), with the CPUs they may run on
    work = []
    for pid, v in per_proc.items():
        if v[1] == 'workers':
            g_ = sum(x[0] for q, x in per_proc.items() if x[1] == 'game' and _under(q, pid, per_proc))
            work.append(dict(pid=pid, worker_cores=round(v[0] / CLK / win, 3) if win else None,
                             game_cores=round(g_ / CLK / win, 3) if win else None, cpus=allowed(pid)))
    out = dict(match=a.match, trainer_pid=root, processes=nk, per_cpu_busy=per_cpu,
               per_physical_core={k: v for k, v in cores_.items()},
               top_threads=top_threads, workers=sorted(work, key=lambda d: -(d['worker_cores'] or 0)),
               window_s=round(win, 1), updates=[a.from_update, a.to_update], cpus=cpus, capacity=cap,
               total_cores=round(total, 3), idle_cores=round(cap - total, 3), groups={k: round(v, 3) for k, v in groups.items()},
               roles=dict(sorted(roles.items(), key=lambda kv: -(kv[1]['cores'] or 0))))
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
