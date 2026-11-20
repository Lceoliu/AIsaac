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
'unattributed' (3b): the lane's busy CPU (/proc/stat) split into the threads sampled, the container's own CPU the
sampler missed (its cgroup's cpu.stat minus the sampled) and the CPU outside the container (the rest); the
descendants born / exited per second by kind and the host's forks per second.
Per role: cores (CPU seconds / window seconds), processes / threads seen; for the trainer's and the learner's threads,
the share of samples on a CPU outside any syscall ('running': user code -- a spin wait counts here), in sched_yield
(a yielding wait) and blocked in a syscall (futex / poll / ioctl ...: waiting without the CPU). The capacity is the
lane: the trainer's container's effective cpuset (cgroup v2 cpuset.cpus.effective), else the union of the CPUs its
process tree may run on (Codex round 22: not the trainer's own mask -- ISAAC_TOK_PIN narrows that to the actor's two
CPUs); idle = capacity - all roles. 'masks': each role's threads' actual masks (Cpus_allowed_list), as last seen.

   usage: python3 p_cpuacct.py --match <str> --progress <progress.csv> --out <f.json> [--from-update 2]
          [--to-update 6] [--interval 1] [--timeout 1800]"""
import argparse
import json
import os
import time

CLK = os.sysconf('SC_CLK_TCK') if hasattr(os, 'sysconf') else 100   # (imported by tests elsewhere too)
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
                cpu=int(f[36]) if len(f) > 36 else -1, cticks=int(f[13]) + int(f[14]))


def cpu_stat():
    """/proc/stat per logical CPU: (busy ticks, all ticks)."""
    out = {}
    for line in (rd('/proc/stat') or b'').decode('latin-1').splitlines():
        if line.startswith('cpu') and line[3:4].isdigit():
            f = line.split()
            v = [int(x) for x in f[1:]]
            idle = v[3] + (v[4] if len(v) > 4 else 0)
            out[int(f[0][3:])] = (sum(v[:8]) - idle, sum(v[:8]), v[0] + v[1], v[2], v[5] + v[6])
    return out


def cg_stat(pid):
    """(3b) The container's own CPU from its cgroup (v2 cpu.stat, as the host sees it): every process in it, the
    short-lived forks a sampler misses included; None when not readable."""
    for line in (rd(f'/proc/{pid}/cgroup') or b'').decode('latin-1').splitlines():
        if line.startswith('0::'):
            s = rd('/sys/fs/cgroup' + line[3:].strip() + '/cpu.stat')
            if s is None:
                return None
            d = {}
            for l_ in s.decode('latin-1').splitlines():
                k, _, v = l_.partition(' ')
                if k in ('usage_usec', 'user_usec', 'system_usec'):
                    d[k] = int(v)
            return d or None
    return None


def host_forks():
    """/proc/stat 'processes': the host's forks since boot (every tenant's)."""
    for line in (rd('/proc/stat') or b'').decode('latin-1').splitlines():
        if line.startswith('processes'):
            return int(line.split()[1])
    return None


def siblings(c):
    s = rd(f'/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list')
    return s.decode().strip() if s else str(c)


def lane_of(pid):
    """(Codex round 22) The trainer's container's effective cpuset (cgroup v2), None when not readable."""
    for line in (rd(f'/proc/{pid}/cgroup') or b'').decode('latin-1').splitlines():
        if line.startswith('0::'):
            s = rd('/sys/fs/cgroup' + line[3:].strip() + '/cpuset.cpus.effective')
            if s and s.strip():
                return s.decode().strip()
    return None


def container_ancestors(pid):
    """(3b) The trainer's ancestors in its own cgroup, nearest first; the last is the container's init (docker --init:
    the reaper of every orphaned clone -- their CPU lands in its cutime, outside the trainer's tree)."""
    cg = rd(f'/proc/{pid}/cgroup')
    out, q = [], pid
    for _ in range(64):
        st = stat_of(f'/proc/{q}/stat')
        if st is None or st['ppid'] <= 1 or rd(f"/proc/{st['ppid']}/cgroup") != cg:
            break
        q = st['ppid']
        out.append(q)
    return out


def cpu_set(spec):
    out = set()
    for part in (spec or '').split(','):
        if '-' in part:
            lo, hi = part.split('-')
            out.update(range(int(lo), int(hi) + 1))
        elif part.strip().isdigit():
            out.add(int(part))
    return out


def spec_of(cpus):
    return ','.join(str(c) for c in sorted(cpus))


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
    cpus = lane_of(root)
    lane_source = 'cgroup cpuset.cpus.effective'
    if not cpus:   # (the union of the masks the trainer's tree may run on, grown while sampling)
        lane_source = "the trainer's tree's masks"
        cpus = spec_of(set().union(*(cpu_set(allowed(q)) for q in [root] + descendants(root))))

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
    child_acc = {}       # kind -> reaped children's ticks within the window (cutime + cstime: short-lived forks)
    clast = {}           # pid -> cticks
    cs0 = cs1 = None
    cg0 = cg1 = fk0 = fk1 = None
    anc = container_ancestors(root)   # (3b: the orphans' reaper and the wrappers between it and the trainer)
    anc0 = anc1 = None
    alive = None         # (3b) the descendants at the sample before: births / exits within the window by kind
    born, gone = {}, {}
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
            cg0, fk0 = cg_stat(root), host_forks()
            anc0 = {q: (stat_of(f'/proc/{q}/stat') or {}).get('cticks') for q in anc}
        if open_ and n_rows >= a.to_update:
            t1 = time.time()
            cs1 = cpu_stat()
            cg1, fk1 = cg_stat(root), host_forks()
            anc1 = {q: (stat_of(f'/proc/{q}/stat') or {}).get('cticks') for q in anc}
            break
        if not os.path.exists(f'/proc/{root}'):
            break
        procs = descendants(root)
        if open_ and alive is not None:
            now_ = set(procs)
            for q in now_ - alive:
                born[q] = True
            for q in alive - now_:
                k_ = procs_seen.get(q, ('?', ''))[0]
                gone[k_] = gone.get(k_, 0) + 1
        alive = set(procs)
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
            cp_ = clast.get(pid)
            clast[pid] = st['cticks']
            if open_ and cp_ is not None and st['cticks'] > cp_:
                child_acc[kind] = child_acc.get(kind, 0) + st['cticks'] - cp_
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
    # (Codex round 22) each role's threads' actual masks, as they stand at the window's end
    masks = {}
    for pid in descendants(root) if os.path.exists(f'/proc/{root}') else []:
        kind = procs_seen.get(pid, ('other', ''))[0]
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
                    'driver thread' if ts['comm'] == 'tok-driver' else 'trainer: other threads')
            elif kind == 'learner':
                role = 'learner process'
            else:
                role = {'game': 'game engines', 'workers': 'python workers'}.get(kind, 'other')
            m = '?'
            for line in (rd(f'/proc/{pid}/task/{t}/status') or b'').decode('latin-1').splitlines():
                if line.startswith('Cpus_allowed_list'):
                    m = line.split(':', 1)[1].strip()
            d = masks.setdefault(role, {})
            d[m] = d.get(m, 0) + 1
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
    per_cpu, kern = {}, dict(user=0.0, system=0.0, irq_softirq=0.0)
    if cs0:
        for c in cpu_list:
            x0, x1 = cs0.get(c, (0, 0, 0, 0, 0)), cs1.get(c, (0, 0, 0, 0, 0))
            per_cpu[c] = round((x1[0] - x0[0]) / max(x1[1] - x0[1], 1), 3)
            kern['user'] += (x1[2] - x0[2]) / CLK / win if win else 0
            kern['system'] += (x1[3] - x0[3]) / CLK / win if win else 0
            kern['irq_softirq'] += (x1[4] - x0[4]) / CLK / win if win else 0
    kern = {k: round(v, 3) for k, v in kern.items()}
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
    # (3b) where the lane's busy CPU goes: the threads sampled; the container's own CPU the sampler missed (forks
    # living less than a sample, a process's ticks after its last sample); outside the container (other tenants'
    # unpinned processes, kernel threads, interrupts) = the lane's busy CPU - the container's
    lane_busy = round(sum(kern.values()), 3)
    split = dict(lane_busy_cores=lane_busy, sampled_cores=round(total, 3))
    if cg0 and cg1 and win:
        cc = {k: round((cg1[k] - cg0[k]) / 1e6 / win, 3) for k in cg0 if k in cg1}
        split.update(container_cores=cc, missed_in_container_cores=round(cc['usage_usec'] - total, 3),
                     outside_container_cores=round(lane_busy - cc['usage_usec'], 3))
    else:
        split['container_cores'] = None
    bk = {}
    for q in born:
        k_ = procs_seen.get(q, ('?', ''))[0]
        bk[k_] = bk.get(k_, 0) + 1
    if anc0 and anc1 and win:   # (3b) children reaped above the trainer: the init's are the orphaned clones
        split['reaped_above_trainer_cores'] = {
            f"{q} {(stat_of(f'/proc/{q}/stat') or {}).get('comm', '?')}{' (init)' if q == anc[-1] else ''}":
                round((anc1[q] - anc0[q]) / CLK / win, 3)
            for q in anc if anc0.get(q) is not None and anc1.get(q) is not None}
    split.update(born_per_s={k: round(v / win, 2) for k, v in bk.items()} if win else None,
                 exited_per_s={k: round(v / win, 2) for k, v in gone.items()} if win else None,
                 host_forks_per_s=round((fk1 - fk0) / win, 1) if (fk0 is not None and fk1 is not None and win) else None)
    out = dict(match=a.match, trainer_pid=root, processes=nk, per_cpu_busy=per_cpu, unattributed=split,
               lane_cpu_by_mode_cores=kern,
               reaped_children_cores={k: round(v / CLK / win, 3) if win else None for k, v in child_acc.items()},
               per_physical_core={k: v for k, v in cores_.items()},
               top_threads=top_threads, workers=sorted(work, key=lambda d: -(d['worker_cores'] or 0)),
               window_s=round(win, 1), updates=[a.from_update, a.to_update], cpus=cpus, capacity=cap,
               lane_source=lane_source, masks=masks,
               total_cores=round(total, 3), idle_cores=round(cap - total, 3), groups={k: round(v, 3) for k, v in groups.items()},
               roles=dict(sorted(roles.items(), key=lambda kv: -(kv[1]['cores'] or 0))))
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
