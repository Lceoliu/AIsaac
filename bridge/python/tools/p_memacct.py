"""(2026-11-20, the coordinator's memory ledger) Where a training run's memory goes -- read only, on the host, standard
library: /proc/<pid>/smaps_rollup and status of every process in the trainer's container (its cgroup's cgroup.procs:
orphaned clones included), the trainer's and the learner's shared-memory / memfd mappings one by one (smaps), and the
container's cgroup memory.stat / memory.current / memory.peak. Nothing is written but --out; no signal, no ptrace.

Kinds: trainer (train_tok.py's process), learner (a python process with CUDA's threads), workers (the other python
processes: the sampler workers, their forked search / cf children), game root (an isaac* process whose parent is not
an isaac* one: an instance), game clone (an isaac* process whose parent is one: a fork), orphan clone (an isaac*
process whose parent is the container's init), other. Per kind: processes, RSS, PSS (anonymous / file / shmem),
private, shared (MB). PSS sums to the container's mapped memory without double counting; the cgroup's anon + file
(+ shmem inside file) + kernel is what its limit counts.

   usage: python3 p_memacct.py --match <a unique part of the trainer's command line> --out f.json
          [--every 600 --times 6]   (several samples: the growth)"""
import argparse
import json
import os
import time


def rd(path):
    try:
        with open(path, 'rb') as f:
            return f.read()
    except OSError:
        return None


def stat_of(pid):
    s = rd(f'/proc/{pid}/stat')
    if s is None:
        return None
    s = s.decode('latin-1')
    r = s.rfind(')')
    f = s[r + 2:].split()
    return dict(comm=s[s.find('(') + 1:r], ppid=int(f[1]))


def cmdline(pid):
    return (rd(f'/proc/{pid}/cmdline') or b'').replace(b'\0', b' ').decode('latin-1')


def find_trainer(match):
    best = None
    for p in os.listdir('/proc'):
        if p.isdigit():
            c = cmdline(p)
            if 'train_tok.py' in c and match in c:
                st = stat_of(p)
                if st and st['comm'].startswith('python'):
                    best = int(p) if best is None else min(best, int(p))
    return best


def cgroup_path(pid):
    for line in (rd(f'/proc/{pid}/cgroup') or b'').decode('latin-1').splitlines():
        if line.startswith('0::'):
            return '/sys/fs/cgroup' + line[3:].strip()
    return None


def kv(text, sep=None):
    out = {}
    for line in (text or b'').decode('latin-1').splitlines():
        parts = line.split(sep) if sep else line.split()
        if len(parts) >= 2:
            k = parts[0].rstrip(':')
            try:
                out[k] = int(parts[1])
            except ValueError:
                pass
    return out


def uses_cuda(pid):
    try:
        for t in os.listdir(f'/proc/{pid}/task'):
            c = (rd(f'/proc/{pid}/task/{t}/comm') or b'').decode('latin-1').strip()
            if c.startswith('cuda-EvtHandlr'):
                return True
    except OSError:
        pass
    return False


ROLL = ('Rss', 'Pss', 'Pss_Anon', 'Pss_File', 'Pss_Shmem', 'Shared_Clean', 'Shared_Dirty', 'Private_Clean',
        'Private_Dirty', 'Anonymous', 'Swap')


def mappings(pid):
    """Shared-memory / memfd / deleted-file mappings of a process: name -> {Rss, Pss, Size} (kB), from smaps."""
    out = {}
    cur = None
    for line in (rd(f'/proc/{pid}/smaps') or b'').decode('latin-1').splitlines():
        parts = line.split()
        if len(parts) >= 5 and '-' in parts[0] and len(parts[0]) > 8 and ':' in parts[3]:   # a mapping header
            name = ' '.join(parts[5:]) if len(parts) > 5 else ''
            cur = name if ('/dev/shm' in name or 'memfd:' in name or '(deleted)' in name or 'SYSV' in name) else None
            if cur is not None:
                out.setdefault(cur, dict(Size=0, Rss=0, Pss=0, maps=0))['maps'] += 1
        elif cur is not None and len(parts) >= 2 and parts[0].rstrip(':') in ('Size', 'Rss', 'Pss'):
            key = parts[0].rstrip(':')
            if key == 'Size':   # (11-20 fix: a segment mapped twice in one process -- tok_sampler creates it, then
                #               attach_* opens it by name -- reserves it once: the largest mapping, not the sum)
                out[cur]['Size'] = max(out[cur]['Size'], int(parts[1]))
            else:   # (Pss: exact for pages seen through several mappings; Rss would count such a page per mapping)
                out[cur][key] += int(parts[1])
    return out


def sample(root):
    cg = cgroup_path(root)
    pids = []
    if cg:
        pids = [int(x) for x in (rd(cg + '/cgroup.procs') or b'').split() if x.strip()]
    if not pids:
        pids = [root]
    info = {p: stat_of(p) for p in pids}
    info = {p: v for p, v in info.items() if v}
    tops = [p for p, v in info.items() if v['ppid'] not in info]   # (the container's init: its parent is outside)
    init = min(tops) if tops else None
    kinds = {}
    per_kind = {}
    unread = 0
    for p, st in info.items():
        comm = st['comm']
        par = info.get(st['ppid'])
        if p == root:
            k = 'trainer'
        elif comm.startswith('isaac'):
            if st['ppid'] == init:
                k = 'orphan clone'
            elif par is not None and par['comm'].startswith('isaac'):
                k = 'game clone'
            else:
                k = 'game root'
        elif comm.startswith('python'):
            k = 'learner' if uses_cuda(p) else 'workers'
        else:
            k = 'other'
        kinds[p] = k
        r = kv(rd(f'/proc/{p}/smaps_rollup'))
        if not r:
            unread += 1
            continue
        d = per_kind.setdefault(k, dict({x: 0 for x in ROLL}, n=0))
        d['n'] += 1
        for x in ROLL:
            d[x] += r.get(x, 0)
    mb = lambda v: round(v / 1024.0, 1)   # noqa: E731
    table = {k: dict(n=d['n'], rss_mb=mb(d['Rss']), pss_mb=mb(d['Pss']), pss_anon_mb=mb(d['Pss_Anon']),
                     pss_file_mb=mb(d['Pss_File']), pss_shmem_mb=mb(d['Pss_Shmem']),
                     private_mb=mb(d['Private_Clean'] + d['Private_Dirty']),
                     shared_mb=mb(d['Shared_Clean'] + d['Shared_Dirty']), swap_mb=mb(d['Swap']))
             for k, d in sorted(per_kind.items())}
    status = {}
    for p, k in kinds.items():
        if k in ('trainer', 'learner'):
            s = kv(rd(f'/proc/{p}/status'))
            status[f'{k} {p}'] = {x: s.get(x) for x in ('VmRSS', 'RssAnon', 'RssFile', 'RssShmem', 'VmPin', 'VmLck',
                                                        'VmSwap', 'VmHWM')}
    maps = {}
    for p, k in kinds.items():
        if k in ('trainer', 'learner'):
            for name, m in mappings(p).items():
                e = maps.setdefault(name, dict(Size_mb=0.0, Rss_mb=0.0, Pss_mb=0.0, by=[], maps=[]))
                e['maps'].append(m['maps'])
                e['Size_mb'] = max(e['Size_mb'], mb(m['Size']))
                e['Rss_mb'] = max(e['Rss_mb'], mb(m['Rss']))
                e['Pss_mb'] = round(e['Pss_mb'] + mb(m['Pss']), 1)
                e['by'].append(k)
    big = dict(sorted(maps.items(), key=lambda kv_: -kv_[1]['Rss_mb'])[:40])
    mstat = kv(rd(cg + '/memory.stat')) if cg else {}
    keep = ('anon', 'file', 'kernel', 'kernel_stack', 'pagetables', 'sec_pagetables', 'percpu', 'sock', 'shmem',
            'file_mapped', 'file_dirty', 'slab', 'slab_reclaimable', 'slab_unreclaimable', 'active_anon',
            'inactive_anon', 'active_file', 'inactive_file', 'unevictable', 'anon_thp')
    cgm = {k: round(mstat[k] / 2 ** 20, 1) for k in keep if k in mstat}
    cur = rd(cg + '/memory.current') if cg else None
    peak = rd(cg + '/memory.peak') if cg else None
    return dict(time=time.time(), processes=len(info), unreadable=unread, kinds=table, trainer_learner_status=status,
                shared_mappings_mb=big, cgroup_mb=cgm,
                cgroup_current_mb=round(int(cur) / 2 ** 20, 1) if cur and cur.strip().isdigit() else None,
                cgroup_peak_mb=round(int(peak) / 2 ** 20, 1) if peak and peak.strip().isdigit() else None,
                pss_total_mb=round(sum(v['pss_mb'] for v in table.values()), 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--match', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=float, default=0)
    ap.add_argument('--times', type=int, default=1)
    a = ap.parse_args()
    root = find_trainer(a.match)
    if root is None:
        json.dump(dict(error='no trainer'), open(a.out, 'w'))
        return
    out = dict(match=a.match, trainer_pid=root, samples=[])
    for i in range(max(1, a.times)):
        if not os.path.exists(f'/proc/{root}'):
            break
        out['samples'].append(sample(root))
        with open(a.out, 'w') as f:
            json.dump(out, f, indent=1)
        if i + 1 < a.times:
            time.sleep(a.every)


if __name__ == '__main__':
    main()
