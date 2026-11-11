"""Reader of abp_turbo's ABP_CLONE_STATS files (clone-stats-<root pid>.bin, one per game instance; abp_turbo.c cs_*).

The instance runs with ABP_CLONE_STATS=<dir> (off by default); every process of the instance (the root, its clones,
their clones) adds to that file, per fork purpose (the bridge command's "purpose": episode, snap, hurt, fatal, random,
cf, fork_many, other). This sums any number of files (an instance each, e.g. every worker of a host) and prints per
purpose:
  forks / clones / exits / reaped (+ reaped without ABP_EXIT) / frames, frames per clone;
  the clones' CPU per clone (ms): start (the fork's return to the first frame: isolation, the bridge's answer,
    the first copy-on-write faults), sim (first frame to ABP_EXIT), teardown (the reaped whole minus the CPU at
    ABP_EXIT: exit_mmap), whole; sim CPU per frame (us); minor faults per clone (start, sim);
  the parents' side per fork: wall (ms; mean and max), the fork syscall's wall, the forking thread's CPU across fork()
    (the page-table / VMA copy), the parent's VmRSS and VmPTE (MB) and map count at the fork.
  usage: python3 clone_stats.py [--json out.json] [--since earlier.json] <file or dir> ...
--json writes the summed raw counters (to diff a window later with --since: the counters minus that snapshot's)."""
import glob
import json
import os
import struct
import sys

NAMES = ('other', 'episode', 'snap', 'hurt', 'fatal', 'random', 'cf', 'fork_many')
FIELDS = ('forks', 'fork_wall_ns', 'fork_sys_wall_ns', 'fork_cpu_ns', 'fork_wall_max_ns', 'rss_kb', 'pte_kb', 'maps', 'maps_n',
          'clones', 'frames', 'cpu_start_ns', 'cpu_sim_ns', 'flt_start', 'flt_sim', 'exits',
          'reaped', 'cpu_total_ns', 'cpu_teardown_ns', 'reaped_noexit', 'cpu_noexit_ns')
HEAD = struct.Struct('<8s7Q')


def read(path):
    with open(path, 'rb') as f:
        raw = f.read()
    magic, version, root, n_p, n_f, n_slots, lost, _ = HEAD.unpack_from(raw, 0)
    if magic != b'ABPCLST1' or version != 1:
        raise ValueError(f'{path}: not an ABP_CLONE_STATS v1 file')
    vals = struct.unpack_from(f'<{n_p*n_f}Q', raw, HEAD.size)
    out = {}
    for p in range(n_p):
        out[NAMES[p] if p < len(NAMES) else f'p{p}'] = {FIELDS[k]: vals[p*n_f+k] for k in range(min(n_f, len(FIELDS)))}
    return out, dict(root=root, slots_lost=lost)


def files(args):
    for a in args:
        if os.path.isdir(a):
            yield from sorted(glob.glob(os.path.join(a, 'clone-stats-*.bin')))
        else:
            yield a


def summed(paths):
    tot, meta = {}, dict(files=0, slots_lost=0)
    for path in paths:
        t, m = read(path)
        meta['files'] += 1; meta['slots_lost'] += m['slots_lost']
        for p, d in t.items():
            acc = tot.setdefault(p, dict.fromkeys(FIELDS, 0))
            for k, v in d.items():
                acc[k] = max(acc[k], v) if k == 'fork_wall_max_ns' else acc[k]+v
    return tot, meta


def report(tot, meta):
    def r(a, b, scale=1.0):
        return f'{a/b*scale:9.3f}' if b else f'{"-":>9s}'
    print(f'files {meta["files"]}, exit slots lost {meta["slots_lost"]}')
    print(f'{"purpose":10s} {"forks":>7s} {"clones":>7s} {"exits":>7s} {"reaped":>7s} {"noexit":>6s} {"frames":>9s} {"fr/clone":>9s} '
          f'{"start ms":>9s} {"sim ms":>9s} {"tear ms":>9s} {"whole ms":>9s} {"us/frame":>9s} {"flt st":>9s} {"flt sim":>9s}')
    for p in NAMES:
        d = tot.get(p)
        if not d or not (d['forks'] or d['clones']):
            continue
        c = d['clones']
        print(f'{p:10s} {d["forks"]:7d} {c:7d} {d["exits"]:7d} {d["reaped"]:7d} {d["reaped_noexit"]:6d} {d["frames"]:9d} '
              f'{r(d["frames"], d["exits"])} {r(d["cpu_start_ns"], d["exits"], 1e-6)} {r(d["cpu_sim_ns"], d["exits"], 1e-6)} '
              f'{r(d["cpu_teardown_ns"], d["reaped"], 1e-6)} {r(d["cpu_total_ns"], d["reaped"], 1e-6)} '
              f'{r(d["cpu_sim_ns"], d["frames"], 1e-3)} {r(d["flt_start"], d["exits"])} {r(d["flt_sim"], d["exits"])}')
    print(f'{"purpose":10s} {"fork ms":>9s} {"max ms":>9s} {"sys ms":>9s} {"cpu ms":>9s} {"RSS MB":>9s} {"PTE MB":>9s} {"maps":>9s}   (the parent, per fork)')
    for p in NAMES:
        d = tot.get(p)
        if not d or not d['forks']:
            continue
        n = d['forks']
        print(f'{p:10s} {r(d["fork_wall_ns"], n, 1e-6)} {d["fork_wall_max_ns"]/1e6:9.3f} {r(d["fork_sys_wall_ns"], n, 1e-6)} '
              f'{r(d["fork_cpu_ns"], n, 1e-6)} {r(d["rss_kb"], n, 1/1024)} {r(d["pte_kb"], n, 1/1024)} {r(d["maps"], d["maps_n"])}')
    cpu = sum(d['cpu_total_ns']+d['cpu_noexit_ns']+d['fork_cpu_ns'] for d in tot.values())
    print(f'all reaped clones\' CPU + the parents\' fork CPU: {cpu/1e9:.3f} s')


def main(argv):
    out = since = None
    args = []
    i = 0
    while i < len(argv):
        if argv[i] == '--json':
            out = argv[i+1]; i += 2
        elif argv[i] == '--since':
            since = argv[i+1]; i += 2
        else:
            args.append(argv[i]); i += 1
    tot, meta = summed(list(files(args)))
    if out:
        with open(out, 'w') as f:
            json.dump(dict(tot=tot, meta=meta), f)
    if since:
        with open(since) as f:
            old = json.load(f)['tot']
        for p, d in tot.items():
            for k in d:
                if k != 'fork_wall_max_ns':
                    d[k] -= old.get(p, {}).get(k, 0)
    report(tot, meta)


if __name__ == '__main__':
    main(sys.argv[1:])
