"""Memory of a running whole-floor training by role (diagnostic, 2026-10-06; no dependencies, any python3).

Every --every s one JSON line to --out:
- the host: /proc/meminfo (MemAvailable, AnonPages, Shmem, PageTables, Slab, swap ...);
- every isaac.x64 process: Pss / Rss / Private_Dirty / SwapPss (smaps_rollup), VmPTE, state, parent, its instance
  name (XDG_DATA_HOME .../instances/<name>/data);
- the training's process tree (train_tok.py whose --name is --name, its descendants that are not game processes:
  trainer, workers) and an evaluation's (eval_tok.py with --name <name>e);
- the role of each game process of the training, from the workers' dumps (tok_floor ISAAC_RL_MEM_LOG, --mem-log):
  root (the instance a worker launched), template, archive (parked room entries), queue (restore points of queued
  hurts that are not archive entries or the template), parked_other (a parked clone the worker still holds that is
  none of these: a leak), episode (a child of a template or archive entry or other parked clone), other (the running
  episode's restore point, search clones, entries parked after the last dump), zombie;
- per worker: parked clones it holds against its caps (archive size + distinct queue bases + template), the root's RSS;
- the TCP sockets in the ephemeral port range (their local ports can collide with a bridge port).
usage: python3 abplus_probe_memory.py --name mema --mem-log <dir> --out <file> [--every 120] [--once]
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROLL = ('Rss', 'Pss', 'Pss_Anon', 'Pss_File', 'Pss_Shmem', 'Shared_Clean', 'Shared_Dirty', 'Private_Clean',
        'Private_Dirty', 'Swap', 'SwapPss')
MEMINFO = ('MemTotal', 'MemFree', 'MemAvailable', 'Buffers', 'Cached', 'SwapCached', 'AnonPages', 'Shmem', 'Slab',
           'SReclaimable', 'SUnreclaim', 'KernelStack', 'PageTables', 'SwapTotal', 'SwapFree', 'Committed_AS',
           'AnonHugePages')


def read(path, binary=False):
    try:
        with open(path, 'rb' if binary else 'r') as f:
            return f.read()
    except OSError:
        return None


def kb_fields(text, keys):
    out = {}
    for line in (text or '').splitlines():
        k, _, v = line.partition(':')
        if k in keys:
            try:
                out[k] = int(v.split()[0])
            except (ValueError, IndexError):
                pass
    return out


def procs():
    """pid -> dict(comm, ppid, state, cmd) of every process."""
    out = {}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        stat = read(f'/proc/{d}/stat')
        if not stat:
            continue
        try:
            comm = stat[stat.index('(') + 1:stat.rindex(')')]
            f = stat[stat.rindex(')') + 2:].split()
            out[int(d)] = dict(comm=comm, state=f[0], ppid=int(f[1]), start=int(f[19]))
        except (ValueError, IndexError):
            continue
    return out


def cmdline(pid):
    raw = read(f'/proc/{pid}/cmdline', True) or b''
    return raw.replace(b'\0', b' ').decode('utf-8', 'replace').strip()


def instance_name(pid):
    raw = read(f'/proc/{pid}/environ', True) or b''
    m = re.search(rb'XDG_DATA_HOME=[^\0]*/instances/([^/\0]+)/data', raw)
    return m.group(1).decode() if m else None


def mem_of(pid):
    out = kb_fields(read(f'/proc/{pid}/smaps_rollup'), ROLL)
    out.update(kb_fields(read(f'/proc/{pid}/status'), ('VmPTE', 'VmRSS')))
    return out


def descendants(table, root):
    kids = {}
    for p, v in table.items():
        kids.setdefault(v['ppid'], []).append(p)
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(kids.get(p, []))
    return out


def last_line(path):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            lines = f.read().splitlines()
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError, IndexError):
        return None


def ephemeral_sockets():
    """TCP sockets by state, and the ephemeral local ports held by sockets that block a later bind() of that port
    (every state but TIME_WAIT: luasocket's bind sets SO_REUSEADDR, which passes over TIME_WAIT only)."""
    lo, hi = (int(v) for v in (read('/proc/sys/net/ipv4/ip_local_port_range') or '32768 60999').split())
    blocking, waiting = set(), set()
    states = {}
    names = {'01': 'established', '06': 'time_wait', '0A': 'listen', '08': 'close_wait'}
    for path in ('/proc/net/tcp', '/proc/net/tcp6'):
        for line in (read(path) or '').splitlines()[1:]:
            parts = line.split()
            try:
                port = int(parts[1].split(':')[1], 16)
                st = names.get(parts[3], parts[3])
            except (IndexError, ValueError):
                continue
            states[st] = states.get(st, 0) + 1
            if lo <= port <= hi:
                (waiting if st == 'time_wait' else blocking).add(port)
    return dict(states=states, ephemeral_ports_blocking=len(blocking), ephemeral_ports_time_wait=len(waiting - blocking),
                ephemeral_range=hi - lo + 1)


def sample(args):
    t = time.time()
    table = procs()
    info = kb_fields(read('/proc/meminfo'), MEMINFO)
    trainer = evals = None
    for p, v in table.items():
        if v['comm'].startswith('python') or 'python' in v['comm']:
            c = cmdline(p)
            if 'train_tok.py' in c and f'--name {args.name} ' in c + ' ':
                trainer = p
            elif 'eval_tok.py' in c and f'--name {args.name}e ' in c + ' ':
                evals = p
    games = {p: v for p, v in table.items() if v['comm'] == 'isaac.x64'}
    names = {p: instance_name(p) for p in games}
    mems = {p: mem_of(p) for p in games if games[p]['state'] != 'Z'}
    tree = set(descendants(table, trainer)) if trainer else set()
    etree = set(descendants(table, evals)) if evals else set()
    # the workers' dumps
    dumps = {}
    if args.mem_log:
        for f in Path(args.mem_log).glob('w*.jsonl'):
            rec = last_line(f)
            if rec and t - rec.get('t', 0) < 4 * args.every + 120:
                dumps[rec['w']] = rec
    role = {}
    per_worker = {}
    pat = re.compile(rf'^{re.escape(args.name)}(\d+)$')
    epat = re.compile(rf'^{re.escape(args.name)}e(\d+)$')
    parked_parent = set()
    for w, rec in dumps.items():
        archive = {a[0] for a in rec.get('archive', []) if a[0]}
        queue = {q[0] for q in rec.get('queue', []) if q[0]}
        live = {l[0] for l in rec.get('live', []) if l[0]}
        tpl = rec.get('template')
        if rec.get('root'):
            role[rec['root']] = 'root'
        for p in live:
            role.setdefault(p, 'parked_other')
        for p in rec.get('hurts') or []:   # restore points the running episode's hurts hold
            if p:
                role[p] = 'episode_hurts'
        if rec.get('base'):
            role[rec['base']] = 'episode_base'
        if rec.get('episode'):
            role[rec['episode']] = 'episode'
        for p in queue:
            role[p] = 'queue'
        for p in archive:
            role[p] = 'archive'
        if tpl:
            role[tpl] = 'template'
        held = {p for p in (rec.get('hurts') or []) + [rec.get('base')] if p}
        parked_parent |= live | archive | queue | ({tpl} if tpl else set())
        per_worker[w] = dict(live=len(live), archive=len(archive), queue_bases=len(queue), queue=len(rec.get('queue', [])),
                             queue_not_archive=len(queue - archive - ({tpl} if tpl else set())),
                             episode_held=len(held - archive - queue - ({tpl} if tpl else set())),
                             cap=len(archive) + len(queue - archive) + (1 if tpl else 0),
                             over_cap=len(live - archive - queue - held - ({tpl} if tpl else set())),
                             root_rss=rec.get('root_rss'), launches=rec.get('launches'), builds=rec.get('builds'),
                             done=rec.get('done'), episodes=rec.get('episodes'), age=round(t - rec['t'], 1))
    roles = {}
    alive_by_worker = {}
    for p, v in games.items():
        nm = names.get(p) or '?'
        if v['state'] == 'Z':
            r = 'zombie'
        elif epat.match(nm):
            r = 'evaluation'
        elif pat.match(nm):
            r = role.get(p)
            if r is None:
                if v['ppid'] in tree and v['ppid'] != trainer and table.get(v['ppid'], {}).get('comm', '') != 'isaac.x64':
                    r = 'root'
                elif v['ppid'] in parked_parent:
                    r = 'episode'
                else:
                    r = 'other'
            w = int(pat.match(nm).group(1))
            alive_by_worker[w] = alive_by_worker.get(w, 0) + 1
        else:
            r = 'foreign'
        m = mems.get(p, {})
        s = roles.setdefault(r, dict(n=0, Pss=0, Rss=0, Private_Dirty=0, SwapPss=0, VmPTE=0, Pss_Shmem=0))
        s['n'] += 1
        for k in ('Pss', 'Rss', 'Private_Dirty', 'SwapPss', 'VmPTE', 'Pss_Shmem'):
            s[k] += m.get(k, 0)
    for w, n in alive_by_worker.items():
        per_worker.setdefault(w, {})['game_processes'] = n
    tree_py = [p for p in tree if games.get(p) is None]
    tree_mem = {p: mem_of(p) for p in tree_py}
    trainer_mem = mem_of(trainer) if trainer else {}
    rec = dict(t=t, time=time.strftime('%H:%M:%S'), meminfo=info, roles=roles, workers=per_worker,
               trainer=dict(pid=trainer, Pss=trainer_mem.get('Pss'), Rss=trainer_mem.get('Rss'),
                            SwapPss=trainer_mem.get('SwapPss')),
               tree_python=dict(n=len(tree_py), Pss=sum(m.get('Pss', 0) for m in tree_mem.values()),
                                SwapPss=sum(m.get('SwapPss', 0) for m in tree_mem.values())),
               tree_games=dict(n=sum(1 for p in tree if p in games),
                               Pss=sum(mems.get(p, {}).get('Pss', 0) for p in tree if p in games)),
               outside_tree_games=dict(n=sum(1 for p in games if p not in tree and pat.match(names.get(p) or '')),
                                       Pss=sum(mems.get(p, {}).get('Pss', 0) for p in games
                                               if p not in tree and pat.match(names.get(p) or ''))),
               eval_tree=dict(n=len(etree), Pss=sum(mem_of(p).get('Pss', 0) for p in etree)),
               pss_all_games=sum(m.get('Pss', 0) for m in mems.values()),
               png_files_kib={f.name: os.stat(f).st_blocks // 2 for f in Path('/dev/shm').glob('abp-png-*')},
               root_anon_kib=sum(kb_fields(read(f'/proc/{p}/status'), ('RssAnon',)).get('RssAnon', 0)
                                 for p, r in role.items() if r == 'root' and p in games),
               sockets=ephemeral_sockets())
    if args.roots:
        rec['root_detail'] = {str(p): dict(name=names.get(p), **{k: mems.get(p, {}).get(k) for k in ('Rss', 'Pss',
                                                                                                    'Private_Dirty')})
                              for p, r in role.items() if r == 'root' and p in games}
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, help="the training's instance name prefix (train_tok --name)")
    p.add_argument('--mem-log', default='', help='the workers\' ISAAC_RL_MEM_LOG directory')
    p.add_argument('--out', required=True)
    p.add_argument('--every', type=float, default=120.0)
    p.add_argument('--once', action='store_true')
    p.add_argument('--roots', type=int, default=1, help='1: per-root RSS / Pss in every line')
    args = p.parse_args()
    while True:
        t0 = time.time()
        rec = sample(args)
        rec['sample_s'] = round(time.time() - t0, 2)
        with open(args.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        if args.once:
            print(json.dumps(rec, indent=1))
            return
        time.sleep(max(1.0, args.every - (time.time() - t0)))


if __name__ == '__main__':
    main()
