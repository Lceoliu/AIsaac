"""Inside-container memory audit (docker exec, read-only): one JSON line on stdout.

Every process's /proc/<pid>/smaps_rollup (Pss, Private_Dirty, ...), grouped by
role. Game roles come from the workers' own role dumps (ISAAC_RL_MEM_LOG,
tok_floor.mem_dump: root, template, archive, hurt-search queue, live parked
entries, running episode, base); python roles from the process tree. Python
processes also get /proc/<pid>/smaps summed by mapping class, and /dev/shm use.

usage: python audit_inside.py <memlog dir>
"""
import json, os, sys
from pathlib import Path

KEYS = ('Rss', 'Pss', 'Private_Clean', 'Private_Dirty', 'Shared_Clean', 'Shared_Dirty', 'Swap', 'Anonymous')


def rollup(pid):
    out = {}
    try:
        for line in open(f'/proc/{pid}/smaps_rollup'):
            k = line.split(':')[0]
            if k in KEYS: out[k] = int(line.split()[1]) * 1024
    except OSError:
        return None
    return out


def mapping_classes(pid):
    """Private_Dirty + Private_Clean by mapping kind (python processes only), and
    the largest file-backed mappings by path (private bytes, top 12)."""
    out = {}; kind = 'anon'; path = ''; files = {}
    try:
        for line in open(f'/proc/{pid}/smaps'):
            head = line.split()
            if len(head) >= 5 and '-' in head[0] and ':' in head[3]:
                path = ' '.join(head[5:]) if len(head) > 5 else ''
                kind = ('anon' if not path or path.startswith('[anon') else 'heap' if path == '[heap]'
                        else 'stack' if path.startswith('[stack') else 'nvidia_dev' if path.startswith('/dev/nvidia')
                        else 'shm' if path.startswith(('/dev/shm', '/memfd:', '/SYSV')) else 'file')
            elif head and head[0] in ('Private_Dirty:', 'Private_Clean:', 'Shared_Dirty:'):
                size = int(head[1]) * 1024
                if head[0] != 'Shared_Dirty:': out[kind] = out.get(kind, 0) + size
                if kind in ('file', 'shm'):
                    files[path] = files.get(path, 0) + size
    except OSError:
        pass
    # collapse per-object names (e.g. /tmp/xyz123) to their directory for a stable summary
    dirs = {}
    for p, size in files.items():
        key = p if p.startswith('/bench/home') or p.endswith('.so') or '.so.' in p else os.path.dirname(p) + '/*'
        dirs[key] = dirs.get(key, 0) + size
    out['top_paths'] = dict(sorted(dirs.items(), key=lambda kv: -kv[1])[:12])
    return out


def game_roles(memlog):
    roles = {}
    prio = ['hurt_base', 'base', 'episode', 'live_parked', 'queue', 'root', 'template', 'archive']   # later wins
    latest = {}
    for f in Path(memlog).glob('w*.jsonl'):
        try:
            last = f.read_text().strip().splitlines()[-1]
            latest[f.stem] = json.loads(last)
        except (OSError, IndexError, ValueError):
            continue
    for rec in latest.values():
        marks = dict(hurt_base=rec.get('hurts') or [], base=[rec.get('base')], episode=[rec.get('episode')],
                     live_parked=[p[0] for p in rec.get('live') or []], queue=[q[0] for q in rec.get('queue') or []],
                     root=[rec.get('root')], template=[rec.get('template')], archive=[a[0] for a in rec.get('archive') or []])
        for role in prio:
            for pid in marks[role]:
                if pid: roles[int(pid)] = role
    return roles, len(latest), max((r.get('t', 0) for r in latest.values()), default=0)


def main():
    memlog = sys.argv[1]
    roles, workers, newest = game_roles(memlog)
    procs = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            args = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            stat = (p / 'stat').read_text(); ppid = int(stat.rsplit(')', 1)[1].split()[1])
        except OSError:
            continue
        procs[int(p.name)] = (ppid, args)
    trainer = {pid for pid, (pp, a) in procs.items() if 'train_tok.py' in a and a.split()[0].endswith('python')}
    spawned = {pid for pid, (pp, a) in procs.items() if 'spawn_main' in a}
    big = {}
    groups = {}
    for pid, (ppid, args) in procs.items():
        r = rollup(pid)
        if r is None: continue
        if 'isaac.x64' in args:
            role = 'game:' + roles.get(pid, 'unmapped')
        elif 'torch/_inductor' in args: role = 'compile'
        elif pid in trainer: role = 'trainer'
        elif pid in spawned and ppid in trainer: role = 'spawn_of_trainer'
        elif pid in spawned: role = 'spawn_other'
        elif 'python' in args.split()[0] if args else False: role = 'python_other'
        else: role = 'other:' + (args.split()[0].split('/')[-1] if args else '?')
        g = groups.setdefault(role, dict(count=0, **{k: 0 for k in KEYS}))
        g['count'] += 1
        for k in KEYS: g[k] += r.get(k, 0)
        if not role.startswith(('game', 'other')) and r.get('Rss', 0) > 300 * 2**20:
            big[pid] = dict(role=role, ppid=ppid, rollup=r, maps=mapping_classes(pid))
    # source workers: spawned children of the learner (the spawn child of trainer with big spawned children)
    shm = os.statvfs('/dev/shm')
    shm_files = sum(f.stat().st_size for f in Path('/dev/shm').rglob('*') if f.is_file())
    print(json.dumps(dict(groups=groups, python=big, dev_shm_used=(shm.f_blocks - shm.f_bfree) * shm.f_frsize,
                          dev_shm_files=shm_files, role_workers=workers, role_newest=newest, mapped_games=len(roles))))


if __name__ == '__main__':
    main()
