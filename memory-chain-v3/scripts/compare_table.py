"""Old vs new architecture: per-update increments by memory category (markdown).

usage: python -X utf8 compare_table.py label=dir:from:to ...
Categories are Pss sums (smaps_rollup) of the process groups of audit_inside.py;
coverage = summed Pss increment / cgroup (anon + shmem) increment.
"""
import json, sys

G = 2**30
CATS = [('games: archive (parked room entries)', lambda g: g == 'game:archive'),
        ('games: other (unmapped, episodes, queues, root, template)', lambda g: g.startswith('game:') and g != 'game:archive'),
        ('python: trainer (= actor0)', lambda g: g == 'trainer'),
        ('python: spawned by trainer (learner, actor1, 64 game bridges)', lambda g: g == 'spawn_of_trainer'),
        ('python: other spawned (3 source workers, ...)', lambda g: g == 'spawn_other'),
        ('compile + other', lambda g: g in ('compile', 'python_other') or g.startswith('other'))]


def load(d):
    stat, audit = {}, {}
    for line in open(f'{d}/memory.jsonl'):
        m = json.loads(line)
        if 'audit' in m:
            if 'error' not in m['audit']: audit[m['progress_rows']] = m['audit']
        else: stat[m['progress_rows']] = m['stat']
    return stat, audit


cols = []
for spec in sys.argv[1:]:
    label, rest = spec.split('=', 1)
    d, a, b = rest.rsplit(':', 2); a, b = int(a), int(b)
    stat, audit = load(d)
    n = b - a
    def cat(u, test):
        return sum(v['Pss'] for g, v in audit[u]['groups'].items() if test(g)) / G
    row = {name: (cat(b, t) - cat(a, t)) / n for name, t in CATS}
    row['Pss sum'] = sum(row[name] for name, _ in CATS)
    row['cgroup anon'] = (stat[b]['anon'] - stat[a]['anon']) / G / n
    row['cgroup shmem'] = (stat[b]['shmem'] - stat[a]['shmem']) / G / n
    row['coverage (Pss / anon+shmem)'] = row['Pss sum'] / (row['cgroup anon'] + row['cgroup shmem']) * 100
    row['archive procs'] = f"{audit[a]['groups'].get('game:archive', {}).get('count', 0)}->{audit[b]['groups'].get('game:archive', {}).get('count', 0)}"
    row['level at end (cgroup anon+shmem GiB)'] = (stat[b]['anon'] + stat[b]['shmem']) / G
    cols.append((f'{label} ({a}->{b})', row))

names = [n for n, _ in CATS] + ['Pss sum', 'cgroup anon', 'cgroup shmem', 'coverage (Pss / anon+shmem)', 'archive procs',
                               'level at end (cgroup anon+shmem GiB)']
print('| GiB per update | ' + ' | '.join(c for c, _ in cols) + ' |')
print('|---|' + '---|' * len(cols))
for n in names:
    cells = []
    for _, row in cols:
        v = row[n]
        cells.append(v if isinstance(v, str) else (f'{v:.0f}%' if 'coverage' in n else f'{v:.1f}' if 'level' in n else f'{v:+.3f}'))
    print(f'| {n} | ' + ' | '.join(cells) + ' |')
