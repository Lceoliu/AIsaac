"""Live objects vs RSS per process role, and the allocator A/B (markdown).

usage: python -X utf8 census_table.py label=<results/case> ... [--window 9:20]
"""
import csv, json, sys
import numpy as np

G = 2**30
args = [a for a in sys.argv[1:] if '=' in a]
lo, hi = map(int, (sys.argv[sys.argv.index('--window') + 1] if '--window' in sys.argv else '9:20').split(':'))


def load(d):
    prog = list(csv.DictReader(open(f'{d}/progress.csv')))
    upd = {int(r['update']): i + 1 for i, r in enumerate(prog)}   # update number -> progress row (1-based)
    acct = {}
    for line in open(f'{d}/train.log', errors='replace'):
        if 'MEMORY_ACCOUNT ' not in line: continue
        a = json.loads(line.split('MEMORY_ACCOUNT ', 1)[1])
        role = a['role'] if not a['role'].startswith('source-') else a['role']
        acct.setdefault(role, []).append(a)
    stat, audit = {}, {}
    for line in open(f'{d}/memory.jsonl'):
        m = json.loads(line)
        if 'audit' in m:
            if 'error' not in m['audit']: audit[m['progress_rows']] = m['audit']
        else: stat[m['progress_rows']] = m['stat']
    return prog, upd, acct, stat, audit


def slope(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 2 else float('nan')


report = {}
for spec in args:
    label, d = spec.split('=', 1)
    prog, upd, acct, stat, audit = load(d)
    out = {}
    rows = [n for n in sorted(set(stat) & set(audit)) if lo <= n <= hi]
    out['updates'] = len(prog)
    out['anon'] = slope(rows, [stat[n]['anon'] / G for n in rows])
    out['shmem'] = slope(rows, [stat[n]['shmem'] / G for n in rows])
    def pss(n, test): return sum(v['Pss'] for g, v in audit[n]['groups'].items() if test(g)) / G
    out['python Pss'] = slope(rows, [pss(n, lambda g: not g.startswith(('game', 'other'))) for n in rows])
    out['games Pss'] = slope(rows, [pss(n, lambda g: g.startswith('game')) for n in rows])
    for role, lines in sorted(acct.items()):
        if role.startswith('source-'):
            xs = [a['step'] for a in lines]   # retire count ~ updates
        elif role == 'learner':
            xs = [upd.get(a['step'], None) for a in lines]
        else:
            xs = [a['step'] for a in lines]   # actor hand-offs ~ updates
        pts = [(x, a) for x, a in zip(xs, lines) if x is not None and lo <= x <= hi]
        if len(pts) < 3: continue
        x = [p[0] for p in pts]
        out[f'{role} RssAnon'] = slope(x, [p[1]['proc'].get('RssAnon', 0) / G for p in pts])
        out[f'{role} RawHistory raw'] = slope(x, [p[1]['census']['raw_bytes'] / G for p in pts])
        out[f'{role} RawHistory objects'] = slope(x, [p[1]['census']['counts']['RawHistory'] for p in pts])
        out[f'{role} tuples'] = slope(x, [p[1]['census']['containers'].get('tuple', 0) for p in pts])
        out[f'{role} census s'] = float(np.median([p[1]['census']['seconds'] for p in pts]))
        if 'lane_refs_prev' in pts[-1][1]:
            out[f'{role} lane refs alive'] = '%d/%d' % (pts[-1][1]['lane_refs_alive'], pts[-1][1]['lane_refs_prev'])
            out[f'{role} rebuild_contexts'] = pts[-1][1].get('rebuild_contexts')
    report[label] = out

keys = []
for o in report.values():
    for k in o:
        if k not in keys: keys.append(k)
print(f'| per update, updates {lo}-{hi} | ' + ' | '.join(report) + ' |')
print('|---|' + '---|' * len(report))
for k in keys:
    cells = []
    for o in report.values():
        v = o.get(k, '')
        cells.append(v if isinstance(v, (str, int)) or v == '' else (f'{v:+.3f}' if 'objects' not in k and 'tuples' not in k and ' s' not in k else f'{v:+.0f}' if ' s' not in k else f'{v:.2f}'))
    print(f'| {k} | ' + ' | '.join(str(c) for c in cells) + ' |')
