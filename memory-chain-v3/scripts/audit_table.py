"""Per-update memory by category from guard memory.jsonl (stat + smaps audit).

usage: python -X utf8 audit_table.py <label>=<results/case> ... [--from 2] [--to N]
Prints, per run, rows of cgroup anon/shmem/file/kernel and Pss/Private_Dirty per
process group, then the increment table (from -> to) and the coverage of the
cgroup's anon+shmem increment by the summed Pss increment.
"""
import json, sys

args = [a for a in sys.argv[1:] if '=' in a]
lo = int(sys.argv[sys.argv.index('--from') + 1]) if '--from' in sys.argv else 2
hi = int(sys.argv[sys.argv.index('--to') + 1]) if '--to' in sys.argv else None
G = 2**30


def merge(group):
    """Fold process groups into report categories."""
    if group.startswith('game:'): return 'game_' + group[5:]
    if group in ('trainer',): return 'py_trainer(actor0)'
    if group == 'spawn_of_trainer': return 'py_spawn_of_trainer(learner,actor1,game_bridges)'
    if group == 'spawn_other': return 'py_spawn_other(sources,...)'
    if group == 'compile': return 'py_compile'
    if group.startswith('other'): return 'other'
    return 'py_other'


results = {}
for spec in args:
    label, d = spec.split('=', 1)
    stat, audit = {}, {}
    for line in open(f'{d}/memory.jsonl'):
        m = json.loads(line)
        if 'audit' in m:
            if 'error' not in m['audit']: audit[m['progress_rows']] = m['audit']
        else:
            stat[m['progress_rows']] = m['stat']
    rows = {}
    for n in sorted(set(stat) & set(audit)):
        s, a = stat[n], audit[n]
        r = dict(anon=s['anon'] / G, shmem=s['shmem'] / G, file=s['file'] / G, kernel=s.get('kernel', 0) / G,
                 dev_shm=a['dev_shm_used'] / G)
        pss, dirty, count = {}, {}, {}
        for g, v in a['groups'].items():
            c = merge(g)
            pss[c] = pss.get(c, 0) + v['Pss'] / G
            dirty[c] = dirty.get(c, 0) + v['Private_Dirty'] / G
            count[c] = count.get(c, 0) + v['count']
        r['pss'], r['dirty'], r['count'] = pss, dirty, count
        r['pss_total'] = sum(pss.values())
        rows[n] = r
    results[label] = rows
    print(f'== {label} ({d}): {len(rows)} updates with stat+audit')
    cats = sorted({c for r in rows.values() for c in r['pss']})
    for n, r in rows.items():
        print(f"  {n:>2} anon {r['anon']:5.1f} shmem {r['shmem']:4.1f} file {r['file']:4.1f} kernel {r['kernel']:4.2f} "
              f"/dev/shm {r['dev_shm']:4.2f} | Pss sum {r['pss_total']:5.1f} | "
              + ' '.join(f"{c.split('(')[0]}={r['pss'].get(c, 0):.2f}/{r['count'].get(c, 0)}" for c in cats))

print('\n== increments (GiB per update; Pss by category; count = processes at the end)')
for label, rows in results.items():
    keys = sorted(rows)
    a, b = (lo if lo in rows else keys[0]), (hi if hi in rows else keys[-1])
    n = b - a
    ra, rb = rows[a], rows[b]
    d_anon = rb['anon'] - ra['anon']; d_shm = rb['shmem'] - ra['shmem']; d_pss = rb['pss_total'] - ra['pss_total']
    print(f'-- {label}: updates {a}->{b} ({n})  anon {d_anon / n:+.3f}  shmem {d_shm / n:+.3f}  file {(rb["file"] - ra["file"]) / n:+.3f}  '
          f'kernel {(rb["kernel"] - ra["kernel"]) / n:+.3f}  | Pss sum {d_pss / n:+.3f}  coverage of anon+shmem by Pss: '
          f'{d_pss / (d_anon + d_shm) * 100 if d_anon + d_shm else float("nan"):.0f}%')
    for c in sorted(set(ra['pss']) | set(rb['pss']), key=lambda c: -(rb['pss'].get(c, 0) - ra['pss'].get(c, 0))):
        dp = rb['pss'].get(c, 0) - ra['pss'].get(c, 0); dd = rb['dirty'].get(c, 0) - ra['dirty'].get(c, 0)
        print(f"     {c:<52} Pss {dp / n:+.3f}  Private_Dirty {dd / n:+.3f}  share of anon+shmem {dp / (d_anon + d_shm) * 100:5.1f}%  "
              f"procs {ra['count'].get(c, 0)}->{rb['count'].get(c, 0)}")
