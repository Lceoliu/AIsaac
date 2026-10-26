"""Per-update host memory of ablation runs: anon slope, game processes, records.

usage: python -X utf8 ablation_summary.py <results/case> [...]
"""
import csv, json, sys
import numpy as np

for d in sys.argv[1:]:
    prog = list(csv.DictReader(open(f'{d}/progress.csv')))
    mem = {}
    for line in open(f'{d}/memory.jsonl'):
        m = json.loads(line); mem[m['progress_rows']] = m
    learn = {}
    for line in open(f'{d}/train.log', errors='replace'):
        if 'MEMORY_ACCOUNT ' in line:
            a = json.loads(line.split('MEMORY_ACCOUNT ', 1)[1])
            if a['role'] == 'learner': learn[a['step']] = a
    rows = []
    for i, r in enumerate(prog, 1):
        m = mem.get(i); a = learn.get(int(r['update']))
        if not m: continue
        rows.append(dict(i=i, anon=m['stat']['anon'] / 2**30, shmem=m['stat']['shmem'] / 2**30, games=m['rss']['count']['game'],
                         archive=float(r['archive'] or 0), teach_held=int(float(r['teach_held'] or 0)),
                         v2_held=int(float(r.get('v2_held') or 0)), src_eps=a['source_live_episodes'] if a else None,
                         learner_anon=a['proc'].get('RssAnon', 0) / 2**30 if a else None,
                         rss={k: m['rss'][k] / 2**30 for k in ('trainer', 'learner', 'source', 'actor', 'worker')}))
    x = np.array([r['i'] for r in rows if r['i'] >= 2], float)
    y = np.array([r['anon'] for r in rows if r['i'] >= 2])
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 2 else float('nan')
    print(f'== {d}: updates {len(prog)}, anon slope (updates 2..) {slope:.3f} GiB/update')
    for r in rows:
        print(f"  {r['i']:>2} anon {r['anon']:6.2f} shmem {r['shmem']:5.2f} games {r['games']:4} archive {r['archive']:5.0f} "
              f"teach_held {r['teach_held']:5} v2_held {r['v2_held']:4} src_eps {r['src_eps']} learnerAnon {r['learner_anon'] and round(r['learner_anon'], 2)} "
              + ' '.join(f'{k}={v:.2f}' for k, v in r['rss'].items()))
