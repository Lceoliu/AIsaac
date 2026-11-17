"""C109: launch check of an ab4 arm (read-only). ab4_check.py <run name> [N=20]"""
import csv
import json
import os
import statistics as st
import sys

name = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
d = f'/root/AIsaac/train/ab4/{name}/{name}-s0'
R = list(csv.DictReader(open(d + '/progress.csv')))
eps = []
for fn in sorted(os.listdir(d)):
    if fn.startswith('episodes.jsonl'):
        for line in open(os.path.join(d, fn)):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if int(e.get('update', 0)) <= N:
                eps.append(e)
by = {int(r['update']): r for r in R}
u = N if N in by else int(R[-1]['update'])
ERR = ('errors', 'stalls', 'cf_errors', 'v2_errors', 'broken_episodes', 'w_overrun', 'ctr_oom_kills')
c = json.load(open(d + '/config.json'))
a = c.get('args', c)
print(f'{name}: {len(R)} rows, u{R[-1]["update"]} {float(R[-1]["game_hours"]):.1f} gh')
print(f'  u1-{N}: {len(eps)} episodes, forced {sum(1 for e in eps if e.get("pickup_forced")) / max(len(eps), 1):.3f}')
print(f'  pickup_force_p u1 {by[1]["pickup_force_p"]} u{u} {by[u]["pickup_force_p"]} (1 - gh/25000 = '
      f'{1 - float(by[u]["game_hours"]) / 25000:.6f})')
print('  errors', {k: sum(float(r[k] or 0) for r in R) for k in ERR if k in R[0]})
print('  stale_share median', st.median(float(r['stale_share']) for r in R), 'lr', sorted({r['lr'] for r in R}))
print('  x_real median last 10', round(st.median(float(r['x_real_time']) for r in R[-10:]), 1),
      'ctr_mem_gb last', R[-1].get('ctr_mem_gb', '')[:5])
print('  grad_norm logged rows', sum(1 for r in R if r.get('grad_norm') not in (None, '')),
      'aux cols present', any(k.startswith('aux_') for k in R[0]), 'priv cols present', any('priv' in k for k in R[0]))
print('  config', {k: a.get(k) for k in ('overlap', 'decoupled', 'learner_device', 'aux_predict', 'priv_critic', 'lr',
                                          'lr_final', 'pickup_force', 'checkpoint_every', 'seed', 'workers')})
