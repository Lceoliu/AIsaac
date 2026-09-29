"""Summary of a goexplore_abplus.py run (EXPERIMENTS.md A8): per room its cells, wins, the best win (half hearts lost,
seconds), the exploration at which the first win and the first damage-free win were found, the share of returns whose
digest differed at first, failed returns, retired cells and the final check; then the totals and the last progress line.

usage: python goexplore_summary.py <run dir> [<room catalog, e.g. ../abplus/catalog/abplus_basement1_rooms.json>]
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
cat = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
names = {}
if cat:
    for r in cat['normal'] + cat.get('boss', []):
        c = Counter(f"{e['type']}.{e['variant']}" for e in r['enemies'])
        names[r['variant']] = ' '.join(f'{k}x{v}' for k, v in sorted(c.items()))
rows = []
for line in open(run / 'results.jsonl'):
    r = json.loads(line)
    if 'failed' in r:
        print('FAILED', r['seed'], r['failed'][:200])
        continue
    d = json.loads((run / 'rooms' / f"{r['seed']}.json").read_text())
    wins = [c for c in d['cells'] if c['outcome'] == 'win']
    first = min((c['found'] for c in wins), default=None)
    clean = [c for c in wins if c.get('hurt', 0) == 0]   # no damage taken (hearts picked up do not count)
    first_clean = min((c['found'] for c in clean), default=None)
    best = r['best_cell']
    tasks = r['dispatched'] + 1
    retries = sum(r.get(k, 0) for k in ('retry_match', 'retry_same', 'retry_other'))
    won = best['outcome'] == 'win'
    rows.append(dict(seed=r['seed'], variant=r['room']['variant'], enemies=names.get(r['room']['variant'], ''),
                     cells=r['cells'], wins=len(wins), best_health=best['health'] if won else None,
                     lost=best.get('hurt') if won else None, verified=r.get('best_verified'),
                     best_s=round(best['frames'] / 30, 1) if best['outcome'] == 'win' else None,
                     progress=best['progress'], first_win=first, first_clean=first_clean,
                     mismatch=round(retries / tasks, 4), fails=r.get('return_fails', 0), retired=r['retired'],
                     verify=(r.get('verify') or {}).get('status'), game_h=r['game_hours']))
print(f"{'seed':>10} {'var':>4} {'cells':>5} {'wins':>4} {'lost':>4} {'best s':>6} {'1st win':>7} {'1st clean':>9} "
      f"{'mism':>6} {'fail':>4} {'ret':>3} {'verify':>6}  enemies")
for x in rows:
    print(f"{x['seed']:>10} {x['variant']:>4} {x['cells']:>5} {x['wins']:>4} {str(x['lost']):>4} {str(x['best_s']):>6} "
          f"{str(x['first_win']):>7} {str(x['first_clean']):>9} {x['mismatch']:>6} {x['fails']:>4} {x['retired']:>3} "
          f"{str(x['verify']):>6}  {x['enemies'][:60]}")
n = len(rows)
won = [x for x in rows if x['best_health'] is not None]
clean = [x for x in won if x['lost'] == 0]
print(f"\nrooms {n}; won {len(won)} ({len(won) / max(n, 1):.1%}); damage-free win {len(clean)} ({len(clean) / max(n, 1):.1%})")
if won:
    print('best win seconds: median', statistics.median(x['best_s'] for x in won),
          '| first win at iteration: median', statistics.median(x['first_win'] for x in won))
if clean:
    print('first damage-free win at iteration: median', statistics.median(x['first_clean'] for x in clean))
print('verify:', Counter(x['verify'] for x in rows), '| best win verified:', sum(bool(x['verified']) for x in won),
      'of', len(won))
print('rooms with mismatches:', sum(x['mismatch'] > 0 for x in rows), '| retired cells total', sum(x['retired'] for x in rows),
      '| failed returns total', sum(x['fails'] for x in rows))
print('game hours:', round(sum(x['game_h'] for x in rows), 2))
prog = [json.loads(l) for l in open(run / 'progress.jsonl')]
last = prog[-1]
print('last progress:', {k: last[k] for k in ('t', 'tasks', 'tasks_per_s', 'ok', 'fail', 'error', 'retries', 'game_hours',
                                               'x_realtime', 'reset_ms', 'return_fps', 'explore_fps', 'digest_ms')})
