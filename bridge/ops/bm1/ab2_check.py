#!/usr/bin/env python3
"""ab2 (C107, 2026-11-14): launch / progress check of both ab2 arms, read-only, standard library.
  ab2_check.py [N=20]
Per arm (/root/AIsaac/train/ab2/<arm>/<arm>-s0): progress rows, the last update and its game hours; the episodes of
updates 1..N: their count and forced-pickup share; pickup_force_p at u1, uN and the last row (the line 1 -> 0.952 over
1,200 game hours: about 0.999 at u20); errors (sum, and the arm's other error columns that exist); x_real_time median
of the last 20 rows; the guard's last line."""
import csv
import json
import os
import statistics as st
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
T = '/root/AIsaac/train/ab2'
ERR = ('errors', 'stalls', 'cf_errors', 'v2_errors', 'broken_episodes', 'w_overrun')
for arm in ('ab2-c99old', 'ab2-bm'):
    d = f'{T}/{arm}/{arm}-s0'
    try:
        rows = list(csv.DictReader(open(f'{d}/progress.csv')))
    except OSError:
        print(f'{arm}: no progress.csv yet')
        continue
    eps = []
    for name in sorted(os.listdir(d)):
        if name.startswith('episodes.jsonl'):
            for line in open(f'{d}/{name}'):
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if int(e.get('update', 0)) <= N:
                    eps.append(e)
    by = {int(r['update']): r for r in rows}
    last = rows[-1]

    def g(r, k):
        try:
            return float(r.get(k) or 'nan')
        except (TypeError, ValueError):
            return float('nan')
    errs = ', '.join(f'{k} {sum(g(r, k) for r in rows if g(r, k) == g(r, k)):.0f}' for k in ERR if k in rows[0])
    xr = [g(r, 'x_real_time') for r in rows[-20:]]
    pf = lambda u: f'{g(by[u], "pickup_force_p"):.4f}' if u in by else '-'
    gl = open(f'{T}/{arm}/guard.log').read().strip().splitlines()[-1]
    print(f'{arm}: {len(rows)} rows, u{last["update"]} {g(last, "game_hours"):.1f} gh | u1-{N}: {len(eps)} episodes, '
          f'forced {sum(1 for e in eps if e.get("pickup_forced")) / max(len(eps), 1):.3f} | pickup_force_p u1 {pf(1)} '
          f'u{N} {pf(N)} u{last["update"]} {g(last, "pickup_force_p"):.4f} | {errs} | x_real median(last 20) '
          f'{st.median(xr):.0f} | guard: {gl[:160]}')
