#!/usr/bin/env python3
"""ab1 (2026-11-14, C106): a K=1 run's training log in windows of W (200) updates, read-only, standard library.
  ab1_win.py <run dir>/<name>-s0 [W]
Per window, from the episodes (episodes.jsonl and its rotated files): episodes, the share with a room cleared (rooms
>= 1; and among the fresh starts, start == 'fresh'), items per episode, floors passed (exits per episode, and episodes reaching stage >= 2), deaths / no-progress
ends, forced-pickup share; from progress.csv: the medians of x real, move + shoot entropy (and each head), bomb / item /
pill entropy, KL, game hours at the window's end, and the rows with v2_loss_pi."""
import csv
import json
import os
import statistics as st
import sys

d = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 200
eps = []
for name in sorted(os.listdir(d)):
    if name.startswith('episodes.jsonl'):
        with open(os.path.join(d, name)) as f:
            for line in f:
                try:
                    eps.append(json.loads(line))
                except ValueError:
                    pass
prog = {}
with open(os.path.join(d, 'progress.csv')) as f:
    for x in csv.DictReader(f):
        prog[int(x['update'])] = x


def fl(x, k):
    try:
        return float(x.get(k) or 'nan')
    except ValueError:
        return float('nan')


def med(v):
    v = [a for a in v if a == a]
    return st.median(v) if v else float('nan')


end = max(prog)
print(f'{d}: {len(eps)} episodes, {len(prog)} progress rows (u{min(prog)}-u{end})')
print('window | gh_end | eps | rooms>=1 | fresh: eps rooms>=1 | items | exits/ep stage>=2 | death stall | forced | x_real | move+shoot (move shoot) '
      '| bomb item pill | kl | v2_loss_pi rows')
for lo in range(1, end + 1, W):
    hi = min(lo + W - 1, end)
    E = [e for e in eps if lo <= int(e.get('update', 0)) <= hi]
    R = [prog[u] for u in range(lo, hi + 1) if u in prog]
    m = max(len(E), 1)
    F = [e for e in E if e.get('start') == 'fresh']   # (the others start from the archive: already inside a floor)
    mv, sh = med([fl(a, 'ent_move') for a in R]), med([fl(a, 'ent_shoot') for a in R])
    print(f'u{lo}-{hi} | {fl(R[-1], "game_hours"):.0f} | {len(E)} | {sum(e.get("rooms", 0) >= 1 for e in E) / m:.3f} | '
          f'{len(F)} {sum(e.get("rooms", 0) >= 1 for e in F) / max(len(F), 1):.3f} | '
          f'{sum(e.get("items", 0) for e in E) / m:.3f} | {sum(e.get("exits", 0) for e in E) / m:.3f} '
          f'{sum(e.get("stage", 1) >= 2 for e in E)} | {sum(e.get("done") == 2 for e in E) / m:.3f} '
          f'{sum(e.get("done") == 3 for e in E) / m:.3f} | {sum(1 for e in E if e.get("pickup_forced")) / m:.3f} | '
          f'{med([fl(a, "x_real_time") for a in R]):.0f} | {mv + sh:.3f} ({mv:.3f} {sh:.3f}) | '
          + ' '.join(f'{med([fl(a, k) for a in R]):.3f}' for k in ('ent_bomb', 'ent_item', 'ent_pill'))
          + f' | {med([fl(a, "kl") for a in R]):.4f} | {sum(1 for a in R if a.get("v2_loss_pi"))}/{len(R)}')
