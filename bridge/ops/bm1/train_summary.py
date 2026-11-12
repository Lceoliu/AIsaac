#!/usr/bin/env python3
"""Offline summary of a training run's own episodes (no online evaluation; read-only: it never touches the trainers).

  python3 train_summary.py <run dir> [--window 400] [--csv out.csv]

Reads <run dir>/<name>-s<r>/episodes*.jsonl (rotated files too) and progress.csv of every rank. Per window of updates
(default 400): for the episodes started from a floor start (episodes.jsonl 'start' fresh) the mean deepest stage,
new rooms entered ('rooms': the ROW's EV_NEW_ROOM count; the episode summaries have no cleared-room count), bosses
cleared, collectibles gained, half hearts lost, and the shares of the ends (death, stall, the time cap, broken off,
other); the same per character (char0); the zero-progress share over all episodes (a stall end with no monster hurt,
no new room, no item, no exit); and from progress.csv the window's median KL, entropy, combined x real time and the
last container memory. Standard library only (the host's python3)."""
import argparse
import csv
import glob
import json
import os
import statistics as st
import sys

CHAR = {0: 'Isaac', 1: 'Magdalene', 2: 'Cain'}


def load(rd):
    eps, prog = [], {}
    for d in sorted(glob.glob(os.path.join(rd, '*-s[0-9]*'))):
        if not os.path.isdir(d):
            continue
        r = d.rsplit('-s', 1)[1]
        for f in sorted(glob.glob(os.path.join(d, 'episodes*.jsonl'))):
            with open(f) as fh:
                for line in fh:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue   # (a line being written)
                    e['_rank'] = r
                    eps.append(e)
        p = os.path.join(d, 'progress.csv')
        if os.path.exists(p):
            with open(p) as fh:
                prog[r] = list(csv.DictReader(fh))
    return eps, prog


def end_kind(e):
    if e.get('broken'):
        return 'broken'
    if e.get('done') == 2:
        return 'death'
    if e.get('stall'):
        return 'stall'
    if e.get('cap'):
        return 'cap'
    return 'other'


def zero_progress(e):
    return bool(e.get('stall')) and not e.get('damage') and not e.get('rooms') and not e.get('items') \
        and not e.get('exits')


def stats(E):
    if not E:
        return None
    n = len(E)
    ends = {k: sum(end_kind(e) == k for e in E) / n for k in ('death', 'stall', 'cap', 'broken', 'other')}
    m = lambda k: st.mean(float(e.get(k) or 0) for e in E)
    return dict(n=n, stage=m('stage'), rooms=m('rooms'), boss=m('boss'), items=m('items'), hurt=m('hurt'), **ends)


def med(rows, k):
    v = []
    for x in rows:
        try:
            v.append(float(x[k]))
        except (KeyError, TypeError, ValueError):
            pass
    return st.median(v) if v else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rd')
    ap.add_argument('--window', type=int, default=400)
    ap.add_argument('--csv', default='')
    a = ap.parse_args()
    eps, prog = load(a.rd)
    if not eps:
        sys.exit(f'no episodes under {a.rd}')
    last = max(int(e.get('update', 0)) for e in eps)
    out, W = [], a.window
    for lo in range(1, last + 1, W):
        hi = min(lo + W - 1, last)
        E = [e for e in eps if lo <= int(e.get('update', 0)) <= hi]
        fresh = [e for e in E if e.get('start') == 'fresh']
        P = [x for rows in prog.values() for x in rows if lo <= int(x['update']) <= hi]
        p0 = [x for x in prog.get('0', []) if lo <= int(x['update']) <= hi]
        x_sum = sum(med([x for x in prog[r] if lo <= int(x['update']) <= hi], 'x_real_time') for r in prog)
        ent = ' '.join(f'{k[4:]} {med(P, k):.2f}' for k in ('ent_move', 'ent_shoot', 'ent_bomb', 'ent_item', 'ent_pill')
                       if P and k in P[0])
        row = dict(window=f'u{lo}-{hi}', episodes=len(E), fresh=len(fresh),
                   zero_progress=sum(zero_progress(e) for e in E) / max(len(E), 1),
                   kl_median=med(P, 'kl'), entropy=med(P, 'entropy'), x_real=x_sum,
                   ctr_mem_gb=float(p0[-1]['ctr_mem_gb']) if p0 and p0[-1].get('ctr_mem_gb') else float('nan'))
        s = stats(fresh)
        if s:
            row.update({f'fresh_{k}': v for k, v in s.items()})
        for c, name in CHAR.items():
            s = stats([e for e in fresh if e.get('char0') == c])
            if s:
                row.update({f'{name}_{k}': v for k, v in s.items()})
        out.append(row)
        print(f"== {row['window']}: {row['episodes']} episodes ({row['fresh']} fresh starts); zero-progress "
              f"{row['zero_progress']:.2f} of all; KL median {row['kl_median']:.4f}, entropy {row['entropy']:.2f} "
              f"({ent}); x real {row['x_real']:.0f}; container {row['ctr_mem_gb']:.1f} GB")
        print(f"   {'':10s} {'n':>6s} {'stage':>6s} {'rooms':>6s} {'boss':>6s} {'items':>6s} {'hurt':>6s}   ends: "
              f"{'death':>6s} {'stall':>6s} {'cap':>6s} {'broken':>6s} {'other':>6s}")
        for name in ['fresh'] + list(CHAR.values()):
            if f'{name}_n' not in row:
                continue
            g = lambda k: row[f'{name}_{k}']
            print(f"   {name:10s} {g('n'):6d} {g('stage'):6.2f} {g('rooms'):6.2f} {g('boss'):6.2f} {g('items'):6.2f} "
                  f"{g('hurt'):6.2f}         {g('death'):6.2f} {g('stall'):6.2f} {g('cap'):6.2f} {g('broken'):6.2f} "
                  f"{g('other'):6.2f}")
    if a.csv:
        keys = sorted({k for r in out for k in r}, key=lambda k: (k != 'window', k))
        with open(a.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(out)


if __name__ == '__main__':
    main()
