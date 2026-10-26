"""Per source card per update: busy seconds (encode + writer + VJP), graph frames, allocation; spread across cards.

usage: python -X utf8 source_cards_table.py label=<run dir> ... [--steady 8:20]
From progress.csv (source_busy_w{i}: the previous cycle's encode + writer + source_backward seconds per card,
written by --source-balance builds; source_graph_frames_w{i}; source_balance_w{i}) and the SOURCE_PIPELINE lines
(worker_seconds per device and kind) of train.log. Spread = (max - min) / mean of the per-card medians.
"""
import csv, json, sys
import numpy as np

lo, hi = map(int, (sys.argv[sys.argv.index('--steady') + 1] if '--steady' in sys.argv else '8:20').split(':'))
for spec in [a for a in sys.argv[1:] if '=' in a and not a.startswith('--')]:
    label, d = spec.split('=', 1)
    rows = list(csv.DictReader(open(f'{d}/progress.csv')))[lo:hi]
    per = []
    for line in open(f'{d}/train.log', errors='replace'):
        if line.startswith('SOURCE_PIPELINE '):
            ws = json.JSONDecoder().raw_decode(line.split(' ', 1)[1])[0].get('worker_seconds', {})
            dev = {}
            for k, v in ws.items():
                dv, kind = k.rsplit(':', 1)
                if dv != 'host': dev.setdefault(dv, {})[kind] = v
            per.append(dev)
    per = per[lo:hi]
    devs = sorted({k for p in per for k in p})
    out = {}
    for dv in devs:
        kinds = {k: float(np.median([p.get(dv, {}).get(k, 0.) for p in per])) for k in ('encode', 'writer', 'source_backward')}
        out[dv] = dict(busy=float(np.median([sum(p.get(dv, {}).values()) for p in per])), **kinds)
    busy = np.asarray([o['busy'] for o in out.values()])
    n = len([k for k in rows[0] if k.startswith('source_graph_frames_w')]) if rows else 0
    frames = [float(np.median([float(r[f'source_graph_frames_w{i}']) for r in rows])) for i in range(n)]
    weights = [float(np.median([float(r.get(f'source_balance_w{i}') or 'nan') for r in rows])) for i in range(n)]
    print(f'## {label} ({d}, rows {lo}:{hi})')
    print('| card | busy s | encode | writer | VJP | graph frames | balance weight |')
    print('|---|---|---|---|---|---|---|')
    for i, (dv, o) in enumerate(out.items()):
        print(f"| {dv} | {o['busy']:.2f} | {o['encode']:.2f} | {o['writer']:.2f} | {o['source_backward']:.2f} | "
              f"{frames[i] if i < len(frames) else float('nan'):.0f} | {weights[i] if i < len(weights) else float('nan'):.2f} |")
    per_update = np.asarray([[sum(p.get(dv, {}).values()) for dv in devs] for p in per])
    upd_max = per_update.max(1); upd_spread = (per_update.max(1) - per_update.min(1)) / per_update.mean(1)
    print(f'per update: busiest card median {np.median(upd_max):.2f} s (q75 {np.percentile(upd_max, 75):.2f}); '
          f'spread median {np.median(upd_spread):.1%}')
    print(f'spread (max - min) / mean of busy: {(busy.max() - busy.min()) / busy.mean():.1%}; '
          f'max {busy.max():.2f} s; wall median {np.median([float(r["wall_s"]) for r in rows]):.2f} s\n')
