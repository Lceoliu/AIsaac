"""B42 steady summary: skip the first 40 updates with both teachers in the loss, then the next 40 updates."""
import csv, json, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1] + '/progress.csv')))
def f(r, k):
    v = r.get(k, '')
    try: return float(v)
    except ValueError: return float('nan')
def joint(r):
    return f(r, 'teach_step') > 0 and f(r, 'v2_step') > 0 and f(r, 'teach_loss') > 0
J = [i for i, r in enumerate(rows) if joint(r)]
out = dict(updates=len(rows), decisions=int(f(rows[-1], 'decisions')), game_hours=f(rows[-1], 'decisions') / 27000,
           first_joint=int(rows[J[0]]['update']) if J else None, joint_updates=len(J))
if len(J) >= 40:
    start = J[39] + 1
    win = rows[start:start + 40]
    if len(win) == 40:
        prev = f(rows[start - 1], 'decisions')
        dec = f(win[-1], 'decisions') - prev
        wall = sum(f(r, 'wall_s') for r in win)
        out.update(window=f"u{win[0]['update']}-u{win[-1]['update']}", window_decisions=int(dec),
                   window_game_hours=dec / 27000, window_wall_s=wall, x_real_time=dec / (wall * 7.5),
                   joint_in_window=sum(joint(r) for r in win))
        for k in ('collect_s', 'update_s', 'wall_s', 'infer_ms', 'gpu_reserved_gb', 'teach_step', 'v2_step', 'searches',
                  'block_gc_s', 'block_kv_rows', 'block_raw_rows', 'idle_share'):
            v = [f(r, k) for r in win if r.get(k, '') != '']
            if v: out['median_' + k] = st.median(v)
        if any('p95' in k for k in win[0]): pass
print(json.dumps(out, indent=1))
