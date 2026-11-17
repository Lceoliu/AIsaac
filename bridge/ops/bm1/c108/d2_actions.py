"""C108 D2 (2026-11-15): what the agent's actions do. Read-only analysis of saved records, no game, no GPU.
  d2_actions.py table <out.json> <label>=<source> ...      action vs effect tables
  d2_actions.py draw <out dir> <label> <record dir> [k=12]  trajectory pictures + one HTML page
<source>: a training rollout dump (train_tok --dump-rollouts: rollouts/rollout-*.npz, its rows / pending / actions; the
production path) or an eval_tok --record directory (ep*.npz, the eval path).

Effect of a move: the player's displacement from record u to u+1 (ROW player x, y: the room-relative position, scaled
to pixels by the room's size in cells, player cols 26 / 27), as one of the 9 MOVES (abplus_geometry: 0 stop, 1 up,
2 up-right, ... 8 up-left; still when under 2 px). Effect of a shot: the player's tears (entity type 2) new at record
u+1 (age <= 4 frames), their direction by the larger velocity axis (1 up, 2 right, 3 down, 4 left; 0 none).
The action decided at record t is paired with the step u -> u+1 for u = t-1, t, t+1, t+2 (lag -1 .. +2): tok_record's
lag-1 protocol says the decided action runs during t+1 -> t+2 (lag +1). Pairs across a room change (another grid, a
jump over a quarter room) or an episode start are left out; forced pickup moves (pforce > 0) are counted apart.
Rollouts also give pending (the action under way at a record): share pending[t+1] == action[t] per head."""
import base64
import glob
import io
import json
import os
import sys

import numpy as np

MOVES = ((0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
LAGS = (-1, 0, 1, 2)
TEAR = 2


def octant(dx, dy, still=2.0):
    """MOVES index of a displacement (pixels; y down), 0 under `still` px."""
    out = np.zeros(len(dx), np.int64)
    mv = np.hypot(dx, dy) >= still
    ang = np.degrees(np.arctan2(dx, -dy)) % 360   # 0 = up, 90 = right
    k = (np.round(ang / 45).astype(np.int64) % 8) + 1
    out[mv] = k[mv]
    return out


def episodes_from_record(d):
    for f in sorted(glob.glob(os.path.join(d, 'ep*.npz'))):
        z = np.load(f, allow_pickle=True)
        n = z['n_ent'].astype(np.int64)   # (int16 in the file: its cumulative sum overflowed in a long episode)
        at = np.concatenate([[0], np.cumsum(n, dtype=np.int64)])
        E, I = z['ent'], z['ent_id']   # (once: every z[...] reads the whole array from the file again)
        ents = [(E[at[k]:at[k + 1]], I[at[k]:at[k + 1]]) for k in range(len(n))]
        act = z['act'].astype(np.int64)
        yield dict(name=os.path.basename(f), player=z['player'], act=act, pending=None, ents=ents,
                   room=z['grid_idx'].astype(np.int64), pforce=np.zeros(len(act), np.int64),
                   grids=z['grids'], grid_idx=z['grid_idx'], doors=z['doors'], n_doors=z['n_doors'],
                   events=z['events'], bombs=z['bombs'], done=z['done'],
                   meta=json.loads(str(z['meta'])) if 'meta' in z.files else {})


def episodes_from_rollouts(paths):
    for f in paths:
        z = np.load(f, allow_pickle=True)
        rows, acts, pend, length = z['rows'], z['actions'], z['pending'], z['length']
        for w in range(rows.shape[0]):
            r = rows[w, :int(length[w])]
            if not len(r):
                continue
            starts = list(np.flatnonzero(r['first'])) + [len(r)]
            if starts[0] != 0:
                starts = [0] + starts
            for a, b in zip(starts[:-1], starts[1:]):
                if b - a < 3:
                    continue
                rr = r[a:b]
                gk = {}
                room = np.array([gk.setdefault(rr['grid'][k].tobytes(), len(gk)) for k in range(len(rr))])
                ents = [(rr['ent'][k, :int(rr['n_ent'][k])], rr['ent_id'][k, :int(rr['n_ent'][k])])
                        for k in range(len(rr))]
                yield dict(name=f'{os.path.basename(f)}:w{w}:{a}', player=rr['player'],
                           act=acts[w, a:b].astype(np.int64), pending=pend[w, a:b].astype(np.int64), ents=ents,
                           room=room, pforce=rr['pforce'].astype(np.int64) if 'pforce' in rr.dtype.names else
                           np.zeros(b - a, np.int64))


def steps(ep):
    """Per record u: (move effect of u -> u+1, tear effect at u+1, valid) arrays of length T (the last invalid)."""
    p = ep['player'].astype(np.float64)
    T = len(p)
    wpx = np.maximum(p[:, 26] * 28 - 2, 1) * 40
    hpx = np.maximum(p[:, 27] * 16 - 2, 1) * 40
    dx = np.zeros(T); dy = np.zeros(T)
    dx[:-1] = (p[1:, 0] - p[:-1, 0]) * wpx[:-1]
    dy[:-1] = (p[1:, 1] - p[:-1, 1]) * hpx[:-1]
    valid = np.zeros(T, bool)
    valid[:-1] = (ep['room'][1:] == ep['room'][:-1]) & (np.abs(p[1:, 0] - p[:-1, 0]) < 0.25) & \
        (np.abs(p[1:, 1] - p[:-1, 1]) < 0.25)
    mv = octant(dx, dy)
    tear = np.zeros(T, np.int64)
    for u in range(T - 1):
        e, i = ep['ents'][u + 1]
        if not len(e):
            continue
        new = (i[:, 0] == TEAR) & (e[:, 15] * 90 <= 4.5)
        if new.any():
            vx, vy = e[new, 3].mean(), e[new, 4].mean()
            tear[u] = (2 if vx > 0 else 4) if abs(vx) > abs(vy) else (3 if vy > 0 else 1)
    return mv, tear, valid, dx, dy


def table(out, sources):
    res = {}
    for label, src in sources:
        if os.path.isdir(src) and glob.glob(os.path.join(src, 'ep*.npz')):
            eps, kind = episodes_from_record(src), 'eval record'
        else:
            paths = sorted(glob.glob(os.path.join(src, 'rollout-*.npz'))) if os.path.isdir(src) else [src]
            eps, kind = episodes_from_rollouts(paths), 'training rollout'
        conf_m = {L: np.zeros((9, 9), np.int64) for L in LAGS}
        conf_mf = {L: np.zeros((9, 9), np.int64) for L in LAGS}
        conf_s = {L: np.zeros((5, 5), np.int64) for L in LAGS}
        cos = {L: [] for L in LAGS}
        pend_ok = None
        n_ep = n_rec = 0
        for ep in eps:
            n_ep += 1
            mv, tear, valid, dx, dy = steps(ep)
            a = ep['act']
            T = len(a)
            n_rec += T
            if ep['pending'] is not None:
                pd = ep['pending']
                ok = (pd[1:] == a[:-1])
                pend_ok = (ok.sum(0), len(ok)) if pend_ok is None else (pend_ok[0] + ok.sum(0), pend_ok[1] + len(ok))
            for L in LAGS:
                for t in range(T):
                    u = t + L
                    if u < 0 or u >= T - 1 or not valid[u] or a[t, 0] < 0:
                        continue
                    m = int(a[t, 0])
                    if ep['pforce'][t] > 0:
                        conf_mf[L][m, mv[u]] += 1
                        continue
                    conf_m[L][m, mv[u]] += 1
                    if a[t, 1] >= 0:
                        conf_s[L][int(a[t, 1]), tear[u]] += 1
                    if m > 0 and mv[u] > 0:
                        v = np.array(MOVES[m], float); v /= np.linalg.norm(v)
                        d = np.array([dx[u], dy[u]]); d /= np.linalg.norm(d)
                        cos[L].append(float(v @ d))
        r = dict(kind=kind, source=src, episodes=n_ep, records=n_rec, lags={})
        if pend_ok is not None:
            r['pending_next_equals_action'] = dict(share=(pend_ok[0] / max(pend_ok[1], 1)).tolist(), n=int(pend_ok[1]))
        for L in LAGS:
            cm = conf_m[L]
            moving = cm[1:, :].sum()
            r['lags'][str(L)] = dict(
                move_pairs=int(cm.sum()), move_same_dir=float(np.trace(cm[1:, 1:]) / max(moving, 1)),
                move_still_when_moving=float(cm[1:, 0].sum() / max(moving, 1)),
                move_cos_mean=float(np.mean(cos[L])) if cos[L] else None,
                stop_moved=float(cm[0, 1:].sum() / max(cm[0].sum(), 1)),
                move_confusion=cm.tolist(), forced_confusion=conf_mf[L].tolist(),
                forced_same_dir=float(np.trace(conf_mf[L][1:, 1:]) / max(conf_mf[L][1:].sum(), 1)),
                shoot_confusion=conf_s[L].tolist(),
                shoot_tear_same_dir=float(np.trace(conf_s[L][1:, 1:]) / max(conf_s[L][1:].sum(), 1)),
                shoot_none_tears=float(conf_s[L][0, 1:].sum() / max(conf_s[L][0].sum(), 1)))
        res[label] = r
        print(label, kind, n_ep, 'episodes', n_rec, 'records', {L: (round(r['lags'][str(L)]['move_same_dir'], 3),
              r['lags'][str(L)]['move_cos_mean'] and round(r['lags'][str(L)]['move_cos_mean'], 3),
              round(r['lags'][str(L)]['shoot_tear_same_dir'], 3)) for L in LAGS},
              r.get('pending_next_equals_action'), flush=True)
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)


def draw(outdir, label, src, k=12):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    html = [f'<html><head><meta charset="utf-8"><title>C108 D2 {label}</title></head><body>',
            f'<h2>C108 D2: {label} -- eval episodes ({src})</h2>',
            '<p>Each panel is one room visit. Grey: walls / rocks (solid), black: pits, brown: destructible, red: '
            'hazards, orange: closed doors, green squares: open doors. The line is the player\'s path coloured by time '
            '(dark = early, yellow = late); arrows: the move decided at that record (every 6th); small ticks: shots '
            'decided (direction); red x: a bomb placed (the bomb count fell); blue star: an active item used; magenta '
            'diamond: a pill / card used. Positions are room-relative (player cols 0 / 1) mapped onto the room grid.</p>']
    eps = list(episodes_from_record(src))[:k]
    for ep in eps:
        p, a, room = ep['player'], ep['act'], ep['room']
        segs, s0 = [], 0
        for t in range(1, len(p) + 1):
            if t == len(p) or room[t] != room[t - 1]:
                segs.append((s0, t)); s0 = t
        segs = segs[:8]
        fig, axes = plt.subplots(1, len(segs), figsize=(4.2 * len(segs), 3.4), squeeze=False)
        T = len(p)
        for ax, (lo, hi) in zip(axes[0], segs):
            g = ep['grids'][ep['grid_idx'][lo]].astype(float)
            inside = g[0] > 0
            H = int(inside.any(1).sum()) or 16
            W = int(inside.any(0).sum()) or 28
            img = np.ones((16, 28, 3))
            img[inside] = (0.93, 0.93, 0.9)
            for ch, col in ((2, (0.45, 0.45, 0.45)), (3, (0.05, 0.05, 0.05)), (4, (0.6, 0.4, 0.2)),
                            (5, (0.85, 0.2, 0.2)), (6, (1.0, 0.6, 0.0))):
                img[g[ch] > 0] = col
            ax.imshow(img[:H, :W], extent=(0, W, H, 0), interpolation='nearest')
            xs = 1 + p[lo:hi, 0] * (W - 2)
            ys = 1 + p[lo:hi, 1] * (H - 2)
            tt = np.arange(lo, hi) / max(T - 1, 1)
            ax.scatter(xs, ys, c=tt, cmap='viridis', s=3, vmin=0, vmax=1, zorder=3)
            ax.plot(xs, ys, color='k', lw=0.3, alpha=0.4, zorder=2)
            nd = int(ep['n_doors'][lo])
            for j in range(nd):
                dr = ep['doors'][lo, j]
                xd = xs[0] + dr[0] * 200 / 40
                yd = ys[0] + dr[1] * 200 / 40
                ax.scatter([xd], [yd], marker='s', s=40, c='g' if dr[2] > 0.5 else 'orange', edgecolors='k', zorder=4)
            for t in range(lo, hi, 6):
                m = int(a[t, 0])
                if m > 0:
                    dx, dy = MOVES[m]
                    ax.annotate('', xy=(xs[t - lo] + 0.6 * dx, ys[t - lo] + 0.6 * dy), xytext=(xs[t - lo], ys[t - lo]),
                                arrowprops=dict(arrowstyle='->', color='royalblue', lw=0.7), zorder=5)
            sdir = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}
            for t in range(lo, hi, 3):
                s = int(a[t, 1])
                if s > 0:
                    dx, dy = sdir[s]
                    ax.plot([xs[t - lo], xs[t - lo] + 0.35 * dx], [ys[t - lo], ys[t - lo] + 0.35 * dy],
                            color='crimson', lw=0.5, alpha=0.6, zorder=4)
            b = ep['bombs'].astype(int)
            for t in range(max(lo, 1), hi):
                if b[t] < b[t - 1]:
                    ax.scatter([xs[t - lo]], [ys[t - lo]], marker='x', c='red', s=50, zorder=6)
                ev = int(ep['events'][t])
                if ev & 32:
                    ax.scatter([xs[t - lo]], [ys[t - lo]], marker='*', c='blue', s=70, zorder=6)
                if ev & 64:
                    ax.scatter([xs[t - lo]], [ys[t - lo]], marker='D', c='magenta', s=30, zorder=6)
            ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'records {lo}-{hi - 1}', fontsize=8)
        mt = ep['meta']
        use = (a[:, 3] == 1).mean() if a.shape[1] > 3 else float('nan')
        fig.suptitle(f"{label} {ep['name']}: {T} records, done {mt.get('done', '?')}, rooms cleared "
                     f"{mt.get('rooms', '?')}, rooms entered {mt.get('entered', '?')}, items {mt.get('items', '?')}; "
                     f"moves 0 (stop) {np.mean(a[:-1, 0] == 0):.2f}, shoot none {np.mean(a[:-1, 1] == 0):.2f}, "
                     f"item use decided {use:.2f}", fontsize=8)
        fig.tight_layout()
        png = os.path.join(outdir, f"{label}_{ep['name'].replace('.npz', '')}.png")
        fig.savefig(png, dpi=110)
        plt.close(fig)
        b64 = base64.b64encode(open(png, 'rb').read()).decode()
        html.append(f'<h3>{ep["name"]}</h3><img src="data:image/png;base64,{b64}" style="max-width:100%">')
    html.append('</body></html>')
    with open(os.path.join(outdir, f'{label}.html'), 'w') as f:
        f.write('\n'.join(html))
    print('drew', len(eps), 'episodes ->', outdir)


if __name__ == '__main__':
    if sys.argv[1] == 'table':
        table(sys.argv[2], [tuple(s.split('=', 1)) for s in sys.argv[3:]])
    else:
        draw(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 12)
