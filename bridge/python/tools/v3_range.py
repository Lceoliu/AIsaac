"""The first v3 arm's pre-run check (seq-v1, the coordinator 2026-11-18): the first layer's inputs over real frames, and
an initialisation probe.

Reads train_tok --dump-rollouts output (<dir>/rollout-*.npz: the records the learner got, their actions under way;
rollout-*.pt: the learner's weights before that update -- the first one is the initialisation).

1. Range table (contract V4 / V8, light): every numeric input the frame encoder reads, decoded exactly as the actor and
   the learner decode it (v3_input.batch_from_rows), over the real records (padding excluded): per column, over all
   records and per room type (ROW room_type), mean / std / min / max / count, the share at the clamp (|x| = NUM_CLAMP)
   and outside schema_v3.NUM_RANGE where the schema states one. Flagged: std < 0.02 (a column that barely moves: a
   scale too large, or a constant) and |x| >= 8 anywhere (at the clamp: a scale too small). Also: non-finite wire floats,
   frames at the entity cap (CAP_ENTITY: the producer may have cut entities), inventory / slot rows cut at
   N_MAX_V3 / S_MAX_V3, and the workers' handed-back (reference encoder) frames from progress.csv.
2. Initialisation probe: a fresh model of the dump's configuration (TokPolicy(**config), torch seed --seed: the arm's
   initialisation), and for comparison the dump's own weights, on a sample of those records -- each token kind's input
   to the blocks (std, max |x|), the controlled body's and the global token's output, the logits per head (std,
   max |logit|, the entropy as a share of the uniform one), the values; anything non-finite.

   usage: python tools/v3_range.py <rollouts dir> [--progress progress.csv] [--out <stem>] [--probe-n 2048]
   writes <stem>.md (the tables) and <stem>.csv (every column x room type row)."""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from isaac_bridge import schema_v3 as V3          # noqa: E402
from isaac_bridge import v3_input as VI           # noqa: E402

STD_LOW, ABS_HIGH = 0.02, 8.0


def _names():
    """group -> (decoded key, column names, schema table or None)."""
    g = {}
    for t, cols in (('P', V3.P_COLS), ('E', V3.E_COLS), ('N', V3.N_COLS), ('S', V3.S_COLS)):
        g[f'{t}_num'] = ([cols[i][0] for i in VI.PLAN[t]['num']], t)
    g['P_off'] = (['off_x', 'off_y'], None)
    g['P_rel'] = (['rel_dx', 'rel_dy', 'rel_dist'], None)
    g['E_price'] = (['price_coins'], None)
    g['E_rel'] = (['rel_dx', 'rel_dy', 'rel_dist'], None)
    g['E_laser_rel'] = (['laser_dx', 'laser_dy', 'laser_dist'], None)
    g['room'] = (list(V3.ROOM_CHANNELS), None)
    g['local'] = (list(V3.ROOM_CHANNELS), None)
    g['room_rel'] = (['rel_dx', 'rel_dy', 'rel_dist'], None)
    g['map_bits'] = ([n for n, _ in V3.MAP_BITS[:4]], None)
    return g


GROUPS = _names()


class Acc:
    """Streaming per-column sums (float64) for one (group, room type)."""
    def __init__(self, k):
        self.n = np.zeros(k)
        self.s = np.zeros(k)
        self.ss = np.zeros(k)
        self.lo = np.full(k, np.inf)
        self.hi = np.full(k, -np.inf)
        self.clamp = np.zeros(k)
        self.out = np.zeros(k)

    def add(self, x, rng):
        """x [m, k] (the real rows only)."""
        if not len(x):
            return
        x = x.astype(np.float64)
        self.n += len(x)
        self.s += x.sum(0)
        self.ss += (x * x).sum(0)
        self.lo = np.minimum(self.lo, x.min(0))
        self.hi = np.maximum(self.hi, x.max(0))
        self.clamp += (np.abs(x) >= V3.NUM_CLAMP).sum(0)
        if rng is not None:
            # (the bounds rounded to float32 as the decoded values are: 15 / 28 or 0.99 on a bound is inside -- the
            #  first table counted float32 values at a bound as outside, the coordinator 2026-11-18)
            lo = np.array([r[0] if r else -np.inf for r in rng], np.float32).astype(np.float64)
            hi = np.array([r[1] if r else np.inf for r in rng], np.float32).astype(np.float64)
            self.out += ((x < lo) | (x > hi)).sum(0)

    def rows(self):
        n = np.maximum(self.n, 1)
        mean = self.s / n
        std = np.sqrt(np.maximum(self.ss / n - mean * mean, 0))
        return mean, std


def _real(key, d):
    """The decoded array of a group as [m, k] over its real rows (padding excluded), and each row's record index."""
    a = d[key].numpy()
    B = a.shape[0]
    if key.startswith(('P_', 'E_', 'N_', 'S_')):
        m = d[key[0] + '_mask'].numpy()
        if key == 'E_price':
            a = a.reshape(*a.shape[:2], -1)
        rec = np.broadcast_to(np.arange(B)[:, None], m.shape)
        return a[m].reshape(-1, a.shape[-1]), rec[m]
    if key == 'local':   # [B, P, 9, 9, RC]: the bodies with a position (local_pad False), every cell
        m = d['P_mask'].numpy() & ~d['local_pad'].numpy()
        x = a[m]                                            # [k, 9, 9, RC]
        rec = np.broadcast_to(np.arange(B)[:, None], m.shape)[m]
        cells = x.shape[1] * x.shape[2]
        return x.reshape(-1, a.shape[-1]), np.repeat(rec, cells)
    x = a.reshape(B, -1, a.shape[-1])                       # room [B, H, W, RC], room_rel [B, 28, 3], map_bits [B, 169, 4]
    return x.reshape(-1, a.shape[-1]), np.repeat(np.arange(B), x.shape[1])


def table(paths, progress=None, chunk=2048):
    import torch
    acc = {}                     # (group, room type or 'all') -> Acc
    rt_n = {}
    extra = dict(records=0, nonfinite=np.zeros(5, np.int64), e_cap=0, e_max=0, trunc_n=0, trunc_s=0, n_max=0, s_max=0)
    for p in paths:
        z = np.load(p, allow_pickle=False)
        rows, length = z['rows'], z['length']
        keep = np.arange(rows.shape[1])[None, :] < length[:, None]
        recs = rows[keep]
        for i in range(0, len(recs), chunk):
            r = np.ascontiguousarray(recs[i:i + chunk])
            frames = r['visible_frame'].view('<f4')
            counts, trunc = VI.check_batch(frames[:, :V3.HEADER], r['visible_schema'], r['visible_bytes'], VI.P_MAX,
                                           VI.N_MAX_V3, VI.S_MAX_V3)
            with torch.no_grad():
                d = VI.batch_from_rows(r, 'cpu')
            extra['records'] += len(r)
            extra['nonfinite'] += d['nonfinite'].numpy().sum(0)
            extra['e_cap'] += int((counts[:, 1] >= V3.CAP_ENTITY).sum())
            extra['e_max'] = max(extra['e_max'], int(counts[:, 1].max()))
            extra['n_max'] = max(extra['n_max'], int(counts[:, 2].max()))
            extra['s_max'] = max(extra['s_max'], int(counts[:, 3].max()))
            extra['trunc_n'] += trunc['n']
            extra['trunc_s'] += trunc['s']
            rtype = r['room_type'].astype(np.int64)
            for t in np.unique(rtype):
                rt_n[int(t)] = rt_n.get(int(t), 0) + int((rtype == t).sum())
            for g, (names, tab) in GROUPS.items():
                x, rec = _real(g, d)
                rng = [V3.NUM_RANGE[tab].get(n) for n in names] if tab else None
                acc.setdefault((g, 'all'), Acc(len(names))).add(x, rng)
                rr = rtype[rec]
                for t in np.unique(rr):
                    acc.setdefault((g, int(t)), Acc(len(names))).add(x[rr == t], rng)
    ref = None
    if progress and os.path.exists(progress):
        with open(progress) as f:
            pr = list(csv.DictReader(f))
        if pr and 'v3_reference' in pr[-1]:
            ref = (float(pr[-1].get('v3_native') or 0), float(pr[-1].get('v3_reference') or 0))
    return acc, rt_n, extra, ref


def probe(pt, paths, n=2048, seed=0, fresh=True):
    """The probe of a model on the first dump's first n records: fresh -- TokPolicy(**the dump's config) under torch seed
    `seed` (the initialisation); else the dump's weights."""
    import torch
    from isaac_bridge import tok_policy as TP
    ck = torch.load(pt, map_location='cpu', weights_only=False)
    torch.manual_seed(seed)
    m = TP.TokPolicy(**ck['config'])
    if not fresh:
        m.load_state_dict(ck['model'])
    m.eval()
    z = np.load(paths[0], allow_pickle=False)
    keep = np.arange(z['rows'].shape[1])[None, :] < z['length'][:, None]
    rows = np.ascontiguousarray(z['rows'][keep][:n])
    pend = np.ascontiguousarray(z['pending'][keep][:n])
    b = TP.to_batch(rows, pend, 'cpu')
    out = {}
    with torch.no_grad():
        v = VI.get_batch(b)
        x, pad, g = m.frame_v3.tokens(v, b['pending'], m.gtok.tok)
        kinds = [('P', v['P_mask'].shape[1]), ('L', v['P_mask'].shape[1]), ('E', v['E_mask'].shape[1]),
                 ('N', v['N_mask'].shape[1]), ('S', v['S_mask'].shape[1]), ('R', VI.R_ROWS * VI.R_COLS), ('M', 1),
                 ('G', 1)]
        at = 0
        for k, w in kinds:
            sl, real = x[:, at:at + w], ~pad[:, at:at + w]
            t_ = sl[real]
            out[f'tokens {k}'] = dict(n=int(real.sum()), std=float(t_.std()) if t_.numel() > 1 else float('nan'),
                                      absmax=float(t_.abs().max()) if t_.numel() else float('nan'))
            at += w
        y, _pad, g = m.frame_v3(v, b['pending'], m.gtok.tok)
        for k, t_ in (('out P0', y[:, 0]), ('out G', y[:, g])):
            out[k] = dict(n=int(t_.shape[0]), std=float(t_.std()), absmax=float(t_.abs().max()))
        logits, value, _danger, vg = m(b, need_vg=True)
        at = 0
        for h, w in enumerate(m.heads):
            lg = logits[:, at:at + w].float()
            p = torch.softmax(lg, -1)
            ent = float((-(p * torch.log(p.clamp_min(1e-30))).sum(-1)).mean())
            out[f'head {h} ({w})'] = dict(std=float(lg.std()), absmax=float(lg.abs().max()),
                                          entropy_share=ent / math.log(w))
            at += w
        out['value'] = dict(mean=float(value.float().mean()), std=float(value.float().std()))
        out['nonfinite'] = int((~torch.isfinite(logits)).sum() + (~torch.isfinite(value)).sum()
                               + (~torch.isfinite(y)).sum())
        out['records'] = int(len(rows))
        out['weights'] = f'fresh (seed {seed})' if fresh else f'the dump (update {int(ck.get("update", -1))})'
    return out


def _fmt(v):
    return 'nan' if not np.isfinite(v) else (f'{v:.4g}' if abs(v) >= 1e-3 or v == 0 else f'{v:.2e}')


def write(stem, acc, rt_n, extra, ref, pr):
    lines, flagged = [], []
    with open(stem + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'column', 'room_type', 'n', 'mean', 'std', 'min', 'max', 'at_clamp', 'outside_range', 'flag'])
        for (g, rt), a in sorted(acc.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            mean, std = a.rows()
            for j, name in enumerate(GROUPS[g][0]):
                if not a.n[j]:
                    continue
                fl = []
                if std[j] < STD_LOW:
                    fl.append('std<0.02')
                if max(abs(a.lo[j]), abs(a.hi[j])) >= ABS_HIGH:
                    fl.append('|x|>=8')
                w.writerow([g, name, rt, int(a.n[j]), mean[j], std[j], a.lo[j], a.hi[j], int(a.clamp[j]), int(a.out[j]),
                            ' '.join(fl)])
                if fl and rt == 'all':
                    flagged.append((g, name, fl))
    L = lines.append
    L(f'# v3 first-layer input ranges ({extra["records"]} records)')
    L('')
    L(f'- room types (ROW room_type: records): ' + ', '.join(f'{k}: {v}' for k, v in sorted(rt_n.items())))
    L(f'- non-finite wire floats (P / E / N / S / canvas+map+local): {extra["nonfinite"].tolist()}')
    L(f'- frames at the entity cap ({V3.CAP_ENTITY}): {extra["e_cap"]}; largest e {extra["e_max"]}')
    L(f'- rows cut: inventory {extra["trunc_n"]} frames (> {VI.N_MAX_V3}; largest n {extra["n_max"]}), slots '
      f'{extra["trunc_s"]} frames (> {VI.S_MAX_V3}; largest s {extra["s_max"]})')
    L(f'- the workers\' frames (progress.csv, last row): native / handed back (reference) {ref if ref else "n/a"}')
    L('')
    L(f'## Flagged columns (all records): std < {STD_LOW} or |x| >= {ABS_HIGH:g}')
    L('')
    L('| group | column | flags | mean | std | min | max | n |')
    L('|---|---|---|---|---|---|---|---|')
    for g, name, fl in flagged:
        a = acc[(g, 'all')]
        j = GROUPS[g][0].index(name)
        mean, std = a.rows()
        L(f'| {g} | {name} | {" ".join(fl)} | {_fmt(mean[j])} | {_fmt(std[j])} | {_fmt(a.lo[j])} | {_fmt(a.hi[j])} | {int(a.n[j])} |')
    L('')
    L('## Every column, all records (per room type: the .csv)')
    L('')
    L('| group | column | mean | std | min | max | at clamp | outside NUM_RANGE | n |')
    L('|---|---|---|---|---|---|---|---|---|')
    for g in GROUPS:
        a = acc.get((g, 'all'))
        if a is None:
            continue
        mean, std = a.rows()
        for j, name in enumerate(GROUPS[g][0]):
            if a.n[j]:
                L(f'| {g} | {name} | {_fmt(mean[j])} | {_fmt(std[j])} | {_fmt(a.lo[j])} | {_fmt(a.hi[j])} | '
                  f'{int(a.clamp[j])} | {int(a.out[j])} | {int(a.n[j])} |')
    for one in pr or ():
        L('')
        L(f'## Probe: {one.get("weights")} weights, {one.get("records")} records')
        L('')
        L('| item | values |')
        L('|---|---|')
        for k, v in one.items():
            if k not in ('weights', 'records'):
                L(f'| {k} | ' + (', '.join(f'{a} {_fmt(b) if isinstance(b, float) else b}' for a, b in v.items())
                                 if isinstance(v, dict) else str(v)) + ' |')
    with open(stem + '.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('rollouts')
    ap.add_argument('--progress', default=None)
    ap.add_argument('--out', default='v3_range')
    ap.add_argument('--probe-n', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    paths = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.npz')))
    if not paths:
        ap.error(f'no rollout-*.npz in {a.rollouts}')
    acc, rt_n, extra, ref = table(paths, a.progress)
    pts = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.pt')))
    pr = [probe(pts[0], paths, a.probe_n, a.seed, True), probe(pts[0], paths, a.probe_n, a.seed, False)]         if pts and a.probe_n else None
    write(a.out, acc, rt_n, extra, ref, pr)
    print(f'{a.out}.md: {extra["records"]} records, {sum(1 for k in acc if k[1] == "all")} groups')


if __name__ == '__main__':
    main()
