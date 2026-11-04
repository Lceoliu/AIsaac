"""Delayed-cue probe of a frozen policy on real play (rl/docs/PREDICTIVE_MEMORY_DESIGN.md section 2): a diagnostic, never a verdict.

Input: `eval_tok --dump-memory <dir>`, one npz per episode: per decision the temporal output o_t and what the record showed.
A cue is an observable property of the room the player was in d decisions ago, while the player is now in ANOTHER room (so the
current frame does not show it; rows still in that room are masked, and reported): `doors` (the room had 3 or more doors),
`big` (its width or height is beyond a 1x1 room's), `enemies` (a monster on screen at t - d), `special` (its type is not a
normal room's). A linear read-out (logistic regression) on o_t is fitted on 3/4 of the episodes and tested on the other 1/4, per
cue and delay. The same read-out on the current record's explicit fields is the control: what the current frame alone gives
(the minimap keeps some rooms' shapes and types, a door the way back). Reports balanced accuracy and AUC, the rows used, and the
masked share of the rows old enough for the delay.

usage: python abplus_delay_probe.py <dump dir> [--delays 8,32,64,128,255,512,1024] [--out result.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

CUES = ('doors', 'big', 'enemies', 'special')


def load(root):
    eps = []
    for p in sorted(Path(root).glob('ep-*.npz')):
        with np.load(p) as z:
            eps.append({k: z[k] for k in z.files})
    return eps


def cues_of(e):
    """Per decision of an episode: [T, len(CUES)] bool."""
    # (a 1x1 room's width and height in the record's scale: room[2] / 28 and room[3] / 16 of its grid)
    return np.stack([e['n_doors'] >= 3, (e['w'] > e['w_min']) | (e['h'] > e['h_min']), e['npc'] > 0, e['room_type'] > 1], 1)


def rows_for(eps, d):
    """(o, control, cue labels, episode index) of the rows with a delay d into another room; and the masked share."""
    X, C, Y, E = [], [], [], []
    old, kept = 0, 0
    for i, e in enumerate(eps):
        T = len(e['o'])
        if T <= d:
            continue
        rk = e['stage'].astype(np.int64) * 1000 + e['room']
        t = np.arange(d, T)
        old += len(t)
        keep = t[(rk[t] != rk[t - d])]
        kept += len(keep)
        if not len(keep):
            continue
        X.append(e['o'][keep])
        C.append(e['control'][keep])
        Y.append(e['cues'][keep - d])
        E.append(np.full(len(keep), i))
    masked = 1.0 - kept / max(old, 1)
    if not X:
        return None, masked
    return (np.concatenate(X).astype(np.float32), np.concatenate(C).astype(np.float32), np.concatenate(Y), np.concatenate(E)), masked


def auc(score, y):
    order = np.argsort(score)
    rank = np.empty(len(score))
    rank[order] = np.arange(len(score))
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float('nan')
    return float((rank[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def fit(x, y, xt, yt, device, steps=400, lr=0.05, wd=1e-4):
    """Balanced logistic regression; standardised on the train rows. (balanced accuracy, AUC) on the test rows."""
    mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True) + 1e-6
    X = torch.from_numpy((x - mu) / sd).to(device)
    Xt = torch.from_numpy((xt - mu) / sd).to(device)
    Y = torch.from_numpy(y.astype(np.float32)).to(device)
    w = torch.zeros(X.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    pw = torch.tensor([float((~y).sum()) / max(float(y.sum()), 1.0)], device=device)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, Y, pos_weight=pw).backward()
        opt.step()
    with torch.no_grad():
        s = (Xt @ w + b).cpu().numpy()
    tpr = float((s[yt] > 0).mean()) if yt.any() else float('nan')
    tnr = float((s[~yt] <= 0).mean()) if (~yt).any() else float('nan')
    return 0.5 * (tpr + tnr), auc(s, yt)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('root')
    a.add_argument('--delays', default='8,32,64,128,255,512,1024')
    a.add_argument('--out', default='')
    a.add_argument('--max-rows', type=int, default=200000)
    args = a.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eps = load(args.root)
    w_all = np.concatenate([e['w'] for e in eps])
    h_all = np.concatenate([e['h'] for e in eps])
    w_min, h_min = float(np.percentile(w_all, 5)), float(np.percentile(h_all, 5))   # (the 1x1 room's size, as the records hold it)
    for e in eps:
        e['w_min'], e['h_min'] = w_min, h_min
        e['cues'] = cues_of(e)
        e['control'] = e['state']
    n_rows = sum(len(e['o']) for e in eps)
    print(f'{len(eps)} episodes, {n_rows} decisions, device {device}', flush=True)
    rng = np.random.default_rng(1)
    is_test = rng.random(len(eps)) < 0.25
    out = dict(episodes=len(eps), decisions=n_rows, cues=list(CUES), delays={})
    for d in (int(v) for v in args.delays.split(',')):
        got, masked = rows_for(eps, d)
        rec = dict(masked_share=round(masked, 4))
        if got is None:
            out['delays'][d] = rec
            print(f'd={d}: no rows (masked {masked:.3f})', flush=True)
            continue
        X, C, Y, E = got
        if len(X) > args.max_rows:
            pick = rng.choice(len(X), args.max_rows, replace=False)
            X, C, Y, E = X[pick], C[pick], Y[pick], E[pick]
        te = is_test[E]
        rec['rows'] = int(len(X))
        for j, cue in enumerate(CUES):
            y = Y[:, j]
            if te.sum() < 50 or (~te).sum() < 50 or y[~te].mean() in (0.0, 1.0) or y[te].mean() in (0.0, 1.0):
                rec[cue] = None
                continue
            acc, a_ = fit(X[~te], y[~te], X[te], y[te], device)
            acc_c, a_c = fit(C[~te], y[~te], C[te], y[te], device)
            rec[cue] = dict(base_rate=round(float(y.mean()), 3), o_bal_acc=round(acc, 3), o_auc=round(a_, 3), control_bal_acc=round(acc_c, 3),
                            control_auc=round(a_c, 3))
        out['delays'][d] = rec
        print(f'd={d}', json.dumps(rec), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
