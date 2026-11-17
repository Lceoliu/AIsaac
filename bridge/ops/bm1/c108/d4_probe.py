"""C108 D4b: linear probes on frozen representations (d4hook dumps), numpy only (no sklearn / scipy in the venv).
  d4_probe.py <out.json> <label>=<dump dir> ...
Per dump dir (its d4_u*.npz together): the decisions with every representation (block: the re-encoded subsample;
old: up to N_MAX (12000) decisions drawn at random); train / test by episode ((worker, episode) hashed, 80 / 20), the
regularisation picked on a validation fifth of the train episodes. Features standardised on the train part.
Probes and their trivial baselines:
  px, py; door_dx, door_dy (rows with a door): ridge, test R^2 (baseline 0: the train mean);
  door_dir, npc_dir (rows with a door / an NPC; 8 classes): multinomial logistic, test accuracy (baseline: the train
  majority class's share of the test rows);
  room_next (rows with a next record): logistic, test AUC and balanced accuracy (baseline 0.5), with the positives.
Representations: block -- emb_body, emb_local, emb_room (28 tokens flattened), emb_all (body + local + map + entity
mean + room), spat_body, spat_global, z (4 x 256), o, zr; old -- emb_player, doors_raw, emb_all (player + raw
doors), zr."""
import glob
import json
import os
import sys

import numpy as np

N_MAX = 12000
LAB = ('px', 'py', 'door_dx', 'door_dy', 'door_dir', 'npc_dx', 'npc_dy', 'npc_dir', 'room_next', 'has_next',
       'episode', 'worker')


def load(d):
    fs = sorted(glob.glob(os.path.join(d, 'd4_u*.npz')))
    parts = []
    for f in fs:
        z = np.load(f, allow_pickle=False)
        kind = str(z['kind'])
        if kind == 'block':
            sub = z['sub']
            rep = dict(emb_body=z['emb_body'], emb_local=z['emb_local'], emb_room=z['emb_room'].reshape(len(sub), -1),
                       spat_body=z['spat_body'], spat_global=z['spat_global'], z=z['z'].reshape(len(sub), -1),
                       o=z['o'][sub], zr=z['zr'][sub])
            rep['emb_all'] = np.concatenate([z['emb_body'], z['emb_local'], z['emb_map'], z['emb_entmean'],
                                             rep['emb_room']], 1)
            lab = {k: z[k][sub] for k in LAB}
            full = dict(zr_all=z['zr'], o_all=z['o']), {k: z[k] for k in LAB}
        else:
            n = len(z['zr'])
            rep = dict(emb_player=z['emb_player'], doors_raw=z['doors_raw'].reshape(n, -1),
                       zr=z['zr'])
            rep['emb_all'] = np.concatenate([z['emb_player'], rep['doors_raw'].astype(np.float16)], 1)
            lab = {k: z[k] for k in LAB}
            full = None
        parts.append((kind, rep, lab, os.path.basename(f), full))
    kind = parts[0][0]
    rep = {k: np.concatenate([p[1][k] for p in parts]).astype(np.float32) for k in parts[0][1]}
    lab = {k: np.concatenate([p[2][k] for p in parts]) for k in LAB}
    n = len(lab['px'])
    if n > N_MAX:
        keep = np.sort(np.random.default_rng(0).choice(n, N_MAX, replace=False))
        rep = {k: v[keep] for k, v in rep.items()}
        lab = {k: v[keep] for k, v in lab.items()}
    full = None
    if parts[0][4] is not None:   # (block: zr / o of every decision, for the rare room_next positives)
        frep = {k: np.concatenate([p[4][0][k] for p in parts]).astype(np.float32) for k in parts[0][4][0]}
        flab = {k: np.concatenate([p[4][1][k] for p in parts]) for k in LAB}
        full = (frep, flab)
    return kind, rep, lab, [p[3] for p in parts], full


def split(lab):
    key = (lab['worker'].astype(np.int64) * 1000003 + lab['episode'].astype(np.int64)) % 1000
    h = (key * 2654435761) % 1000
    test = h < 200
    val = (~test) & (h < 360)
    return ~test & ~val, val, test


def std(X, tr):
    m, s = X[tr].mean(0), X[tr].std(0) + 1e-6
    return (X - m) / s


def ridge(X, y, tr, va, te):
    """Ridge regression (float64, torch: GPU when there is one), lambda picked on the validation episodes."""
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xd = torch.as_tensor(X, dtype=torch.float64, device=dev)
    yd = torch.as_tensor(y, dtype=torch.float64, device=dev)
    trd, vad, ted = (torch.as_tensor(m, device=dev) for m in (tr, va, te))
    Xt, yt = Xd[trd], yd[trd]
    mu = yt.mean()
    G = Xt.T @ Xt
    bvec = Xt.T @ (yt - mu)
    eye = torch.eye(len(G), dtype=torch.float64, device=dev)
    best = None
    for lam in (1e-1, 1, 10, 1e2, 1e3, 1e4):
        w = torch.linalg.solve(G + lam * len(Xt) / 1000 * eye, bvec)
        yv = yd[vad]
        r2v = float(1 - ((Xd[vad] @ w + mu - yv) ** 2).mean() / max(float(yv.var()), 1e-12))
        if best is None or r2v > best[0]:
            best = (r2v, lam, w)
    _, lam, w = best
    ye = yd[ted]
    pred = Xd[ted] @ w + mu
    return dict(r2=float(1 - ((pred - ye) ** 2).mean() / max(float(ye.var()), 1e-12)), lam=lam, n_train=int(tr.sum()),
                n_test=int(te.sum()))


def softmax_fit(X, y, k, l2, iters=300, lr=0.05, weight=None):
    """Multinomial logistic regression with L2, full-batch Adam, in torch (GPU when there is one: the venv's numpy runs
    on one thread in the container). weight: per-class weights (balanced for room_next)."""
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.long, device=dev)
    torch.manual_seed(0)
    W = torch.zeros(X.shape[1], k, device=dev, requires_grad=True)
    b = torch.zeros(k, device=dev, requires_grad=True)
    cw = None if weight is None else torch.as_tensor(weight, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt, weight=cw) + 0.5 * l2 * (W * W).sum()
        loss.backward()
        opt.step()
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def classify(X, y, k, tr, va, te):
    best = None
    cnt = np.bincount(y[tr], minlength=k).astype(np.float64)
    weight = (cnt.sum() / np.maximum(cnt, 1) / k) if k == 2 else None   # (room_next: balanced classes)
    for l2 in (1e-4, 1e-3, 1e-2, 1e-1):
        W, b = softmax_fit(X[tr], y[tr], k, l2, weight=weight)
        pv = (X[va] @ W + b).argmax(1)
        acc = (float(np.mean([(pv[y[va] == c] == c).mean() for c in range(k) if (y[va] == c).any()])) if k == 2
               else float((pv == y[va]).mean())) if va.any() else 0.0
        if best is None or acc > best[0]:
            best = (acc, l2, W, b)
    _, l2, W, b = best
    s = X[te] @ W + b
    pred = s.argmax(1)
    major = np.bincount(y[tr], minlength=k).argmax()
    out = dict(acc=float((pred == y[te]).mean()), base=float((y[te] == major).mean()), l2=l2, n_train=int(tr.sum()),
               n_test=int(te.sum()))
    if k == 2:
        sc = s[:, 1] - s[:, 0]
        pos, neg = sc[y[te] == 1], sc[y[te] == 0]
        if len(pos) and len(neg):
            out['auc'] = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
        out['bal_acc'] = float(np.mean([(pred[y[te] == c] == c).mean() for c in (0, 1) if (y[te] == c).any()]))
        out['positives_train'], out['positives_test'] = int(y[tr].sum()), int(y[te].sum())
    return out


def probe(kind, rep, lab):
    tr, va, te = split(lab)
    res = {}
    has_door = ~np.isnan(lab['door_dx'])
    has_npc = ~np.isnan(lab['npc_dx'])
    nx = lab['has_next'] > 0
    for name, X in rep.items():
        X = std(X, tr)
        r = {}
        for t in ('px', 'py'):
            r[t] = ridge(X, lab[t].astype(np.float64), tr, va, te)
        for t in ('door_dx', 'door_dy'):
            m = has_door
            r[t] = ridge(X[m], lab[t][m].astype(np.float64), tr[m], va[m], te[m])
        m = has_door & (lab['door_dir'] > 0)
        r['door_dir'] = classify(X[m], lab['door_dir'][m] - 1, 8, tr[m], va[m], te[m])
        m = has_npc & (lab['npc_dir'] > 0)
        r['npc_dir'] = classify(X[m], lab['npc_dir'][m] - 1, 8, tr[m], va[m], te[m])
        r['room_next'] = classify(X[nx], lab['room_next'][nx].astype(np.int64), 2, tr[nx], va[nx], te[nx])
        res[name] = r
        print(name, X.shape[1], {k: round(v.get('r2', v.get('acc', 0)), 3) for k, v in r.items()},
              'room_next auc', round(r['room_next'].get('auc', float('nan')), 3), flush=True)
    return res


if __name__ == '__main__':
    out = {}
    for spec in sys.argv[2:]:
        label, d = spec.split('=', 1)
        kind, rep, lab, files, full = load(d)
        print('==', label, kind, len(lab['px']), 'decisions', files, 'room_next positives', int(lab['room_next'].sum()),
              flush=True)
        out[label] = dict(kind=kind, files=files, n=len(lab['px']), room_next_pos=int(lab['room_next'].sum()),
                          probes=probe(kind, rep, lab))
        if full is not None:
            print('== ', label, 'every decision (zr, o):', len(full[1]['px']), 'room_next positives',
                  int(full[1]['room_next'].sum()), flush=True)
            out[label]['n_all'] = len(full[1]['px'])
            out[label]['room_next_pos_all'] = int(full[1]['room_next'].sum())
            out[label]['probes_all'] = probe(kind, full[0], full[1])
        with open(sys.argv[1], 'w') as f:
            json.dump(out, f, indent=1)
