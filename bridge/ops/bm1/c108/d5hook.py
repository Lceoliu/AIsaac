"""C109 D5 (2026-11-15, throwaway: only in the tree copies maa1e98a-d5 / m1ae82d3-d5, never committed).
v2 (the coordinator, 11-15): 16 updates per configuration, pairwise Gram matrices, a second (odd / even) worker split,
per-worker row statistics, and a same-batch check of the donated-buffer switch.

ISAAC_D5_GRAD=1, with frozen weights (--lr 0): gradient consistency of the PPO update, hooked into every PPO part right
before its loss.backward(), first epoch only (a part whose decisions were seen already in this update is skipped).
The block tree's compiled pieces (flex attention, block_compile) donate their backward buffers, which forbids the
hook's retain_graph=True: with ISAAC_D5_GRAD=1 torch._functorch.config.donated_buffer is set False at import.

Per part, from the part's own tensors (the advantages exactly as the trainer normalised them):
  per-row policy surrogate at the update's first step (ratio to the update-start log-probability with --decoupled,
  importance weight exp(prox - behaviour) clamped at --decoupled-max-weight; forced rows contribute 0), weighted as the
  trainer weights the part's rows (pm / pn or 1 / k, times the part's share w);
  worker = rollout position // capacity (the rollout is worker-major); splits: A = workers 0..7 / B = 8..15, and
  E = even / O = odd workers;
  gradients (torch.autograd.grad, retain_graph) of: pg_A, pg_B, pg_E, pg_O; full_A, full_B, full_E, full_O
  (pg + vf_coef * vf - ent_coef * ent, the per-row PPO terms); and of each whole-part term as the trainer adds it:
  vf, ent, imit, glob (value_global), priv, aux, teacher (incl. danger), v2 (teacher v2 / CF), total (the part's loss).
Summed over the first epoch's parts per parameter group. At the update's end one JSON line in <out>/d5_grad.jsonl: per
group the norms of every term, cos of the two splits' halves (pg, full), cos with the previous update (pg, total); the
rows per half / per worker (all, PPO-trained = not forced) and per worker sum(rw), sum(rw |A|), sum((rw A)^2), mean
importance weight. <out>/d5_gram.json (rewritten after every update): per group the Gram matrices (dot products,
float64 from float32 vectors) of the updates' whole pg (= pg_A + pg_B) and total gradients, the update list. No raw
gradient vectors are kept on disk.

ISAAC_D5_CHECK=1 (block tree, without ISAAC_D5_GRAD): before the first PPO part's forward, the part's chunks go through
forward_bucketed four times at the same weights and seeds -- donated_buffer True, True, False, False (dynamo reset
before each, so every pass compiles anew under its setting) -- and the gradient of a fixed random projection of every
output (logits, value, danger, o, z) is taken; <out>/d5_check.json: per pass the projection's value, the gradient's
norm per group, and its cos / relative difference to pass 1's. The trainer then goes on with the production setting."""
import json
import os

import numpy as np

# C110 (2026-11-15, the coordinator) v3: ISAAC_D5_HEADS=1 -- the policy term split by action head instead (part_heads /
# _flush_heads below; the v2 terms are not computed then). Per part, ratio = 1 at --lr 0, so the surrogate's gradient
# is linear in the heads: g_h = grad sum_rows(rw * wgt * keep * (-A) * logp_h) with logp_h = log_softmax(split(logits,
# gate)[h])[action_h] (the trainer's own split and gate); the check terms: pg_trainer (the trainer's pg * w, as it enters
# the loss) and total (the part's loss). <out>/d5h_grad.jsonl per update: per group the norm of every head / ms
# (move + shoot) / sum (all heads) / pg_trainer / total and cos(sum, pg_trainer), cos(ms, pg_trainer); per head the
# rows whose "use" entry is gated off (share), mean p(use) over the PPO-trained rows and over all rows, rows.
# <out>/d5h_gram.json: per key (each head, ms, sum, pg_trainer, total) and group the Gram matrix over the updates.
HEADS_ON = os.environ.get('ISAAC_D5_HEADS', '0') == '1'
HEAD_NAMES = ('move', 'shoot', 'bomb', 'item', 'pill')
ON = os.environ.get('ISAAC_D5_GRAD', '0') == '1' or HEADS_ON
CHECK = os.environ.get('ISAAC_D5_CHECK', '0') == '1'
if ON:
    import torch._functorch.config as _fc
    _fc.donated_buffer = False
_s = dict(update=None, seen=None, acc=None, prev=None, out=None, groups=None, parts=0, ws=None, vecs=None, ups=[],
          checked=False)
TERMS = ('pg_A', 'pg_B', 'pg_E', 'pg_O', 'full_A', 'full_B', 'full_E', 'full_O', 'vf', 'ent', 'imit', 'glob', 'priv',
         'aux', 'teacher', 'v2', 'total')
NW = 16


def _groups(model, kind):
    g = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('pi.'):
            k = 'heads'
        elif kind == 'new':
            k = ('zr_proj' if name.startswith('block_to_spatial.') else
                 'readout4' if name.startswith('block_memory.resampler.') else
                 'writer' if name.startswith('block_memory.writer.') else
                 'reader' if name.startswith('block_memory.') else
                 'spatial' if name.startswith('structured_spatial.') else 'other')
        else:
            k = ('temporal' if name.startswith('tatt.') else
                 'other' if name.split('.')[0] in ('value', 'danger', 'gtok', 'norm') else 'trunk')
        g.setdefault(k, []).append((name, p))
    return g


def part(d, flat, cap, update, out, args, kind):
    import torch
    if not ON:
        return
    u = int(update)
    if _s['update'] is not None and u != _s['update']:
        _flush()
    if _s['update'] != u:
        _s.update(update=u, seen=set(), acc=None, out=out, parts=0, ws=None)
    model = d['cx'].model if kind == 'new' else d['learner_model']
    if _s['groups'] is None:
        _s['groups'] = _groups(model, kind)
    if kind == 'new':
        R = d['R']
        prt = d['sp_']['part']
        b_logp, pf, prox, ret = R['logp'], R.get('pf'), R.get('prox'), R['ret']
    else:
        prt = d['part']
        b_logp = d['b_logp'][prt]
        pf = d['b_pf'][prt] if d.get('b_pf') is not None else None
        prox = d['prox'][prt] if d.get('prox') is not None else None
        ret = d['b_ret'][prt]
    pidx = prt.detach().cpu().numpy() if torch.is_tensor(prt) else np.asarray(prt)
    pos = np.asarray(flat)[pidx]
    if any(int(p) in _s['seen'] for p in pos[:50]):   # (a later epoch: these decisions were taken already)
        return
    _s['seen'].update(int(p) for p in pos)
    k_ = int(d['k_'])
    logp, entropy, value, a = d['logp'], d['entropy'], d['value'], d['a']
    w, pm, pn = d['w'], d['pm'], d['pn']
    dev = logp.device
    forced = pf.bool() if pf is not None else torch.zeros(k_, dtype=torch.bool, device=dev)
    keep = (~forced).float()
    lp_d = logp.detach()
    b = torch.where(forced, lp_d, b_logp)
    if prox is not None:
        px = torch.where(forced, lp_d, prox)
        ratio = torch.exp(logp - px)
        wgt = torch.exp(px - b).clamp(max=args.decoupled_max_weight)
    else:
        ratio = torch.exp(logp - b)
        wgt = torch.ones_like(ratio)
    if HEADS_ON:
        return _part_heads(d, kind, model, pos, cap, k_, a, w, pm, pn, keep, wgt, dev)
    c = args.clip
    per_pg = -wgt * torch.min(ratio * a, torch.clamp(ratio, 1 - c, 1 + c) * a)
    per_full = per_pg + args.vf_coef * 0.5 * (value - ret) ** 2 - args.ent_coef * entropy.sum(-1) * keep
    rw = (pm / pn if pm is not None else torch.full((k_,), 1.0 / k_, device=dev)) * w
    wk = pos // cap
    wkt = torch.as_tensor(wk, device=dev, dtype=torch.long)
    sel = dict(A=wkt < NW // 2, B=wkt >= NW // 2, E=wkt % 2 == 0, O=wkt % 2 == 1)
    T = {}
    for h, m in sel.items():
        T['pg_' + h] = (per_pg * rw * m.float()).sum()
        T['full_' + h] = (per_full * rw * m.float()).sum()
    vc, ec = args.vf_coef, args.ent_coef
    if d.get('vf') is not None:
        T['vf'] = vc * d['vf'] * w
    if d.get('ent') is not None:
        T['ent'] = -ec * d['ent'] * w
    if d.get('imit') is not None:
        T['imit'] = args.pickup_force_imitation * d['imit'] * w
    if d.get('vgf') is not None:
        T['glob'] = args.vf_global_coef * d['vgf'] * w
    if d.get('pl_v') is not None:
        T['priv'] = (vc * d['pl_v'] + args.vf_global_coef * d['pl_g']) * w
    if d.get('aloss') is not None:
        T['aux'] = args.aux_coef * d['aloss']
    if d.get('tloss') is not None and d.get('merged'):
        T['teacher'] = d['tloss']
    if d.get('vloss') is not None and d.get('vmerged'):
        T['v2'] = d['vloss']
    T['total'] = d['loss']
    # per-worker row statistics (all rows / PPO-trained rows, the pg weights)
    with torch.no_grad():
        ad = a.detach().float()
        rwd = rw.detach().float() if torch.is_tensor(rw) else torch.full((k_,), float(rw), device=dev)
        kp = keep.float()
        cols = dict(rows=torch.ones_like(ad), free=kp, sum_rw=rwd * kp, sum_rw_abs_a=(rwd * ad.abs()) * kp,
                    sum_rw2_a2=(rwd * ad) ** 2 * kp, sum_wgt=wgt.detach().float() * kp,
                    sum_pg=(per_pg.detach().float() * rwd))
        ws = {k: torch.bincount(wkt, weights=v, minlength=NW).double().cpu().numpy() for k, v in cols.items()}
        if _s['ws'] is None:
            _s['ws'] = ws
        else:
            for k in ws:
                n = max(len(_s['ws'][k]), len(ws[k]))
                _s['ws'][k] = np.pad(_s['ws'][k], (0, n - len(_s['ws'][k]))) + np.pad(ws[k], (0, n - len(ws[k])))
    G = _s['groups']
    params = [p for gl in G.values() for _, p in gl]
    if _s['acc'] is None:
        _s['acc'] = {t: {g: torch.zeros(sum(p.numel() for _, p in gl), device=dev, dtype=torch.float32)
                         for g, gl in G.items()} for t in TERMS}
    for t, val in T.items():
        if not torch.is_tensor(val) or not val.requires_grad:
            continue
        grads = torch.autograd.grad(val, params, retain_graph=True, allow_unused=True)
        it = iter(grads)
        for g, gl in G.items():
            parts = []
            for _, p in gl:
                gr = next(it)
                parts.append((gr if gr is not None else torch.zeros_like(p)).reshape(-1).float().to(dev))
            _s['acc'][t][g] += torch.cat(parts)
    _s['parts'] += 1


def _part_heads(d, kind, model, pos, cap, k_, a, w, pm, pn, keep, wgt, dev):
    """ISAAC_D5_HEADS: one PPO part's policy term per head (module comment at the top)."""
    import torch
    import torch.nn.functional as F
    logits = d['logits_'][:k_]
    if kind == 'new':
        gate, act = d['R']['gate'][:k_], d['R']['act']
    else:
        gate, act = model.gate(d['mb'])[:k_], d['b_act'][d['part']]
    hs = model.split(logits.float(), gate)
    names = [HEAD_NAMES[i] if i < len(HEAD_NAMES) else f'h{i}' for i in range(len(hs))]
    rw = (pm / pn if pm is not None else torch.full((k_,), 1.0 / k_, device=dev)) * w
    coef = (rw * wgt * keep * a).detach().float()
    T = {}
    st = _s.setdefault('hst', {})
    g2 = gate if gate.dim() == 2 else (gate > 0)[:, None].expand(-1, 3)
    kp = keep.detach().float()
    for i, (nm, h) in enumerate(zip(names, hs)):
        logs = F.log_softmax(h, -1)
        lp = logs.gather(-1, act[:, i:i + 1].long())[:, 0]
        T['pg_' + nm] = -(coef * lp).sum()
        s_ = st.setdefault(nm, dict(rows=0.0, free=0.0, gated_off=0.0, p_use_all=0.0, p_use_free=0.0))
        s_['rows'] += float(k_)
        s_['free'] += float(kp.sum())
        if i >= 2 and h.shape[-1] >= 2:
            with torch.no_grad():
                pu = logs[:, 1].exp().float()
                s_['gated_off'] += float((~g2[:, i - 2].bool()).float().sum()) if i - 2 < g2.shape[1] else 0.0
                s_['p_use_all'] += float(pu.sum())
                s_['p_use_free'] += float((pu * kp).sum())
    if d.get('pg') is not None:
        T['pg_trainer'] = d['pg'] * w
    T['total'] = d['loss']
    _s['hnames'] = names
    G = _s['groups']
    params = [p for gl in G.values() for _, p in gl]
    if _s['acc'] is None:
        _s['acc'] = {}
    for t, val in T.items():
        if not torch.is_tensor(val) or not val.requires_grad:
            continue
        grads = torch.autograd.grad(val, params, retain_graph=True, allow_unused=True)
        it = iter(grads)
        acc = _s['acc'].setdefault(t, {g: torch.zeros(sum(p.numel() for _, p in gl), device=dev, dtype=torch.float32)
                                       for g, gl in G.items()})
        for g, gl in G.items():
            parts = []
            for _, p in gl:
                gr = next(it)
                parts.append((gr if gr is not None else torch.zeros_like(p)).reshape(-1).float().to(dev))
            acc[g] += torch.cat(parts)
    _s['parts'] += 1


def _flush_heads():
    import torch
    acc = _s['acc']
    if acc is None or _s['out'] is None:
        return
    names = _s.get('hnames') or []
    G = list(next(iter(acc.values())))
    shared = [g for g in G if g not in ('heads', 'other')]
    vec = {t: dict(acc[t]) for t in acc}
    hk = ['pg_' + n for n in names if 'pg_' + n in vec]
    vec['ms'] = {g: sum(vec[k][g] for k in ('pg_move', 'pg_shoot') if k in vec) for g in G}
    vec['sum'] = {g: sum(vec[k][g] for k in hk) for g in G}
    for t in vec:
        vec[t]['shared'] = torch.cat([vec[t][g] for g in shared])
        vec[t]['all'] = torch.cat([vec[t][g] for g in G])
    keys = hk + ['ms', 'sum'] + [k for k in ('pg_trainer', 'total') if k in vec]
    rec = dict(update=_s['update'], parts=_s['parts'], decisions=len(_s['seen']), heads=names,
               head_stats=_s.get('hst', {}), groups={})
    for g in G + ['shared', 'all']:
        r = dict(n_params=int(vec['sum'][g].numel()), norm={k: float(vec[k][g].norm()) for k in keys})
        if 'pg_trainer' in vec:
            r['cos_sum_pg_trainer'] = _cos(vec['sum'][g], vec['pg_trainer'][g])
            r['rel_diff_sum_pg_trainer'] = (float((vec['sum'][g] - vec['pg_trainer'][g]).norm() /
                                                  vec['pg_trainer'][g].norm()) if float(vec['pg_trainer'][g].norm()) > 0
                                            else None)
            r['cos_ms_pg_trainer'] = _cos(vec['ms'][g], vec['pg_trainer'][g])
        r['cos_ms_rest'] = _cos(vec['ms'][g], vec['sum'][g] - vec['ms'][g])
        rec['groups'][g] = r
    out = str(_s['out'])
    with open(os.path.join(out, 'd5h_grad.jsonl'), 'a') as f:
        f.write(json.dumps(rec) + '\n')
    if _s['vecs'] is None:
        _s['vecs'] = {k: {g: [] for g in G} for k in keys}
        _s['gram'] = {k: {g: [] for g in G} for k in keys}
    for k in keys:
        for g in G:
            cur = vec[k][g].cpu()
            vs = _s['vecs'][k][g] + [cur]
            row = [float(torch.dot(cur.double(), v.double())) for v in vs]
            gm = _s['gram'][k][g]
            for i, x in enumerate(row[:-1]):
                gm[i].append(x)
            gm.append(row)
            _s['vecs'][k][g] = vs
    _s['ups'].append(_s['update'])
    tmp = os.path.join(out, 'd5h_gram.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(dict(updates=_s['ups'], groups=G, shared=shared, keys=keys,
                       n_params={g: int(vec['sum'][g].numel()) for g in G}, gram=_s['gram']), f)
    os.replace(tmp, os.path.join(out, 'd5h_gram.json'))
    _s['acc'] = None
    _s['hst'] = {}


def _cos(x, y):
    import torch
    nx, ny = float(x.norm()), float(y.norm())
    return float(torch.dot(x, y) / (nx * ny)) if nx > 0 and ny > 0 else None


def _flush():
    import torch
    if HEADS_ON:
        return _flush_heads()
    acc = _s['acc']
    if acc is None or _s['out'] is None:
        return
    G = list(acc['pg_A'])
    shared = [g for g in G if g not in ('heads', 'other')]
    vec = {t: {g: acc[t][g] for g in G} for t in TERMS}
    for t in TERMS:
        vec[t]['shared'] = torch.cat([acc[t][g] for g in shared])
        vec[t]['all'] = torch.cat([acc[t][g] for g in G])
    pg = {g: vec['pg_A'][g] + vec['pg_B'][g] for g in vec['pg_A']}
    ws = {k: [float(x) for x in v] for k, v in (_s['ws'] or {}).items()}
    nw = len(ws.get('rows', []))
    half = {}
    for h, f in (('A', lambda i: i < NW // 2), ('B', lambda i: i >= NW // 2), ('E', lambda i: i % 2 == 0),
                 ('O', lambda i: i % 2 == 1)):
        half[h] = dict(rows=sum(ws['rows'][i] for i in range(nw) if f(i)),
                       free=sum(ws['free'][i] for i in range(nw) if f(i)))
    rec = dict(update=_s['update'], parts=_s['parts'], decisions=len(_s['seen']), halves=half, workers=ws, groups={})
    for g in vec['pg_A']:
        r = dict(n_params=int(vec['pg_A'][g].numel()),
                 cos_pg_halves=_cos(vec['pg_A'][g], vec['pg_B'][g]),
                 cos_full_halves=_cos(vec['full_A'][g], vec['full_B'][g]),
                 cos_pg_eo=_cos(vec['pg_E'][g], vec['pg_O'][g]),
                 cos_full_eo=_cos(vec['full_E'][g], vec['full_O'][g]),
                 norm={t: float(vec[t][g].norm()) for t in TERMS}, norm_pg=float(pg[g].norm()))
        if _s['prev'] is not None:
            r['cos_pg_prev'] = _cos(pg[g], _s['prev']['pg'][g])
            r['cos_total_prev'] = _cos(vec['total'][g], _s['prev']['total'][g])
        r['cos_pg_total'] = _cos(pg[g], vec['total'][g])
        rec['groups'][g] = r
    out = str(_s['out'])
    with open(os.path.join(out, 'd5_grad.jsonl'), 'a') as f:
        f.write(json.dumps(rec) + '\n')
    # Gram matrices over the updates (per group; float32 vectors on the host, float64 dot products)
    cur = dict(pg={g: pg[g].cpu() for g in G}, total={g: vec['total'][g].cpu() for g in G})
    if _s['vecs'] is None:
        _s['vecs'] = dict(pg={g: [] for g in G}, total={g: [] for g in G})
        _s['gram'] = dict(pg={g: [] for g in G}, total={g: [] for g in G})
    for t in ('pg', 'total'):
        for g in G:
            vs = _s['vecs'][t][g] + [cur[t][g]]
            row = [float(torch.dot(cur[t][g].double(), v.double())) for v in vs]
            gm = _s['gram'][t][g]
            for i, x in enumerate(row[:-1]):
                gm[i].append(x)
            gm.append(row)
            _s['vecs'][t][g] = vs
    _s['ups'].append(_s['update'])
    tmp = os.path.join(out, 'd5_gram.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(dict(updates=_s['ups'], groups=G, shared=shared,
                       n_params={g: int(vec['pg_A'][g].numel()) for g in G}, gram=_s['gram']), f)
    os.replace(tmp, os.path.join(out, 'd5_gram.json'))
    _s['prev'] = dict(pg=pg, total=vec['total'])
    _s['acc'] = None


def check(model, fwd, chunks, max_rows, update, history_mode, out):
    """ISAAC_D5_CHECK: the donated-buffer switch on one batch (module docstring); once per process."""
    import random
    import torch
    import torch._dynamo
    import torch._functorch.config as fc
    if not CHECK or _s['checked']:
        return
    _s['checked'] = True
    G = _groups(model, 'new')
    params = [p for gl in G.values() for _, p in gl]
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    np_rng, py_rng = np.random.get_state(), random.getstate()
    old = fc.donated_buffer
    res, proj, g1 = [], None, None
    for i, don in enumerate((True, True, False, False)):
        torch._dynamo.reset()
        fc.donated_buffer = don
        torch.manual_seed(1109)
        np.random.seed(1109)
        random.seed(1109)
        o, ctx = fwd(model, chunks, max_rows=max_rows, version=update, history_mode=history_mode)
        ts = [x for x in list(o) + [ctx['o_seq']] + list(ctx['z_out'])
              if torch.is_tensor(x) and x.is_floating_point() and x.requires_grad]
        if proj is None:
            gen = torch.Generator(device=ts[0].device).manual_seed(5)
            proj = [torch.randn(x.shape, generator=gen, device=x.device, dtype=torch.float32) for x in ts]
        s = sum((x.float() * r).sum() for x, r in zip(ts, proj))
        grads = torch.autograd.grad(s, params, allow_unused=True)
        it = iter(grads)
        per, flat_ = {}, []
        for g, gl in G.items():
            v = torch.cat([(gr if gr is not None else torch.zeros_like(p)).reshape(-1).float()
                           for gr, (_, p) in zip([next(it) for _ in gl], gl)])
            per[g] = float(v.norm())
            flat_.append(v)
        v = torch.cat(flat_)
        r = dict(donated_buffer=don, s=float(s), shapes=[list(x.shape) for x in ts], norm=per, norm_all=float(v.norm()))
        if g1 is None:
            g1 = v
        else:
            r['cos_to_pass1'] = _cos(v, g1)
            r['rel_diff_to_pass1'] = float((v - g1).norm() / g1.norm())
            r['max_abs_diff_to_pass1'] = float((v - g1).abs().max())
        res.append(r)
        del o, ctx, ts, s, grads
    torch._dynamo.reset()
    fc.donated_buffer = old
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    np.random.set_state(np_rng)
    random.setstate(py_rng)
    with open(os.path.join(str(out), 'd5_check.json'), 'w') as f:
        json.dump(dict(update=int(update), chunks=len(chunks), max_rows=max_rows, history_mode=history_mode,
                       passes=res), f, indent=1)


import atexit  # noqa: E402
atexit.register(_flush)
