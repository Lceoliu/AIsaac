"""C108 D4b (2026-11-15, throwaway: only in the tree copies maa1e98a-d4 / m1ae82d3-d4, never committed).
ISAAC_D4_DUMP=1: with frozen weights (--lr 0), dump per decision the representations at several depths and the
labels the probes need, into <out>/d4_u<update>.npz (one file per update, the first ISAAC_D4_UPDATES (2) updates).

Block tree (new_group / new_prox, hooked into the learner's start-of-update pass, every decision of the rollout once):
  zr   [n, 192]  the policy heads' input (block_to_spatial(o)); o [n, 256] the Reader's pooled output (aux heads' input)
  on a subsample of ISAAC_D4_SUB (6000) decisions, re-encoded from their frames with the same weights, fp32:
  emb_body / emb_local / emb_map / emb_entmean [m, 192], emb_room [m, 28, 192]: StructuredSpatial's tokens right after
    the embedding (blocks skipped: the trunk's input); spat_body / spat_global [m, 192]: the same tokens after the
    trunk; z [m, 4, 256]: the 4-query readout (the Reader's per-step input).
Old tree (old_part, hooked into every PPO part; the first occurrence of each decision kept):
  zr   [n, 192]  the policy head's input (TokPolicy.pi's input, after the temporal attention)
  emb_player [n, 192] the player token's embedding (TokPolicy.player MLP: player fields, pending one-hot, the 9x9
    egocentric patch CNN); raw doors [n, 8, 7] (the door tokens' input). (The room CNN token is not taken: the
    learner's fast path runs it on the batch's distinct grids only.)
Labels for every dumped decision, from its record and the worker's next record (legacy ROW fields, both trees):
  px, py (room-relative 0..1), door (dx, dy) px to the nearest door (nan: none), its 8-way direction (MOVES order,
  1 up .. 8 up-left), npc (dx, dy) px to the nearest visible NPC and its direction, room_next (the next record of the
  same episode is in another room: minimap channel 3 moved or EV_NEW_ROOM / EV_EXIT), episode id, worker, step."""
import os

import numpy as np

ON = os.environ.get('ISAAC_D4_DUMP', '0') == '1'
UPDATES = int(os.environ.get('ISAAC_D4_UPDATES', '2'))
SUB = int(os.environ.get('ISAAC_D4_SUB', '6000'))
_s = dict(groups=[], n=0, old=None, hooks=False, cap={})


def _dir8(dx, dy):
    ang = np.degrees(np.arctan2(dx, -dy)) % 360
    return (np.round(ang / 45).astype(np.int64) % 8) + 1


def labels(rows_flat, nxt_flat, has_next):
    """rows_flat: the decisions' records; nxt_flat: the next record of the same worker (any), has_next: it exists and
    is of the same episode."""
    from isaac_bridge.tok_obs import EV_NEW_ROOM, EV_EXIT, KIND_COLUMN
    n = len(rows_flat)
    pl = np.asarray(rows_flat['player'], np.float64)
    out = dict(px=pl[:, 0].astype(np.float32), py=pl[:, 1].astype(np.float32))
    nd = np.asarray(rows_flat['n_doors'], np.int64)
    d = np.asarray(rows_flat['doors'], np.float64)
    ddx = np.full(n, np.nan); ddy = np.full(n, np.nan)
    for i in range(n):
        k = int(nd[i])
        if k:
            dd = d[i, :k, :2] * 200
            j = int(np.argmin((dd ** 2).sum(1)))
            ddx[i], ddy[i] = dd[j]
    out.update(door_dx=ddx.astype(np.float32), door_dy=ddy.astype(np.float32),
               door_dir=np.where(np.isnan(ddx), 0, _dir8(np.nan_to_num(ddx), np.nan_to_num(ddy))))
    ne = np.minimum(np.asarray(rows_flat['n_ent'], np.int64), rows_flat['ent'].shape[1])
    edx = np.full(n, np.nan); edy = np.full(n, np.nan)
    npc_col = KIND_COLUMN + 6
    for i in range(n):
        e = rows_flat['ent'][i, :ne[i]]
        npc = e[:, npc_col] > 0
        if npc.any():
            ee = e[npc][:, :2].astype(np.float64) * 200
            j = int(np.argmin((ee ** 2).sum(1)))
            edx[i], edy[i] = ee[j]
    out.update(npc_dx=edx.astype(np.float32), npc_dy=edy.astype(np.float32),
               npc_dir=np.where(np.isnan(edx), 0, _dir8(np.nan_to_num(edx), np.nan_to_num(edy))))
    m0 = np.asarray(rows_flat['map'][:, 3]).reshape(n, -1)
    m1 = np.asarray(nxt_flat['map'][:, 3]).reshape(n, -1)
    ev = np.asarray(nxt_flat['events'], np.int64)
    moved = (m0.any(1) & m1.any(1) & (m0 != m1).any(1)) | ((ev & (EV_NEW_ROOM | EV_EXIT)) > 0)
    out['room_next'] = (moved & has_next).astype(np.int8)
    out['has_next'] = has_next.astype(np.int8)
    return out


def _rollout_labels(rl, flat):
    flat = np.asarray(flat, np.int64)
    w, s = flat // rl.cap, flat % rl.cap
    rows = rl.rows
    nx = np.minimum(s + 1, rl.cap - 1)
    has = (s + 1 < np.asarray(rl.length)[w]) & (rows['first'][w, nx] == 0)
    lab = labels(rows[w, s], rows[w, nx], has)
    lab.update(worker=w.astype(np.int32), step=s.astype(np.int32), episode=np.asarray(rows['episode'][w, s], np.int64),
               t=np.asarray(rows['t'][w, s], np.int32))
    return lab


def new_group(ctx):
    if ON and _s['n'] < UPDATES:
        _s['groups'].append((ctx['z_out'][0].detach().float().cpu().numpy().astype(np.float16),
                             ctx['o_seq'].detach().float().cpu().numpy().astype(np.float16)))


def new_prox(rl, rp, flat, update, out, model):
    if not ON or _s['n'] >= UPDATES:
        _s['groups'] = []
        return
    import torch
    from torch import nn
    from isaac_bridge.structured_input import from_rows
    zr = np.concatenate([g[0] for g in _s['groups']]); o = np.concatenate([g[1] for g in _s['groups']])
    _s['groups'] = []
    assert len(zr) == len(flat), (len(zr), len(flat))
    lab = _rollout_labels(rl, flat)
    rng = np.random.default_rng(int(update))
    sub = np.sort(rng.choice(len(flat), min(SUB, len(flat)), replace=False))
    # the subsample's frames: through the plan's chunks (each decision's chunk and offset)
    where = {}
    for (off, n_, w_, b_), ch in zip(rp['tch'], rp['chunks']):
        for k in range(n_):
            where[off + k] = (ch, k)
    ss, bm = model.structured_spatial, model.block_memory
    feats = {k: [] for k in ('emb_body', 'emb_local', 'emb_map', 'emb_entmean', 'emb_room', 'spat_body', 'spat_global', 'z')}
    dev = next(model.parameters()).device
    fcache = {}
    for lo in range(0, len(sub), 256):
        idx = sub[lo:lo + 256]
        recs = [where[int(i)] for i in idx]
        # every record as a frame of the batch's widest size (source_dtype records differ in their frame width)
        framed = []
        for ch, k in recs:
            if id(ch) not in fcache:
                fcache[id(ch)] = ch.framed()
            framed.append(fcache[id(ch)][k:k + 1])
        wide = max(framed, key=lambda r: r.dtype['visible_frame'].shape[0]).dtype
        rows = np.zeros(len(framed), wide)
        for j, r in enumerate(framed):
            for name in ('first', 'visible_schema', 'visible_bytes'):
                rows[name][j] = r[name][0]
            rows['visible_frame'][j, :r.dtype['visible_frame'].shape[0]] = r['visible_frame'][0]
        parts = [(rows, np.concatenate([ch.pending[k:k + 1] for ch, k in recs]))]
        for rws, pend in parts:
            with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                b = from_rows(rws, dev, pend)
                bb = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in b.items()}
                saved = ss.blocks
                ss.blocks = nn.ModuleList()
                x0, pad0 = ss.forward_eager(bb)
                ss.blocks = saved
                x1, pad1 = ss.forward_eager(bb)
                z = bm.resampler(x1, pad1)
            Bw, Lw = b['v_body'].shape[1], b['v_local'].shape[1]
            Ew = b['v_entity'].shape[1]
            r0 = Bw; l0 = r0 + 28; m0 = l0 + Lw; e0 = m0 + 1
            em = b['v_entity_mask'].float()[:, :, None]
            ent = (x0[:, e0:e0 + Ew] * em).sum(1) / em.sum(1).clamp(min=1)
            for k_, v in (('emb_body', x0[:, 0]), ('emb_local', x0[:, l0]), ('emb_map', x0[:, m0]), ('emb_entmean', ent),
                          ('emb_room', x0[:, r0:r0 + 28]), ('spat_body', x1[:, 0]), ('spat_global', x1[:, -1]), ('z', z)):
                feats[k_].append(v.float().cpu().numpy().astype(np.float16))
    np.savez(os.path.join(str(out), f'd4_u{int(update):06d}.npz'), kind='block', update=int(update), zr=zr, o=o, sub=sub,
             **{k: np.concatenate(v) for k, v in feats.items()}, **lab)
    _s['n'] += 1


def _hooks(model):
    if _s['hooks']:
        return

    def pre(name):
        def f(mod, inp):
            _s['cap'][name] = inp[0].detach()
        return f

    def post(name):
        def f(mod, inp, outp):
            _s['cap'][name] = outp.detach()
        return f
    model.pi.register_forward_pre_hook(pre('pi_in'))
    model.player.register_forward_hook(post('player'))
    _s['hooks'] = True


def old_part(model, k_, part, flat, rl, update, out):
    if not ON:
        return
    _hooks(model)
    u = int(update)
    if _s['old'] is not None and _s['old']['update'] != u:
        _old_flush()
    if _s['n'] >= UPDATES:
        return
    if _s['old'] is None:
        _s['old'] = dict(update=u, out=out, pos={}, zr=[], player=[], lab=[], doors=[])
    c = _s['cap']
    pos = np.asarray(flat)[part.detach().cpu().numpy() if hasattr(part, 'detach') else np.asarray(part)]
    h, pl = c.get('pi_in'), c.get('player')
    if h is None or pl is None or min(len(h), len(pl)) < k_:
        if not _s.get('warned'):
            import sys
            print(f'd4hook: part skipped: k_ {k_}, pi_in {None if h is None else tuple(h.shape)}, player '
                  f'{None if pl is None else tuple(pl.shape)}', file=sys.stderr, flush=True)
            _s['warned'] = True
        return
    st = _s['old']
    keep = [i for i, p in enumerate(pos[:k_]) if int(p) not in st['pos']]
    for i in keep:
        st['pos'][int(pos[i])] = len(st['pos'])
    if keep:
        ki = np.asarray(keep)
        st['zr'].append(h[:k_][ki].float().cpu().numpy().astype(np.float16))
        st['player'].append(pl[:k_][ki].float().cpu().numpy().astype(np.float16))
        pk = pos[ki].astype(np.int64)   # (the labels now: the rollout's arrays are refilled after the update)
        st['lab'].append(_rollout_labels(rl, pk))
        st['doors'].append(np.asarray(rl.rows['doors'][pk // rl.cap, pk % rl.cap], np.float32))


def _old_flush():
    st = _s['old']
    _s['old'] = None
    if st is None or not st['pos']:
        return
    lab = {k: np.concatenate([d[k] for d in st['lab']]) for k in st['lab'][0]}
    np.savez(os.path.join(str(st['out']), f'd4_u{st["update"]:06d}.npz'), kind='old', update=st['update'],
             zr=np.concatenate(st['zr']), emb_player=np.concatenate(st['player']),
             doors_raw=np.concatenate(st['doors']), **lab)
    _s['n'] += 1


import atexit  # noqa: E402
atexit.register(_old_flush)
