"""B44 measurement only: one real update's learner inputs for the fixed-input learner benchmark.

ISAAC_LEARNER_DUMP=<dir> [ISAAC_LEARNER_DUMP_AT=<update>, default 3]: during that update every
forward_bucketed call on paged chunks is recorded (its chunks, grad mode and keyword arguments);
at the update's end the referenced part of the page store (K/V and latent rows, block values and
sources, raw frames), the chunks' rows and snapshots, the call list and the learner weights are
saved to <dir>/learner_inputs.pt. bridge/ops/b44_learner_bench.py replays it (load()).

Rows are stored as their frame bytes (not the 128 KiB padded records); store ids are compacted.
"""
import os

import numpy as np
import torch

DIR = os.environ.get('ISAAC_LEARNER_DUMP', '')
AT = int(os.environ.get('ISAAC_LEARNER_DUMP_AT', '3'))
STATE = dict(active=False, calls=[], done=False)


def begin(update, model=None):
    STATE['active'] = bool(DIR) and not STATE['done'] and int(update) == AT
    STATE['calls'] = []
    if STATE['active'] and model is not None:   # the weights the update starts from
        STATE['model'] = dict(config=model.config,
                              model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})


def record(chunks, kwargs):
    if STATE['active']:
        STATE['calls'].append((list(chunks), torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
                               {k: v for k, v in kwargs.items() if k != 'generator'}))


def _rows_blob(rows):
    rows = np.asarray(rows)
    size = np.asarray(rows['visible_bytes'], np.int64)
    frame = rows['visible_frame'].reshape(len(rows), -1)
    blob = np.concatenate([frame[i, :size[i]] for i in range(len(rows))]) if len(rows) else np.empty(0, np.uint8)
    return dict(first=np.asarray(rows['first']).copy(), schema=np.asarray(rows['visible_schema']).copy(),
                size=size, blob=blob)


def _rows_from_blob(d):
    from .block_training import source_dtype
    n = len(d['size'])
    rows = np.zeros(n, dtype=source_dtype(max([4] + d['size'].tolist()), True))
    rows['first'] = d['first']; rows['visible_schema'] = d['schema']; rows['visible_bytes'] = d['size']
    at = np.concatenate(([0], np.cumsum(d['size'])))
    for i in range(n):
        rows['visible_frame'][i, :d['size'][i]] = d['blob'][at[i]:at[i+1]]
    return rows


def end(model, extra=None):
    """Save the recorded update (once); no-op when not recording."""
    if not STATE['active']:
        return None
    STATE['active'] = False
    STATE['done'] = True
    calls = STATE['calls']; STATE['calls'] = []
    if not calls:
        return None
    store = calls[0][0][0].snapshot.store
    cfg = store.config
    chunk_ids, chunk_list = {}, []
    for chunks, *_ in calls:
        for c in chunks:
            if id(c) not in chunk_ids:
                chunk_ids[id(c)] = len(chunk_list); chunk_list.append(c)
    # referenced store rows
    kv_ids, blk_ids, raw_ids = [], [], []
    logs, log_ids = [], {}
    for c in chunk_list:
        snap = c.snapshot
        if snap.store is not store:
            raise ValueError('one page store per dumped update')
        if id(snap.log) not in log_ids:
            log_ids[id(snap.log)] = len(logs); logs.append(snap.log)
        p = store.snapshot_parts(snap)
        kv_ids += [p['kvs'], p['pkvs']]; blk_ids.append(p['bid']); raw_ids.append(p['pdec'])
    kv_u = np.unique(np.concatenate(kv_ids)) if kv_ids else np.empty(0, np.int64)
    blk_u = np.unique(np.concatenate(blk_ids)) if blk_ids else np.empty(0, np.int64)
    raw_u = np.unique(np.concatenate(raw_ids + [store.b_src[blk_u].reshape(-1)]))
    raw_u = raw_u[raw_u >= 0]

    def remap(values, table):
        values = np.asarray(values, np.int64)
        at = np.clip(np.searchsorted(table, values), 0, max(len(table)-1, 0))
        ok = (table[at] == values) if len(table) else np.zeros(len(values), bool)
        return np.where(ok, at, 0)
    log_out = [dict(episode=l.episode, offset=l.offset, dec=l.dec[:l.n].copy(), kvs=remap(l.kvs[:l.n], kv_u),
                    blk=remap(l.blk[:l.nb], blk_u), blk_end=l.blk_end[:l.nb].copy()) for l in logs]
    rows_by_key, rows_out, chunk_out = {}, [], []
    for c in chunk_list:
        r = np.asarray(c.rows)
        key = (r.__array_interface__['data'][0], r.shape, r.strides)
        if key not in rows_by_key:   # (B45: compact records' frames from their arena)
            rows_by_key[key] = len(rows_out); rows_out.append(_rows_blob(c.framed()))
        s = c.snapshot
        chunk_out.append(dict(rows=rows_by_key[key], pending=np.asarray(c.pending, np.int64).copy(),
                              log=log_ids[id(s.log)], step=s.step, spending=np.asarray(s.pending).copy(),
                              version=s.policy_version))
    raw = {int(d): store.raw[int(d)] for d in raw_u.tolist()}
    out = dict(config=cfg.__dict__.copy(), nh=store.nh, kv_dtype=str(store.kv_dtype), page=store.page,
               kv=store.kv(kv_u), z=store.z(kv_u), d_meta=store.d_meta[kv_u].copy(),
               blocks=store.blocks(blk_u), b_meta=store.b_meta[blk_u].copy(), b_src=store.b_src[blk_u].copy(),
               b_srcver=store.b_srcver[blk_u].copy(), raw=raw, logs=log_out, rows=rows_out, chunks=chunk_out,
               calls=[dict(chunks=[chunk_ids[id(c)] for c in ch], grad=g, inference=i, kwargs=kw)
                      for ch, g, i, kw in calls],
               model=STATE.pop('model', None) or dict(config=model.config, model={
                   k: v.detach().cpu() for k, v in model.state_dict().items()}),
               extra=extra or {})
    os.makedirs(DIR, exist_ok=True)
    path = os.path.join(DIR, 'learner_inputs.pt')
    torch.save(out, path)
    print(f'B44 learner inputs: {len(calls)} calls, {len(chunk_list)} chunks, {len(kv_u)} K/V rows, '
          f'{len(blk_u)} blocks, {len(raw)} raw frames -> {path}', flush=True)
    return path


def load(path):
    """(model checkpoint dict, store, chunks, calls) of a saved update; calls: dicts with
    'chunks' (Chunk objects), 'grad', 'inference', 'kwargs'."""
    from .block_memory import BlockConfig
    from .block_paged import PagedStore, EpisodeLog, PagedSnapshot
    from .block_training import Chunk
    d = torch.load(path, map_location='cpu', weights_only=False)
    cfg = BlockConfig(**d['config'])
    kv_dtype = getattr(torch, d['kv_dtype'].replace('torch.', ''))
    store = PagedStore(cfg, d['nh'], kv_dtype=kv_dtype, page=d['page'])
    n_kv, n_b = len(d['kv']), len(d['blocks'])
    kv_ids = np.sort(store._alloc(max(n_kv, 1), 'd'))[:n_kv]   # a fresh store hands out 0..n-1
    if not np.array_equal(kv_ids, np.arange(n_kv)):
        raise RuntimeError('fresh store ids are not 0..n-1')
    store._scatter(store.kv_pages, kv_ids, d['kv']); store._scatter(store.z_pages, kv_ids, d['z'])
    store.d_meta[:n_kv] = d['d_meta']
    bids = np.sort(store._alloc(max(n_b, 1), 'b'))[:n_b]
    store._scatter(store.b_pages, bids, d['blocks'])
    store.b_meta[:n_b] = d['b_meta']; store.b_src[:n_b] = d['b_src']; store.b_srcver[:n_b] = d['b_srcver']
    store.raw = dict(d['raw'])
    store.next_decision = max(store.raw) + 1 if store.raw else 0
    logs = [EpisodeLog(l['episode'], l['offset'], l['dec'], l['kvs'], l['blk'], l['blk_end']) for l in d['logs']]
    rows = [_rows_from_blob(r) for r in d['rows']]
    chunks = [Chunk(rows[c['rows']], c['pending'], PagedSnapshot(store, logs[c['log']], c['step'], c['spending'],
                                                                 c['version'])) for c in d['chunks']]
    calls = [dict(chunks=[chunks[i] for i in c['chunks']], grad=c['grad'], inference=c['inference'],
                  kwargs=c['kwargs']) for c in d['calls']]
    return d['model'], store, chunks, calls, d.get('extra', {})
