"""B52: glue between train_tok and the C++ actor driver (actor_native.ActorDriver).

make_driver wires the driver to the sampler's shared memory (records, actions, main_value / main_vg, aux, cf_snap,
priv), the worker pipes, the main actor's arena (reset tensors, scratch row, step counters) and the actor stream,
registers the step graphs captured so far and returns it (None when the extension is unavailable: the Python path).
The driver thread then polls the workers, steps the main actor's graphs, replies before writing the pages and hands
each step to the Python consumer (train_tok's run_driver_collect) as a record: the consumer does the bookkeeping the
Python actor loop did (Episodes.arrive, store into the rollout, lineage, snapshots, lane observe / record, the CF
point capture) from the record, never from the live arena. precapture_lane_generations pre-creates both teacher
generations' lane actors (and their graphs) at start-up instead of on the actor thread when a generation first
appears (B50 phase 0: a 9.2 s stall).
"""
import copy
import os

import numpy as np
import torch

from . import actor_native


def reason():
    return actor_native.REASON


def _ptr(a):
    return 0 if a is None else int(a.ctypes.data)


def make_driver(actor, sampler, stream, seed, pf_on, aux):
    """The driver for actor (the main SlotBlockActor, its store native) and sampler, on `stream`; None when off."""
    ext = actor_native.load()
    if ext is None or getattr(actor.store, 'native', None) is None or actor.step_graph is None:
        return None
    from .structured_input import BUCKETS, FIXED, FRAME_BYTES, VERSION
    from .tok_obs import ROW, DONE_BRANCH
    from .tok_sampler import _REPLY
    d = ext.ActorDriver(actor.store.native)
    d.set_stream(int(stream.stream_id), int(stream.device_index))
    d.set_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    d.set_buckets(list(BUCKETS['body']), list(BUCKETS['entity']), list(BUCKETS['inventory']), list(BUCKETS['slots']), int(FIXED),
                  int(FRAME_BYTES), int(VERSION))
    fields = {n: (int(ROW.fields[n][1]), 1 if n == 'visible_frame' else int(ROW.fields[n][0].itemsize))
              for n in ('first', 'done', 'pforce', 'visible_schema', 'visible_bytes', 'visible_frame')}
    main_value, main_vg, cf_snap = getattr(sampler, 'main_value', None), getattr(sampler, 'main_vg', None), getattr(sampler, 'cf_snap', None)
    for name, a in (('main_value', main_value), ('main_vg', main_vg), ('aux', aux)):
        if a is not None and a.dtype != np.float64:
            raise TypeError(f'{name}: float64 expected, got {a.dtype}')
    if cf_snap is not None and cf_snap.dtype != np.int64:
        raise TypeError('cf_snap: int64 expected')
    if sampler.actions.dtype != np.int32:
        raise TypeError('actions: int32 expected')
    priv = getattr(sampler, 'priv', None)
    model = actor.model
    nh, heads = len(model.heads), sum(model.heads)
    # (set_arena before set_sampler: pending_cur needs nh)
    st = actor.state
    tensors = [st.kv] + [getattr(st, name) for name in st._rows]
    dims = [2] + [0]*len(st._rows)
    fills = [0.0] + [0.0 if name in ('bank', 'partial', 'steps') else -1.0 for name in st._rows]
    d.set_arena(tensors, dims, fills, int(st.slots-1), int(actor.base), int(st.step_host.ctypes.data), heads, nh,
                int(model.block_memory.config.block_steps), int(st.slots), id(actor))
    d.set_sampler(int(sampler.rows.ctypes.data), int(ROW.itemsize), int(sampler.cfg.workers), [int(f) for f in sampler._fds],
                  int(sampler.actions.ctypes.data), int(sampler.actions.shape[1]), _ptr(main_value), _ptr(main_vg), _ptr(aux),
                  int(aux.shape[1]) if aux is not None else 0, _ptr(cf_snap), _ptr(priv), int(priv.dtype.itemsize) if priv is not None else 0,
                  fields, _REPLY, int(DONE_BRANCH), bool(pf_on))
    actor.step_graph.attach_driver(d)
    return d


def precapture_lane_generations(lanes, model):
    """Both teacher generations' policies and lane actors (graphs pre-captured) at start-up. TeacherGenerations.publish
    later loads weights into the pre-created policy in place, so the captured graphs stay valid."""
    if os.environ.get('ISAAC_BLOCK_LANE_PRECAPTURE', '1') != '1' or lanes.main_actor is None:
        return 0
    gens = lanes.generations
    made = 0
    for i in range(len(gens.slots)):
        if gens.slots[i] is None:
            gens.slots[i] = copy.deepcopy(model).eval().requires_grad_(False)
        if i not in lanes.actors:
            lanes.actors[i] = lanes.main_actor.lane_actor(gens.slots[i], len(lanes.s.lane_io), generation=i)
            made += 1
    return made


def record_rows(rec, ROW):
    """The ROW records of a driver record (header + used frame bytes copied by the driver)."""
    return rec['rows'].numpy().view(ROW)
