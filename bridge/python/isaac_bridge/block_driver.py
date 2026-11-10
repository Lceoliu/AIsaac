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


# (B54) the driver's per-collect counters: their progress.csv columns, the deltas, the stdout line
DRV_FIELDS = ('drv_pause_s', 'drv_pauses', 'drv_lane_jobs', 'drv_lane_rows', 'drv_merged_rows', 'drv_merged_calls',
              'drv_lane_calls', 'drv_captures', 'drv_lane_python', 'drv_lane_s', 'drv_lane_hist',
              'drv_py_lane_rounds', 'drv_py_lane_rows', 'drv_py_lane_flushed_rows', 'drv_py_lane_share',
              'drv_lane_deferred_requests', 'drv_lane_orphan_requests', 'drv_steps', 'drv_sync_s', 'drv_pages_s', 'drv_host_s',
              'drv_cap_holds')


def collect_counters(c0, c1, handoff, stat, py_mark):
    """(B54) One collect's counters into stat (the lane_stat dict train_tok writes to progress.csv): c0 / c1 the
    driver's counters() at the collect's start / end, handoff the seconds the driver was paused for the hand-off
    before this collect (excluded from the paused seconds), py_mark a one-element list holding the Python-path lane
    counters at the last collect's end. Returns the stdout line's counter part."""
    d = {k: c1[k]-c0[k] for k in c1 if not isinstance(c1[k], (list, tuple))}
    stat['drv_pause_s'] = max(0.0, d['pause_s']-handoff)
    stat['drv_pauses'] = d['pauses']
    for k in ('lane_jobs', 'lane_rows', 'merged_rows', 'merged_calls', 'lane_calls', 'captures', 'lane_python', 'lane_s',
              'steps', 'sync_s', 'pages_s', 'host_s'):
        stat['drv_'+k] = d[k]
    stat['drv_cap_holds'] = d.get('cap_holds', 0)   # (B57: main steps' host / GPU-wait / page seconds, held records)
    stat['drv_lane_hist'] = '/'.join(str(b-a) for a, b in zip(c0['lane_hist'], c1['lane_hist']))
    # the lane rows the Python path stepped since the last collect (the actor waiting for the learner: rounds on the
    # Python path, the round in flight flushed), their share of all lane rows
    now = {k: stat.get(k, 0) for k in ('py_lane_rounds', 'py_lane_rows', 'py_lane_flushed_rows', 'lane_deferred_requests',
                                       'lane_orphan_requests')}
    for k, v in now.items():
        stat['drv_'+k] = v-py_mark[0].get(k, 0)
    py_mark[0] = now
    py_rows = stat['drv_py_lane_rows']+stat['drv_py_lane_flushed_rows']
    stat['drv_py_lane_share'] = py_rows/max(py_rows+d['lane_rows'], 1)
    return (f"paused {stat['drv_pause_s']:.2f} s ({d['pauses']} pauses), lane jobs {d['lane_jobs']} ({d['lane_rows']} rows: "
            f"{d['merged_rows']} merged in {d['merged_calls']} main batches, {d['lane_calls']} lane-only calls), run-time "
            f"captures {d['captures']}, lane jobs stepped in Python {d['lane_python']}; lane-only {d['lane_s']:.2f} s, rows "
            f"histogram 1/2/3-4/5-8/9-16/>16 {stat['drv_lane_hist']}; Python path while waiting for the learner "
            f"{stat['drv_py_lane_rounds']} rounds {stat['drv_py_lane_rows']} rows + {stat['drv_py_lane_flushed_rows']} flushed "
            f"({100*stat['drv_py_lane_share']:.1f}% of lane rows); branch requests waiting for their capture "
            f"{stat['drv_lane_deferred_requests']}, cancelled for a point that never came {stat['drv_lane_orphan_requests']}")


def policy_lag_violations(roll_version, roll_length, min_version):
    """(B54, the policy-lag constraint) The decisions of a rollout made with weights older than min_version: every
    decision of the h-th hand-off (0-based, resumed after U updates) is made with weights of at least U + h - 1
    updates, i.e. at most one update behind the update that trains on it. Column 0 is left out: a worker's open
    decision carried over from the rollout before (spare.carry) was made in that collect. Returns (violations, oldest
    version among the checked decisions, -1 when none)."""
    import numpy as np
    v = np.asarray(roll_version)
    length = np.asarray(roll_length)
    cols = np.arange(v.shape[1])[None, :]
    mask = (cols >= 1) & (cols < length[:, None])
    if not mask.any():
        return 0, -1
    used = v[mask]
    return int((used < min_version).sum()), int(used.min())
