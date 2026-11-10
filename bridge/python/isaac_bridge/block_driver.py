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
              'drv_cap_holds', 'drv_poll_s', 'drv_prep_s', 'drv_reply_s', 'drv_cons_s', 'drv_push_s', 'drv_mlane_s',
              'drv_hold_s', 'drv_loop_s', 'drv_polls', 'drv_idle_polls', 'drv_lo_cur_rows', 'drv_lo_old_rows', 'drv_putback_rows',
              'drv_enq_in_step', 'drv_enq_outside', 'b58e_batches', 'b58e_ev_rows', 'drv_m1_jobs', 'drv_m1_rows',
              'drv_lo_old_jobs', 'drv_lo_switch_jobs', 'drv_lo_refused_jobs', 'drv_lo_idle_jobs', 'drv_lo_other_jobs')
# (B58 step 0) the driver thread's time by phase (abp_actor.cpp: t_*_s); pages_s leaves the consumer out since step 0
DRV_PHASES = ('poll', 'prep', 'sync', 'reply', 'pages', 'cons', 'push', 'mlane', 'lane', 'hold', 'pause')


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
    for k in ('poll_s', 'prep_s', 'reply_s', 'cons_s', 'push_s', 'mlane_s', 'hold_s', 'loop_s', 'polls', 'idle_polls',
              'lo_cur_rows', 'lo_old_rows', 'putback_rows', 'enq_in_step', 'enq_outside',   # (B58d plan: merging)
              'm1_jobs', 'm1_rows', 'lo_old_jobs', 'lo_switch_jobs', 'lo_refused_jobs', 'lo_idle_jobs', 'lo_other_jobs'):   # (d2 M1)
        stat['drv_'+k] = d.get(k, 0)   # (B58 step 0; absent from an older extension's counters)
    # the lane rows the Python path stepped since the last collect (the actor waiting for the learner: rounds on the
    # Python path, the round in flight flushed), their share of all lane rows
    now = {k: stat.get(k, 0) for k in ('py_lane_rounds', 'py_lane_rows', 'py_lane_flushed_rows', 'lane_deferred_requests',
                                       'lane_orphan_requests')}
    for k, v in now.items():
        stat['drv_'+k] = v-py_mark[0].get(k, 0)
    py_mark[0] = now
    py_rows = stat['drv_py_lane_rows']+stat['drv_py_lane_flushed_rows']
    phases = '/'.join('%.2f' % (stat['drv_pause_s'] if k == 'pause' else d.get(k+'_s', 0)) for k in DRV_PHASES)
    stat['drv_py_lane_share'] = py_rows/max(py_rows+d['lane_rows'], 1)
    return (f"paused {stat['drv_pause_s']:.2f} s ({d['pauses']} pauses), lane jobs {d['lane_jobs']} ({d['lane_rows']} rows: "
            f"{d['merged_rows']} merged in {d['merged_calls']} main batches, {d['lane_calls']} lane-only calls), run-time "
            f"captures {d['captures']}, lane jobs stepped in Python {d['lane_python']}; lane-only {d['lane_s']:.2f} s, rows "
            f"histogram 1/2/3-4/5-8/9-16/>16 {stat['drv_lane_hist']}; Python path while waiting for the learner "
            f"{stat['drv_py_lane_rounds']} rounds {stat['drv_py_lane_rows']} rows + {stat['drv_py_lane_flushed_rows']} flushed "
            f"({100*stat['drv_py_lane_share']:.1f}% of lane rows); branch requests waiting for their capture "
            f"{stat['drv_lane_deferred_requests']}, cancelled for a point that never came {stat['drv_lane_orphan_requests']}"
            f"; driver s by phase {'/'.join(DRV_PHASES)} "
            f"{phases} of loop {d.get('loop_s', 0):.2f} "
            f"({d.get('polls', 0)} polls, {d.get('idle_polls', 0)} idle); M1 merged {d.get('m1_jobs', 0)} jobs "
            f"({d.get('m1_rows', 0)} rows), lane-only jobs old / switched / refused / idle / other "
            f"{'/'.join(str(d.get(k, 0)) for k in ('lo_old_jobs', 'lo_switch_jobs', 'lo_refused_jobs', 'lo_idle_jobs', 'lo_other_jobs'))}")


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


# ------------------------------------------------------------------------------------------- B58a+b native consumer

class NativeFrameArena:
    """(B58a+b) FrameArena's interface over the driver's frame arena of a rollout slot: the driver appends the stored
    decisions' frames; Python appends the hand-off's carried frames and reads buf / used once the rollout is handed."""
    ALIGN = 8

    def __init__(self, driver, slot):
        self.d, self.slot = driver, int(slot)

    @property
    def buf(self):
        return self.d.arena_buf(self.slot).numpy()

    @property
    def used(self):
        return int(self.d.arena_used(self.slot))

    def reset(self):
        self.d.arena_reset(self.slot)

    def append_from(self, buf, offs, sizes):
        import torch
        return self.d.arena_append_from(self.slot, torch.from_numpy(np.ascontiguousarray(buf)),
                                        torch.from_numpy(np.asarray(offs, np.int64)),
                                        torch.from_numpy(np.asarray(sizes, np.int64))).numpy()

    def append(self, rows):
        frames = np.ascontiguousarray(rows['visible_frame']).reshape(-1)
        width = rows.dtype['visible_frame'].shape[0]
        return self.append_from(frames, np.arange(len(rows), dtype=np.int64)*width, rows['visible_bytes'])


def native_consumer_reason(cpp_driver, rolls, lab, t_on, mem_on, proc, n_act):
    """(B58a+b) None when the native consumer can take the records, else why not (the Python consumer runs)."""
    if os.environ.get('ISAAC_BLOCK_CPP_CONSUMER', '1') == '0':
        return 'ISAAC_BLOCK_CPP_CONSUMER=0'
    if cpp_driver is None:
        return 'no C++ driver'
    for why, bad in (('the lab', lab is not None), ('temporal models', t_on), ('GRU memory', mem_on),
                     ('a learner process', proc), ('several actor processes', n_act > 1)):
        if bad:
            return why
    if any(getattr(r, 'frames', None) is None or r.h is not None or r.t_kv is not None for r in rolls):
        return 'a non-compact rollout'
    return None


def attach_consumer(d, book, rolls, gamma):
    """(B58a+b) The driver's native consumer: Episodes' arrays and reward weights, both rollouts' arrays (slots 0 / 1:
    rolls[i].native_slot), their frame arenas moved into the driver (NativeFrameArena)."""
    from .tok_obs import ROW, ROWC, ROW_FRAME_AT, EV_EXIT, EV_ITEM, EV_ACTIVE, EV_PILL, DONE_BRANCH

    def ptr(a, dtype=None, shape=None):
        if a is None:
            return None
        if dtype is not None and a.dtype != dtype:
            raise TypeError(f'native consumer: {dtype} expected, got {a.dtype}')
        if not a.flags['C_CONTIGUOUS']:
            raise TypeError('native consumer: a contiguous array expected')
        return int(a.ctypes.data)
    rw = book.rw
    f = lambda n: int(ROW.fields[n][1])
    cfg = dict(o_first=f('first'), o_done=f('done'), o_damage=f('damage'), o_hurt=f('hurt'), o_events=f('events'),
               o_stage=f('stage'), o_player=f('player'), o_pforce=f('pforce'), row_frame_at=int(ROW_FRAME_AT),
               **{f'w_{k}': float(rw.get(k, 0.0)) for k in ('damage', 'hurt', 'time', 'death', 'timeout', 'room', 'explore',
                                                             'boss', 'exit', 'heal', 'resource')},
               end_bonus=float(book.end_bonus), gamma=float(gamma), floor=bool(book.floor), run=bool(book.run),
               items=bool(book.items), gains_on=book.gains is not None, plr_on=book.plr is not None,
               pick_on=book.pick is not None, ev_exit=int(EV_EXIT), ev_item=int(EV_ITEM), ev_active=int(EV_ACTIVE),
               ev_pill=int(EV_PILL), done_branch=int(DONE_BRANCH),
               ret=ptr(book.ret, np.float64), hurt=ptr(book.hurt, np.float64), damage=ptr(book.damage, np.float64),
               gains=ptr(book.gains, np.float64), healed=ptr(book.healed, np.float64), gathered=ptr(book.gathered, np.float64),
               v0=ptr(book.v0, np.float64), gret=ptr(book.gret, np.float64), rooms=ptr(book.rooms, np.int64),
               boss=ptr(book.boss, np.int64), extra=ptr(book.extra, np.int64), steps=ptr(book.steps, np.int64),
               pick_steps=ptr(book.pick['steps'], np.int64) if book.pick is not None else None)
    if book.plr is not None:
        cfg['gamma'] = float(book.plr[2])
    d.set_consumer(cfg)
    for slot, r in enumerate(rolls):
        if r.rows.dtype != ROWC:
            raise TypeError('native consumer: ROWC rollouts')
        a = dict(cap=int(r.cap), rowc_size=int(ROWC.itemsize), off_visible_off=int(ROWC.fields['visible_off'][1]),
                 priv_size=int(r.priv.dtype.itemsize) if r.priv is not None else 0,
                 rows=ptr(r.rows), terminal=ptr(r.terminal, np.bool_), close_done=ptr(r.close_done, np.uint8),
                 close_ev=ptr(r.close_ev, np.uint8), priv=ptr(r.priv), boot=ptr(r.boot, np.bool_),
                 reward=ptr(r.reward, np.float32), close_hurt=ptr(r.close_hurt, np.float32), logp=ptr(r.logp, np.float32),
                 value=ptr(r.value, np.float32), value_g=ptr(r.value_g, np.float32), open=ptr(r.open, np.int64),
                 length=ptr(r.length, np.int64), pending=ptr(r.pending, np.int64), actions=ptr(r.actions, np.int64),
                 version=ptr(r.version, np.int64))
        d.set_rollout(slot, a, int(len(r.frames.buf)))
        r.frames = NativeFrameArena(d, slot)
        r.native_slot = slot


# ------------------------------------------------------------------------------------------------ B58c native lineage

def native_lineage_reason(cpp_driver, native_cons, lanes):
    """(B58c) None when the driver can keep the main rows' lineage (logs, snapshots, CF captures, injury sources, the
    rollout sidecar) in C++, else why not (consume_record does it in Python, as B54)."""
    if os.environ.get('ISAAC_BLOCK_CPP_LINEAGE', '1') == '0':
        return 'ISAAC_BLOCK_CPP_LINEAGE=0'
    if cpp_driver is None:
        return 'no C++ driver'
    if not native_cons:
        return 'the Python consumer (the sidecar needs the native rollout indices)'
    if lanes is None or lanes.main_actor is None:
        return 'no block lanes'
    return None


class NativeBlockRollout:
    """(B58c) BlockRollout over the store's native sidecar of a rollout slot: the driver records the chunk starts and
    open decisions; reset / carry act on the native sidecar (the driver paused); materialize() (at the hand-off, in
    the actor thread) turns them into Python PagedSnapshots, after which payload / plan are BlockRollout's."""

    def __init__(self, store, slot, chunk_steps):
        from .block_training import BlockRollout
        self.store, self.slot = store, int(slot)
        self.py = BlockRollout(chunk_steps)
        self.chunk_steps = int(chunk_steps)

    @property
    def starts(self):
        return self.py.starts

    @property
    def open(self):
        return self.py.open

    def record(self, *a, **k):
        raise RuntimeError('native lineage: the driver records the sidecar')

    def reset(self):
        self.store.native.side_reset(self.slot)
        self.py.reset()

    def carry(self, other, workers, indices):
        import torch
        self.store.native.side_carry(self.slot, other.slot, torch.from_numpy(np.asarray(workers, np.int64)),
                                     torch.from_numpy(np.asarray(indices, np.int64)))
        self.py.reset()

    def materialize(self):
        """The native sidecar as BlockRollout's starts / open (one native call; a snapshot shared by a start and an open
        decision is one Python object, as in BlockRollout)."""
        from .block_paged import native_snapshots
        keys, snaps, ow, oi, os_ = (x.numpy() for x in self.store.native.side_payload(self.slot))
        ids = list(dict.fromkeys(snaps.tolist() + os_.tolist()))
        objs = dict(zip(ids, native_snapshots(self.store, ids)))
        self.py.reset()
        for (w, i), s in zip(keys.tolist(), snaps.tolist()):
            self.py.starts.setdefault(int(w), {})[int(i)] = objs[s]
        for w, i, s in zip(ow.tolist(), oi.tolist(), os_.tolist()):
            self.py.open[int(w)] = (int(i), objs[s])
        return self

    def payload(self):
        return self.py.payload()

    def plan(self, rollout, mask):
        return self.py.plan(rollout, mask)


def attach_lineage(d, actor, lanes, sampler, rolls):
    """(B58c) The main rows' lineage into the driver: the store's lineage configured (window, blocks, the arena rows,
    the injury depths), the CF points' status array, the teacher rings / injury pins for the reclamation, the driver's
    ROW fields, the rollouts' sidecars native (NativeBlockRollout); the lanes' points / generations / injury sources
    read from it (BlockLaneService.attach_lineage)."""
    from .tok_obs import ROW
    from .tok_sampler import ST, TEACH_RING
    st = actor.store
    cfg = actor.model.block_memory.config
    nh = len(actor.model.heads)
    st.native.lineage_configure(int(cfg.window), int(cfg.block_steps), int(cfg.memory_blocks), nh, int(actor.state.slots),
                                int(sampler.cfg.workers), [int(x) for x in lanes.depths], int(rolls[0].block.chunk_steps),
                                len(lanes.generations.slots))
    status, mode, jobs, epoch, kept, commit = sampler.block_arrays[:6]
    st.native.lineage_status(int(status.ctypes.data))
    tr = sampler.teach_rows
    st.native.lineage_teach(dict(stats=int(sampler.stats.ctypes.data), stats_stride=int(sampler.stats.shape[1]),
                                 st_teach=int(ST['teach']), commit=int(commit.ctypes.data), ring=int(TEACH_RING) if tr is not None else 0,
                                 rows=int(tr.ctypes.data) if tr is not None else 0, row_size=int(ROW.itemsize), o_episode=int(ROW.fields['episode'][1]),
                                 o_t=int(ROW.fields['t'][1]), s_episode=int(ROW.fields['episode'][0].itemsize),
                                 s_t=int(ROW.fields['t'][0].itemsize), jobs=int(jobs.ctypes.data), jobs_k=int(jobs.shape[1]),
                                 epoch=int(epoch.ctypes.data)))
    fields = {n: (int(ROW.fields[n][1]), int(ROW.fields[n][0].itemsize)) for n in ('episode', 't', 'hurt')}
    d.set_lineage(True, os.getpid(), fields)
    for r in rolls:
        r.block = NativeBlockRollout(st, r.native_slot, r.block.chunk_steps)
    lanes.attach_lineage(st)


# --------------------------------------------------------------------------------------------- B58d native lanes

class NativeContexts:
    """(B58d) ContextRegistry over the native one (the lane service registers its answers' contexts in C++):
    get(ident) materializes the snapshot, retire / complete act on the native registry."""

    def __init__(self, store):
        self.store = store

    def get(self, ident):
        from .block_paged import native_snapshots
        return native_snapshots(self.store, [self.store.native.ctx_get(int(ident))])[0]

    def complete(self, owner):
        self.store.native.ctx_complete(int(owner))

    def retire(self, live, transient=False):
        self.store.native.ctx_retire(sorted(int(x) for x in live), bool(transient))

    def put(self, owner, snapshot):
        raise RuntimeError('native lanes: the lane service registers its contexts in C++')

    @property
    def values(self):   # (counts only)
        return range(int(self.store.native.lin_stats()['contexts']))


def native_lanes_reason(cpp_driver, native_lin, lanes):
    """(B58d) None when the lane service can run in C++ (the driver thread: d1; its own thread: d2), else why not."""
    if os.environ.get('ISAAC_BLOCK_CPP_LANES', '1') == '0':   # (B58d d1: on by default since its gate)
        return 'ISAAC_BLOCK_CPP_LANES=0'
    if not native_lin:
        return 'the Python lineage'
    gens = range(len(lanes.generations.slots))
    if not lanes.actors or any(g not in lanes.actors for g in gens):
        return 'the teacher generations not pre-created'
    if any(lanes.actors[g].step_graph is None or getattr(lanes.actors[g].step_graph, 'pool_tag', 'main') != 'lane' for g in gens):
        return 'lane graphs without their own pool'
    s = lanes.s
    if getattr(s, 'lane_t', None) is None or getattr(s, 'block_arrays', None) is None:
        return 'no lane_t / block arrays'
    return None


def attach_lanes(d, lanes, sampler, stream):
    """(B58d) The lane service in C++: its window constants, the lanes' shared arrays and pipes (taken out of the
    sampler's poll sets: only the native service reads them now), both teacher generations' graph owners and arena
    rows, the lane stream; the lane service's Python side switches to its native mode."""
    from .tok_obs import ROW
    from .tok_lanes import ACTIVE_S
    from .tok_teacher2 import V2_RING
    status, mode, jobs, epoch, kept = sampler.block_arrays[:5]
    cfg = dict(lanes=len(sampler.lane_io), tw=int(sampler.lane_t.shape[1]), wait=float(lanes.wait), quiet=float(lanes.quiet),
               active_s=float(ACTIVE_S), defer_s=float(lanes.DEFER_S), reclaim_s=float(lanes.RECLAIM_S), pid=os.getpid())
    for name, a in (('lane_rows', sampler.lane_rows), ('lane_io', sampler.lane_io), ('lane_out', sampler.lane_out),
                    ('lane_t', sampler.lane_t), ('v2_t', sampler.v2_t)):
        if a is not None and not a.flags['C_CONTIGUOUS']:
            raise TypeError(f'native lanes: {name} not contiguous')
    sh = dict(rows=int(sampler.lane_rows.ctypes.data), o_episode=int(ROW.fields['episode'][1]),
              s_episode=int(ROW.fields['episode'][0].itemsize), io=int(sampler.lane_io.ctypes.data), io_cols=int(sampler.lane_io.shape[1]),
              out=int(sampler.lane_out.ctypes.data), out_cols=int(sampler.lane_out.shape[1]), t=int(sampler.lane_t.ctypes.data),
              v2_t=int(sampler.v2_t.ctypes.data) if sampler.v2_t is not None else 0, v2_ring=int(V2_RING) if sampler.v2_t is not None else 0,
              mode=int(mode.ctypes.data), status=int(status.ctypes.data), pts_per=int(status.shape[1]), kept=int(kept.ctypes.data),
              kept_cols=int(kept.shape[1]), cf_snap=int(sampler.cf_snap.ctypes.data))
    fds = [int(c.fileno()) for c in sampler.lane_conns]
    for fd in fds:   # (Python never reads a lane request again)
        for p in (getattr(sampler, '_poller', None), getattr(sampler, '_lane_poller', None)):
            if p is not None:
                try:
                    p.unregister(fd)
                except (KeyError, ValueError, OSError):
                    pass
        sampler._lanes.pop(fd, None)
    gens = range(len(lanes.generations.slots))
    missing = [g for g in gens if g not in lanes.actors]
    if missing:   # (on_publish must never pause the driver from inside a record: see ActorDriver.pause)
        raise RuntimeError(f'native lanes: generations {missing} have no lane actor (ISAAC_BLOCK_LANE_PRECAPTURE=1)')
    owners = [int(lanes.actors[g].step_graph.owner) for g in gens]
    bases = [int(lanes.actors[g].base) for g in gens]
    d.lanes_configure(cfg, sh, fds, int(stream.stream_id), owners, bases)
    # (d2 M1: a lane job of the main version rides the next main batch. Off until M1's own gate and timing:
    # ISAAC_BLOCK_LANE_MERGE=1 turns it on; unset / 0: every job lane-only, the d1 behaviour)
    d.set_m1_merge(os.environ.get('ISAAC_BLOCK_LANE_MERGE', '0') == '1')
    lanes.attach_native_lanes(d, stream)
    sampler.block_contexts = lanes.contexts = NativeContexts(lanes.lin)


# (D1 / B58d) per collect: lane request timings (window / behind the caller / GPU / read -> reply) and the CF points'
# (capture -> first lane step -> release, life in seconds and policy versions), refused captures, seconds every
# teacher generation was pinned (ActorDriver.lane_timing), as progress.csv columns
LANE_TIMING_FIELDS = tuple(f'{k}_{s}' for k in ('m1_wait_s', 'lt_worker_s', 'lt_window_s', 'lt_driver_s', 'lt_gpu_s', 'lt_rt_s', 'pt_first_s', 'pt_run_s',
                                               'pt_life_s', 'pt_life_v') for s in ('n', 'p50', 'p90')) + ('pt_refused', 'pt_all_pinned_s',
                                                                                              'pt_live_mean', 'pt_live_max')


def check_versions(driver, actor):
    """(B58d) The driver's policy version (the native lane service orders a round's generations by it) and the main
    actor's (the Python lane service's) are one counter set in two places: a mismatch stops the run."""
    if driver is not None and actor is not None and int(driver.version) != int(actor.version):
        raise RuntimeError(f'policy versions out of step: driver {int(driver.version)}, main actor {int(actor.version)}')


# ------------------------------------------------------------------------------------------- B58e batched records

def batch_reason(cpp_driver, native_cons, native_lin_lanes):
    """(B58e) None when the driver can keep the per-step counters and hand the episode events over in batches (the
    native consumer, lineage and lane service: Python then needs nothing per step). ISAAC_BLOCK_CPP_BATCH=0: a record
    per step (the B58d path)."""
    if os.environ.get('ISAAC_BLOCK_CPP_BATCH', '1') == '0':
        return 'ISAAC_BLOCK_CPP_BATCH=0'
    if cpp_driver is None or not native_cons or not native_lin_lanes:
        return 'the native consumer, lineage and lanes first'
    return None


def batch_wait(d, on_events, on_record, timeout=0.5):
    """(B58e) One blocking wait of the main thread in batch mode: 'target' (the collect's rollout is full: kind 7),
    'wake' (kind 9: new weights to take), 'idle' (nothing within timeout: the caller checks its liveness conditions).
    Event batches (kind 8) go to on_events(rows, workers, snap); every other record (captures, lane records, publish,
    errors) to on_record. After such a record the wait polls every 2 ms until the queue is empty and resumes a driver
    that paused itself for it (a capture request, a Python-stepped job), as the per-record loop did. A driver that
    stopped or failed without a record raises here (the liveness timeout: at most `timeout` later)."""
    short = False
    while True:
        rec = d.next_record(0.002 if short else timeout)
        if rec is None:
            if short:
                short = False
                if d.is_paused() and d.queue_size() == 0:
                    d.resume()
                continue
            if not d.running or d.last_error():
                raise RuntimeError('cpp actor driver stopped: ' + (d.last_error() or 'no error message'))
            return 'idle'
        k = int(rec['kind'])
        if k == 7:
            return 'target'
        if k == 9:
            return 'wake'
        if k == 8:
            on_events(rec['rows'], rec['workers'], rec['snap'])
            continue
        on_record(rec)
        short = True


def pause_serving(d, consume):
    """The driver paused between steps, its records consumed meanwhile (consume(rec)). The driver thread can be blocked
    on Python: d1's lane service waits for the answer to a lane record (kind 1 / 5) that sits in the queue this caller
    consumes, so a plain pause() cannot end there (ActorDriver.pause raises in that state). Returns when the driver is
    paused (or stopped / failed: the caller's drain raises the error record). The records consumed here are the ones a
    drain after pause() would have consumed, in the same order."""
    d.request_pause()
    while not d.is_paused():
        if not d.running or d.last_error():
            return
        rec = d.next_record(0.002)
        if rec is not None:
            consume(rec)


def event_chunks(ws):
    """(B58e) [start, end) runs of an event batch in which no worker appears twice. _native_events takes a record's
    first rows before its ends; a batch spans steps, so a worker's first row after its end (the next episode) must be
    taken after that end: chunk by chunk, as the per-record path did (one record holds a worker once)."""
    out, seen, s = [], set(), 0
    for i, w in enumerate(ws.tolist()):
        if w in seen:
            out.append((s, i)); seen = set(); s = i
        seen.add(w)
    if len(ws) > s:
        out.append((s, len(ws)))
    return out


B58E_COUNTERS = ('polls', 'calls', 'rows', 'calls_busy', 'latency', 'act', 'act_busy')


def batch_prof(c0, c1, prof):
    """(B58e) The driver's per-step counters of a collect into train_tok's prof (the same progress.csv columns as the
    per-record path: infer_ms from latency / calls, rows_per_call, infer_upd_ms, calls_upd, polls)."""
    for k in B58E_COUNTERS:
        prof[k] += c1[k]-c0[k]
    prof['main_ready'] += c1['latency']-c0['latency']
