"""B58d gate (d1): the native lane service (abp_lanes.h LaneCore, run by the driver) == the Python lane service it
replaces (BlockLaneService), byte for byte, on one scripted request stream.

Two identical worlds (same seeds): a main actor with both teacher generations' lane actors in its arena, the main
rows' histories, CF points captured at two policy versions (two generations). World A serves the lanes in Python
(BlockLaneService, the lane-only path); world B in C++ (the driver's LaneCore with an injected clock: lane_test_intake /
lane_test_tick; the main lineage copied into the native lineage, its points in the native point table). After every
round: the answers (lane_io actions, lane_out value / logp / version / value_global / sequence / cancelled), lane_t (the
context idents), every answer's context (its snapshot window), each lane's lineage (its log's window), the lane steps.
The script's rounds: branch begins at two generations, continuations, a replay row, a repeated request (last answer), a
cancellation (the worker releases the lane's point while the request waits), a begin before its point (deferred,
then served once captured), a begin whose point never comes (an orphan: cancelled), an episode's first row in a lane,
a main version change between rounds (the generations' order), a job the graphs cannot take (forced: kind 5, stepped
in Python, its decisions back to the native lineage). The window rule itself: test_window_rule_matches_python.
Skipped without the extension (no CUDA torch, no g++, ISAAC_BLOCK_CPP_ACTOR=0)."""
import copy
import select
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_block_memory import make_policy
from test_visible_input import synth_rows, synth_pending
from test_lane_batch_smoke import FakeSampler
from test_b52_cpp_actor import EXT, fake_sampler, make_driver_for, frame
from isaac_bridge.block_paged import SlotBlockActor, PagedSnapshot, native_snapshots
from isaac_bridge.block_lanes import BlockLaneService, publish_slot
from isaac_bridge.tok_lanes import LaneService, ACTIVE_S
from isaac_bridge.tok_teacher2 import V2_RING
from isaac_bridge.tok_sampler import TEACH_RING, ST
from isaac_bridge.tok_obs import ROW

pytestmark = pytest.mark.skipif(EXT is None, reason='cpp actor extension unavailable')


def gpu_fake_driver(svc):
    """test_b54_lane_bookkeeping's FakeDriver (the B54 lane jobs stepped by their lane actors, records as kind 4) under
    the actors' autocast: world A's lane service is the B54 driver mode it replaces (deferral, tokens, rings)."""
    from test_b54_lane_bookkeeping import FakeDriver

    class GpuFake(FakeDriver):
        def step(self, k):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return super().step(k)
    return GpuFake(svc)

W, BANK, TAG = 4, 8, 99


def lane_sampler():
    s = FakeSampler(W, D=16)
    s.cfg = SimpleNamespace(workers=W, memory_dim=0, teacher_depths=(2,), teacher_death_depths=(4,))
    s.v2_t = np.zeros((W, V2_RING, s.lane_t.shape[1]), np.float32)
    s.block_arrays = [np.zeros((W, BANK), np.int64), np.zeros(W, np.int64), np.full((W, 6, 2), -1, np.int64),
                      np.zeros(W, np.int64), np.zeros((W, 4), np.int64), np.zeros((W, TEACH_RING), np.int64)]
    s.cf_snap = np.zeros(W, np.int64)
    s.stats = np.zeros((W, max(ST.values())+1))     # (the reclamation reads the teacher rings: none here)
    s.teach_rows = np.zeros((W, TEACH_RING), ROW)
    return s


def world(native, lane_pipes=False):
    torch.manual_seed(77)
    m = make_policy('cuda').eval()
    m1 = copy.deepcopy(m)
    with torch.no_grad():
        for p_ in m1.parameters():
            p_.mul_(1.01)
    main = SlotBlockActor(m, W, graph_entities=128, bf16=True, lane_slots=W, lane_generations=2)
    rng = np.random.default_rng(77)
    nh = len(m.heads)
    d = None
    if native:
        d = make_driver_for(main, fake_sampler(W, nh))
    for t in range(40):   # (the main rows' histories, the Python path)
        rows = frame(rng, W, t, np.arange(W)+1, np.array([int(t == 0)]*W), seed=4000+t)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            main.step(rows, synth_pending(rng, W, m.heads), list(range(W)), ukeys=(np.full(W, t), np.arange(W)))
    s = lane_sampler()
    svc = BlockLaneService(s, m, 'cuda', wait=0, quiet=0, main_actor=main)
    svc.RECLAIM_S = 0.0   # (the contexts' reclamation at every control pass, in both worlds)
    gens = svc.generations
    for g in (0, 1):   # (both generations' lane actors, as precapture_lane_generations)
        gens.slots[g] = copy.deepcopy(m).eval().requires_grad_(False)
        svc.actors[g] = main.lane_actor(gens.slots[g], W, generation=g)
    w = SimpleNamespace(m=m, m1=m1, main=main, s=s, svc=svc, d=d, rng=np.random.default_rng(78), nh=nh, store=main.store)
    if not native:
        w.fake = gpu_fake_driver(svc)
        svc.attach_driver(w.fake)
    if native:
        st = main.store
        cfg = m.block_memory.config
        st.native.lineage_configure(int(cfg.window), int(cfg.block_steps), int(cfg.memory_blocks), nh, int(main.state.slots), W,
                                    [2, 4], 8)
        st.native.lineage_status(int(s.block_arrays[0].ctypes.data))
        for q in range(W):   # (the main lineage into the native one)
            log = main.logs[q]
            st.native.lin_new_head(q, *[int(x) for x in (log.episode if isinstance(log.episode, tuple) else (TAG, q, 1))])
            for i in range(log.n):
                st.native.lin_push(q, int(log.dec[i]), int(log.kvs[i]))
            for i in range(log.nb):   # (a block's push comes with its last decision's: the order does not matter here)
                st.native.lin_push_block(q, int(log.blk[i]), int(log.blk_end[i]))
        svc.attach_lineage(st)
        stream = torch.cuda.Stream()
        sh = dict(rows=int(s.lane_rows.ctypes.data), o_episode=int(ROW.fields['episode'][1]), s_episode=int(ROW.fields['episode'][0].itemsize),
                  io=int(s.lane_io.ctypes.data), io_cols=int(s.lane_io.shape[1]), out=int(s.lane_out.ctypes.data),
                  out_cols=int(s.lane_out.shape[1]), t=int(s.lane_t.ctypes.data), v2_t=int(s.v2_t.ctypes.data), v2_ring=V2_RING,
                  mode=int(s.block_arrays[1].ctypes.data), status=int(s.block_arrays[0].ctypes.data), pts_per=BANK,
                  kept=int(s.block_arrays[4].ctypes.data), kept_cols=4, cf_snap=int(s.cf_snap.ctypes.data))
        cfg_l = dict(lanes=W, tw=int(s.lane_t.shape[1]), wait=0.0, quiet=0.0, active_s=float(ACTIVE_S), defer_s=float(svc.DEFER_S),
                     reclaim_s=0.0, pid=TAG)
        pipes = [socket.socketpair() for _ in range(W)] if lane_pipes else []
        w_pipes = pipes
        d.lanes_configure(cfg_l, sh, [a_.fileno() for a_, _ in pipes], int(stream.stream_id), [int(svc.actors[g].step_graph.owner) for g in (0, 1)],
                          [int(svc.actors[g].base) for g in (0, 1)])
        svc.attach_driver(d)
        svc.attach_native_lanes(d, stream)
        w.stream = stream
        w.pipes = w_pipes
    return w


def capture(w, point, worker, version):
    """A CF point of a main row at a policy version (generation publish included), in both worlds' ways."""
    model = w.m if version == 0 else w.m1
    w.main.version = version
    if w.d is not None:   # (the driver's version is the main actor's: train_tok sets both from one counter)
        w.d.set_version(version)
    if w.d is None:
        ok = w.svc.capture_point(point, w.main.snapshot(worker), model, version)
        w.s.block_arrays[0].flat[point] = 2 if ok else -1
        return ok
    nat = w.store.native
    snap = nat.lin_snap_row(worker, torch.from_numpy(w.main.pending[worker][0].astype(np.int64)), version)
    c = nat.pt_capture(point, snap, version)
    if c <= -2:
        publish_slot(w.svc.generations, model, version, -2-c)
        torch.cuda.synchronize()
        nat.pt_published(-2-c, version)
        c = nat.pt_capture(point, snap, version)
    nat.lin_release(snap)
    assert c in (2, -1)
    return c == 2


def native_tick(w, now):
    """d.lane_test_tick on a thread (GIL released) while this thread answers its lane records (captures / kind 5)."""
    out, err = [], []

    def run():
        try:
            out.append(w.d.lane_test_tick(now))
        except Exception as e:   # noqa: BLE001
            err.append(e)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    t0 = time.perf_counter()
    while th.is_alive():
        rec = w.d.next_record(0.01)
        if rec is not None:
            assert int(rec['lane_src']) == 1, rec['kind']
            w.svc.on_native_lane_record(rec)
        assert time.perf_counter()-t0 < 60, 'native lane tick hung'
    th.join()
    if err:
        raise err[0]
    return out[0]


def request(w, lane, row, pend, seq, begin=0, replay=0, now=0.0):
    w.s.block_arrays[1][lane] = replay
    if w.d is None:
        w.s.request(lane, row, pend, seq=seq, tbegin=begin, now=now)
    else:
        w.s.lane_rows[lane] = row; w.s.lane_io[lane, :5] = pend; w.s.lane_io[lane, 10] = seq
        if begin:
            w.s.lane_io[lane, 11] = begin
        w.d.lane_test_intake(lane, now)


def tick(w, now):
    if w.d is None:   # (B54 driver mode: submit, the fake driver's records, the answers)
        w.svc.DEFER_S = w.defer
        w.svc.tick()
        w.fake.run()
        return w.svc.tick()
    return native_tick(w, now)


def same_snap(a, b, cfg):
    assert a.step == b.step and a.policy_version == b.policy_version
    np.testing.assert_array_equal(np.asarray(a.pending).reshape(-1), np.asarray(b.pending).reshape(-1))
    ra, rb = a.log.branch(a.step, cfg), b.log.branch(b.step, cfg)
    assert ra.offset == rb.offset
    for name, n in (('dec', 'n'), ('kvs', 'n'), ('blk', 'nb'), ('blk_end', 'nb')):
        np.testing.assert_array_equal(getattr(ra, name)[:getattr(ra, n)], getattr(rb, name)[:getattr(rb, n)], err_msg=name)


def compare(A, B, k):
    cfg = A.m.block_memory.config
    np.testing.assert_array_equal(A.s.lane_io[:, 5:10], B.s.lane_io[:, 5:10], err_msg=f'round {k}: actions')
    np.testing.assert_array_equal(A.s.lane_out, B.s.lane_out, err_msg=f'round {k}: lane_out')
    np.testing.assert_array_equal(A.s.lane_t, B.s.lane_t, err_msg=f'round {k}: lane_t')
    for lane in range(W):
        if A.s.lane_t[lane, -1] == 1.0:   # the answer's context
            pr = np.array(A.s.lane_t[lane, :2], np.uint32)
            ident = int(pr[0] | (pr[1] << 16))
            ca = A.svc.contexts.get(ident)
            cb = native_snapshots(B.store, [B.store.native.ctx_get(ident)])[0]
            same_snap(ca, cb, cfg)
    for g, ga in A.svc.actors.items():   # each lane's lineage (log window at its step)
        for lane in range(W):
            la = ga.logs[lane]
            row = ga.base+lane
            end = B.store.native.lin_head_end(row)
            if la.episode is None and la.n == 0:
                assert end in (-1, 0), (g, lane, end)
                continue
            assert end == la.end, (k, g, lane, end, la.end)
            sa = PagedSnapshot(A.store, la, la.end, np.zeros((1, A.nh), np.int64), 0)
            sn = B.store.native.lin_snap_row(row, torch.zeros(A.nh, dtype=torch.long), 0)
            sb = native_snapshots(B.store, [sn])[0]; B.store.native.lin_release(sn)
            same_snap(sa, sb, cfg)
    assert len(A.svc.contexts.values) == int(B.store.native.lin_stats()['contexts']), k   # (after the reclamations)
    np.testing.assert_array_equal(A.s.block_arrays[0], B.s.block_arrays[0], err_msg=f'round {k}: point status')
    va, ra = A.svc.generations.versions, A.svc.generations.refs
    vb, rb = B.store.native.pt_gens()
    assert list(va) == list(vb) and list(ra) == list(rb), (k, va, ra, vb, rb)
    st = B.d.lane_state()
    assert dict(st['lane_steps']) == {int(q): int(v) for q, v in A.svc.lane_steps.items()}, k
    assert dict(st['lane_point']) == {int(q): int(v) for q, v in A.svc.lane_point.items()}, k


def test_native_lane_service_matches_python_lane_service():
    A, B = world(False), world(True)
    for w in (A, B):
        w.defer = 1e9
        capture(w, 0, 0, 0)            # (generation 0: version 0)
        capture(w, 1*BANK+0, 1, 0)
        capture(w, 2*BANK+0, 2, 1)     # (generation 1: version 1)
    rng = np.random.default_rng(91)
    nh = A.nh
    seq = [100]

    def rows_for(lanes, first=()):
        r = synth_rows(rng, len(lanes)); r['first'] = 0; r['episode'] = 7
        for i, q in enumerate(lanes):
            if q in first:
                r['first'][i] = 1; r['episode'][i] = 8
        return r

    def round_(k, reqs, now, events=None, force_python=False):
        """reqs: (lane, begin, replay, repeat): the same rows / pending / sequence numbers in both worlds"""
        rows = rows_for([q for q, *_ in reqs], first=[q for q, b_, r_, rep_, f_ in reqs if f_])
        pend = synth_pending(rng, len(reqs), A.m.heads)
        for w in (A, B):
            if events:
                events(w)
            if B is w:
                w.d.set_lane_force_python(-2 if force_python is True else (-1 if force_python is False else int(force_python)))
        for i, (lane, begin, replay, repeat, _) in enumerate(reqs):
            sq = seq[0]-1 if repeat else seq[0]
            if not repeat:
                seq[0] += 1
            for w in (A, B):
                request(w, lane, rows[i], pend[i], sq, begin, replay, now)
        for w in (A, B):
            tick(w, now)
        compare(A, B, k)

    t = 1.0
    round_(1, [(0, 1, 0, 0, 0), (1, BANK+1, 0, 0, 0), (2, 2*BANK+1, 0, 0, 0)], t)                  # begins, two generations
    round_(2, [(0, 0, 0, 0, 0), (1, 0, 1, 0, 0), (2, 0, 0, 0, 0)], t+1)                             # continuations, a replay row
    round_(3, [(0, 0, 0, 1, 0), (2, 0, 0, 0, 0)], t+2)                                              # a repeated request
    round_(4, [(0, 0, 0, 0, 0), (1, 0, 0, 0, 0)], t+3,
           events=lambda w: w.s.block_arrays[0].__setitem__((1, 0), 3))                             # lane 1's point released: cancelled
    assert A.s.lane_out[1, 5] == -1
    round_(5, [(3, 3*BANK+1, 0, 0, 0)], t+4)                                                        # its point not captured: deferred
    assert A.s.lane_out[3, 4] == 0
    for w in (A, B):
        capture(w, 3*BANK, 3, 1)
    round_(6, [(2, 0, 0, 0, 0)], t+5)                                                               # (the deferred one served too)
    assert A.s.lane_out[3, 4] != 0
    round_(7, [(3, 0, 0, 0, 1), (2, 0, 0, 0, 0)], t+6,
           events=lambda w: setattr(w.main, 'version', 1) or (w.d.set_version(1) if w.d is not None else None))  # first row; main version 1
    round_(8, [(2, 0, 0, 0, 0), (3, 0, 0, 0, 0)], t+7, force_python=True)                           # kind 5: stepped in Python
    A.defer = 0.0
    B.d.set_lane_defer(0.0)
    round_(9, [(0, 5, 0, 0, 0), (2, 0, 0, 0, 0)], t+8)                                              # a point that never comes
    assert A.s.lane_out[0, 5] == -1 and A.svc.stat.get('lane_orphan_requests', 0) == 1 and B.d.lane_counters()['orphan'] == 1
    # both generations pinned (versions 0 and 1): a capture at version 2 refused in both worlds (status -1)
    assert capture(A, 1*BANK+1, 1, 2) is False and capture(B, 1*BANK+1, 1, 2) is False
    def ver(v):
        return lambda w: setattr(w.main, 'version', v) or (w.d.set_version(v) if w.d is not None else None)
    for w in (A, B):   # (lane 1 begins again at a new point of version 0: its generation still holds it)
        assert capture(w, BANK+3, 1, 0)
    round_(10, [(1, BANK+4, 0, 0, 0), (2, 0, 0, 0, 0)], t+9, events=ver(2))                       # version 2: lanes on v0 / v1 points
    round_(11, [(1, 0, 0, 0, 0), (2, 0, 0, 0, 0), (3, 0, 0, 0, 0)], t+10, events=ver(3),
           force_python=0)                                                                           # version 3; gen 0 in Python, gen 1 native
    for w in (A, B):   # (a point released and a new capture in the same round as the reclamation)
        w.s.block_arrays[0][2, 0] = 3
    round_(12, [(1, 0, 0, 0, 0), (3, 0, 0, 0, 0)], t+11, events=lambda w: capture(w, 2*BANK+2, 2, 3))
    c = B.d.lane_counters()
    assert c['python_jobs'] >= 2 and c['rounds'] >= 11 and c['begins'] >= 5


def test_window_rule_matches_python():
    """LaneCore.due / the take's seen times == LaneService.due on random arrival / check times (wait, quiet, ACTIVE_S)."""
    rng = np.random.default_rng(5)
    for wait, quiet in ((0.002, 0.0005), (0.002, 0.0), (0.0, 0.0005), (0.004, 0.001)):
        s = FakeSampler(W)
        svc = SimpleNamespace(s=s, wait=wait, quiet=quiet, seen=np.full(W, -1e9))
        probe = EXT.LaneWindowProbe(W, wait, quiet, ACTIVE_S)
        now = 10.0
        for _ in range(400):
            now += float(rng.exponential(0.0007))
            if rng.random() < 0.6:
                lane = int(rng.integers(0, W))
                s.arrive(lane, now); probe.intake(lane, now)
            due_py = LaneService.due(svc, now)
            assert due_py == probe.due(now), (wait, quiet, now)
            if due_py:
                idx = s.take_lanes(); svc.seen[idx] = now
                assert list(idx) == probe.take(now)


def test_policy_version_guard():
    """The driver's version and the main actor's are one counter set in two places: check_versions stops a run where
    only one of them was set."""
    from isaac_bridge.block_driver import check_versions
    torch.manual_seed(3)
    m = make_policy('cuda').eval()
    main = SlotBlockActor(m, 2, graph_entities=128, bf16=True, lane_slots=0)
    d = make_driver_for(main, fake_sampler(2, len(m.heads)))
    d.set_version(4); main.version = 4
    check_versions(d, main)
    main.version = 5   # (only one set)
    with pytest.raises(RuntimeError, match='out of step'):
        check_versions(d, main)
    d.set_version(5)
    check_versions(d, main)


def test_native_lanes_only_mode_matches_the_harness():
    """d1 while the actor waits for the learner: the driver in main_hold runs the lane passes itself, the requests
    through the lanes' pipes (real socketpairs), each round's messages sent while the driver is paused (one round);
    the answers equal the harness's (lane_test_tick, itself equal to the Python service)."""
    H, D = world(True), world(True, lane_pipes=True)
    for w in (H, D):
        w.defer = 1e9
        capture(w, 0, 0, 0); capture(w, BANK, 1, 0); capture(w, 2*BANK, 2, 1)
    D.d.start(); D.d.set_main_hold(True)
    rng = np.random.default_rng(93)
    seq = [500]
    msg = b'\x00\x00\x00\x010'
    try:
        for k, reqs in enumerate(([(0, 1), (1, BANK+1), (2, 2*BANK+1)], [(0, 0), (1, 0), (2, 0)], [(0, 0), (2, 0)],
                                  [(1, 0), (2, 0)])):
            rows = synth_rows(rng, len(reqs)); rows['first'] = 0; rows['episode'] = 7
            pend = synth_pending(rng, len(reqs), H.m.heads)
            D.d.pause()
            for i, (lane, begin) in enumerate(reqs):
                for w in (H, D):
                    w.s.lane_rows[lane] = rows[i]; w.s.lane_io[lane, :5] = pend[i]; w.s.lane_io[lane, 10] = seq[0]
                    if begin:
                        w.s.lane_io[lane, 11] = begin
                H.d.lane_test_intake(lane, 1.0+k)
                D.pipes[lane][1].sendall(msg)
                seq[0] += 1
            D.d.resume()
            native_tick(H, 1.0+k)
            waiting = {D.pipes[lane][1]: lane for lane, _ in reqs}
            t0 = time.perf_counter()
            while waiting:   # (the replies; lane records answered meanwhile)
                rd, _, _ = select.select(list(waiting), [], [], 0.01)
                for sk in rd:
                    got = b''
                    while len(got) < 5:
                        got += sk.recv(5-len(got))
                    assert got == b'\x00\x00\x00\x01a'
                    del waiting[sk]
                rec = D.d.next_record(0.0)
                if rec is not None:
                    assert int(rec['lane_src']) == 1, rec['kind']
                    D.svc.on_native_lane_record(rec)
                assert time.perf_counter()-t0 < 60, 'lanes-only answers missing'
            np.testing.assert_array_equal(H.s.lane_io[:, 5:10], D.s.lane_io[:, 5:10], err_msg=f'round {k}')
            np.testing.assert_array_equal(H.s.lane_out, D.s.lane_out, err_msg=f'round {k}')
            np.testing.assert_array_equal(H.s.lane_t, D.s.lane_t, err_msg=f'round {k}')
    finally:
        D.d.set_main_hold(False); D.d.pause(); D.d.stop()
    assert D.d.last_error() == '' and D.d.lane_error() == ''
