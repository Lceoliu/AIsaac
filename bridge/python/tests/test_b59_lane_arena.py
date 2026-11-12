"""Lane GPU, stage 1: the teacher generations' lane rows in arenas of their own (SlotBlockActor lane_arena='own', the
driver's set_lane_arena), on the actor's device, == the lane rows in the main arena (the current d2), byte for byte.

Two worlds of test_b58d_lanes (same seeds, both native): S with the lane rows in the main arena, O with an arena per
generation (row ids past the main arena: lin_base; the graphs address local rows). On test_b58d's scripted rounds with
the injected clock (lane_test_tick: begins at two generations, continuations, a replay row, a repeat, a cancellation, a
deferral, an orphan, an episode's first row, main version changes, kind 5 jobs stepped in Python, a capture refused)
and then through the lane thread (d2, real lane pipes, lanes_start), after every round: the answers (lane_io actions,
lane_out, lane_t), every answer's context, each lane's lineage window, the point status, the generations, the lane
bookkeeping, every lane row's arena contents (kv and every BlockSlotState row tensor, S's row base + lane against O's
row lane) and its step counter, and the page store (decisions' K/V, z, metadata; blocks, their sources). O: M1 off.
Skipped without the extension."""
import time

import numpy as np
import pytest
import torch

from test_b58d_lanes import world, capture, run_script, same_snap, W, BANK
from test_b52_cpp_actor import EXT
from test_visible_input import synth_rows, synth_pending
from isaac_bridge.block_paged import native_snapshots

pytestmark = pytest.mark.skipif(EXT is None, reason='cpp actor extension unavailable')

MSG = b'\x00\x00\x00\x010'


def lane_rows(w, g, lane):
    """generation g's lane row: (its arena, the arena row)"""
    a = w.svc.actors[g]
    return a.state, a.base+lane


def same_arenas(S, O, k):
    torch.cuda.synchronize()
    for g in (0, 1):
        for lane in range(W):
            (ss, rs), (so, ro) = lane_rows(S, g, lane), lane_rows(O, g, lane)
            assert so is not ss and ss is S.main.state and so is not O.main.state
            assert int(ss.step_host[rs]) == int(so.step_host[ro]), (k, g, lane)
            assert torch.equal(ss.kv.select(2, rs).cpu(), so.kv.select(2, ro).cpu()), (k, g, lane, 'kv')
            for name in ss._rows:
                assert torch.equal(getattr(ss, name)[rs].cpu(), getattr(so, name)[ro].cpu()), (k, g, lane, name)
            rS = S.d.get_ring(S.svc.actors[g].lin_base+lane); rO = O.d.get_ring(O.svc.actors[g].lin_base+lane)
            assert torch.equal(rS, rO), (k, g, lane, 'ring')


def same_pages(S, O, k):
    a, b = S.store, O.store
    np.testing.assert_array_equal(a.d_used, b.d_used, err_msg=f'round {k}: d_used')
    np.testing.assert_array_equal(a.b_used, b.b_used, err_msg=f'round {k}: b_used')
    d = np.flatnonzero(a.d_used).astype(np.int64)
    np.testing.assert_array_equal(a.d_meta[d], b.d_meta[d], err_msg=f'round {k}: d_meta')
    if len(d):
        assert torch.equal(a.kv(d).view(torch.uint8) if a.kv_dtype == torch.bfloat16 else a.kv(d),
                           b.kv(d).view(torch.uint8) if b.kv_dtype == torch.bfloat16 else b.kv(d)), k
        assert torch.equal(a.z(d), b.z(d)), k
    bl = np.flatnonzero(a.b_used).astype(np.int64)
    for name in ('b_meta', 'b_src', 'b_srcver'):
        np.testing.assert_array_equal(getattr(a, name)[bl], getattr(b, name)[bl], err_msg=f'round {k}: {name}')
    if len(bl):
        assert torch.equal(a.blocks(bl), b.blocks(bl)), k


def compare_native(S, O, k):
    cfg = S.m.block_memory.config
    np.testing.assert_array_equal(S.s.lane_io[:, 5:10], O.s.lane_io[:, 5:10], err_msg=f'round {k}: actions')
    np.testing.assert_array_equal(S.s.lane_out, O.s.lane_out, err_msg=f'round {k}: lane_out')
    np.testing.assert_array_equal(S.s.lane_t, O.s.lane_t, err_msg=f'round {k}: lane_t')
    for lane in range(W):
        if S.s.lane_t[lane, -1] == 1.0:
            pr = np.array(S.s.lane_t[lane, :2], np.uint32)
            ident = int(pr[0] | (pr[1] << 16))
            ca = native_snapshots(S.store, [S.store.native.ctx_get(ident)])[0]
            cb = native_snapshots(O.store, [O.store.native.ctx_get(ident)])[0]
            same_snap(ca, cb, cfg)
    for g in (0, 1):
        for lane in range(W):
            rs, ro = S.svc.actors[g].lin_base+lane, O.svc.actors[g].lin_base+lane
            es, eo = S.store.native.lin_head_end(rs), O.store.native.lin_head_end(ro)
            assert es == eo, (k, g, lane, es, eo)
            if es <= 0:
                continue
            z = torch.zeros(S.nh, dtype=torch.long)
            a_ = S.store.native.lin_snap_row(rs, z, 0); b_ = O.store.native.lin_snap_row(ro, z, 0)
            sa = native_snapshots(S.store, [a_])[0]; sb = native_snapshots(O.store, [b_])[0]
            S.store.native.lin_release(a_); O.store.native.lin_release(b_)
            same_snap(sa, sb, cfg)
    assert int(S.store.native.lin_stats()['contexts']) == int(O.store.native.lin_stats()['contexts']), k
    np.testing.assert_array_equal(S.s.block_arrays[0], O.s.block_arrays[0], err_msg=f'round {k}: point status')
    assert [list(x) for x in S.store.native.pt_gens()] == [list(x) for x in O.store.native.pt_gens()], k
    a_, b_ = S.d.lane_state(), O.d.lane_state()
    for key in ('lane_steps', 'lane_point', 'cancelled'):
        assert dict(a_[key]) == dict(b_[key]), (k, key)
    same_arenas(S, O, k)
    same_pages(S, O, k)


def worlds(pipes=False):
    S, O = world(True, lane_pipes=pipes, lane_arena='shared'), world(True, lane_pipes=pipes, lane_arena='own')
    assert S.main.lane_arena == 'shared' and O.main.lane_arena == 'own'
    assert S.d.lane_arena_count() == 0 and O.d.lane_arena_count() == 2
    assert O.main.state.slots == W+1 and O.main.lin_rows == W+1+2*W   # (the main arena: main rows and its scratch row)
    assert [O.svc.actors[g].lin_base for g in (0, 1)] == [W+1, 2*W+1]
    assert O.d.lane_device() == S.d.lane_device() == torch.cuda.current_device()
    return S, O


def test_own_lane_arenas_match_shared_on_the_scripted_rounds():
    S, O = worlds()
    run_script(S, O, compare_native)
    cs, co = S.d.lane_counters(), O.d.lane_counters()
    for key in ('rounds', 'rows', 'jobs', 'job_rows', 'restores', 'python_jobs', 'captures', 'begins', 'orphan', 'deferred'):
        assert cs[key] == co[key], (key, cs[key], co[key])
    assert co['python_jobs'] >= 2 and co['restores'] >= 5


def serve(w, lanes, t_max=60):
    """the lane thread's answers to the requests just sent: the replies read, its records answered meanwhile"""
    import select
    waiting = {w.pipes[lane][1]: lane for lane in lanes}
    t0 = time.perf_counter()
    while waiting:
        rd, _, _ = select.select(list(waiting), [], [], 0.01)
        for sk in rd:
            got = b''
            while len(got) < 5:
                got += sk.recv(5-len(got))
            assert got == b'\x00\x00\x00\x01a'
            del waiting[sk]
        rec = w.d.next_record(0.0)
        if rec is not None:
            assert int(rec['lane_src']) == 1, rec['kind']
            w.svc.on_native_lane_record(rec)
        assert time.perf_counter()-t0 < t_max, ('lane answers missing', w.d.lane_error())


def test_own_lane_arenas_match_shared_through_the_lane_thread():
    """d2: the lane thread serves real lane pipes in both worlds (one round at a time: the requests written, the replies
    awaited); a kind 5 round (every job to Python) in the middle; the same comparisons after every round."""
    from isaac_bridge import block_driver as bdrv
    S, O = worlds(pipes=True)
    for w in (S, O):
        w.defer = 1e9
        capture(w, 0, 0, 0); capture(w, BANK, 1, 0); capture(w, 2*BANK, 2, 1)
    rng = np.random.default_rng(95)
    seq = [700]
    for w in (S, O):
        w.d.lanes_start()
    try:
        for k, (reqs, force) in enumerate((([(0, 1), (1, BANK+1), (2, 2*BANK+1)], -1), ([(0, 0), (1, 0), (2, 0)], -1),
                                           ([(0, 0), (2, 0)], -2), ([(1, 0), (2, 0)], -1), ([(0, 0), (1, 0), (2, 0)], 1))):
            rows = synth_rows(rng, len(reqs)); rows['first'] = 0; rows['episode'] = 7
            pend = synth_pending(rng, len(reqs), S.m.heads)
            for w in (S, O):
                bdrv.lanes_pause_serving(w.d, w.svc.on_native_lane_record)
                w.d.set_lane_force_python(force)
                for i, (lane, begin) in enumerate(reqs):
                    w.s.lane_rows[lane] = rows[i]; w.s.lane_io[lane, :5] = pend[i]; w.s.lane_io[lane, 10] = seq[0]+i
                    if begin:
                        w.s.lane_io[lane, 11] = begin
                    w.pipes[lane][1].sendall(MSG)
                w.d.lanes_resume()
                serve(w, [lane for lane, _ in reqs])
            seq[0] += len(reqs)
            for w in (S, O):   # (the comparisons with both lane threads parked)
                bdrv.lanes_pause_serving(w.d, w.svc.on_native_lane_record)
            compare_native(S, O, k)
            for w in (S, O):
                w.d.lanes_resume()
    finally:
        for w in (S, O):
            bdrv.lanes_pause_serving(w.d, w.svc.on_native_lane_record)
            w.d.lanes_stop()
    assert S.d.lane_error() == '' and O.d.lane_error() == ''
    assert O.d.lane_counters()['python_jobs'] >= 1 and O.d.lane_counters()['jobs'] == S.d.lane_counters()['jobs']


def test_own_lane_arena_turns_m1_off():
    """d1 with M1 on (the driver thread runs the lane service): with lane arenas of their own a job of the main version
    never waits for a main batch; it goes lane-only at once, as the Python lane-only call (test_b58f_m1's round)."""
    from test_b58f_m1 import pair, Script
    A, B = pair('own')
    assert B.d.lane_arena_count() == 2
    for w in (A, B):
        capture(w, 0, 0, 0); capture(w, BANK, 1, 0)
    B.d.start()
    try:
        c = Script(A, B).round([(0, 1, 0), (1, BANK+1, 0)], mode='lane')
        assert c['m1_jobs'] == 0 and c['m1_waiting'] == 0 and c['lo_other_jobs'] == 1, c
    finally:
        B.d.pause(); B.d.stop()
    assert B.d.last_error() == ''


def test_set_lane_arena_checks():
    S, O = worlds()
    with pytest.raises(RuntimeError, match='lanes_configure first'):
        S.d.set_lane_arena(5, [S.svc.actors[0].state.kv], [2], [0.0], 0, 0, 1, 0)
    with pytest.raises(RuntimeError, match='another lane arena'):
        a = O.svc.actors[1]
        O.d.set_lane_arena(0, [a.state.kv], [2], [0.0], int(a.state.step_host.ctypes.data), int(O.svc.actors[0].lin_base), W+1,
                           int(a.state.kv.device.index))
