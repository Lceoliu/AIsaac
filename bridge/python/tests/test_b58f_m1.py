"""d2 M1 gate: a lane job of the main version rides the driver's next main batch (one replay with the main rows),
byte for byte the Python B54 merged path; every other case lane-only at once.

Two worlds as test_b58d_lanes (same seeds): A serves the lanes in Python (BlockLaneService in driver mode) and steps a
round's job merged with one main row exactly as the B54 driver's merged batch does (main.step over [main row, lane
rows] with the lane actor as the lanes' owner, the main row's uniforms 0.5, the lanes' CRN uniforms); B runs the real
driver thread with the native lane service and M1, the lane requests through real pipes, the main row through the
main worker's pipe. set_m1_hold keeps a queued job waiting for the main row the test sends (no idle flush), so every
round is deterministic. After every round: the answers, lane_t, the answers' contexts, the lanes' lineage, the point
table, the generations (test_b58d_lanes.compare) and the main row's outputs.

Rounds: begins (two lanes, generation 0 at the main version: merged), continuations with a replay row (merged), a
point released while its job waits for the main batch (the job is in flight: answered, the point gone after it; its
generation stays pinned by the waiting job), the main version changed while the job waits (taken weights: the job
goes lane-only on its generation's graph, as A's lane-only call), a hand-off with a job waiting (the driver paused,
the weights taken, the driver in main_hold: the job lane-only at once), an older generation's job (lane-only, never
queued). Also: the restore's order control (the restore on the lane stream behind a long kernel, nothing waiting:
the merged answers must differ), ISAAC_BLOCK_LANE_MERGE=0 (set_m1_merge(False): nothing merged), and the M1
counters. Skipped without the extension."""
import select
import time

import numpy as np
import pytest
import torch

from test_b58d_lanes import world, capture, compare, W, BANK
from test_b52_cpp_actor import EXT, frame
from test_visible_input import synth_rows, synth_pending
from isaac_bridge.tok_policy import crn_uniforms

pytestmark = pytest.mark.skipif(EXT is None, reason='cpp actor extension unavailable')

MSG = b'\x00\x00\x00\x010'
MAIN = 3   # (the main worker whose rows ride with the lanes; no lane of the script begins at its points)


def pair():
    A, B = world(False), world(True, lane_pipes=True)
    A.defer = 1e9
    B.d.set_fixed_uniform(0.5)
    B.d.set_m1_hold(True); B.d.set_m1_test_capture(True)
    return A, B


def run_merged(A, mrow):
    """world A: the round's one job stepped merged with the main row (the B54 driver's merged batch in Python)"""
    fake, svc = A.fake, A.svc
    assert len(fake.queue) == 1, len(fake.queue)
    j = fake.queue.pop(0)
    gen, positions, version, lanes, rows, pend, keys, replay = svc.drv_jobs[j['job']][:8]
    actor = next(a for a in svc.actors.values() if id(a) == j['owner'])
    main = A.main
    heads, nh = sum(A.m.heads), A.nh
    bs = actor.model.block_memory.config.block_steps
    step_b = np.array([int(actor.state.step_host[actor.base+q]) for q in lanes])
    u = torch.cat((torch.full((1, heads), .5), torch.as_tensor(crn_uniforms(np.asarray(keys[0]), np.asarray(keys[1]), heads))))
    allr = np.concatenate((mrow, rows))
    allp = np.concatenate((np.zeros((1, nh), np.int64), pend))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        main.step(allr, allp, [MAIN, *[int(q) for q in lanes]], owners=[main]+[actor]*len(lanes), uniform=u,
                  evaluate=np.concatenate(([False], np.asarray(replay, bool))))
    main_out = main.host_output.copy()
    dec, kvs, bid = [], [], []
    for i_, q in enumerate(lanes.tolist()):   # (on_lane_record pushes the step's entries)
        log = actor.logs[q]
        dec.append(int(log.dec[log.n-1])); kvs.append(int(log.kvs[log.n-1])); log.n -= 1
        if (step_b[i_]+1) % bs == 0:
            bid.append(int(log.blk[log.nb-1])); log.nb -= 1
        else:
            bid.append(-1)
    first = rows['first'].astype(np.uint8)
    if first.any():
        step_b = np.where(first > 0, 0, step_b)
    svc.on_lane_record(dict(kind=4, job=j['job'], owner=j['owner'], merged=1, version=j['version'],
                            out=torch.from_numpy(actor.host_output.copy()), step_before=torch.from_numpy(step_b),
                            first=torch.from_numpy(first), complete=torch.from_numpy(((step_b+1) % bs == 0).astype(np.uint8)),
                            dec=torch.tensor(dec), kvs=torch.tensor(kvs), bid=torch.tensor(bid),
                            pending_before=torch.from_numpy(pend.copy())))
    return main_out


def records(B, until, t_max=60):
    """B's driver records until until(rec) is true for one (kind 0 / lane records answered meanwhile)"""
    t0 = time.perf_counter()
    while True:
        assert time.perf_counter()-t0 < t_max, (B.d.last_error(), B.d.lane_error(), B.d.counters())
        rec = B.d.next_record(0.01)
        if rec is None:
            continue
        if int(rec['lane_src']):
            B.svc.on_native_lane_record(rec)
            continue
        if int(rec['kind']) == 1:   # (a main / merged graph to capture: the batch is retried)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                assert B.main.step_graph.capture_key(rec['cap'], rec['widths']) is not None
            B.d.resume()
            continue
        if until(rec):
            return rec


def wait_for(B, cond, t_max=60):
    t0 = time.perf_counter()
    while not cond():
        assert time.perf_counter()-t0 < t_max, (B.d.last_error(), B.d.lane_error(), B.d.counters())
        rec = B.d.next_record(0.005)
        if rec is not None:
            assert int(rec['lane_src']), int(rec['kind'])
            B.svc.on_native_lane_record(rec)


def replies(B, lanes, t_max=60):
    waiting = {B.pipes[q][1]: q for q in lanes}
    t0 = time.perf_counter()
    while waiting:
        rd, _, _ = select.select(list(waiting), [], [], 0.01)
        for sk in rd:
            got = b''
            while len(got) < 5:
                got += sk.recv(5-len(got))
            assert got == b'\x00\x00\x00\x01a'
            del waiting[sk]
        rec = B.d.next_record(0.0)
        if rec is not None:
            assert int(rec['lane_src']), int(rec['kind'])
            B.svc.on_native_lane_record(rec)
        assert time.perf_counter()-t0 < t_max, ('lane replies missing', B.d.last_error(), B.d.counters())


class Script:
    def __init__(self, A, B):
        self.A, self.B = A, B
        self.rng = np.random.default_rng(95)
        self.seq = 700
        self.k = 0
        self.slot = 0

    def lane_rows(self, reqs):
        r = synth_rows(self.rng, len(reqs)); r['first'] = 0; r['episode'] = 7
        return r, synth_pending(self.rng, len(reqs), self.A.m.heads)

    def main_row(self):
        r = frame(self.rng, 1, 0, np.array([50+self.k]), np.array([1]), seed=9000+self.k)   # (a new episode: pending 0)
        return r

    @staticmethod
    def put(w, lane, row, pend, seq, begin, replay):
        w.s.block_arrays[1][lane] = replay
        w.s.lane_rows[lane] = row; w.s.lane_io[lane, :5] = pend; w.s.lane_io[lane, 10] = seq
        if begin:
            w.s.lane_io[lane, 11] = begin

    def round(self, reqs, mode='merged', events=None, b_events=None, a_events=None):
        """reqs: (lane, begin, replay). mode: 'merged' (A steps the job merged with the main row) or 'lane' (A's job
        lane-only; the main row, when B sends one, stepped alone before it, as B's main step precedes its lane-only
        calls)."""
        A, B = self.A, self.B
        self.k += 1
        rows, pend = self.lane_rows(reqs)
        mrow = self.main_row()
        for i, (lane, begin, replay) in enumerate(reqs):
            self.put(B, lane, rows[i], pend[i], self.seq, begin, replay)
            A.s.block_arrays[1][lane] = replay
            A.s.request(lane, rows[i], pend[i], seq=self.seq, tbegin=begin, now=float(self.k))
            self.seq += 1
        A.svc.DEFER_S = A.defer
        A.svc.tick()   # (A: the round submitted, its jobs queued in the fake driver)
        m0 = lo_sum(B.d.counters())
        B.d.pause()
        for lane, *_ in reqs:
            B.pipes[lane][1].sendall(MSG)
        B.d.resume()   # (B: the driver submits the round; a job of the main version waits for a main batch)
        wait_for(B, lambda: B.d.counters()['m1_waiting'] > 0 or lo_sum(B.d.counters()) > m0)
        for w in (A, B):
            if events is not None:
                events(w)
        if b_events is not None:
            b_events(B)
        if a_events is not None:
            a_events(A)
        send_main = B.d.counters()['m1_waiting'] > 0
        assert send_main or mode == 'lane'
        main_out = None
        if mode == 'merged':
            main_out = run_merged(A, mrow)
        else:
            if send_main:
                heads = sum(A.m.heads)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    A.main.step(mrow, np.zeros((1, A.nh), np.int64), [MAIN], uniform=torch.full((1, heads), .5))
                main_out = A.main.host_output.copy()
            A.fake.run()
        A.svc.tick()   # (A: control, the round finished)
        if send_main:   # (B: the main row; its batch takes the waiting job or runs it lane-only after its reply)
            s_ = B.ms
            s_.rows[2*MAIN+self.slot] = mrow[0]
            s_._pairs[MAIN][1].sendall(b'\x00\x00\x00\x01' + (b'1' if self.slot else b'0'))
            r0 = records(B, lambda r: int(r['kind']) == 0)
            got = b''
            while len(got) < 5:
                got += s_._pairs[MAIN][1].recv(5-len(got))
            self.slot ^= 1
            np.testing.assert_array_equal(r0['out'].numpy(), main_out, err_msg=f'round {self.k}: main row')
        replies(B, [q for q, *_ in reqs])
        compare(A, B, self.k)
        return B.d.counters()


def lo_sum(c):
    return sum(c[k] for k in ('lo_old_jobs', 'lo_switch_jobs', 'lo_refused_jobs', 'lo_idle_jobs', 'lo_other_jobs'))


def test_m1_merged_rounds_match_python_merged_path():
    A, B = pair()
    for w in (A, B):
        capture(w, 0, 0, 0)            # (generation 0 at version 0 = the main version)
        capture(w, BANK, 1, 0)
    B.d.start()
    sc = Script(A, B)
    try:
        c = sc.round([(0, 1, 0), (1, BANK+1, 0)])                       # begins: merged
        assert c['m1_jobs'] == 1 and c['m1_rows'] == 2
        c = sc.round([(0, 0, 0), (1, 0, 1)])                            # continuations, a replay row: merged
        assert c['m1_jobs'] == 2

        def release_1(w):   # (worker 1 releases its point while lane 1's job waits for the main batch)
            w.s.block_arrays[0][1, 0] = 3

        def pinned(B_):     # (point 1 released while the job waits: refs = point 0 + the waiting job's pin)
            wait_for(B_, lambda: B_.store.native.pt_gens()[1][0] == 2)
        c = sc.round([(0, 0, 0), (1, 0, 0)], events=release_1, b_events=pinned)   # in flight: answered, released after
        assert c['m1_jobs'] == 3 and B.store.native.pt_gens()[1][0] == 1

        def ver(v):
            return lambda w: setattr(w.main, 'version', v) or (w.d.set_version(v) if w.d is not None else None)
        c = sc.round([(0, 0, 0)], mode='lane', events=ver(1))               # the version taken while it waits
        assert c['lo_switch_jobs'] == 1 and c['m1_jobs'] == 3
        for w in (A, B):
            capture(w, 2*BANK, 2, 1)   # (generation 1 at version 1 = the main version)

        def handoff(B_):    # (the collect's end with the job waiting: paused, the weights taken, lanes only)
            B_.d.pause(); B_.d.set_version(2); B_.main.version = 2; B_.d.set_main_hold(True); B_.d.resume()
            wait_for(B_, lambda: B_.d.counters()['m1_waiting'] == 0)
        c = sc.round([(2, 2*BANK+1, 0)], mode='lane', b_events=handoff, a_events=ver(2))
        assert c['lo_switch_jobs'] == 2 and c['m1_waiting'] == 0
        B.d.pause(); B.d.set_main_hold(False); B.d.resume()
        c = sc.round([(0, 0, 0), (2, 0, 0)], mode='lane')               # older generations: lane-only, never queued
        assert c['lo_old_jobs'] >= 2 and c['m1_jobs'] == 3
    finally:
        B.d.pause(); B.d.stop()
    assert B.d.last_error() == '' and B.d.lane_error() == ''
    t = B.d.lane_timing()
    assert t['m1_wait_s_n'] == 3


def test_m1_restore_order_control_fails():
    """The restore's copies on the lane stream behind a long kernel, nothing ordering the merged step after them: the
    merged answers differ from the Python merged path (M1 does the restore on the main stream: the first round of the
    test above)."""
    A, B = pair()
    for w in (A, B):
        capture(w, 0, 0, 0); capture(w, BANK, 1, 0)
    B.d.set_restore_test(1)
    B.d.start()
    sc = Script(A, B)
    try:
        with pytest.raises(AssertionError, match='round 1'):
            sc.round([(0, 1, 0), (1, BANK+1, 0)])
    finally:
        torch.cuda.synchronize()
        B.d.pause(); B.d.stop()


def test_m1_off_is_d1():
    """set_m1_merge(False) (ISAAC_BLOCK_LANE_MERGE=0): the same round lane-only, as world A's lane-only call."""
    A, B = pair()
    B.d.set_m1_merge(False)
    for w in (A, B):
        capture(w, 0, 0, 0); capture(w, BANK, 1, 0)
    B.d.start()
    sc = Script(A, B)
    try:
        c = sc.round([(0, 1, 0), (1, BANK+1, 0)], mode='lane')
        assert c['m1_jobs'] == 0 and c['lo_other_jobs'] == 1 and c['m1_waiting'] == 0
    finally:
        B.d.pause(); B.d.stop()
    assert B.d.last_error() == ''
