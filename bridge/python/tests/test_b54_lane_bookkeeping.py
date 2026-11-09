"""B54 bookkeeping without a GPU or a game: the Python side of the driver path, run on CPU.

The two B54 smoke failures were in the Python bookkeeping (serve_wait with a round in flight; the per-collect
counter deltas), each found only after a 10-minute compile on the 4090. These run anywhere:

- block_driver.collect_counters (the per-collect deltas train_tok writes): fake driver counters (the lane-only rows
  histogram a list), two collects, the hand-off's pause excluded, the Python-path lane counters and their share,
  every DRV_FIELDS column written by csv.DictWriter;
- BlockLaneService in driver mode with a fake driver (the lane actors stepped on CPU by the fake, records as the
  driver's kind 4): tick (submit, no wait, answers from the records), the control's context reclamation between the
  records and the answer (B54 smoke 3: an ident retired before lane_t named it), serve_wait with the round in flight
  (dropped from the driver, stepped on the Python path) and with new rounds (the Python path, the rings re-set before
  the next driver job), the counters; every answer equal to the same requests served by a service without a driver.
"""
import csv
import io
from types import SimpleNamespace

import numpy as np
import torch

from test_block_memory import make_policy
from test_block_lanes import ring_sampler
from test_visible_input import synth_rows, synth_pending
from isaac_bridge.block_driver import collect_counters, DRV_FIELDS
from isaac_bridge.block_lanes import BlockLaneService
from isaac_bridge.block_paged import SlotBlockActor


def counters(**kw):
    c = dict(steps=0, calls=0, rows=0, terminal=0, sync_s=0.0, pages_s=0.0, host_s=0.0, lane_jobs=0, lane_rows=0,
             merged_rows=0, merged_calls=0, lane_calls=0, captures=0, lane_python=0, pauses=0, pause_s=0.0, lane_s=0.0,
             lane_hist=[0]*6)
    c.update(kw)
    return c


def test_collect_counters_and_progress_columns():
    stat, mark = {}, [{}]
    c0 = counters()
    c1 = counters(pause_s=5.25, pauses=3, lane_jobs=10, lane_rows=40, merged_rows=12, merged_calls=4, lane_calls=6,
                  captures=0, lane_s=1.5, lane_hist=[2, 1, 1, 1, 1, 0])
    stat['py_lane_rounds'], stat['py_lane_rows'], stat['py_lane_flushed_rows'] = 2, 8, 2
    line = collect_counters(c0, c1, 5.0, stat, mark)
    assert abs(stat['drv_pause_s']-0.25) < 1e-9 and stat['drv_pauses'] == 3
    assert stat['drv_lane_hist'] == '2/1/1/1/1/0' and stat['drv_lane_rows'] == 40
    assert stat['drv_py_lane_rows'] == 8 and stat['drv_py_lane_flushed_rows'] == 2
    assert abs(stat['drv_py_lane_share']-10/50) < 1e-12
    assert 'lane jobs 10' in line and '20.0% of lane rows' in line
    # the next collect: deltas against the last one; no Python-path rows meanwhile
    c2 = counters(pause_s=9.0, pauses=4, lane_jobs=15, lane_rows=60, merged_rows=12, merged_calls=4, lane_calls=11,
                  lane_s=2.0, lane_hist=[5, 3, 1, 1, 1, 0])
    collect_counters(c1, c2, 3.0, stat, mark)
    assert abs(stat['drv_pause_s']-0.75) < 1e-9 and stat['drv_lane_hist'] == '3/2/0/0/0/0'
    assert stat['drv_py_lane_rows'] == 0 and stat['drv_py_lane_share'] == 0.0
    # a hand-off longer than the paused seconds: never negative
    collect_counters(c2, c2, 1.0, stat, mark)
    assert stat['drv_pause_s'] == 0.0
    # the progress row's columns
    row = {'update': 1, **{k: stat.get(k, '') for k in DRV_FIELDS}}
    f = io.StringIO()
    w = csv.DictWriter(f, fieldnames=list(row))
    w.writeheader(); w.writerow(row)
    back = list(csv.DictReader(io.StringIO(f.getvalue())))[0]
    assert set(DRV_FIELDS) <= set(back) and back['drv_lane_hist'] == '0/0/0/0/0/0'


class FakeDriver:
    """The C++ driver's lane interface, the jobs stepped on CPU by their lane actors (as a lane-only call) when run()
    is called; records as the driver's kind 4 (the step's log entries handed to on_lane_record, not pushed twice)."""
    def __init__(self, service):
        self.service, self.queue, self.rings, self.dropped = service, [], [], []
        self.merge = True

    def set_lane_merge(self, on):
        self.merge = on

    def submit_lanes(self, job, owner, version, rows, n, pend, ids, seq, lane, replay):
        self.queue.append(dict(job=job, owner=owner, version=version, n=n, pend=pend.numpy().reshape(n, -1).copy(),
                               ids=ids.numpy().copy(), seq=seq.numpy().copy(), lane=lane.numpy().copy(),
                               replay=replay.numpy().astype(bool)))

    def drop_lane_job(self, job):
        for i, j in enumerate(self.queue):
            if j['job'] == job:
                del self.queue[i]; self.dropped.append(job)
                return True
        return False

    def set_ring(self, row, dec, kv, n):
        self.rings.append(int(row))

    def pause(self):
        pass

    def resume(self):
        pass

    def run(self):
        recs = []
        while self.queue:
            j = self.queue.pop(0)
            gen, positions, version, lanes, rows, pend, keys, replay = self.service.drv_jobs[j['job']][:8]
            actor = next(a for a in self.service.actors.values() if id(a) == j['owner'])
            bs = actor.model.block_memory.config.block_steps
            step_b = np.array([int(actor.state.step_host[actor.base+q]) for q in lanes])
            actor.step(rows, pend, lanes, ukeys=keys, evaluate=replay)
            dec, kvs, bid = [], [], []
            for k, q in enumerate(lanes.tolist()):
                log = actor.logs[q]
                dec.append(int(log.dec[log.n-1])); kvs.append(int(log.kvs[log.n-1]))
                log.n -= 1   # (on_lane_record pushes the step's entries)
                if (step_b[k]+1) % bs == 0:
                    bid.append(int(log.blk[log.nb-1])); log.nb -= 1
                else:
                    bid.append(-1)
            first = rows['first'].astype(np.uint8)
            if first.any():
                step_b = np.where(first > 0, 0, step_b)
            recs.append(dict(kind=4, job=j['job'], owner=j['owner'], merged=0, version=j['version'],
                             out=torch.from_numpy(actor.host_output.copy()), step_before=torch.from_numpy(step_b),
                             first=torch.from_numpy(first), complete=torch.from_numpy(((step_b+1) % bs == 0).astype(np.uint8)),
                             dec=torch.tensor(dec), kvs=torch.tensor(kvs), bid=torch.tensor(bid),
                             pending_before=torch.from_numpy(pend.copy())))
        for r in recs:
            self.service.on_lane_record(r)
        return len(recs)


def setup(driver):
    torch.manual_seed(61)
    m = make_policy('cpu').eval()
    main = SlotBlockActor(m, 2, lane_slots=2, lane_generations=2)
    rng = np.random.default_rng(61)
    for t in range(6):
        rows = synth_rows(rng, 2); rows['first'] = int(t == 0); rows['episode'] = 3
        main.step(rows, synth_pending(rng, 2, m.heads), [0, 1], ukeys=(np.array([t, t]), np.array([0, 1])))
    s = ring_sampler(2)
    s.cfg = SimpleNamespace(workers=2, memory_dim=0, teacher_depths=(2,), teacher_death_depths=(4,))
    s.block_arrays = [np.zeros((2, 4), np.int64), np.zeros(2, np.int64)]
    s.cf_snap = np.zeros(2, np.int64)
    service = BlockLaneService(s, m, 'cpu', wait=0, main_actor=main)
    service.service_control = lambda: None   # (the control's reclamation needs the full sampler; unchanged by B54)
    assert service.capture_point(0, main.snapshot(0), m, model_version=0)
    if driver:
        fake = FakeDriver(service)
        service.attach_driver(fake)
        return m, main, s, service, fake
    return m, main, s, service, None


def script(m, n_rounds=7):
    rng = np.random.default_rng(62)
    out = []
    for k in range(n_rounds):
        lanes = [0] if k < 2 else [0, 1]
        rows = synth_rows(rng, len(lanes)); rows['first'] = 0
        out.append([(lane, rows[i], synth_pending(rng, 1, m.heads)[0], 40+10*k+lane, 1 if (k, lane) in ((0, 0), (2, 1)) else 0)
                    for i, lane in enumerate(lanes)])
    return out


def reclaim(s, svc):
    """BlockLaneService._service_control's context reclamation: idents no lane_t / v2_t row names are retired."""
    live = set()
    for records in (s.lane_t, s.v2_t):
        pairs = np.array(records[..., :2], dtype=np.uint32, copy=True).reshape(-1, 2)
        live.update((pairs[:, 0] | (pairs[:, 1] << 16)).tolist())
    svc.contexts.retire(live, transient=True)


def answers(s, lanes):
    return s.lane_io[lanes, 5:10].copy(), s.lane_out[lanes, :5].copy()


def test_lane_service_driver_mode_matches_python_path_with_fallbacks():
    m, _, s_ref, ref, _ = setup(False)
    m2, _, s, svc, fake = setup(True)
    rounds = script(m)
    for k, reqs in enumerate(rounds):
        lanes = [q for q, *_ in reqs]
        for lane, row, pend, seq, begin in reqs:
            s_ref.request(lane, row, pend, seq=seq, tbegin=begin)
            s.request(lane, row, pend, seq=seq, tbegin=begin)
        assert ref.tick()
        if k in (3, 4):
            # the actor waits for the learner (driver paused): k = 3 a round already queued on the driver (flushed to
            # the Python path), k = 4 a new round on the Python path
            if k == 3:
                assert svc.tick() and svc.fly is not None and not svc.fly['ready'] and fake.queue
            assert svc.serve_wait(0.0)
            assert svc.fly is None
        else:
            assert svc.tick()                      # (submitted: the answers wait for the records)
            assert svc.fly is not None and not svc.fly['ready'] and len(fake.queue) >= 1
            assert not svc.tick()                  # (never waits for the driver)
            assert fake.run() >= 1 and svc.fly['ready']
            reclaim(s, svc)   # (the control's reclamation between the records and the answer, as in train_tok)
            assert svc.tick() is not None and svc.fly is None
        for lane in lanes:   # every context a lane's answer names is still registered (the v2 ring reads it later)
            if s.lane_t[lane, -1] == 1.0:
                pair_ = np.array(s.lane_t[lane, :2], np.uint32)
                assert int(pair_[0] | (pair_[1] << 16)) in svc.contexts.values, f'round {k} lane {lane}'
        a_ref, o_ref = answers(s_ref, lanes)
        a, o = answers(s, lanes)
        np.testing.assert_array_equal(a, a_ref, err_msg=f'round {k}')
        np.testing.assert_array_equal(o, o_ref, err_msg=f'round {k}')
    st = svc.stat
    assert fake.dropped and st['py_lane_flushed_rows'] == 2 and st['py_lane_rounds'] == 1 and st['py_lane_rows'] == 2
    # the rows the Python path stepped got their rings back before the next driver job
    gen_actor = next(iter(svc.actors.values()))
    assert {gen_actor.base, gen_actor.base+1} <= set(fake.rings) and not svc.ring_dirty


def test_lane_service_driver_mode_point_released_while_in_flight():
    """A cancellation (the control releases the lane's point) between the submit and the job's record: the answer
    still goes out (the context owner taken at the submit), nothing raises."""
    m, main, s, svc, fake = setup(True)
    rng = np.random.default_rng(63)
    rows = synth_rows(rng, 1); rows['first'] = 0
    s.request(0, rows[0], synth_pending(rng, 1, m.heads)[0], seq=77, tbegin=1)
    assert svc.tick() and svc.fly is not None and not svc.fly['ready']
    svc.release_point(0)
    assert fake.run() == 1 and svc.fly['ready']
    reclaim(s, svc)
    svc.tick()
    assert svc.fly is None and s.lane_out[0, 4] == 77
