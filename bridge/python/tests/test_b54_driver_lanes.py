"""B54 gates: the lane rounds on the C++ driver (bridge/engine/abp_actor.cpp: submit_lanes, merged main + lane batches,
lane-only calls, kind 4 records, set_ring). Skipped without the extension (no CUDA torch, no g++, ISAAC_BLOCK_CPP_ACTOR=0).

- crn_uniforms in C++ == tok_policy.crn_uniforms bit for bit (thousands of random keys / lanes / columns);
- lane-only jobs == SlotBlockActor.step of the lane actor (ukeys, evaluate rows) bit for bit: outputs, pages, decision ids,
  blocks and their sources, also after a restore of another lane from a snapshot (the lineage ring set by driver_ring);
- merged batches (main rows first, then the lane job of the main policy's version) == the Python merged call
  (main_actor.step with owners): outputs, pages, ids; the order invariant: main replied, main pages and record before the
  lane rows' pages and record; a job of another version is not merged;
- same seed, same scripted rounds: identical main and lane outputs; another seed: other main samples, the same lane
  outputs (CRN);
- pressure: workers stepping while a thread restores lane rows and queues lane jobs (merged and lane-only): every lane
  record's pages complete and its blocks' sources equal the mirror logs; main records' lineage as in B52.
"""
import copy
import threading
import time

import numpy as np
import pytest
import torch

from test_block_memory import make_policy
from test_visible_input import synth_pending
from test_b52_cpp_actor import EXT, frame, fake_sampler, make_driver_for, FakeWorker
from isaac_bridge.block_paged import SlotBlockActor, PagedSnapshot, EpisodeLog
from isaac_bridge.block_lanes import driver_ring
from isaac_bridge.tok_policy import crn_uniforms

pytestmark = pytest.mark.skipif(EXT is None, reason='cpp actor extension unavailable')

MSG = (b'\x00\x00\x00\x010', b'\x00\x00\x00\x011')


def test_crn_cpp_matches_python():
    rng = np.random.default_rng(54)
    for columns in (1, 20, 33, 256):
        n = 3000
        keys = np.concatenate([rng.integers(-2**62, 2**62, n//2), rng.integers(0, 10**6, n - n//2)]).astype(np.int64)
        t = np.concatenate([rng.integers(0, 4096, n//2), rng.integers(0, 2**40, n - n//2)]).astype(np.int64)
        a = EXT.crn_uniforms(torch.from_numpy(keys), torch.from_numpy(t), columns).numpy()
        b = crn_uniforms(keys, t, columns)
        assert a.dtype == b.dtype == np.float32
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


def wait_kind(d, kinds, actors, timeout=30.0):
    """The next record of one of `kinds`; capture requests (kind 1) answered on their owner's graph."""
    t0 = time.perf_counter()
    while True:
        r = d.next_record(0.05)
        if r is None:
            assert time.perf_counter()-t0 < timeout, f'no record; driver error {d.last_error()!r}'
            continue
        if r['kind'] == 1:
            own = next(a for a in actors if a.step_graph.owner == int(r['owner']))
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=own.bf16):
                assert own.step_graph.capture_key(r['cap'], r['widths']) is not None
            d.resume()
            continue
        assert r['kind'] in kinds, (r['kind'], d.last_error())
        return r


def submit(d, job, actor, rows, pend, lanes, seq, replay, version):
    rows = np.ascontiguousarray(rows)
    d.submit_lanes(job, int(actor.step_graph.owner), int(version), torch.from_numpy(rows.view(np.uint8).reshape(-1)), len(lanes),
                   torch.from_numpy(np.ascontiguousarray(pend, np.int64).reshape(-1)),
                   torch.from_numpy((actor.base+np.asarray(lanes)).astype(np.int64)),
                   torch.from_numpy(np.asarray(seq, np.int64)), torch.from_numpy(np.asarray(lanes, np.int64)),
                   torch.from_numpy(np.asarray(replay, np.uint8)))


def mirror(actor, logs, rec, lanes, rows, version):
    """on_lane_record's lineage: logs per lane, the pre-step contexts."""
    ctx = []
    for k, lane in enumerate(lanes):
        if rec['first'][k]:
            logs[lane] = EpisodeLog((0, lane, int(rows['episode'][k])))
        ctx.append(PagedSnapshot(actor.store, logs[lane], int(rec['step_before'][k]), rec['pending_before'][k:k+1].numpy().copy(), version))
        logs[lane].push(rec['dec'][k:k+1].numpy(), rec['kvs'][k:k+1].numpy())
        if rec['complete'][k]:
            logs[lane].push_block(int(rec['bid'][k]), int(rec['step_before'][k]))
    return ctx


def same_pages(sa, sb, kva, kvb):
    torch.testing.assert_close(sa.kv(kva), sb.kv(kvb), rtol=0, atol=0)
    torch.testing.assert_close(sa.z(kva), sb.z(kvb), rtol=0, atol=0)
    np.testing.assert_array_equal(sa.d_meta[kva], sb.d_meta[kvb])


def same_block(sa, sb, ba, bb):
    torch.testing.assert_close(sa.blocks(np.array([ba])), sb.blocks(np.array([bb])), rtol=0, atol=0)
    np.testing.assert_array_equal(sa.b_src[ba], sb.b_src[bb]); np.testing.assert_array_equal(sa.b_srcver[ba], sb.b_srcver[bb])
    np.testing.assert_array_equal(sa.b_meta[ba], sb.b_meta[bb])


def pair(m, lane_model):
    """(driven main, its lane actor), (reference main, its lane actor): fresh stores each."""
    out = []
    for _ in range(2):
        a = SlotBlockActor(m, 1, graph_entities=128, bf16=True, lane_slots=2, lane_generations=1)
        la = a.lane_actor(lane_model, 2, generation=0)
        out.append((a, la))
    return out


def test_lane_only_jobs_match_python_lane_step_with_restore():
    torch.manual_seed(51)
    m = make_policy('cuda').eval()
    lm = copy.deepcopy(m)
    (a, la), (b, lb) = pair(m, lm)
    la.version = lb.version = 5
    rng = np.random.default_rng(51)
    nh = len(m.heads); bs = m.block_memory.config.block_steps
    s = fake_sampler(1, nh)
    d = make_driver_for(a, s)
    la.step_graph.attach_driver(d)
    d.set_version(3)   # (main version 3, lane jobs at 5: never merged)
    d.start()
    logs = {0: EpisodeLog(None), 1: EpisodeLog(None)}
    snap_d = snap_p = None
    job = 0

    def one(lane, k, first, ep):
        nonlocal job
        rows = frame(rng, 1, k, np.array([ep]), np.array([first]), seed=5000+100*lane+k)
        pend = synth_pending(rng, 1, m.heads)
        seq, rep = np.array([200+k]), np.array([k % 7 == 3])
        job += 1
        submit(d, job, la, rows, pend, [lane], seq, rep, 5)
        r = wait_kind(d, (4,), [a, la])
        assert int(r['job']) == job and int(r['merged']) == 0 and int(r['version']) == 5
        ctx = mirror(la, logs, r, [lane], rows, 5)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            lb.step(rows, pend, [lane], ukeys=(seq, np.array([lane])), evaluate=rep)
        np.testing.assert_array_equal(r['out'].numpy(), lb.host_output)
        kb = lb.logs[lane].kvs[lb.logs[lane].n-1:lb.logs[lane].n]
        db = lb.logs[lane].dec[lb.logs[lane].n-1:lb.logs[lane].n]
        np.testing.assert_array_equal(r['dec'].numpy(), db)
        same_pages(a.store, b.store, r['kvs'].numpy(), kb)
        if r['complete'][0]:
            same_block(a.store, b.store, int(r['bid'][0]), int(lb.logs[lane].blk[lb.logs[lane].nb-1]))
        return ctx[0], lb.last_contexts[0]

    for k in range(24):
        cd, cp = one(0, k, int(k == 0), 1)
        if k == 13:
            snap_d, snap_p = cd, cp
    # lane 1 restored from lane 0's step-13 context (another branch of the same episode), on both sides
    la.restore(1, snap_d); driver_ring(d, la, 1)
    lb.restore(1, snap_p)
    logs[1] = la.logs[1]
    for k in range(13, 13+3*bs+3):
        one(1, k, 0, 1)
    d.pause(); d.stop()
    assert d.last_error() == ''
    ga, gb = a.state.gather(torch.tensor([la.base, la.base+1], device='cuda')), b.state.gather(torch.tensor([lb.base, lb.base+1], device='cuda'))
    for k_ in ('kv', 'bank', 'partial', 'kv_steps', 'ends', 'bank_versions', 'partial_versions'):
        torch.testing.assert_close(ga[k_], gb[k_], rtol=0, atol=0, msg=k_)


def drive_merged(m, lane_model, K, lane_version, seed=5, fixed=0.5, rng_seed=52, check=None):
    """One worker's main rows and a two-lane job per step, the job queued and the worker's message sent while the
    driver is paused, so every step is one merged batch (when the versions match). Returns the per-step
    (main record, lane record, rows, pending, lane rows, lane pending, seq, replay)."""
    rng = np.random.default_rng(rng_seed)
    nh = len(m.heads)
    a = SlotBlockActor(m, 1, graph_entities=128, bf16=True, lane_slots=2, lane_generations=1)
    la = a.lane_actor(lane_model, 2, generation=0)
    la.version = lane_version
    s = fake_sampler(1, nh)
    d = make_driver_for(a, s, seed)
    la.step_graph.attach_driver(d)
    d.set_fixed_uniform(fixed)
    d.set_version(3)
    d.start()
    sock = s._pairs[0][1]
    steps = []
    for k in range(K):
        rows = frame(rng, 1, k, np.array([1]), np.array([int(k == 0)]), seed=6000+k)
        lrows = frame(rng, 2, k, np.array([1, 1]), np.array([int(k == 0)]*2), seed=7000+k)
        lpend = synth_pending(rng, 2, m.heads)
        seq, rep = np.array([300+k, 301+k]), np.array([False, k % 5 == 2])
        d.pause()
        submit(d, k+1, la, lrows, lpend, [0, 1], seq, rep, lane_version)
        s.rows[k & 1] = rows[0]
        sock.sendall(MSG[k & 1])
        d.resume()
        r0 = wait_kind(d, (0,), [a, la])
        r4 = wait_kind(d, (4,), [a, la])
        rep_ = b''
        while len(rep_) < 5:
            rep_ += sock.recv(5-len(rep_))
        steps.append((r0, r4, rows, lrows, lpend, seq, rep))
        if check is not None:
            check(a, la, k, r0, r4)
    d.pause(); d.stop()
    assert d.last_error() == ''
    return a, la, steps


def test_merged_batches_match_python_merged_call_and_keep_the_reply_order():
    torch.manual_seed(52)
    m = make_policy('cuda').eval()
    nh = len(m.heads); heads = sum(m.heads)
    b = SlotBlockActor(m, 1, graph_entities=128, bf16=True, lane_slots=2, lane_generations=1)
    lb = b.lane_actor(m, 2, generation=0)
    b.version = lb.version = 3
    K = 9
    order = []

    def check(a, la, k, r0, r4):
        assert int(r4['merged']) == 1 and int(r4['version']) == 3
        # B54 condition a: the main rows were replied, their pages written and their record queued before the lane
        # rows' pages; the lane record carries the main reply's time
        assert r4['t_main_reply'] == r0['t_reply'] and r0['t_done'] <= r4['t_reply'] <= r4['t_pages']
        order.append(k)

    a, la, steps = drive_merged(m, m, K, 3, check=check)
    assert order == list(range(K))
    pending = np.zeros((1, nh), np.int64)
    logs = {0: EpisodeLog(None), 1: EpisodeLog(None)}
    for k, (r0, r4, rows, lrows, lpend, seq, rep) in enumerate(steps):
        if rows['first'][0]:
            pending[:] = 0
        np.testing.assert_array_equal(r0['pending_before'].numpy(), pending)
        u = torch.cat((torch.full((1, heads), .5), torch.as_tensor(crn_uniforms(seq, np.array([0, 1]), heads))))
        allr = np.concatenate((rows, lrows))
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            b.step(allr, np.concatenate((pending, lpend)), [0, 0, 1], owners=[b, lb, lb], uniform=u,
                   evaluate=np.concatenate(([False], rep)))
        np.testing.assert_array_equal(r0['out'].numpy(), b.host_output)
        np.testing.assert_array_equal(r4['out'].numpy(), lb.host_output)
        # decision ids: main first, then the lanes (the same allocation order)
        np.testing.assert_array_equal(r0['dec'].numpy(), b.logs[0].dec[b.logs[0].n-1:b.logs[0].n])
        np.testing.assert_array_equal(r4['dec'].numpy(), [lb.logs[0].dec[lb.logs[0].n-1], lb.logs[1].dec[lb.logs[1].n-1]])
        same_pages(a.store, b.store, r0['kvs'].numpy(), b.logs[0].kvs[b.logs[0].n-1:b.logs[0].n])
        same_pages(a.store, b.store, r4['kvs'].numpy(), np.array([lb.logs[0].kvs[lb.logs[0].n-1], lb.logs[1].kvs[lb.logs[1].n-1]]))
        mirror(la, logs, r4, [0, 1], lrows, 3)
        for q, lane in enumerate((0, 1)):
            if r4['complete'][q]:
                same_block(a.store, b.store, int(r4['bid'][q]), int(lb.logs[lane].blk[lb.logs[lane].nb-1]))
        if r0['complete'][0]:
            same_block(a.store, b.store, int(r0['bid'][0]), int(b.logs[0].blk[b.logs[0].nb-1]))
        pending = r0['actions'].numpy().copy()
    # a lane job of another version is not merged: a lane-only call
    _, _, steps2 = drive_merged(m, copy.deepcopy(m), 2, 4)
    assert all(int(r4['merged']) == 0 for _, r4, *_ in steps2)


def test_driver_with_lanes_same_seed_reproducible():
    torch.manual_seed(53)
    m = make_policy('cuda').eval()

    def outs(seed):
        _, _, steps = drive_merged(m, m, 8, 3, seed=seed, fixed=-1.0, rng_seed=53)
        return (np.stack([r0['out'].numpy() for r0, *_ in steps]), np.stack([r4['out'].numpy() for _, r4, *_ in steps]))

    m1, l1 = outs(77)
    m2, l2 = outs(77)
    m3, l3 = outs(78)
    np.testing.assert_array_equal(m1, m2); np.testing.assert_array_equal(l1, l2)
    nh = len(m.heads)
    assert not np.array_equal(m1[..., :nh], m3[..., :nh]), 'another seed sampled the same main actions'
    np.testing.assert_array_equal(l1, l3)   # (the lanes' CRN uniforms do not depend on the driver's seed)


def test_driver_lanes_pressure_restores_and_jobs_while_workers_step():
    torch.manual_seed(55)
    m = make_policy('cuda').eval()
    lm = copy.deepcopy(m)
    W, K = 4, 50
    a = SlotBlockActor(m, W, graph_entities=128, bf16=True, lane_slots=4, lane_generations=2)
    gens = [a.lane_actor(m, 4, generation=0), a.lane_actor(lm, 4, generation=1)]
    gens[0].version, gens[1].version = 2, 9   # (generation 0 merges with main batches at version 2, generation 1 never)
    nh = len(m.heads); bs = m.block_memory.config.block_steps
    rng = np.random.default_rng(55)
    frames = {(w, k): frame(rng, 1, k, np.array([1]), np.array([int(k == 0)]), seed=9000+w*K+k) for w in range(W) for k in range(K)}
    s = fake_sampler(W, nh)
    d = make_driver_for(a, s)
    for g in gens:
        g.step_graph.attach_driver(d)
    d.set_fixed_uniform(-1)
    d.set_version(2)
    store = a.store
    logs = [EpisodeLog(None) for _ in range(W)]
    lane_logs = [{q: EpisodeLog(None) for q in range(4)} for _ in gens]
    jobs = {}
    lane_recs = [0, 0]   # merged, lane-only
    restores = [0]
    submitted = [0]
    state_lock = threading.Lock()   # (the test's mirror of the consumer: one thread at a time)
    ctxs = []                      # recent main contexts to restore lanes from
    stop = threading.Event()
    done_jobs = threading.Semaphore(0)

    def lane_thread():
        """Restores idle lanes from recent main contexts and queues their jobs (the consumer's _begin + submit)."""
        jid, lrng = 0, np.random.default_rng(56)
        while not stop.is_set():
            with state_lock:
                snap = ctxs[-1] if ctxs else None
            if snap is None:
                time.sleep(0.001); continue
            g = int(lrng.integers(0, 2)); ga = gens[g]
            lanes = [int(x) for x in lrng.choice(4, size=int(lrng.integers(1, 4)), replace=False)]
            with state_lock:
                for lane in lanes:
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                        ga.restore(lane, snap)
                    driver_ring(d, ga, lane)
                    lane_logs[g][lane] = ga.logs[lane]
                    restores[0] += 1
            for rep_k in range(3):
                if stop.is_set():   # (the test's end: no new job; conftest checks for leftover threads)
                    return
                rows = frame(lrng, len(lanes), 100+rep_k, np.ones(len(lanes), np.int64), np.zeros(len(lanes), np.int64), seed=int(lrng.integers(1, 10**6)))
                pend = synth_pending(lrng, len(lanes), m.heads)
                jid += 1
                with state_lock:
                    jobs[jid] = (g, lanes, rows)
                    submitted[0] += 1
                submit(d, jid, ga, rows, pend, lanes, lrng.integers(0, 10**6, len(lanes)), np.zeros(len(lanes), bool), ga.version)
                while not done_jobs.acquire(timeout=0.05):   # (one job of these lanes in flight at a time; the end too)
                    if stop.is_set():
                        return

    workers = [FakeWorker(w, s._pairs[w][1], s, lambda w_, k_: frames[w_, k_], K, delay=0.002) for w in range(W)]
    d.start()
    for wk in workers:
        wk.start()
    lt = threading.Thread(target=lane_thread, daemon=True); lt.start()
    t0 = time.perf_counter()
    while sum(len(wk.got) for wk in workers) < W*K or d.queue_size() > 0:
        r = d.next_record(0.05)
        if r is None:
            assert time.perf_counter()-t0 < 90, f'driver error {d.last_error()!r}'
            if any(wk.error for wk in workers):
                raise next(wk.error for wk in workers if wk.error)
            continue
        if r['kind'] == 1:
            own = next(x for x in [a, *gens] if x.step_graph.owner == int(r['owner']))
            with state_lock, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                assert own.step_graph.capture_key(r['cap'], r['widths']) is not None
            d.resume(); continue
        if r['kind'] == 4:
            with state_lock:
                g, lanes, rows = jobs.pop(int(r['job']))
                ga = gens[g]
                mirror(ga, lane_logs[g], r, lanes, rows, ga.version)
                for k, lane in enumerate(lanes):
                    kv = int(r['kvs'][k])
                    assert store.d_meta[kv, 0] == int(r['step_before'][k]) and store.d_meta[kv, 1] == ga.version
                    assert int(r['dec'][k]) in store.raw
                    if r['complete'][k]:
                        step = int(r['step_before'][k]); bid = int(r['bid'][k])
                        d_ids, k_ids = lane_logs[g][lane].decisions(step-bs+1, step+1)
                        np.testing.assert_array_equal(store.b_src[bid], d_ids)
                        np.testing.assert_array_equal(store.b_srcver[bid], store.d_meta[k_ids, 1])
                lane_recs[int(r['merged']) == 0] += 1
                assert int(r['merged']) == 0 or g == 0
            done_jobs.release()
            continue
        assert r['kind'] == 0, d.last_error()
        with state_lock:
            for j, w in enumerate(r['workers'].numpy().tolist()):
                if r['first'][j]:
                    logs[w] = EpisodeLog((0, w, 1))
                step = int(r['step_before'][j])
                snap = PagedSnapshot(store, logs[w], step, r['pending_before'][j:j+1].numpy().copy(), 2)
                logs[w].push(r['dec'][j:j+1].numpy(), r['kvs'][j:j+1].numpy())
                if r['complete'][j]:
                    bid = int(r['bid'][j]); logs[w].push_block(bid, step)
                    d_ids, _ = logs[w].decisions(step-bs+1, step+1)
                    np.testing.assert_array_equal(store.b_src[bid], d_ids)
                ctxs.append(snap)
                del ctxs[:-16]
    stop.set()
    done_jobs.release()
    lt.join(10)
    assert not lt.is_alive(), 'the test lane thread did not end'
    for wk in workers:
        wk.join(10); assert wk.error is None, wk.error
    d.pause()
    # the jobs still queued or in flight: their records
    while d.lane_queue_size() > 0 or jobs:
        d.resume()
        r = d.next_record(0.2)
        if r is None:
            break
        if r['kind'] == 4:
            with state_lock:
                jobs.pop(int(r['job']))   # (KeyError: a second record of a job)
                lane_recs[int(r['merged']) == 0] += 1
    d.pause(); d.stop()
    assert d.last_error() == ''
    c = d.counters()
    # every submitted job answered exactly once (none lost, none twice); how many merged is the race's (the merged path's
    # correctness: test_merged_batches_match_python_merged_call_and_keep_the_reply_order, which merges every step)
    assert not jobs and sum(lane_recs) == submitted[0] == c['lane_jobs'], (lane_recs, submitted, c['lane_jobs'], len(jobs))
    assert restores[0] > 5 and lane_recs[1] > 0, (restores, lane_recs, c)
    for w in range(W):
        assert int(a.state.step_host[w]) == K


def test_driver_python_switch_matches_driver_only():
    """The driver <-> Python switch of serve_wait (the actor waiting for the learner): rounds stepped on the Python
    path in the middle of a lane's run, the rings re-set from the logs (driver_ring), then back on the driver. Same
    scripted rounds with and without the switch: outputs, decision ids, pages, blocks and their sources, arena rows
    bit for bit."""
    torch.manual_seed(57)
    m = make_policy('cuda').eval()
    lm = copy.deepcopy(m)
    nh = len(m.heads)
    rng = np.random.default_rng(57)
    K = 30
    script = []
    for k in range(K):
        lanes = [0] if k < 10 else [0, 1]
        rows = frame(rng, len(lanes), k, np.ones(len(lanes), np.int64), np.array([int(k == 0)]*len(lanes)), seed=8000+k)
        script.append((lanes, rows, synth_pending(rng, len(lanes), m.heads), np.array([500+k+10*q for q in lanes]),
                       np.array([k % 6 == 4 and q == 0 for q in lanes])))

    def run(python_steps):
        a = SlotBlockActor(m, 1, graph_entities=128, bf16=True, lane_slots=2, lane_generations=1)
        la = a.lane_actor(lm, 2, generation=0)
        la.version = 5
        s = fake_sampler(1, nh)
        d = make_driver_for(a, s)
        la.step_graph.attach_driver(d)
        d.set_version(3)
        d.start()
        outs, decs, kvs, bids = [], [], [], []
        snap = None
        for k, (lanes, rows, pend, seq, rep) in enumerate(script):
            if k == 10:   # lane 1: a branch of lane 0 restored from its step-9 context
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                    la.restore(1, snap)
                driver_ring(d, la, 1)
            if k in python_steps:
                d.pause()
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                    la.step(rows, pend, lanes, ukeys=(seq, np.array(lanes)), evaluate=rep)
                out = la.host_output.copy()
                dec = np.array([la.logs[q].dec[la.logs[q].n-1] for q in lanes])
                kv = np.array([la.logs[q].kvs[la.logs[q].n-1] for q in lanes])
                bid = np.array([la.logs[q].blk[la.logs[q].nb-1] if la.logs[q].nb and la.logs[q].blk_end[la.logs[q].nb-1] == int(a.state.step_host[la.base+q])-1
                                else -1 for q in lanes])
                ctx = la.last_contexts
                for q in lanes:
                    driver_ring(d, la, q)
                d.resume()
            else:
                submit(d, k+1, la, rows, pend, lanes, seq, rep, 5)
                r = wait_kind(d, (4,), [a, la])
                ctx = mirror(la, la.logs, r, lanes, rows, 5)
                out, dec, kv, bid = r['out'].numpy().copy(), r['dec'].numpy().copy(), r['kvs'].numpy().copy(), r['bid'].numpy().copy()
            if k == 9:
                snap = ctx[0]
            outs.append(out); decs.append(dec); kvs.append(kv); bids.append(bid)
        d.pause(); d.stop()
        assert d.last_error() == ''
        return a, la, outs, decs, kvs, bids

    a1, l1, o1, d1, k1, b1 = run(set())
    a2, l2, o2, d2, k2, b2 = run({12, 13, 14, 15, 16, 17, 25})
    bs = m.block_memory.config.block_steps
    for k in range(K):
        np.testing.assert_array_equal(o1[k], o2[k], err_msg=f'step {k}')
        np.testing.assert_array_equal(d1[k], d2[k]); np.testing.assert_array_equal(k1[k], k2[k])
        same_pages(a1.store, a2.store, k1[k], k2[k])
        np.testing.assert_array_equal(b1[k] >= 0, b2[k] >= 0, err_msg=f'step {k}: blocks')
        for x, y in zip(b1[k], b2[k]):
            if x >= 0:
                same_block(a1.store, a2.store, int(x), int(y))
    assert sum(int((b >= 0).sum()) for b in b1) >= K//bs   # (blocks completed on both sides of the switch)
    ids = torch.tensor([l1.base, l1.base+1], device='cuda')
    g1, g2 = a1.state.gather(ids), a2.state.gather(ids)
    for k_ in ('kv', 'bank', 'partial', 'kv_steps', 'ends', 'bank_versions', 'partial_versions'):
        torch.testing.assert_close(g1[k_], g2[k_], rtol=0, atol=0, msg=k_)


def test_driver_holds_a_worker_until_its_capture_and_runs_lanes_only_on_hold():
    """B54: (1) a record asking for a CF capture (cf_snap): the worker's next record waits until Python reports the
    capture done (capture_done), as the Python path captures before replying; (2) main_hold: lane jobs run, worker
    records wait until the hold ends."""
    import socket as socket_
    torch.manual_seed(58)
    m = make_policy('cuda').eval()
    nh = len(m.heads)
    a = SlotBlockActor(m, 1, graph_entities=128, bf16=True, lane_slots=2, lane_generations=1)
    la = a.lane_actor(copy.deepcopy(m), 2, generation=0)
    la.version = 7
    rng = np.random.default_rng(58)
    s = fake_sampler(1, nh)
    d = make_driver_for(a, s)
    la.step_graph.attach_driver(d)
    d.set_fixed_uniform(.5); d.set_version(3)
    d.start()
    sock = s._pairs[0][1]

    def send(k):
        s.rows[k & 1] = frame(rng, 1, k, np.array([1]), np.array([int(k == 0)]), seed=11000+k)[0]
        sock.sendall(MSG[k & 1])

    def reply(timeout):
        sock.settimeout(timeout)
        try:
            return sock.recv(5)
        except socket_.timeout:
            return None
        finally:
            sock.settimeout(None)

    s.cf_snap[0] = 4
    send(0)
    r0 = wait_kind(d, (0,), [a, la])
    assert int(r0['cf_snap'][0]) == 4 and s.cf_snap[0] == 0 and reply(5) is not None
    send(1)
    assert reply(0.3) is None and d.queue_size() == 0   # (held: the capture is not done)
    d.capture_done([0])
    r1 = wait_kind(d, (0,), [a, la])
    assert reply(5) is not None and int(r1['step_before'][0]) == 1 and d.counters()['cap_holds'] >= 1
    # lanes only
    d.set_main_hold(True)
    send(2)
    rows = frame(rng, 1, 0, np.array([1]), np.array([1]), seed=12000)
    submit(d, 1, la, rows, synth_pending(rng, 1, m.heads), [0], np.array([5]), np.array([False]), 7)
    r4 = wait_kind(d, (4,), [a, la])
    assert int(r4['job']) == 1 and reply(0.3) is None   # (the lane job ran, the worker's record waits)
    d.set_main_hold(False)
    r2 = wait_kind(d, (0,), [a, la])
    assert reply(5) is not None and int(r2['step_before'][0]) == 2
    d.pause(); d.stop()
    assert d.last_error() == ''
