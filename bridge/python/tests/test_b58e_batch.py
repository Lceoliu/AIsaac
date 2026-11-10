"""B58e gate: the driver's batched records (set_batch: the per-step counters kept by the driver, the episode-event rows
handed over in kind 8 batches / take_events, a kind 7 record when the collect's rollout is full, a kind 9 wake) give
Python what the per-record path (a kind 0 record per step) gave it, and the main thread's blocking wait (batch_wait)
never sleeps through a driver error, a stopped driver or the learner's wake.

1. One worker (every step one row: both runs step the same batches): the per-record run and the batch run (small
   ev_max: many kind 8 batches) on two actors of the same weights with the native consumer, fixed uniforms. Equal:
   the event rows (worker, ROW header bytes, the consumer's per-row snapshot), in order; the counters (polls = records,
   calls = records with a go row, rows = go rows); the finished episodes' summaries; the rollouts byte for byte.
2. Three workers, batch mode: each worker's event rows are its script's first / end rows in order; rows = go rows; the
   kind 7 record comes once, when usable_now() >= the target.
3. A driver error (set_test_fail) / a stopped driver / a wake from another thread end batch_wait within 1 s.
Skipped without the extension."""
import threading
import time

import numpy as np
import pytest
import torch

from test_block_memory import make_policy
from test_b52_cpp_actor import EXT, fake_sampler, make_driver_for, FakeWorker, frame, ACTOR
from isaac_bridge.block_paged import SlotBlockActor
from isaac_bridge import block_driver as bdrv
from isaac_bridge.tok_obs import ROW, ROW_FRAME_AT, DONE_BRANCH

pytestmark = pytest.mark.skipif(EXT is None, reason='cpp actor extension unavailable')

CAP = 64


def script(W, K, seed):
    """per (worker, step): one ROW record; episodes end with done 1 / 2 / 3 (the next record a first), branch
    truncations (DONE_BRANCH: the record still steps), breaks (a first without an end)."""
    rng = np.random.default_rng(seed)
    out = {}
    for w in range(W):
        ep, t, new = 0, 0, True
        for k in range(K):
            first = new or rng.random() < 0.03
            if first:
                ep += 1; t = 0
            r = frame(rng, 1, t, np.array([ep]), np.array([int(first)]), seed=700+131*w+k)
            d = 0
            if not first and rng.random() < 0.12:
                d = int(rng.choice([1, 2, 3, DONE_BRANCH]))
            r['done'] = d
            r['group'] = 0; r['seed'] = 1000+w; r['lab'] = 0; r['point'] = 0
            r['branch'] = 1 if d == DONE_BRANCH else 0
            r['damage'] = np.float32(rng.random()*2); r['hurt'] = np.float32(rng.random() < 0.2)
            out[w, k] = r
            t += 1
            new = d not in (0, DONE_BRANCH)
    return out


def make_rolls(W, nh):
    import train_tok as T
    return [T.Rollout(W, CAP, nh, 0, True, None, close=True, priv=None, block_steps=128) for _ in range(2)]


def make_book(W):
    import train_tok as T
    rw = dict(damage=1.0, hurt=0.75, room=1.0, explore=0.5, boss=2.0, exit=3.0, death=2.0, timeout=1.0, time=0.002,
              heal=0.5, resource=0.25, clear=1.0)
    return T.Episodes(W, rw, True, run=True, items=True)


def capture(r):
    """a kind 1 record: the graph captured on the test actor (the driver retries the batch once resumed)"""
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=ACTOR[0].bf16):
        assert ACTOR[0].step_graph.capture_key(r['cap'], r['widths']) is not None


def setup(m, W, seed=5):
    nh = len(m.heads)
    actor = SlotBlockActor(m, W, graph_entities=128, bf16=True, lane_slots=0)
    s = fake_sampler(W, nh)
    d = make_driver_for(actor, s, seed)
    d.set_fixed_uniform(0.5); d.set_version(3)
    book, rolls = make_book(W), make_rolls(W, nh)
    bdrv.attach_consumer(d, book, rolls, 0.99)
    d.use_rollout(rolls[0].native_slot)
    ACTOR[0] = actor
    return actor, s, d, book, rolls


def start_workers(s, d, W, K, rows):
    workers = [FakeWorker(w, s._pairs[w][1], s, lambda w_, k_: rows[w_, k_], K) for w in range(W)]
    d.start()
    for wk in workers:
        wk.start()
    return workers


def header(rows):
    return rows.view(np.uint8).reshape(len(rows), -1)[:, :ROW_FRAME_AT].copy()


def run_per_record(m, W, K, rows):
    """the per-record path: kind 0 records; the event rows / counters Python took from them"""
    import train_tok as T
    actor, s, d, book, rolls = setup(m, W)
    workers = start_workers(s, d, W, K, rows)
    ev, cnt = [], dict(polls=0, calls=0, rows=0)
    t0 = time.perf_counter()
    paused = False
    try:
        while True:   # (until every worker is answered; then paused: its last step's record queued, then drained)
            assert time.perf_counter()-t0 < 60, d.last_error()
            if not paused and not any(wk.is_alive() for wk in workers):
                d.pause(); paused = True
            r = d.next_record(0.0 if paused else 0.05)
            if r is None:
                if paused:
                    break
                continue
            if r['kind'] == 1:
                capture(r); d.resume(); continue
            assert r['kind'] == 0, (r['kind'], d.last_error())
            idx = r['workers'].numpy(); go = r['go'].numpy().astype(bool)
            rr = r['rows'].numpy().view(ROW); snap = r['snap'].numpy()
            T._native_events(book, rr, idx, snap)
            cnt['polls'] += 1; cnt['calls'] += int(go.any()); cnt['rows'] += int(go.sum())
            for j in np.flatnonzero((rr['first'] != 0) | (rr['done'] != 0)):
                ev.append((int(idx[j]), header(rr[j:j+1])[0], snap[j].copy()))
    finally:
        d.pause(); d.stop()
    for wk in workers:
        wk.join(10); assert wk.error is None
    assert d.last_error() == ''
    return ev, cnt, book, rolls[0]


def run_batch(m, W, K, rows, ev_max, target=-1):
    import train_tok as T
    actor, s, d, book, rolls = setup(m, W)
    d.set_batch(True, ev_max)
    d.set_collect_target(target)
    c0 = d.batch_counters()
    ev, got = [], dict(batches=0, target=0, usable_at_target=None)

    def on_events(rows_t, ws_t, snap_t):
        ws_ = ws_t.numpy(); rr = rows_t.numpy().view(ROW); snap = snap_t.numpy()
        assert len(ws_) == len(rr) == len(snap) and len(ws_) >= 1
        T._native_events(book, rr, ws_, snap)
        got['batches'] += 1
        for j in range(len(ws_)):
            ev.append((int(ws_[j]), header(rr[j:j+1])[0], snap[j].copy()))

    def on_record(r):
        if r['kind'] == 1:
            capture(r); return
        raise AssertionError((int(r['kind']), d.last_error()))
    workers = start_workers(s, d, W, K, rows)
    t0 = time.perf_counter()
    try:
        while any(wk.is_alive() for wk in workers):
            assert time.perf_counter()-t0 < 60, d.last_error()
            st = bdrv.batch_wait(d, on_events, on_record, 0.5)
            if st == 'target':
                got['target'] += 1
                got['usable_at_target'] = (d.usable_now(), rolls[0].usable())
        d.pause()
        while True:   # (drain: the records still queued; kind 8 batches among them)
            r = d.next_record(0.0)
            if r is None:
                break
            if r['kind'] == 8:
                on_events(r['rows'], r['workers'], r['snap'])
            elif r['kind'] != 7:
                on_record(r)
        on_events_tail = d.take_events()
        if len(on_events_tail[1]):
            on_events(*on_events_tail)
        assert len(d.take_events()[1]) == 0   # (taken once)
        c1 = d.batch_counters()
    finally:
        d.pause(); d.stop()
    for wk in workers:
        wk.join(10); assert wk.error is None
    assert d.last_error() == ''
    cnt = {k: c1[k]-c0[k] for k in ('polls', 'calls', 'rows')}
    assert c1['ev_rows']-c0['ev_rows'] == len(ev)
    assert c1['latency'] > 0 and c1['act'] >= c1['latency'] and c1['calls_busy'] == 0
    return ev, cnt, book, rolls[0], got


def same_finished(fa, fb):
    """the finished episodes' summaries equal (a NaN equals a NaN: the snapshots were compared bitwise already)"""
    import math
    assert len(fa) == len(fb)
    for i, (a, b) in enumerate(zip(fa, fb)):
        assert a.keys() == b.keys(), (i, sorted(a), sorted(b))
        for k in a:
            x, y = a[k], b[k]
            if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
                continue
            assert x == y, (i, k, x, y)


def test_batch_matches_per_record_one_worker():
    from test_b58_native_consumer import same_rollout
    torch.manual_seed(41)
    m = make_policy('cuda').eval()
    K = 50
    rows = script(1, K, 41)
    ev_a, cnt_a, book_a, roll_a = run_per_record(m, 1, K, rows)
    ev_b, cnt_b, book_b, roll_b, got = run_batch(m, 1, K, rows, ev_max=3)
    assert len(ev_a) == len(ev_b) > 6 and got['batches'] >= 3
    for (wa, ha, sa), (wb, hb, sb) in zip(ev_a, ev_b):
        assert wa == wb
        np.testing.assert_array_equal(ha, hb)
        np.testing.assert_array_equal(sa, sb)
    assert cnt_a == cnt_b and cnt_a['polls'] == K
    same_finished(book_a.finished, book_b.finished)
    assert len(book_a.finished) > 2
    same_finished(book_a.branch_finished, book_b.branch_finished)
    assert book_a.inv0 == book_b.inv0
    same_rollout(roll_a, roll_b)


def test_batch_events_and_target_three_workers():
    torch.manual_seed(43)
    m = make_policy('cuda').eval()
    W, K = 3, 45
    rows = script(W, K, 43)
    ev, cnt, book, roll, got = run_batch(m, W, K, rows, ev_max=4, target=30)
    for w in range(W):
        want = [header(rows[w, k])[0] for k in range(K) if rows[w, k]['first'][0] != 0 or rows[w, k]['done'][0] != 0]
        mine = [h for w_, h, _ in ev if w_ == w]
        assert len(mine) == len(want)
        for a, b in zip(mine, want):
            np.testing.assert_array_equal(a, b)
    n_go = sum(int(rows[w, k]['done'][0] in (0, DONE_BRANCH)) for w in range(W) for k in range(K))
    assert cnt['rows'] == n_go and cnt['calls'] <= cnt['polls']
    assert got['target'] == 1 and got['usable_at_target'][0] >= 30 and got['usable_at_target'][1] >= 30
    assert len(book.finished) > 2


KEEP = []


def idle_driver(m):
    """a started driver in batch mode whose two workers send nothing (the sampler's arrays kept alive)"""
    actor, s, d, book, rolls = setup(m, 2)
    KEEP[:] = [actor, s, book, rolls]
    d.set_batch(True, 256)
    d.start()
    return d


def raise_on(r):
    if int(r['kind']) == 3:
        raise RuntimeError('cpp actor driver: an error record')
    raise AssertionError(int(r['kind']))


def timed(fn, after, d):
    """fn(d) on another thread `after` s from now; batch_wait's outcome and its delay after fn"""
    t_fn = []
    th = threading.Thread(target=lambda: (time.sleep(after), t_fn.append(time.perf_counter()), fn(d)), daemon=True)
    th.start()
    out = None
    t_end = time.perf_counter()+10
    try:
        while time.perf_counter() < t_end:
            st = bdrv.batch_wait(d, lambda *a: None, raise_on, 0.5)
            if st != 'idle':
                out = st
                break
    except RuntimeError as e:
        out = e
    t_out = time.perf_counter()
    th.join(10)
    return out, t_out-t_fn[0]


def test_driver_error_wakes_the_main_thread():
    torch.manual_seed(45)
    m = make_policy('cuda').eval()
    d = idle_driver(m)
    out, dt = timed(lambda d_: d_.set_test_fail(True), 0.7, d)
    assert isinstance(out, RuntimeError) and dt < 1.0, (out, dt)
    assert 'set_test_fail' in d.last_error()
    d.stop()


def test_stopped_driver_wakes_the_main_thread():
    torch.manual_seed(46)
    m = make_policy('cuda').eval()
    d = idle_driver(m)
    out, dt = timed(lambda d_: d_.stop(), 0.7, d)
    assert isinstance(out, RuntimeError) and dt < 1.0, (out, dt)


def test_wake_ends_the_wait():
    torch.manual_seed(47)
    m = make_policy('cuda').eval()
    d = idle_driver(m)
    out, dt = timed(lambda d_: d_.wake(), 0.7, d)
    assert out == 'wake' and dt < 0.1, (out, dt)
    d.pause(); d.stop()
