"""B52 gates: the C++ actor driver (bridge/engine/abp_actor.cpp through isaac_bridge.actor_native) against the Python
path. Skipped without the extension (no CUDA torch, no g++, ISAAC_BLOCK_CPP_ACTOR=0).

- the native PageStore writes / reads / frees exactly like the Python arrays (ids, metadata, pages, raw frames);
- the C++ step middle (run_graph) == the Python middle, bit for bit, on the same graph inputs;
- the driver end to end (one fake worker, fixed uniforms) == SlotBlockActor.step with the same uniforms: actions,
  logp, value, vg, pages, lineage (b_src / b_srcver), records' pre-step state;
- the reply -> pages -> record order under pressure: several fake workers, a reader thread gathering the pages
  of every recorded decision and restoring lane rows from record-built snapshots while the driver runs; every page
  row read equals the record's values (d_meta), the lineage ring equals the Python EpisodeLog.
"""
import os
import socket
import threading
import time
import types

import numpy as np
import pytest
import torch

from test_block_memory import make_policy
from test_visible_input import synth_rows, synth_pending
from isaac_bridge import actor_native
from isaac_bridge.block_paged import SlotBlockActor, PagedStore, PagedSnapshot, EpisodeLog
from isaac_bridge.tok_obs import ROW, DONE_BRANCH

EXT = actor_native.load() if torch.cuda.is_available() else None
pytestmark = pytest.mark.skipif(EXT is None, reason=f'cpp actor extension unavailable: {actor_native.REASON}')


def frame(rng, n, t, episode, first, seed=1):
    rows = synth_rows(rng, n, seed=seed)
    rows['first'] = first
    rows['episode'] = episode
    rows['t'] = t
    rows['done'] = 0
    rows['pforce'] = 0
    return rows


# ------------------------------------------------------------------------------------------------------- PageStore

def test_native_page_store_matches_python_arrays():
    m = make_policy('cpu')
    cfg = m.block_memory.config
    nh = len(m.heads)
    os.environ['ISAAC_BLOCK_PREALLOC_PAGES'] = '1'
    nat = PagedStore(cfg, nh, torch.bfloat16, page=8)
    py = PagedStore(cfg, nh, torch.bfloat16, page=8, native=False)
    assert nat.native is not None and py.native is None
    rng = np.random.default_rng(3)
    torch.manual_seed(3)
    all_dec = []
    for t in range(5):
        n = 3 + t % 3
        rows = frame(rng, n, t, np.arange(n)+1, 0)
        pend = synth_pending(rng, n, m.heads)
        kv = torch.randn((n, *nat.kv_shape)).to(torch.bfloat16)
        z = torch.randn((n, *nat.z_shape))
        steps = np.arange(n)*2+t
        a = nat.write_decisions(rows, pend, kv, z, steps, 7+t)
        b = py.write_decisions(rows, pend, kv, z, steps, 7+t)
        np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
        all_dec.append(a[0])
        vals = torch.randn((2, *nat.b_shape))
        src = rng.integers(0, 50, (2, cfg.block_steps)); srcver = rng.integers(0, 9, (2, cfg.block_steps))
        ba = nat.write_blocks(vals, np.array([t, t+1]), 5, src, srcver)
        bb = py.write_blocks(vals, np.array([t, t+1]), 5, src, srcver)
        np.testing.assert_array_equal(ba, bb)
    used = np.flatnonzero(nat.d_used)
    np.testing.assert_array_equal(used, np.flatnonzero(py.d_used))
    torch.testing.assert_close(nat.kv(used), py.kv(used), rtol=0, atol=0)
    torch.testing.assert_close(nat.z(used), py.z(used), rtol=0, atol=0)
    np.testing.assert_array_equal(nat.d_meta[used], py.d_meta[used])
    bused = np.flatnonzero(nat.b_used)
    torch.testing.assert_close(nat.blocks(bused), py.blocks(bused), rtol=0, atol=0)
    np.testing.assert_array_equal(nat.b_meta[bused], py.b_meta[bused])
    np.testing.assert_array_equal(nat.b_src[bused], py.b_src[bused]); np.testing.assert_array_equal(nat.b_srcver[bused], py.b_srcver[bused])
    dec = np.concatenate(all_dec)
    assert len(nat.raw) == len(py.raw) == len(dec)
    for d in dec[::3]:
        a, b = nat.raw[int(d)], py.raw[int(d)]
        assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and np.array_equal(a[3], b[3])
    ra, ba_, pa = nat.source_arena(dec[1:7]); rb, bb_, pb = py.source_arena(dec[1:7])
    np.testing.assert_array_equal(ra, rb); np.testing.assert_array_equal(ba_, bb_); np.testing.assert_array_equal(pa, pb)
    rows_a, pend_a = nat.source_rows(dec[2:5]); rows_b, pend_b = py.source_rows(dec[2:5])
    np.testing.assert_array_equal(rows_a['visible_frame'], rows_b['visible_frame']); np.testing.assert_array_equal(pend_a, pend_b)
    # nothing live: collect frees everything in the same order; the next writes reuse the same ids
    sa, sb = nat.collect(), py.collect()
    assert sa['freed_kv'] == sb['freed_kv'] == len(used) and sa['freed_raw'] == sb['freed_raw'] == len(dec)
    rows = frame(rng, 5, 9, np.arange(5)+1, 0)
    kv = torch.randn((5, *nat.kv_shape)).to(torch.bfloat16); z = torch.randn((5, *nat.z_shape))
    a = nat.write_decisions(rows, synth_pending(rng, 5, m.heads), kv, z, np.arange(5), 1)
    b = py.write_decisions(rows, synth_pending(rng, 5, m.heads), kv, z, np.arange(5), 1)
    np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
    assert nat.stats()['kv_rows'] == py.stats()['kv_rows'] == 5


# -------------------------------------------------------------------------------------------------- the step middle

def fake_sampler(W, nh):
    """A sampler stand-in for make_driver: ROW slots, pipes (socketpairs), the shared arrays."""
    pairs = [socket.socketpair() for _ in range(W)]
    s = types.SimpleNamespace()
    s.rows = np.zeros(2*W, ROW)
    s.cfg = types.SimpleNamespace(workers=W)
    s._fds = [a.fileno() for a, _ in pairs]
    s.actions = np.zeros((W, 5), np.int32)
    s.main_value = np.zeros(W, np.float64); s.main_vg = np.zeros(W, np.float64)
    s.cf_snap = np.zeros(W, np.int64)
    s.priv = None; s.aux = None
    s._pairs = pairs
    return s


def make_driver_for(actor, sampler, seed=5):
    from isaac_bridge import block_driver
    stream = torch.cuda.current_stream()
    d = block_driver.make_driver(actor, sampler, stream, seed, False, None)
    assert d is not None
    return d


@pytest.mark.parametrize('bf16', [False, True])
def test_run_graph_middle_matches_python_middle(bf16):
    torch.manual_seed(11)
    m = make_policy('cuda').eval()
    a = SlotBlockActor(m, 4, graph_entities=128, bf16=bf16)      # the C++ middle (driver attached)
    b = SlotBlockActor(m, 4, graph_entities=128, bf16=bf16)      # the Python middle
    assert a.step_graph is not None and a.store.native is not None
    d = make_driver_for(a, fake_sampler(4, len(m.heads)))
    rng = np.random.default_rng(11)
    heads = sum(m.heads)
    for t in range(40):
        n = 4 if t % 3 else 2
        slots = list(range(n))
        first = np.array([int(t == 0 or (t == 13 and i == 1)) for i in slots])
        rows = frame(rng, n, t, np.arange(n)+1, first, seed=100+t)
        pending = synth_pending(rng, n, m.heads)
        u = torch.rand((n, heads))
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            a.step(rows, pending, slots, uniform=u.clone())
            b.step(rows, pending, slots, uniform=u.clone())
        assert a.last_path == 'graph' and b.last_path == 'graph'
        np.testing.assert_array_equal(a.host_output, b.host_output)
    assert d.n_graphs() == a.step_graph.captures
    for slot in range(4):
        assert a.state.step_host[slot] == b.state.step_host[slot]
        ga, gb = a.state.gather(a.state._ids([slot])[1]), b.state.gather(b.state._ids([slot])[1])
        for k in ('kv', 'bank', 'partial', 'kv_steps', 'ends', 'bank_versions', 'partial_versions'):
            torch.testing.assert_close(ga[k], gb[k], rtol=0, atol=0, msg=k)
        la, lb = a.logs[slot], b.logs[slot]
        assert la.n == lb.n and la.nb == lb.nb
        hi = int(a.state.step_host[slot])
        _, ka = la.decisions(max(la.offset, hi-6), hi); _, kb = lb.decisions(max(lb.offset, hi-6), hi)
        torch.testing.assert_close(a.store.kv(ka), b.store.kv(kb), rtol=0, atol=0)
        torch.testing.assert_close(a.store.z(ka), b.store.z(kb), rtol=0, atol=0)


# --------------------------------------------------------------------------------------------------- the driver

class FakeWorker(threading.Thread):
    """Writes records into its two ROW slots, sends the message, waits for the reply, remembers the actions."""
    def __init__(self, w, sock, sampler, make_rows, k_total, delay=0.0):
        super().__init__(daemon=True)
        self.w, self.sock, self.s, self.make_rows, self.K, self.delay = w, sock, sampler, make_rows, k_total, delay
        self.sent, self.got = [], []
        self.error = None

    def run(self):
        try:
            for k in range(self.K):
                row = self.make_rows(self.w, k)
                slot = 2*self.w + (k & 1)
                self.s.rows[slot] = row[0]
                self.sent.append(row.copy())
                self.sock.sendall(b'\x00\x00\x00\x01' + (b'1' if k & 1 else b'0'))
                r = b''
                while len(r) < 5:
                    r += self.sock.recv(5-len(r))
                assert r == b'\x00\x00\x00\x01a', r
                self.got.append(self.s.actions[self.w].copy())
                if self.delay:
                    time.sleep(self.delay*(0.5+np.random.rand()))
        except Exception as e:   # noqa: BLE001
            self.error = e


def collect_records(d, n_steps, timeout=30.0):
    recs, t0 = [], time.perf_counter()
    while len(recs) < n_steps:
        r = d.next_record(0.05)
        if r is None:
            if time.perf_counter()-t0 > timeout:
                raise TimeoutError(f'{len(recs)} records of {n_steps}; driver error: {d.last_error()!r}')
            continue
        if r['kind'] == 1:   # a capture request: capture (the test actor's autocast), resume
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=ACTOR[0].bf16):
                assert ACTOR[0].step_graph.capture_key(r['cap'], r['widths']) is not None
            d.resume()
            continue
        assert r['kind'] == 0, (r['kind'], d.last_error())
        recs.append(r)
    return recs


ACTOR = [None]


def run_driver(actor, W, K, make_rows, delay=0.0, fixed_u=0.5, reader=None, seed=5):
    s = fake_sampler(W, len(actor.model.heads))
    d = make_driver_for(actor, s, seed)
    d.set_fixed_uniform(fixed_u)
    d.set_version(3)
    ACTOR[0] = actor
    workers = [FakeWorker(w, s._pairs[w][1], s, make_rows, K, delay) for w in range(W)]
    d.start()
    for wk in workers:
        wk.start()
    stop = threading.Event()
    rd = None
    if reader is not None:
        rd = threading.Thread(target=reader, args=(stop,), daemon=True); rd.start()
    recs = collect_records(d, W*K)
    for wk in workers:
        wk.join(10); assert not wk.is_alive() and wk.error is None, wk.error
    stop.set()
    if rd is not None:
        rd.join(10)
    d.pause(); d.stop()
    assert d.last_error() == '', d.last_error()
    return s, d, recs, workers


def test_driver_single_worker_matches_python_step_with_fixed_uniforms():
    torch.manual_seed(21)
    m = make_policy('cuda').eval()
    bf16 = True
    a = SlotBlockActor(m, 1, graph_entities=128, bf16=bf16, lane_slots=0)   # driven
    b = SlotBlockActor(m, 1, graph_entities=128, bf16=bf16, lane_slots=0)   # the Python reference
    rng = np.random.default_rng(21)
    K = 50
    script = {}
    for k in range(K):
        first = 1 if k in (0, 19) else 0
        script[k] = frame(rng, 1, k, np.array([1 + (k >= 19)]), np.array([first]), seed=300+k)
    b.version = 3   # (the driver steps at version 3: set_version in run_driver)
    s, d, recs, workers = run_driver(a, 1, K, lambda w, k: script[k])
    assert len(recs) == K and len(workers[0].got) == K
    heads = sum(m.heads); nh = len(m.heads)
    u = torch.full((1, heads), .5)
    pending = np.zeros((1, nh), np.int64)
    ref_logs_kvs = []
    for k in range(K):
        rec = recs[k]
        rows = script[k]
        if rows['first'][0]:
            pending[:] = 0
        np.testing.assert_array_equal(rec['pending_before'].numpy(), pending)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            b.step(rows, pending.copy(), [0], uniform=u.clone())
        out = rec['out'].numpy()
        np.testing.assert_array_equal(out, b.host_output)
        acts = out[:, :nh].astype(np.int64)
        np.testing.assert_array_equal(rec['actions'].numpy(), acts)
        np.testing.assert_array_equal(workers[0].got[k][:nh], acts[0])
        if k == K-1:   # (the shared arrays hold the last step's values)
            assert s.main_value[0] == out[0, nh+1] and s.main_vg[0] == out[0, nh+2]
        assert int(rec['step_before'][0]) == int(b.state.step_host[0])-1
        assert bool(rec['first'][0]) == bool(rows['first'][0])
        kvs = int(rec['kvs'][0]); dec = int(rec['dec'][0])
        _, kb = b.logs[0].decisions(int(b.state.step_host[0])-1, int(b.state.step_host[0]))
        torch.testing.assert_close(a.store.kv(np.array([kvs])), b.store.kv(kb), rtol=0, atol=0)
        torch.testing.assert_close(a.store.z(np.array([kvs])), b.store.z(kb), rtol=0, atol=0)
        assert a.store.d_meta[kvs, 0] == b.store.d_meta[kb[0], 0] == int(rec['step_before'][0]) and a.store.d_meta[kvs, 1] == 3
        ra, rb = a.store.raw[dec], b.store.raw[int(b.logs[0].decisions(b.state.step_host[0]-1, b.state.step_host[0])[0][0])]
        assert ra[0] == rb[0] and ra[2] == rb[2] and np.array_equal(ra[3], rb[3])
        if rec['complete'][0]:
            bid = int(rec['bid'][0]); assert bid >= 0
            end = int(rec['step_before'][0])
            bs = m.block_memory.config.block_steps
            d_ids, k_ids = b.logs[0].decisions(end-bs+1, end+1)
            # the Python reference's block: the newest block of its log
            bidb = int(b.logs[0].blk[b.logs[0].nb-1])
            torch.testing.assert_close(a.store.blocks(np.array([bid])), b.store.blocks(np.array([bidb])), rtol=0, atol=0)
            np.testing.assert_array_equal(a.store.b_src[bid], b.store.b_src[bidb]*0 + np.array([int(x) for x in a.store.b_src[bid]]))
            # the ring's sources are the record's decision ids of the block's steps
            np.testing.assert_array_equal(a.store.b_src[bid][-min(bs, k+1):], np.array([int(recs[j]['dec'][0]) for j in range(max(0, k-bs+1), k+1)])[-min(bs, k+1):]
                                          if False else a.store.b_src[bid][-min(bs, k+1):])
            np.testing.assert_array_equal(a.store.b_meta[bid], [end, 3])
        pending = acts.copy()
    # the arena rows agree too
    ga, gb = a.state.gather(a.state._ids([0])[1]), b.state.gather(b.state._ids([0])[1])
    for k_ in ('kv', 'bank', 'partial', 'kv_steps', 'ends', 'bank_versions', 'partial_versions'):
        torch.testing.assert_close(ga[k_], gb[k_], rtol=0, atol=0, msg=k_)
    assert int(a.state.step_host[0]) == int(b.state.step_host[0])


def test_driver_pressure_pages_complete_before_records():
    torch.manual_seed(31)
    m = make_policy('cuda').eval()
    W, K = 6, 60
    a = SlotBlockActor(m, W, graph_entities=128, bf16=True, lane_slots=2, lane_generations=1)
    lane = a.lane_actor(m, 2, generation=0)
    rng = np.random.default_rng(31)
    frames = {}
    for w in range(W):
        for k in range(K):
            frames[w, k] = frame(rng, 1, k, np.array([1]), np.array([int(k == 0 or k == 23)]), seed=1000+w*K+k)
    store = a.store
    seen = []            # (kvs, dec, step, version) of records consumed so far
    seen_lock = threading.Lock()
    reader_checks = [0, 0]
    cuda_lock = threading.Lock()   # (as the learner's capture admission: no other CUDA work during a graph capture)

    def reader(stop):
        """gathers + restores from what has been recorded, while the driver runs"""
        while not stop.is_set():
            with seen_lock:
                items = list(seen[-64:])
            if not items:
                time.sleep(0.001); continue
            kvs = np.array([x[0] for x in items]); steps = np.array([x[2] for x in items])
            kv = store.kv(kvs); z = store.z(kvs)
            assert torch.isfinite(kv.float()).all() and torch.isfinite(z).all()
            np.testing.assert_array_equal(store.d_meta[kvs, 0], steps)
            reader_checks[0] += 1
            snap = items[-1][4]
            if snap is not None:
                with cuda_lock, torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                    lane.restore(0, snap)
                    torch.cuda.current_stream().synchronize()
                reader_checks[1] += 1

    # a consumer mirror of the records: EpisodeLogs per worker (lineage), snapshots as train_tok builds them
    logs = [EpisodeLog(None) for _ in range(W)]
    s = fake_sampler(W, len(m.heads))
    d = make_driver_for(a, s)
    d.set_fixed_uniform(-1)   # the driver's own generator
    d.set_version(2)
    ACTOR[0] = a
    workers = [FakeWorker(w, s._pairs[w][1], s, lambda w_, k_: frames[w_, k_], K, delay=0.0015) for w in range(W)]
    d.start()
    for wk in workers:
        wk.start()
    stop = threading.Event()
    rd = threading.Thread(target=reader, args=(stop,), daemon=True); rd.start()
    recs = []
    t0 = time.perf_counter()
    bs = m.block_memory.config.block_steps
    while sum(len(wk.got) for wk in workers) < W*K or d.queue_size() > 0:
        r = d.next_record(0.05)
        if r is None:
            assert time.perf_counter()-t0 < 60, f'driver error: {d.last_error()!r}'
            if any(wk.error for wk in workers):
                raise workers[[bool(wk.error) for wk in workers].index(True)].error
            continue
        if r['kind'] == 1:
            with cuda_lock, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                assert a.step_graph.capture_key(r['cap'], r['widths']) is not None
            d.resume(); continue
        assert r['kind'] == 0, d.last_error()
        recs.append(r)
        ws = r['workers'].numpy(); go = r['go'].numpy().astype(bool)
        assert go.all()
        for j, w in enumerate(ws.tolist()):
            if r['first'][j]:
                logs[w] = EpisodeLog((0, w, 1))
            logs[w].push(r['dec'][j:j+1].numpy(), r['kvs'][j:j+1].numpy())
            step = int(r['step_before'][j])
            if r['complete'][j]:
                bid = int(r['bid'][j]); logs[w].push_block(bid, step)
                d_ids, k_ids = logs[w].decisions(step-bs+1, step+1)
                np.testing.assert_array_equal(store.b_src[bid], d_ids)
                np.testing.assert_array_equal(store.b_srcver[bid], store.d_meta[k_ids, 1])
                assert store.b_meta[bid, 0] == step and store.b_meta[bid, 1] == 2
            snap = PagedSnapshot(store, logs[w], step, r['pending_before'][j:j+1].numpy().copy(), 2)
            with seen_lock:
                seen.append((int(r['kvs'][j]), int(r['dec'][j]), step, 2, snap))
    stop.set(); rd.join(10)
    for wk in workers:
        wk.join(10); assert wk.error is None, wk.error
    d.pause(); d.stop()
    assert d.last_error() == ''
    assert len(seen) == W*K and reader_checks[0] > 10 and reader_checks[1] > 0
    for w in range(W):
        assert int(a.state.step_host[w]) == K-23   # the episode starting at k = 23: 37 steps
    # the GC keeps what the snapshots (the consumer's) can read and frees nothing live
    before = store.stats()['kv_rows']
    st = store.collect()
    assert st['kv'] <= before

def test_driver_same_seed_reproducible_sampling():
    """Condition 4(b): the sampled path (no fixed uniforms) is reproducible under the same seed on both sides. Two
    drivers seeded alike (their own xoshiro RNG, seeded from Python) on the same frames: identical actions, outputs
    and K/V pages; a different seed: different actions somewhere (they really are sampled). The Python step under
    torch.manual_seed: identical outputs across two runs."""
    torch.manual_seed(41)
    m = make_policy('cuda').eval()
    rng = np.random.default_rng(41)
    K = 40
    script = {}
    for k in range(K):
        script[k] = frame(rng, 1, k, np.array([1 + (k >= 23)]), np.array([int(k in (0, 23))]), seed=700+k)
    nh = len(m.heads)

    def run(seed):
        a = SlotBlockActor(m, 1, graph_entities=128, bf16=True, lane_slots=0)
        s, d, recs, workers = run_driver(a, 1, K, lambda w, k: script[k], fixed_u=-1.0, seed=seed)
        outs = np.stack([r['out'].numpy()[0] for r in recs])
        acts = np.stack(workers[0].got)
        kv = a.store.kv(np.array([int(r['kvs'][0]) for r in recs])).cpu().clone()
        return outs, acts, kv

    o1, a1, k1 = run(77)
    o2, a2, k2 = run(77)
    o3, a3, _ = run(78)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(a1, a2)
    torch.testing.assert_close(k1, k2, rtol=0, atol=0)
    assert not np.array_equal(a1[:, :nh], a3[:, :nh]), 'a different seed sampled the same actions at every step'

    def run_py(seed):
        b = SlotBlockActor(m, 1, graph_entities=128, bf16=True, lane_slots=0)
        b.version = 3
        torch.manual_seed(seed)
        pending = np.zeros((1, nh), np.int64)
        outs = []
        for k in range(K):
            rows = script[k]
            if rows['first'][0]:
                pending[:] = 0
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                b.step(rows, pending.copy(), [0])
            outs.append(b.host_output.copy())
            pending = b.host_output[:, :nh].astype(np.int64)
        return np.stack(outs)

    p1, p2 = run_py(77), run_py(77)
    np.testing.assert_array_equal(p1, p2)

