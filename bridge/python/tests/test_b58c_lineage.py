"""B58c gate: the native lineage (abp_lineage.h) == the Python lineage it replaces, on one scripted stream.

Two page stores driven identically: A keeps the Python lineage (EpisodeLog per worker, PagedSnapshot per row,
BlockRollout sidecars, ContextRegistry, BlockLaneService's injury bookkeeping, PagedStore.collect over the Python
snapshots); B keeps the native one (heads, snapshots, sidecars, contexts, injury table, the native collect). Every
step's page writes must get the same ids in both (so the GCs free the same rows in the same order), and at every
hand-off: the sidecars' payload (keys, steps, pending, versions, each snapshot's log window), the contexts, the
injury sources and hurt candidates, the pages / blocks / raw frames left after the GC.
Second test: the driver's lineage (lineage_main) == consume_record's Python lineage on the records of a driven run,
including the CF captures (status written, point snapshots) and the injury sources.
Skipped without the extension (no CUDA torch, no g++, ISAAC_BLOCK_CPP_ACTOR=0)."""
import types
import weakref

import numpy as np
import pytest
import torch

from test_block_memory import make_policy
from test_visible_input import synth_rows
from test_b52_cpp_actor import EXT
from isaac_bridge.block_paged import PagedStore, PagedSnapshot, EpisodeLog, native_snapshots
from isaac_bridge.block_training import BlockRollout
from isaac_bridge.block_lanes import BlockLaneService, ContextRegistry
from isaac_bridge.block_driver import NativeBlockRollout

pytestmark = pytest.mark.skipif(EXT is None, reason='cpp actor extension unavailable')

TAG = 77


class Heads:
    """A's slot logs as the actor's GC root (SlotBlockActor._live)."""
    def __init__(self, n):
        self.logs = [EpisodeLog(None) for _ in range(n)]

    def live(self):
        return [(log, log.end) for log in self.logs]


def injury_ns(W, depths):
    ns = types.SimpleNamespace(lin=None, episode={}, recent={}, hurt_candidates={}, injury_sources={},
                               depths=tuple(sorted(set(depths))))
    ns.recent_size = max(ns.depths, default=0)+2
    return ns


def same_window(snap_a, snap_b, cfg):
    """A's snapshot (full log) and B's materialized one (the branch window) read the same rows."""
    assert snap_a.step == snap_b.step and snap_a.policy_version == snap_b.policy_version
    np.testing.assert_array_equal(snap_a.pending, snap_b.pending)
    ref = snap_a.log.branch(snap_a.step, cfg)
    got = snap_b.log
    assert ref.offset == got.offset and ref.episode == got.episode
    np.testing.assert_array_equal(ref.dec[:ref.n], got.dec[:got.n])
    np.testing.assert_array_equal(ref.kvs[:ref.n], got.kvs[:got.n])
    np.testing.assert_array_equal(ref.blk[:ref.nb], got.blk[:got.nb])
    np.testing.assert_array_equal(ref.blk_end[:ref.nb], got.blk_end[:got.nb])


def same_store(a, b):
    np.testing.assert_array_equal(a.d_used, b.d_used)
    np.testing.assert_array_equal(a.b_used, b.b_used)
    np.testing.assert_array_equal(np.sort(a.native.raw_keys().numpy()), np.sort(b.native.raw_keys().numpy()))


def test_native_lineage_matches_python_lineage():
    torch.manual_seed(5)
    m = make_policy('cuda').eval()
    cfg = m.block_memory.config
    nh = len(m.heads)
    W, CH, P = 4, 8, 64
    A = PagedStore(cfg, nh, page=P)
    B = PagedStore(cfg, nh, page=P)
    assert A.native is not None and B.native is not None
    depths = [1, 3]
    B.native.lineage_configure(int(cfg.window), int(cfg.block_steps), int(cfg.memory_blocks), nh, W, W, depths, CH)
    heads = Heads(W)
    A.roots.append(weakref.WeakMethod(heads.live))
    side_a = [BlockRollout(CH), BlockRollout(CH)]
    side_b = [NativeBlockRollout(B, 0, CH), NativeBlockRollout(B, 1, CH)]
    ctx_a = ContextRegistry()
    inj = injury_ns(W, depths)
    rng = np.random.default_rng(58)
    kv_shape, z_shape, b_shape = A.kv_shape, A.z_shape, A.b_shape
    bs = int(cfg.block_steps)
    ep = np.zeros(W, np.int64)
    t = np.zeros(W, np.int64)
    idx = np.zeros(W, np.int64)     # each worker's rollout index
    cur = 0
    ver = 3
    owners = [101, 102, 103]
    idents = []
    for k in range(260):
        ws = np.sort(rng.choice(W, size=int(rng.integers(1, W+1)), replace=False))
        rows = synth_rows(rng, len(ws), seed=900+k)
        for j, w in enumerate(ws):
            first = k == 0 or t[w] == 0 or rng.random() < 0.03
            if first:
                ep[w] += 1; t[w] = 0
            rows[j]['first'] = int(first); rows[j]['episode'] = ep[w]; rows[j]['t'] = t[w]
            rows[j]['hurt'] = np.float32(rng.random() < 0.2)
        pend = rng.integers(0, 4, (len(ws), nh)).astype(np.int64)
        # observe (every row), the snapshots before the step
        BlockLaneService.observe_rows(inj, ws, rows)
        for j, w in enumerate(ws):
            B.native.inj_observe(int(w), bool(rows[j]['first']), int(rows[j]['episode']), int(rows[j]['t']), bool(rows[j]['hurt'] > 0))
        snaps_a, snaps_b = [], []
        for j, w in enumerate(ws):
            if rows[j]['first']:
                heads.logs[w] = EpisodeLog((TAG, int(w), int(ep[w])))
                B.native.lin_new_head(int(w), TAG, int(w), int(ep[w]))
            assert B.native.lin_head_end(int(w)) == heads.logs[w].end
            snaps_a.append(PagedSnapshot(A, heads.logs[w], heads.logs[w].end, pend[j:j+1].copy(), ver))
            snaps_b.append(B.native.lin_snap_row(int(w), torch.from_numpy(pend[j]), ver))
        # the step's pages: the same ids in both stores
        n = len(ws)
        kv = torch.randn((n, *kv_shape)); z = torch.randn((n, *z_shape))
        steps = np.array([heads.logs[w].end for w in ws], np.int64)
        da, ka = A.write_decisions(rows, pend, kv, z, steps, ver)
        db, kb = B.write_decisions(rows, pend, kv, z, steps, ver)
        np.testing.assert_array_equal(da, db); np.testing.assert_array_equal(ka, kb)
        # injury sources (record_main), contexts, the sidecar, the log pushes
        BlockLaneService.record_main(inj, ws, rows, snaps_a)
        for j, w in enumerate(ws):
            B.native.inj_record(int(w), int(rows[j]['episode']), int(rows[j]['t']), snaps_b[j])
            if rng.random() < 0.3:
                owner = int(rng.choice(owners))
                ia = ctx_a.put(owner, snaps_a[j]); ib = B.native.ctx_put(owner, snaps_b[j])
                assert ia == ib
                idents.append(ia)
            side_a[cur].record(w, idx[w], rows[j], snaps_a[j])
            B.native.side_record(cur, int(w), int(idx[w]), bool(rows[j]['first']), snaps_b[j])
            idx[w] += 1
            heads.logs[w].push(da[j:j+1], ka[j:j+1])
            B.native.lin_push(int(w), int(da[j]), int(ka[j]))
            if (heads.logs[w].end) % bs == 0:   # a block completes at this step
                vals = torch.randn((1, *b_shape))
                src = np.full((1, bs), -1, np.int64); srcv = np.full((1, bs), -1, np.int64)
                ba = A.write_blocks(vals, [heads.logs[w].end-1], ver, src, srcv)
                bb = B.write_blocks(vals, [heads.logs[w].end-1], ver, src, srcv)
                np.testing.assert_array_equal(ba, bb)
                heads.logs[w].push_block(int(ba[0]), heads.logs[w].end-1)
                B.native.lin_push_block(int(w), int(bb[0]), heads.logs[w].end-1)
            t[w] += 1
        for s in snaps_b:
            B.native.lin_release(s)
        del snaps_a
        if k % 37 == 36:   # the contexts' reclamation: an owner completes, a live set
            owner = int(rng.choice(owners))
            ctx_a.complete(owner); B.native.ctx_complete(owner)
            live = set(int(x) for x in rng.choice(idents, size=min(len(idents), 5), replace=False)) if idents else set()
            transient = bool(rng.random() < 0.5)
            ctx_a.retire(live, transient=transient)
            B.native.ctx_retire(sorted(live), transient)
            assert set(ctx_a.values) == {i for i in idents if i in ctx_a.values}
            for i in idents:
                if i in ctx_a.values:
                    same_window(ctx_a.get(i), native_snapshots(B, [B.native.ctx_get(i)])[0], cfg)
                else:
                    with pytest.raises(KeyError):
                        B.native.ctx_get(i)
        if k % 64 == 63:   # a hand-off: payloads, carry, the GC
            pa = side_a[cur]
            side_b[cur].materialize()
            assert list(pa.starts) == list(side_b[cur].starts)
            for w in pa.starts:
                assert list(pa.starts[w]) == list(side_b[cur].starts[w])
                for i in pa.starts[w]:
                    same_window(pa.starts[w][i], side_b[cur].starts[w][i], cfg)
            assert pa.open.keys() == side_b[cur].open.keys()
            for w in pa.open:
                assert pa.open[w][0] == side_b[cur].open[w][0]
                same_window(pa.open[w][1], side_b[cur].open[w][1], cfg)
            # injury sources and candidates
            src_b, hurt_b = (x.numpy() for x in B.native.inj_keys())
            assert set(map(tuple, src_b.tolist())) == set(inj.injury_sources)
            assert set(map(tuple, hurt_b.tolist())) == {k_ for s_ in inj.hurt_candidates.values() for k_ in s_}
            for key in list(inj.injury_sources)[:20]:
                same_window(inj.injury_sources[key], native_snapshots(B, [B.native.inj_get(*key)])[0], cfg)
            ws_o = np.flatnonzero(idx > 0); ks_o = idx[ws_o]-1
            nxt = 1-cur
            side_a[nxt].carry(pa, ws_o, ks_o)
            side_b[nxt].carry(side_b[cur], ws_o, ks_o)
            idx[:] = 0; idx[ws_o] = 1
            cur = nxt
            payload_b = side_b[1-cur]   # (the handed rollout's materialized snapshots stay alive, as a job's)
            ga = A.collect(); gb = B.collect()
            assert (ga['freed_kv'], ga['freed_blocks'], ga['freed_raw']) == (gb['freed_kv'], gb['freed_blocks'], gb['freed_raw'])
            assert ga['freed_kv'] > 0 or k < 100
            same_store(A, B)
            del payload_b
    st = B.native.lin_stats()
    assert st['heads'] == W and st['contexts'] == len(ctx_a.values) and st['injury'] == len(inj.injury_sources)


def test_native_points_follow_teacher_generations():
    """capture / publish / release: TeacherGenerations' choice of slot, refs, backpressure; the status array."""
    torch.manual_seed(6)
    m = make_policy('cuda').eval()
    cfg = m.block_memory.config
    nh = len(m.heads)
    S = PagedStore(cfg, nh, page=64)
    S.native.lineage_configure(int(cfg.window), int(cfg.block_steps), int(cfg.memory_blocks), nh, 2, 2, [1], 8)
    status = np.zeros((2, 8), np.int64)
    S.native.lineage_status(int(status.ctypes.data))
    S.native.lin_new_head(0, TAG, 0, 1)
    snaps = {v: S.native.lin_snap_row(0, torch.zeros(nh, dtype=torch.long), v) for v in (1, 2, 3)}
    from isaac_bridge.block_memory import TeacherGenerations
    g = TeacherGenerations()
    # version 1: no generation yet -> publish slot 0 (min version among free: both -1, the first)
    assert S.native.pt_capture(0, snaps[1], 1) == -2-(g.versions.index(min(g.versions)))
    assert g.publish(m, 1); i = g.versions.index(1); S.native.pt_published(i, 1)
    assert S.native.pt_capture(0, snaps[1], 1) == 2 and status.flat[0] == 2
    g.acquire(1)
    # version 2 -> the other slot
    c = S.native.pt_capture(1, snaps[2], 2)
    assert g.publish(m, 2); j = g.versions.index(2)
    assert c == -2-j and j != i
    S.native.pt_published(j, 2)
    assert S.native.pt_capture(1, snaps[2], 2) == 2; g.acquire(2)
    # version 3 with both slots pinned: refused (backpressure), status -1
    assert not g.publish(m, 3)
    assert S.native.pt_capture(2, snaps[3], 3) == -1 and status.flat[2] == -1
    with pytest.raises(RuntimeError):
        S.native.pt_capture(0, snaps[1], 1)   # (point 0 not released yet)
    tok = S.native.pt_release(0); g.release(i)
    assert tok == 1 and S.native.pt_release(0) == -1
    versions, refs = S.native.pt_gens()
    assert list(versions) == list(g.versions) and list(refs) == list(g.refs)
    for s in snaps.values():
        S.native.lin_release(s)
    assert S.native.lin_stats()['points'] == 1


def test_driver_lineage_matches_consume_record():
    """The driver's lineage_main on a driven run (several fake workers, episode starts, hurts, CF capture requests) ==
    consume_record's Python lineage over the same records: each worker's log end, every injury source and hurt
    candidate (same snapshot windows), every captured point's snapshot and status."""
    from test_b52_cpp_actor import fake_sampler, make_driver_for, FakeWorker, collect_records, frame, ACTOR
    from isaac_bridge.block_paged import SlotBlockActor
    from isaac_bridge.tok_obs import ROW
    torch.manual_seed(31)
    m = make_policy('cuda').eval()
    cfg = m.block_memory.config
    nh = len(m.heads)
    W, K = 3, 70
    actor = SlotBlockActor(m, W, graph_entities=128, bf16=True, lane_slots=0)
    st = actor.store
    depths = [1, 2]
    st.native.lineage_configure(int(cfg.window), int(cfg.block_steps), int(cfg.memory_blocks), nh, int(actor.state.slots), W,
                                depths, 8)
    status = np.zeros((W, K), np.int64)
    st.native.lineage_status(int(status.ctypes.data))
    st.native.pt_published(0, 3)   # (the driver steps at version 3: its generation exists)
    s = fake_sampler(W, nh)
    d = make_driver_for(actor, s)
    d.set_fixed_uniform(0.5); d.set_version(3)
    fields = {n: (int(ROW.fields[n][1]), int(ROW.fields[n][0].itemsize)) for n in ('episode', 't', 'hurt')}
    d.set_lineage(True, TAG, fields)
    ACTOR[0] = actor
    rng = np.random.default_rng(31)
    script, cap = {}, {}
    for w in range(W):
        ep, t = 0, 0
        for k in range(K):
            first = k == 0 or rng.random() < 0.04
            if first:
                ep += 1; t = 0
            r = frame(rng, 1, t, np.array([ep]), np.array([int(first)]), seed=500+97*w+k)
            r['hurt'] = np.float32(rng.random() < 0.25)
            script[w, k] = r
            if rng.random() < 0.08:
                cap[w, k] = w*K+k   # (a point of its own)
            t += 1

    def make_rows(w, k):
        if (w, k) in cap:
            s.cf_snap[w] = cap[w, k]+1
        return script[w, k]
    workers = [FakeWorker(w, s._pairs[w][1], s, make_rows, K) for w in range(W)]
    d.start()
    for wk in workers:
        wk.start()
    import time as time_
    recs, t0 = [], time_.perf_counter()
    try:   # (records until every worker's every row was stepped: a step takes several workers' rows)
        while sum(len(r['workers']) for r in recs) < W*K:
            assert time_.perf_counter()-t0 < 60, (f'driver error {d.last_error()!r}, counters {d.counters()}, workers got '
                                                  f'{[len(wk.got) for wk in workers]}')
            recs += collect_records(d, 1, timeout=30.0)
        for wk in workers:
            wk.join(10); assert wk.error is None
    finally:
        d.pause(); d.stop()
    assert d.last_error() == ''
    # the Python lineage over the records (consume_record's order)
    inj = injury_ns(W, depths)
    logs = {}
    points = {}
    for rec in recs:
        idx = rec['workers'].numpy(); go = rec['go'].numpy().astype(bool)
        rows = np.frombuffer(rec['rows'].numpy().tobytes(), ROW)
        BlockLaneService.observe_rows(inj, idx, rows)
        a = idx[go]; sub = rows[go]
        sb = rec['step_before'].numpy(); fr = rec['first'].numpy().astype(bool); pb = rec['pending_before'].numpy()
        ver = int(rec['version'])
        ctx = []
        for j, w in enumerate(a.tolist()):
            if fr[j]:
                logs[w] = EpisodeLog((TAG, w, int(sub['episode'][j])))
            ctx.append(PagedSnapshot(st, logs[w], int(sb[j]), pb[j:j+1].copy(), ver))
        for j, req in enumerate(rec['cf_snap'].numpy()[go].tolist()):
            if req:
                points[req-1] = ctx[j]
        BlockLaneService.record_main(inj, a, sub, ctx)
        dec = rec['dec'].numpy(); kvs = rec['kvs'].numpy(); bid = rec['bid'].numpy(); comp = rec['complete'].numpy().astype(bool)
        for j, w in enumerate(a.tolist()):
            logs[w].push(dec[j:j+1], kvs[j:j+1])
            if comp[j]:
                logs[w].push_block(int(bid[j]), int(sb[j]))
    for w in range(W):
        assert st.native.lin_head_end(actor.base+w) == logs[w].end
    src_b, hurt_b = (x.numpy() for x in st.native.inj_keys())
    assert set(map(tuple, src_b.tolist())) == set(inj.injury_sources)
    assert set(map(tuple, hurt_b.tolist())) == {k_ for s_ in inj.hurt_candidates.values() for k_ in s_}
    for key, snap in inj.injury_sources.items():
        same_window(snap, native_snapshots(st, [st.native.inj_get(*key)])[0], cfg)
    assert sorted(points) == sorted(cap.values()) == st.native.pt_points()
    for p, snap in points.items():
        got = st.native.pt_get(p)
        assert got is not None and int(got[1]) == 0 and int(got[2]) == 3
        same_window(snap, native_snapshots(st, [int(got[0])])[0], cfg)
        assert status.flat[p] == 2
    c = d.counters()
    assert c['cap_native'] == len(cap) and c['cap_publish'] == 0 and c['cap_refused'] == 0


def test_native_points_with_n_generations():
    """The point table with N = 3 teacher generations (configurable, default 2): three versions pinned, the fourth
    refused; a release frees its slot (the oldest free version chosen)."""
    torch.manual_seed(7)
    m = make_policy('cuda').eval()
    cfg = m.block_memory.config
    nh = len(m.heads)
    S = PagedStore(cfg, nh, page=64)
    S.native.lineage_configure(int(cfg.window), int(cfg.block_steps), int(cfg.memory_blocks), nh, 2, 2, [1], 8, generations=3)
    S.native.lin_new_head(0, TAG, 0, 1)
    snaps = {v: S.native.lin_snap_row(0, torch.zeros(nh, dtype=torch.long), v) for v in range(1, 6)}
    for p, v in enumerate((1, 2, 3)):
        c = S.native.pt_capture(p, snaps[v], v)
        assert c == -2-p
        S.native.pt_published(-2-c, v)
        assert S.native.pt_capture(p, snaps[v], v) == 2
    assert S.native.pt_capture(3, snaps[4], 4) == -1
    assert S.native.pt_release(1) >= 0                 # (version 2's slot free)
    assert S.native.pt_capture(3, snaps[4], 4) == -2-1
    versions, refs = S.native.pt_gens()
    assert list(versions) == [1, 2, 3] and list(refs) == [1, 0, 1]
    for s in snaps.values():
        S.native.lin_release(s)
