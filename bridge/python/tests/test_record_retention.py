"""Record retention (approved 2026-10-24): references not copies, release after
consumption, a byte budget. CPU, small model.

1. share_prefix: a pickled branch history (full prefix copy) shares its
   ancestor's arrays afterwards; rows are unchanged byte for byte.
2. A branch's prefix frames are its parent's source frames: encoding only its
   own frames; an unconsumed record's loss reaches the parent's graphs.
3. After the consuming update the record's own graphs detach (frontier inf);
   a later read of it is frozen: no source cotangent, zero dropped gradient,
   Reader weights still learn.
4. RecordLedger budget evicts the oldest records whole; draws are unchanged
   while nothing was evicted.
"""
import copy, json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch, SourceView, share_prefix
from isaac_bridge.record_retention import RecordLedger
from isaac_bridge.source_pipeline import SourcePipeline

BLOCK = 4
CFG = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
           temporal_heads=4, temporal_window=4, rebuild_memory=True, memory_block=BLOCK, memory_tokens=2, source_cache=True)


def main():
    torch.set_num_threads(2); torch.manual_seed(7)
    model = TokPolicy(**CFG).eval()
    rng = np.random.default_rng(7)
    seq = TS.make_seq(rng, nw=1, T=40, changes={(0, 9), (0, 21)}, resets=set(), exits=set())
    pend = [M.synth_pending(rng, 1, model.heads)[0] for _ in seq]
    parent = RawHistory.empty()
    prefixes = []
    for t, rows in enumerate(seq[:30]):
        prefixes.append(parent); parent = parent.append(rows[0], pend[t], BLOCK)
    # a branch at frame 18 (inside block 4), then 12 own frames
    branch = prefixes[18].fork()
    assert branch.origin == ((parent.uid, 18),) and branch.key(17) == (parent.uid, 17) and branch.key(18) == (branch.uid, 18)
    records = []
    for t in range(18, 30):
        records.append(branch); branch = branch.append(seq[t][0], pend[t], BLOCK)
    # 1. a pickled copy shares the parent's arrays again
    copied = pickle.loads(pickle.dumps(records[-1]))
    shared, n = share_prefix(copied, parent, BLOCK)
    assert n == 16, n   # whole blocks 0-3 (frames 0-15); block 4 mixes parent rows 16-17 and own rows
    assert all(a[0] is b[0] for a, b in zip(shared.blocks[:4], parent.blocks[:4]))
    r0, p0 = records[-1].arrays(); r1, p1 = shared.arrays()
    assert r0.tobytes() == r1.tobytes() and np.array_equal(p0, p1)
    ledger = RecordLedger(16, 16, float('inf'), r0.dtype.itemsize, BLOCK)
    ledger.on_rollout([parent])
    deduped = ledger.share([pickle.loads(pickle.dumps(h)) for h in records[-3:]])
    assert ledger.shared_frames >= 3 * 16
    # 2. sources: the branch's prefix is the parent's
    pipe = SourcePipeline(model, ['cpu', 'cpu'], period=0, micro=8, update_rollouts=1, horizon=8)
    opt = torch.optim.Adam(model.main_parameters(), lr=1e-3, eps=1e-5)
    main_req = [(prefixes[t], seq[t][0], pend[t]) for t in range(22, 30)]
    with torch.no_grad(): bank = SourceBatch(model, main_req, cache=False, pipeline=pipe)
    encoded = pipe.encoded_rows
    rec_req = [(h, seq[h.length][0], pend[h.length]) for h in deduped]
    extra = SourceBatch(model, rec_req, cache=False, pipeline=pipe)
    own = deduped[-1].length + 1 - 18
    assert pipe.encoded_rows - encoded == own, (pipe.encoded_rows - encoded, own)
    parent_cids = {pipe.frames[(parent.uid, t)][0] for t in range(18)}
    keep = {}
    for cid in parent_cids:
        u = pipe.chunks[cid]['future'].result()['u']; u.retain_grad(); keep[cid] = u
    view = SourceView(bank, np.arange(len(main_req)))
    out = view.forward(np.arange(len(main_req)), need_vg=True)
    o2 = extra.forward(np.arange(len(rec_req)), need_vg=True)
    (out[0].square().mean() + o2[0].square().mean() + o2[1].square().mean()).backward()
    view.backward(); extra.backward(); pipe.fast_step(opt, 1e9); pipe.finish_rollout(opt, 1e9)
    reached = sum(float(u.grad.abs().sum()) for u in keep.values() if u.grad is not None)
    assert reached > 0, 'record loss did not reach the parent graph'
    # 3. consumed: frontier inf for the branch -> its own chunks detach
    slots = np.arange(len(deduped))
    ledger.admit_v2(slots); ledger.picked(v2=slots); ledger.end_update()
    front = ledger.frontier({parent.uid: 29}, None, list(deduped))
    assert front[deduped[0].uid] == float('inf')
    pipe.retain_histories({parent.uid} | {h.uid for h in deduped}, frontiers=front)
    for f in pipe.control_futures: f.result()
    own_chunks = [c for c in pipe.chunks if c is not None and c.get('uid') == deduped[0].uid]
    assert own_chunks and all(c.get('detached') for c in own_chunks)
    frozen = ledger.frozen(v2=slots)
    assert frozen == list(range(len(slots)))
    again = SourceBatch(model, rec_req, cache=False, pipeline=pipe, frozen=frozen)
    model.zero_grad(set_to_none=True)
    o3 = again.forward(np.arange(len(rec_req)), need_vg=True)
    (o3[0].square().mean() + o3[1].square().mean()).backward()
    again.backward(); pipe.finish_rollout(opt, 1e9)
    dropped = pipe.horizon_stats()['source_horizon_dropped_grad']
    assert dropped == 0., dropped
    reader = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters() if n.startswith('reader.layers') and p.grad is not None)
    assert reader > 0, 'frozen reads must still train the Reader'
    pipe.close()
    # 4. budget: oldest whole records go first; draws unchanged before eviction
    led = RecordLedger(4, 8, budget_bytes=10 * 100, row_bytes=100, block=BLOCK)
    hs = [RawHistory((1, i), length=4, origin=(((0, 0), 2),)) for i in range(8)]   # 3 own frames each
    led.admit_v2(np.arange(8))
    g1, g2 = np.random.default_rng(3), np.random.default_rng(3)
    assert np.array_equal(led.sample_v2(g1, 8, 16), g2.integers(0, 8, 16))
    contexts = list(hs)
    evicted = led.enforce(contexts, 0, 0.)
    assert evicted == list(range(5)), evicted   # 8 x 300 B > 1000 B: oldest go until 3 x 300 <= 1000
    assert led.last['record_bytes'] <= 1000 and led.last['record_evicted'] == 5
    draw = led.sample_v2(np.random.default_rng(4), 8, 64)
    assert set(draw) <= {5, 6, 7}
    print(json.dumps(dict(result='PASS', shared_frames=int(ledger.shared_frames), own_encoded=own,
                          parent_graph_gradient=reached, frozen_dropped=dropped, reader_gradient=reader,
                          evicted=evicted)))


if __name__ == '__main__':
    main()
