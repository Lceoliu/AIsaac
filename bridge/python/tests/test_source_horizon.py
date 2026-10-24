"""Bounded source gradient horizon: exact reads, per-pair masked credit, freed graphs.

Synthetic episodes of 48 decisions, block 4, recent window 4, horizon H=12.
1. Reader forward under H equals H=0 bit for bit (same parameters).
2. The batched masked source cotangent equals an independent per-decision
   reference: native H=0 autograd for one decision at a time, its leaf rows
   older than H zeroed. Reader weights receive every pair's gradient.
3. Graphs no future in-horizon decision can reach are released (weakrefs die,
   values remain readable and unchanged); their pending credit is settled in
   the same Adam interval (parameters equal a pipeline that never detaches).
4. One decision's loss: frames inside the horizon get non-zero gradient,
   frames outside get none (detached frames have no graph at all).
5. Source-process workers detach the same graphs (parameters match local).
"""
import argparse, copy, gc, json, sys, weakref
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch, SourceView
from isaac_bridge.source_pipeline import SourcePipeline, source_parameter

H, T, BLOCK, W, MICRO = 12, 48, 4, 4, 8
CFG = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True,
           temporal_width=32, temporal_layers=2, temporal_heads=4, temporal_window=W,
           rebuild_memory=True, memory_block=BLOCK, memory_tokens=2, source_cache=True)


def episodes(seed, heads):
    rng = np.random.default_rng(seed)
    seq = TS.make_seq(rng, nw=2, T=T, changes={(0, 8), (0, 20), (1, 12), (1, 30)}, resets=set(), exits=set())
    hs = [RawHistory.empty(), RawHistory.empty()]
    rollouts = {1: [], 2: []}
    for t, rows in enumerate(seq):
        pending = M.synth_pending(rng, 2, heads)
        for w in range(2):
            if 24 <= t < 32: rollouts[1].append((hs[w], rows[w], pending[w]))
            if 40 <= t < 48: rollouts[2].append((hs[w], rows[w], pending[w]))
            hs[w] = hs[w].append(rows[w], pending[w], BLOCK)
    return hs, rollouts


def loss_of(out, weights):
    per = out[0].square().mean(-1) + out[1].reshape(len(weights), -1).square().mean(-1) + out[-1].reshape(len(weights), -1).square().mean(-1)
    return (per * weights).sum()


def frontiers(req):
    f = {}
    for h, _, _ in req: f[h.uid] = min(f.get(h.uid, h.length), h.length)
    return f


def run():
    p = argparse.ArgumentParser()
    p.add_argument('--no-process', action='store_true')
    p.add_argument('--learner', default='cpu')
    p.add_argument('--sources', default='cpu,cpu')
    a = p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(29)
    base = TokPolicy(**CFG).to(a.learner).eval()
    sources = a.sources.split(',')
    hs, rollouts = episodes(29, base.heads)
    req1, req2 = rollouts[1], rollouts[2]
    n1, n2 = len(req1), len(req2)
    weights1 = (torch.rand(n1) + .5).to(a.learner)
    weights2 = (torch.rand(n2) + .5).to(a.learner)
    names = ['detach', 'reference', 'unbounded'] + ([] if a.no_process else ['process'])
    models = {k: copy.deepcopy(base) for k in names}
    pipes = {k: SourcePipeline(models[k], sources, period=0, micro=MICRO,
                               update_rollouts=2, horizon=0 if k == 'unbounded' else H, processes=k == 'process')
             for k in names}
    opts = {k: torch.optim.Adam(models[k].main_parameters(), lr=1e-3, eps=1e-5) for k in names}
    report = {}

    # ---- rollout 1 -------------------------------------------------------
    banks = {}
    for k in names:
        pipes[k].retain_histories({h.uid for h, _, _ in req1}, frontiers=None if k == 'reference' else frontiers(req1))
        with torch.no_grad(): banks[k] = SourceBatch(models[k], req1, cache=False, pipeline=pipes[k])
    views = {k: SourceView(banks[k], np.arange(n1)) for k in names}
    outs = {k: views[k].forward(np.arange(n1), need_vg=True) for k in names}
    # 1. reads: bitwise identical to the unbounded pipeline
    for k in names:
        for x, y in zip(outs[k], outs['unbounded']):
            if k == 'process':   # separate-process encoding: v2 itself is not bitwise to in-process
                torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-6)
            else:
                assert torch.equal(x, y), f'{k}: horizon changed the Reader forward'
    report['forward_bitwise_equal'] = True
    hb = banks['detach']
    layout, _ = hb.grouped_layout(np.arange(n1))
    report['masked_visible_pairs'] = int((layout['valid'] & ~layout['grad_valid']).sum())
    assert report['masked_visible_pairs'] > 0, 'test does not exercise the horizon mask'

    # 2. per-pair masked cotangent vs per-decision native reference
    for k in ('detach',):
        loss_of(outs[k], weights1).backward()
    mH = models['detach']
    ref_u = torch.zeros_like(hb.u); ref_m = torch.zeros_like(hb.mem); ref_y = torch.zeros_like(hb.y)
    full_m = torch.zeros_like(hb.mem)
    ub = banks['unbounded']; m0 = models['unbounded']
    frame_age_key = [ub.frame_keys[i] for i in ub.u_frames]
    block_start = [key[0] for key in ub.block_keys for _ in range(m0.reader.tokens)]
    for i in range(n1):
        ub.u.grad = ub.mem.grad = ub.y.grad = None
        view = SourceView(ub, np.asarray([i]))
        out = view.forward(np.asarray([0]), need_vg=True)
        loss_of(out, weights1[i:i+1]).backward()
        uid, t = req1[i][0].uid, req1[i][0].length
        keep_u = torch.tensor([u == uid and t - s <= H for u, s in frame_age_key], device=a.learner)[:, None]
        keep_m = torch.tensor([u == uid and t - s <= H for u, s in block_start], device=a.learner)[:, None]
        if ub.u.grad is not None: ref_u += ub.u.grad * keep_u
        if ub.mem.grad is not None:
            ref_m += ub.mem.grad * keep_m; full_m += ub.mem.grad
        if ub.y.grad is not None: ref_y += ub.y.grad
    torch.testing.assert_close(hb.u.grad, ref_u, rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(hb.mem.grad, ref_m, rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(hb.y.grad, ref_y, rtol=1e-4, atol=1e-7)
    report['masked_memory_grad_removed'] = float((full_m - ref_m).abs().sum())
    assert report['masked_memory_grad_removed'] > 0, 'mask removed nothing'
    worst = 0.
    for (name, x), (_, y) in zip(mH.named_parameters(), m0.named_parameters()):
        if x.grad is None and y.grad is None: continue
        worst = max(worst, float((x.grad - y.grad).abs().max()))
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-6, msg=name)
    report['reader_weight_grad_max_abs_vs_unbounded'] = worst
    report['per_pair_reference'] = 'PASS'
    for k in ('reference', 'process'):
        if k in names: loss_of(outs[k], weights1).backward()
    for k in names:
        if k == 'unbounded': continue
        views[k].backward(); pipes[k].fast_step(opts[k], 1e9); pipes[k].finish_rollout(opts[k], 1e9)
    del views, outs, view, out, banks, ub, hb, layout

    # ---- rollout 2: retire (detach) before the next rollout's reads ---------
    del models['unbounded'], pipes['unbounded'], opts['unbounded']; names.remove('unbounded'); gc.collect()
    pD = pipes['detach']
    old_u = {i: weakref.ref(c['future'].result()['u']) for i, c in enumerate(pD.chunks) if c is not None}
    old_m = {i: weakref.ref(c['future'].result()['mem']) for i, c in enumerate(pD.writes) if c is not None}
    for k in names:
        pipes[k].retain_histories({h.uid for h, _, _ in req2}, frontiers=None if k == 'reference' else frontiers(req2))
    for future in pD.control_futures: future.result()   # detach runs FIFO after the settling VJP
    gc.collect()
    detached_c = [i for i, c in enumerate(pD.chunks) if c is not None and c.get('detached')]
    detached_w = [i for i, c in enumerate(pD.writes) if c is not None and c.get('detached')]
    limit = 40 - H
    assert detached_c and detached_w, 'nothing detached'
    for i, c in enumerate(pD.chunks):
        if c is None: continue
        r = c['future'].result()
        if c.get('detached'):
            assert c['last'] < limit and r['u'].grad_fn is None and not r['u'].requires_grad and r['y'] is None
            assert old_u[i]() is None, f'detached chunk {i} graph still referenced'
        else:
            assert r['u'].grad_fn is not None
    for i, c in enumerate(pD.writes):
        if c is None: continue
        r = c['future'].result()
        if c.get('detached'):
            assert c['last_start'] < limit and r['mem'].grad_fn is None
            assert old_m[i]() is None, f'detached write {i} graph still referenced'
    pinned = [i for i, c in enumerate(pD.chunks) if c is not None and not c.get('detached') and c['last'] < limit]
    report['detached_chunks'], report['detached_writes'] = len(detached_c), len(detached_w)
    report['pinned_by_live_writer'] = len(pinned)
    report['settled_packets'] = pD.horizon_detached['packets']
    assert pD.horizon_detached['packets'] > 0, 'accumulated credit of detached graphs was not settled'
    banks = {}
    for k in names:
        with torch.no_grad(): banks[k] = SourceBatch(models[k], req2, cache=False, pipeline=pipes[k])
    stats = pD.horizon_stats()
    # Exact invariant: a graph is kept only if some future decision can reach
    # it, or a reachable Writer graph (last block start >= limit) consumes it.
    live_writes = [c for c in pD.writes if c is not None and not c.get('detached')]
    assert all(c['last_start'] >= limit for c in live_writes)
    pinning = set().union(*(c['chunks'] for c in live_writes))
    for i, c in enumerate(pD.chunks):
        if c is not None and not c.get('detached'):
            assert c['last'] >= limit or i in pinning, (i, c['last'], limit)
    report['graph_frames_uid_max'] = stats['source_graph_frames_uid_max']
    report['graph_frames_reachable_per_uid'] = T - limit
    views = {k: SourceView(banks[k], np.arange(n2)) for k in names}
    outs = {k: views[k].forward(np.arange(n2), need_vg=True) for k in names}
    # Detached values are read exactly as the never-detached reference reads them.
    for x, y in zip(outs['detach'], outs['reference']):
        assert torch.equal(x, y), 'detached values read differently'
    for k in names:
        loss_of(outs[k], weights2).backward()
        views[k].backward(); pipes[k].fast_step(opts[k], 1e9); pipes[k].finish_rollout(opts[k], 1e9)
    for k in names: pipes[k].finish(opts[k], 1e9)
    for k in names:
        if k != 'reference': assert pipes[k].version == 1, (k, pipes[k].version)
    dropped = pD.horizon_stats()['source_horizon_dropped_grad']
    assert dropped == 0., f'in-horizon gradient dropped: {dropped}'
    report['dropped_grad'] = dropped
    worst = 0.
    for (name, x), (_, y) in zip(models['detach'].named_parameters(), models['reference'].named_parameters()):
        worst = max(worst, float((x - y).detach().abs().max()))
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-7, msg=name)
    report['detach_vs_never_detached_weight_max_abs'] = worst
    if 'process' in names:
        worst = 0.
        for (name, x), (_, y) in zip(models['detach'].named_parameters(), models['process'].named_parameters()):
            worst = max(worst, float((x - y).detach().abs().max()))
            torch.testing.assert_close(x, y, rtol=1e-4, atol=2e-6, msg=name)
        report['process_vs_local_weight_max_abs'] = worst
        memory = pipes['process'].remote_memory
        report['process_detached_chunks'] = sum(m['detached_chunks'] for m in memory)
        report['process_detached_writes'] = sum(m['detached_writes'] for m in memory)
        assert report['process_detached_chunks'] == len(detached_c), (memory, detached_c)
        assert report['process_detached_writes'] == len(detached_w), (memory, detached_w)
    del views, outs, banks
    for k in names: pipes[k].close()

    # ---- 4. one decision's loss --------------------------------------------
    per_frame = {}
    for horizon in (H, 0):
        torch.manual_seed(31)
        model = copy.deepcopy(base)
        pipe = SourcePipeline(model, sources, period=0, micro=MICRO, update_rollouts=1, horizon=horizon)
        opt = torch.optim.Adam(model.main_parameters(), lr=1e-3, eps=1e-5)
        pipe.retain_histories({h.uid for h, _, _ in req1}, frontiers=frontiers(req1))
        with torch.no_grad(): bank = SourceBatch(model, req1, cache=False, pipeline=pipe)
        view = SourceView(bank, np.arange(n1)); loss_of(view.forward(np.arange(n1), need_vg=True), weights1).backward()
        view.backward(); pipe.fast_step(opt, 1e9); pipe.finish_rollout(opt, 1e9)
        target = next(r for r in req2 if r[0].length == 47 and r[0].uid == hs[0].uid)
        pipe.retain_histories({target[0].uid}, frontiers={target[0].uid: 47})
        with torch.no_grad(): bank = SourceBatch(model, [target], cache=False, pipeline=pipe)
        outputs = {}
        for i, c in enumerate(pipe.chunks):
            if c is None or c['uid'] != target[0].uid: continue
            u = c['future'].result()['u']
            if u.requires_grad: u.retain_grad()
            outputs[i] = u
        view = SourceView(bank, np.asarray([0]))
        loss_of(view.forward(np.asarray([0]), need_vg=True), torch.ones(1, device=a.learner)).backward()
        view.backward(); pipe.fast_step(opt, 1e9); pipe.finish_rollout(opt, 1e9)
        grads = {}
        for (uid, s), (cid, offset) in pipe.frames.items():
            if uid != target[0].uid: continue
            u = outputs[cid]
            grads[s] = (None if not u.requires_grad else
                        0. if u.grad is None else float(u.grad[offset].abs().sum()))
        per_frame[horizon] = grads
        pipe.close()
    inside = set(range(36, 48))   # blocks 9-10 (start >= 47-12), recent 43-46, query 47
    for s, g in per_frame[H].items():
        if s in inside: assert g is not None and g > 0, (s, g)
        else: assert not g, (s, g)   # None: no graph; 0.0: graph kept by its chunk, zero gradient
    assert all(per_frame[H][s] is None for s in range(0, 47 - H) if s < 24), 'old chunk graph not released'
    reached_unbounded = [s for s, g in per_frame[0].items() if g]
    assert min(reached_unbounded) < 47 - H, 'H=0 control does not reach beyond the horizon'
    report['single_decision'] = dict(t=47, horizon=H, nonzero=sorted(s for s, g in per_frame[H].items() if g),
                                     no_graph=sorted(s for s, g in per_frame[H].items() if g is None),
                                     graph_zero=sorted(s for s, g in per_frame[H].items() if g == 0.),
                                     unbounded_nonzero_min=min(reached_unbounded))
    print(json.dumps(dict(result='PASS', learner=a.learner, sources=a.sources, **report)))


if __name__ == '__main__':
    run()
