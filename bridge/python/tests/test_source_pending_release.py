"""retain_histories while an interval's VJP is pending (CPU, small model).

The learner's order with a deferred source VJP (update every rollout, overlap): retain, the
rollout's SourceBatch, dispatch the previous interval's VJP, Reader steps, commit, close the
interval. Pipeline A releases while the VJP is pending (chunks of its packets and their episodes
kept); pipeline B releases only without a pending VJP (the behaviour before 2026-10-25).
1. Same parameters after every rollout up to float summation order (sources and Reader; observed
   max difference ~1e-8 from the third interval on).
2. A frees an ended episode without pending credit at once, one with pending credit one rollout
   later; B keeps graphs while anything is pending.
"""
import copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch, SourceView
from isaac_bridge.source_pipeline import SourcePipeline

torch.set_num_threads(2); torch.manual_seed(31)
cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
           temporal_heads=4, temporal_window=4, rebuild_memory=True, memory_block=4, memory_tokens=2, source_cache=True)
base = TokPolicy(**cfg).eval()
rng = np.random.default_rng(31)
W = 3
seq = TS.make_seq(rng, nw=W, T=40, changes={(0, 6), (1, 9), (2, 13)}, resets=set(), exits=set())
pend = [M.synth_pending(rng, W, base.heads) for _ in seq]
runs = {}
for name in ('A', 'B'):
    model = copy.deepcopy(base)
    pipe = SourcePipeline(model, ['cpu', 'cpu'], period=0, micro=8, update_rollouts=1, defer_backward=True)
    opt = torch.optim.Adam(model.main_parameters(), lr=1e-3, eps=1e-5)
    hs = [RawHistory.empty() for _ in range(W)]
    alive = [True] * W
    end_at = {0: 0, 1: 1}   # worker 0's episode ends after rollout 0, worker 1's after rollout 1; 2 lives on
    graph_frames, params = [], []
    for rollout in range(4):
        live = {hs[w].uid for w in range(W) if alive[w]}
        if name == 'A' or pipe.pending is None:
            pipe.retain_histories(live)
        graph_frames.append({w: sum(c['rows'] for c in pipe.chunks if c is not None and not c.get('detached')
                                    and c.get('uid') == hs[w].uid) for w in range(W)})
        req = []
        for t in range(rollout * 8, rollout * 8 + 8):
            for w in range(W):
                if alive[w]:
                    req.append((hs[w], seq[t][w], pend[t][w])); hs[w] = hs[w].append(seq[t][w], pend[t][w], 4)
        with torch.no_grad(): bank = SourceBatch(model, req, cache=False, pipeline=pipe)
        pipe.dispatch_finish()
        view = SourceView(bank, np.arange(len(req)))
        out = view.forward(np.arange(len(req)), need_vg=True)
        (out[0].square().mean() + out[1].square().mean()).backward()
        view.backward(); pipe.fast_step(opt, 1e9)
        pipe.poll_finish(opt, 1e9, wait=True)
        pipe.finish_rollout(opt, 1e9, overlap=True)
        for w, r_ in end_at.items():
            if r_ == rollout: alive[w] = False
        params.append([p.detach().clone() for p in model.parameters()])
    pipe.close()
    runs[name] = (graph_frames, params, hs)
gA, pA, hA = runs['A']; gB, pB, _ = runs['B']
diffs = []
for r, (a_, b_) in enumerate(zip(pA, pB)):
    # the same credit; releasing earlier changes which chunks one VJP groups, i.e. float summation order only
    diffs.append(max(float((x - y).abs().max()) for x, y in zip(a_, b_)))
    assert all(torch.allclose(x, y, rtol=1e-5, atol=1e-7) for x, y in zip(a_, b_)), (r, diffs)
# worker 0 ended after rollout 0 with pending credit from rollout 0's reads: A keeps it at rollout 1, frees at 2
assert gA[1][0] > 0 and gA[2][0] == 0, gA
# B, with the deferred VJP pending at every retain, never released anything after rollout 0
assert gB[2][0] > 0 and gB[3][0] > 0, gB
print(json.dumps(dict(result='PASS', max_param_diff=diffs, graph_frames_A=gA, graph_frames_B=gB)))
