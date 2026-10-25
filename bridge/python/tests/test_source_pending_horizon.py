"""A horizon detach while a deferred VJP is pending (CPU; threads and spawned source processes).

Rollout 0 reads episode `a`; its interval closes deferred (its packets hold a's chunks). Rollout 1
reads episode `b` only. retain_histories then runs with frontiers so far ahead that every chunk of
both episodes is out of the horizon: b's chunks (no pending credit) detach, a's (in the pending
packets) stay graphs; the deferred VJP then dispatches and commits without error (2026-10-25: a
name clash let the detach take the pending chunks too, and the VJP met a released output).
  python tests/test_source_pending_horizon.py [--process]
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch, SourceView
from isaac_bridge.source_pipeline import SourcePipeline


def main(processes):
    torch.set_num_threads(2); torch.manual_seed(41)
    cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
               temporal_heads=4, temporal_window=4, rebuild_memory=True, memory_block=4, memory_tokens=2, source_cache=True)
    model = TokPolicy(**cfg).eval()
    rng = np.random.default_rng(41)
    seq = TS.make_seq(rng, nw=2, T=24, changes={(0, 5), (1, 7)}, resets=set(), exits=set())
    pend = [M.synth_pending(rng, 2, model.heads) for _ in seq]
    pipe = SourcePipeline(model, ['cpu', 'cpu'], period=0, micro=8, update_rollouts=1, defer_backward=True,
                          processes=processes, horizon=2)
    opt = torch.optim.Adam(model.main_parameters(), lr=1e-3, eps=1e-5)
    hs = [RawHistory.empty(), RawHistory.empty()]

    def rollout(workers, t0):
        req = []
        for t in range(t0, t0 + 8):
            for w in workers:
                req.append((hs[w], seq[t][w], pend[t][w])); hs[w] = hs[w].append(seq[t][w], pend[t][w], 4)
        with torch.no_grad(): bank = SourceBatch(model, req, cache=False, pipeline=pipe)
        return bank, req

    def learn(bank, req):
        view = SourceView(bank, np.arange(len(req)))
        out = view.forward(np.arange(len(req)), need_vg=True)
        (out[0].square().mean() + out[1].square().mean()).backward()
        view.backward(); pipe.fast_step(opt, 1e9)

    # both episodes encoded; rollout 0's credit for episode a only
    bank, req = rollout([0], 0); rollout([1], 0)
    learn(bank, req)
    pipe.finish_rollout(opt, 1e9, overlap=True)          # deferred: pending holds a's chunks
    assert pipe.pending is not None and pipe.deferred_packets
    a, b = hs[0].uid, hs[1].uid
    far = {a: 10 ** 6, b: 10 ** 6}
    pipe.retain_histories({a, b}, frontiers=far)
    for f in pipe.control_futures: f.result()
    state = {u: [bool(c.get('detached')) for c in pipe.chunks if c is not None and c.get('uid') == u] for u in (a, b)}
    assert state[a] and not any(state[a]), ('pending chunks were detached', state)
    assert state[b] and all(state[b]), ('chunks without pending credit stayed graphs', state)
    bank, req = rollout([1], 8)
    pipe.dispatch_finish()
    learn(bank, req)
    pipe.poll_finish(opt, 1e9, wait=True)                # the deferred VJP over a's graphs
    assert pipe.pending is None
    pipe.retain_histories({a, b}, frontiers=far)          # nothing pending: a's chunks detach now
    for f in pipe.control_futures: f.result()
    assert all(c.get('detached') for c in pipe.chunks if c is not None and c.get('uid') == a)
    pipe.close()
    print(json.dumps(dict(result='PASS', processes=processes, detached_while_pending=dict(a=state[a], b=state[b]))))


if __name__ == '__main__':
    if sys.platform == 'win32': os.environ.setdefault('ISAAC_HOST_TRIM_INTERVAL', '0')
    main('--process' in sys.argv)
