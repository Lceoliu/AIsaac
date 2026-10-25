"""--source-dispatch-at-close changes when a closed interval's VJP runs, nothing else (CPU; threads and processes).

The learner's order with deferred source VJPs (update every rollout, overlap): retain, the rollout's
SourceBatch, dispatch_finish, Reader steps with commit polls, close the interval. Pipeline D dispatches each
closed interval at once and gates its commit to dispatch_finish; pipeline E defers the dispatch. Between
closing an interval and the next rollout, D's VJP is already done (or running), E's has not started; polls
before dispatch_finish commit in neither. Committed at the same minibatch boundary (the first after
dispatch_finish; which boundary a VJP commits at always depended on when it finished), parameters, source
versions and encodes are the same bit for bit.
  python tests/test_source_dispatch_at_close.py [--process]
"""
import copy, json, os, sys
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
    torch.set_num_threads(2); torch.manual_seed(53)
    cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
               temporal_heads=4, temporal_window=4, rebuild_memory=True, memory_block=4, memory_tokens=2, source_cache=True)
    base = TokPolicy(**cfg).eval()
    rng = np.random.default_rng(53)
    W = 3
    seq = TS.make_seq(rng, nw=W, T=40, changes={(0, 6), (1, 9), (2, 13)}, resets=set(), exits=set())
    pend = [M.synth_pending(rng, W, base.heads) for _ in seq]
    runs = {}
    for name, early in (('E', False), ('D', True)):
        model = copy.deepcopy(base)
        pipe = SourcePipeline(model, ['cpu', 'cpu'], period=0, micro=8, update_rollouts=1, defer_backward=True,
                              processes=processes, horizon=6, dispatch_at_close=early)
        opt = torch.optim.Adam(model.main_parameters(), lr=1e-3, eps=1e-5)
        hs = [RawHistory.empty() for _ in range(W)]
        trace = []
        for rollout in range(4):
            pipe.poll_finish(opt, 1e9)                    # (a poll during collection / at the update's start)
            trace.append(('version before sources', pipe.version))
            live = {h.uid for h in hs}
            req = []
            for t in range(rollout * 8, rollout * 8 + 8):
                for w in range(W):
                    req.append((hs[w], seq[t][w], pend[t][w])); hs[w] = hs[w].append(seq[t][w], pend[t][w], 4)
            with torch.no_grad(): bank = SourceBatch(model, req, cache=False, pipeline=pipe)
            pipe.retain_histories(live, overlap=True, frontiers={h.uid: h.length - 8 for h in hs})
            pipe.dispatch_finish()
            for step in range(2):
                view = SourceView(bank, np.arange(len(req)))
                out = view.forward(np.arange(len(req)), need_vg=True)
                (out[0].square().mean() + out[1].square().mean()).backward()
                view.backward(); pipe.fast_step(opt, 1e9)
                # (the commit at the first minibatch boundary in both: the boundary a VJP commits at depends on
                # when it finishes; dispatched at close it is done by then, deferred it is waited for here)
                pipe.poll_finish(opt, 1e9, wait=(step == 0))
            trace.append(('version after minibatches', pipe.version))
            pipe.finish_rollout(opt, 1e9, overlap=True)
            trace.append(('dispatched at close', bool(pipe.pending is not None and pipe.deferred_packets is None)))
        params = [p.detach().clone() for p in model.parameters()]
        runs[name] = (trace, params, pipe.encoded_rows)
        pipe.close()
    (tE, pE, eE), (tD, pD, eD) = runs['E'], runs['D']
    assert all(torch.equal(a, b) for a, b in zip(pE, pD)), 'parameters differ'
    assert eE == eD, (eE, eD)
    versions = lambda tr: [v for k, v in tr if k.startswith('version')]
    assert versions(tE) == versions(tD), (tE, tD)
    assert all(v for k, v in tD if k == 'dispatched at close') and not any(v for k, v in tE if k == 'dispatched at close')
    print(json.dumps(dict(result='PASS', processes=processes, versions=versions(tD), encoded_rows=eD)))


if __name__ == '__main__':
    if sys.platform == 'win32': os.environ.setdefault('ISAAC_HOST_TRIM_INTERVAL', '0')
    main('--process' in sys.argv)
