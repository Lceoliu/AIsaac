"""--source-vjp-merge / --source-fused-cast: the same source gradients with fewer autograd calls and nodes (2026-10-27).

The long runs' learner order (dispatch at close, deferred VJP, update every rollout, horizon 6 so that detaches
happen), BF16 sources (amp: the BF16 parameter casts exist), several episodes per source worker so a VJP has
several episode groups. Variants on the same weights and records:
  base     one autograd.backward per episode group, one cast node per parameter (the default)
  merge    --source-vjp-merge 1: one autograd.backward per worker and VJP
  pack     --source-vjp-merge 24: episode groups packed into calls of >= 24 encoded frames
  fused    --source-fused-cast: one cast node per encode
  both
  batch    (source processes) both + --source-encode-batch: one RPC per process and scheduling call
Checks against base, every commit: source gradients within 1e-6 relative (float summation order only; lr 0 so
every interval runs on the same weights), the audit exactly zero, the same source versions and encoded rows;
and the counts (from a profiler around each VJP: threads mode only) of AccumulateGrad and cast nodes drop.
  python tests/test_source_vjp_merge.py [--process] [--sources cuda:1,cuda:2]
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
from isaac_bridge.source_pipeline import SourcePipeline, source_parameter


COUNTS = None   # (the variant being counted: cast backward calls and engine calls, threads mode)


def _count_patches():
    import isaac_bridge.source_pipeline as SP
    for cls in (SP.ImmutableSourceCast, SP.ImmutableSourceCastFused):
        original = cls.backward
        def wrapped(ctx, *g, original=original):
            if COUNTS is not None: COUNTS['cast'] += 1
            return original(ctx, *g)
        cls.backward = staticmethod(wrapped)
    engine = torch.autograd.backward
    def call(*a, **k):
        if COUNTS is not None: COUNTS['calls'] += 1
        return engine(*a, **k)
    torch.autograd.backward = call


def main(processes, sources):
    global COUNTS
    if not processes: _count_patches()
    cuda = sources[0].startswith('cuda')
    if processes: os.environ['ISAAC_SOURCE_PREEMPT_VJP'] = '1'
    torch.set_num_threads(2); torch.manual_seed(71)
    learner = 'cuda:0' if cuda else 'cpu'
    cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
               temporal_heads=4, temporal_window=4, rebuild_memory=True, memory_block=4, memory_tokens=2, source_cache=True)
    base = TokPolicy(**cfg).to(learner).eval()
    rng = np.random.default_rng(71)
    W, R = 6, 4
    seq = TS.make_seq(rng, nw=W, T=8 * R + 1, changes={(0, 6), (1, 9), (2, 13), (4, 3)}, resets=set(), exits=set())
    pend = [M.synth_pending(rng, W, base.heads) for _ in seq]
    runs = {}
    variants = [('base', 0, False, False), ('merge', 1, False, False), ('pack', 24, False, False), ('fused', 0, True, False),
                ('both', 1, True, False)] + ([('batch', 1, True, True)] if processes else [])
    for name, merge, fused, batch in variants:
        model = copy.deepcopy(base)
        pipe = SourcePipeline(model, sources, period=0, micro=8, update_rollouts=1, defer_backward=True, amp=True,
                              processes=processes, horizon=6, dispatch_at_close=True, vjp_merge=merge, fused_cast=fused,
                              encode_batch=batch)
        if not processes and not cuda: pipe.ipc_async = True   # (cpu threads: group by episode as a source process does)
        opt = torch.optim.SGD(model.main_parameters(), lr=0.)
        counts = dict(accumulate=0, cast=0, calls=0); COUNTS = counts
        if not processes:   # (counts in this process's worker threads: AccumulateGrad via post-accumulate hooks,
            # cast nodes via their backward, engine calls via torch.autograd.backward)
            import isaac_bridge.source_pipeline as SP
            backward = pipe._backward
            def counted(worker, *a, **k):
                for replicas in pipe.versions.values():
                    r = replicas[worker]
                    if not getattr(r, '_counted', False):
                        r._counted = True
                        for _, q in r.named_parameters():
                            if q.requires_grad:
                                q.register_post_accumulate_grad_hook(lambda q: counts.__setitem__('accumulate', counts['accumulate'] + 1))
                return backward(worker, *a, **k)
            pipe._backward = counted
        hs = [RawHistory.empty() for _ in range(W)]
        grads, versions = [], []
        for rollout in range(R):
            pipe.poll_finish(opt, 1e9)
            req = []
            for t in range(rollout * 8, rollout * 8 + 8):
                for w in range(W):
                    req.append((hs[w], seq[t][w], pend[t][w])); hs[w] = hs[w].append(seq[t][w], pend[t][w], 4)
            with torch.no_grad(): bank = SourceBatch(model, req, cache=False, pipeline=pipe)
            pipe.retain_histories({h.uid for h in hs}, overlap=True, frontiers={h.uid: h.length - 8 for h in hs})
            pipe.dispatch_finish()
            for step in range(2):
                view = SourceView(bank, np.arange(len(req)))
                out = view.forward(np.arange(len(req)), need_vg=True)
                (out[0].float().square().mean() + out[1].float().square().mean()).backward()
                view.backward(); pipe.fast_step(opt, 1e9)
                before = pipe.version
                pipe.poll_finish(opt, 1e9, wait=(step == 0))
                if pipe.version != before:
                    grads.append({n: q.grad.detach().double().clone() for n, q in model.named_parameters()
                                  if source_parameter(n) and q.grad is not None})
            pipe.finish_rollout(opt, 1e9, overlap=True)
            versions.append(pipe.version)
        pipe.poll_finish(opt, 1e9, wait=True)
        runs[name] = dict(requests=pipe.encode_requests, grads=grads, versions=versions, encoded=pipe.encoded_rows, counts=counts,
                          dropped=float(pipe.horizon_stats()['source_horizon_dropped_grad']))
        pipe.close()
    B = runs['base']
    assert B['grads'], 'no commit with source gradients'
    report = {}
    for name in [v[0] for v in variants[1:]]:
        C = runs[name]
        assert C['versions'] == B['versions'] and C['encoded'] == B['encoded'], (name, C['versions'], B['versions'])
        assert len(C['grads']) == len(B['grads'])
        rel = 0.
        for a, b in zip(C['grads'], B['grads']):
            assert set(a) == set(b), (name, sorted(set(a) ^ set(b)))
            ga = torch.cat([a[n].flatten() for n in sorted(a)]); gb = torch.cat([b[n].flatten() for n in sorted(b)])
            rel = max(rel, float((ga - gb).norm() / gb.norm().clamp_min(1e-30)))
        assert rel <= 1e-6, (name, rel)
        report[name] = dict(source_grad_rel_diff=rel, counts=C['counts'])
    if processes: assert runs['batch']['requests'] < runs['both']['requests'], (runs['batch']['requests'], runs['both']['requests'])
    if processes: report['encode_requests'] = {k: r['requests'] for k, r in runs.items()}
    assert all(r['dropped'] == 0 for r in runs.values()), {k: r['dropped'] for k, r in runs.items()}
    if not processes:
        b, m, f = B['counts'], runs['merge']['counts'], runs['fused']['counts']
        p_ = runs['pack']['counts']
        assert m['accumulate'] < b['accumulate'] and f['cast'] < b['cast'], (b, m, f)
        assert m['calls'] <= p_['calls'] < b['calls'], (b, p_, m)
    print(json.dumps(dict(result='PASS', processes=processes, sources=sources, base_counts=B['counts'], **report)))


if __name__ == '__main__':
    if sys.platform == 'win32': os.environ.setdefault('ISAAC_HOST_TRIM_INTERVAL', '0')
    arg = sys.argv[sys.argv.index('--sources') + 1] if '--sources' in sys.argv else 'cpu,cpu'
    main('--process' in sys.argv or arg.startswith('cuda'), arg.split(','))
    if arg.startswith('cuda'):   # (as test_source_streams: skip the interpreter's CUDA / IPC teardown after PASS)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
