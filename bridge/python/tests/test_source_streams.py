"""--source-streams 2: a VJP concurrent with encodes and Writers gives the serial schedule's result (2026-10-26).

Source processes with the separate VJP connection (ISAAC_SOURCE_PREEMPT_VJP=1), the long runs' learner order
(dispatch at close, deferred VJP, update every rollout, horizon 6 so that detaches and retires arrive while a
VJP runs): per rollout, the rollout's SourceBatch is encoded while the previous interval's VJP runs, then
retain (detach / retire), dispatch_finish, Reader steps with commit polls, close the interval.
  S   streams 0: one stream; the VJP serves encodes only between its episode groups (as before)
  C1  streams 2, schedule shuffled (ISAAC_SOURCE_STREAM_SHUFFLE: random host sleeps and, on CUDA, random
  C2  device sleeps on the issuing stream before every encode / Writer and between the VJP's episode groups)
Checks: parameters after every rollout equal to S's (bitwise; on CUDA within 1e-6 relative), source versions
and encoded rows equal, the audit (dropped gradient) exactly zero in all, and in C the VJP ran concurrently
with encodes (a worker's encode interval inside one of its VJP intervals, the VJP event marked concurrent).
(The extra workspace per card is measured in the long runs: memory breakdown and nvidia-smi peaks.)
  python tests/test_source_streams.py [--sources cpu,cpu | cuda:1,cuda:2]
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


def main(sources):
    os.environ['ISAAC_SOURCE_PREEMPT_VJP'] = '1'
    cuda = sources[0].startswith('cuda')
    torch.set_num_threads(2); torch.manual_seed(59)
    learner = 'cuda:0' if cuda else 'cpu'
    cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
               temporal_heads=4, temporal_window=4, rebuild_memory=True, memory_block=4, memory_tokens=2, source_cache=True)
    base = TokPolicy(**cfg).to(learner).eval()
    rng = np.random.default_rng(59)
    W, R = 3, 5
    seq = TS.make_seq(rng, nw=W, T=8 * R + 1, changes={(0, 6), (1, 9), (2, 13)}, resets=set(), exits=set())
    pend = [M.synth_pending(rng, W, base.heads) for _ in seq]
    runs = {}
    variants = [('S', 0, None), ('C1', 2, '1'), ('C2', 2, '2')] + ([('S2', 0, None)] if cuda else [])
    for name, streams, shuffle in variants:
        if shuffle: os.environ['ISAAC_SOURCE_STREAM_SHUFFLE'] = shuffle
        else: os.environ.pop('ISAAC_SOURCE_STREAM_SHUFFLE', None)
        print('RUN', name, flush=True)
        model = copy.deepcopy(base)
        pipe = SourcePipeline(model, sources, period=0, micro=8, update_rollouts=1, defer_backward=True,
                              processes=True, horizon=6, dispatch_at_close=True, streams=streams)
        opt = torch.optim.Adam(model.main_parameters(), lr=1e-3, eps=1e-5)
        hs = [RawHistory.empty() for _ in range(W)]
        params, versions = [], []
        for rollout in range(R):
            pipe.poll_finish(opt, 1e9)
            req = []
            for t in range(rollout * 8, rollout * 8 + 8):
                for w in range(W):
                    req.append((hs[w], seq[t][w], pend[t][w])); hs[w] = hs[w].append(seq[t][w], pend[t][w], 4)
            with torch.no_grad(): bank = SourceBatch(model, req, cache=False, pipeline=pipe)   # (during the VJP)
            pipe.retain_histories({h.uid for h in hs}, overlap=True, frontiers={h.uid: h.length - 8 for h in hs})
            pipe.dispatch_finish()
            for step in range(2):
                view = SourceView(bank, np.arange(len(req)))
                out = view.forward(np.arange(len(req)), need_vg=True)
                (out[0].float().square().mean() + out[1].float().square().mean()).backward()
                view.backward(); pipe.fast_step(opt, 1e9)
                pipe.poll_finish(opt, 1e9, wait=(step == 0))
            pipe.finish_rollout(opt, 1e9, overlap=True)
            params.append([p.detach().double().clone() for p in model.parameters()])
            versions.append(pipe.version)
        pipe.poll_finish(opt, 1e9, wait=True)
        params.append([p.detach().double().clone() for p in model.parameters()])
        ev = pipe.events
        vjps = [e for e in ev if e['kind'] == 'source_backward']
        overlap = sum(1 for e in ev if e['kind'] in ('encode', 'writer') for v in vjps
                      if v['device'] == e['device'] and v['start'] < e['start'] and e['end'] < v['end'])
        runs[name] = dict(params=params, versions=versions, encoded=pipe.encoded_rows, overlap=overlap,
                          concurrent=any(v.get('concurrent') for v in vjps), vjps=len(vjps),
                          dropped=float(pipe.horizon_stats()['source_horizon_dropped_grad']))
        pipe.close()
    os.environ.pop('ISAAC_SOURCE_STREAM_SHUFFLE', None)
    S = runs['S']
    report = {}
    def rel(P, Q):
        return max(float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))
                   for pa, pb in zip(P['params'], Q['params']) for a, b in zip(pa, pb))
    # (CUDA: the serial schedule run twice; atomics in index_add_ / scatter make it not bitwise reproducible)
    noise = rel(runs['S2'], S) if cuda else 0.
    for name in ('C1', 'C2'):
        C = runs[name]
        assert C['versions'] == S['versions'] and C['encoded'] == S['encoded'], (name, C['versions'], S['versions'])
        diff = max(float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))
                   for pa, pb in zip(C['params'], S['params']) for a, b in zip(pa, pb))
        bitwise = all(torch.equal(a, b) for pa, pb in zip(C['params'], S['params']) for a, b in zip(pa, pb))
        print('DIFF', name, diff, 'serial-vs-serial', noise, flush=True)
        assert bitwise if not cuda else diff <= max(1e-6, 2 * noise), (name, diff, noise)
        assert C['concurrent'] and C['overlap'] > 0, (name, C['concurrent'], C['overlap'])
        report[name] = dict(bitwise=bitwise, max_rel_param_diff=diff, encodes_inside_vjp=C['overlap'], vjps=C['vjps'])
    assert all(r['dropped'] == 0 for r in runs.values()), {k: r['dropped'] for k, r in runs.items()}
    assert not S['concurrent']
    print(json.dumps(dict(result='PASS', sources=sources, serial_encodes_inside_vjp=S['overlap'],
                          serial_repeat_rel_diff=noise, **report)))


if __name__ == '__main__':
    if sys.platform == 'win32': os.environ.setdefault('ISAAC_HOST_TRIM_INTERVAL', '0')
    arg = sys.argv[sys.argv.index('--sources') + 1] if '--sources' in sys.argv else 'cpu,cpu'
    main(arg.split(','))
