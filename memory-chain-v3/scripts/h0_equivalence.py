"""H=0 equivalence: v3 with --memory-horizon 0 against the unmodified v2 code.

Runs the same deterministic pipeline scenario (two episodes, 4 rollouts of 8
decisions each, update_rollouts=2, two source workers; local threads and,
with --process, spawned source processes) inside one checkout and saves every
Reader output and the final parameters. `compare` checks them bit for bit.

  python -X utf8 h0_equivalence.py run <checkout> <out.pt> [--process]
  python -X utf8 h0_equivalence.py compare <a.pt> <b.pt>
"""
import inspect, os, sys
from pathlib import Path

if __name__ == '__mp_main__' and os.environ.get('H0_TESTS'):
    # Spawned source workers re-import this module: same paths and the test
    # package's import-time stubs (e.g. gymnasium) as the parent.
    sys.path[:0] = os.environ['H0_TESTS'].split(os.pathsep)
    import test_memory_smoke  # noqa: F401


def scenario(checkout, out, process):
    root = Path(checkout).resolve() / 'bridge' / 'python'
    target = str(Path(out).resolve())
    sys.path[:0] = [str(root), str(root / 'tests')]
    os.environ['H0_TESTS'] = os.environ['PYTHONPATH'] = os.pathsep.join([str(root), str(root / 'tests')])
    os.chdir(root)
    if sys.platform == 'win32': os.environ.setdefault('ISAAC_HOST_TRIM_INTERVAL', '0')
    import numpy as np, torch
    import test_memory_smoke as M
    import test_temporal_smoke as TS
    from isaac_bridge.tok_policy import TokPolicy
    from isaac_bridge.rebuild_memory import RawHistory, SourceBatch, SourceView
    from isaac_bridge.source_pipeline import SourcePipeline
    torch.set_num_threads(2); torch.manual_seed(53)
    model = TokPolicy(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32,
                      temporal_layers=2, temporal_heads=4, temporal_window=4, rebuild_memory=True,
                      memory_block=4, memory_tokens=2, source_cache=True).eval()
    v3 = 'horizon' in inspect.signature(SourcePipeline).parameters
    kw = dict(horizon=0) if v3 else {}
    pipe = SourcePipeline(model, ['cpu', 'cpu'], period=0, micro=8, update_rollouts=2, processes=process, **kw)
    opt = torch.optim.Adam(model.main_parameters(), lr=1e-3, eps=1e-5)
    rng = np.random.default_rng(53)
    seq = TS.make_seq(rng, nw=2, T=40, changes={(0, 6), (0, 19), (1, 11), (1, 27)}, resets=set(), exits=set())
    hs = [RawHistory.empty(), RawHistory.empty()]; rollouts = [[] for _ in range(4)]
    for t, rows in enumerate(seq):
        pending = M.synth_pending(rng, 2, model.heads)
        for w in range(2):
            if t >= 8: rollouts[(t - 8) // 8].append((hs[w], rows[w], pending[w]))
            hs[w] = hs[w].append(rows[w], pending[w], 4)
    outputs = []
    for req in rollouts:
        pipe.poll_finish(opt, 1e9, wait=True)
        front = {}
        for h, _, _ in req: front[h.uid] = min(front.get(h.uid, h.length), h.length)
        pipe.retain_histories({h.uid for h, _, _ in req}, **(dict(frontiers=front) if v3 else {}))
        with torch.no_grad(): bank = SourceBatch(model, req, cache=False, pipeline=pipe)
        for epoch in range(2):
            opt.zero_grad(set_to_none=True)
            view = SourceView(bank, np.arange(len(req)))
            out = view.forward(np.arange(len(req)), need_vg=True)
            (out[0].square().mean() + out[1].square().mean() + out[-1].square().mean()).backward()
            outputs.append([x.detach().clone() for x in out])
            view.backward(); pipe.fast_step(opt, 1e9)
        pipe.finish_rollout(opt, 1e9)
    pipe.finish(opt, 1e9)
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    torch.save(dict(outputs=outputs, params=params, version=pipe.version, encoded=pipe.encoded_rows,
                    v3=v3, process=process), target)
    pipe.close()
    print('SAVED', target, 'v3' if v3 else 'v2', 'process' if process else 'local', 'version', pipe.version)


def compare(a, b):
    import torch
    x, y = torch.load(a), torch.load(b)
    assert (x['version'], x['encoded']) == (y['version'], y['encoded']), (x['version'], y['version'], x['encoded'], y['encoded'])
    assert len(x['outputs']) == len(y['outputs'])
    for i, (p, q) in enumerate(zip(x['outputs'], y['outputs'])):
        for j, (u, v) in enumerate(zip(p, q)):
            assert torch.equal(u, v), f'reader output {i}.{j} differs: {(u - v).abs().max()}'
    for n in x['params']:
        assert torch.equal(x['params'][n], y['params'][n]), f'parameter {n} differs'
    print(f"BITWISE_EQUAL outputs={len(x['outputs'])} params={len(x['params'])} source_version={x['version']} "
          f"encoded_rows={x['encoded']} ({'v3' if x['v3'] else 'v2'} vs {'v3' if y['v3'] else 'v2'}, "
          f"{'process' if x['process'] else 'local'})")


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        if __import__('multiprocessing').current_process().name == 'MainProcess':
            scenario(sys.argv[2], sys.argv[3], '--process' in sys.argv[4:])
    else:
        compare(sys.argv[2], sys.argv[3])
