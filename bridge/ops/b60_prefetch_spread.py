"""B60 evidence for the B48 prefetch test's CUDA tolerance: the run-to-run spread of serial runs and of prefetched
runs, in FP32 ulps, on any code version (measurement only).

  python b60_prefetch_spread.py <code_dir> <out.json> [--pairs 20]

<code_dir>: a candidate checkout (its bridge/python and bridge/python/tests are imported, so the same script measures
the code before and after B60). The setting is test_b48_prefetch's (other_call False, CUDA): the rollout fixture
(seed 37), its chunks + teacher records in 3 parts, per part forward_bucketed (generator seeded 9, version 3,
history_mode random, max_rows 4096) + the test's loss + backward + Adam step; one intra-op thread, deterministic
algorithms (warn only), CUBLAS_WORKSPACE_CONFIG as the runner. A reference serial run R, then `pairs` serial runs
and `pairs` prefetched runs (each part's preparation queued on PREFETCH after the previous part's forward), each
compared with R item by item in order (per part: loss, outputs, gradients in named_parameters order): max |diff|
in ulps (np.spacing of the larger magnitude, float32) and the first item that differs."""
import json
import os
import sys
import time

code = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(code, 'bridge', 'python', 'tests'))
sys.path.insert(0, os.path.join(code, 'bridge', 'python'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np  # noqa: E402
import torch  # noqa: E402


def run(prefetch):
    from test_block_packed_learner import _rollout_and_teacher
    from test_block_slot_integration import _losses
    from isaac_bridge import block_learner
    from isaac_bridge.block_training import forward_bucketed, source_rows
    from isaac_bridge.structured_input import from_rows
    block_learner.PACKED = True
    m_, ch_, te_ = _rollout_and_teacher('cuda', seed=37)
    batch = ch_+te_
    idx = np.array_split(np.arange(len(batch)), 3)
    parts = [[batch[i] for i in ix] for ix in idx]
    m_.train()
    torch.manual_seed(0)
    opt = torch.optim.Adam(m_.main_parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(9)
    kw = dict(generator=gen, version=3, history_mode='random', max_rows=4096)
    items = []
    for k, part in enumerate(parts):
        out, ctx = forward_bucketed(m_, part, **kw)
        if prefetch and k+1 < len(parts):
            block_learner.PREFETCH.submit(m_, parts[k+1], **kw)
        rows = source_rows([c.framed() for c in part]); pend = np.concatenate([c.pending for c in part])
        loss = _losses(m_, out, from_rows(rows, 'cuda', pend), 3)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        items.append((f'p{k}:loss', loss.detach().float().cpu().numpy().reshape(-1)))
        for i, o in enumerate(out):
            items.append((f'p{k}:out{i}', o.detach().float().cpu().numpy().reshape(-1)))
        for n, p in m_.named_parameters():
            if p.grad is not None:
                items.append((f'p{k}:grad:{n}', p.grad.detach().float().cpu().numpy().reshape(-1)))
        opt.step()
    torch.cuda.synchronize()
    return items


def compare(a, b):
    """(max ulps over every item, first differing item or None, its ulps)."""
    worst, first, first_u = 0.0, None, 0.0
    for (na, x), (nb, y) in zip(a, b):
        assert na == nb and x.shape == y.shape
        d = np.abs(x.astype(np.float64)-y.astype(np.float64))
        if not d.any():
            continue
        sp = np.spacing(np.maximum(np.abs(x), np.abs(y)).astype(np.float32)).astype(np.float64)
        u = float((d/np.maximum(sp, np.finfo(np.float32).tiny)).max())
        if first is None:
            first, first_u = na, u
        worst = max(worst, u)
    return worst, first, first_u


def main():
    out, pairs = sys.argv[2], (int(sys.argv[sys.argv.index('--pairs')+1]) if '--pairs' in sys.argv else 20)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    t0 = time.perf_counter()
    ref = run(False)
    res = dict(code=code, torch=torch.__version__, pairs=pairs, serial=[], prefetched=[])
    for i in range(pairs):
        for kind, pf in (('serial', False), ('prefetched', True)):
            w, first, fu = compare(ref, run(pf))
            res[kind].append(dict(max_ulps=w, first=first, first_ulps=fu))
            print(json.dumps(dict(i=i, kind=kind, max_ulps=round(w, 2), first=first)), flush=True)
    for kind in ('serial', 'prefetched'):
        u = [round(r['max_ulps']) for r in res[kind]]
        res[kind+'_hist'] = {'0': u.count(0), '1': u.count(1), '2': u.count(2), '>2': sum(x > 2 for x in u)}
        firsts = {}
        for r in res[kind]:
            if r['first']:
                firsts[r['first']] = firsts.get(r['first'], 0)+1
        res[kind+'_first_items'] = firsts
    res['seconds'] = round(time.perf_counter()-t0, 1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('serial', 'prefetched')}), flush=True)
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
