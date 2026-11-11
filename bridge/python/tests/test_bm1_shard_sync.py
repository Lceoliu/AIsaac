"""BM1 C (isaac_bridge.shard_sync), two ranks on the CPU (gloo): the weighted gradient mean equals one process's
gradient of the concatenated minibatch; after Adam steps the ranks' parameters and Adam states are bit-identical; a
flipped bit is found by the hash check and repaired from rank 0; AuxNorm's merged update equals one update with the
concatenated batch; the agreements (MAX, MIN); a rank whose process ends is noticed by the other's watchdog."""
import os
import socket
import sys
import time

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from isaac_bridge import shard_sync   # noqa: E402
from isaac_bridge.shard_sync import ShardSync, auxnorm_apply, auxnorm_moments   # noqa: E402

if sys.platform == 'win32' and os.environ.get('ISAAC_TEST_DIST_WIN') != '1':
    pytest.skip('torch.distributed gloo: run on Linux (the gate)', allow_module_level=True)


def _port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _data(seed, n):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 6, generator=g), torch.randn(n, 2, generator=g)


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))


def _loss(m, x, y):
    return ((m(x) - y) ** 2).sum(1).mean()   # (a minibatch mean: the trainer's normalisation)


SIZES = (5, 9)


def _np(x):
    if torch.is_tensor(x):
        return x.detach().numpy().copy()
    if isinstance(x, (list, tuple)):
        return type(x)(_np(v) for v in x)
    if isinstance(x, dict):
        return {k: _np(v) for k, v in x.items()}
    return x


def _worker(rank, port, d, q):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    try:
        s = ShardSync(rank, 2, 'cpu', timeout=60, shard_dir=d).init()
        m = _model()
        if rank == 1:   # (different start: broadcast_state makes them equal)
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(1.0)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        s.broadcast_state([m], [opt])
        out = {}
        for step in range(3):
            x, y = _data(100 + 10 * step + rank, SIZES[rank])
            opt.zero_grad(set_to_none=True)
            _loss(m, x, y).backward()
            s.reduce_grads(list(m.parameters()), weight=SIZES[rank])
            if step == 0:
                out['grad0'] = [p.grad.clone() for p in m.parameters()]
            opt.step()
        # a zero-weight step (a rank with fewer minibatches): its gradient does not count
        x, y = _data(999, 4)
        opt.zero_grad(set_to_none=True)
        if rank == 0:
            _loss(m, x, y).backward()
        s.reduce_grads(list(m.parameters()), weight=4 if rank == 0 else 0)
        out['grad_pad'] = [p.grad.clone() for p in m.parameters()]
        opt.step()
        mis, _ = s.check([m], [opt])
        out['mismatch_clean'] = mis
        s.flip = (1, 7, 0)
        s.maybe_flip(7, 0, list(m.parameters()))
        mis, _ = s.check([m], [opt])
        out['mismatch_flip'] = mis
        mis, _ = s.check([m], [opt])
        out['mismatch_after'] = mis
        out['params'] = [p.detach().clone() for p in m.parameters()]
        out['adam'] = [v.clone() for st in opt.state.values() for k, v in sorted(st.items()) if torch.is_tensor(v)]
        # AuxNorm merge
        from isaac_bridge.tok_aux import AuxNorm
        rng = np.random.default_rng(rank)
        st, ret, mev = rng.normal(size=(7 + rank, 3)), rng.normal(size=(7 + rank, 2)), rng.random((7 + rank, 2)) > 0.3
        n = AuxNorm(3, 2)
        auxnorm_apply(n, np.asarray(s.sum(auxnorm_moments(st, ret, mev).tolist())))
        out['aux'] = (n.n, n.mean.copy(), n.m2.copy(), n.rn.copy(), n.rm2.copy())
        out['max'] = s.max_int(3 + rank)
        out['all_true'] = s.all_true(rank == 0)
        q.put((rank, _np(out)))   # (by value: a worker's shared-memory tensors would die with it)
        s.close()
    except Exception as e:   # (reported to the test)
        import traceback
        q.put((rank, dict(error=traceback.format_exc())))


def test_two_ranks_cpu(tmp_path):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    port = _port()
    ps = [ctx.Process(target=_worker, args=(r, port, str(tmp_path), q)) for r in (0, 1)]
    for p in ps:
        p.start()
    res = dict(q.get(timeout=120) for _ in ps)
    for p in ps:
        p.join(30)
    for r in (0, 1):
        assert 'error' not in res[r], res[r].get('error')
    a, b = res[0], res[1]
    # the weighted mean = one process's gradient of the concatenated minibatch (the same start: rank 0's)
    m = _model()
    xs, ys = zip(*(_data(100 + r, SIZES[r]) for r in (0, 1)))
    _loss(m, torch.cat(xs), torch.cat(ys)).backward()
    for g_ref, g0, g1 in zip([p.grad for p in m.parameters()], a['grad0'], b['grad0']):
        assert np.array_equal(g0, g1)
        np.testing.assert_allclose(g0, g_ref.numpy(), rtol=1e-5, atol=1e-6)
    # the zero-weight step: rank 0's gradient alone
    for g0, g1 in zip(a['grad_pad'], b['grad_pad']):
        assert np.array_equal(g0, g1)
    assert a['mismatch_clean'] is False and b['mismatch_clean'] is False
    assert a['mismatch_flip'] is True and b['mismatch_flip'] is True
    assert a['mismatch_after'] is False and b['mismatch_after'] is False
    for x, y in zip(a['params'] + a['adam'], b['params'] + b['adam']):
        assert np.array_equal(x, y)
    # AuxNorm: the merged update = one update with the two batches concatenated
    from isaac_bridge.tok_aux import AuxNorm
    ref = AuxNorm(3, 2)
    parts = []
    for r in (0, 1):
        rng = np.random.default_rng(r)
        parts.append((rng.normal(size=(7 + r, 3)), rng.normal(size=(7 + r, 2)), rng.random((7 + r, 2)) > 0.3))
    ref.update(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
               np.concatenate([p[2] for p in parts]))
    for got in (a['aux'], b['aux']):
        np.testing.assert_allclose(got[0], ref.n)
        for x, y in zip(got[1:], (ref.mean, ref.m2, ref.rn, ref.rm2)):
            np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-12)
    assert a['max'] == b['max'] == 4
    assert a['all_true'] is False and b['all_true'] is False


def _dies(rank, port, d, q):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    s = ShardSync(rank, 2, 'cpu', timeout=60, shard_dir=d).init()
    if rank == 1:
        os._exit(3)   # (an abnormal end)
    t0 = time.time()
    while s.peer_lost is None and time.time() - t0 < 20:
        time.sleep(0.2)
    q.put((rank, dict(lost=s.peer_lost is not None, after=time.time() - t0)))


def test_peer_end_noticed(tmp_path):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    port = _port()
    ps = [ctx.Process(target=_dies, args=(r, port, str(tmp_path), q)) for r in (0, 1)]
    for p in ps:
        p.start()
    rank, res = q.get(timeout=120)
    for p in ps:
        p.join(30)
    assert rank == 0 and res['lost'] and res['after'] < 10, res
