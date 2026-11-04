"""CUDA seconds of the privileged critic's parts on a part of 4,096 records (a measurement aid, host 2).
usage (bridge's python dir): PYTHONPATH=. python tests/priv_time_bench.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_memory_smoke as M   # noqa: E402,F401

import numpy as np   # noqa: E402
import torch   # noqa: E402

from isaac_bridge import tok_priv as P   # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = torch.device('cuda')
n, D = 4096, int(sys.argv[1]) if len(sys.argv) > 1 else 64
roll = type('R', (), {})()
roll.cap = 512
roll.priv = np.zeros((8, 512), P.PRIV)
g = np.random.default_rng(1)
pv = roll.priv
pv['map'] = g.random(pv['map'].shape) < 0.03
pv['map_ver'] = np.arange(512)[None] // (4096 // D // 8 + 1)
pv['hp'] = g.random(pv['hp'].shape)
pv['scal'] = g.random(pv['scal'].shape)
pv['items'] = g.integers(0, 1024, pv['items'].shape) * (g.random(pv['items'].shape) < 0.01)
flat = np.arange(n, dtype=np.int64)
t0 = time.perf_counter()
b = P.PrivBatch(roll, flat, dev)
torch.cuda.synchronize()
print(f'PrivBatch {1000 * (time.perf_counter() - t0):.1f} ms, {b.maps.shape[0]} distinct maps')
c = P.make_critic(192).to(dev)
idx = torch.randperm(n, device=dev)
zr, zg = torch.randn(n, 192, device=dev), torch.randn(n, 192, device=dev)


def timed(name, fn, reps=30):
    out = None
    t = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    print(f'{name:28s} {1000 * float(np.median(t)):8.2f} ms   first {1000 * t[0]:8.2f}')
    return out


with torch.inference_mode():
    u, inv = timed('unique', lambda: torch.unique(b.ids.index_select(0, idx), return_inverse=True))
    pad = u
    if pad.shape[0] % 64:
        pad = torch.cat([pad, pad[:1].expand(64 - pad.shape[0] % 64)])
    print('distinct', u.shape[0], 'padded', pad.shape[0])
    maps = timed('maps gather .float', lambda: b.maps.index_select(0, pad).float())
    m0 = timed('map cnn', lambda: c.map(maps))
    m = m0.index_select(0, inv)
    it = timed('item embedding sum', lambda: c.item(b.items.index_select(0, idx)).sum(1))
    fe = timed('features gather cat', lambda: torch.cat([b.hp.index_select(0, idx), b.scal.index_select(0, idx), it,
                                                         b.missing.index_select(0, idx)[:, None]], -1))
    r = timed('rest mlp', lambda: c.rest(fe))
    p = timed('join', lambda: c.join(torch.cat([m, r], -1)))
    timed('value heads', lambda: c(zr, zg, p))
    timed('encode all', lambda: c.encode(b, idx))
    timed('encode + heads', lambda: c(zr, zg, c.encode(b, idx)))


def fb():
    v, vg = c(zr, zg, c.encode(b, idx))
    (v.sum() + vg.sum()).backward()
    c.zero_grad(set_to_none=True)


timed('forward + backward', fb)
print('reserved GiB', round(torch.cuda.max_memory_reserved() / 2 ** 30, 2))
