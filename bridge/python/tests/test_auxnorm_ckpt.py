"""AuxNorm's checkpoint state loads with torch.load's default (weights_only) after a BM1 C shard reduce too: r1b /
r2b (2026-11-12) failed to resume a K=2 checkpoint because the reduced batch count made AuxNorm.n a numpy scalar."""
import io

import numpy as np
import torch

from isaac_bridge.tok_aux import AuxNorm


def _roundtrip(norm):
    buf = io.BytesIO()
    torch.save({'norm': norm.state()}, buf)
    buf.seek(0)
    return torch.load(buf, map_location='cpu')['norm']   # (the default: weights_only)


def test_state_loads_weights_only_after_a_shard_reduce():
    rng = np.random.default_rng(0)
    for reduce in (None, lambda m: m):   # (one rank's sums as the global batch: shard_sync's path, K=1 world)
        norm = AuxNorm(5, 3)
        norm.reduce = reduce
        for _ in range(3):
            norm.update(rng.normal(size=(64, 5)).astype(np.float32), rng.normal(size=(8, 8, 3)),
                        rng.random((8, 8, 3)) > 0.3)
        d = _roundtrip(norm)
        assert type(d['n']) is float and d['n'] == 192.0
        back = AuxNorm(5, 3)
        back.load(d)
        assert np.allclose(back.state_std(), norm.state_std()) and np.allclose(back.ret_std(), norm.ret_std())
