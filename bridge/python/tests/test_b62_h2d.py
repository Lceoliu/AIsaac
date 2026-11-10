"""B62: PinnedH2D (ISAAC_LEARN_ASYNC_H2D) gives the arrays' exact bytes on the device; its pinned buffers are reused
(no allocation per call once large enough) and never rewritten while a copy that reads them is pending (the source
array changed right after a call does not change what arrives)."""
import numpy as np
import pytest
import torch

from train_tok import PinnedH2D


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
@pytest.mark.parametrize('slots', [1, 2])
def test_pinned_h2d_bytes_reuse_and_lifetime(device, slots):
    h = PinnedH2D(torch.device(device), slots=slots)
    rng = np.random.default_rng(0)
    outs, refs = [], []
    for i in range(12):
        a = rng.integers(-2**40, 2**40, (64+i, 17)).astype(np.int64) if i % 2 else rng.standard_normal((50, 3)).astype(np.float32)
        refs.append(a.copy())
        outs.append(h(a))
        a[...] = 0   # (the source changed right after the call: the copy must already own its bytes)
    if device == 'cuda':
        torch.cuda.synchronize()
    for o, r in zip(outs, refs):
        assert o.dtype == torch.from_numpy(r).dtype and tuple(o.shape) == r.shape
        assert o.cpu().numpy().tobytes() == r.tobytes()
    if device == 'cuda':
        allocs = h.allocs
        for _ in range(10):
            h(refs[-1])
        assert h.allocs == allocs   # (no allocation per call once the buffers are large enough)
        assert h.pinned_bytes > 0
