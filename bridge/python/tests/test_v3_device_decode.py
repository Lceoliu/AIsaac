"""Bitwise and operator-count regression against the frozen 7ea9bd9 torch decoder."""
from collections import Counter

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import _v3_device_baseline as old
from isaac_bridge import v3_input as VI
from isaac_bridge.tok_obs import ROW
from isaac_bridge.visible_producer import write_frame_v3
from test_v3_input import NAMES, PC, V3, _devices, _frames_matrix, payload


def assert_bits(got, expected):
    assert got.keys() == expected.keys()
    for k, x in got.items():
        y = expected[k]
        assert (x.dtype, x.shape) == (y.dtype, y.shape), k
        if x.dtype == torch.float32:
            # Numeric equality would miss signed zero and reject matching NaN payloads.
            x, y = x.view(torch.int32), y.view(torch.int32)
        assert torch.equal(x, y), k


def packed(bufs, device):
    f = _frames_matrix(bufs)
    c, _ = VI.check_batch(f[:, :V3.HEADER], [V3.VERSION] * len(bufs), [len(b) for b in bufs])
    return torch.from_numpy(f).to(device), torch.from_numpy(c).to(device)


def wire(p, e, n, s):
    """A contract-valid header with deliberately bad wire values at every data offset."""
    f = np.empty(V3.frame_floats(p, e, n, s), dtype='<f4')
    rng = np.random.default_rng(e + n + s)
    f[V3.HEADER:] = rng.uniform(-4000, 4000, len(f) - V3.HEADER)
    edges = np.array([np.nan, np.inf, -np.inf, -0.0, 0, 1, -0.5, -1.001, -1, 3, 3.001,
                      -2, -3, -1000, -1001, 99.9, 100, 2.0**31, -2.0**31, 2.0**63,
                      np.nextafter(np.float32(V3.CELL), np.float32(0))], dtype='<f4')
    f[V3.HEADER::2] = np.resize(edges, f[V3.HEADER::2].shape)
    f[:V3.HEADER] = [V3.MAGIC, len(f), p, e, n, s, V3.VERSION_WORD, 1]
    bodies = f[V3.FIXED:V3.FIXED + p * V3.PF].reshape(p, V3.PF)
    bodies[:, PC('position_known')] = 0
    bodies[0, [PC('position_known'), PC('body_role')]] = 1
    eat = V3.FIXED + p * V3.BODY
    ent = f[eat:eat + e * V3.EF].reshape(e, V3.EF)
    ent[::3, V3.col(V3.E_COLS, 'kind')] = V3.ENTITY_KIND['laser']
    return f.tobytes()


def test_golden_batch_and_unpadded_frames_match_frozen():
    assert len(NAMES) >= 19
    bufs = [payload(n) for n in NAMES]
    for dev in _devices():
        f, c = packed(bufs, dev)
        assert_bits(VI.decode_frames_device(f, c, 128, 64, 32), old.decode_frames_device(f, c, 128, 64, 32))
        for b in bufs + [wire(1, 0, 0, 0), wire(2, 7, 9, 11)]:
            assert_bits(VI.decode_frame_torch(b, V3.VERSION, device=dev),
                        old.decode_frame_torch(b, V3.VERSION, device=dev))


@pytest.mark.parametrize('counts,buckets', [
    ((0, 0, 0), (16, 8, 8)),
    ((16, 8, 8), (16, 8, 8)),
    ((17, 9, 9), (32, 16, 16)),
    ((33, 17, 17), (64, 32, 32)),
    ((128, 64, 32), (128, 64, 32)),
    ((16, 33, 17), (16, 8, 8)),   # N/S offsets still use the original counts when truncated
])
def test_wire_edges_and_bucket_shapes_match_frozen(counts, buckets):
    for dev in _devices():
        f, c = packed([wire(2, *counts), wire(1, 0, 0, 0), wire(1, 1, 1, 2)], dev)
        assert_bits(VI.decode_frames_device(f, c, *buckets, check=False),
                    old.decode_frames_device(f, c, *buckets, check=False))


def test_lazy_materializes_only_selected_rows_against_frozen():
    bufs = [wire(2, 65, 33, 17), wire(2, 17, 9, 9), wire(1, 0, 0, 0)]
    rows = np.zeros(len(bufs), dtype=ROW)
    for i, buf in enumerate(bufs):
        write_frame_v3(rows[i:i + 1], buf)
        rows['visible_frame'][i, len(buf):] = 255
    for dev in _devices():
        raw = torch.from_numpy(rows.view(np.uint8).reshape(len(rows), -1)).to(dev)
        at = [ROW.fields[n][1] for n in ('visible_frame', 'visible_schema', 'visible_bytes')]
        lazy = VI.lazy_from_raw(raw, *at)
        idx = torch.tensor([2, 1, 2], device=dev)
        selected = {k: v[idx] for k, v in lazy.items()}
        f, c = packed([bufs[i] for i in (2, 1, 2)], dev)
        assert_bits(VI.materialize(selected), old.decode_frames_device(f, c, 32, 16, 16))


def test_no_new_host_constants_when_switching_buckets(monkeypatch):
    monkeypatch.setattr(VI, '_DEVC', {})
    for dev in _devices():
        f, c = packed([wire(1, 0, 0, 0)], dev)
        VI.decode_frames_device(f, c, 16, 8, 8, check=False)

        def no(*a, **k):
            raise AssertionError('host data copied to a warmed-up decoder')

        with monkeypatch.context() as m:
            for name in ('tensor', 'as_tensor', 'from_numpy'):
                m.setattr(torch, name, no)
            VI.decode_frames_device(f, c, 128, 64, 32, check=False)


class CountATen(TorchDispatchMode):
    """Number of dispatched ATen operators after warm-up, including view/shape ops."""

    def __init__(self):
        self.ops = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops[str(func)] += 1
        return func(*args, **(kwargs or {}))


def test_warm_actor_dispatch_count_drops():
    b = payload('forgotten')
    f, c = packed([b] * 8, 'cpu')

    def measure(decode):
        decode(f, c, 16, 64, 32, check=False)
        with CountATen() as monitor:
            decode(f, c, 16, 64, 32, check=False)
        return sum(monitor.ops.values())

    baseline, current = measure(old.decode_frames_device), measure(VI.decode_frames_device)
    print(f'Warmed CPU ATen dispatches (8x E16/N64/S32): {baseline} -> {current}')
    assert current < 0.65 * baseline, (baseline, current)
