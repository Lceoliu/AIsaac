"""(A-4, 2026-11-19) take_prefix: a call's records taken up to their frame's end into a reused buffer -- every field
byte and each frame equal src[slots] whatever the buffer held before (the frame field's tail past a frame is not
copied: nothing reads it); a record with an impossible visible_bytes is copied whole. And GraphActor's v3 call (the
staging copy up to the call's longest frame) gives the same answers whatever the frame field's tail holds (CUDA)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import test_v3_policy as VP   # noqa: E402
from isaac_bridge import tok_policy as TP   # noqa: E402
from isaac_bridge.tok_obs import ROW, ROW_FRAME_AT, take_prefix   # noqa: E402

FIELD = np.zeros(ROW_FRAME_AT, bool)
for _n in ROW.names:
    _o = ROW.fields[_n][1]
    if _o < ROW_FRAME_AT:
        FIELD[_o:_o + ROW.fields[_n][0].itemsize] = True


def _src(k=8):
    src = VP._rows_with_frames([VP.NAMES[i % len(VP.NAMES)] for i in range(k)], seed=3)
    for i in range(k):   # (a worker's slot holds an earlier, longer frame's bytes past this one)
        nb = int(src['visible_bytes'][i])
        src['visible_frame'][i, nb:nb + 4000] = 0x5A
    return src


def test_take_prefix_equals_fancy_indexing_on_every_read_byte():
    src = _src()
    out = np.frombuffer(np.random.default_rng(0).integers(0, 256, 8 * ROW.itemsize, dtype=np.uint8).tobytes(),
                        ROW).copy()
    for slots in (np.array([3, 0, 7]), np.arange(8), np.array([5]), np.array([6, 6 - 1, 1, 2, 4])):
        got, want = take_prefix(src, slots, out), src[slots]
        g = got.view(np.uint8).reshape(len(slots), -1)
        w = want.view(np.uint8).reshape(len(slots), -1)
        for i in range(len(slots)):
            k = ROW_FRAME_AT + ((int(want['visible_bytes'][i]) + 3) & ~3)
            assert np.array_equal(g[i, :ROW_FRAME_AT][FIELD], w[i, :ROW_FRAME_AT][FIELD])
            assert np.array_equal(g[i, ROW_FRAME_AT:k], w[i, ROW_FRAME_AT:k])
        for name in ROW.names:   # (and as structured records)
            if name != 'visible_frame':
                assert np.array_equal(got[name], want[name]), name
    bad = src.copy()
    bad['visible_bytes'][2] = -4
    got = take_prefix(bad, np.array([2]), out)
    g, w = got.view(np.uint8).reshape(-1), bad.view(np.uint8).reshape(len(bad), -1)[2]
    assert np.array_equal(g, w), (np.flatnonzero(g != w)[:8], len(g), len(w))   # (whole)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA: GraphActor')
@pytest.mark.parametrize('fold', [False, True])
def test_graph_actor_ignores_the_frame_fields_tail(fold):
    cuda = torch.device('cuda')
    m = VP.build().to(cuda)
    outs = []
    for tail in (0x00, 0xFF):
        st = TP.TemporalState(m, 6, cuda, torch.float32)
        actor = TP.GraphActor(m, 6, 32, stream=torch.cuda.Stream(), crn=False, temporal=st, fold=fold)
        rows = VP._rows_with_frames(VP.NAMES[:4], seed=1)
        rows['first'] = 1
        for i in range(4):
            rows['visible_frame'][i, int(rows['visible_bytes'][i]):] = tail
        pend = np.zeros((4, len(m.heads)), np.int64)
        ukeys = (np.arange(4, dtype=np.int64) + 11, np.zeros(4, np.int64))
        a, lp, v = actor.act(rows, pend, np.array([0, 1, 2, 3]), None, None, ukeys=ukeys)
        torch.cuda.synchronize()
        outs.append((a, lp, v))
    for x, y in zip(*outs):
        assert np.array_equal(x, y)
