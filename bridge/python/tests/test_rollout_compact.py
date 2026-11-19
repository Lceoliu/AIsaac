"""(A-3, 2026-11-19) v3-frame rollouts stored compact (ROWC + FrameArena, B45's format): the learner's bytes
(rows_to_device, with and without a staging buffer and with_raw), the model's input built from them (decode_rows, the
lazy v3 decode), the dumps' records (expand_rows) and a carried open decision equal those of a ROW rollout of the same
records, the frame field's tail past each frame aside (never decoded)."""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import test_v3_policy as VP   # noqa: E402
import train_tok as T   # noqa: E402
from isaac_bridge import tok_policy as TP   # noqa: E402
from isaac_bridge import v3_input as VI   # noqa: E402
from isaac_bridge.tok_obs import ROW, ROWC   # noqa: E402

FA = ROW.fields['visible_frame'][1]


def _fill(compact, steps=4, W=5, seed=1):
    rng = np.random.default_rng(seed)
    roll = T.Rollout(W, 16, 5, compact=compact)
    assert roll.rows.dtype == (ROWC if compact else ROW)
    for t in range(steps):
        names = [VP.NAMES[(t * W + i) % len(VP.NAMES)] for i in range(W)]
        rows = VP._rows_with_frames(names, seed=t)
        # (a stale tail past each frame, as a worker's shared-memory slot holds: the ROW rollout keeps it)
        for i in range(W):
            nb = int(rows['visible_bytes'][i])
            rows['visible_frame'][i, nb:nb + 64] = 0xEE
        a = np.arange(W)
        pend = rng.integers(0, 2, (W, 5))
        T.store(roll, a, rows, pend, pend, np.zeros(W, np.float32), np.zeros(W, np.float32), 0)
    return roll


FIELD = np.zeros(FA, bool)   # (ROW's field bytes before the frame: not the alignment padding, which a structured
for _n in ROW.names:          #  assignment does not copy)
    _o = ROW.fields[_n][1]
    if _o < FA:
        FIELD[_o:_o + ROW.fields[_n][0].itemsize] = True


def _prefix_equal(a, b, nbytes):
    """[n, >= FA] byte rows equal on every field byte before the frame and up to each record's frame end."""
    a, b = np.asarray(a), np.asarray(b)
    for i, k in enumerate(nbytes):
        assert np.array_equal(a[i, :FA][FIELD], b[i, :FA][FIELD]), i
        assert np.array_equal(a[i, FA:FA + k], b[i, FA:FA + k]), i


def test_compact_rollout_gives_the_learner_the_same_records():
    rc, rr = _fill(True), _fill(False)
    flat = np.array([0, 3, 17, 18, 5, 33, 49])
    flat = flat[flat < rr.cap * rr.n]
    flat = flat[(flat % rr.cap) < rr.length[flat // rr.cap]]
    pend = rr.pending.reshape(-1, 5)[flat]
    nb = rr.rows.reshape(-1)['visible_bytes'][flat].astype(np.int64)
    for stage in (None, torch.zeros((64, TP.ROW_BYTES), dtype=torch.uint8)):
        bc, raw_c, _ = TP.rows_to_device(rc.rows, flat, pend, 'cpu', stage=stage, with_raw=True, frames=rc.frames)
        br, raw_r, _ = TP.rows_to_device(rr.rows, flat, pend, 'cpu', with_raw=True)
        assert raw_c.shape == raw_r.shape
        _prefix_equal(raw_c.numpy(), raw_r.numpy(), nb)
        assert not raw_c.numpy()[0, FA + nb[0]:].any() or nb[0] == raw_c.shape[1] - FA   # (zeros past the frame)
        assert set(bc) == set(br)
        for k in bc:
            if k == 'v3.raw':
                continue
            x, y = bc[k], br[k]
            assert (torch.equal(x, y) if torch.is_tensor(x) else x == y), k
        vc, vr = VI.materialize(VI.get_batch(bc)), VI.materialize(VI.get_batch(br))
        assert set(vc) == set(vr) and vc
        for k in vc:
            assert torch.equal(vc[k], vr[k]), k
        m = VP.build()
        with torch.no_grad():
            for x, y in zip(m(bc, need_vg=True), m(br, need_vg=True)):
                if torch.is_tensor(x):
                    assert torch.equal(x, y)


def test_expand_rows_and_carry():
    rc, rr = _fill(True), _fill(False)
    m = int(rr.length.max())
    ec = T.expand_rows(rc, m)
    assert ec.dtype == ROW and ec.shape == rr.rows[:, :m].shape
    nb = rr.rows[:, :m]['visible_bytes'].reshape(-1).astype(np.int64)
    _prefix_equal(ec.reshape(-1).view(np.uint8).reshape(-1, ROW.itemsize),
                  rr.rows[:, :m].reshape(-1).view(np.uint8).reshape(-1, ROW.itemsize), nb)
    # carry: an open decision moves with its frame
    rc2, rr2 = T.Rollout(rc.n, 16, 5, compact=True), T.Rollout(rr.n, 16, 5, compact=False)
    rc2.carry(rc)
    rr2.carry(rr)
    assert rc2.length.sum() == rr2.length.sum() > 0
    i = np.flatnonzero(rr2.length > 0)
    flat = i * rr2.cap
    pend = rr2.pending.reshape(-1, 5)[flat]
    _, raw_c, _ = TP.rows_to_device(rc2.rows, flat, pend, 'cpu', with_raw=True, frames=rc2.frames)
    _, raw_r, _ = TP.rows_to_device(rr2.rows, flat, pend, 'cpu', with_raw=True)
    _prefix_equal(raw_c.numpy(), raw_r.numpy(), rr2.rows.reshape(-1)['visible_bytes'][flat].astype(np.int64))
