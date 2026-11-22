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


def test_recompute_stage_chunks_equal_take():
    """(2026-11-22, train_tok --recompute-stage 1) the recompute pass's records staged once (rows_to_device into a stage
    of stage_cols' bytes per record, with_raw) and each chunk decoded from a slice of the bytes with the host's entity
    counts give take(every, part)'s batch: every field, the lazy v3 tensors materialized and the model's outputs equal,
    for compact and ROW v3 rollouts."""
    for compact in (True, False):
        roll = _fill(compact, steps=6)
        cap = roll.cap
        flat = np.concatenate([w * cap + np.arange(int(roll.length[w])) for w in range(roll.n)]).astype(np.int64)
        pend = roll.pending.reshape(-1, 5)[flat]
        cols = TP.stage_cols(roll.rows, flat)
        stage = torch.zeros(len(flat) * cols, dtype=torch.uint8)   # (1-D, exactly the bytes: the train_tok layout)
        every, raw_e, pend_e = TP.rows_to_device(roll.rows, flat, pend, 'cpu', stage=stage, with_raw=True,
                                                 frames=roll.frames)
        assert raw_e.shape[1] == cols, (compact, raw_e.shape, cols)
        every0 = TP.rows_to_device(roll.rows, flat, pend, 'cpu', frames=roll.frames)   # (as with the switch off)
        ne = roll.rows.reshape(-1)['n_ent'][flat].astype(np.int64)
        wid = every['ent'].shape[1]
        m = VP.build()
        for lo, hi in ((0, 7), (7, 19), (19, len(flat))):
            part = torch.arange(lo, hi)
            want = TP.take(every0, part)
            got = TP.decode_rows(raw_e[lo:hi], pending=pend_e[lo:hi], entities=min(wid, TP.used_entities(ne[lo:hi])))
            assert set(got) == set(want), (set(got) ^ set(want))
            for k in want:
                if k == 'v3.raw':
                    continue
                x, y = got[k], want[k]
                assert (torch.equal(x, y) if torch.is_tensor(x) else x == y), (compact, k)
            vg, vw = VI.materialize(VI.get_batch(got)), VI.materialize(VI.get_batch(want))
            assert set(vg) == set(vw) and vg
            for k in vw:
                assert torch.equal(vg[k], vw[k]), (compact, k)
            with torch.no_grad():
                for x, y in zip(m(got, need_vg=True), m(want, need_vg=True)):
                    if torch.is_tensor(x):
                        assert torch.equal(x, y), compact


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


def test_shared_memory_rollout_is_compact_and_the_learner_view_reads_it():
    """(A-6, 2026-11-19) --learner-process 1: a Rollout over an allocator (SharedArrays in train_tok; a plain one
    here) is compact too -- its arena a fixed byte array among the shared arrays (frames_buf), never in the scalars the
    jobs carry; the learner process's view (rollout_view over the arrays and scalars) rebuilds the arena and gives the
    learner the same bytes as a ROW rollout."""
    made = {}

    def alloc(shape, dtype):
        a = np.zeros(shape, dtype)
        made[id(a)] = a
        return a
    rr = _fill(False)
    rs = T.Rollout(5, 16, 5, compact=True, alloc=alloc)
    assert rs.rows.dtype == ROWC and rs.frames.fixed and rs.frames_buf is rs.frames.buf
    for t in range(4):   # (the records _fill stores, with the ROW rollout's pending)
        names = [VP.NAMES[(t * 5 + i) % len(VP.NAMES)] for i in range(5)]
        rows = VP._rows_with_frames(names, seed=t)
        for i in range(5):
            nb = int(rows['visible_bytes'][i])
            rows['visible_frame'][i, nb:nb + 64] = 0xEE
        T.store(rs, np.arange(5), rows, rr.pending[:, t], rr.pending[:, t], np.zeros(5, np.float32),
                np.zeros(5, np.float32), 0)
    scal = T.rollout_scalars(rs)
    assert 'frames' not in scal and 'frames_buf' not in scal
    arrays = {k: v for k, v in vars(rs).items() if isinstance(v, np.ndarray)}
    assert 'frames_buf' in arrays
    view = T.rollout_view(arrays, scal, None)
    assert view.frames is not None and view.frames.buf is rs.frames_buf
    flat = np.array([0, 3, 17, 18, 33, 49])
    flat = flat[(flat % rr.cap) < rr.length[flat // rr.cap]]
    pend = rr.pending.reshape(-1, 5)[flat]
    _, raw_v, _ = TP.rows_to_device(view.rows, flat, pend, 'cpu', with_raw=True, frames=view.frames)
    _, raw_r, _ = TP.rows_to_device(rr.rows, flat, pend, 'cpu', with_raw=True)
    _prefix_equal(raw_v.numpy(), raw_r.numpy(), rr.rows.reshape(-1)['visible_bytes'][flat].astype(np.int64))


def test_a_fixed_arena_never_grows():
    from isaac_bridge.tok_obs import FrameArena
    a = FrameArena(buf=np.zeros(64, np.uint8))
    rows = VP._rows_with_frames(VP.NAMES[:1])
    try:
        a.append(rows)
    except RuntimeError as e:
        assert 'fixed arena' in str(e)
    else:
        raise AssertionError('a full fixed arena must raise')
