"""B45 compact records: ROWC + FrameArena hold exactly the bytes ROW held (head fields at the same offsets, each
frame's visible_bytes), the compact decoders read the same values as the inline ones, and a learner part over compact
rollout chunks gives the outputs of the same chunks with their frames inline."""
import numpy as np
import torch

from test_visible_input import synth_rows
from isaac_bridge.tok_obs import ROW, ROWC, ROW_FRAME_AT, FrameArena, compact_rows, compact_copy, expand_rows
from isaac_bridge.structured_input import decode_frame_compact, decode_frames_compact
from isaac_bridge.block_training import Chunk, forward_bucketed, frame_rows


def test_round_trip_keeps_every_byte():
    rows = synth_rows(np.random.default_rng(1), 9)
    arena = FrameArena(64)   # (grows)
    c = compact_rows(rows, arena)
    assert c.dtype == ROWC and ROWC.itemsize < 21000 and arena.used < 9 * 40000
    back = expand_rows(c, arena.buf)
    assert back.tobytes() == rows.tobytes()
    assert (c['visible_off'] % 8 == 0).all()
    one, buf = compact_copy(rows[3:4])
    assert expand_rows(one, buf).tobytes() == rows[3:4].tobytes()
    np.testing.assert_array_equal(c.view(np.uint8).reshape(9, -1)[:, :ROW_FRAME_AT],
                                  rows.view(np.uint8).reshape(9, -1)[:, :ROW_FRAME_AT])


def test_compact_decode_equals_inline_decode():
    rng = np.random.default_rng(2)
    parts = [synth_rows(rng, 5), synth_rows(rng, 3)]
    arenas = [FrameArena(), FrameArena()]
    compact = [(compact_rows(r, a), a.buf) for r, a in zip(parts, arenas)]
    for packed in (False, True):
        for a, b in zip(decode_frames_compact(parts, packed), decode_frames_compact(compact, packed)):
            for k in a:
                np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)
    fr = frame_rows(*compact[0])
    for k, v in decode_frame_compact(parts[0]).items():
        np.testing.assert_array_equal(v, decode_frame_compact(fr)[k], err_msg=k)


def test_rollout_stores_compact_records_and_learner_reads_them():
    from test_block_slot_integration import _collect_rollout
    from train_tok import gae
    m, ref, new, old, rolls = _collect_rollout('cpu')
    roll = rolls[0]
    assert roll.rows.dtype == ROWC and roll.frames is not None
    n = roll.total()
    assert roll.frames.used <= n * 40000   # frames at their real size, not 128 KiB each
    _, _, mask = gae(roll, .99, .95)
    plan = roll.block.plan(roll, mask)
    assert all(c.frames is not None for c in plan['chunks'])
    inline = [Chunk(c.framed(), c.pending, c.snapshot) for c in plan['chunks']]
    m.eval()
    with torch.no_grad():
        a, _ = forward_bucketed(m, plan['chunks'], train_history=False)
        b, _ = forward_bucketed(m, inline, train_history=False)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
