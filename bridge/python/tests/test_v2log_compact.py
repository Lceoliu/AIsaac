"""V2Log.keep_row / put_row (2026-11-13): a block-mode worker keeps each record of its episode for the teacher's
replay_value as its bytes up to the end of its frame (ROW_FRAME_AT + visible_bytes), not the whole 151,880-byte ROW --
main1's 3600 s episodes held ~4 GB in one worker. The worker's ROW slot is reused (slots[t & 1]): the frame budget past
visible_bytes holds stale bytes of earlier records, which the old full copy carried along and the readers never read.

Checked here, with the real LanePolicy.replay_value and a stand-in actor that reads the lane slot as the actor does
(every field before the frame, the frame's first visible_bytes): over episodes ending normally, by death, by the time cap
(done 3 at t >= the cap), by the stall rule (done 3 before it), as a branch's end, with small / typical / multi-body /
128-KiB frames -- the compact log and the old full copy give the actor the same bytes and the same values, step by step
(so the same teacher-v2 records: their values / actions are what the actor answers, every other field comes from the
log's other lists, unchanged), and the decoded frames (structured_input.decode_frame_host) are equal; the slot's bytes
past visible_bytes differ (the old copy's stale tail against the slot's own) and nothing reads them. So may the ROW's
alignment padding between fields (numpy's structured copy / assignment leaves it; the raw-byte copy keeps the slot's):
every differing byte before the frame is checked to lie in no field. A kept record costs ROW_FRAME_AT + visible_bytes
bytes."""
import zlib

import numpy as np
import pytest

from isaac_bridge.structured_input import decode_frame_host, pack_frame, write_frame
from isaac_bridge.tok_obs import DONE_BRANCH, ROW, ROW_FRAME_AT
from isaac_bridge.tok_teacher2 import LanePolicy, V2Log
from test_visible_input import components

FRAME_CAP = ROW.itemsize - ROW_FRAME_AT - 4   # (131072: the budget; 4 bytes of padding after it)
CAP_T = 40
FIELDS = [n for n in ROW.names if n != 'visible_frame']
IN_FIELD = np.zeros(ROW_FRAME_AT, bool)   # (the bytes before the frame that belong to a field: the rest is padding)
for _n in FIELDS:
    _dt, _off = ROW.fields[_n][:2]
    IN_FIELD[_off:_off + _dt.itemsize] = True


def actor_view(slot):
    """what the actor reads of a ROW slot: every field before the frame, the frame's first visible_bytes"""
    n = int(slot['visible_bytes'][0])
    return b''.join(slot[name].tobytes() for name in FIELDS) + slot['visible_frame'][0, :n].tobytes()


def valid_frame(seed, bodies=1, entities=3):
    rows = np.zeros(1, ROW)
    write_frame(rows, pack_frame(*components(bodies=bodies, entities=entities, seed=seed)))
    n = int(rows['visible_bytes'][0])
    return rows['visible_schema'][0], rows['visible_frame'][0, :n].copy()


def episodes(rng):
    """(length, end done code, frame kind) per episode: normal ends, death, the cap, the stall rule, a branch end"""
    return [(12, 1, 'valid'), (9, 2, 'multi'), (CAP_T + 1, 3, 'valid'), (7, 3, 'large'), (10, DONE_BRANCH, 'valid'),
            (6, 0, 'tiny')]


class FakeConn:
    """The lane's pipe and a stand-in actor: on each request it reads the lane slot as the actor does (the bytes before
    the frame and the frame's first visible_bytes), answers a value from their CRC and echoes the request's number."""
    def __init__(self, slot, io, out):
        self.slot, self.io, self.out = slot, io, out
        self.seen, self.raw = [], []

    def send_bytes(self, b):
        read = actor_view(self.slot)
        self.seen.append(read)
        self.raw.append(self.slot.view(np.uint8).reshape(-1)[:ROW_FRAME_AT].copy())
        v = float(zlib.crc32(read))
        self.out[0, 0], self.out[0, 1], self.out[0, 2], self.out[0, 3] = v, -v, 1, v + 0.5
        self.out[0, 4] = self.io[0, 10]
        self.out[0, 5] = 0

    def poll(self, timeout=0):
        return True

    def recv_bytes(self):
        return b'1'


def build_logs(seed):
    """A worker's episode stream through one reused ROW slot pair (stale bytes everywhere at the start): the old full
    copies and the compact log of the same records"""
    rng = np.random.default_rng(seed)
    slots_all = np.zeros(2, ROW)
    slots_all.view(np.uint8)[:] = rng.integers(0, 256, slots_all.nbytes, dtype=np.uint8)   # (stale slot bytes)
    slots = (slots_all[0:1], slots_all[1:2])
    full, compact, kinds = [], V2Log(), []
    t_all = 0
    for e, (length, end, kind) in enumerate(episodes(rng)):
        for t in range(length):
            row = slots[t_all & 1]
            head = row.view(np.uint8).reshape(-1)
            head[:ROW_FRAME_AT] = rng.integers(0, 256, ROW_FRAME_AT, dtype=np.uint8)   # (every field before the frame)
            if kind == 'large':
                schema, fr = 3, rng.integers(0, 256, FRAME_CAP, dtype=np.uint8)
            elif kind == 'tiny':
                schema, fr = 3, rng.integers(0, 256, int(rng.integers(0, 64)), dtype=np.uint8)
            else:
                schema, fr = valid_frame(1000 * seed + t_all, bodies=3 if kind == 'multi' else 1,
                                         entities=int(rng.integers(1, 40)))
            row['visible_schema'], row['visible_bytes'] = schema, len(fr)
            row['visible_frame'][0, :len(fr)] = fr   # (the rest of the budget: the slot's earlier bytes)
            row['t'], row['episode'], row['first'] = t, e + 1, int(t == 0)
            row['done'] = end if t == length - 1 else 0
            full.append(row.copy())
            compact.keep_row(row)
            compact.pending.append(tuple(int(x) for x in rng.integers(0, 5, 5)))
            kinds.append(kind)
            t_all += 1
    return full, compact, kinds


def lane(slot):
    io = np.zeros((1, 12), np.int64)
    out = np.zeros((1, 6), np.float64)
    conn = FakeConn(slot, io, out)
    pol = LanePolicy(conn, io, out, 0, lambda: False, block_mode=np.zeros(1, np.int64), block_keep=np.zeros((1, 4)))
    return pol, conn


class FullLog:
    """the old V2Log rows (whole ROW copies), replayed the old way (row_buffer[:] = rows[step])"""
    def __init__(self, rows, pending):
        self.rows, self.pending = rows, pending

    def put_row(self, step, row_buffer):
        row_buffer[:] = self.rows[step]


@pytest.mark.parametrize('seed', [1, 2])
def test_compact_log_replays_the_same(seed):
    full, compact, kinds = build_logs(seed)
    K = len(full)
    assert len(compact.rows) == K
    for i in range(K):   # a kept record: its bytes up to the end of its frame; its fields as the old full copy's
        assert len(compact.rows[i]) == ROW_FRAME_AT + int(full[i]['visible_bytes'][0])
        a = full[i].view(np.uint8).reshape(-1)[:ROW_FRAME_AT]
        assert not ((a != compact.rows[i][:ROW_FRAME_AT]) & IN_FIELD).any()
    rng = np.random.default_rng(100 + seed)
    old_slot, new_slot = np.zeros(1, ROW), np.zeros(1, ROW)
    old_slot.view(np.uint8)[:] = rng.integers(0, 256, old_slot.nbytes, dtype=np.uint8)
    new_slot.view(np.uint8)[:] = rng.integers(0, 256, new_slot.nbytes, dtype=np.uint8)
    po, co = lane(old_slot)
    pn, cn = lane(new_slot)
    old_log = FullLog(full, compact.pending)
    tails_differ = 0
    for start, end in ((0, K - 1), (5, 20), (K - 9, K - 1), (13, 13)):
        vo = po.replay_value(old_log, start, end, None, old_slot, global_value=True)
        vn = pn.replay_value(compact, start, end, None, new_slot, global_value=True)
        assert vo == vn
        vo = po.replay_value(old_log, start, end, None, old_slot, global_value=False)
        vn = pn.replay_value(compact, start, end, None, new_slot, global_value=False)
        assert vo == vn
        n = int(new_slot['visible_bytes'][0])
        tails_differ += not np.array_equal(old_slot['visible_frame'][0, n:], new_slot['visible_frame'][0, n:])
    assert len(co.seen) == len(cn.seen) > K
    for a, b in zip(co.seen, cn.seen):   # every request: the same bytes read by the actor
        assert a == b
    pad = 0
    for a, b in zip(co.raw, cn.raw):   # the bytes before the frame that differ: alignment padding only
        d = a != b
        assert not (d & IN_FIELD).any()
        pad += int(d.sum())
    assert tails_differ > 0   # (the unread tails do differ: stale slot bytes either way)
    # the decoded frames of the valid records: equal, from the compact log written into a slot of other stale bytes
    for i in range(K):
        if kinds[i] in ('valid', 'multi'):
            compact.put_row(i, new_slot)
            a, b = decode_frame_host(full[i]), decode_frame_host(new_slot)
            assert a.keys() == b.keys()
            for k in a:
                np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    # every ROW field but visible_frame equal; the frame equal up to visible_bytes
    for i in range(K):
        compact.put_row(i, new_slot)
        for name in ROW.names:
            if name != 'visible_frame':
                np.testing.assert_array_equal(full[i][name], new_slot[name], err_msg=name)
        n = int(full[i]['visible_bytes'][0])
        np.testing.assert_array_equal(full[i]['visible_frame'][0, :n], new_slot['visible_frame'][0, :n])


def test_put_row_writes_the_shared_slot_in_place():
    lane_rows = np.zeros(4, ROW)   # (attach_v2's lane rows: the worker's slot is lane_rows[index:index + 1])
    slot = lane_rows[2:3]
    log = V2Log()
    src = np.zeros(1, ROW)
    src['visible_bytes'] = 100
    src['visible_frame'][0, :100] = 7
    src['t'] = 5
    log.keep_row(src)
    log.put_row(0, slot)
    assert lane_rows['t'][2] == 5 and (lane_rows['visible_frame'][2, :100] == 7).all()
    assert lane_rows['t'][1] == 0 and lane_rows['t'][3] == 0


def test_bytes_per_record():
    """the kept size: ROW_FRAME_AT + visible_bytes (20,804 + the frame) against 151,880"""
    log = V2Log()
    for n in (0, 11 * 1024, FRAME_CAP):
        r = np.zeros(1, ROW)
        r['visible_bytes'] = n
        log.keep_row(r)
    assert [len(b) for b in log.rows] == [ROW_FRAME_AT, ROW_FRAME_AT + 11 * 1024, ROW_FRAME_AT + FRAME_CAP]
    assert ROW_FRAME_AT == 20804 and ROW.itemsize == 151880
