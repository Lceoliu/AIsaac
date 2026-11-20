"""(2026-11-20, the memory ledger: ISAAC_RL_RING_COMPACT; Codex round 25: the rings' slot generations) The workers'
teacher and V2 rings written / read only up to each record's frame end, and every record a reader takes checked against
its slot's generation (tok_sampler.ring_open / ring_publish) before and after it is copied, in both modes:
  (a) every record taken from a ring: its fixed fields (field by field) and its frame up to visible_bytes are the same
      bytes as the record written (real v3 frames of different lengths); past visible_bytes compact mode has zeros
      where the whole-slot copy had the slot's stale bytes of earlier records. The dtype's alignment padding is not
      compared: the readers assemble their output with np.stack, which leaves it uninitialised in either mode. The
      readers downstream (tok_obs.compact_copy, the trainer's chunks; tok_policy.to_batch, the learner's batches) give
      the same results from both;
  (b) the lengths: a record without a frame (visible_schema 0) ends at the frame whatever its visible_bytes; a frame
      length outside [0, 131,072] is refused (the readers count it, ring_bad, and skip it);
  (c) the sampler's readers (teacher_records / v2_records) on stand-ins: a full ring, a ring that wrapped, a reader
      that fell more than a ring behind -- the records taken checked one by one (their ids and meta), the ones written
      over counted (ring_*_lapped);
  (d) a writer in a forked child (shared memory, as the workers write) reusing the slot a reader is copying, forced at
      two points -- after the reader read the record's length, and after it copied the ROW, before the meta: the
      reader takes the complete older record or drops it (ring_*_torn), never the newer one's bytes as the older
      record; a negative control (the same interleaving with the generations not kept) shows the check is what
      catches it;
  (e) a writer in a forked child writing without pause while the reader reads: every record taken is one record
      whole, and taken + lapped + torn = written;
  (f) Linux: the slot pages a reader touches: in compact mode the frame budget's tail stays unresident.
The (d) interleavings take the reader as in 6aca452 (no generations) as their negative control."""
import json
import mmap
import os
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from isaac_bridge import abplus_lean as AL   # noqa: E402
from isaac_bridge import tok_obs as TO   # noqa: E402
from isaac_bridge import tok_sampler as TS   # noqa: E402
from isaac_bridge import tok_teacher2 as T2   # noqa: E402
from isaac_bridge import visible_producer as VP   # noqa: E402
from isaac_bridge.tok_obs import FRAME_W, ROW, ROW_FRAME_AT   # noqa: E402

_NO_FORK = pytest.mark.skipif(not hasattr(os, 'fork'), reason='os.fork / shared memory: Linux')


def FORK(f):
    """Linux only, and (912ad10) fork_in_process: these tests fork the pytest process, so they run where no device
    context exists (conftest skips them while a device is visible; gate.sh runs them in its second pass)."""
    return pytest.mark.fork_in_process(_NO_FORK(f))
FIELDS = [n for n in ROW.names if n != 'visible_frame']


def _raw():
    hits = list(HERE.rglob('forgotten.raw.json'))
    return json.loads(hits[0].read_text())['raw']


def real_rows(n, seed=0):
    """n ROW records encode_row wrote from synthetic v3 lean payloads (test_fast_row_v9's world): lean entities varied,
    and the v3 frames' entities too (frames of 15-25 KB, each length different)."""
    import test_obs_v2_smoke as V
    old = VP.VISIBLE_V3_MODE
    VP.VISIBLE_V3_MODE = 2
    try:
        raw = _raw()
        frames = {}
        rng = np.random.default_rng(seed)
        rows = np.zeros(n, ROW)
        dec, st = AL.LeanDecoder(), TO.EpisodeState(10 ** 6, floor=True, stall=6)
        for k in range(n):
            ne = int(rng.integers(1, 40))
            if ne not in frames:
                frames[ne] = VP.observation_frame_v3(dict(raw, entities=[raw['entities'][0]] * ne))
            frame = frames[ne]
            p, d, e, room = V.world(0)
            e = np.resize(e, int(rng.integers(1, 40)))
            p['x'] = 300.0 + 2.0 * k
            flags = 64 | (2 if k == 0 else 0)
            out = struct.pack('<IIIBBBB', AL.MAGIC, 100 + 4 * k, 200 + 4 * k, flags, len(p), len(d), 0)
            out += struct.pack('<iiiiiiiiffff', *room)
            out += struct.pack('<dddd', 0.0, 30.0, 30.0, 1.0)
            out += p.tobytes() + d.tobytes() + struct.pack('<H', len(e))
            out += np.ascontiguousarray(e).view(np.uint8).reshape(len(e), AL.ENTITY.itemsize)[:, :148].tobytes()
            if k == 0:
                tj, gj = V.terrain_json()
                out += struct.pack('<II', 7, len(tj)) + tj + struct.pack('<I', len(gj)) + gj
            out += struct.pack('<I', len(frame)) + frame
            TO.encode_row(dec.decode(out), st, rows[k:k + 1], k)
        return rows
    finally:
        VP.VISIBLE_V3_MODE = old


@pytest.fixture(scope='module', autouse=True)
def torch_pools():
    """tok_policy.to_batch's first call starts torch's process-lifetime CPU pool (6 native threads when this file
    runs alone; an earlier test starts it in the whole suite): started here, before any test's own thread check
    (conftest.nothing_left_running), as conftest's session warm-up does for the pools it knows. No thread let off."""
    try:
        import torch
    except ImportError:
        return
    from isaac_bridge import tok_policy as TP
    rows = real_rows(2, seed=9)
    TP.to_batch(rows, np.zeros((2, 5), np.int64), torch.device('cpu'))


def stale_ring(shape, seed=1):
    """A ring whose slots hold earlier records with large frames and noise past them (a long-running ring)."""
    rng = np.random.default_rng(seed)
    ring = np.zeros(shape, ROW)
    raw = ring.view(np.uint8).reshape(-1, ROW.itemsize)
    raw[:] = rng.integers(0, 256, raw.shape, dtype=np.uint8)
    ring['visible_bytes'] = FRAME_W
    ring['visible_schema'] = 3
    return ring


def put_take(rows, compact, monkeypatch, ring_shape=(1, 8)):
    monkeypatch.setattr(TO, 'RING_COMPACT', compact)
    ring = stale_ring(ring_shape)
    out = np.zeros(len(rows), ROW)   # (byte copies of what ring_take returned: np.stack leaves padding uninitialised)
    raw = out.view(np.uint8).reshape(len(rows), -1)
    for j in range(len(rows)):
        slot = j % ring_shape[1]
        TO.ring_put(ring[0, slot:slot + 1], rows[j:j + 1])
        raw[j] = np.frombuffer(TO.ring_take(ring, 0, slot).tobytes(), np.uint8)
    return out


def same_fields(x, y, k):
    """Every fixed field of record k equal (field by field: the dtype's alignment padding is not a field)."""
    return all(x[n][k].tobytes() == y[n][k].tobytes() for n in FIELDS)


def test_a_records_taken_are_the_same_up_to_their_frame_end(monkeypatch):
    """Every field and the frame up to visible_bytes: the same, on real v3 frames of different lengths."""
    rows = real_rows(24)
    assert len(set(int(v) for v in rows['visible_bytes'])) > 5   # (frames of different lengths)
    whole = put_take(rows, False, monkeypatch)
    comp = put_take(rows, True, monkeypatch)
    bw, bc = whole.view(np.uint8).reshape(len(rows), -1), comp.view(np.uint8).reshape(len(rows), -1)
    br = rows.view(np.uint8).reshape(len(rows), -1)
    for k in range(len(rows)):
        n = TO.row_frame_end(rows[k:k + 1])
        assert n < ROW.itemsize   # (real frames: far below the budget)
        assert same_fields(whole, comp, k) and same_fields(rows, comp, k)
        assert bw[k, ROW_FRAME_AT:n].tobytes() == bc[k, ROW_FRAME_AT:n].tobytes() == br[k, ROW_FRAME_AT:n].tobytes()
        assert bc[k, :n].tobytes() == br[k, :n].tobytes()   # (compact: the record's own bytes, padding included)
        assert not bc[k, n:].any()   # compact: zeros past the frame end
    # the readers downstream: the trainer's compact chunks and the learner's batches
    cw, fw = TO.compact_copy(whole)
    cc, fc = TO.compact_copy(comp)
    assert fw.tobytes() == fc.tobytes()
    assert all(cw[n].tobytes() == cc[n].tobytes() for n in cw.dtype.names)
    torch = pytest.importorskip('torch')
    from isaac_bridge import tok_policy as TP
    pend = np.zeros((len(rows), 5), np.int64)
    bw_, bc_ = TP.to_batch(whole, pend, torch.device('cpu')), TP.to_batch(comp, pend, torch.device('cpu'))
    assert set(bw_) == set(bc_)
    for k in bw_:
        a, b = bw_[k], bc_[k]
        if torch.is_tensor(a):
            assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), k
        else:
            assert type(a) is type(b), k


def test_b_lengths_checked_against_the_frame_field(monkeypatch):
    """No frame: the record ends at the frame (a reused slot's visible_bytes ignored). A frame: [0, 131,072] taken,
    anything else refused, by the writer and the reader alike; the sampler's reader counts and skips it."""
    assert FRAME_W == 131072
    r = np.zeros(1, ROW)
    for schema, vb, want in ((0, 0, ROW_FRAME_AT), (0, 9999, ROW_FRAME_AT), (0, -5, ROW_FRAME_AT),
                             (3, 0, ROW_FRAME_AT), (3, 14000, ROW_FRAME_AT + 14000), (3, FRAME_W, ROW_FRAME_AT + FRAME_W)):
        r['visible_schema'], r['visible_bytes'] = schema, vb
        assert TO.row_frame_end(r) == want, (schema, vb)
    for vb in (-1, FRAME_W + 1, FRAME_W + 4, 2 ** 31 - 1):
        r['visible_schema'], r['visible_bytes'] = 3, vb
        with pytest.raises(ValueError):
            TO.row_frame_end(r)
    monkeypatch.setattr(TO, 'RING_COMPACT', True)
    ring = stale_ring((1, 2))
    with pytest.raises(ValueError):
        TO.ring_put(ring[0, 0:1], r)
    with pytest.raises(ValueError):   # (the stale slot's own length is good; this one's is not)
        ring['visible_bytes'][0, 1] = -7
        TO.ring_take(ring, 0, 1)
    # the reader: a published record with a bad length counted (ring_teach_bad) and skipped, the next one taken
    S = stand_in(1, 4, 4, np.zeros)
    for c, vb in enumerate((100, FRAME_W + 1, 200)):
        write_record(S, 'teach', 0, c, frame_len=vb)
    S.stats[0, TS.ST['teach']] = 3
    monkeypatch.setattr(TS, 'TEACH_RING', 4)
    rows, meta = TS.TokSampler.teacher_records(S)
    assert [int(x) for x in meta[:, 0]] == [rid(0, 0), rid(0, 2)]
    assert S.ring_skips['teach_bad'] == 1 and S.ring_skips['teach_torn'] == 0


# ---------------------------------------------------------------- stand-in samplers and the workers' writer protocol
def rid(w, c):
    return 1 + w * 100000 + c


def frame_of(ident, n):
    return ((np.arange(n, dtype=np.int64) * (ident % 7 + 1) + ident) % 251 + 1).astype(np.uint8)


def frame_len_of(ident):
    return 9000 + (ident * 7919) % 30000


def shared_zeros(shape, dtype):
    """An array in anonymous shared memory (MAP_SHARED: a forked child writes the parent's pages)."""
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buf = mmap.mmap(-1, max(n, 1)) if hasattr(os, 'fork') else bytearray(max(n, 1))
    return np.frombuffer(buf, dtype, count=int(np.prod(shape))).reshape(shape)


def stand_in(W, RT, RV, alloc):
    """The sampler's ring state as the readers see it (TokSampler.teacher_records / v2_records, called unbound)."""
    from isaac_bridge.tok_teacher2 import V2_META
    return types.SimpleNamespace(
        cfg=types.SimpleNamespace(workers=W, memory_dim=0), stats=alloc((W, len(TS.STATS)), np.float64),
        teach_rows=alloc((W, RT), ROW), teach_meta=alloc((W, RT, TS.TEACH_META), np.int32),
        teach_gen=alloc((W, RT), np.int64), teach_seen=np.zeros(W, np.int64),
        v2_rows=alloc((W, RV), ROW), v2_meta=alloc((W, RV, V2_META), np.float64), v2_gen=alloc((W, RV), np.int64),
        v2_t=None, v2_h=None, v2_seen=np.zeros(W, np.int64), ring_skips=dict.fromkeys(TS.RING_SKIPS, 0))


def write_record(S, kind, w, c, frame_len=None, gens=True, hook=None):
    """Record c of worker w into its ring as the workers write it (tok_floor.ring_write / the teacher's encode):
    ring_open, the ROW (ring_put), the meta, ring_publish; its id in episode and meta[0], a frame of its own (length
    frame_len_of(id) unless given) -- everything a reader takes is checkable against the id alone. The count is the
    caller's (the worker raises it after the record is published). hook(stage): between the parts (tests)."""
    rows, meta, gen = ((S.teach_rows, S.teach_meta, S.teach_gen) if kind == 'teach' else
                       (S.v2_rows, S.v2_meta, S.v2_gen))
    R = rows.shape[1]
    slot, ident = c % R, rid(w, c)
    n = frame_len_of(ident) if frame_len is None else frame_len
    src = np.zeros(1, ROW)
    src['episode'], src['t'], src['visible_schema'], src['visible_bytes'] = ident, c, 3, n
    if 0 <= n <= FRAME_W:
        src['visible_frame'][0, :n] = frame_of(ident, n)
    if gens:
        TS.ring_open(gen, w, slot, c)
    if 0 <= n <= FRAME_W or not TO.RING_COMPACT:
        TO.ring_put(rows[w, slot:slot + 1], src)
    else:   # (a bad length: written as is, bytes past the frame field not, for the reader's check)
        rows[w, slot:slot + 1].view(np.uint8).reshape(-1)[:ROW_FRAME_AT] = src.view(np.uint8)[:ROW_FRAME_AT]
    if hook:
        hook('row')
    meta[w, slot] = 0
    meta[w, slot, 0] = ident
    if gens:
        TS.ring_publish(gen, w, slot, c)


def check_whole(row, meta_row):
    """A record taken is one record whole: its id in episode and meta, its frame that id's, zeros past it."""
    ident = int(row['episode'])
    n = int(row['visible_bytes'])
    frame = np.frombuffer(row['visible_frame'].tobytes(), np.uint8)
    assert int(meta_row[0]) == ident, ('meta of another record', ident, int(meta_row[0]))
    assert n == frame_len_of(ident), ('length of another record', ident, n)
    assert frame[:n].tobytes() == frame_of(ident, n).tobytes(), ('frame of another record', ident)
    return ident


def take(S, kind):
    return (TS.TokSampler.teacher_records(S) if kind == 'teach' else TS.TokSampler.v2_records(S))[:2]


@pytest.mark.parametrize('compact', (False, True))
def test_c_full_wrapped_and_lagging_readers(monkeypatch, compact):
    """3 workers, rings of 8; bursts (5, 8, 9) then (0, 0, 12): worker 2 falls 1, then 4 records behind (9 and 12
    written, 8 kept). Both readers take exactly the last <= 8 of each worker's writes since its last read, each one
    whole; the others are counted lapped."""
    monkeypatch.setattr(TO, 'RING_COMPACT', compact)
    monkeypatch.setattr(TS, 'TEACH_RING', 8)
    monkeypatch.setattr(T2, 'V2_RING', 8)
    W, R = 3, 8
    S = stand_in(W, R, R, lambda shape, dt: (stale_ring(shape) if dt == ROW else np.zeros(shape, dt)))
    written = [0] * W
    for burst in ((5, 8, 9), (0, 0, 12)):
        since = list(written)
        for w, k in enumerate(burst):
            for _ in range(k):
                for kind in ('teach', 'v2'):
                    write_record(S, kind, w, written[w])
                written[w] += 1
            S.stats[w, TS.ST['teach']] = S.stats[w, TS.ST['v2_records']] = written[w]
        want = [rid(w, c) for w in range(W) for c in range(max(since[w], written[w] - R), written[w])]
        for kind in ('teach', 'v2'):
            rows, meta = take(S, kind)
            got = [check_whole(rows[j], meta[j]) for j in range(len(rows))]
            assert got == want, (kind, got, want)
            if compact:   # (the frame past visible_bytes: zeros; not the dtype's padding, np.stack's)
                assert all(not rows[j]['visible_frame'][int(rows[j]['visible_bytes']):].any() for j in range(len(rows)))
    assert S.ring_skips['teach_lapped'] == S.ring_skips['v2_lapped'] == (9 - 8) + (12 - 8)
    assert S.ring_skips['teach_torn'] == S.ring_skips['v2_torn'] == 0


# ---------------------------------------------------------------- (d) forced interleavings, a forked writer
class ForkWriter:
    """A child process that writes record (kind, w, c) into the shared rings on request, then answers (the worker);
    the parent reads (the trainer). gens False: the writer keeps no generations (the reader as before this check)."""

    def __init__(self, S, gens=True):
        self.S = S
        self.req_r, self.req_w = os.pipe()
        self.ans_r, self.ans_w = os.pipe()
        self.pid = os.fork()
        if self.pid == 0:
            code = 1
            try:
                os.close(self.req_w)
                os.close(self.ans_r)
                f = os.fdopen(self.req_r, 'rb')
                while True:
                    line = f.readline()
                    if not line:
                        break
                    kind, w, c = line.split()
                    w, c = int(w), int(c)
                    write_record(S, kind.decode(), w, c, gens=gens)
                    S.stats[w, TS.ST['teach' if kind == b'teach' else 'v2_records']] = c + 1
                    os.write(self.ans_w, b'k')
                code = 0
            finally:
                os._exit(code)
        os.close(self.req_r)
        os.close(self.ans_w)

    def write(self, kind, w, c):
        os.write(self.req_w, f'{kind} {w} {c}\n'.encode())
        assert os.read(self.ans_r, 1) == b'k'

    def close(self):
        os.close(self.req_w)
        _, status = os.waitpid(self.pid, 0)
        os.close(self.ans_r)
        assert os.waitstatus_to_exitcode(status) == 0


INTERLEAVINGS = [('teach', 'after_length', True), ('v2', 'after_length', True),
                 ('teach', 'after_row', False), ('teach', 'after_row', True),
                 ('v2', 'after_row', False), ('v2', 'after_row', True)]


def interleave(monkeypatch, kind, point, compact, gens):
    """Record 0 published; the reader starts on it; at `point` the writer (another process) publishes record R in
    the same slot, whole, before the reader goes on. What the reader returns, and its counters."""
    monkeypatch.setattr(TO, 'RING_COMPACT', compact)
    R = 4
    monkeypatch.setattr(TS, 'TEACH_RING', R)
    monkeypatch.setattr(T2, 'V2_RING', R)
    S = stand_in(1, R, R, shared_zeros)
    child = ForkWriter(S, gens=gens)
    try:
        child.write(kind, 0, 0)
        if not gens:   # (the negative control: the reader's generations say record 0, as nothing changes them)
            (S.teach_gen if kind == 'teach' else S.v2_gen)[0, 0] = 1
        S.stats[0, TS.ST['teach' if kind == 'teach' else 'v2_records']] = 1   # (the reader's count snapshot: 1)
        fired = []

        def fire():
            if not fired:
                fired.append(1)
                child.write(kind, 0, R)   # (record R: slot 0 again, a longer frame of its own)
        if point == 'after_length':
            orig = TO.row_frame_end

            def after_length(row1):
                n = orig(row1)
                fire()
                return n
            monkeypatch.setattr(TO, 'row_frame_end', after_length)
        else:
            orig_take = TS.ring_take

            def after_row(ring, i, slot):
                got = orig_take(ring, i, slot)
                fire()
                return got
            monkeypatch.setattr(TS, 'ring_take', after_row)
        rows, meta = take(S, kind)   # (its count snapshot: 1, record R not yet written)
        assert fired, 'the interleaving never happened'
        return rows, meta, S.ring_skips
    finally:
        monkeypatch.undo()
        child.close()


@FORK
@pytest.mark.parametrize('kind,point,compact', INTERLEAVINGS)
def test_d_slot_reused_while_read_older_record_whole_or_dropped(monkeypatch, kind, point, compact):
    """Codex round 25's case (the compact read took the old length, then the new record's bytes): with the slots'
    generations the record is dropped and counted, never taken as a mix."""
    rows, meta, sk = interleave(monkeypatch, kind, point, compact, gens=True)
    for j in range(len(rows)):
        assert check_whole(rows[j], meta[j]) == rid(0, 0)   # (only ever the older record, whole)
    assert len(rows) == 0 and sk[f'{kind}_torn'] == 1


@FORK
@pytest.mark.parametrize('kind,point,compact', INTERLEAVINGS)
def test_d_negative_control_without_generations_a_mix_is_taken(monkeypatch, kind, point, compact):
    """The same interleavings with the generations not kept (the readers as in 6aca452): a record that is not one
    record whole is taken -- the check in test_d above is what refuses it."""
    rows, meta, sk = interleave(monkeypatch, kind, point, compact, gens=False)
    assert len(rows) == 1
    with pytest.raises(AssertionError):
        check_whole(rows[0], meta[0])


# ---------------------------------------------------------------- (e) a writer that never waits
@FORK
@pytest.mark.parametrize('compact', (False, True))
def test_e_concurrent_writer_every_record_taken_is_whole(monkeypatch, compact):
    """A forked writer writes 3000 records per ring (2 workers, rings of 8: it laps the reader over and over) while
    the parent reads with the sampler's readers as fast as it can: each record taken is one record whole, in order,
    and taken + lapped + torn = written."""
    monkeypatch.setattr(TO, 'RING_COMPACT', compact)
    W, R, N = 2, 8, 3000
    monkeypatch.setattr(TS, 'TEACH_RING', R)
    monkeypatch.setattr(T2, 'V2_RING', R)
    S = stand_in(W, R, R, shared_zeros)
    done = shared_zeros((1,), np.int64)
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            for c in range(N):
                for w in range(W):
                    for kind, st in (('teach', 'teach'), ('v2', 'v2_records')):
                        write_record(S, kind, w, c)
                        S.stats[w, TS.ST[st]] = c + 1
            done[0] = 1
            code = 0
        finally:
            os._exit(code)
    taken = {('teach', w): [] for w in range(W)}
    taken.update({('v2', w): [] for w in range(W)})
    import time
    deadline = time.monotonic() + 300   # (Codex round 26: a writer that failed before setting done ends the wait)
    try:
        while True:
            assert time.monotonic() < deadline, 'the writer never finished'
            if pid is not None:
                got_pid, status = os.waitpid(pid, os.WNOHANG)
                if got_pid:   # (the writer ended: whole, or the test fails here instead of waiting forever)
                    pid = None
                    assert os.waitstatus_to_exitcode(status) == 0 and int(done[0]), 'the writer failed'
            last = int(done[0])
            for kind in ('teach', 'v2'):
                rows, meta = take(S, kind)
                for j in range(len(rows)):
                    ident = check_whole(rows[j], meta[j])
                    taken[(kind, (ident - 1) // 100000)].append(ident)
            if last:
                break
    finally:
        if pid is not None:
            _, status = os.waitpid(pid, 0)
            assert os.waitstatus_to_exitcode(status) == 0
    for (kind, w), ids in taken.items():
        assert ids == sorted(ids) and len(set(ids)) == len(ids), (kind, w)
    sk = S.ring_skips
    for kind in ('teach', 'v2'):
        n_taken = sum(len(v) for (k, _), v in taken.items() if k == kind)
        assert n_taken + sk[f'{kind}_lapped'] + sk[f'{kind}_torn'] == W * N, (kind, n_taken, sk)
        assert n_taken > 0


# ---------------------------------------------------------------- (f) resident pages
@FORK
def test_f_another_process_writes_and_the_tail_stays_unresident(monkeypatch):
    """A writer in a forked child (as a worker), the reader here (as the trainer): the same records; and the ring's
    shared-memory pages resident after a full read: compact mode leaves the frame budget's tail out."""
    from multiprocessing import shared_memory
    rows = real_rows(16, seed=5)
    W, R = 2, 16
    res = {}
    for compact in (False, True):
        monkeypatch.setattr(TO, 'RING_COMPACT', compact)
        shm = shared_memory.SharedMemory(create=True, size=ROW.itemsize * W * R)
        try:
            ring = np.ndarray((W, R), ROW, buffer=shm.buf)
            pid = os.fork()
            if pid == 0:
                code = 1
                try:
                    for w in range(W):
                        for j in range(R):
                            TO.ring_put(ring[w, j:j + 1], rows[j:j + 1])
                    code = 0
                finally:
                    os._exit(code)
            _, status = os.waitpid(pid, 0)
            assert os.waitstatus_to_exitcode(status) == 0
            taken = [TO.ring_take(ring, w, j) for w in range(W) for j in range(R)]
            for i, a in enumerate(taken):
                src = rows[i % R:i % R + 1]
                n = TO.row_frame_end(src)
                assert all(a[f].tobytes() == src[f][0].tobytes() for f in FIELDS)
                assert a.tobytes()[ROW_FRAME_AT:n] == src.tobytes()[ROW_FRAME_AT:n]
            rss = 0
            name = '/dev/shm/' + shm.name.lstrip('/')
            cur = None
            for line in open('/proc/self/smaps'):
                f = line.split()
                if len(f) >= 5 and '-' in f[0] and ':' in f[3]:
                    cur = (f[-1] if len(f) > 5 else '') == name
                elif cur and f[0] == 'Rss:':
                    rss += int(f[1])
            res[compact] = rss
            del ring, taken
        finally:
            shm.close()
            shm.unlink()
    full_kb = ROW.itemsize * W * R // 1024
    assert res[False] >= 0.9 * full_kb, res         # whole slots: every page resident
    assert res[True] <= 0.5 * full_kb, res           # compact: the frame budget's tail never touched


def test_g_workers_refuse_another_ring_mode(monkeypatch):
    """The trainer's ISAAC_RL_RING_COMPACT in cfg; a worker with another refuses to start; -1 (unset): no check."""
    cfg = types.SimpleNamespace(ring_compact=1)
    monkeypatch.setattr(TO, 'RING_COMPACT', False)
    with pytest.raises(RuntimeError):
        TS.check_ring_mode(cfg)
    monkeypatch.setattr(TO, 'RING_COMPACT', True)
    TS.check_ring_mode(cfg)
    TS.check_ring_mode(types.SimpleNamespace(ring_compact=-1))
    TS.check_ring_mode(types.SimpleNamespace())


def test_h_ring_open_counts_a_slot_already_open():
    """(A1's review) ring_open on a slot whose generation is negative (another writer has it open, or the previous
    attempt at the record failed midway) returns 1 -- the worker's ring_open_busy; a published or fresh slot 0."""
    gen = np.zeros((1, 2), np.int64)
    assert TS.ring_open(gen, 0, 0, 0) == 0 and gen[0, 0] == -1
    assert TS.ring_open(gen, 0, 0, 0) == 1        # (opened again before its publish)
    TS.ring_publish(gen, 0, 0, 0)
    assert gen[0, 0] == 1 and TS.ring_open(gen, 0, 0, 2) == 0 and gen[0, 0] == -3
    assert TS.ring_open(None, 0, 0, 5) == 0
    assert 'ring_open_busy' in TS.ST and TS.STATS[-2:] == ('np_runs', 'np_removed')


# ---------------------------------------------------------------- (i) the workers' ring confirmation (Codex round 26)
def _wait_closed(conn):
    try:
        conn.recv_bytes()
    except (EOFError, OSError):
        pass


def stub_worker_old(index, cfg, names, conn, teacher_names=None, *args, **kw):
    """A worker of a tree without the ring protocol (Codex round 26's stand-in): writes a teacher record and raises
    the ring's count, no generation, no confirmation."""
    from isaac_bridge import tok_sampler as S_
    blocks, rows, actions, stats, control = S_.attach(names, cfg.workers)
    tb, trows, tmeta = S_.attach_teacher(teacher_names, cfg.workers)
    trows[index, 0]['episode'] = 7
    stats[index, S_.ST['teach']] = 1
    del rows, actions, stats, control, trows, tmeta
    _wait_closed(conn)


def stub_worker_new(index, cfg, names, conn, teacher_names=None, *args, **kw):
    """A worker of this tree up to its games: confirms the protocol, then the mode check (tok_floor / worker_main's
    order). ISAAC_RL_TEST_BAD_WORKER=<i>: worker i runs the other ring mode (its own environment, as a mixed launch)."""
    from isaac_bridge import tok_obs as O_
    from isaac_bridge import tok_sampler as S_
    if str(index) == os.environ.get('ISAAC_RL_TEST_BAD_WORKER', ''):
        O_.RING_COMPACT = not O_.RING_COMPACT
    gb, tg, vg, ack = S_.attach_ring_gen(cfg.ring_gen_block, cfg.workers)
    S_.ring_confirm(ack, index)
    del tg, vg, ack
    S_.check_ring_mode(cfg)   # (the bad one ends here: RuntimeError, exit 1)
    _wait_closed(conn)


def sampler_with(monkeypatch, tmp_path, target, workers=2, confirm_s='4'):
    import select
    if not hasattr(select, 'poll'):
        pytest.skip('TokSampler: select.poll (Linux)')
    monkeypatch.setenv('ISAAC_RL_FATAL_DIR', str(tmp_path))
    monkeypatch.setenv('ISAAC_RL_PNG_SHARED', '0')
    monkeypatch.setenv('ISAAC_RL_RING_CONFIRM_S', confirm_s)
    monkeypatch.setattr(TS, 'worker_main', target)
    cfg = TS.TokSamplerConfig(workers=workers, teacher=True, name='ringtest', port=os.getpid() % 30000 + 20000)
    return TS.TokSampler(cfg)


def test_i_a_worker_without_the_ring_protocol_ends_the_run(monkeypatch, tmp_path):
    """Codex round 26: a worker that writes records without generations (an older tree) and a trainer of this one:
    before, every record it wrote was counted torn and dropped, silently (1 written, 0 taken); now the trainer stops
    at its start with the reason, nothing sampled."""
    from isaac_bridge.visible_producer import VisibleV3Fatal
    with pytest.raises(VisibleV3Fatal, match='confirmed nothing'):
        sampler_with(monkeypatch, tmp_path, stub_worker_old)


def test_i_one_worker_of_another_ring_mode_ends_the_run(monkeypatch, tmp_path):
    """Codex round 26: one bad worker (another ISAAC_RL_RING_COMPACT), one good: the run ends with the bad worker's
    confirmation in the reason -- not the bad worker alone dropped and the run going on with the other."""
    from isaac_bridge.visible_producer import VisibleV3Fatal
    monkeypatch.setenv('ISAAC_RL_TEST_BAD_WORKER', '0')
    with pytest.raises(VisibleV3Fatal, match=r'worker 0: ring protocol 2, ISAAC_RL_RING_COMPACT'):
        sampler_with(monkeypatch, tmp_path, stub_worker_new, confirm_s='60')


def test_i_workers_that_confirm_start(monkeypatch, tmp_path, capfd):
    """Both workers confirm the trainer's protocol and mode: the sampler starts (and closes cleanly), and says so
    in its log (the launch check after the merge: every worker's confirmation in the log)."""
    monkeypatch.delenv('ISAAC_RL_TEST_BAD_WORKER', raising=False)
    s = sampler_with(monkeypatch, tmp_path, stub_worker_new, confirm_s='60')
    try:
        assert [int(a) for a in s.ring_ack] == [TS.ring_ack_value(TO.RING_COMPACT)] * 2
        assert 'ring protocol confirmed by all 2 workers (ring protocol 2' in capfd.readouterr().err
    finally:
        s.close(timeout=2.0)


def test_j_two_actors_ring_skips_summed():
    """Codex round 26: with two actor processes (merge_jobs) the rings' skip counters are every actor's -- rows
    dropped by the second actor only reach progress (before: the first actor's dict was kept, zeros)."""
    pytest.importorskip('torch')
    import train_tok as TT
    zero = dict.fromkeys(TS.RING_SKIPS, 0)
    zero['open_busy'] = 0.0
    h0 = dict(roll=None, bid=1, aid=0, seq=4, finished=[], branch_finished=[], collect_s=1.5, wait_learner_s=0.2,
              cycle_s=2.0, cycle_wait_s=0.0, take_s=0.01, t_put=10.0, t_start=1.0, prof={}, worker={}, lane_stat={},
              stats=np.ones((3, 4)), teach=None, v2=None, choices=None, cpu_collect=0.2, proc_cores=1.0, dec_min=5.0,
              dec_max=9.0, dec_w='5 9 7', actor_mem=100, ring_skips=dict(zero))
    h1 = dict(h0, aid=1, ring_skips=dict(zero, v2_torn=3, teach_lapped=2, open_busy=1.0))
    j = TT.merge_jobs([h0, h1])
    assert j['ring_skips'] == dict(zero, v2_torn=3, teach_lapped=2, open_busy=1.0)
    assert TT.merge_jobs([h1, h1])['ring_skips']['v2_torn'] == 6
