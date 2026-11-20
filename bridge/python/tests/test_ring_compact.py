"""(2026-11-20, the memory ledger: ISAAC_RL_RING_COMPACT) The workers' teacher and V2 rings written / read only up to
each record's frame end. Checked against the whole-slot behaviour (the switch off) on the same records:
  (a) every record taken from a ring: its fixed fields (every byte before the frame, padding included) and its frame up
      to visible_bytes are the same bytes; past visible_bytes compact mode has zeros where the whole-slot copy had the
      slot's stale bytes of earlier records -- and the readers downstream (tok_obs.compact_copy, the trainer's chunks;
      tok_policy.to_batch, the learner's batches) give the same results from both;
  (b) a full ring, a ring that wrapped, a reader that fell behind by more than the ring (the sampler's own readers,
      teacher_records / v2_records, on stand-in samplers);
  (c) Linux: a writer in another process (shared memory, as the workers write), and the slot pages a reader touches:
      in compact mode the frame budget's tail stays unresident after the ring is read."""
import json
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
from isaac_bridge import visible_producer as VP   # noqa: E402
from isaac_bridge.tok_obs import ROW, ROW_FRAME_AT   # noqa: E402


def _raw():
    hits = list(HERE.rglob('forgotten.raw.json'))
    return json.loads(hits[0].read_text())['raw']


def real_rows(n, seed=0):
    """n ROW records encode_row wrote from synthetic v3 lean payloads (test_fast_row_v9's world), entities varied."""
    import test_obs_v2_smoke as V
    old = VP.VISIBLE_V3_MODE
    VP.VISIBLE_V3_MODE = 2
    try:
        frame = VP.observation_frame_v3(_raw())
        rng = np.random.default_rng(seed)
        rows = np.zeros(n, ROW)
        dec, st = AL.LeanDecoder(), TO.EpisodeState(10 ** 6, floor=True, stall=6)
        for k in range(n):
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


def stale_ring(shape, seed=1):
    """A ring whose slots hold earlier records with large frames and noise past them (a long-running ring)."""
    rng = np.random.default_rng(seed)
    ring = np.zeros(shape, ROW)
    raw = ring.view(np.uint8).reshape(-1, ROW.itemsize)
    raw[:] = rng.integers(0, 256, raw.shape, dtype=np.uint8)
    ring['visible_bytes'] = ROW.itemsize - ROW_FRAME_AT
    return ring


def put_take(rows, compact, monkeypatch, ring_shape=(1, 8)):
    monkeypatch.setattr(TO, 'RING_COMPACT', compact)
    ring = stale_ring(ring_shape)
    out = []
    for j in range(len(rows)):
        slot = j % ring_shape[1]
        TO.ring_put(ring[0, slot:slot + 1], rows[j:j + 1])
        out.append(TO.ring_take(ring[0, slot:slot + 1]))
    return np.stack(out)


FIELDS = [n for n in ROW.names if n != 'visible_frame']


def same_fields(x, y, k):
    """Every fixed field of record k equal (field by field: the dtype's alignment padding is not a field)."""
    return all(x[n][k].tobytes() == y[n][k].tobytes() for n in FIELDS)


def test_a_records_taken_are_the_same_up_to_their_frame_end(monkeypatch):
    """Every field and the frame up to visible_bytes: the same. Not the same, by design: (1) past visible_bytes the
    whole-slot copy has the slot's stale bytes, compact mode zeros; (2) the dtype's alignment padding between fields:
    the whole-slot write is a structured assignment (fields only: the slot keeps stale padding, as before this
    change), the compact write copies the record's bytes (its own padding). No reader reads either (the consumers
    below give the same results)."""
    rows = real_rows(24)
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


def test_b_full_wrapped_and_lagging_readers(monkeypatch):
    """The sampler's readers on stand-ins: 3 workers, rings of 8, writes 5 / 8 / 21 (the last wrapped twice); the
    reader called after each batch of writes and once after falling behind by more than the ring."""
    from isaac_bridge import tok_sampler as TS
    from isaac_bridge.tok_teacher2 import V2_META
    rows = real_rows(30, seed=3)
    for compact in (False, True):
        monkeypatch.setattr(TO, 'RING_COMPACT', compact)
        monkeypatch.setattr(TS, 'TEACH_RING', 8)
        W, R = 3, 8
        S = types.SimpleNamespace(cfg=types.SimpleNamespace(workers=W, memory_dim=0),
                                  stats=np.zeros((W, len(TS.STATS))), teach_rows=stale_ring((W, R)),
                                  teach_meta=np.zeros((W, R, TS.TEACH_META), np.int32), teach_seen=np.zeros(W, np.int64),
                                  v2_rows=stale_ring((W, R)), v2_meta=np.zeros((W, R, V2_META)), v2_t=None, v2_h=None,
                                  v2_seen=np.zeros(W, np.int64))
        got_t = []
        written = [0, 0, 0]
        for burst in ((5, 8, 9), (0, 0, 12)):   # (the second: worker 2 falls 12 > 8 behind its last read)
            for w, k in enumerate(burst):
                for _ in range(k):
                    c = written[w]
                    src = rows[(w * 7 + c) % len(rows):(w * 7 + c) % len(rows) + 1]
                    slot = c % R
                    S.teach_rows[w, slot:slot + 1] = src   # (the floor worker encodes in place: the same bytes)
                    TO.ring_put(S.v2_rows[w, slot:slot + 1], src)
                    written[w] += 1
                S.stats[w, TS.ST['teach']] = written[w]
                S.stats[w, TS.ST['v2_records']] = written[w]
            import isaac_bridge.tok_teacher2 as T2
            monkeypatch.setattr(T2, 'V2_RING', R)
            tr, _ = TS.TokSampler.teacher_records(S)
            vr, _ = TS.TokSampler.v2_records(S)
            got_t.append((tr, vr))
        # a reader sees at most the last R of each worker's writes since its last read
        flat_t = np.concatenate([g[0] for g in got_t])
        flat_v = np.concatenate([g[1] for g in got_t])
        assert len(flat_t) == len(flat_v) == 5 + 8 + 8 + 8   # (worker 2: 8 of its 9, then 8 of its 12)
        for a, b in zip(flat_t, flat_v):
            n = TO.row_frame_end(a.reshape(1))
            assert all(a[f].tobytes() == b[f].tobytes() for f in FIELDS)
            assert a.tobytes()[ROW_FRAME_AT:n] == b.tobytes()[ROW_FRAME_AT:n]
            if compact:
                assert not np.frombuffer(b.tobytes()[n:], np.uint8).any()


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='os.fork / shared memory: Linux')
def test_c_another_process_writes_and_the_tail_stays_unresident(monkeypatch):
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
                try:
                    for w in range(W):
                        for j in range(R):
                            TO.ring_put(ring[w, j:j + 1], rows[j:j + 1])
                finally:
                    os._exit(0)
            os.waitpid(pid, 0)
            taken = [TO.ring_take(ring[w, j:j + 1]) for w in range(W) for j in range(R)]
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
