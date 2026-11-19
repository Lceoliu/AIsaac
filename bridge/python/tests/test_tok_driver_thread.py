"""P package phase 1, milestone B (2026-11-19): the driver thread (abp_tok_thread.h) without a GPU -- TokMiddle in fake
mode (each record's outputs from its 't'), fake workers on socketpairs writing real v3 frames (the fixtures) into
shared ROW slots. The asynchronous gate (P-driver-design 11.5, third gate): every record answered once and committed
once, in order per worker, no record lost or doubled, whatever the credits (blocks, queue, ring rows), the consumer's
lag, the workers' pace; episode ends without an action replied at once; pause / resume; stop and fatal wake every
waiter (the thread exits promptly, even when short of credits); 'F' + 'R', an 'F' without its 'R', a malformed
message, a worker's end and EOF. Skipped where the extension cannot be built."""
import os
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from isaac_bridge import tok_driver, tok_native
from isaac_bridge import schema_v3 as V3
from isaac_bridge.tok_obs import ROW
from isaac_bridge.tok_policy import FoldLayout

M = tok_driver.load(need_cuda=False) if os.name != 'nt' else None
pytestmark = pytest.mark.skipif(M is None, reason=f'tok_driver: {tok_driver.REASON}')

HEADS = (9, 9, 2, 3, 6)
NH = len(HEADS)
COLS = NH + 7   # (actions, logp, value, vg, the four temporal counters)
FIX = Path(__file__).resolve().parent / 'fixtures' / 'visible_v3'
REPLY = b'\x00\x00\x00\x01a'


def frame():
    b = (FIX / 'doorframe.v3.bin').read_bytes()
    assert len(b) % 4 == 0
    return b


def middle():
    lay = FoldLayout(16, HEADS, 0, 1, 0, (16, 32, 64, 128), (8, 16))
    m = M.TokMiddle(tok_native.fold_layout(lay), tok_native.v3_consts(), tok_driver.row_fields(), COLS,
                    4 * 2 * 192, 192)
    m.fake = True
    return m


class Rig:
    """W fake workers, the shared rows / actions, a driver with M blocks of n rows and K ring rows."""

    def __init__(self, W=6, blocks=4, n=8, K=64, q_cap=0, episode=7, ends=(), done_kind=1):
        self.W, self.n = W, n
        self.rows = np.zeros(2 * W, ROW)
        self.actions = np.zeros((W, 6), np.int32)
        self.main_value = np.zeros(W)
        self.main_vg = np.zeros(W)
        self.cf_snap = np.zeros(W, np.int64)
        self.blocks = np.zeros((blocks, n), ROW)
        pairs = [socket.socketpair() for _ in range(W)]
        self.drv_socks, self.wk_socks = [p[0] for p in pairs], [p[1] for p in pairs]
        self.mid = middle()
        d = self.d = M.TokDriver(self.mid)
        d.set_sampler(self.rows.ctypes.data, W, [s.fileno() for s in self.drv_socks], self.actions.ctypes.data,
                      self.actions.shape[1], self.main_value.ctypes.data, self.main_vg.ctypes.data,
                      self.cf_snap.ctypes.data)
        d.set_rules(ROW.fields['done'][1], ROW.fields['first'][1], ROW.fields['t'][1], 4, 0, 0)
        d.set_blocks(self.blocks.ctypes.data, blocks, n, q_cap)
        d.set_ring_rows(K)
        self.fr = frame()
        self.episode, self.done_kind = episode, done_kind
        self.got = {w: [] for w in range(W)}     # (t, actions) the workers saw
        self.errors = []

    def record(self, w, t):
        slot = 2 * w + (t & 1)
        r = self.rows[slot:slot + 1]
        r[:] = 0
        b = r.view(np.uint8).reshape(-1)
        fa = ROW.fields['visible_frame'][1]
        b[fa:fa + len(self.fr)] = np.frombuffer(self.fr, np.uint8)
        r['visible_schema'] = V3.VERSION
        r['visible_bytes'] = len(self.fr)
        r['t'] = t
        r['first'] = 1 if t % self.episode == 0 else 0
        r['done'] = self.done_kind if t % self.episode == self.episode - 1 else 0
        return slot

    def worker(self, w, T, pace=0.0, end_with=b'x', fail_at=None):
        s = self.wk_socks[w]
        try:
            for t in range(T):
                if fail_at is not None and t == fail_at[0]:
                    for m in fail_at[1]:
                        s.sendall(m)
                    return
                slot = self.record(w, t)
                s.sendall(struct.pack('!i', 1) + (b'0' if slot % 2 == 0 else b'1'))
                got = b''
                while len(got) < 5:
                    c = s.recv(5 - len(got))
                    if not c:
                        return
                    got += c
                if got != REPLY:
                    self.errors.append((w, t, got))
                self.got[w].append((t, self.actions[w, :NH].copy()))
                if pace:
                    time.sleep(pace)
            if end_with:
                s.sendall(struct.pack('!i', len(end_with)) + end_with)
        except OSError as e:
            self.errors.append((w, 'os', str(e)))

    def consume(self, until, lag=0.0, timeout=20.0, release=True):
        """Records until `until(recs)` or the timeout; checks the order per worker; releases the credits."""
        recs, end = [], time.time() + timeout
        seen = getattr(self, '_seen', {})
        self._seen = seen
        while time.time() < end and not until(recs):
            d = self.d.take(64, 0.05)
            if len(d['seq']) == 0:
                if self.d.fatal()[0]:
                    break
                continue
            if lag:
                time.sleep(lag)
            for i in range(len(d['seq'])):
                recs.append({k: v[i] for k, v in d.items()})
            if release:   # (a call's block back once all its records came -- a take may end inside a call)
                for c_, b_, k_ in zip(d['call'], d['block'], d['n_call']):
                    if b_ < 0:
                        continue
                    seen[int(c_)] = seen.get(int(c_), 0) + 1
                    if seen[int(c_)] == int(k_):
                        self.d.release_block(int(b_))
                rr = d['ring'][d['ring'] >= 0]
                if len(rr):
                    self.d.release_ring(int(rr.max()) + 1)
        return recs

    def close(self):
        self.d.stop()
        for s in self.drv_socks + self.wk_socks:
            s.close()


def expected_actions(t):
    return np.array([(t + c) % 3 for c in range(NH)])


def check_records(rig, recs, T, episode):
    seqs = [r['seq'] for r in recs]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), 'sequence order, no duplicate'
    by_w = {w: [r for r in recs if r['worker'] == w and r['kind'] == 0] for w in range(rig.W)}
    for w, rs in by_w.items():
        assert len(rs) == T, (w, len(rs))
        for t, r in enumerate(rs):
            ends = t % episode == episode - 1
            assert r['go'] == (0 if ends else 1), (w, t)
            if r['go']:
                assert (r['actions'] == expected_actions(t)).all(), (w, t, r['actions'])
                assert r['value'] == t and r['logp'] == -t


@pytest.mark.parametrize('blocks,K,lag,pace', [(4, 64, 0.0, 0.0), (1, 8, 0.002, 0.0), (2, 16, 0.0, 0.003),
                                               (2, 9, 0.0, 0.0)])   # (K = 9: the ring wraps inside calls)
def test_every_record_once_in_order(blocks, K, lag, pace):
    W, T, ep = 6, 40, 7
    rig = Rig(W=W, blocks=blocks, n=8, K=K, episode=ep)
    rig.d.start()
    ths = [threading.Thread(daemon=True, target=rig.worker, args=(w, T, pace * (w % 2))) for w in range(W)]
    for t_ in ths:
        t_.start()
    try:
        recs = rig.consume(lambda rs: sum(r['kind'] == 1 for r in rs) == W, lag=lag)
        c = rig.d.counters()
    finally:
        rig.close()   # (a stall fails here, never hangs: the driver joined, the sockets closed, the workers daemons)
    assert sum(r['kind'] == 1 for r in recs) == W, ('stalled', c)
    for t_ in ths:
        t_.join(5)
    assert not rig.errors, rig.errors[:3]
    check_records(rig, recs, T, ep)
    assert sum(r['kind'] == 1 for r in recs) == W   # (each worker's 'x')
    for w in range(W):   # (the workers saw their own actions, the episode's ends without one)
        assert [t for t, _ in rig.got[w]] == list(range(T))
        for t, a in rig.got[w]:
            if t % ep != ep - 1:
                assert (a == expected_actions(t)).all()
    assert c['accepted'] == W * T and c['replies'] == W * T
    if blocks == 1:
        assert c['credit_waits'] > 0   # (the credits were short: the workers waited, nothing was lost)


def test_pause_and_resume():
    W, T, ep = 4, 30, 5
    rig = Rig(W=W, episode=ep)
    rig.d.start()
    ths = [threading.Thread(daemon=True, target=rig.worker, args=(w, T, 0.002)) for w in range(W)]
    for t_ in ths:
        t_.start()
    recs = rig.consume(lambda rs: len(rs) >= 20)
    rig.d.pause()
    t0 = time.time()
    while not rig.d.is_paused() and time.time() - t0 < 5:
        recs += rig.consume(lambda rs: True, timeout=0.05)
    assert rig.d.is_paused()
    recs += rig.consume(lambda rs: False, timeout=0.3)   # (drained: nothing more while paused)
    n_paused = len(recs)
    time.sleep(0.2)
    assert len(rig.consume(lambda rs: False, timeout=0.2)) == 0 and rig.d.last_seq() == recs[-1]['seq'] + 1
    rig.d.resume()
    recs += rig.consume(lambda rs: sum(r['kind'] == 1 for r in rs) == W)
    for t_ in ths:
        t_.join(5)
    rig.close()
    assert n_paused < W * T
    check_records(rig, recs, T, ep)


def test_stop_wakes_a_driver_short_of_credits():
    rig = Rig(W=4, blocks=1, n=4, K=4)
    rig.d.start()
    ths = [threading.Thread(daemon=True, target=rig.worker, args=(w, 50)) for w in range(4)]
    for t_ in ths:
        t_.start()
    time.sleep(0.3)   # (nothing consumed: the one block and the ring are held -- the driver waits for credits)
    assert rig.d.counters()['credit_waits'] >= 1
    t0 = time.time()
    rig.d.stop()
    assert time.time() - t0 < 2.0
    for s in rig.drv_socks + rig.wk_socks:
        s.close()
    for t_ in ths:
        t_.join(5)


@pytest.mark.parametrize('msgs,code,text', [
    ([struct.pack('!i', 1) + b'F', struct.pack('!i', 9) + b'Rbad v3 f'], 86, b'bad v3 f'),
    ([struct.pack('!i', 1) + b'F'], 86, b'never came'),
    ([struct.pack('!i', 2) + b'zz'], 2, b'unknown'),
])
def test_fatal_messages_stop_the_driver(msgs, code, text):
    rig = Rig(W=3)
    rig.d.r_timeout_s = 0.3
    rig.d.start()
    ths = [threading.Thread(daemon=True, target=rig.worker, args=(w, 20, 0.0, b'x', (5, msgs) if w == 1 else None))
           for w in range(3)]
    for t_ in ths:
        t_.start()
    rig.consume(lambda rs: rig.d.fatal()[0] != 0, timeout=10)
    f = rig.d.fatal()
    t0 = time.time()
    rig.close()
    assert time.time() - t0 < 2.0
    for t_ in ths:
        t_.join(5)
    assert f[0] == code and f[1] == 1 and text in f[2], f


def test_a_worker_gone_is_reported():
    rig = Rig(W=3)
    rig.d.start()
    ths = [threading.Thread(daemon=True, target=rig.worker, args=(w, 10, 0.0, None if w == 2 else b'x')) for w in range(3)]
    for t_ in ths:
        t_.start()
    for t_ in ths:
        t_.join(5)
    rig.wk_socks[2].close()   # (worker 2 dies without its 'x': EOF)
    recs = rig.consume(lambda rs: sum(r['kind'] in (1, 2) for r in rs) == 3)
    rig.close()
    kinds = {int(r['worker']): int(r['kind']) for r in recs if r['kind'] in (1, 2)}
    assert kinds == {0: 1, 1: 1, 2: 2}
    assert rig.d.fatal()[0] == 0


def test_a_ring_smaller_than_a_call_is_refused():
    rig = Rig(W=2, n=8, K=6)
    with pytest.raises(Exception, match='must hold a call'):
        rig.d.start()
    for s_ in rig.drv_socks + rig.wk_socks:
        s_.close()


def test_each_records_priv_is_copied_beside_it():
    """(tok_priv) The worker's slot can be overwritten once it is answered: the driver copies its PRIV record into the
    priv block row of the record (the consumer reads that, never the slot)."""
    W, T, ep = 4, 25, 6
    PRIVD = np.dtype([('t', np.int64), ('w', np.int64), ('pad', np.uint8, 40)])
    rig = Rig(W=W, blocks=2, n=8, K=16, episode=ep)
    priv = np.zeros(2 * W, PRIVD)
    pblocks = np.zeros((2, 8), PRIVD)
    rig.d.set_priv(priv.ctypes.data, PRIVD.itemsize, pblocks.ctypes.data)
    orig = rig.record

    def record(w, t):
        slot = orig(w, t)
        priv[slot]['t'], priv[slot]['w'] = t, w
        return slot
    rig.record = record
    seen = []
    rig.d.start()
    ths = [threading.Thread(daemon=True, target=rig.worker, args=(w, T)) for w in range(W)]
    for t_ in ths:
        t_.start()
    try:
        end = time.time() + 20
        ends = 0
        while ends < W and time.time() < end:
            d = rig.d.take(64, 0.05)
            for i in range(len(d['seq'])):
                if d['kind'][i] == 1:
                    ends += 1
                    continue
                b, r = int(d['block'][i]), int(d['brow'][i])
                seen.append((int(d['worker'][i]), int(pblocks[b, r]['w']), int(pblocks[b, r]['t'])))
                time.sleep(0.001)   # (a slow consumer: the workers move on meanwhile)
            for c_, b_, k_ in zip(d['call'], d['block'], d['n_call']):
                if b_ >= 0:
                    rig._seen = getattr(rig, '_seen', {})
                    rig._seen[int(c_)] = rig._seen.get(int(c_), 0) + 1
                    if rig._seen[int(c_)] == int(k_):
                        rig.d.release_block(int(b_))
            rr = d['ring'][d['ring'] >= 0]
            if len(rr):
                rig.d.release_ring(int(rr.max()) + 1)
    finally:
        rig.close()
    for w in range(W):
        mine = [(pw, pt) for sw, pw, pt in seen if sw == w]
        assert mine == [(w, t) for t in range(T)], (w, mine[:5])


# ---- Codex round 17 (2026-11-19): the handshake, the fds read while waiting, the reply's outcome, a worker's end after
#      its record, the end's cancels, each record's ep_archive column

def _send_record(rig, w, t):
    slot = rig.record(w, t)
    rig.wk_socks[w].sendall(struct.pack('!i', 1) + (b'0' if slot % 2 == 0 else b'1'))


def _recv_reply(rig, w, timeout=5.0):
    s = rig.wk_socks[w]
    s.settimeout(timeout)
    got = b''
    while len(got) < 5:
        c = s.recv(5 - len(got))
        if not c:
            break
        got += c
    return got


def test_resume_then_pause_is_acknowledged():
    """(17-2) A pause right after a resume, before the thread woke: the new pause is acknowledged (a generation of its
    own), never a drain waiting forever; nothing lost or doubled."""
    W, T, ep = 4, 60, 7
    rig = Rig(W=W, episode=ep)
    rig.d.start()
    ths = [threading.Thread(daemon=True, target=rig.worker, args=(w, T, 0.001)) for w in range(W)]
    for t_ in ths:
        t_.start()
    recs = []
    try:
        for _ in range(60):
            rig.d.pause()
            t0 = time.time()
            while not rig.d.is_paused() and time.time() - t0 < 3:
                recs += rig.consume(lambda rs: True, timeout=0.01)
            assert rig.d.is_paused(), 'a pause never acknowledged'
            rig.d.resume()
            rig.d.pause()   # (at once: the thread may not have woken from the last pause yet)
            t0 = time.time()
            while not rig.d.is_paused() and time.time() - t0 < 3:
                recs += rig.consume(lambda rs: True, timeout=0.01)
            assert rig.d.is_paused(), 'the pause after a resume never acknowledged'
            rig.d.resume()
            recs += rig.consume(lambda rs: True, timeout=0.01)
        recs += rig.consume(lambda rs: sum(r['kind'] == 1 for r in rs) == W, timeout=20)
    finally:
        rig.close()
    check_records(rig, recs, T, ep)
    assert rig.d.counters()['pause_gen'] == 120


def test_a_workers_eof_comes_after_its_record():
    """(17-3) EOF read while the worker's record waits for credits: the record first (its reply failed: replied 0,
    counted), then the worker's GONE."""
    rig = Rig(W=2, blocks=1, n=2, K=4)
    rig.d.start()
    try:
        _send_record(rig, 1, 0)
        assert _recv_reply(rig, 1) == REPLY   # (worker 1's call holds the only block: nothing released)
        _send_record(rig, 0, 0)
        rig.wk_socks[0].close()               # (worker 0 dies right after sending)
        time.sleep(0.3)                       # (the driver, short of credits, reads the EOF meanwhile)
        recs = rig.consume(lambda rs: any(r['kind'] == 2 for r in rs), timeout=10)
        c = rig.d.counters()
    finally:
        rig.close()
    mine = [(int(r['kind']), int(r['replied'])) for r in recs if int(r['worker']) == 0]
    assert mine == [(0, 0), (2, 1)], mine
    assert c['reply_fail'] == 1


def test_f_and_half_r_advance_while_short_of_credits():
    """(17-3) An 'F' and part of its 'R' while the driver waits for credits: the reader still runs, the missing rest
    times out into the fatal (86) -- before, the driver slept on the credits and never saw it."""
    rig = Rig(W=3, blocks=1, n=2, K=4)
    rig.d.r_timeout_s = 0.3
    rig.d.start()
    try:
        _send_record(rig, 1, 0)
        assert _recv_reply(rig, 1) == REPLY   # (the only block held)
        _send_record(rig, 2, 0)               # (waits for credits)
        rig.wk_socks[0].sendall(struct.pack('!i', 1) + b'F' + struct.pack('!i', 20) + b'Rhalf')
        t0 = time.time()
        while rig.d.fatal()[0] == 0 and time.time() - t0 < 5:
            time.sleep(0.05)
        f = rig.d.fatal()
    finally:
        rig.close()
    assert f[0] == 86 and f[1] == 0, f


def test_close_serve_cancels_the_records_read():
    """(17-2) Records the driver read but never took (credits short at the stop) and records arriving after the stop:
    answered without an action until the workers end -- none waits for a reply nobody writes."""
    W = 3
    rig = Rig(W=W, blocks=1, n=2, K=4)
    rig.d.start()
    got = {}

    def late(w):   # (a record, its reply, then the end)
        _send_record(rig, w, 0)
        got[w] = _recv_reply(rig, w, 10)
        rig.wk_socks[w].sendall(struct.pack('!i', 1) + b'x')
    try:
        _send_record(rig, 0, 0)
        assert _recv_reply(rig, 0) == REPLY   # (the only block held)
        ths = [threading.Thread(daemon=True, target=late, args=(w,)) for w in (1, 2)]
        for t_ in ths:
            t_.start()
        time.sleep(0.3)   # (1 and 2 read, waiting for credits)
        rig.d.stop()
        rig.wk_socks[0].sendall(struct.pack('!i', 1) + b'x')
        n_c, open_w = rig.d.close_serve(5.0)
        for t_ in ths:
            t_.join(5)
        recs = rig.consume(lambda rs: sum(r['kind'] == 1 for r in rs) == W, timeout=5, release=False)
    finally:
        for s in rig.drv_socks + rig.wk_socks:
            s.close()
    assert got == {1: REPLY, 2: REPLY}
    assert n_c == 2 and open_w == 0, (n_c, open_w)
    assert sorted(int(r['worker']) for r in recs if r['kind'] == 1) == [0, 1, 2]
    assert rig.d.counters()['cancelled'] == 2


def test_each_records_origin_is_copied_at_its_take():
    """(17-5) The worker's ep_archive column as it was when its record was taken: the worker may change it right after
    its reply (a new episode's start), the consumer reads the record's copy."""
    W, T = 3, 20
    rig = Rig(W=W, blocks=2, n=4, K=16)
    stats = np.zeros((W, 5))
    col = stats[:, 3]
    rig.d.set_origin(col.ctypes.data, col.strides[0])
    orig = rig.record

    def record(w, t):
        stats[w, 3] = 1000 * w + t
        return orig(w, t)
    rig.record = record
    rig.d.start()
    ths = [threading.Thread(daemon=True, target=rig.worker, args=(w, T)) for w in range(W)]
    for t_ in ths:
        t_.start()
    try:
        recs = rig.consume(lambda rs: sum(r['kind'] == 1 for r in rs) == W, lag=0.002)
    finally:
        rig.close()
    for w in range(W):
        mine = [float(r['origin']) for r in recs if int(r['worker']) == w and r['kind'] == 0]
        assert mine == [1000.0 * w + t for t in range(T)], (w, mine[:5])
