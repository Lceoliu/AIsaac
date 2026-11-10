"""B58a+b gate: the driver's native consumer (Episodes.arrive's arithmetic and store, in C++) == the Python consumer,
byte for byte, on one record stream (no timing involved: driver.consume_rows runs the consumer the driver runs per step).

Two rollout pairs and two Episodes objects with the production options (run mode, items, heal / resource gains, the
pedestal pick, the PLR score, aux close arrays, value_global, privileged records, compact ROWC rollouts with frame
arenas); a random stream of records over several workers (episode starts, ends of every kind incl. branch truncations,
breaks, damage / hurt / events / stages / player gains, forced pickups, sampled actions and outputs); the Python side
runs Episodes.arrive + store, the native side consume_rows + _native_events + the sidecar record. Compared: every
rollout array (ROWC records incl. visible_off, the frame arenas' used bytes), every Episodes accumulator, the finished
episodes' summaries, the first-record snapshots, the PLR scores; then a hand-off (carry into the other rollout of each
pair) and more records into it. Also: the page store's vectors never reallocate (reserve to the caps).
Skipped without the extension (no CUDA torch, no g++, ISAAC_BLOCK_CPP_ACTOR=0)."""
import numpy as np
import pytest
import torch

from test_block_memory import make_policy
from test_visible_input import synth_rows
from test_b52_cpp_actor import EXT, fake_sampler, make_driver_for
from isaac_bridge.block_paged import SlotBlockActor
from isaac_bridge import block_driver as bdrv
from isaac_bridge.tok_obs import DONE_BRANCH, ROWC

pytestmark = pytest.mark.skipif(EXT is None, reason='cpp actor extension unavailable')

W, CAP = 5, 48


def make_pair(nh, priv_dtype):
    import train_tok as T
    return [T.Rollout(W, CAP, nh, 0, True, None, close=True, priv=priv_dtype, block_steps=128) for _ in range(2)]


def make_book(stats):
    import train_tok as T
    rw = dict(damage=1.0, hurt=0.75, room=1.0, explore=0.5, boss=2.0, exit=3.0, death=2.0, timeout=1.0, time=0.002,
              heal=0.5, resource=0.25, clear=1.0)
    b = T.Episodes(W, rw, True, run=True, items=True)
    b.plr = (stats, {'plr_score': 0, 'plr_episode': 1}, 0.99)
    b.stats0, b.char0, b.active0 = {}, {}, {}
    b.pick = dict(bped={}, steps=np.zeros(W, np.int64))
    b.seen_inv = []
    return b


def stream(rng, n_steps, nh):
    """per step: (workers, ROW records, go mask, outputs [n_go, nh+3], actions / pending [n_go, nh], priv bytes)"""
    started = np.zeros(W, bool)
    t = np.zeros(W, np.int64)
    ep = np.zeros(W, np.int64)
    out = []
    for k in range(n_steps):
        ws = np.sort(rng.choice(W, size=int(rng.integers(1, W+1)), replace=False))
        rows = synth_rows(rng, len(ws), seed=20000+k)
        for j, w in enumerate(ws):
            first = (not started[w]) or rng.random() < 0.05   # (a break: a new episode with one open)
            if first:
                ep[w] += 1; t[w] = 0; started[w] = True
            r = rows[j:j+1]
            r['first'] = int(first)
            r['episode'] = ep[w]; r['t'] = t[w]; r['group'] = 0; r['seed'] = 1000+w
            d = 0
            if not first and rng.random() < 0.12:
                d = int(rng.choice([1, 2, 3, DONE_BRANCH]))
            r['done'] = d
            r['damage'] = np.float32(rng.random()*3)
            r['hurt'] = np.float32(rng.random() < 0.2)
            r['events'] = int(rng.integers(0, 128))
            r['stage'] = int(rng.integers(1, 9))
            pl = r['player'][0]
            pl[7:11] = rng.random(4).astype(np.float32)
            pl[11:16] = rng.random(5).astype(np.float32)
            pl[21] = np.float32(rng.random() < 0.5)
            r['pforce'] = int(rng.random() < 0.1)
            r['pmode'] = int(rng.integers(0, 2))
            r['bped'] = rng.integers(0, 50, 3)
            r['inv'] = rng.integers(0, 4, (16, 2))
            r['pchar'] = int(rng.integers(0, 5))
            r['lab'] = 0
            r['branch'] = 1 if d == DONE_BRANCH else 0
            r['point'] = 0
            t[w] += 1
            if d not in (0, DONE_BRANCH):
                started[w] = False
        go = ~(rows['done'] != 0) | (rows['done'] == DONE_BRANCH)
        n = int(go.sum())
        o = np.zeros((n, nh+3), np.float32)
        o[:, :nh] = rng.integers(0, 4, (n, nh))
        o[:, nh:] = rng.standard_normal((n, 3)).astype(np.float32)
        acts = o[:, :nh].astype(np.int64)
        acts[:, 0] = np.where(rows['pforce'][go] > 0, 9, acts[:, 0])
        pend = rng.integers(0, 4, (n, nh))
        out.append((ws, rows, go, o, acts, pend))
    return out


def run_python(book, rolls, steps, nh, priv_dtype, ver0):
    import train_tok as T
    roll = rolls[0]
    pending = np.zeros((W, nh), np.int64)
    for k, (ws, rows, go, o, acts, pend) in enumerate(steps):
        g = book.arrive(roll, rows, ws, pending)
        assert np.array_equal(g, go)
        a = ws[go]
        priv = np.zeros(len(a), priv_dtype)
        priv.view(np.uint8)[:] = (np.arange(priv.nbytes) + k) % 251
        T.store(roll, a, rows[go], pend, acts, o[:, nh], o[:, nh+1], ver0+k, vg=o[:, nh+2], priv=priv,
                block_context=[('ctx', int(w), k) for w in a])
        if k == len(steps)//2:   # a hand-off
            rolls[1].carry(roll)
            roll = rolls[1]
    return roll


def run_native(d, book, rolls, steps, nh, priv_dtype, ver0):
    import train_tok as T
    d.use_rollout(rolls[0].native_slot)
    roll = rolls[0]
    for k, (ws, rows, go, o, acts, pend) in enumerate(steps):
        rows_c = np.ascontiguousarray(rows)
        n_all = len(ws)
        priv_all = np.zeros(n_all, priv_dtype)
        priv_go = np.zeros(int(go.sum()), priv_dtype)
        priv_go.view(np.uint8)[:] = (np.arange(priv_go.nbytes) + k) % 251
        priv_all[go] = priv_go
        kk, snap = d.consume_rows(torch.from_numpy(rows_c.view(np.uint8).reshape(-1)), [int(w) for w in ws],
                                  torch.from_numpy(go.astype(np.uint8)), torch.from_numpy(o), torch.from_numpy(acts),
                                  torch.from_numpy(np.ascontiguousarray(pend, np.int64)), ver0+k,
                                  torch.from_numpy(priv_all.view(np.uint8).copy()))
        T._native_events(book, rows, ws, snap.numpy())
        for j, (w, k_) in enumerate(zip(ws[go].tolist(), kk.numpy().tolist())):
            if k_ >= 0:
                roll.block.record(w, k_, rows[go][j], ('ctx', int(w), k))
        if k == len(steps)//2:
            rolls[1].carry(roll)
            roll = rolls[1]
            d.use_rollout(roll.native_slot)
    return roll


def same_rollout(a, b):
    for name in ('pending', 'actions', 'logp', 'value', 'value_g', 'reward', 'terminal', 'version', 'boot', 'close_hurt',
                 'close_done', 'close_ev', 'length', 'open'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    np.testing.assert_array_equal(a.priv.view(np.uint8), b.priv.view(np.uint8), err_msg='priv')
    assert a.rows.dtype == b.rows.dtype == ROWC
    np.testing.assert_array_equal(a.rows.view(np.uint8), b.rows.view(np.uint8), err_msg='ROWC records')
    assert a.frames.used == b.frames.used
    np.testing.assert_array_equal(a.frames.buf[:a.frames.used], b.frames.buf[:b.frames.used], err_msg='frames')
    assert a.block.starts.keys() == b.block.starts.keys()
    for w in a.block.starts:
        assert a.block.starts[w] == b.block.starts[w]
    assert a.block.open == b.block.open


def test_native_consumer_matches_python_consumer():
    from isaac_bridge.tok_priv import PRIV as PRIV_DTYPE
    torch.manual_seed(71)
    m = make_policy('cuda').eval()
    nh = len(m.heads)
    actor = SlotBlockActor(m, W, graph_entities=128, bf16=True, lane_slots=0)
    s = fake_sampler(W, nh)
    s.priv = np.zeros(2*W, PRIV_DTYPE)   # (the records' privileged rows: the driver copies them, the consumer stores them)
    d = make_driver_for(actor, s)
    steps = stream(np.random.default_rng(71), 120, nh)
    stats_a, stats_b = np.zeros((W, 2)), np.zeros((W, 2))
    book_a, book_b = make_book(stats_a), make_book(stats_b)
    rolls_a, rolls_b = make_pair(nh, PRIV_DTYPE), make_pair(nh, PRIV_DTYPE)
    bdrv.attach_consumer(d, book_b, rolls_b, 0.99)
    assert d.consumer_on and isinstance(rolls_b[0].frames, bdrv.NativeFrameArena)
    ra = run_python(book_a, rolls_a, steps, nh, PRIV_DTYPE, 5)
    rb = run_native(d, book_b, rolls_b, steps, nh, PRIV_DTYPE, 5)
    same_rollout(rolls_a[0], rolls_b[0])   # (the first rollout as handed off: carried from, then left alone)
    same_rollout(ra, rb)
    for name in ('ret', 'hurt', 'damage', 'rooms', 'boss', 'extra', 'gains', 'healed', 'gathered', 'v0', 'gret', 'steps'):
        np.testing.assert_array_equal(getattr(book_a, name), getattr(book_b, name), err_msg=name)
    np.testing.assert_array_equal(book_a.pick['steps'], book_b.pick['steps'])
    assert book_a.finished == book_b.finished and len(book_a.finished) > 5
    assert book_a.branch_finished == book_b.branch_finished and len(book_a.branch_finished) > 0
    assert book_a.inv0 == book_b.inv0 and book_a.stats0 == book_b.stats0 and book_a.char0 == book_b.char0
    assert book_a.active0 == book_b.active0 and book_a.pick['bped'] == book_b.pick['bped'] and book_a.seen_inv == book_b.seen_inv
    np.testing.assert_array_equal(stats_a, stats_b)
    st = actor.store.native.stats()
    assert st['reallocs'] == 0 and st['cap_pages_d'] >= st['kv_pages']
