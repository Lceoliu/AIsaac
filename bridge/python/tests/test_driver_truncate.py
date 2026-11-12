"""--cap-truncate / --stall-truncate on the C++ actor driver's path (2026-11-12, the coordinator: 5ee81f3 / 6f8e099's
driver part, ActorDriver.set_cap and the evaluate decision's test, had never been compiled -- the build failed and every
test of it skipped). One worker, a fixed script of episodes ending by the time cap (done 3 at t >= CAP), by the stall
rule (done 3 before it), by death, by the floor's end, as a branch's truncated end (DONE_BRANCH); TRUNC 0-3:

1. the driver without the native consumer (train_tok's Python-consumer path): every record's go mask (the end records
   evaluated or not) equals Episodes.arrive's on the same rows, with train_tok.CAP_T / TRUNC set alike -- as train_tok
   checks it in the run ('the go mask differs from Episodes.arrive');
2. the driver with the native consumer (the training runs' path): the rollout's decisions are the go rows; a truncated
   end is a boot decision the one before bootstraps from (not terminal), any other end makes the decision before it
   terminal (DONE_BRANCH: boot as always); the episode summaries mark 'cap' / 'stall'.

Not skipped on the gate: gate.sh fails a run whose native component's tests skip (native_check.py)."""
import time

import numpy as np
import pytest
import torch

import train_tok as T
from test_block_memory import make_policy
from test_b52_cpp_actor import EXT, ACTOR, FakeWorker, fake_sampler, frame, make_driver_for, collect_records
import test_b58e_batch as B
from isaac_bridge import block_driver as bdrv
from isaac_bridge.block_paged import SlotBlockActor
from isaac_bridge.tok_obs import ROW, DONE_BRANCH

pytestmark = pytest.mark.skipif(EXT is None, reason='cpp actor extension unavailable')

CAP = 6
# (length, the end record's done): the end record's t = length - 1
EPISODES = [(7, 3), (4, 3), (5, 2), (9, 3), (3, 3), (4, DONE_BRANCH), (6, 1), (2, 3)]


def script():
    rng = np.random.default_rng(77)
    rows, k = {}, 0
    for e, (n, end) in enumerate(EPISODES):
        for t in range(n):
            r = frame(rng, 1, t, np.array([e + 1]), np.array([int(t == 0)]), seed=900 + k)
            r['done'] = end if t == n - 1 else 0
            r['group'] = 0; r['seed'] = 1000; r['lab'] = 0; r['point'] = 0
            r['branch'] = 1 if r['done'][0] == DONE_BRANCH else 0
            r['damage'] = np.float32(0.5); r['hurt'] = np.float32(0.0)
            rows[0, k] = r
            k += 1
    return rows, k


def truncated(done, t, trunc):
    return done == 3 and ((t >= CAP and bool(trunc & 1)) or (t < CAP and bool(trunc & 2)))


def want_go(rows, K, trunc):
    out = []
    for k in range(K):
        d, t = int(rows[0, k]['done'][0]), int(rows[0, k]['t'][0])
        out.append(d == 0 or d == DONE_BRANCH or truncated(d, t, trunc))
    return np.array(out)


@pytest.mark.parametrize('trunc', [0, 1, 2, 3])
def test_driver_go_matches_episodes_arrive(trunc, monkeypatch):
    monkeypatch.setattr(T, 'CAP_T', CAP)
    monkeypatch.setattr(T, 'TRUNC', trunc)
    torch.manual_seed(51)
    m = make_policy('cuda').eval()
    rows, K = script()
    actor = SlotBlockActor(m, 1, graph_entities=128, bf16=True, lane_slots=0)
    s = fake_sampler(1, len(m.heads))
    d = make_driver_for(actor, s, 5)
    d.set_fixed_uniform(0.5); d.set_version(3)
    d.set_cap(CAP, trunc)
    ACTOR[0] = actor
    wk = FakeWorker(0, s._pairs[0][1], s, lambda w_, k_: rows[w_, k_], K)
    d.start(); wk.start()
    try:
        recs = collect_records(d, K)
        wk.join(10); assert wk.error is None
    finally:
        d.pause(); d.stop()
    assert d.last_error() == ''
    book = B.make_book(1)
    roll = B.make_rolls(1, len(m.heads))[0]
    pending = np.zeros((1, len(m.heads)), np.int64)
    got = []
    for rec in recs:
        idx = rec['workers'].numpy(); go_r = rec['go'].numpy().astype(bool)
        rr = bdrv.record_rows(rec, ROW)
        go_b = book.arrive(roll, rr, idx, pending)
        np.testing.assert_array_equal(go_r, go_b)
        got.append(bool(go_r[0]))
    np.testing.assert_array_equal(np.array(got), want_go(rows, K, trunc))
    n_trunc = sum(truncated(int(rows[0, k]['done'][0]), int(rows[0, k]['t'][0]), trunc) for k in range(K))
    assert n_trunc == {0: 0, 1: 2, 2: 3, 3: 5}[trunc]


@pytest.mark.parametrize('trunc', [0, 1, 2, 3])
def test_driver_native_consumer_truncated_ends(trunc, monkeypatch):
    monkeypatch.setattr(T, 'CAP_T', CAP)
    monkeypatch.setattr(T, 'TRUNC', trunc)

    def setup(m, W, seed=5):   # (test_b58e_batch.setup with the cap and TRUNC on the driver and its native consumer)
        actor = SlotBlockActor(m, W, graph_entities=128, bf16=True, lane_slots=0)
        s = fake_sampler(W, len(m.heads))
        d = make_driver_for(actor, s, seed)
        d.set_fixed_uniform(0.5); d.set_version(3)
        d.set_cap(CAP, trunc)
        book, rolls = B.make_book(W), B.make_rolls(W, len(m.heads))
        bdrv.attach_consumer(d, book, rolls, 0.99, cap_t=CAP, trunc=trunc)
        d.use_rollout(rolls[0].native_slot)
        ACTOR[0] = actor
        return actor, s, d, book, rolls
    monkeypatch.setattr(B, 'setup', setup)
    torch.manual_seed(52)
    m = make_policy('cuda').eval()
    rows, K = script()
    ev, cnt, book, roll = B.run_per_record(m, 1, K, rows)
    go = want_go(rows, K, trunc)
    assert cnt['polls'] == K and cnt['rows'] == int(go.sum())
    assert int(roll.length[0]) == int(go.sum())
    dec = np.cumsum(go) - 1   # (each go row's decision index)
    for k in range(K):
        dn, t = int(rows[0, k]['done'][0]), int(rows[0, k]['t'][0])
        if dn == 0:
            continue
        before = dec[k - 1]   # (the decision the end record closes)
        if dn == DONE_BRANCH or truncated(dn, t, trunc):
            assert go[k] and bool(roll.boot[0, dec[k]]), (k, dn, t)
            assert not bool(roll.terminal[0, before]), (k, dn, t)
        else:
            assert not go[k] and bool(roll.terminal[0, before]), (k, dn, t)
    ends3 = [(int(rows[0, k]['t'][0]) >= CAP) for k in range(K) if int(rows[0, k]['done'][0]) == 3]
    fin = [f for f in book.finished if f.get('cap') or f.get('stall')]
    assert sorted(bool(f.get('cap')) for f in fin) == sorted(ends3)
