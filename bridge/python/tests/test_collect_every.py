"""ISAAC_BLOCK_COLLECT_EVERY (train_tok.COLLECT_EVERY): the page store collected every N-th hand-off instead of every
one only changes when dead page rows are freed and their ids reused -- never a value. The same rollout (an episode
reset, teacher records kept a while and then let go, so rows die and ids get reused) is built twice, the store
collected after every step and after every fourth: every action / value the actor computed, every rollout array, and
the learner's outputs and gradients on the chunks and the teacher records still held are equal bit for bit."""
import gc

import numpy as np
import pytest
import torch

from test_block_slot_integration import pair, frame, DEV
from test_visible_input import synth_pending
from test_block_packed_learner import _run, _grads
from isaac_bridge.block_training import Chunk
from train_tok import Rollout, gae, store


def _build(device, every, seed=31):
    m, ref, new, old = pair(device, slots=3, seed=seed)
    rng = np.random.default_rng(seed)
    roll = Rollout(3, 64, 5, value_g=True, block_steps=7)
    episodes = np.array([1, 1, 1])
    teach, outs, collects = [], [], 0
    for t in range(48):
        first = np.array([int(t == 0)] * 3)
        if t in (11, 23, 35):
            first[t % 3] = 1
            episodes[t % 3] += 1
        rows = frame(rng, 3, t, episodes, first)
        pending = synth_pending(rng, 3, m.heads)
        if t >= 6 and t % 3 == 0:
            for i in range(3):
                teach.append(Chunk(rows[i:i + 1].copy(), pending[i:i + 1].copy(), new.snapshot(i)))
            teach = teach[-9:]   # (a small ring: older records let go, their rows die)
        result = new.step(rows, pending, [0, 1, 2], ukeys=(np.full(3, t), np.arange(3)))
        action, lp, v, vg = [x.float().cpu().numpy() for x in result]
        outs.append((action.copy(), lp.copy(), v.copy(), vg.copy()))
        store(roll, np.arange(3), rows, pending, action.astype(np.int64), lp, v, 0, vg=vg,
              block_context=new.last_contexts)
        roll.reward[:, t] = .1
        if (t + 1) % every == 0:   # (a hand-off's collect)
            gc.collect()
            new.store.collect()
            collects += 1
    _, _, mask = gae(roll, .99, .95)
    plan = roll.block.plan(roll, mask)
    return m, new, roll, outs, plan['chunks'], teach, collects


@pytest.mark.parametrize('device', DEV)
def test_collect_every_four_changes_no_value(device, monkeypatch):
    # (one CPU thread: two identical builds' backward is then bitwise reproducible -- with several, the scatter-adds of
    #  memory_age / short_relative accumulate in another order run to run, ~5e-10, whatever the collects)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _check(device, monkeypatch)
    finally:
        torch.set_num_threads(n_threads)


def _check(device, monkeypatch):
    a = _build(device, 1)
    b = _build(device, 4)
    assert a[6] == 48 and b[6] == 12
    for (x, y) in zip(a[3], b[3]):   # the actor's actions, log-probs, values, global values, step by step
        for p, q in zip(x, y):
            np.testing.assert_array_equal(p, q)
    for name in ('actions', 'logp', 'value', 'value_g', 'length', 'open', 'boot'):
        np.testing.assert_array_equal(getattr(a[2], name), getattr(b[2], name), err_msg=name)
    assert a[1].store.freed['kv'] > 0   # (rows did die and were freed: ids got reused in between)
    a[0].train(); b[0].train()
    (oa, _) = _run(a[0], a[4] + a[5], True, monkeypatch)
    sum(x.float().sum() for x in oa if x.is_floating_point()).backward()
    oa = [x.detach() for x in oa]
    ga = _grads(a[0])
    (ob, _) = _run(b[0], b[4] + b[5], True, monkeypatch)
    sum(x.float().sum() for x in ob if x.is_floating_point()).backward()
    gb = _grads(b[0])
    assert len(oa) >= 2 and len(oa) == len(ob)
    for x, y in zip(oa, ob):
        torch.testing.assert_close(x, y.detach(), rtol=0, atol=0)
    assert ga.keys() == gb.keys() and len(ga) > 10
    for k in ga:
        torch.testing.assert_close(ga[k], gb[k], rtol=0, atol=0, msg=k)
    a[1].store.collect(); b[1].store.collect()   # (after a final collect both stores hold the same live rows)
    assert int(a[1].store.d_used.sum()) == int(b[1].store.d_used.sum())
    assert int(a[1].store.b_used.sum()) == int(b[1].store.b_used.sum())
