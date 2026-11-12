"""--cap-truncate / --stall-truncate (train_tok.CAP_T, TRUNC, is_cap, is_trunc): an episode ended by the time cap
(done 3 at t >= the cap) or by the floor's stall rule (done 3 before it) is, when its switch is on, a truncation -- no
timeout penalty, not terminal, its last record evaluated as a boot decision the one before bootstraps from (as a
branch's truncated end); with its switch off a terminal timeout as before. The ends are marked 'cap' / 'stall' in the
episode summaries either way. The native consumer is held to the same by test_b58_native_consumer (cap 6, TRUNC 0-3)."""
import numpy as np
import pytest

import train_tok as T
from test_broken_episode import GAMMA, RW, Run


def _episode(r, end_t):
    r.rec(0.5, first=1, t=0)
    r.rec(0.6, damage=1.0, t=1)
    r.rec(0.8, done=3, t=end_t)        # (the end: the cap at t 10, a stall before it)
    r.rec(0.1, first=1, seed=8, t=0)   # (the next episode)


@pytest.mark.parametrize('trunc', [0, 1, 2, 3])
def test_cap_and_stall_ends(trunc, monkeypatch):
    monkeypatch.setattr(T, 'CAP_T', 10)
    monkeypatch.setattr(T, 'TRUNC', trunc)
    for end_t, kind in ((10, 'cap'), (5, 'stall')):
        r = Run()
        _episode(r, end_t)
        ro = r.roll
        truncated = bool(trunc & (1 if kind == 'cap' else 2))
        base = -RW['time']
        assert np.isclose(ro.reward[0, 1], base if truncated else base - RW['timeout'])
        assert bool(ro.terminal[0, 1]) == (not truncated)
        adv, ret, mask = r.gae()
        if truncated:   # the end record evaluated: a boot decision (value 0.8), the one before bootstraps from it
            assert ro.length[0] == 4 and ro.boot[0, 2] and not mask[0, 2]
            assert np.isclose(ret[0, 1], ro.reward[0, 1] + GAMMA * 0.8)
        else:
            assert ro.length[0] == 3 and np.isclose(ret[0, 1], ro.reward[0, 1])
        e = r.book.finished[0]
        assert e.get(kind) == 1 and not e.get('stall' if kind == 'cap' else 'cap')
        assert r.book.broken == []


def test_outside_floor_modes_nothing(monkeypatch):
    monkeypatch.setattr(T, 'CAP_T', 0)
    monkeypatch.setattr(T, 'TRUNC', 3)
    r = Run()
    _episode(r, 10)
    assert r.roll.terminal[0, 1] and 'cap' not in r.book.finished[0] and 'stall' not in r.book.finished[0]
