"""--cap-truncate (train_tok.CAP_T, is_cap): an episode ended by the time cap (done 3 at t >= the cap) is a
truncation -- no timeout penalty, not terminal, its last record evaluated as a boot decision the one before
bootstraps from (as a branch's truncated end); a stall (done 3 before the cap) stays a terminal timeout; off, both
are terminal timeouts as before. The native consumer is held to the same by test_b58_native_consumer (cap 6)."""
import numpy as np
import pytest

import train_tok as T
from test_broken_episode import GAMMA, RW, Run


def _episode(r, end_t):
    r.rec(0.5, first=1, t=0)
    r.rec(0.6, damage=1.0, t=1)
    r.rec(0.8, done=3, t=end_t)   # (the end: the cap at t 10, a stall before it)
    r.rec(0.1, first=1, seed=8, t=0)   # (the next episode)


@pytest.mark.parametrize('cap_on', [0, 1])
def test_time_cap_truncation_and_stall(cap_on, monkeypatch):
    monkeypatch.setattr(T, 'CAP_T', 10 if cap_on else 0)
    for end_t, is_cap in ((10, True), (5, False)):
        r = Run()
        _episode(r, end_t)
        ro = r.roll
        trunc = cap_on and is_cap
        # decision 1 (closed by the end record): the timeout penalty and terminal unless truncated
        base = RW['damage'] * 0 - RW['time']
        assert np.isclose(ro.reward[0, 1], base if trunc else base - RW['timeout'])
        assert bool(ro.terminal[0, 1]) == (not trunc)
        adv, ret, mask = r.gae()
        if trunc:   # the end record evaluated: a boot decision (value 0.8), the one before bootstraps from it
            assert ro.length[0] == 4 and ro.boot[0, 2] and not mask[0, 2]
            assert np.isclose(ret[0, 1], ro.reward[0, 1] + GAMMA * 0.8)
            assert r.book.finished[0].get('cap') == 1
        else:
            assert ro.length[0] == 3 and np.isclose(ret[0, 1], ro.reward[0, 1])
            assert 'cap' not in r.book.finished[0]
        assert r.book.broken == []   # (the boot decision is closed as a truncation, not a broken episode)
