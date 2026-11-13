"""An episode that breaks off (its game or worker fails: the next record of the worker is a new episode's first while
a decision is still open) is a truncation, not a death (2026-11-12, the memory-limit test on gpu57): the open decision
only bootstraps the decision before it (boot, not terminal, never trained), its accumulators go to a line of their own
(done DONE_BROKE, broken 1) and never into the next episode's; a normal end and a branch's truncated end are as before.
The Python consumer here (Episodes.arrive); the native one (abp_core.h consume) is held to it byte for byte by
test_b58_native_consumer (its record stream breaks episodes too)."""
import numpy as np

import train_tok as T
from isaac_bridge.tok_obs import DONE_BRANCH, ROW

RW = dict(damage=1.0, hurt=0.75, room=1.0, explore=0.5, boss=2.0, exit=3.0, death=2.0, timeout=1.0, time=0.002,
          heal=0.0, resource=0.0, clear=1.0)
GAMMA, LAM = 0.9, 0.95


def _row(first=0, done=0, damage=0.0, seed=7, branch=0, t=0):
    r = np.zeros(1, ROW)
    r['t'] = t
    r['first'], r['done'], r['damage'], r['seed'], r['group'], r['branch'] = first, done, damage, seed, 3, branch
    return r


class Run:
    """One worker: records in, decisions opened as the store would (value v each), then gae."""

    def __init__(self):
        self.roll = T.Rollout(1, 32, close=True)
        self.book = T.Episodes(1, RW, True, run=True, items=True)
        self.pending = np.zeros((1, 3), np.int64)

    def rec(self, value, **kw):
        rows = _row(**kw)
        go = self.book.arrive(self.roll, rows, np.array([0]), self.pending)
        if go[0]:   # (the store: a new open decision with this value; DONE_BRANCH's too, as a boot)
            k = int(self.roll.length[0])
            self.roll.value[0, k] = value
            self.roll.boot[0, k] = (rows['done'][0] == DONE_BRANCH) | T.is_trunc(rows)[0]
            self.roll.open[0], self.roll.length[0] = k, k + 1

    def gae(self):
        return T.gae(self.roll, GAMMA, LAM)


def test_broken_episode_bootstraps_not_a_death():
    r = Run()
    r.rec(0.5, first=1)
    r.rec(0.6, damage=1.0)
    r.rec(0.7, damage=2.0)            # (decision 2 open when the game dies)
    r.rec(0.1, first=1, seed=8)       # (the worker's next record: a new episode's first)
    adv, ret, mask = r.gae()
    ro = r.roll
    assert ro.boot[0, 2] and not ro.terminal[0, 2] and ro.reward[0, 2] == 0.0
    assert not mask[0, 2]                                   # (never trained)
    assert mask[0, 1] and mask[0, 0]
    # decision 1's target: its reward + gamma * the broken decision's value (0.7) -- not reward + 0
    assert np.isclose(ret[0, 1], ro.reward[0, 1] + GAMMA * 0.7)
    assert ro.close_done[0, 2] == T.CLOSE_BROKE


def test_broken_accumulators_their_own_line_not_the_next_episode():
    r = Run()
    r.rec(0.5, first=1, seed=7)
    r.rec(0.6, damage=1.0)
    r.rec(0.7, damage=2.0)
    r.rec(0.1, first=1, seed=8)       # (broke off: 3 damage so far)
    r.rec(0.2, damage=0.5, seed=8)
    r.rec(0.3, damage=0.25, done=2, seed=8)   # (the next episode dies normally)
    b = r.book
    assert len(b.broken) == 1 and b.broken_total == 1
    e = b.broken[0]
    assert e['done'] == T.DONE_BROKE and e['broken'] == 1 and e['seed'] == 7 and e['group'] == 3
    assert e['damage'] == 3.0
    assert len(b.finished) == 1
    f = b.finished[0]
    assert f['seed'] == 8 and f['damage'] == 0.75 and 'broken' not in f   # (only its own)


def test_normal_end_and_branch_truncation_unchanged():
    """The two normal paths: what the code before this change did, value by value."""
    r = Run()
    r.rec(0.5, first=1)
    r.rec(0.6, damage=1.0, done=DONE_BRANCH, branch=1)   # (a branch's truncated end: a boot decision opened)
    r.rec(0.1, first=1, seed=8)       # (closes the boot decision as always: reward 0, terminal, no broken line)
    r.rec(0.2, damage=0.5, done=2)    # (a death)
    ro, b = r.roll, r.book
    assert ro.boot[0, 1] and ro.terminal[0, 1] and ro.reward[0, 1] == 0.0 and ro.close_done[0, 1] == T.CLOSE_BROKE
    assert not ro.boot[0, 2] and ro.terminal[0, 2]
    assert b.broken == [] and b.broken_total == 0
    assert len(b.branch_finished) == 1 and len(b.finished) == 1
    adv, ret, mask = r.gae()
    assert mask[0, 0] and not mask[0, 1] and mask[0, 2]
    assert np.isclose(ret[0, 0], ro.reward[0, 0] + GAMMA * 0.6)      # (bootstraps from the boot decision's value)
    assert np.isclose(ret[0, 2], ro.reward[0, 2])                    # (terminal)


def test_episode_start_from_the_workers_column():
    """(2026-11-12) episodes.jsonl 'start': the worker's ep_archive stats column at the episode's first record."""
    r = Run()
    origin = np.array([1.0])
    r.book.origin = origin
    r.rec(0.5, first=1, seed=7)
    origin[0] = 0.0   # (the worker's next episode, a floor start: written before its first record)
    r.rec(0.6, damage=1.0, done=2)
    r.rec(0.1, first=1, seed=8)
    r.rec(0.2, done=2, seed=8)
    assert [e['start'] for e in r.book.finished] == ['archive', 'fresh']
