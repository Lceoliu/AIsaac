"""The long run's reward terms (2026-11-19, the user / the coordinator; train_tok NEW_TERMS, Episodes.arrive): item /
trinket (each id once per episode, the start build not rewarded), depth (the exit and boss terms x 1 + depth x
(n - 1)), nohit (a room cleared with no hurt in it over every visit). Each is 0 by default and then every reward is
the old one. The three ways to farm an item -- the same active put down and taken again, two actives swapped back and
forth, a trinket dropped and taken again -- pay once."""
import numpy as np
import pytest

import train_tok as T
from isaac_bridge.tok_obs import EV_CLEARED, EV_BOSS, EV_EXIT, ROW

OLD = dict(damage=1.0, hurt=0.75, room=1.0, explore=0.5, boss=2.0, exit=3.0, death=2.0, timeout=1.0, time=0.002,
           heal=0.0, resource=0.0, clear=1.0)


def _row(first=0, inv=(), active=0, trinkets=(0, 0), stage=1, cell=0, events=0, hurt=0.0, damage=0.0, done=0):
    r = np.zeros(1, ROW)
    r['first'], r['done'], r['seed'], r['group'] = first, done, 7, 0
    for k, v in enumerate(inv):
        r['inv'][0, k] = (v, 1)
    r['pitem'][0, 0] = active
    r['pitem'][0, 1], r['pitem'][0, 2] = trinkets
    r['stage'] = stage
    if cell is not None:
        r['map'][0, 3].reshape(-1)[cell] = 1
    r['events'], r['hurt'], r['damage'] = events, hurt, damage
    return r


class Run:
    def __init__(self, **new):
        self.rw = dict(OLD, **new)
        self.roll = T.Rollout(1, 64, close=True)
        self.book = T.Episodes(1, self.rw, True, run=True, items=True)
        self.pending = np.zeros((1, 3), np.int64)
        self.rewards = []

    def rec(self, **kw):
        rows = _row(**kw)
        k_before = int(self.roll.open[0])
        go = self.book.arrive(self.roll, rows, np.array([0]), self.pending)
        if k_before >= 0:
            self.rewards.append(float(self.roll.reward[0, k_before]))
        if go[0]:
            k = int(self.roll.length[0])
            self.roll.open[0], self.roll.length[0] = k, k + 1
        return self


def _old_reward(hurt=0.0, damage=0.0, events=0):
    """The reward before the new terms (Episodes.arrive's formula, a running record, run mode, no gains)."""
    rw = OLD
    r = rw['damage'] * damage - rw['hurt'] * hurt - rw['time']
    return (r + (rw['room'] * ((events & 1) > 0) + rw['explore'] * ((events & 2) > 0) + rw['boss'] * ((events & 4) > 0))
            + rw['exit'] * ((events & EV_EXIT) > 0))


def test_defaults_are_the_old_rewards_bit_for_bit():
    r = Run()
    assert not r.book.new_on
    steps = [dict(events=EV_CLEARED, damage=0.5), dict(hurt=1.0, cell=5), dict(events=EV_BOSS | EV_CLEARED, stage=3),
             dict(events=EV_EXIT, stage=4, inv=(9,)), dict(active=12, trinkets=(3, 0))]
    r.rec(first=1, inv=(1,))
    for s in steps:
        r.rec(**s)
    want = [_old_reward(s.get('hurt', 0.0), s.get('damage', 0.0), s.get('events', 0)) for s in steps]
    assert r.rewards == [float(np.float32(x)) for x in want]   # (exactly, as Rollout stores it)
    r.rec(done=2)
    assert 'r' not in r.book.finished[0] and 'items_new' not in r.book.finished[0]


def test_item_and_trinket_once_per_id_and_the_three_farming_ways():
    r = Run(item=1.0, trinket=0.3)
    r.rec(first=1, inv=(10,), active=0)                       # (the start build: not rewarded)
    r.rec(inv=(10, 20))                                       # a new passive: +1
    r.rec(inv=(10, 20), active=30)                            # an active taken: +1
    r.rec(inv=(10, 20), active=0)                             # put down ...
    r.rec(inv=(10, 20), active=30)                            # ... and taken again: 0
    r.rec(inv=(10, 20), active=31)                            # a second active (swap): +1
    r.rec(inv=(10, 20), active=30)                            # swapped back: 0
    r.rec(inv=(10, 20), active=31)                            # and again: 0
    r.rec(inv=(10, 20), active=31, trinkets=(5, 0))           # a trinket: +0.3
    r.rec(inv=(10, 20), active=31, trinkets=(0, 0))           # dropped ...
    r.rec(inv=(10, 20), active=31, trinkets=(5, 0))           # ... and taken again: 0
    r.rec(inv=(10, 21), active=31, trinkets=(5, 0))           # a reroll (20 -> 21): a new id, +1
    base = _old_reward()
    extra = [x - base for x in r.rewards]
    assert extra == pytest.approx([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3, 0.0, 0.0, 1.0], abs=1e-5)
    r.rec(done=2)
    f = r.book.finished[0]
    assert f['items_new'] == 4 and f['trinkets_new'] == 1 and f['r']['item'] == 4.0 and f['r']['trinket'] == 0.3


def test_item_set_starts_again_each_episode():
    r = Run(item=1.0)
    r.rec(first=1, inv=(10,))
    r.rec(inv=(10, 20))
    r.rec(done=2, inv=(10, 20))
    r.rec(first=1, inv=(10,))                                 # a new episode: 20 is new again
    r.rec(inv=(10, 20))
    assert r.rewards[-1] - _old_reward() == pytest.approx(1.0, abs=1e-5)


def test_depth_scales_exit_and_boss_by_the_floor():
    r = Run(depth=0.25)
    r.rec(first=1, stage=1)
    r.rec(stage=1, events=EV_BOSS)                            # floor 1: unchanged
    r.rec(stage=2, events=EV_EXIT)                            # left floor 1: unchanged
    r.rec(stage=3, events=EV_BOSS)                            # floor 3: boss x (1 + 0.25 x 2)
    r.rec(stage=4, events=EV_EXIT)                            # left floor 3: exit x (1 + 0.25 x 2)
    ex = [x - _old_reward(events=e) for x, e in zip(r.rewards, (EV_BOSS, EV_EXIT, EV_BOSS, EV_EXIT))]
    assert ex == pytest.approx([0.0, 0.0, 2.0 * 0.5, 3.0 * 0.5], abs=1e-5)


def test_nohit_per_room_over_every_visit():
    r = Run(nohit=0.5)
    r.rec(first=1, cell=0)
    r.rec(cell=0, events=EV_CLEARED)                          # room 0 cleared, never hurt: +0.5
    r.rec(cell=1, hurt=1.0)                                   # hurt in room 1
    r.rec(cell=2)                                             # out ...
    r.rec(cell=1)                                             # ... and back in
    r.rec(cell=1, events=EV_CLEARED)                          # cleared: 0 (hurt there before)
    r.rec(cell=2, events=EV_CLEARED | EV_BOSS)                # a boss room, never hurt in: +0.5
    r.rec(cell=None, events=EV_CLEARED)                       # no room known: 0
    r.rec(stage=2, cell=1, events=EV_CLEARED)                 # the next floor's room at the same cell: a new room
    olds = [_old_reward(events=EV_CLEARED), _old_reward(hurt=1.0), _old_reward(), _old_reward(),
            _old_reward(events=EV_CLEARED), _old_reward(events=EV_CLEARED | EV_BOSS), _old_reward(events=EV_CLEARED),
            _old_reward(events=EV_CLEARED)]
    assert [a - b for a, b in zip(r.rewards, olds)] == pytest.approx([0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5], abs=1e-5)
    r.rec(done=2)
    assert r.book.finished[0]['nohit_rooms'] == 3


def test_new_terms_refused_outside_floor_modes(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv('ISAAC_RL_VISIBLE_V3', raising=False)
    # (with --block-memory refused too; an earlier block-memory check stops a bare command line before it)
    for i, (extra, msg) in enumerate(((['--reward', 'item=1'], 'floor or run only'),
                                      (['--reward', 'depth=0.25,nohit=0.5'], 'floor or run only'))):
        with pytest.raises(SystemExit):
            T.main(['--groups-file', str(tmp_path / 'x.json'), '--out', str(tmp_path / f'o{i}')] + extra)
        assert msg in capsys.readouterr().err
