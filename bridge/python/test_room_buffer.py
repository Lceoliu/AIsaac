"""C39 Room Buffer (user design 2026-09-29): priorities, EMAs and priors, admission, the fresh share of the steps, the
workers' draw and checkpoints. No game instance or GPU needed."""
import json
import unittest

import numpy as np


def buffer(**kw):
    from isaac_bridge.room_buffer import RoomBuffer
    return RoomBuffer(['a', 'b'], **{'capacity': 4, **kw})


class PriorityTest(unittest.TestCase):
    def test_user_formula(self):
        b = buffer(lam=0.5, d0=2.0, eta=0.1, eps=0.02)
        self.assertAlmostEqual(b.value(0.5, 0.0), 1.0)                  # the learning frontier
        self.assertAlmostEqual(b.value(1.0, 4.0), 0.5)                  # always cleared, 2 hearts lost: lam p clip = 0.5
        self.assertAlmostEqual(b.value(1.0, 1.0), 0.25)
        self.assertAlmostEqual(b.value(0.0, 4.0), 0.0)                  # never cleared
        b.observe(0, 10, True, 0.0, 100, 'r1', False)
        P = b.priorities(0)
        p, d = b.p[0, 0], b.d[0, 0]
        self.assertAlmostEqual(P[0], 4 * p * (1 - p) + 0.5 * p * min(1.0, d / 2.0) + 0.0 + 0.02)
        self.assertEqual(P[1:].tolist(), [0.0, 0.0, 0.0])               # empty slots

    def test_staleness_grows_with_the_groups_episodes(self):
        b = buffer(capacity=4, eta=0.1, eps=0.02)
        b.observe(0, 10, True, 0.0, 100, 'r1', False)
        before = b.priorities(0)[0]
        for seed in (11, 12):
            b.observe(0, seed, False, 0.0, 100, 'r1', False)
        self.assertAlmostEqual(b.priorities(0)[0] - before, 0.1 * 2 / 4)
        for seed in (13, 14, 15, 16):
            b.observe(0, seed, False, 0.0, 100, 'r1', False)
        slot = b.where[0].get(10)
        if slot is not None:                                             # capped at eta once a capacity of episodes passed
            self.assertLessEqual(b.priorities(0)[slot], b.value(b.p[0, slot], b.d[0, slot]) + 0.1 + 0.02 + 1e-12)


class StatisticsTest(unittest.TestCase):
    def test_ema_and_damage_only_on_clears(self):
        b = buffer(alpha=0.25)
        b.observe(0, 10, True, 2.0, 80, 'r1', False)                    # first: from the default prior (0.5, d0, -)
        slot = b.where[0][10]
        self.assertAlmostEqual(b.p[0, slot], 0.5 + 0.25 * 0.5)
        self.assertAlmostEqual(b.d[0, slot], 2.0)
        self.assertEqual(b.L[0, slot], 80)
        b.observe(0, 10, False, 6.0, 40, 'r1', True)                    # a loss: d unchanged
        self.assertAlmostEqual(b.p[0, slot], 0.625 * 0.75)
        self.assertAlmostEqual(b.d[0, slot], 2.0)
        self.assertAlmostEqual(b.L[0, slot], 80 + 0.25 * (40 - 80))
        b.observe(0, 10, True, 0.0, 40, 'r1', True)
        self.assertAlmostEqual(b.d[0, slot], 2.0 * 0.75)
        self.assertEqual(b.plays[0, slot], 3)

    def test_prior_is_the_family_then_the_group(self):
        b = buffer(capacity=50)
        for seed in range(3):                                             # family r1: always cleared, 20 steps
            b.observe(0, seed, True, 1.0, 20, 'r1', False)
        p, d, L = b.prior(0, 'r1')
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(d, 1.0)
        self.assertAlmostEqual(L, 20.0)
        self.assertEqual(b.prior(0, 'r2'), (b.group[0]['p'], b.group[0]['d'], b.group[0]['L']))   # unknown family
        self.assertEqual(b.prior(1, 'r1'), (0.5, 2.0, 1.0))               # nothing in group b yet
        b.observe(0, 99, False, 0.0, 30, 'r1', False)                     # a new seed of r1 starts from r1's EMAs
        slot = b.where[0][99]
        self.assertAlmostEqual(b.p[0, slot], 1.0 + 0.25 * (0.0 - 1.0))

    def test_admission_replaces_the_lowest_value(self):
        b = buffer(capacity=2)
        b.observe(0, 1, True, 0.0, 10, 'r1', False)
        b.observe(0, 2, False, 0.0, 10, 'r2', False)                     # a frontier seed (p 0.75)
        for _ in range(12):                                               # seed 1 becomes solved: value -> 0
            b.observe(0, 1, True, 0.0, 10, 'r1', True)
        b.observe(0, 3, False, 0.0, 10, 'r3', False)                      # a new frontier-ish seed
        self.assertIn(3, b.where[0])
        self.assertNotIn(1, b.where[0])
        self.assertIn(2, b.where[0])
        self.assertEqual(int(b.evicted[0]), 1)


class TableTest(unittest.TestCase):
    def test_start_probability_and_fresh_share_by_steps(self):
        b = buffer(capacity=4, fresh_share=0.3, eta=0.0, eps=0.02)
        b.observe(0, 1, True, 0.0, 30, 'r1', False)
        b.observe(0, 2, True, 0.0, 10, 'r2', False)
        seeds, probs, fresh = b.table()
        P = b.priorities(0)
        lengths = b.lengths(0)
        w = P[:2] / lengths[:2]
        np.testing.assert_allclose(probs[0, :2], w / w.sum())
        self.assertEqual(probs[0, 2:].tolist(), [0.0, 0.0])
        replay_length = (probs[0, :2] * lengths[:2]).sum()                # no replay finished yet: the model's mean
        q = fresh[0]
        share = q * b.fresh_L[0] / (q * b.fresh_L[0] + (1 - q) * replay_length)
        self.assertAlmostEqual(share, 0.3)
        self.assertEqual(fresh[1], 1.0)                                   # an empty group plays fresh seeds only
        self.assertEqual(seeds[0, :2].tolist(), [1, 2])
        # Once replays finished, their realised mean length sets the fresh probability.
        b.observe(0, 1, False, 0.0, 200, 'r1', True)
        q = b.table()[2][0]
        share = q * b.fresh_L[0] / (q * b.fresh_L[0] + (1 - q) * b.replay_L[0])
        self.assertAlmostEqual(b.replay_L[0], 200.0)
        self.assertAlmostEqual(share, 0.3)

    def test_lengths_shrink_towards_the_family(self):
        from isaac_bridge.room_buffer import L_SHRINK
        b = buffer(capacity=10)
        for seed in range(4):                                             # family r1: 100 steps
            b.observe(0, seed, False, 0.0, 100, 'r1', False)
        b.observe(0, 50, False, 0.0, 10, 'r1', False)                     # one short episode of a new r1 seed
        slot = b.where[0][50]
        fam = b.prior(0, 'r1')[2]
        expected = (1 * b.L[0, slot] + L_SHRINK * fam) / (1 + L_SHRINK)
        self.assertAlmostEqual(b.lengths(0)[slot], expected)
        self.assertGreater(b.lengths(0)[slot], b.L[0, slot])

    def test_worker_draw_and_fresh_budget(self):
        """Each slot splits a group's steps between fresh and replayed seeds by its own step budget (30% fresh here),
        whatever the lengths: fresh episodes of 10 steps next to replays of 100 still take 30% of the steps."""
        from isaac_bridge.abplus_groups import GroupScheduler
        from isaac_bridge.abplus_worker import Worker
        seeds = np.array([[7, 8, -1], [-1, -1, -1]], np.int64)
        probs = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        w = Worker.__new__(Worker)
        w.index, w.config = 0, dict(base_seed=1000, num_envs=4, start_randomization=None, start_bombs=None)
        w.scheduler = GroupScheduler([0.5, 0.5], 'slots', 0)            # always group 0
        w.buffer = (seeds, probs, np.array([0.3, 1.0]))
        w.kinds = [GroupScheduler([0.3, 0.7], 'steps') for _ in range(2)]
        steps = {0: 0, 1: 0}
        for k in range(400):
            seed, _, _, group, replay = w.schedule(k)
            self.assertEqual(group, 0)
            self.assertIn(replay, (0, 1))
            self.assertEqual(seed, 8 if replay == 1 else 1000 + 4 * k)   # replays draw seed 8, fresh ones the slot's own
            length = 100 if replay else 10
            w.kinds[group].start(replay)
            w.kinds[group].finish(replay, length)
            steps[replay] += length
        self.assertAlmostEqual(steps[0] / (steps[0] + steps[1]), 0.3, delta=0.01)
        # An empty group plays fresh seeds outside the budget (-1), which stays untouched.
        w.scheduler = GroupScheduler([0.5, 0.5], 'slots', 1)
        self.assertEqual({w.schedule(k)[4] for k in range(20)}, {-1})
        self.assertEqual(w.kinds[1].decisions.tolist(), [0.0, 0.0])


class CheckpointTest(unittest.TestCase):
    def test_round_trip(self):
        from isaac_bridge.room_buffer import RoomBuffer
        b = buffer(capacity=3)
        for seed, win in ((1, True), (2, False), (3, True), (4, False)):
            b.observe(seed % 2, seed, win, 1.0, 12, f'r{seed}', False)
        state = json.loads(json.dumps(b.state_dict()))
        c = RoomBuffer.from_state(state)
        for key in ('seeds', 'p', 'd', 'L', 'plays', 'last', 'episodes'):
            np.testing.assert_array_equal(getattr(c, key), getattr(b, key))
        self.assertEqual(c.where, b.where)
        np.testing.assert_allclose(c.table()[1], b.table()[1])
        with self.assertRaises(ValueError):
            RoomBuffer.from_state(state, capacity=5)

    def test_family_key(self):
        from isaac_bridge.abplus_tasks import target_arm
        from isaac_bridge.room_buffer import family_key
        arms = [{'rooms': [1]}, {'rooms': [1]}, {'rooms': [1]}]
        self.assertEqual(family_key({'target': None}, 5, 77), '77')
        self.assertEqual(family_key({'target': {'arms': arms}}, 5, 77), f'77:{target_arm(5, arms)}')


if __name__ == '__main__':
    unittest.main()
