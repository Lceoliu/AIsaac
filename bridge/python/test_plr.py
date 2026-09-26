"""Prioritized Level Replay (isaac_bridge/plr.py) and the worker's start/room helpers; no engine needed."""
import unittest

import numpy as np

from isaac_bridge import abplus_worker as W
from isaac_bridge.plr import PrioritizedLevels, mixture_levels

LEVELS = [('arena', 0), ('normal', 11), ('normal', 12), ('boss', 21)]
WEIGHTS = {'arena': 0.2, 'normal': 0.45, 'boss': 0.35}
ROOMS = [('arena', 0), ('normal', 1), ('normal', 2), ('normal', 3), ('normal', 4), ('boss', 7), ('boss', 8)]


def shares(plr, p):
    return {kind: float(p[rooms].sum()) for kind, rooms in plr.groups.items()}


class PrioritizedLevelsTest(unittest.TestCase):
    def test_score_is_the_mean_positive_advantage_of_each_episode(self):
        plr = PrioritizedLevels(LEVELS, WEIGHTS)
        adv = np.array([[1.0, -2.0], [-1.0, 4.0], [3.0, 0.0], [5.0, 2.0]])
        starts = np.array([[1, 1], [0, 0], [1, 0], [0, 0]])
        levels = np.array([[1, 3], [1, 3], [2, 3], [2, 3]])
        plr.update(adv, starts, levels)
        # env 0: level 1 finished with (1 + 0) / 2; level 2 is still running.
        self.assertAlmostEqual(plr.scores[1], 0.5)
        self.assertEqual(plr.seen.tolist(), [False, True, False, False])
        # Next rollout: env 0's level 2 ends after one more step, env 1's level 3 after one.
        plr.update(np.array([[1.0, -1.0], [9.0, 9.0]]), np.array([[0, 0], [1, 1]]), np.array([[2, 3], [0, 1]]))
        self.assertAlmostEqual(plr.scores[2], (3 + 5 + 1) / 3)
        self.assertAlmostEqual(plr.scores[3], (0 + 4 + 0 + 2 + 0) / 5)
        self.assertEqual(plr.finished, 3)

    def test_distribution_ranks_scores_within_a_kind_tries_unseen_rooms_and_keeps_a_floor(self):
        plr = PrioritizedLevels(ROOMS, WEIGHTS, beta=1.0, staleness=0.0, floor=0.1)
        p = plr.probabilities()
        # Nothing seen: each kind's weight spread uniformly over its rooms (the fixed mixture).
        self.assertTrue(np.allclose(p, [0.2] + [0.45 / 4] * 4 + [0.35 / 2] * 2))
        for level, score in ((1, 3.0), (2, 1.0)):
            plr._finish(level, score)
        p = plr.probabilities()
        self.assertAlmostEqual(p.sum(), 1.0)
        self.assertGreater(p[1], p[2])                            # higher score, higher rank
        self.assertAlmostEqual(p[3], p[4])                        # the unseen share is uniform
        # half the normal rooms are unseen: 0.9 * 0.5 / 2 + 0.1 / 4 of the normal weight
        self.assertAlmostEqual(p[3], 0.45 * (0.9 * 0.5 / 2 + 0.1 / 4))
        self.assertAlmostEqual(p[0], 0.2)                         # other kinds are untouched
        self.assertTrue(np.allclose(p[5:], 0.35 / 2))
        for level in (3, 4):
            plr._finish(level, 0.0)
        self.assertGreaterEqual(plr.probabilities()[1:5].min(), 0.45 * 0.1 / 4 - 1e-12)

    def test_kind_shares_are_the_mixture_weights_whatever_the_scores(self):
        # abp-mix-04: one distribution over all rooms gave the arena ~0.1% and boss rooms ~8%.
        levels = [('arena', 0)] + [('normal', v) for v in range(495)] + [('boss', 1000 + v) for v in range(77)]
        plr = PrioritizedLevels(levels, WEIGHTS)
        rng = np.random.default_rng(3)
        for _ in range(5000):
            level = int(rng.integers(len(levels)))
            plr._finish(level, float(rng.exponential(0.05 if level < 496 else 0.001)))
        p = plr.probabilities()
        for kind, share in shares(plr, p).items():
            self.assertAlmostEqual(share, WEIGHTS[kind], places=12)
        self.assertAlmostEqual(p[0], 0.2, places=12)
        summary = plr.summary(p)
        self.assertAlmostEqual(summary['share_boss'], 0.35, places=12)
        self.assertAlmostEqual(summary['effective_arena'], 1.0)
        self.assertLess(summary['effective_normal'], 495)
        # Weights are normalised over the kinds present.
        two = PrioritizedLevels(ROOMS[1:], {'normal': 0.45, 'boss': 0.35, 'arena': 0.2})
        self.assertAlmostEqual(shares(two, two.probabilities())['normal'], 0.45 / 0.8)

    def test_floor_one_is_the_fixed_mixture(self):
        plr = PrioritizedLevels(ROOMS, WEIGHTS, floor=1.0)
        for level, score in ((1, 5.0), (5, 2.0), (6, 0.0)):
            plr._finish(level, score)
        self.assertTrue(np.allclose(plr.probabilities(), [0.2] + [0.45 / 4] * 4 + [0.35 / 2] * 2))

    def test_every_kind_needs_a_weight(self):
        with self.assertRaises(ValueError):
            PrioritizedLevels(LEVELS, {'arena': 0.2, 'normal': 0.8})

    def test_staleness_favours_rooms_not_played_for_long(self):
        plr = PrioritizedLevels([('normal', 11), ('normal', 12)], {'normal': 1.0}, beta=1.0, staleness=1.0, floor=0.0)
        plr._finish(0, 1.0)
        for _ in range(3):
            plr._finish(1, 1.0)
        p = plr.probabilities()
        self.assertGreater(p[0], p[1])

    def test_state_round_trip_and_summary(self):
        plr = PrioritizedLevels(ROOMS, WEIGHTS)
        for level, score in ((0, 0.1), (1, 0.2), (2, 0.1), (5, 2.0)):
            plr._finish(level, score)
        state = plr.state_dict()
        copy = PrioritizedLevels.from_state(state)
        self.assertTrue(np.allclose(copy.probabilities(), plr.probabilities()))
        summary = plr.summary()
        self.assertAlmostEqual(summary['share_arena'] + summary['share_normal'] + summary['share_boss'], 1.0)
        self.assertEqual(summary['seen'], 4)
        self.assertEqual(plr.top(1)[0]['level'], ['arena', 0])
        # States saved before the kinds kept their weights (abp-mix-04) need the weights passed.
        del state['weights']
        with self.assertRaises(ValueError):
            PrioritizedLevels.from_state(state)
        old = PrioritizedLevels.from_state(state, WEIGHTS)
        self.assertTrue(np.allclose(old.probabilities(), plr.probabilities()))

    def test_mixture_levels_follow_the_spec(self):
        spec = {'weights': {'arena': 0.2, 'normal': 0.45, 'boss': 0.35}, 'normal': [5, 6], 'boss': [9]}
        self.assertEqual(mixture_levels(spec), [('arena', 0), ('normal', 5), ('normal', 6), ('boss', 9)])
        spec['weights']['arena'] = 0
        self.assertEqual(mixture_levels(spec)[0], ('normal', 5))


class WorkerHelpersTest(unittest.TestCase):
    def test_start_bombs_are_seeded_half_zero_otherwise_one_to_three(self):
        config = {'zero_prob': 0.5, 'max': 3}
        draws = [W.sample_bombs(seed, config) for seed in range(4000)]
        self.assertEqual(draws, [W.sample_bombs(seed, config) for seed in range(4000)])
        self.assertAlmostEqual(draws.count(0) / len(draws), 0.5, delta=0.03)
        self.assertEqual(set(draws), {0, 1, 2, 3})
        self.assertEqual(W.sample_bombs(7, None), 1)

    def test_plr_chooser_follows_the_shared_distribution(self):
        probabilities = np.array([0.0, 0.0, 1.0, 0.0])
        chooser = W.PlrTaskChooser(LEVELS, probabilities)
        tasks = [chooser.choose(seed) for seed in range(50)]
        self.assertTrue(all((t.kind, t.variant) == ('normal', 12) for t in tasks))
        self.assertEqual(chooser.choose(3, 1), chooser.choose(3, 1))
        probabilities[:] = [1.0, 0.0, 0.0, 0.0]                   # the learner rewrote the table
        self.assertEqual(chooser.choose(3).kind, 'arena')
        levels = {level: i for i, level in enumerate(LEVELS)}
        self.assertEqual(W.level_index(levels, {'task': 'arena', 'room_variant': 1010}), 0)
        self.assertEqual(W.level_index(levels, {'task': 'boss', 'room_variant': 21}), 3)
        self.assertEqual(W.level_index({}, {'task': 'boss', 'room_variant': 21}), -1)

    def test_episode_stats(self):
        def raw(x, y, blocking, bombs=1):
            return {'players': [{'pos': [x, y], 'bombs': bombs}], 'combat': {'blocking_hp': blocking}}
        stats = W.EpisodeStats(raw(100, 100, 30.0))
        for _ in range(300):                                    # 20 s parked, nothing hit
            stats.step(raw(100, 100, 30.0), 2)
        stats.step(raw(200, 100, 26.5, bombs=0), 2)             # moves, hits, drops the bomb
        for _ in range(15):
            stats.step(raw(200, 100, 26.5, bombs=0), 2)
        first, no_hit, stationary, cells, bombs = stats.array()
        self.assertAlmostEqual(first, 301 * 2 / 30, places=4)
        self.assertAlmostEqual(no_hit, 301 * 2 / 30, places=4)
        self.assertAlmostEqual(stationary, 301 * 2 / 30, places=4)
        self.assertEqual(cells, 2)
        self.assertEqual(bombs, 1)

    def test_combat_frame_layout_extends_the_simulator_record(self):
        dtype, keys = W.frame_layout('combat-v3')
        self.assertEqual(dtype.names[:len(W.FRAME_DTYPE.names)], W.FRAME_DTYPE.names)
        self.assertEqual(dtype.itemsize, W.FRAME_DTYPE.itemsize + 4 * 4)
        self.assertEqual(keys[-1], 'combat')
        self.assertEqual(W.frame_layout('combat-v2'), (W.FRAME_DTYPE, W.FRAME_KEYS))
        v4, _ = W.frame_layout('combat-v4')
        self.assertEqual(v4.itemsize, W.FRAME_DTYPE.itemsize + 2 * 4)

    def test_observation_options_per_profile(self):
        from isaac_bridge.transformer_obs import COMBAT_FIELDS, COMBAT_FIELDS_V4, VisibleHistory
        self.assertEqual(W.observation_options('combat-v4'), dict(deadline=False, combat_state=COMBAT_FIELDS_V4))
        self.assertEqual(W.observation_options('combat-v3'), dict(deadline=True, combat_state=COMBAT_FIELDS))
        self.assertEqual(W.observation_options('combat-v2'), dict(deadline=True, combat_state=False))
        space = VisibleHistory(64, 256, **W.observation_options('combat-v4')).space
        self.assertNotIn('remaining_time', space.spaces)           # truncation: no elapsed-time input
        self.assertEqual(space['combat'].shape, (64, 2))


if __name__ == '__main__':
    unittest.main()
