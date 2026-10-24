"""C39 (user decisions 2026-09-28): combat-hp, PLR by steps within parallel groups, the stored remaining time and
the frames per decision. No game instance or GPU needed."""
import json
import unittest
from pathlib import Path

import numpy as np

CATALOG = Path(__file__).resolve().parent.parent / 'abplus' / 'catalog'


def combat(monster=0.0, player=0.0):
    return {'combat': {'monster_damage': monster, 'player_damage': player}}


class CombatHpTest(unittest.TestCase):
    def test_components(self):
        from isaac_bridge.abplus_reward import COMPONENTS_HP, CombatHp
        r = CombatHp()
        r.reset(combat(5.0, 1.0), 'normal')
        got = r.step(combat(12.0, 1.0), 'running', 4)            # 7 HP lost by monsters: +2
        self.assertEqual(set(got), set(COMPONENTS_HP))
        self.assertAlmostEqual(got['damage'], 2.0)
        self.assertEqual(got['hurt'], 0.0)
        got = r.step(combat(12.0, 3.0), 'running', 4)            # two half hearts: -1
        self.assertAlmostEqual(got['hurt'], -1.0)
        self.assertAlmostEqual(r.step(combat(15.5, 3.0), 'win', 4)['clear'], 100.0)
        self.assertAlmostEqual(sum(r.totals.values()), 2.0 - 1.0 + 1.0 + 100.0)
        self.assertEqual(len(r.totals_array()), len(COMPONENTS_HP))

    def test_timeout_and_death_are_both_minus_100(self):
        from isaac_bridge.abplus_reward import CombatHp
        for outcome, key in (('time_limit', 'timeout'), ('death', 'death')):
            r = CombatHp()
            r.reset(combat(), 'normal')
            got = r.step(combat(0.0, 1.0), outcome, 4)
            self.assertEqual(got[key], -100.0)
            self.assertAlmostEqual(sum(got.values()), -100.5)

    def test_needs_the_monster_counter(self):
        from isaac_bridge.abplus_reward import CombatHp
        with self.assertRaisesRegex(ValueError, 'abp-0.2.8-hp'):
            CombatHp().reset({'combat': {'player_damage': 0.0}}, 'normal')

    def test_profile_is_registered(self):
        from isaac_bridge import abplus_worker as W
        from isaac_bridge.abplus_reward import REWARDS, CombatHp, describe_hp
        self.assertIs(REWARDS['combat-hp'], CombatHp)
        for group in (W.REWARD_PROFILES, W.DEADLINE_PROFILES, W.GEOMETRY_PROFILES, W.WALK_PROFILES,
                      W.STORED_DEADLINE_PROFILES):
            self.assertIn('combat-hp', group)
        self.assertNotIn('combat-hp', W.TRUNCATING_PROFILES)     # the deadline is a termination
        options = W.observation_options('combat-hp')
        self.assertTrue(options['deadline'])
        self.assertEqual(options['combat_state'], ('blocking_count', 'blocking_fraction'))
        dtype, _ = W.frame_layout('combat-hp')
        self.assertIn('remaining_time', dtype.names)
        self.assertNotIn('remaining_time', W.frame_layout('combat-hitrate-hurt')[0].names)
        self.assertIn('terminal', describe_hp()['death'])


class RemainingTimeTest(unittest.TestCase):
    def test_history_uses_its_deadline(self):
        from isaac_bridge.transformer_obs import VisibleHistory
        h = VisibleHistory(deadline=True, deadline_s=180.0)
        self.assertEqual(h.deadline_s, 180.0)
        self.assertEqual(VisibleHistory(deadline=True).deadline_s, 120.0)

    def test_frames_carry_it_and_the_learner_reads_it(self):
        import torch
        from isaac_bridge import abplus_worker as W
        from isaac_bridge.gpu_env import decode_frame
        from isaac_bridge.transformer_obs import VisibleHistory
        dtype, _ = W.frame_layout('combat-hp')
        rows = np.zeros(2, dtype)
        rows['time'] = (90.0, 30.0)
        rows['remaining_time'] = (0.5, 1 - 30 / 180)
        space = VisibleHistory(**W.observation_options('combat-hp')).space
        words = torch.from_numpy(rows.view(np.int32).reshape(2, -1).copy())
        got = decode_frame(words, space, dtype)
        self.assertTrue(torch.allclose(got['remaining_time'], torch.tensor([0.5, 1 - 30 / 180])))
        # Profiles without the field keep 1 - time / 120.
        old, _ = W.frame_layout('combat-v3')
        rows = np.zeros(1, old)
        rows['time'] = 60.0
        got = decode_frame(torch.from_numpy(rows.view(np.int32).reshape(1, -1).copy()),
                           VisibleHistory(**W.observation_options('combat-v3')).space, old)
        self.assertAlmostEqual(float(got['remaining_time'][0]), 0.5)

    def test_write_frame_copies_it(self):
        from isaac_bridge import abplus_worker as W
        dtype, _ = W.frame_layout('combat-hp')
        row = np.zeros(1, dtype)[0]
        frame = {k: np.zeros(dtype[k].shape, dtype[k].base) for k in dtype.names}
        frame['remaining_time'] = np.float32(0.25)
        W.write_frame(row, frame, 0.0, False, False, 0, 0, 0, 0)
        self.assertEqual(float(row['remaining_time']), 0.25)


class PlrByStepsTest(unittest.TestCase):
    def test_start_probability_divides_by_length(self):
        from isaac_bridge.plr import PrioritizedLevels
        levels = [('a', 0, -1), ('a', 1, -1), ('b', 0, -1)]
        plr = PrioritizedLevels(levels, {'a': 0.5, 'b': 0.5}, floor=1.0, by_steps=True)   # floor 1: P uniform per kind
        np.testing.assert_allclose(plr.probabilities(), [0.25, 0.25, 0.5])   # no length known yet
        plr._finish(0, 1.0, 30)
        plr._finish(1, 1.0, 10)
        np.testing.assert_allclose(plr.step_shares(), [0.25, 0.25, 0.5])
        # Within kind a: 1/30 : 1/10, so both collect the same steps.
        np.testing.assert_allclose(plr.probabilities(), [0.125, 0.375, 0.5])
        q = plr.probabilities()[:2] * plr.durations[:2]
        self.assertAlmostEqual(q[0], q[1])

    def test_length_is_a_mean_then_an_exponential_average(self):
        from isaac_bridge.plr import DURATION_WINDOW, PrioritizedLevels
        plr = PrioritizedLevels([('a', 0, -1)], {'a': 1.0}, by_steps=True)
        for steps in (10, 20, 30):
            plr._finish(0, 0.0, steps)
        self.assertAlmostEqual(plr.durations[0], 20.0)
        for _ in range(DURATION_WINDOW - 3):
            plr._finish(0, 0.0, 20)
        plr._finish(0, 0.0, 120)
        self.assertAlmostEqual(plr.durations[0], 20.0 + (120 - 20) / DURATION_WINDOW)

    def test_unknown_length_is_the_kind_mean(self):
        from isaac_bridge.plr import PrioritizedLevels
        plr = PrioritizedLevels([('a', v, -1) for v in range(3)], {'a': 1.0}, floor=1.0, by_steps=True)
        plr._finish(0, 0.0, 10)
        plr._finish(1, 0.0, 30)
        np.testing.assert_allclose(plr.lengths(np.arange(3)), [10, 30, 20])

    def test_update_measures_the_episodes(self):
        from isaac_bridge.plr import PrioritizedLevels
        plr = PrioritizedLevels([('a', 0, -1), ('a', 1, -1)], {'a': 1.0}, by_steps=True)
        starts = np.zeros((10, 1))
        starts[[0, 4]] = 1
        levels = np.array([[0]] * 4 + [[1]] * 6)
        plr.update(np.ones((10, 1)), starts, levels)
        self.assertEqual(plr.durations.tolist(), [4.0, 0.0])      # the second episode is still running
        state = json.loads(json.dumps(plr.state_dict()))
        again = PrioritizedLevels.from_state(state)
        self.assertTrue(again.by_steps)
        self.assertEqual(again.durations.tolist(), [4.0, 0.0])

    def test_steps_follow_the_shares_in_a_simulation(self):
        """Two rooms, 30- and 3-step episodes: by episode starts the long room takes ~91% of the steps, by steps
        50% (the user's example of 2026-09-28)."""
        from isaac_bridge.plr import PrioritizedLevels
        rng = np.random.default_rng(0)
        for by_steps, expected in ((False, 30 / 33), (True, 0.5)):
            plr = PrioritizedLevels([('a', 0, -1), ('a', 1, -1)], {'a': 1.0}, floor=1.0, by_steps=by_steps)
            steps = np.zeros(2)
            for _ in range(4000):
                i = int(rng.random() >= plr.probabilities()[0])
                length = (30, 3)[i]
                plr._finish(i, 0.0, length)
                steps[i] += length
            self.assertAlmostEqual(steps[0] / steps.sum(), expected, delta=0.02)


class GroupLevelsTest(unittest.TestCase):
    def test_c39_groups(self):
        from isaac_bridge.abplus_groups import load_groups
        from isaac_bridge.plr import group_levels
        groups = load_groups(CATALOG / 'scaling_groups.json')
        self.assertEqual([g['name'] for g in groups], ['normal', 'horf', 'horf_rocks', 'spawners'])
        self.assertEqual({g['share'] for g in groups}, {0.25})
        self.assertEqual({g['seconds'] for g in groups}, {180.0})
        levels, weights = group_levels(groups)
        count = {name: sum(1 for level in levels if level[0] == name) for name in weights}
        self.assertEqual(count, {'normal': 491, 'horf': 63, 'horf_rocks': 3 * 365, 'spawners': 4 * 63})
        self.assertEqual(len(set(levels)), len(levels))
        self.assertTrue(all(level[2] == -1 for level in levels if level[0] in ('normal', 'horf')))
        spawners = groups[3]['target']['arms']
        self.assertEqual([a['type'] for a in spawners], [306, 16, 205, 206])
        self.assertEqual(spawners[0]['extras']['counts'], [0, 1, 2, 3])
        rocks = groups[2]['target']['arms']
        self.assertEqual([a.get('extras', {}).get('counts') for a in rocks], [None, [1], [2]])
        self.assertTrue(all(a['obstacles'] for a in rocks))

    def test_chooser_stays_in_its_group(self):
        from isaac_bridge.abplus_worker import GroupPlrChooser
        levels = [('a', 5, -1), ('b', 7, 0), ('b', 7, 1), ('b', 9, 1)]
        p = np.array([0.7, 0.1, 0.1, 0.1])
        chooser = GroupPlrChooser(levels, p, [1, 2, 3])
        tasks = {(t.variant, t.arm) for t in (chooser.choose(seed) for seed in range(200))}
        self.assertEqual(tasks, {(7, 0), (7, 1), (9, 1)})
        self.assertEqual({chooser.choose(3).kind}, {'normal'})
        self.assertEqual(chooser.choose(11), chooser.choose(11))         # a function of the seed
        with self.assertRaises(ValueError):
            GroupPlrChooser(levels, p, [])


class FramesPerDecisionTest(unittest.TestCase):
    def test_env_holds_actions_for_the_given_frames(self):
        from isaac_bridge.abplus import AbplusTransformerEnv
        env = AbplusTransformerEnv(port=1, frames_per_decision=4, deadline_s=180.0,
                                   deadline=True, combat_state=('blocking_count', 'blocking_fraction'))
        self.assertEqual(env.bridge.action_repeat, 4)
        self.assertEqual(env.history.deadline_s, 180.0)
        self.assertEqual(AbplusTransformerEnv(port=1).bridge.action_repeat, 2)
        with self.assertRaises(ValueError):
            AbplusTransformerEnv(port=1, frames_per_decision=0)


class MemoryRecycleTest(unittest.TestCase):
    """C39 t3r (09-29 14:59 out of memory): the resident-memory cap on AB+ processes and the staggered recycling."""

    @unittest.skipUnless(Path('/proc/self/status').exists(), 'needs Linux /proc')
    def test_rss_of_the_process(self):
        import os
        from types import SimpleNamespace
        from isaac_bridge.abplus_worker import Instance
        instance = Instance.__new__(Instance)          # no game process: rss_mib only
        instance.proc = SimpleNamespace(pid=os.getpid())
        self.assertGreater(instance.rss_mib(), 1.0)
        instance.proc = SimpleNamespace(pid=2 ** 22 + 12345)   # above pid_max: no such process
        self.assertEqual(instance.rss_mib(), 0.0)
        instance.proc = None
        self.assertEqual(instance.rss_mib(), 0.0)

    def test_meta_and_default(self):
        from isaac_bridge.abplus_worker import META_DTYPE, RECYCLE_RSS_MIB
        for name in ('rss_mib', 'instance', 'instance_episodes', 'memory_recycles'):
            self.assertIn(name, META_DTYPE.names)
        self.assertEqual(RECYCLE_RSS_MIB, 0)   # off unless a run sets --recycle-rss-mib

    def test_staggered_thresholds(self):
        from isaac_bridge.abplus_worker import recycle_threshold
        got = [recycle_threshold(200, i) for i in range(32)]
        self.assertTrue(all(200 <= t < 400 for t in got))
        self.assertGreater(len(set(got)), 28)
        self.assertEqual(recycle_threshold(0, 5), 0)


if __name__ == '__main__':
    unittest.main()
