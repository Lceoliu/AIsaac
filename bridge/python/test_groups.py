"""Parallel task groups (abplus_groups), the model size options and the distillation KL (user plan 2026-09-28)."""
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
GROUPS = HERE.parent / 'abplus' / 'catalog' / 'parallel_groups.json'


def simulate(scheduler, lengths, episodes, seed=0):
    """A worker's schedule: the next episode is chosen when the current one begins (the standby prepares it)."""
    played = np.zeros(len(lengths))
    upcoming = scheduler.choose(seed)
    for k in range(episodes):
        current = upcoming
        scheduler.start(current)
        upcoming = scheduler.choose(seed + k + 1)
        scheduler.finish(current, lengths[current])
        played[current] += lengths[current]
    return played / played.sum()


class GroupsFileTest(unittest.TestCase):
    def test_default_groups_resolve(self):
        from isaac_bridge.abplus_groups import describe_groups, load_groups
        groups = load_groups(GROUPS)
        self.assertEqual([g['name'] for g in groups], ['normal', 'obstacles', 'dodge'])
        self.assertAlmostEqual(sum(g['share'] for g in groups), 1.0)
        self.assertEqual([g['frames'] for g in groups], [5400, 900, 900])
        self.assertEqual([bool(g['target']) for g in groups], [False, True, True])
        self.assertEqual(len(groups[0]['spec']['normal']), 202)
        self.assertEqual(len(groups[1]['target']['arms']), 3)
        described = describe_groups(groups, 'steps')
        self.assertEqual(described['budget'], 'steps')
        self.assertEqual([g['has_target'] for g in described['groups']], [False, True, True])

    def test_bad_files_are_refused(self):
        import json
        import tempfile
        from isaac_bridge.abplus_groups import load_groups
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'g.json'
            for groups in ([], [{'name': 'a', 'tasks': str(GROUPS.parent / 'mixture_target_dodge.json'), 'share': 0}],
                           [{'name': 'a', 'tasks': str(GROUPS.parent / 'mixture_target_dodge.json'), 'share': 1}] * 2,
                           [{'name': 'a/b', 'tasks': str(GROUPS.parent / 'mixture_target_dodge.json'), 'share': 1}]):
                path.write_text(json.dumps({'groups': groups}), encoding='utf8')
                with self.assertRaises(ValueError):
                    load_groups(path)


class BudgetTest(unittest.TestCase):
    def test_slots_are_apportioned_and_interleaved(self):
        from isaac_bridge.abplus_groups import slot_groups
        slots = slot_groups([0.5, 0.25, 0.25], 16)
        self.assertEqual(np.bincount(slots).tolist(), [8, 4, 4])
        self.assertEqual(sorted(set(slots[:4])), [0, 1, 2])          # the near-greedy actors see every group
        self.assertEqual(np.bincount(slot_groups([0.5, 0.25, 0.25], 64)).tolist(), [32, 16, 16])
        self.assertEqual(np.bincount(slot_groups([1, 1, 1], 16)).tolist(), [6, 5, 5])

    def test_episode_budget_gives_the_long_group_most_steps(self):
        """The user's example: 50 % of the starts, 3 s against 30 s episodes -> ~91 % of the steps."""
        from isaac_bridge.abplus_groups import GroupScheduler
        shares = simulate(GroupScheduler([0.5, 0.5], 'episodes'), [45, 450], 4000)
        self.assertAlmostEqual(shares[1], 450 / 495, delta=0.02)

    def test_step_budget_follows_the_shares(self):
        from isaac_bridge.abplus_groups import GroupScheduler
        for target, lengths in (([0.5, 0.5], [45, 450]), ([0.5, 0.25, 0.25], [2700, 60, 75]), ([0.2, 0.8], [900, 30])):
            shares = simulate(GroupScheduler(target, 'steps'), lengths, 3000)
            np.testing.assert_allclose(shares, target, atol=0.02)

    def test_step_budget_uses_the_running_length_of_the_episode_in_progress(self):
        from isaac_bridge.abplus_groups import GroupScheduler
        s = GroupScheduler([0.5, 0.5], 'steps')
        for _ in range(3):
            s.start(0)
            s.finish(0, 100)
        s.start(1)
        s.finish(1, 10)
        self.assertEqual(s.choose(0), 1)
        s.start(1)                     # in progress: counted at group 1's mean length (10)
        self.assertEqual(s.choose(1), 1)
        self.assertAlmostEqual(s.mean_length(1), 10.0)

    def test_slot_budget_keeps_the_slot_group(self):
        from isaac_bridge.abplus_groups import GroupScheduler
        s = GroupScheduler([0.5, 0.5], 'slots', slot_group=1)
        self.assertEqual({s.choose(seed) for seed in range(50)}, {1})
        with self.assertRaises(ValueError):
            GroupScheduler([1.0], 'rooms')


class ModelSizeTest(unittest.TestCase):
    def space(self):
        from isaac_bridge.abplus_worker import observation_options
        from isaac_bridge.transformer_obs import VisibleHistory
        return VisibleHistory(64, 256, **observation_options('combat-hitrate-miss')).space

    def test_defaults_are_the_old_architecture(self):
        from isaac_bridge.transformer_policy import CombatTransformer
        m = CombatTransformer(self.space(), features_dim=256, layers=4, heads=8)
        self.assertEqual(tuple(m.queries.shape), (4, 128))
        self.assertEqual(m.fusion[0].in_features, 64 + 512 + 128 + 32)
        self.assertEqual(len(m.temporal), 4)

    def test_larger_student_builds_and_runs(self):
        from isaac_bridge.transformer_policy import CombatTransformer
        space = self.space()
        small = CombatTransformer(space, features_dim=256, layers=4, heads=8)
        big = CombatTransformer(space, features_dim=384, layers=6, heads=8, entity_queries=8)
        self.assertEqual(tuple(big.queries.shape), (8, 128))
        self.assertEqual(big.fusion[0].in_features, 64 + 8 * 128 + 128 + 32)
        count = lambda m: sum(p.numel() for p in m.parameters())
        self.assertGreater(count(big), 2.5 * count(small))
        # Integer inputs (kinds, animations) at 0, the floats random; 5 entities in every frame.
        obs = {k: torch.zeros((2, *v.shape), dtype=torch.as_tensor(v.sample()).dtype) for k, v in space.spaces.items()}
        for k in ('player', 'entities', 'terrain'):
            obs[k] = torch.rand_like(obs[k])
        obs['history_mask'] = torch.ones_like(obs['history_mask'])
        obs['entity_mask'][..., :5] = 1.0
        obs['time'] = torch.arange(64, dtype=obs['time'].dtype).expand(2, -1) / 15
        with torch.no_grad():
            out = big(obs)
        self.assertEqual(tuple(out.shape), (2, big.features_dim))
        self.assertTrue(torch.isfinite(out).all())


class DistillationKLTest(unittest.TestCase):
    def test_masked_four_head_kl(self):
        from isaac_bridge.gpu_ppo import full_kl
        nvec = (9, 5, 2, 2)
        torch.manual_seed(0)
        logits = torch.randn(3, sum(nvec))
        masks = torch.ones_like(logits, dtype=torch.bool)
        masks[:, 9 + 5 + 2 + 1] = False          # the item is never allowed
        masks[0, :3] = False                     # blocked moves

        def masked(x):
            out = torch.where(masks, x, torch.full_like(x, -1e8))
            return torch.cat([torch.log_softmax(part, -1) for part in torch.split(out, nvec, -1)], -1)

        teacher, student = masked(logits), masked(logits + torch.randn_like(logits))
        self.assertTrue(torch.allclose(full_kl(teacher, teacher, nvec), torch.zeros(3), atol=1e-6))
        kl = full_kl(teacher, student, nvec)
        self.assertTrue(torch.isfinite(kl).all() and (kl > 0).all())
        # equal to the per-head sum computed directly over the allowed actions
        want = 0
        for t, q, m in zip(torch.split(teacher, nvec, -1), torch.split(student, nvec, -1), torch.split(masks, nvec, -1)):
            want = want + (t.exp() * (t - q) * m).sum(-1)
        self.assertTrue(torch.allclose(kl, want, atol=1e-5))
        tau = 2.0   # the trainer softens both by T (masked logits stay ~ -inf)
        self.assertTrue(torch.isfinite(full_kl(teacher / tau, student / tau, nvec)).all())


if __name__ == '__main__':
    unittest.main()
