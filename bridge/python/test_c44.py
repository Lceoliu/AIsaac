"""C44 (goal-hp2, rl/docs/GOAL_CONDITIONED_DESIGN.md 2.2 item 5 and 4.8): the full-room branch joins a trained goal-line
policy at zero (the migrated policy computes the checkpoint's function on every task, both encoding paths), is wired in
(non-zero weights change the outputs and gradients reach it), the optimizer's two named groups ('shared', 'goal') with their
learning rates, and room_bits (the whole room on the 16 x 28 canvas: ROOM_BITS per cell)."""
import unittest

import numpy as np
import torch

from isaac_bridge.abplus_worker import frame_layout, observation_options
from isaac_bridge.gpu_ppo import GroupLearningRate
from isaac_bridge.training_session import _goal_line_key, _optimizer_names, _room_key
from isaac_bridge.transformer_obs import ROOM_BITS, ROOM_CANVAS, VisibleHistory, terrain_channels
from isaac_bridge.transformer_policy import GeometryPolicy
from test_goal_model import ACTIONS, KWARGS, heads, perturb, random_windows, static_heads


def policy(space, seed, **extra):
    torch.manual_seed(seed)
    return GeometryPolicy(space, ACTIONS, lambda _: 1e-4, **KWARGS, **extra).eval()


def bits(value, name):
    return (int(value) >> ROOM_BITS.index(name)) & 1


class RoomBranchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_space = VisibleHistory(**observation_options('goal-hp')).space
        cls.new_space = VisibleHistory(**observation_options('goal-hp2')).space
        cls.old = policy(cls.old_space, 1)
        perturb(cls.old, np.random.default_rng(3))   # a trained goal line (C43): every goal module non-zero
        cls.new = policy(cls.new_space, 2, lr_groups=True)
        source = cls.old.state_dict()
        cls.missing = [k for k in cls.new.state_dict() if k not in source]
        result = cls.new.load_state_dict(source, strict=False)
        assert not result.unexpected_keys and set(result.missing_keys) == set(cls.missing)

    def windows(self, rng, tasks=('combat', 'combat', 'goto_position', 'goto_door')):
        obs = random_windows(self.new_space, len(tasks), tasks, rng)
        obs['room_bits'] = torch.as_tensor(rng.integers(0, 256, tuple(obs['room_bits'].shape)), dtype=torch.float32)
        return obs, {k: v for k, v in obs.items() if k in self.old_space.spaces}

    def test_only_the_room_branch_is_new_and_it_starts_at_zero(self):
        self.assertTrue(self.missing and all(_room_key(k) for k in self.missing))
        self.assertTrue(all(_goal_line_key(k) for k in self.missing))
        self.assertTrue(self.new.features_extractor.room_injection_zero())
        self.assertEqual(set(self.new_space.spaces) - set(self.old_space.spaces), {'room_bits'})
        self.assertEqual(self.new_space.spaces['room_bits'].shape[1:], ROOM_CANVAS)

    def test_migrated_outputs_bit_for_bit(self):
        new_obs, old_obs = self.windows(np.random.default_rng(5))
        with torch.no_grad():
            for path in (heads, static_heads):
                ol, ov = path(self.old, old_obs)
                nl, nv = path(self.new, new_obs)
                self.assertTrue(torch.equal(nl, ol), path.__name__)
                self.assertTrue(torch.equal(nv, ov), path.__name__)

    def test_the_branch_is_wired_in(self):
        new = policy(self.new_space, 4, lr_groups=True)
        new.load_state_dict(self.new.state_dict())
        with torch.no_grad():
            for p in new.features_extractor.room_fusion.parameters():
                p.normal_(0, 0.1)
        new_obs, old_obs = self.windows(np.random.default_rng(6))
        with torch.no_grad():
            nl, _ = heads(new, new_obs)
            ol, _ = heads(self.old, old_obs)
            sl, _ = static_heads(new, new_obs)
        self.assertFalse(torch.equal(nl, ol))
        self.assertTrue(torch.allclose(nl, sl, atol=1e-4))   # both paths see the branch
        new.train()
        logits, values = heads(new, new_obs)
        (logits.square().mean() + values.square().mean()).backward()
        cnn = [p.grad for n, p in new.named_parameters() if 'room_cnn' in n]
        self.assertTrue(all(g is not None and g.abs().sum() > 0 for g in cnn))

    def test_optimizer_groups(self):
        groups = self.new.optimizer.param_groups
        self.assertEqual([g['name'] for g in groups], ['shared', 'goal'])
        names = {id(p): n for n, p in self.new.named_parameters()}
        self.assertTrue(all(_goal_line_key(names[id(p)]) for p in groups[1]['params']))
        self.assertFalse(any(_goal_line_key(names[id(p)]) for p in groups[0]['params']))
        self.assertEqual(sum(len(g['params']) for g in groups), len(list(self.new.parameters())))
        order = _optimizer_names(self.new, [dict(name='shared'), dict(name='goal')])
        self.assertEqual(order, [names[id(p)] for g in groups for p in g['params']])
        plain = _optimizer_names(self.old, [dict()])
        self.assertEqual(plain, [n for n, _ in self.old.named_parameters()])

    def test_group_learning_rate(self):
        shared = GroupLearningRate('cosine', 1e-4, 1e-5, 5, 1e-5)
        self.assertAlmostEqual(shared(1.0, 0), 1e-5)
        self.assertAlmostEqual(shared(1.0, 5), 1e-4)
        self.assertTrue(1e-5 < shared(1.0, 2) < 1e-4)
        self.assertAlmostEqual(shared(0.0, 400), 1e-5)
        goal = GroupLearningRate('cosine', 3e-4, 3e-5)
        self.assertAlmostEqual(goal(1.0, 0), 3e-4)
        self.assertAlmostEqual(goal(0.5, 200), (3e-4 + 3e-5) / 2)

    def test_frame_layout(self):
        dtype, keys = frame_layout('goal-hp2')
        self.assertIn('room_bits', keys)
        self.assertEqual(dtype.fields['room_bits'][0].shape, ROOM_CANVAS)
        old_dtype, old_keys = frame_layout('goal-hp')
        self.assertNotIn('room_bits', old_keys)


def room(height, width, missing=(), rocks=(), doors=()):
    """A raw observation's terrain / doors / player of a height x width grid: the border walls, the missing cells (an L
    shape's cut-out) outside, rocks solid and destructible."""
    cells = []
    for index in range(height * width):
        r, c = divmod(index, width)
        inside = (r, c) not in missing
        wall = r in (0, height - 1) or c in (0, width - 1)
        rock = (r, c) in rocks
        walkable = inside and not wall and not rock
        cells.append([index, 40.0 * c, 40.0 * r, 0, int(inside and not wall), int(walkable), 0, 0,
                      int(wall or rock), int(rock)])
    return dict(terrain=dict(height=height, width=width, cells=cells, version=1),
                doors=[dict(pos=[40.0 * c, 40.0 * r], open=True) for r, c in doors])


class RoomBitsTest(unittest.TestCase):
    def encode(self, obs, player, target=None, row0=0, col0=0):
        h = VisibleHistory(**observation_options('goal-hp2'))
        h.goal_state = dict(task='goto_position' if target else 'combat', target=target, distance=None, source='goto')
        obs = dict(obs, players=[dict(pos=list(player))])
        full = terrain_channels(obs, (obs['terrain']['height'], obs['terrain']['width']))
        return h.room_bits_frame(obs, full, row0, col0).astype(np.int64)

    def test_one_room(self):
        obs = room(9, 15, rocks={(4, 7)}, doors={(0, 7)})
        b = self.encode(obs, (40.0 * 3, 40.0 * 2), target=(40.0 * 10, 40.0 * 6))
        self.assertEqual(b.shape, ROOM_CANVAS)
        self.assertEqual(bits(b[2, 3], 'player'), 1)
        self.assertEqual(bits(b[6, 10], 'goal'), 1)
        self.assertEqual(bits(b[4, 7], 'destructible'), 1)
        self.assertEqual(bits(b[4, 7], 'walkable'), 0)
        self.assertEqual(bits(b[0, 7], 'door'), 1)
        self.assertEqual(bits(b[5, 5], 'walkable'), 1)
        self.assertEqual(bits(b[5, 5], 'window'), 1)             # a 1x1 room is its camera window
        self.assertEqual(int(b[9:].sum() + b[:, 15:].sum()), 0)   # nothing beyond the room's grid
        self.assertEqual(sum(bits(v, 'player') for v in b.flat), 1)

    def test_l_room_and_window(self):
        missing = {(r, c) for r in range(8, 16) for c in range(14, 28)}   # bottom-right quadrant cut away (LBR)
        obs = room(16, 28, missing=missing)
        b = self.encode(obs, (40.0 * 5, 40.0 * 12), row0=7, col0=0)
        self.assertEqual(bits(b[12, 20], 'inside'), 0)
        self.assertEqual(bits(b[12, 20], 'walkable'), 0)
        self.assertEqual(bits(b[3, 20], 'walkable'), 1)
        self.assertEqual(bits(b[12, 5], 'player'), 1)
        window = np.array([[bits(v, 'window') for v in row] for row in b])
        self.assertEqual(int(window.sum()), 9 * 15)
        self.assertEqual(int(window[7:16, 0:15].sum()), 9 * 15)


if __name__ == '__main__':
    unittest.main()
