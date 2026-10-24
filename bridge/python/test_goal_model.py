"""Goal-conditioned line M0 (rl/docs/GOAL_CONDITIONED_DESIGN.md): a goal-line policy migrated from a checkpoint without the
goal modules computes that checkpoint's function bit for bit on COMBAT frames (full windows and the per-frame static path),
also once the new modules are non-zero; GOTO frames change; gradients reach the new modules from GOTO samples only; the
migration accepts exactly the goal-line keys; observations carry the goal fields."""
import math
import unittest

import numpy as np
import torch
from gymnasium import spaces

from isaac_bridge.abplus_worker import observation_options
from isaac_bridge.training_session import _goal_line_key
from isaac_bridge.transformer_obs import GOAL_FIELDS, GOAL_SLOTS, GOAL_TASKS, SOURCES, VisibleHistory
from isaac_bridge.transformer_policy import CombatTransformer, GeometryPolicy, NavActionNet, TaskValueNet

ACTIONS = spaces.MultiDiscrete([9, 5, 2, 2])
KWARGS = dict(features_extractor_class=CombatTransformer,
              features_extractor_kwargs=dict(features_dim=256, layers=4, heads=8, entity_queries=4, entity_dim=128),
              net_arch=dict(pi=[256], vf=[256]), normalize_images=False)


def policy(space, seed):
    torch.manual_seed(seed)
    return GeometryPolicy(space, ACTIONS, lambda _: 1e-4, **KWARGS).eval()


def random_windows(space, batch, tasks, rng, history=16):
    """Random windows (batch, history, ...) of an observation space; row b's goal task is tasks[b]."""
    out = {}
    for key, box in space.spaces.items():
        shape = (batch, history) + box.shape[1:]
        if np.issubdtype(box.dtype, np.integer):
            high = int(min(box.high.max(), 64))
            out[key] = rng.integers(0, high + 1, shape).astype(np.float32)
        else:
            out[key] = rng.normal(0, 0.5, shape).astype(np.float32)
    out['history_mask'][:] = 1
    out['time'][:] = np.arange(history, dtype=np.float32) * 0.133
    out['entity_mask'][:] = 0
    for b in range(batch):
        for t in range(history):
            out['entity_mask'][b, t, :rng.integers(0, 12)] = 1
    out['terrain'] = (rng.random(out['terrain'].shape) < 0.3).astype(np.float32)
    out['previous_action'] = np.zeros_like(out['previous_action'])
    if 'goal' in out:
        out['goal'][:] = 0
        for b, task in enumerate(tasks):
            out['goal'][b, :, GOAL_TASKS.index(task)] = 1
            if task != 'combat':
                out['goal'][b, :, GOAL_SLOTS:] = rng.normal(0, 0.5, (history, GOAL_FIELDS - GOAL_SLOTS))
        out['goal_map'] = (rng.random(out['goal_map'].shape) < 0.05).astype(np.float32)
        out['goal_distance'] = np.abs(out['goal_distance'])
        out['source'][:] = 0
    return {k: torch.as_tensor(v) for k, v in out.items()}


def heads(p, obs):
    features = p.features_extractor(obs)
    pi, vf = p.mlp_extractor(features)
    return p.action_net(pi), p.value_net(vf).flatten()


def static_heads(p, obs):
    """The per-frame static path (graph sampler): every frame encoded alone, then the temporal stage."""
    fe = p.features_extractor
    b, h = obs['history_mask'].shape
    frames = fe.encode_frame_static({k: v.flatten(0, 1) for k, v in obs.items()}).view(b, h, -1)
    latent = fe.temporal_features(frames, obs['time'], obs['history_mask'].bool())[:, -1]
    pi, vf = p.mlp_extractor(latent)
    return p.action_net(pi), p.value_net(vf).flatten()


def migrate(old, new):
    source = old.state_dict()
    missing = [k for k in new.state_dict() if k not in source]
    assert missing and all(_goal_line_key(k) for k in missing), missing
    result = new.load_state_dict(source, strict=False)
    assert not result.unexpected_keys and set(result.missing_keys) == set(missing)
    return missing


def perturb(p, rng):
    """What training does to the new modules: non-zero weights everywhere."""
    with torch.no_grad():
        for name, param in p.named_parameters():
            if _goal_line_key(name):
                param.copy_(torch.as_tensor(rng.normal(0, 0.1, tuple(param.shape)), dtype=param.dtype))


class GoalModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        options = observation_options('combat-hp')
        cls.old_space = VisibleHistory(**options).space
        cls.goal_space = VisibleHistory(**options, goal=True).space
        cls.old = policy(cls.old_space, 1)
        cls.new = policy(cls.goal_space, 2)
        cls.missing = migrate(cls.old, cls.new)

    def test_old_architecture_unchanged(self):
        self.assertEqual(self.old.features_extractor.goal_dim, 0)
        self.assertNotIsInstance(self.old.action_net, NavActionNet)
        self.assertNotIsInstance(self.old.value_net, TaskValueNet)
        names = [n for n, _ in self.old.named_parameters()]
        self.assertFalse(any(_goal_line_key(n) for n in names))

    def test_new_modules_zero_and_keys(self):
        fe = self.new.features_extractor
        self.assertTrue(fe.goal_injections_zero())
        self.assertFalse(self.new.action_net.nav[-1].weight.any())
        self.assertTrue(self.missing)
        # every existing parameter keeps its position in the parameter order (the new ones come after, per module)
        old_names = [n for n, _ in self.old.named_parameters()]
        new_names = [n for n, _ in self.new.named_parameters() if not _goal_line_key(n)]
        self.assertEqual(old_names, new_names)

    def check_combat_equal(self, new, rng, batch_tasks=('combat', 'combat', 'goto_position', 'goto_door')):
        obs = random_windows(self.goal_space, len(batch_tasks), batch_tasks, rng)
        old_obs = {k: v for k, v in obs.items() if k in self.old_space.spaces}
        combat = torch.tensor([t == 'combat' for t in batch_tasks])
        with torch.no_grad():
            for path in (heads, static_heads):
                ol, ov = path(self.old, old_obs)
                nl, nv = path(new, obs)
                self.assertTrue(torch.equal(nl[combat], ol[combat]), path.__name__)
                self.assertTrue(torch.equal(nv[combat], ov[combat]), path.__name__)
        return obs, combat

    def test_combat_bitwise_after_migration(self):
        self.check_combat_equal(self.new, np.random.default_rng(0))

    def test_combat_bitwise_with_trained_goal_modules(self):
        new = policy(self.goal_space, 3)
        migrate(self.old, new)
        perturb(new, np.random.default_rng(1))
        obs, combat = self.check_combat_equal(new, np.random.default_rng(2))
        with torch.no_grad():
            nl, nv = heads(new, obs)
            ol, ov = heads(self.old, {k: v for k, v in obs.items() if k in self.old_space.spaces})
        self.assertFalse(torch.equal(nl[~combat], ol[~combat]))
        self.assertFalse(torch.equal(nv[~combat], ov[~combat]))

    def test_gradients(self):
        new = policy(self.goal_space, 4)
        migrate(self.old, new)
        rng = np.random.default_rng(3)
        tasks = ('combat', 'goto_position')
        obs = random_windows(self.goal_space, 2, tasks, rng)
        new.train()
        logits, values = heads(new, obs)
        # COMBAT rows only: nothing reaches the new modules
        new.zero_grad()
        (logits[0].sum() + values[0]).backward(retain_graph=True)
        for name, param in new.named_parameters():
            if _goal_line_key(name):
                self.assertTrue(param.grad is None or not param.grad.any(), name)
        # GOTO rows: the injections, the residual's last layer and the GOTO critic get gradients (the goal encoder only
        # once an injection is non-zero, after the first step)
        new.zero_grad()
        (logits[1].sum() + values[1]).backward()
        grads = {n: p.grad for n, p in new.named_parameters() if _goal_line_key(n)}
        for key in ('features_extractor.goal_fusion.weight', 'features_extractor.goal_query.weight',
                    'features_extractor.goal_conv.weight', 'action_net.nav.2.weight', 'value_net.goto.2.weight'):
            self.assertTrue(grads[key] is not None and grads[key].any(), key)

    def test_observation_goal_fields(self):
        h = VisibleHistory(**observation_options('combat-hp'), goal=True)
        self.assertEqual(h.space.spaces['goal'].shape[-1], GOAL_FIELDS)
        self.assertEqual(len(SOURCES), 3)
        self.assertEqual(h.space.spaces['goal_map'].shape[1:], (2, 9, 15))


if __name__ == '__main__':
    unittest.main()
