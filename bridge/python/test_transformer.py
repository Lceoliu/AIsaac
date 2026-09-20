import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from isaac_bridge.transformer_obs import (VisibleHistory, TransformerMonstroEnv, ENTITY_FIELDS,
                                         terrain_channels, animation_bytes)
from isaac_bridge.transformer_policy import CombatTransformer
from test_monstro_gym import RecordedBridge

torch.set_num_threads(1)


def bridge_v3():
    bridge = RecordedBridge()
    bridge.original['combat_schema'] = 3
    bridge.original['players'][0]['active_ready'] = False
    for cell in bridge.original['terrain']['cells']:
        cell.extend([int(cell[3] >= 2), 0])
    return bridge


def observations(length=4, history=8, capacity=16):
    bridge = bridge_v3()
    obs, _ = bridge.reset_monstro('fixed')
    window = VisibleHistory(history, capacity)
    for t in range(length):
        current = copy.deepcopy(obs)
        current['logic_frames'] += t*2
        current['players'][0]['pos'][0] += t
        for e in current['entities']:
            e['age'] += t*2
            e['pos'][0] += t*4
        result = window.append(current)
    return window, result


def tensors(obs):
    return {k: torch.as_tensor(v[None]).float() for k, v in obs.items()}


class TransformerObservationTest(unittest.TestCase):
    def test_schema_required_and_no_overflow_truncation(self):
        old = RecordedBridge().original
        with self.assertRaisesRegex(ValueError, 'combat_schema'):
            VisibleHistory().append(old)
        bridge = bridge_v3()
        obs = copy.deepcopy(bridge.original)
        obs['entities'] *= 17
        with self.assertRaisesRegex(ValueError, 'overflow'):
            VisibleHistory(capacity=16).append(obs)

    def test_motion_is_observed_not_engine_velocity_and_reset_clears(self):
        window, result = observations()
        vx = ENTITY_FIELDS.index('vx')
        valid = ENTITY_FIELDS.index('motion_valid')
        self.assertEqual(result['entities'][0, 0, valid], 0)
        width = window.previous['room']['bottom_right'][0]-window.previous['room']['top_left'][0]
        self.assertAlmostEqual(result['entities'][1, 0, vx], 2/width)
        self.assertEqual(result['entities'][1, 0, valid], 1)
        window.clear()
        after = window.append(bridge_v3().original)
        self.assertEqual(after['history_mask'].sum(), 1)
        self.assertEqual(after['entities'][0, 0, valid], 0)
        self.assertFalse(after['previous_action'].any())

    def test_privileged_fields_do_not_enter_actor(self):
        a = copy.deepcopy(bridge_v3().original)
        b = copy.deepcopy(a)
        b['room']['alive'] = 900
        b['combat']['enemy_damage'] = 99999
        b['players'][0]['vel'] = [900, 900]
        b['players'][0]['invulnerable'] = not b['players'][0]['invulnerable']
        for e in b['entities']:
            e.update(vel=[999, 999], hp=999, state=99, seed=999)
        left, right = VisibleHistory().append(a), VisibleHistory().append(b)
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])

    def test_worker_uptime_is_not_in_the_observation(self):
        early = copy.deepcopy(bridge_v3().original)
        late = copy.deepcopy(early)
        late['logic_frames'] += 10_000_000
        a, b = VisibleHistory(), VisibleHistory()
        for step in range(3):
            left, right = a.append(copy.deepcopy(early)), b.append(copy.deepcopy(late))
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
            early['logic_frames'] += 2
            late['logic_frames'] += 2

    def test_terrain_channels_and_laser_segments(self):
        obs = copy.deepcopy(bridge_v3().original)
        cell = obs['terrain']['cells'][67]
        cell[5:10] = [0, 0, 1, 1, 1]
        grid = terrain_channels(obs)
        np.testing.assert_array_equal(grid[[1, 3, 4, 5], 4, 7], [0, 0, 1, 1])
        laser = copy.deepcopy(obs['entities'][0])
        laser.update(type=7, laser={'circle': False, 'radius': 0, 'end': [300, 250],
                                   'samples': [[200, 200], [250, 240], [300, 250]]})
        obs['entities'] = [laser]
        encoded = VisibleHistory().append(obs)
        self.assertEqual(encoded['entity_mask'][0].sum(), 2)
        self.assertGreater(encoded['entities'][0, 0, ENTITY_FIELDS.index('laser_length')], 0)
        self.assertFalse(np.array_equal(encoded['entities'][0, 0], encoded['entities'][0, 1]))

    def test_long_animation_is_not_silently_aliased(self):
        with self.assertRaises(ValueError):
            animation_bytes('x'*33)

    def test_joint_action_masks_and_rewind_boundary(self):
        bridge = bridge_v3()
        env = TransformerMonstroEnv(bridge=bridge, history=8, entity_capacity=16)
        try:
            first, _ = env.reset()
            self.assertEqual(bridge.action_repeat, 2)
            self.assertEqual(env.action_space.nvec.tolist(), [45, 2, 2])
            self.assertFalse(env.action_masks()[48])
            self.assertEqual(env.decode_action([44, 0, 0]), [8, 4, 0, 0])
            obs, _, _, _, _ = env.step([44, 0, 0])
            self.assertEqual(obs['history_mask'].sum(), 2)
            np.testing.assert_array_equal(obs['previous_action'][1], [44, 0, 0, 1])
            bridge.obs['players'][0]['bombs'] = 0
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                env.step([0, 1, 0])
            reset, _ = env.reset()
            self.assertEqual(reset['history_mask'].sum(), 1)
            self.assertFalse(reset['previous_action'].any())
            self.assertTrue(env.observation_space.contains(reset))
        finally:
            env.close()

    def test_missing_entity_breaks_motion_track(self):
        window, _ = observations(length=1)
        vanished = copy.deepcopy(window.previous)
        vanished['logic_frames'] += 2
        entity = vanished['entities'][0]
        vanished['entities'] = []
        window.append(vanished)
        visible = copy.deepcopy(vanished)
        visible['logic_frames'] += 2
        entity['age'] += 4
        visible['entities'] = [entity]
        result = window.append(visible)
        self.assertEqual(result['entities'][2, 0, ENTITY_FIELDS.index('motion_valid')], 0)


class TransformerNetworkTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.window, self.raw = observations()
        self.obs = tensors(self.raw)
        self.net = CombatTransformer(self.window.space).eval()

    def test_causal_attention_cannot_read_future(self):
        future = {k: v.clone() for k, v in self.obs.items()}
        future['player'][:, 3] += 20
        with torch.no_grad():
            before = self.net.sequence_features(self.obs)
            after = self.net.sequence_features(future)
        torch.testing.assert_close(before[:, :3], after[:, :3], atol=1e-5, rtol=1e-5)
        self.assertGreater(float((before[:, 3]-after[:, 3]).abs().max()), 1e-5)

    def test_history_affects_current_prediction(self):
        past = {k: v.clone() for k, v in self.obs.items()}
        past['player'][:, 0] += 5
        with torch.no_grad():
            difference = (self.net(self.obs)-self.net(past)).abs().max()
        self.assertGreater(float(difference), 1e-5)

    def test_padding_and_entity_order_do_not_change_prediction(self):
        # Three distinct real entities, not merely one entity moved through padding.
        for slot in (1, 2):
            for key in ['entities', 'entity_kind', 'entity_anim', 'entity_mask']:
                self.obs[key][:, :, slot] = self.obs[key][:, :, 0]
            self.obs['entities'][:, :, slot, 0] += slot/10
            self.obs['entity_kind'][:, :, slot, 0] = (2, 9)[slot-1]
        other = {k: v.clone() for k, v in self.obs.items()}
        # Move the real entity to the last slot, then poison unused numeric padding.
        for key in ['entities', 'entity_kind', 'entity_anim', 'entity_mask']:
            other[key] = other[key].flip(2)
        other['entities'][~other['entity_mask'].bool()] = 99999
        other['player'][:, 4:] = 99999
        with torch.no_grad():
            torch.testing.assert_close(self.net(self.obs), self.net(other), atol=1e-5, rtol=1e-5)

    def test_empty_room_and_all_encoder_gradients(self):
        self.net.train()
        output = self.net(self.obs)
        (output.square().mean()+output[:, 0].mean()).backward()
        for prefix in ['map_cnn', 'entity_attention', 'entity.', 'entity_type', 'character', 'temporal', 'player']:
            gradients = [p.grad for n, p in self.net.named_parameters() if n.startswith(prefix) and p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        self.obs['entity_mask'].zero_()
        self.assertTrue(torch.isfinite(self.net(self.obs)).all())

    def test_masked_ppo_training_and_checkpoint(self):
        env = TransformerMonstroEnv(bridge=bridge_v3(), history=8, entity_capacity=16,
                                    max_episode_frames=12)
        try:
            model = MaskablePPO('MultiInputPolicy', env, n_steps=8, batch_size=4, n_epochs=1,
                policy_kwargs={'features_extractor_class': CombatTransformer,
                               'net_arch': dict(pi=[32], vf=[32]), 'normalize_images': False}, seed=1)
            before = {n: p.detach().clone() for n, p in model.policy.named_parameters()}
            model.learn(16)
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.policy.named_parameters()))
            obs, _ = env.reset()
            mask = env.action_masks()
            with tempfile.TemporaryDirectory(dir=Path(__file__).parents[2]/'runs') as directory:
                path = Path(directory)/'model'
                model.save(path)
                restored = MaskablePPO.load(path)
                a, _ = model.predict(obs, deterministic=True, action_masks=mask)
                b, _ = restored.predict(obs, deterministic=True, action_masks=mask)
                np.testing.assert_array_equal(a, b)
                self.assertEqual(a[2], 0)
                env.step(a)
        finally:
            env.close()

    def test_full_window_is_not_misclassified_as_an_rgb_image(self):
        env = TransformerMonstroEnv(bridge=bridge_v3())
        try:
            model = MaskablePPO('MultiInputPolicy', env, n_steps=2, batch_size=2,
                policy_kwargs={'features_extractor_class': CombatTransformer,
                               'net_arch': [], 'normalize_images': False})
            obs = model.get_env().reset()
            self.assertEqual(obs['entity_anim'].shape, (1, 64, 256, 32))
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            self.assertEqual(action.shape, (1, 3))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
