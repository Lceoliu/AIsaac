"""CombatTransformer.encode_frame_static (the graph sampler's encoder) against encode_frames."""
import unittest

import torch

from isaac_bridge.transformer_obs import VisibleHistory
from isaac_bridge.transformer_policy import CombatTransformer


def frames(space, n, seed, entities):
    g = torch.Generator().manual_seed(seed)
    obs = {}
    for k, s in space.spaces.items():
        shape = (n, 1) + tuple(s.shape[1:])
        if k in ('player_anim', 'active_kind', 'entity_kind', 'entity_anim'):
            obs[k] = torch.randint(0, 64, shape, generator=g, dtype=torch.int32)
        else:
            obs[k] = torch.randn(shape, generator=g) * 0.5
    obs['entity_kind'][..., 0] = torch.randint(1, 300, obs['entity_kind'][..., 0].shape, generator=g, dtype=torch.int32)
    obs['entity_mask'] = (torch.arange(256)[None, None, :] < torch.tensor(entities)[:, None, None]).float()
    obs['terrain'] = (torch.rand(obs['terrain'].shape, generator=g) > 0.6).float()
    obs['previous_action'][..., 0] = torch.randint(0, 45, (n, 1), generator=g).float()
    obs['previous_action'][..., 1:3] = torch.randint(0, 2, (n, 1, 2), generator=g).float()
    obs['previous_action'][..., 3] = 1.0
    obs['history_mask'] = torch.ones((n, 1))
    obs['remaining_time'] = torch.rand((n, 1), generator=g)
    return obs


class StaticEncodeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.space = VisibleHistory(64, 256, deadline=True).space
        self.encoder = CombatTransformer(self.space).eval()

    def test_matches_encode_frames(self):
        # Rows with no entity, one, many and all 256 slots.
        obs = frames(self.space, 5, 1, [0, 1, 17, 64, 256])
        with torch.no_grad():
            eager = self.encoder.encode_frames(obs)[:, 0]
            static = self.encoder.encode_frame_static({k: v[:, 0] for k, v in obs.items()})
        self.assertLess(float((eager - static).abs().max()), 1e-4)

    def test_invalid_frame_encodes_to_zero_like_encode_frames(self):
        obs = frames(self.space, 3, 2, [5, 5, 5])
        obs['history_mask'][1] = 0
        with torch.no_grad():
            eager = self.encoder.encode_frames(obs)[:, 0]
            static = self.encoder.encode_frame_static({k: v[:, 0] for k, v in obs.items()})
        self.assertTrue(torch.equal(static[1], torch.zeros_like(static[1])))
        self.assertLess(float((eager - static).abs().max()), 1e-4)


class TemperedSampleTest(unittest.TestCase):
    """Near-greedy actors (C29): Gumbel-max at a temperature and the behaviour log-probability it stores."""

    def test_temperature_one_is_the_policy_and_low_ones_approach_the_mode(self):
        import torch.nn.functional as F
        from isaac_bridge.graph_sampler import HUGE_NEG, tempered_sample
        g = torch.Generator().manual_seed(3)
        logits = torch.tensor([[1.0, 0.6, 0.2, HUGE_NEG]]).repeat(200000, 1)
        u = torch.rand(logits.shape, generator=g).clamp_(1e-10, 1 - 1e-7)
        a1, lp1 = tempered_sample(logits, u, torch.ones(len(logits), 1))
        plain = torch.argmax(logits - torch.log(-torch.log(u)), -1)
        self.assertTrue(torch.equal(a1, plain))
        self.assertTrue(torch.equal(lp1, F.log_softmax(logits, -1).gather(-1, plain[:, None])[:, 0]))
        a, lp = tempered_sample(logits, u, torch.full((len(logits), 1), 0.25))
        expected = F.softmax(logits[0] / 0.25, -1)
        freq = torch.bincount(a, minlength=4).float() / len(a)
        self.assertLess(float((freq - expected).abs().max()), 0.005)
        self.assertEqual(int((a == 3).sum()), 0)                    # masked stays masked
        self.assertTrue(torch.allclose(lp, torch.log(expected)[a]))
        self.assertGreater(float(expected[0]), 0.8)                  # near the mode (untempered: 0.47)


if __name__ == '__main__':
    unittest.main()
