"""combat-v5 firing geometry (isaac_bridge/abplus_geometry.py); no engine needed."""
import math
import unittest

from isaac_bridge.abplus_geometry import D_FIRE_MAX, TEAR_RADIUS, fire_geometry

R = 260.0


def obs(px, py, *enemies):
    """enemies: (x, y, size, lineage, blocking)."""
    return {'players': [{'pos': [px, py], 'range': R}],
            'entities': [{'pos': [x, y], 'size': size, 'lineage': lineage, 'blocking': blocking, 'enemy': True}
                         for x, y, size, lineage, blocking in enemies]}


class FireGeometryTest(unittest.TestCase):
    def test_distance_to_the_firing_bands(self):
        tau = 13 + TEAR_RADIUS
        self.assertEqual(fire_geometry(obs(100, 300, (300, 300, 13, True, True)))['d_fire'], 0.0)
        # Off the horizontal band by 10 px (still within range along x): 10.
        self.assertAlmostEqual(fire_geometry(obs(100, 300 + tau + 10, (300, 300, 13, True, True)))['d_fire'], 10.0)
        # In the band but 30 px beyond the tear range.
        self.assertAlmostEqual(fire_geometry(obs(300 - R - 30, 300, (300, 300, 13, True, True)))['d_fire'], 30.0)
        # Both: sqrt(30^2 + 40^2) from the horizontal band; the vertical band is farther.
        self.assertAlmostEqual(fire_geometry(obs(300 - R - 30, 300 + tau + 40, (300, 300, 13, True, True)))['d_fire'], 50.0)
        # The nearer band wins: straight above the target, the vertical band is 0 away.
        self.assertEqual(fire_geometry(obs(300, 100, (300, 300, 13, True, True)))['d_fire'], 0.0)
        self.assertEqual(fire_geometry(obs(0, 0, (5000, 5000, 13, True, True)))['d_fire'], D_FIRE_MAX)

    def test_lineage_targets_first_then_blocking_then_held(self):
        # A lineage NPC far away beats a nearer blocking-only one.
        g = fire_geometry(obs(100, 100, (400, 400, 13, True, True), (110, 100, 13, False, True)))
        self.assertGreater(g['d_fire'], 0)
        # No lineage NPC left: every doors-blocking NPC is a target.
        self.assertEqual(fire_geometry(obs(100, 100, (110, 100, 13, False, True)))['d_fire'], 0.0)
        # Nothing visible: d_fire holds, no labels.
        g = fire_geometry(obs(100, 100), previous=123.0)
        self.assertEqual((g['d_fire'], g['aim'], sum(g['approach']), g['has_target']), (123.0, 0, 0, False))
        self.assertEqual(fire_geometry(obs(100, 100))['d_fire'], D_FIRE_MAX)

    def test_aim_label_is_the_shot_that_hits_an_aligned_target(self):
        e = (300, 300, 13, True, True)
        self.assertEqual(fire_geometry(obs(100, 300, e))['aim'], 2)    # right
        self.assertEqual(fire_geometry(obs(500, 305, e))['aim'], 4)    # left
        self.assertEqual(fire_geometry(obs(298, 100, e))['aim'], 3)    # down
        self.assertEqual(fire_geometry(obs(300, 500, e))['aim'], 1)    # up
        self.assertEqual(fire_geometry(obs(100, 400, e))['aim'], 0)    # not aligned
        self.assertEqual(fire_geometry(obs(300 - R - 30, 300, e))['aim'], 0)  # aligned, out of range
        # Two aligned targets: the nearer one decides.
        self.assertEqual(fire_geometry(obs(300, 300, (100, 300, 13, True, True), (300, 360, 13, True, True)))['aim'], 3)

    def test_approach_marks_the_moves_that_shorten_d_fire(self):
        # 40 px below the target's row (horizontal band 18.8 px away) and 100 px right of its column
        # (vertical band 78.8 px away): up, up-left and up-right close the gap; down moves away;
        # left/right only change the farther band; stop never counts.
        g = fire_geometry(obs(400, 340, (300, 300, 13, True, True)))
        self.assertAlmostEqual(g['d_fire'], 40 - 13 - TEAR_RADIUS)
        self.assertEqual(g['approach'], [0, 1, 1, 0, 0, 0, 0, 0, 1])
        # Already aligned: no move shortens a zero distance.
        self.assertEqual(sum(fire_geometry(obs(100, 300, (300, 300, 13, True, True)))['approach']), 0)

    def test_tear_stopping_grid_cuts_the_band(self):
        # Cell (0, 0) at (40, 120); target in cell (5, 4) = (240, 280); player in (10, 4) = (440, 280).
        def room(px, py, *grid):
            o = obs(px, py, (240.0, 280.0, 13, True, True))
            o['terrain'] = {'cells': [[0, 40.0, 120.0, 0, 0, 0, 0, 0, 0, 0]], 'width': 15}
            o['room'] = {'gw': 15}
            o['grid'] = [[row * 15 + col, gtype, 0, state, 3, 40.0 + 40 * col, 120.0 + 40 * row]
                         for col, row, gtype, state in grid]
            return o
        self.assertEqual(fire_geometry(room(440, 280))['aim'], 4)                  # open row: fire left
        blocked = fire_geometry(room(440, 280, (8, 4, 2, 1)))                      # a rock in (8, 4)
        self.assertEqual(blocked['aim'], 0)
        # The band now ends a tear radius short of the rock's near edge: x <= 360 - 20 - 8.16.
        self.assertAlmostEqual(blocked['d_fire'], 440 - (360 - 20 - TEAR_RADIUS))
        self.assertEqual([m for m, v in enumerate(blocked['approach']) if v], [6, 7, 8])
        for grid in ((8, 4, 14, 0), (8, 4, 2, 2), (8, 4, 12, 0)):                   # poop, rubble, TNT
            self.assertEqual(fire_geometry(room(440, 280, grid))['aim'], 4, grid)
        self.assertEqual(fire_geometry(room(440, 280, (8, 4, 3, 1)))['aim'], 0)    # metal block
        # Same along the column: a rock in (5, 6) between the target and a player below in (5, 7).
        self.assertEqual(fire_geometry(room(240, 400))['aim'], 1)
        self.assertEqual(fire_geometry(room(240, 400, (5, 6, 2, 1)))['aim'], 0)
        # A rock on the other side of the target does not matter.
        self.assertEqual(fire_geometry(room(440, 280, (3, 4, 2, 1)))['aim'], 4)


def raw_obs(px, py, enemies, frame=10):
    """A minimal bridge observation (15x9 room, cell 0 at (40, 120)); enemies: (x, y, lineage, blocking)."""
    cells = [[r * 15 + c, 40.0 + 40 * c, 120.0 + 40 * r, 0, int(0 < r < 8 and 0 < c < 14), int(0 < r < 8 and 0 < c < 14),
              0, 0, 0, 0] for r in range(9) for c in range(15)]
    player = dict(pos=[px, py], size=10.0, hearts=6, max_hearts=6, soul=0, bombs=1, keys=0, coins=0, damage=3.5,
                  speed=1.0, shot_speed=1.0, fire_delay_max=10.0, range=R, can_fly=False, active_charge=0,
                  active_ready=False, active=0, anim='', aframe=0, flip=False)
    entities = [dict(id=i + 1, type=12, variant=0, subtype=0, pos=[x, y], size=13.0, size_multi=[1.0, 1.0], coll=4,
                     gcoll=5, cdmg=1.0, aframe=0, flip=False, age=frame, enemy=True, vulnerable=True, boss=False,
                     lineage=lineage, blocking=blocking) for i, (x, y, lineage, blocking) in enumerate(enemies)]
    return dict(combat_schema=3, logic_frames=frame, players=[player], entities=entities,
                room=dict(top_left=[60.0, 140.0], bottom_right=[580.0, 420.0]),
                terrain=dict(width=15, height=9, cells=cells), doors=[],
                combat=dict(blocking_hp=10.0 * len(enemies), blocking_count=float(len(enemies))))


class CombatV5ObservationTest(unittest.TestCase):
    def history(self, history=64, capacity=256):
        from isaac_bridge.abplus_worker import observation_options
        from isaac_bridge.transformer_obs import VisibleHistory
        return VisibleHistory(history, capacity, **observation_options('combat-v5'))

    def test_flags_distance_labels_and_one_hot_previous_action(self):
        h = self.history()
        f = h.encode(raw_obs(320, 360, [(320, 200, True, True), (100, 200, False, True)]))
        self.assertEqual(f['entity_flags'][:2].tolist(), [[1.0, 1.0], [0.0, 1.0]])
        self.assertEqual(float(f['fire_distance']), 0.0)        # straight below the lineage Horf
        self.assertEqual(float(f['aim_label']), 1.0)            # shoot up
        self.assertEqual(f['previous_action'].tolist(), [0.0] * 18)
        h.set_previous_action(7, 0, 0)                          # move 1 (up), shoot 2 (right)
        f = h.encode(raw_obs(400, 360, [(320, 200, True, True)], frame=12))
        self.assertEqual(sorted(i for i, v in enumerate(f['previous_action']) if v), [1, 11, 14, 16])
        self.assertAlmostEqual(float(f['fire_distance']), (80 - 13 - TEAR_RADIUS) / 40, places=5)
        self.assertEqual(float(f['aim_label']), 0.0)
        self.assertEqual([i for i, v in enumerate(f['approach']) if v], [6, 7, 8])   # the leftward moves

    def test_frame_record_holds_every_observation_key(self):
        from isaac_bridge import abplus_worker as W
        dtype, keys = W.frame_layout('combat-v5')
        self.assertEqual(dtype['previous_action'].shape, (18,))
        for name, shape in (('entity_flags', (256, 2)), ('approach', (9,)), ('fire_distance', ()),
                            ('aim_label', ()), ('hit', ())):
            self.assertEqual(dtype[name].shape, shape)
        self.assertLessEqual(set(self.history().space.spaces), set(dtype.names))
        self.assertEqual(W.frame_layout('combat-v4')[0]['previous_action'].shape, (4,))

    def test_fire_distance_reaches_the_value_branch_only(self):
        try:
            import torch
            from gymnasium import spaces
            from isaac_bridge.transformer_policy import CombatTransformer, GeometryPolicy
        except ImportError:
            self.skipTest('torch / stable-baselines3 not installed')
        h = self.history(8, 16)
        window = h.append(raw_obs(400, 360, [(320, 200, True, True)]))
        policy = GeometryPolicy(h.space, spaces.MultiDiscrete([9, 5, 2, 2]), lambda _: 1e-4,
                                net_arch=dict(pi=[32], vf=[32]), features_extractor_class=CombatTransformer,
                                features_extractor_kwargs=dict(features_dim=32, layers=1, heads=2), normalize_images=False)
        policy.set_training_mode(False)

        def heads(obs):
            with torch.no_grad():
                features = policy.features_extractor({k: torch.as_tensor(v)[None].float() for k, v in obs.items()})
                pi, vf = policy.mlp_extractor(features)
                return features, policy.action_net(pi), policy.value_net(vf), policy.aux_outputs(features)

        features, logits, value, (aim, distance) = heads(window)
        self.assertEqual(features.shape, (1, 33))
        self.assertEqual((aim.shape, distance.shape), ((1, 5), (1,)))
        moved = {k: v.copy() for k, v in window.items()}
        moved['fire_distance'][0] += 5.0
        _, logits2, value2, (aim2, distance2) = heads(moved)
        self.assertTrue(torch.equal(logits, logits2))
        self.assertTrue(torch.equal(aim, aim2) and torch.equal(distance, distance2))
        self.assertFalse(torch.equal(value, value2))


if __name__ == '__main__':
    unittest.main()
