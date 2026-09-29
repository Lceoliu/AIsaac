"""C41 (user decisions 2026-09-29): rooms of every Basement shape (the 16x28 terrain canvas, positions in 1x1-room
units), the combat-hp2 profile (-1 per half heart), per-episode player stat offsets (bridge abp-0.2.11) and the room
groups. No game instance or GPU needed."""
import json
import unittest
from pathlib import Path

import numpy as np

CATALOG = Path(__file__).resolve().parent.parent / 'abplus' / 'catalog'


def room_obs(gw=15, gh=9, player=(200.0, 250.0), enemy=(400.0, 300.0), left=60.0, top=140.0):
    """A minimal bridge observation of an empty room gw x gh cells (walls around, 40 px cells) with one Horf."""
    right, bottom = left + (gw - 2) * 40.0, top + (gh - 2) * 40.0
    cells = []
    for index in range(gw * gh):
        row, col = divmod(index, gw)
        inside = int(1 <= col <= gw - 2 and 1 <= row <= gh - 2)
        cells.append((index, left - 20.0 + col * 40.0, top - 20.0 + row * 40.0, 0, inside, inside, 0, 0, 1 - inside, 0))
    p = dict(pos=list(player), size=10.0, hearts=6, max_hearts=6, soul=0, bombs=1, keys=0, coins=0, damage=3.5,
             speed=1.0, shot_speed=1.0, fire_delay_max=10.0, range=260.0, can_fly=False, active_charge=0,
             active_ready=False, active=0, anim='', aframe=0, flip=False)
    e = dict(id=2, type=12, variant=0, subtype=0, pos=list(enemy), size=13.0, size_multi=[1.0, 1.0], coll=4, gcoll=5,
             cdmg=1.0, aframe=0, flip=False, age=6, enemy=True)
    return dict(combat_schema=3, logic_frames=100, players=[p], entities=[e], doors=[],
                room=dict(top_left=[left, top], bottom_right=[right, bottom], gw=gw, gh=gh),
                terrain=dict(width=gw, height=gh, cells=cells))


class TerrainCanvasTest(unittest.TestCase):
    def test_a_1x1_room_is_unchanged_in_the_corner(self):
        from isaac_bridge.transformer_obs import BIG_TERRAIN, VisibleHistory
        obs = room_obs()
        old = VisibleHistory(deadline=True).encode(obs)
        new = VisibleHistory(deadline=True, terrain_shape=BIG_TERRAIN, room_scale='fixed').encode(obs)
        np.testing.assert_array_equal(new['player'], old['player'])
        np.testing.assert_array_equal(new['entities'], old['entities'])
        self.assertEqual(new['terrain'].shape, (7, 16, 28))
        np.testing.assert_array_equal(new['terrain'][:, :9, :15], old['terrain'])
        self.assertEqual(float(new['terrain'][:, 9:, :].sum() + new['terrain'][:, :, 15:].sum()), 0.0)

    def test_a_2x2_room_fills_the_canvas(self):
        from isaac_bridge.transformer_obs import BIG_TERRAIN, ROOM_1X1, VisibleHistory, terrain_channels
        obs = room_obs(28, 16, player=(1000.0, 600.0), enemy=(100.0, 180.0))
        frame = VisibleHistory(deadline=True, terrain_shape=BIG_TERRAIN, room_scale='fixed').encode(obs)
        inside = frame['terrain'][0]
        self.assertEqual(int(inside.sum()), 26 * 14)
        self.assertEqual(float(inside[14, 26]), 1.0)
        # positions in 1x1-room units: the far corner is about 2 rooms across and down
        self.assertAlmostEqual(float(frame['player'][0]), (1000 - 60) / ROOM_1X1[0], places=5)
        self.assertGreater(float(frame['player'][1]), 1.0)
        self.assertAlmostEqual(float(frame['entities'][0][0]), (100 - 1000) / ROOM_1X1[0], places=5)
        with self.assertRaises(ValueError):          # the 1x1 curriculum's 9x15 canvas rejects it
            VisibleHistory(deadline=True).encode(obs)
        with self.assertRaises(ValueError):          # larger than the canvas
            terrain_channels(room_obs(30, 16), BIG_TERRAIN)

    def test_layout_space_and_learner_agree(self):
        import torch
        from isaac_bridge import abplus_worker as W
        from isaac_bridge.gpu_env import decode_frame
        from isaac_bridge.transformer_obs import VisibleHistory
        dtype, _ = W.frame_layout('combat-hp2')
        options = W.observation_options('combat-hp2')
        self.assertEqual(dtype['terrain'].shape, (7, 16, 28))
        self.assertEqual((options['terrain_shape'], options['room_scale']), ((16, 28), 'fixed'))
        space = VisibleHistory(**options).space
        self.assertEqual(space['terrain'].shape, (64, 7, 16, 28))
        rows = np.zeros(2, dtype)
        rows['terrain'][1, 0, 15, 27] = 1.0
        got = decode_frame(torch.from_numpy(rows.view(np.int32).reshape(2, -1).copy()), space, dtype)
        self.assertEqual(tuple(got['terrain'].shape), (2, 7, 16, 28))
        self.assertEqual(float(got['terrain'][1, 0, 15, 27]), 1.0)
        # combat-hp keeps C39's 9x15 layout
        self.assertEqual(W.frame_layout('combat-hp')[0]['terrain'].shape, (7, 9, 15))
        self.assertNotIn('terrain_shape', W.observation_options('combat-hp'))

    def test_the_policy_takes_the_canvas(self):
        import torch
        from isaac_bridge import abplus_worker as W
        from isaac_bridge.transformer_obs import VisibleHistory
        from isaac_bridge.transformer_policy import CombatTransformer
        for profile, flat in (('combat-hp', 32 * 5 * 8), ('combat-hp2', 32 * 8 * 14)):
            net = CombatTransformer(VisibleHistory(**W.observation_options(profile)).space)
            self.assertEqual(net.map_cnn[-2].in_features, flat)
            shape = VisibleHistory(**W.observation_options(profile)).space['terrain'].shape[1:]
            self.assertEqual(tuple(net.map_cnn(torch.zeros(3, *shape)).shape), (3, 128))


class CameraViewTest(unittest.TestCase):
    """combat-hp2-camera (C41 continuing C39): a 1x1 room encodes exactly as combat-hp; a larger room's terrain is the 15x9
    window around the player, stopping at the room's edges."""

    def test_a_1x1_room_encodes_as_combat_hp(self):
        from isaac_bridge.transformer_obs import VisibleHistory
        obs = room_obs()
        old = VisibleHistory(deadline=True).encode(obs)
        new = VisibleHistory(deadline=True, room_scale='camera').encode(obs)
        for key in old:
            np.testing.assert_array_equal(new[key], old[key], err_msg=key)

    def test_the_window_follows_the_player(self):
        from isaac_bridge.transformer_obs import VisibleHistory, camera_origin, terrain_channels
        cases = [((1000.0, 600.0), (7, 13)),     # bottom-right corner of a 2x2 room: the window stops at the edges
                 ((80.0, 160.0), (0, 0)),        # top-left corner
                 ((545.0, 405.0), (3, 6))]       # the middle: cell (col 13, row 7), window from (row 3, col 6)
        for player, origin in cases:
            obs = room_obs(28, 16, player=player, enemy=(300.0, 300.0))
            self.assertEqual(camera_origin(obs, 16, 28), origin)
            frame = VisibleHistory(deadline=True, room_scale='camera').encode(obs)
            full = terrain_channels(obs, (16, 28))
            r, c = origin
            np.testing.assert_array_equal(frame['terrain'], full[:, r:r + 9, c:c + 15])
        # a wide room: the rows never move, the columns do
        obs = room_obs(28, 9, player=(1000.0, 300.0))
        self.assertEqual(camera_origin(obs, 9, 28), (0, 13))

    def test_same_network_as_c39(self):
        from isaac_bridge import abplus_worker as W
        from isaac_bridge.transformer_obs import VisibleHistory
        self.assertEqual(VisibleHistory(**W.observation_options('combat-hp2-camera')).space,
                         VisibleHistory(**W.observation_options('combat-hp')).space)
        self.assertEqual(W.frame_layout('combat-hp2-camera'), W.frame_layout('combat-hp'))
        self.assertEqual(W.observation_options('combat-hp2-camera')['room_scale'], 'camera')
        with self.assertRaises(ValueError):
            VisibleHistory(terrain_shape=(16, 28), room_scale='camera')


class StatNoiseTest(unittest.TestCase):
    SPEC = dict(speed=0.5, damage=1.0, shot_speed=0.5, tears=1.0, range=1.0)

    def test_offsets_per_seed(self):
        from isaac_bridge.abplus_worker import STAT_KEYS, sample_stats
        self.assertIsNone(sample_stats(5, None))
        self.assertEqual(sample_stats(5, self.SPEC), sample_stats(5, self.SPEC))   # a replayed seed replays its stats
        draws = np.array([sample_stats(s, self.SPEC) for s in range(2000)])
        widths = np.array([self.SPEC[k] for k in STAT_KEYS])
        self.assertTrue((np.abs(draws) <= widths).all())
        self.assertTrue((draws.max(0) > 0.9 * widths).all() and (draws.min(0) < -0.9 * widths).all())
        self.assertLess(np.abs(draws.mean(0)).max(), 0.05)
        only = sample_stats(5, dict(damage=1.0))
        self.assertEqual([i for i, v in enumerate(only) if v != 0], [STAT_KEYS.index('damage')])

    def test_bridge_values(self):
        from isaac_bridge.abplus import ROOM_LUA, STAT_BASE, TARGET_LUA, expected_stats, lua_stats
        self.assertEqual(expected_stats((0, 0, 0, 0, 0)), STAT_BASE)
        speed, damage, shot, delay, reach = expected_stats((0.5, -1.0, -0.5, 1.0, 1.0))
        self.assertEqual((speed, damage, shot, reach), (1.5, 2.5, 0.6, 300.0))   # shot speed stops at 0.6 (engine)
        self.assertEqual(delay, 7.0)                                   # 3.73 shots/s -> MaxFireDelay 7
        self.assertEqual(expected_stats((0, 0, 0, -1.0, -1.0))[3:], (16.0, 220.0))   # 1.73 shots/s, 220 px
        self.assertEqual(lua_stats(None), '0.000000, 0.000000, 0.000000, 0.000000, 0.000000')
        for chunk in (ROOM_LUA, TARGET_LUA):   # the offsets come before the episode reseed
            self.assertLess(chunk.index('AbpSetStats({stats})'), chunk.index('os.getenv("ABP_RESEED:{rng_seed}")'))
            self.assertTrue(chunk.rstrip().endswith('" reseeded=" .. tostring(reseeded)'))

    def test_episode_record_fields(self):
        from isaac_bridge import abplus_worker as W
        self.assertEqual(W.META_DTYPE['stats_mod'].shape, (len(W.STAT_KEYS),))


class RewardTest(unittest.TestCase):
    def test_hurt_costs_one(self):
        from isaac_bridge.abplus_reward import HPR, HPR2, REWARDS, CombatHp2, describe_hp2
        self.assertEqual((HPR['hurt'], HPR2['hurt']), (0.5, 1.0))
        self.assertIs(REWARDS['combat-hp2'], CombatHp2)
        r = CombatHp2()
        r.reset({'combat': {'monster_damage': 0.0, 'player_damage': 0.0}}, 'normal')
        got = r.step({'combat': {'monster_damage': 7.0, 'player_damage': 2.0}}, 'running', 4)
        self.assertAlmostEqual(got['hurt'], -2.0)
        self.assertAlmostEqual(got['damage'], 2.0)
        self.assertEqual(CombatHp2(hurt=0.5).hurt, 0.5)
        d = describe_hp2(hurt=1.0)
        self.assertEqual(d['profile'], 'combat-hp2')
        self.assertIn('-1 per half heart', d['hurt'])


class GroupsTest(unittest.TestCase):
    def test_c41_groups(self):
        from isaac_bridge.abplus_groups import load_groups
        groups = load_groups(CATALOG / 'scaling2_groups.json')
        self.assertEqual([g['name'] for g in groups], ['normal', 'normal_big', 'horf_rocks', 'boss'])
        self.assertTrue(all(abs(g['share'] - 0.25) < 1e-9 and g['seconds'] == 180 for g in groups))
        rooms = {r['variant']: r for r in json.loads((CATALOG / 'abplus_basement1_rooms.json').read_text(
            encoding='utf8'))['normal']}
        big = json.loads((CATALOG / 'scaling2_normal_big_rooms.json').read_text(encoding='utf8'))
        self.assertEqual(len(big['normal']), 358)
        self.assertTrue(all(rooms[v]['shape'] != 1 and not rooms[v]['clear'] for v in big['normal']))
        boss = json.loads((CATALOG / 'scaling2_boss_rooms.json').read_text(encoding='utf8'))
        self.assertEqual((len(boss['boss']), boss['weights']['boss']), (82, 1.0))


if __name__ == '__main__':
    unittest.main()
