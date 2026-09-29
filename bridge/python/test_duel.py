"""The duel arena (isaac_bridge/abplus_duel.py, bridge abp-0.2.9): first-person views, their symmetry, the reward per side,
the episode outcomes, the duel slots' step messages, the scripted duellist and the duel rooms."""
import json
import unittest
from pathlib import Path

import numpy as np

from test_abplus_geometry import raw_obs

HERE = Path(__file__).resolve().parent
DUEL_ROOMS = HERE.parent / 'abplus' / 'catalog' / 'duel_rooms.json'
NPC_ID, PLAYER_ID = 16, 1


def side(shots=0, hits=0, misses=0, miss_units=0, miss_streak=0, damage=0.0, hurt=0, hurt_amount=0.0):
    return dict(shots=shots, hits=hits, misses=misses, miss_units=miss_units, miss_streak=miss_streak, damage=damage,
                hurt=hurt, hurt_amount=hurt_amount)


def duel_obs(player, npc, hearts=6, hp=21.0, tears=(), projectiles=(), player_side=None, npc_side=None, frame=10,
             player_dead=False, npc_dead=False):
    """A duel observation: the player and the duel NPC at the given positions, the player's tears and the NPC's
    projectiles as (x, y, height), the duel counters of both sides (player: damage in NPC HP, hurt in half hearts;
    NPC: damage in half hearts, hurt in NPC HP)."""
    o = raw_obs(player[0], player[1], [], frame=frame)
    o['players'][0].update(id=PLAYER_ID, hearts=hearts, bombs=0, dead=player_dead, ptype=0, luck=0.0, invulnerable=False)
    npc_rec = dict(id=NPC_ID, type=11, variant=1, subtype=0, pos=list(npc), size=10.0, size_multi=[1.0, 1.0], coll=4,
                   gcoll=5, cdmg=0.0, aframe=0, flip=False, age=frame, enemy=True, vulnerable=True, boss=False, champion=-1,
                   lineage=True, blocking=True)
    shots = []
    for i, (x, y, h) in enumerate(tears):
        shots.append(dict(id=100 + i, type=2, variant=0, subtype=0, pos=[x, y], size=8.165, size_multi=[1.0, 1.0],
                          coll=4, gcoll=4, cdmg=3.5, aframe=0, flip=False, age=frame, height=h, fall=0.3,
                          scale=1.0206242799759))
    for i, (x, y, h) in enumerate(projectiles):
        shots.append(dict(id=200 + i, type=9, variant=0, subtype=0, pos=[x, y], size=8.165, size_multi=[1.0, 1.0],
                          coll=4, gcoll=4, cdmg=0.0, aframe=0, flip=False, age=frame, height=h, fall=0.3, scale=1.0,
                          projectile=True))
    o['entities'] = [npc_rec] + shots
    o['duel'] = dict(active=True, dead=npc_dead, npc=NPC_ID, pos=list(npc), size=10.0, hp=0.0 if npc_dead else hp,
                     max_hp=21.0, iframes=0, player_iframes=0, cooldown=-1, vel=[0.0, 0.0], move=0, shoot=0,
                     player=player_side or side(), npc_side=npc_side or side())
    o['events'] = dict(damage=0, tears=0, npc_deaths=0, clears=0)
    return o


def entity_rows(frame):
    """The valid entity rows of an encoded frame (features, kind, flags), rounded and sorted."""
    valid = np.asarray(frame['entity_mask']) > 0
    rows = np.concatenate([frame['entities'][valid], frame['entity_kind'][valid], frame['entity_flags'][valid],
                           frame['entity_anim'][valid]], 1)
    return sorted(tuple(round(float(x), 5) for x in row) for row in rows)


def mirrored(player, npc, hearts, hp, tears, projectiles, player_side, npc_side, frame=10):
    """The same duel with the roles swapped: what the NPC sees in `a` the player sees in `b`, and the other way round."""
    a = duel_obs(player, npc, hearts, hp, tears, projectiles, side(**player_side), side(**npc_side), frame)
    b = duel_obs(npc, player, hp / 3.5, hearts * 3.5, projectiles, tears,
                 side(**{**npc_side, 'damage': npc_side['damage'] * 3.5, 'hurt_amount': npc_side['hurt_amount'] / 3.5}),
                 side(**{**player_side, 'damage': player_side['damage'] / 3.5,
                         'hurt_amount': player_side['hurt_amount'] * 3.5}), frame)
    return a, b


class DuelViewTest(unittest.TestCase):
    def test_npc_view_swaps_the_roles(self):
        from isaac_bridge.abplus_duel import PROJECTILE_FIELDS, TEAR_FIELDS, duel_views
        o = duel_obs((200, 280), (440, 280), hearts=5, hp=17.5, tears=[(300, 279, -15.0)], projectiles=[(350, 281, -20.0)],
                     player_side=side(shots=4, hits=1, misses=2, miss_units=3, miss_streak=1, damage=3.5, hurt=1,
                                      hurt_amount=1.0),
                     npc_side=side(shots=3, hits=1, misses=1, miss_units=1, miss_streak=1, damage=1.0, hurt=1,
                                   hurt_amount=3.5))
        mine, theirs = duel_views(o)
        self.assertEqual(mine['players'][0]['pos'], [200, 280])
        self.assertEqual(mine['combat']['lineage_damage'], 3.5)       # NPC HP
        self.assertEqual(mine['combat']['blocking_hp'], 17.5)
        self.assertEqual(mine['combat']['player_damage'], 1.0)        # half hearts
        self.assertEqual(mine['events']['tears'], 4.0)
        me = theirs['players'][0]
        self.assertEqual(me['pos'], [440, 280])
        self.assertEqual(me['id'], NPC_ID)
        self.assertAlmostEqual(me['hearts'], 5.0)                     # 17.5 HP = 5 half hearts
        self.assertEqual((me['bombs'], me['dead'], me['range'], me['damage']), (0, False, o['players'][0]['range'], 3.5))
        other = theirs['entities'][0]
        self.assertEqual((other['type'], other['variant'], other['id'], other['pos']), (11, 1, PLAYER_ID, [200, 280]))
        self.assertTrue(other['lineage'] and other['blocking'] and other['enemy'])
        kinds = {(e['type'], e['pos'][0]) for e in theirs['entities'][1:]}
        self.assertEqual(kinds, {(2, 350), (9, 300)})                 # own projectile -> tear, the player's tear -> projectile
        own = next(e for e in theirs['entities'] if e['type'] == 2)
        self.assertEqual({k: own[k] for k in TEAR_FIELDS}, TEAR_FIELDS)
        self.assertNotIn('projectile', own)
        theirs_shot = next(e for e in theirs['entities'] if e['type'] == 9)
        self.assertEqual({k: theirs_shot[k] for k in PROJECTILE_FIELDS}, PROJECTILE_FIELDS)
        self.assertTrue(theirs_shot['projectile'])
        self.assertNotIn(NPC_ID, [e['id'] for e in theirs['entities']])
        c = theirs['combat']
        self.assertEqual((c['lineage_damage'], c['blocking_hp'], c['player_damage']), (3.5, 5 * 3.5, 1.0))
        self.assertEqual((c['tear_hits'], c['tear_misses'], c['miss_units'], c['miss_streak']), (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(theirs['events']['tears'], 3.0)

    def test_the_views_of_a_mirrored_duel_are_the_same(self):
        """Each side sees in its view exactly what the other side sees in the duel with the roles swapped, frame by frame
        (positions, velocities, hearts, shots, counters, firing geometry)."""
        from isaac_bridge.abplus_duel import duel_views
        from isaac_bridge.abplus_worker import observation_options
        from isaac_bridge.transformer_obs import VisibleHistory
        ps = dict(shots=6, hits=1, misses=3, miss_units=5, miss_streak=2, damage=3.5, hurt=1, hurt_amount=1.0)
        ns = dict(shots=5, hits=1, misses=2, miss_units=3, miss_streak=0, damage=1.0, hurt=1, hurt_amount=3.5)
        histories = {k: VisibleHistory(64, 256, **observation_options('combat-hitrate-miss')) for k in 'ab'}
        mirror = {k: VisibleHistory(64, 256, **observation_options('combat-hitrate-miss')) for k in 'ab'}
        steps = [((200, 280), (440, 240), 20), ((205, 283), (433, 240), 22), ((212, 290), (426, 236), 24)]
        for player, npc, frame in steps:
            a, b = mirrored(player, npc, 5, 17.5, [(player[0] + 60, player[1] - 4, -15.0)],
                            [(npc[0] - 40, npc[1] + 3, -21.0)], ps, ns, frame)
            a_player, a_npc = duel_views(a)
            b_player, b_npc = duel_views(b)
            for left, right in ((histories['a'].encode(a_player), mirror['b'].encode(b_npc)),
                                (histories['b'].encode(a_npc), mirror['a'].encode(b_player))):
                self.assertEqual(set(left), set(right))
                for key in left:
                    if key.startswith('entit'):
                        continue
                    np.testing.assert_allclose(np.asarray(left[key], np.float64), np.asarray(right[key], np.float64),
                                               atol=1e-6, err_msg=key)
                # The entity attention does not depend on the entities' order: compare the valid rows as a set.
                self.assertEqual(entity_rows(left), entity_rows(right))
                self.assertEqual(len(entity_rows(left)), 3)

    def test_views_need_an_active_duel(self):
        from isaac_bridge.abplus_duel import duel_views
        from isaac_bridge.env import BridgeError
        o = duel_obs((200, 280), (440, 280))
        o['duel']['active'] = False
        with self.assertRaises(BridgeError):
            duel_views(o)
        o = duel_obs((200, 280), (440, 280))
        o['entities'] = o['entities'][1:]
        with self.assertRaises(BridgeError):
            duel_views(o)


class DuelRewardTest(unittest.TestCase):
    def reward(self):
        from isaac_bridge.abplus_reward import REWARDS
        return REWARDS['combat-hitrate-miss'](hit_hp=1.25, align=0.0, gamma=0.9995, miss=0.01, deadline_s=30.0,
                                              hurt_rest=True)

    def test_a_hit_pays_the_same_on_either_side(self):
        """The player's tear hitting the NPC and the NPC's shot hitting the player: the same reward to the hitter and
        the same outcome to the side hit, whichever side it is."""
        from isaac_bridge.abplus_duel import duel_outcomes, duel_views
        start = duel_obs((200, 280), (440, 280), player_side=side(shots=1), npc_side=side(shots=1))
        after_player_hits = duel_obs((200, 280), (440, 280), hp=17.5, frame=12,
                                     player_side=side(shots=1, hits=1, damage=3.5),
                                     npc_side=side(shots=1, hurt=1, hurt_amount=3.5))
        after_npc_hits = duel_obs((200, 280), (440, 280), hearts=5, frame=12,
                                  player_side=side(shots=1, hurt=1, hurt_amount=1.0),
                                  npc_side=side(shots=1, hits=1, damage=1.0))
        paid = {}
        for name, after in (('player_hits', after_player_hits), ('npc_hits', after_npc_hits)):
            rewards = [self.reward(), self.reward()]
            for r, v in zip(rewards, duel_views(start)):
                r.reset(v, 'normal')
            views = duel_views(after)
            outcomes = duel_outcomes(views, 2, 900, hurt_ends=True)
            paid[name] = [r.step(v, o, 2) for r, v, o in zip(rewards, views, outcomes)], outcomes
        (hitter_p, hit_n), outcomes_p = paid['player_hits']
        (hit_p, hitter_n), outcomes_n = paid['npc_hits']
        self.assertEqual(outcomes_p, ['win', 'hurt'])
        self.assertEqual(outcomes_n, ['hurt', 'win'])
        self.assertEqual(hitter_p, hitter_n)
        self.assertEqual(hit_n, hit_p)
        self.assertAlmostEqual(hitter_p['hit'], 3.5 / 1.25)
        self.assertAlmostEqual(hit_p['rest'], -2.0 * (30.0 - 2 / 30))   # the side hit pays the rest of the deadline

    def test_outcomes(self):
        from isaac_bridge.abplus_duel import duel_outcomes, duel_views
        v = duel_views(duel_obs((200, 280), (440, 280)))
        self.assertEqual(duel_outcomes(v, 10, 900, True), ['running', 'running'])
        self.assertEqual(duel_outcomes(v, 900, 900, True), ['time_limit', 'time_limit'])
        both = duel_views(duel_obs((200, 280), (440, 280), hearts=5, hp=17.5,
                                   player_side=side(hurt=1, hurt_amount=1.0, hits=1, damage=3.5),
                                   npc_side=side(hurt=1, hurt_amount=3.5, hits=1, damage=1.0)))
        self.assertEqual(duel_outcomes(both, 10, 900, True), ['hurt', 'hurt'])
        self.assertEqual(duel_outcomes(both, 10, 900, False), ['running', 'running'])   # the game rules: to a death
        dead = duel_views(duel_obs((200, 280), (440, 280), npc_dead=True,
                                   npc_side=side(hurt=6, hurt_amount=21.0), player_side=side(hits=6, damage=21.0)))
        self.assertEqual(duel_outcomes(dead, 10, 900, False), ['win', 'death'])
        self.assertEqual(dead[1]['combat']['lineage_kills'], 0.0)
        self.assertEqual(dead[0]['combat']['lineage_kills'], 1.0)
        self.assertEqual(dead[0]['combat']['blocking_count'], 0.0)


class DuelPlumbingTest(unittest.TestCase):
    def test_step_messages_pack_both_slots_of_a_game(self):
        import struct
        try:
            from isaac_bridge.abplus_vec import step_messages
        except ImportError:
            self.skipTest('torch / stable-baselines3 not installed')
        rows = [[7, 0, 0], [21, 0, 0], [3, 1, 0], [44, 0, 1]]
        self.assertEqual(step_messages(rows), [b'S' + struct.pack('<3i', *r) for r in rows])
        two = step_messages(rows, 2)
        self.assertEqual(len(two), 2)
        self.assertEqual(struct.unpack('<6i', two[1][1:]), (3, 1, 0, 44, 0, 1))

    def test_duel_rooms_and_placement(self):
        from isaac_bridge.abplus import duel_cells
        from isaac_bridge.abplus_duel import duel_tasks
        spec = json.loads(DUEL_ROOMS.read_text(encoding='utf8'))
        arms = spec['duel']['arms']
        self.assertEqual([a['name'] for a in arms], ['open', 'rocks'])
        self.assertEqual([len(a['rooms']) for a in arms], [63, 292])
        self.assertEqual(sorted(set(arms[0]['rooms']) | set(arms[1]['rooms'])), spec['normal'])
        tasks = duel_tasks(spec)
        drawn = [tasks.choose(s) for s in range(2000)]
        share = np.mean([t.arm == 0 for t in drawn])
        self.assertTrue(0.45 < share < 0.55)
        self.assertTrue(all(t.variant in arms[t.arm]['rooms'] for t in drawn))
        obs = raw_obs(320, 280, [])
        obs['room'].update(gw=15, gh=9)
        for arm in arms:   # the rocks arm on an empty room: the walkable placement with the same rules
            pairs = [duel_cells(s, obs, {**spec['duel'], **arm}) for s in range(400)]
            for a, b in pairs:
                self.assertGreaterEqual(max(abs(a % 15 - b % 15), abs(a // 15 - b // 15)), 4)
                self.assertTrue(a % 15 != b % 15 and a // 15 != b // 15)       # never on one row or column
            # swapped half of the time: the first cell is drawn uniformly as often in either role
            first = np.mean([a < b for a, b in pairs])
            self.assertTrue(0.4 < first < 0.6)
        lined = [duel_cells(s, obs, {**arms[0], 'same_line': True}) for s in range(400)]
        self.assertTrue(any(a % 15 == b % 15 or a // 15 == b // 15 for a, b in lined))

    def test_scripted_duellist(self):
        from isaac_bridge.abplus_duel import ScriptedDuellist, duel_views
        s = ScriptedDuellist()
        mine, _ = duel_views(duel_obs((200, 280), (400, 285)))
        self.assertEqual(s(mine), (0, 2))                      # aligned on the row, 200 px: stand and shoot right
        s.reset()
        mine, _ = duel_views(duel_obs((200, 280), (500, 282)))
        self.assertEqual(s(mine), (3, 2))                      # aligned but far: walk closer while shooting
        s.reset()
        mine, _ = duel_views(duel_obs((200, 280), (300, 380)))
        self.assertEqual(s(mine), (3, 0))                      # align the nearer axis (x: 100 < y: 100 ties to x)
        mine, _ = duel_views(duel_obs((217, 280), (300, 380)))
        move, shoot = s(mine)                                  # sliding 8.5 px/frame: brakes before the column
        self.assertEqual(shoot, 0)


if __name__ == '__main__':
    unittest.main()
