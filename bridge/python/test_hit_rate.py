"""The hit-rate test (combat-hitrate): running hit rate, time price, per-step reward, observation."""
import unittest

from isaac_bridge.abplus_reward import COMPONENTS_HR, HR, CombatHitRate, describe_hitrate, time_price
from isaac_bridge.hit_rate import WINDOW_FRAMES, HitRate
from test_abplus_geometry import raw_obs


def counters(frame, shots=0, hits=0, dealt=0.0, kills=0):
    """The bridge counters the hit rate and the reward read, at logic frame `frame`."""
    return {'logic_frames': frame, 'events': {'tears': shots},
            'combat': {'tear_hits': hits, 'lineage_damage': dealt, 'lineage_kills': kills, 'lineage_hp': 30.0 - dealt,
                       'lineage_count': 3}}


class HitRateTest(unittest.TestCase):
    def test_no_shot_is_zero_and_the_window_forgets(self):
        h = HitRate()
        self.assertEqual(h.reset(counters(100)), 0.0)
        self.assertEqual(h.update(counters(102)), 0.0)                     # nothing fired yet
        self.assertEqual(h.update(counters(104, shots=4)), 0.0)            # fired, nothing landed yet
        self.assertEqual(h.update(counters(106, shots=4, hits=3)), 0.75)
        # The shots leave the window 90 frames after the step that fired them.
        self.assertEqual(h.update(counters(104 + WINDOW_FRAMES - 2, shots=4, hits=3)), 0.75)
        self.assertEqual(h.update(counters(104 + WINDOW_FRAMES, shots=4, hits=3)), 0.0)   # hits alone: 0 shots -> 0
        self.assertEqual((h.episode_shots, h.episode_hits), (4, 3))

    def test_hits_that_land_after_the_shots_left_are_capped_at_one(self):
        h = HitRate(window_frames=10)
        h.reset(counters(0))
        h.update(counters(2, shots=1))
        h.update(counters(10, shots=2, hits=1))
        self.assertEqual(h.update(counters(14, shots=2, hits=2)), 1.0)     # 2 hits, 1 shot in the window

    def test_rate_of_steady_fire(self):
        # A tear every 10 frames, every tear lands 6 frames later, 1 in 4 misses.
        h = HitRate()
        h.reset(counters(0))
        shots = hits = 0
        landing = {}
        for t in range(2, 600, 2):
            if t % 10 == 0:
                shots += 1
                if shots % 4:
                    landing[t + 6] = landing.get(t + 6, 0) + 1
            hits += landing.pop(t, 0)
            rate = h.update(counters(t, shots, hits))
        self.assertAlmostEqual(rate, 0.75, delta=0.12)


class CombatHitRateTest(unittest.TestCase):
    def test_price_ends(self):
        self.assertEqual(time_price(0.0), 2.0)
        self.assertEqual(time_price(1.0), 0.5)
        self.assertEqual(time_price(0.5), 1.25)
        self.assertEqual(time_price(3.0), 0.5)
        self.assertEqual((HR['cost_miss'], HR['cost_hit'], HR['deadline_s']), (2.0, 0.5, 180.0))

    def test_every_term_is_paid_at_its_step(self):
        r = CombatHitRate()
        r.reset(counters(0), 'normal')
        step = r.step(counters(2), 'running', 2)                           # no shot: the full price
        self.assertEqual(set(step), set(COMPONENTS_HR))
        self.assertAlmostEqual(step['time'], -2.0 * 2 / 30)
        self.assertEqual((step['hit'], step['kill']), (0.0, 0.0))
        r.step(counters(4, shots=2), 'running', 2)
        step = r.step(counters(6, shots=2, hits=2, dealt=7.0), 'running', 2)   # both tears hit
        self.assertAlmostEqual(step['hit'], 7.0 / 5)
        self.assertAlmostEqual(step['time'], -0.5 * 2 / 30)
        step = r.step(counters(8, shots=2, hits=2, dealt=10.0, kills=1), 'win', 2)
        self.assertAlmostEqual(step['hit'] + step['kill'], 3.0 / 5 + 0.25)
        # The clear ends the episode with no settlement: the same price as any other step.
        self.assertAlmostEqual(step['time'], -0.5 * 2 / 30)
        self.assertAlmostEqual(sum(r.totals.values()),
                               10.0 / 5 + 0.25 - (2.0 + 2.0 + 0.5 + 0.5) * 2 / 30)

    def test_hiding_until_the_deadline_costs_the_full_price(self):
        r = CombatHitRate()
        r.reset(counters(0), 'boss')
        for t in range(2, 5402, 2):
            step = r.step(counters(t), 'time_limit' if t == 5400 else 'running', 2)
        self.assertAlmostEqual(r.totals['time'], -2.0 * 180, places=6)
        self.assertAlmostEqual(step['time'], -2.0 * 2 / 30)
        self.assertEqual(CombatHitRate.scale, 0.1)

    def test_options_and_describe(self):
        r = CombatHitRate(hit_hp=10.0, cost_hit=0.25, cost_miss=4.0)
        r.reset(counters(0), 'arena')
        self.assertAlmostEqual(r.step(counters(2), 'running', 2)['time'], -4.0 * 2 / 30)
        d = describe_hitrate(cost_hit=0.25, cost_miss=4.0)
        self.assertIn('0.25', d['time'])
        self.assertIn('truncation', d['deadline'])


class HitRateObservationTest(unittest.TestCase):
    def test_observation_carries_the_rewards_rate(self):
        from isaac_bridge import abplus_worker as W
        from isaac_bridge.transformer_obs import COMBAT_FIELDS_HITRATE, VisibleHistory
        options = W.observation_options('combat-hitrate')
        self.assertEqual(options['combat_state'], COMBAT_FIELDS_HITRATE)
        self.assertTrue(options['geometry'] and options['factored_actions'] and not options['deadline'])
        history = VisibleHistory(64, 256, **options)
        reward = CombatHitRate()
        index = COMBAT_FIELDS_HITRATE.index('hit_rate')
        schedule = [(0, 0), (1, 0), (2, 1), (2, 2), (3, 2), (3, 2), (5, 3), (5, 4)] + [(5, 4)] * 50

        def frame_obs(frame, shots, hits):
            o = raw_obs(320, 360, [(320, 200, True, True)], frame=frame)
            o['events'] = {'tears': shots}
            o['combat'].update(tear_hits=hits, lineage_damage=3.5 * hits, lineage_kills=0, lineage_hp=10.0,
                               lineage_count=1.0)
            return o

        first = frame_obs(10, 0, 0)
        reward.reset(first, 'normal')
        self.assertEqual(float(history.encode(first)['combat'][index]), 0.0)
        seen = []
        for k, (shots, hits) in enumerate(schedule, start=1):
            o = frame_obs(10 + 2 * k, shots, hits)
            observed = float(history.encode(o)['combat'][index])
            reward.step(o, 'running', 2)
            self.assertAlmostEqual(observed, reward.rate, places=6)
            seen.append(observed)
        self.assertGreater(max(seen), 0.5)
        self.assertEqual(seen[-1], 0.0)            # no tear in the last 3 s

    def test_frame_layout_and_episode_stats(self):
        from isaac_bridge import abplus_worker as W
        dtype, keys = W.frame_layout('combat-hitrate')
        self.assertEqual(dtype['combat'].shape, (3,))
        self.assertEqual(dtype['previous_action'].shape, (18,))
        self.assertIn('fire_distance', keys)
        self.assertIn('combat-hitrate', W.TRUNCATING_PROFILES)
        self.assertEqual(W.EPISODE_STATS[-3:], ('shots', 'tear_hits', 'misses'))
        o = raw_obs(320, 360, [(320, 200, True, True)])
        o['events'], o['combat']['tear_hits'] = {'tears': 7}, 2
        stats = W.EpisodeStats(o)
        o2 = raw_obs(320, 360, [(320, 200, True, True)], frame=12)
        o2['events'], o2['combat']['tear_hits'] = {'tears': 12}, 5
        stats.step(o2, 2)
        self.assertEqual(stats.array()[-3:-1].tolist(), [5.0, 3.0])


class CombatHitRateWalkTest(unittest.TestCase):
    def room(self, x, y, frame, wall=True):
        from test_abplus_geometry import walk_room
        o = walk_room(x, y, (240, 280), rocks=[(c, 5) for c in range(6, 14)] if wall else ())
        o.update(counters(frame))
        return o

    def test_potential_telescopes_and_a_clear_ends_it(self):
        from isaac_bridge.abplus_reward import COMPONENTS_HRW, HRW, CombatHitRateWalk
        from isaac_bridge.abplus_geometry import CELL, TEAR_RADIUS
        r = CombatHitRateWalk()
        r.reset(self.room(440, 360, 0), 'normal')
        self.assertAlmostEqual(r.phi, -HRW['align'] * (200 - 13 - TEAR_RADIUS) / CELL)
        path = [(440 - 8 * k, 360) for k in range(1, 26)] + [(240, 360)] * 5     # walk left along row 6
        g, discounted = 1.0, 0.0
        for k, (x, y) in enumerate(path, start=1):
            step = r.step(self.room(x, y, 2 * k), 'running', 2)
            self.assertEqual(set(step), set(COMPONENTS_HRW))
            discounted += g * step['align']
            g *= HRW['gamma']
        self.assertAlmostEqual(discounted, g * r.phi - (-HRW['align'] * (200 - 13 - TEAR_RADIUS) / CELL), places=9)
        self.assertEqual(r.phi, 0.0)                                  # in the target's column band
        # Walking the wrong way (up into the wall) earns nothing: the walking distance does not shrink.
        r2 = CombatHitRateWalk()
        r2.reset(self.room(440, 360, 0), 'normal')
        self.assertLessEqual(r2.step(self.room(440, 350, 2), 'running', 2)['align'], 1e-9)
        # A clear sets Phi(s') = 0: the remaining potential is paid out.
        r3 = CombatHitRateWalk()
        r3.reset(self.room(440, 360, 0), 'normal')
        self.assertAlmostEqual(r3.step(self.room(440, 360, 2), 'win', 2)['align'], -r3.potential(200 - 13 - TEAR_RADIUS))

    def test_profile_observes_the_walking_distance(self):
        from isaac_bridge import abplus_worker as W
        from isaac_bridge.transformer_obs import VisibleHistory
        from isaac_bridge.abplus_reward import describe_hitrate_walk
        options = W.observation_options('combat-hitrate-walk')
        self.assertEqual(options['geometry'], 'walk')
        self.assertEqual(W.frame_layout('combat-hitrate-walk')[0], W.frame_layout('combat-hitrate')[0])
        self.assertIn('combat-hitrate-walk', W.TRUNCATING_PROFILES)
        o = self.room(440, 360, 10)
        o.update(players=[{**o['players'][0], 'size': 10.0, 'hearts': 6, 'max_hearts': 6, 'soul': 0, 'bombs': 0,
                           'keys': 0, 'coins': 0, 'damage': 3.5, 'speed': 1.0, 'shot_speed': 1.0, 'fire_delay_max': 10.0,
                           'can_fly': False, 'active_charge': 0, 'active_ready': False, 'active': 0, 'anim': '',
                           'aframe': 0, 'flip': False}],
                 combat_schema=3, doors=[], room={'gw': 15, 'top_left': [60.0, 140.0], 'bottom_right': [580.0, 420.0]})
        o['entities'] = [dict(e, id=1, type=12, variant=0, subtype=0, size_multi=[1.0, 1.0], coll=4, gcoll=5, cdmg=1.0,
                              aframe=0, flip=False, age=10) for e in o['entities']]
        o['combat'].update(blocking_hp=10.0, blocking_count=1.0)
        o['terrain'].update(height=9)
        frame = VisibleHistory(64, 256, **options).encode(o)
        from isaac_bridge.abplus_geometry import TEAR_RADIUS
        self.assertAlmostEqual(float(frame['fire_distance']), (200 - 13 - TEAR_RADIUS) / 40, places=5)
        self.assertIn('walking distance', describe_hitrate_walk()['align'])


class CombatHitRateHurtTest(unittest.TestCase):
    """Stage 2 (C33): combat-hitrate-miss without invincibility, combat-v2's health curve, and a death that
    costs the rest of the deadline and keeps the walking potential."""

    def room(self, frame, hearts=6, damage=0.0, px=240, py=360):
        from test_abplus_geometry import walk_room
        o = walk_room(px, py, (240, 280))   # (240, 360) is in the target's column band: walking distance 0
        o.update(counters(frame))
        o['combat'].update(miss_units=0, miss_streak=0, tear_misses=0, player_damage=damage)
        o['players'][0]['hearts'] = hearts
        return o

    def test_hurt_follows_the_health_curve(self):
        from isaac_bridge.abplus_reward import COMPONENTS_HRH, CombatHitRateHurt, health_value
        r = CombatHitRateHurt(deadline_s=180)
        r.reset(self.room(0), 'normal')
        got = r.step(self.room(2, hearts=5, damage=1.0), 'running', 2)
        self.assertEqual(set(got), set(COMPONENTS_HRH))
        self.assertAlmostEqual(got['hurt'], -1.0, places=6)          # the first of six half hearts costs 1
        got = r.step(self.room(4, hearts=3, damage=3.0), 'running', 2)
        self.assertAlmostEqual(got['hurt'], -(health_value(5) - health_value(3)), places=6)
        self.assertEqual(r.step(self.room(6, hearts=3, damage=3.0), 'running', 2)['hurt'], 0.0)
        self.assertAlmostEqual(health_value(1) - health_value(0), 4.09, places=2)   # the last is the dearest
        self.assertAlmostEqual(r.totals['hurt'], -(health_value(6) - health_value(3)), places=6)

    def test_death_costs_the_rest_of_the_deadline_and_keeps_the_potential(self):
        from isaac_bridge.abplus_reward import HRH, CombatHitRateHurt, health_value
        start = self.room(0, px=440, py=360)                        # 80 - tau below the target's row band
        r = CombatHitRateHurt(deadline_s=180)
        r.reset(start, 'normal')
        phi = r.phi
        self.assertLess(phi, 0.0)
        got = r.step(self.room(1800, hearts=0, damage=6.0, px=440, py=360), 'death', 1800)   # dies at 60 s
        self.assertAlmostEqual(got['death'], -2.0 * (180 - 60) - HRH['death'], places=6)
        self.assertAlmostEqual(got['align'], r.gamma * phi - phi, places=6)                # not paid out
        self.assertAlmostEqual(got['hurt'], -(health_value(6) - health_value(0)), places=5)
        # A clear still pays the potential out, and costs no death.
        w = CombatHitRateHurt(deadline_s=180)
        w.reset(start, 'normal')
        got = w.step(self.room(2, px=440, py=360), 'win', 2)
        self.assertAlmostEqual(got['align'], -phi, places=6)
        self.assertEqual(got['death'], 0.0)

    def test_profile_is_registered_like_the_miss_profile(self):
        from isaac_bridge import abplus_worker as W
        from isaac_bridge.abplus_reward import REWARDS, CombatHitRateHurt, describe_hitrate_hurt
        self.assertIs(REWARDS['combat-hitrate-hurt'], CombatHitRateHurt)
        for group in (W.REWARD_PROFILES, W.TRUNCATING_PROFILES, W.GEOMETRY_PROFILES, W.WALK_PROFILES):
            self.assertIn('combat-hitrate-hurt', group)
        self.assertEqual(W.observation_options('combat-hitrate-hurt'), W.observation_options('combat-hitrate-miss'))
        self.assertEqual(W.frame_layout('combat-hitrate-hurt')[0], W.frame_layout('combat-hitrate-miss')[0])
        self.assertIn('180 s deadline', describe_hitrate_hurt(deadline_s=180)['death'])

    def test_hurt_outcome_is_appended(self):
        """C37: frames store the outcome index, so 'hurt' only extends the table."""
        from isaac_bridge import abplus_worker as W
        self.assertEqual(W.OUTCOMES, ('running', 'death', 'win', 'time_limit', 'error', 'hurt'))

    def test_death_cost_can_be_dropped(self):
        """C36: death=False leaves only the health curve at the lethal hit (no rest-of-deadline, no -5)."""
        from isaac_bridge.abplus_reward import CombatHitRateHurt, describe_hitrate_hurt, health_value
        r = CombatHitRateHurt(deadline_s=30, death=False, align=0.0)
        r.reset(self.room(0, px=440, py=360), 'normal')
        self.assertEqual(r.phi, 0.0)                                  # align 0: no potential
        got = r.step(self.room(150, hearts=0, damage=6.0, px=440, py=360), 'death', 150)   # dies at 5 s
        self.assertEqual(got['death'], 0.0)
        self.assertEqual(got['align'], 0.0)
        self.assertAlmostEqual(got['hurt'], -(health_value(6) - health_value(0)), places=5)
        self.assertIn('none', describe_hitrate_hurt(deadline_s=30, death=False)['death'])


class CombatHitRateMissTest(unittest.TestCase):
    def room(self, frame, units=0, streak=0, misses=0, shots=0, hits=0, dealt=0.0):
        from test_abplus_geometry import walk_room
        o = walk_room(240, 360, (240, 280))                   # in the target's column band: no potential change
        o.update(counters(frame, shots, hits, dealt))
        o['combat'].update(miss_units=units, miss_streak=streak, tear_misses=misses)
        return o

    def test_miss_penalty_grows_along_a_run_and_restarts_after_a_hit(self):
        from isaac_bridge.abplus_reward import COMPONENTS_HRM, HRM, CombatHitRateMiss
        r = CombatHitRateMiss()
        r.reset(self.room(0), 'normal')
        # Three misses in a row (bridge miss_units 1, 3, 6), a hit, then another miss (units 7).
        steps = [dict(units=1, streak=1, misses=1, shots=1), dict(units=3, streak=2, misses=2, shots=2),
                 dict(units=6, streak=3, misses=3, shots=3), dict(units=6, streak=0, misses=3, shots=4, hits=1, dealt=3.5),
                 dict(units=7, streak=1, misses=4, shots=5, hits=1, dealt=3.5)]
        got = [r.step(self.room(2 * k, **s), 'running', 2) for k, s in enumerate(steps, start=1)]
        self.assertEqual(set(got[0]), set(COMPONENTS_HRM))
        self.assertEqual([round(g['miss'], 6) for g in got], [-0.01, -0.02, -0.03, 0.0, -0.01])
        self.assertAlmostEqual(got[3]['hit'], 0.7)
        self.assertAlmostEqual(r.totals['miss'], -0.07)
        self.assertEqual((HRM['miss'], HRM['lineage_mode']), (0.01, 3))
        self.assertEqual(CombatHitRateMiss(miss=0.05).miss, 0.05)

    def test_hurt_rest_charges_the_rest_of_the_deadline(self):
        """After C37: with hurt_rest the hurt that ends the episode costs cost_miss per second left."""
        from isaac_bridge.abplus_reward import (COMPONENTS_HRM, COMPONENTS_HRM_REST, CombatHitRateMiss,
                                                describe_hitrate_miss)
        plain = CombatHitRateMiss()
        plain.reset(self.room(0), 'normal')
        self.assertNotIn('rest', plain.step(self.room(2), 'hurt', 2))   # off (C37): nothing more is charged
        self.assertEqual((plain.components, len(plain.totals_array())), (COMPONENTS_HRM, len(COMPONENTS_HRM)))
        r = CombatHitRateMiss(deadline_s=30, hurt_rest=True)
        r.reset(self.room(0), 'normal')
        got = [r.step(self.room(2 * k), 'running', 2) for k in range(1, 75)]
        self.assertEqual({g['rest'] for g in got}, {0.0})
        last = r.step(self.room(150), 'hurt', 2)                          # the first damage at 5 s
        self.assertAlmostEqual(last['rest'], -2.0 * (30 - 5))
        self.assertAlmostEqual(r.totals['rest'], -50.0)
        self.assertEqual(r.components, COMPONENTS_HRM_REST)
        self.assertEqual(len(r.totals_array()), len(COMPONENTS_HRM_REST))
        from isaac_bridge.abplus_worker import META_DTYPE
        self.assertGreaterEqual(META_DTYPE['components'].shape[0], len(COMPONENTS_HRM_REST))
        self.assertIn('30 s deadline', describe_hitrate_miss(deadline_s=30, hurt_rest=True)['rest'])
        self.assertNotIn('rest', describe_hitrate_miss())

    def test_profile_observes_the_miss_streak(self):
        from isaac_bridge import abplus_worker as W
        from isaac_bridge.transformer_obs import COMBAT_FIELDS_MISS, VisibleHistory
        from isaac_bridge.abplus_reward import describe_hitrate_miss
        options = W.observation_options('combat-hitrate-miss')
        self.assertEqual((options['geometry'], options['combat_state']), ('walk', COMBAT_FIELDS_MISS))
        self.assertEqual(W.frame_layout('combat-hitrate-miss')[0]['combat'].shape, (4,))
        h = VisibleHistory(64, 256, **options)
        values = h.combat_features({'blocking_hp': 10.0, 'blocking_count': 1.0, 'miss_streak': 30}, 0.0,
                                   counters(10))
        self.assertAlmostEqual(values[COMBAT_FIELDS_MISS.index('miss_streak')], 1.5)
        self.assertIn('k-th miss', describe_hitrate_miss()['miss'])

    def test_miss_cap_reaches_the_bridge_setup(self):
        # The cap is applied in the bridge (abp-0.2.6): both setup chunks pass it to AbpSetMissCap and
        # keep "reseeded=" last (the reset checks the suffix).
        from isaac_bridge.abplus import ARENA_LUA, ROOM_LUA
        from isaac_bridge.abplus_reward import HRM, describe_hitrate_miss
        common = dict(player_hp=6, boss_hp_fraction='1.0', rng_seed=1, bombs=0, lineage_mode=3, invincible='true',
                      miss_cap=20, stats='0, 0, 0, 0, 0')   # abp-0.2.11: ROOM_LUA's stat offsets (C41)
        for chunk in (ROOM_LUA.format(**common), ARENA_LUA.format(px=1, py=2, bx=3, by=4, **common)):
            self.assertIn('AbpSetMissCap(20)', chunk)
            self.assertTrue(chunk.rstrip().endswith('" reseeded=" .. tostring(reseeded)'))
        self.assertEqual(HRM['miss_cap'], 20)
        self.assertIn('min(k, 20)', describe_hitrate_miss()['miss'])
        self.assertNotIn('min(', describe_hitrate_miss(miss_cap=0)['miss'])


class CombatHitRateFireTest(unittest.TestCase):
    """combat-hitrate-fire: each tear's hit, kill and miss terms carry the step it was fired in."""

    def room(self, frame, credits=(), units=0, streak=0, misses=0, shots=0, hits=0, dealt=0.0, kills=0):
        o = CombatHitRateMissTest.room(self, frame, units, streak, misses, shots, hits, dealt)
        o['combat'].update(lineage_kills=kills, credits=[list(c) for c in credits])
        return o

    def test_credit_goes_back_to_the_transition_that_fired(self):
        from isaac_bridge.abplus_reward import CREDIT_STEPS, CombatHitRateFire
        r = CombatHitRateFire()
        r.reset(self.room(100), 'normal')                    # transition s covers frames (100 + 2s, 102 + 2s]
        for k in range(1, 5):                                # transitions 0..3: the tear fired in frame 101
            r.step(self.room(100 + 2 * k, shots=1), 'running', 2)
            self.assertFalse(r.credit.any())
        # Transition 4 (frames 109, 110) observes the hit (3.5 HP) and the kill of the tear fired in
        # frame 101 (transition 0, 4 steps back), a miss of a tear fired in frame 104 (transition 1,
        # 3 units) and damage charged to the current frame 110 (no tear: stays).
        parts = r.step(self.room(110, credits=[(101, 3.5, 1, 0), (104, 0, 0, 3), (110, 1.5, 0, 0)],
                                 units=3, streak=0, misses=1, shots=2, hits=1, dealt=5.0, kills=1), 'running', 2)
        self.assertEqual(r.credit.shape, (CREDIT_STEPS,))
        self.assertAlmostEqual(float(r.credit[3]), 3.5 / 5 + 0.25, places=6)   # j = 4
        self.assertAlmostEqual(float(r.credit[2]), -0.03, places=6)            # j = 3
        self.assertEqual(int((r.credit != 0).sum()), 2)
        # The step reward itself is unchanged: the learner moves the credit out of it.
        self.assertAlmostEqual(parts['hit'], 1.0)
        self.assertAlmostEqual(parts['kill'], 0.25)
        self.assertAlmostEqual(parts['miss'], -0.03)
        self.assertAlmostEqual(r.credited, 3.5 / 5 + 0.25 - 0.03, places=6)
        # A credit older than CREDIT_STEPS or before the episode is not moved.
        r2 = CombatHitRateFire()
        r2.reset(self.room(100), 'normal')
        for k in range(1, CREDIT_STEPS + 3):
            r2.step(self.room(100 + 2 * k, credits=[(99, 3.5, 0, 0)] if k == 1 else
                              [(101, 3.5, 0, 0)] if k == CREDIT_STEPS + 2 else []), 'running', 2)
            self.assertFalse(r2.credit.any())

    def test_frame_layout_and_rows(self):
        import numpy as np
        from isaac_bridge import abplus_worker as W
        from isaac_bridge.abplus_reward import CREDIT_STEPS
        dtype, _ = W.frame_layout('combat-hitrate-fire')
        self.assertEqual(dtype['credit'].shape, (CREDIT_STEPS,))
        self.assertNotIn('credit', W.frame_layout('combat-hitrate-miss')[0].names)
        self.assertEqual(W.observation_options('combat-hitrate-fire'), W.observation_options('combat-hitrate-miss'))
        row = np.zeros(1, dtype)[0]
        frame = np.zeros(1, dtype)[0]
        credit = np.arange(CREDIT_STEPS, dtype=np.float32)
        W.write_frame(row, frame, 1.0, False, False, 0, 2, 10, 5, 0.0, credit)
        self.assertEqual(float(row['credit'][5]), 5.0)
        W.write_frame(row, frame, 0.0, False, False, 0, 0, 10, 5)                # a reset row clears it
        self.assertFalse(row['credit'].any())

    def test_buffer_moves_the_credit_before_gae(self):
        import types

        import torch
        from isaac_bridge.gpu_buffer import GpuHistoryRolloutBuffer as B
        b = types.SimpleNamespace(device=torch.device('cpu'), rewards=torch.zeros(6, 3), buffer_size=6, n_envs=3)
        workers = torch.tensor([0, 2])
        credit = torch.zeros(2, 4)
        credit[0, 1] = 0.5            # env 0: 2 steps back
        credit[1, 3] = -0.2           # env 2: 4 steps back
        for t in range(6):            # every step observes its full reward; credits only at steps 1 and 5
            b.rewards[t, workers] = torch.tensor([1.0, 0.0])
            B.add_credit(b, t, workers, credit if t in (1, 5) else torch.zeros(2, 4))
        total = float(b.rewards.sum())
        moved, kept = B.apply_credits(b)
        self.assertAlmostEqual(float(b.rewards.sum()), total, places=6)     # nothing lost
        # Step 5: moved to steps 3 (env 0) and 1 (env 2).
        self.assertAlmostEqual(float(b.rewards[5, 0]), 0.5)
        self.assertAlmostEqual(float(b.rewards[3, 0]), 1.5)
        self.assertAlmostEqual(float(b.rewards[5, 2]), 0.2)
        # Step 1: no step 2 or 4 back in this rollout, so its credits stay at step 1; env 2's step 1
        # also receives step 5's -0.2.
        self.assertAlmostEqual(float(b.rewards[1, 0]), 1.0)
        self.assertAlmostEqual(float(b.rewards[1, 2]), -0.2)
        self.assertAlmostEqual(float(moved), 0.3, places=6)
        self.assertAlmostEqual(float(kept), 0.3, places=6)
        self.assertFalse(b.credits.any())

class TargetArenaTest(unittest.TestCase):
    """The single-enemy aiming arena (C22): cells from the seed, the setup chunk, the aux loss split."""

    def test_cells_are_interior_apart_and_replay(self):
        from isaac_bridge.abplus import target_cells
        seen = set()
        for seed in range(500):
            player, target = target_cells(seed, 15, 9, 3)
            self.assertEqual(target_cells(seed, 15, 9, 3), (player, target))
            pc, pr = player % 15, player // 15
            tc, tr = target % 15, target // 15
            for c, r in ((pc, pr), (tc, tr)):
                self.assertTrue(1 <= c <= 13 and 1 <= r <= 7)
            self.assertGreaterEqual(max(abs(pc - tc), abs(pr - tr)), 3)
            seen.add((player, target))
        self.assertGreater(len(seen), 400)

    def test_target_kind_from_the_seed(self):
        from collections import Counter
        from isaac_bridge.abplus import target_kind
        self.assertEqual(target_kind(5, {'type': 12, 'variant': 0}), (12, 0))
        kinds = [{'type': 12}, {'type': 15, 'variant': 0}, {'type': 16}, {'type': 11}]
        picks = [target_kind(seed, {'types': kinds}) for seed in range(4000)]
        self.assertEqual(picks[:50], [target_kind(seed, {'types': kinds}) for seed in range(50)])
        counts = Counter(picks)
        self.assertEqual(set(counts), {(12, 0), (15, 0), (16, 0), (11, 0)})
        self.assertTrue(all(900 < n < 1100 for n in counts.values()), counts)

    def test_walkable_cells_avoid_rocks_and_need_a_way_to_fire(self):
        from isaac_bridge.abplus import target_cells_walkable
        from test_abplus_geometry import walk_room
        # A rock ring around (3, 3) seals the cell: a target there cannot be reached, a player there cannot
        # leave; a full wall on column 9 splits the room, and tears do not pass it.
        ring = [(c, r) for c in (2, 3, 4) for r in (2, 3, 4) if (c, r) != (3, 3)]
        wall = [(9, r) for r in range(1, 8)]
        o = walk_room(200, 200, (400, 300), rocks=ring + wall)
        o['room']['gh'] = 9
        blocked = set(ring + wall)
        seen = set()
        for seed in range(300):
            cells = target_cells_walkable(seed, o, 3)
            self.assertEqual(cells, target_cells_walkable(seed, o, 3))
            (pc, pr), (tc, tr) = [(i % 15, i // 15) for i in cells]
            for c, r in ((pc, pr), (tc, tr)):
                self.assertTrue(1 <= c <= 13 and 1 <= r <= 7 and (c, r) not in blocked)
            self.assertGreaterEqual(max(abs(pc - tc), abs(pr - tr)), 3)
            self.assertNotIn((3, 3), ((pc, pr), (tc, tr)))
            self.assertEqual(pc < 9, tc < 9)          # never across the wall
            seen.add(cells)
        self.assertGreater(len(seen), 200)

    def test_player_is_never_placed_on_spikes(self):
        """C36: spikes are walkable, but a mortal player placed on them is hurt before the episode starts."""
        from isaac_bridge.abplus import target_cells_walkable
        from test_abplus_geometry import walk_room
        plain = walk_room(200, 200, (400, 300))
        spiky = walk_room(200, 200, (400, 300))
        spikes = {(c, r) for c in range(1, 14) for r in (3, 4)}           # two walkable rows of spikes
        spiky['grid'] = spiky['grid'] + [[r * 15 + c, 8, 0, 0, 0, 40.0 + 40 * c, 120.0 + 40 * r] for c, r in spikes]
        same = 0
        for seed in range(300):
            cells = target_cells_walkable(seed, spiky, 3)
            self.assertNotIn((cells[0] % 15, cells[0] // 15), spikes)
            same += cells == target_cells_walkable(seed, plain, 3)
        self.assertGreater(same, 150)        # only the seeds whose player draw hit a spike cell change

    def test_detour_starts_need_a_longer_walk(self):
        from isaac_bridge import abplus_geometry as G
        from isaac_bridge.abplus import target_cells_walkable
        from test_abplus_geometry import walk_room
        # A rock bar on column 7 (rows 1-6) leaves a gap at row 7: from one side to the other the walk
        # goes round its bottom end.
        bar = [(7, r) for r in range(1, 7)]
        o = walk_room(200, 200, (400, 300), rocks=bar)
        o['room']['gh'] = 9
        found = 0
        for seed in range(200):
            cells = target_cells_walkable(seed, o, 3, attempts=400, detour_min=60.0)
            if cells is None:
                continue
            found += 1
            (pc, pr), (tc, tr) = [(i % 15, i // 15) for i in cells]
            grid = G.walk_grid(o)
            (x0, y0), _, _ = grid
            target = (x0 + 40 * tc, y0 + 40 * tr, 13.0)
            stops, origin = G.tear_stops(o)
            limits = [G.target_limits(target, 260.0, stops, origin)]
            field = G.walk_field(grid, [target], limits)
            px, py = x0 + 40 * pc, y0 + 40 * pr
            walk = G.walk_distance(px, py, grid, field, [target], limits)
            self.assertGreaterEqual(walk - G.fire_distance(px, py, [target], limits), 60.0)
        self.assertGreater(found, 150)
        # Without the option the placements are exactly the tier-4 ones.
        self.assertEqual(target_cells_walkable(3, o, 3), target_cells_walkable(3, o, 3, 64, 0.0))

    def test_trap_starts_run_into_a_dead_end(self):
        from isaac_bridge import abplus_geometry as G
        from isaac_bridge.abplus import target_cells_walkable
        from test_abplus_geometry import walk_room
        # The bar room of the detour test: heading straight for a target across the bar ends against it.
        bar = [(7, r) for r in range(1, 7)]
        o = walk_room(200, 200, (400, 300), rocks=bar)
        o['room']['gh'] = 9
        grid = G.walk_grid(o)
        (x0, y0), _, _ = grid
        stops, origin = G.tear_stops(o)
        found = 0
        for seed in range(200):
            cells = target_cells_walkable(seed, o, 3, attempts=600, detour_min=60.0, trap_min=30.0)
            if cells is None:
                continue
            found += 1
            (pc, pr), (tc, tr) = [(i % 15, i // 15) for i in cells]
            target = (x0 + 40 * tc, y0 + 40 * tr, 13.0)
            field = G.walk_field(grid, [target], [G.target_limits(target, 260.0, stops, origin)])
            self.assertGreaterEqual(G.trap_depth(grid, field, (pc, pr), target), 30.0)
        self.assertGreater(found, 150)
        self.assertEqual(target_cells_walkable(3, o, 3), target_cells_walkable(3, o, 3, 64, 0.0, 0.0))

    def test_arms_choose_their_own_rooms(self):
        import json
        import os
        import tempfile
        from collections import Counter

        import numpy as np
        from isaac_bridge.abplus_tasks import Task, TaskSampler, target_arm
        arms = [{'name': 'a', 'weight': 2.0, 'rooms': [1, 2, 3]}, {'name': 'b', 'weight': 1.0, 'rooms': [10, 11]}]
        rooms = [1, 2, 3, 10, 11]
        sampler = TaskSampler({'normal': 1.0}, rooms, [], arms)
        counts = Counter()
        for seed in range(3000):
            task = sampler.choose(seed)
            self.assertEqual(task.arm, target_arm(seed, arms))
            self.assertIn(task.variant, arms[task.arm]['rooms'])
            self.assertEqual(sampler.choose(seed, 2).arm, task.arm)     # a room retry keeps the arm
            counts[task.arm] += 1
        self.assertTrue(0.62 < counts[0] / 3000 < 0.71, counts)
        # Without arms the draw is the old one: the same stream, arm -1.
        plain = TaskSampler({'normal': 1.0}, rooms, [])
        for seed in range(50):
            rng = np.random.default_rng([seed, 0x7A5C])
            rng.random()
            entrance = int(rng.integers(4))
            self.assertEqual(plain.choose(seed), Task('normal', rooms[int(rng.integers(len(rooms)))], entrance))
        # from_file takes the arms from the target spec.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'mix.json')
            with open(path, 'w') as fh:
                json.dump({'weights': {'normal': 1.0}, 'normal': rooms, 'boss': [], 'target': {'arms': arms}}, fh)
            self.assertEqual([TaskSampler.from_file(path).choose(s) for s in range(20)], [sampler.choose(s) for s in range(20)])
        with self.assertRaises(ValueError):
            TaskSampler({'normal': 1.0}, rooms, [], [{'weight': 1.0, 'rooms': []}])

    def test_setup_chunk(self):
        from isaac_bridge.abplus import TARGET_LUA, lua_extras
        chunk = TARGET_LUA.format(player_hp=6, rng_seed=7, bombs=0, player_cell=31, target_cell=100, target_type=12,
                                  target_variant=0, extras=lua_extras([(45, 18, 0), (67, 10, 1)]), lineage_mode=3,
                                  invincible='true', miss_cap=20, stats='0, 0, 0, 0, 0')
        self.assertIn('Isaac.Spawn(12, 0, 0, room:GetGridPosition(100)', chunk)
        self.assertIn('room:GetGridPosition(31)', chunk)
        self.assertIn('local extras, extra_champions = {{18, 0, 45}, {10, 1, 67}}, 0', chunk)
        self.assertIn('AbpSetMissCap(20)', chunk)
        self.assertTrue(chunk.rstrip().endswith('" reseeded=" .. tostring(reseeded)'))
        self.assertEqual(lua_extras([]), '{}')

    def test_extra_npcs_are_reachable_apart_and_replay(self):
        from collections import Counter
        from isaac_bridge.abplus import target_cells_walkable, target_extras
        from test_abplus_geometry import walk_room
        # The walled room of the tier-4 test: a sealed cell (3, 3) and a full wall on column 9.
        ring = [(c, r) for c in (2, 3, 4) for r in (2, 3, 4) if (c, r) != (3, 3)]
        wall = [(9, r) for r in range(1, 8)]
        o = walk_room(200, 200, (400, 300), rocks=ring + wall)
        o['room']['gh'] = 9
        spec = {'counts': [1, 2], 'types': [{'type': 12}, {'type': 18}, {'type': 10, 'variant': 1}], 'min_cells': 3}
        counts, kinds = Counter(), Counter()
        for seed in range(300):
            player, target = target_cells_walkable(seed, o, 3)
            extras = target_extras(seed, o, spec, player, target)
            self.assertEqual(extras, target_extras(seed, o, spec, player, target))
            if extras is None:        # the small side of the wall can run out of cells: the room is skipped
                counts[None] += 1
                continue
            counts[len(extras)] += 1
            (pc, pr), (tc, tr) = (player % 15, player // 15), (target % 15, target // 15)
            cells = [(tc, tr)]
            for cell, t, v in extras:
                c, r = cell % 15, cell // 15
                kinds[(t, v)] += 1
                self.assertTrue(1 <= c <= 13 and 1 <= r <= 7 and (c, r) not in set(ring + wall) and (c, r) != (3, 3))
                self.assertGreaterEqual(max(abs(c - pc), abs(r - pr)), 3)
                self.assertTrue(all(max(abs(c - oc), abs(r - orow)) >= 2 for oc, orow in cells))
                self.assertEqual(c < 9, pc < 9)            # reachable: never across the wall
                cells.append((c, r))
        self.assertLessEqual(counts[None], 15, counts)
        self.assertTrue(counts[1] > 100 and counts[2] > 100, counts)
        self.assertEqual(set(kinds), {(12, 0), (18, 0), (10, 1)})
        self.assertIsNone(target_extras(0, o, dict(spec, min_cells=20), *target_cells_walkable(0, o, 3)))

    def test_blocked_moves_reach_the_masks(self):
        import numpy as np
        import torch
        from isaac_bridge.abplus_geometry import blocked_moves
        from isaac_bridge.abplus_worker import frame_layout, write_frame
        from isaac_bridge.gpu_ppo import action_masks
        from test_abplus_geometry import walk_room
        # The frame carries the block of its own observation, and the learner's masks drop those moves.
        dtype, _ = frame_layout('combat-hitrate-miss')
        self.assertIn('move_block', dtype.names)
        row = np.zeros((), dtype)
        o = walk_room(569.5, 149.5, (160, 280))                  # top right inner corner
        frame = {k: np.zeros(dtype[k].shape, dtype[k].base) for k in dtype.names}
        write_frame(row, frame, 0.0, False, False, 0, 0, 1, 1, raw=o)
        self.assertEqual(row['move_block'].tolist(), [float(b) for b in blocked_moves(o)])
        write_frame(row, frame, 0.0, False, False, 0, 0, 1, 1)   # an error row: nothing blocked
        self.assertFalse(row['move_block'].any())
        block = torch.tensor([[bool(b) for b in blocked_moves(o)], [False] * 9])
        masks = action_masks([9, 5, 2, 2], torch.tensor([True, False]), block)
        self.assertEqual(masks[0, :9].tolist(), [not b for b in blocked_moves(o)])
        self.assertTrue(masks[1, :9].all() and masks[:, 9:14].all())
        self.assertTrue(torch.equal(action_masks([9, 5, 2, 2], torch.tensor([True, False])),
                                    action_masks([9, 5, 2, 2], torch.tensor([True, False]), torch.zeros(2, 9, dtype=torch.bool))))

    def test_full_kl_of_masked_factored_heads(self):
        import torch
        from isaac_bridge.gpu_ppo import full_kl
        # Two heads (3 and 2 actions), the last action of the first head masked (-1e8) in both policies.
        p = torch.tensor([[0.2, 0.5, -1e8, 0.3, -0.1]])
        q = torch.tensor([[0.9, -0.4, -1e8, 0.3, -0.1]])
        kl = full_kl(p, q, [3, 2])
        pp, qq = torch.log_softmax(p[:, :2], -1), torch.log_softmax(q[:, :2], -1)
        self.assertAlmostEqual(float(kl), float((pp.exp() * (pp - qq)).sum()), places=6)
        self.assertEqual(float(full_kl(p, p, [3, 2])), 0.0)
        # An exchange of two near-deterministic choices: the exact KL is large, but finite and per sample.
        a = torch.tensor([[20.0, 0.0, 0.0, 0.0, 0.0]]); b = torch.tensor([[0.0, 20.0, 0.0, 0.0, 0.0]])
        self.assertAlmostEqual(float(full_kl(a, b, [3, 2])), 20.0, delta=0.01)

    def test_aux_distance_weight_defaults_to_the_old_loss(self):
        import inspect
        from isaac_bridge import gpu_ppo
        source = inspect.getsource(gpu_ppo)
        self.assertIn('self.aux_distance_coef=1.0', source)
        self.assertIn('self.aux_balance=False', source)
        self.assertIn('aux=self.aux_coef*(weighted.sum()+self.aux_distance_coef*se.sum())', source)

    def test_balanced_aim_weights_average_one(self):
        import torch
        aim = torch.tensor([0., 0., 0., 0., 0., 0., 0., 2., 0., 3.])        # 2 of 10 aligned
        aligned = (aim > 0).float()
        share = aligned.mean().clamp(0.02, 0.98)
        w = aligned / (2 * share) + (1 - aligned) / (2 * (1 - share))
        self.assertAlmostEqual(float(w.mean()), 1.0, places=6)
        self.assertAlmostEqual(float(w[aligned > 0].sum()), float(w[aligned == 0].sum()), places=5)



class BridgeVersionTest(unittest.TestCase):
    def test_client_and_bridge_versions_match(self):
        """The client refuses any other bridge version, so both sides are bumped together (abp-0.2.8, C37)."""
        import re
        from pathlib import Path
        from isaac_bridge.abplus import BRIDGE_VERSION
        lua = (Path(__file__).resolve().parent.parent / 'abplus' / 'abp_bridge.lua').read_text(encoding='utf8')
        self.assertEqual(re.search(r'^local VERSION = "([^"]+)"', lua, re.M).group(1), BRIDGE_VERSION)


if __name__ == '__main__':
    unittest.main()
