"""combat-v2: the AB+ room-mixture reward (user decisions 2026-09-24), in half-heart units.

The weights are calibrated on the game's own daily-run score (AB+ v1.06 ScoreSheet::Calculate,
checked against the decompile; rl/bridge/abplus/README.md "奖励 combat-v2"). One unit is the first
half heart lost, about 200 score points in a completed run.

  hurt      Health value V(h) = W * f(h / 6), f(x) = (x + 1 - (1 - x)^4) / 2 (OpenAI Five's health
            curve), W = 11.94 so that the first of 6 half hearts costs exactly 1. Losing the half
            hearts of a full 3-heart player costs 1.00, 1.06, 1.29, 1.80, 2.70 and 4.09 (the last);
            above 6 half hearts each one costs 1. Only damage the game itself counts is charged: the
            bridge's player_damage follows Entity_Player::GetTotalDamageTaken, which leaves out
            self-inflicted red-heart damage and curse-room doors (a lethal hit that the bridge
            cancels is added there with its amount). Own bomb blasts count, as in the game.
  death     -5 on top of the health value lost.
  time      -1/170 per game second, every step (the score prices a half heart at 130-220 s).
  bomb      -0.08 per bomb used (a bomb kept to the end of a run is worth ~16 of 200 points).
  clear     (room clear points + kill points of the clear-blocking enemies present at the start
            + floor points) / 200, from the score tables: normal room 40, boss room 100 plus the
            Basement I floor bonus 500 (the Monstro arena is a boss room); kill points
            ceil(5 * MaxHP^0.2) per enemy. Normal rooms get about 0.24-0.8, boss rooms 3.1-3.5.
  timeout   -1 at the 120 s deadline, a termination (as combat-v1).
  progress  Potential shaping on the HP of the NPCs that keep the doors shut (CanShutDoors):
            +1 per 70 HP removed (0.05 per 3.5-damage tear); spawned enemies and regrown HP count
            negative. A death or timeout pays out the remaining potential, so every episode of a
            room earns the same progress total and the shaping only speeds up learning; it does
            not change which behaviour is best.
"""
import bisect

import numpy as np

from .abplus_geometry import CELL, fire_geometry
from .hit_rate import WINDOW_FRAMES, HitRate

PROFILE = 'combat-v2'
COMPONENTS = ('hurt', 'death', 'time', 'bomb', 'clear', 'timeout', 'progress')

HP_FULL = 6                   # half hearts of the curriculum player (3 red heart containers)
DEATH = 5.0
SECONDS_PER_UNIT = 170.0
FRAMES_PER_SECOND = 30
BOMB = 0.08
TIMEOUT = 1.0
POINTS_PER_UNIT = 200.0
ROOM_POINTS = {'normal': 40, 'boss': 100, 'arena': 100}
FLOOR_POINTS = {'normal': 0, 'boss': 500, 'arena': 500}   # Basement I, completed by its boss
PROGRESS_HP_PER_UNIT = 70.0


def _curve(x):
    return (x + 1.0 - (1.0 - x) ** 4) / 2.0


HP_WEIGHT = 1.0 / (_curve(1.0) - _curve((HP_FULL - 1) / HP_FULL))   # 11.94


def health_value(h):
    """Value of h half hearts: concave, the last half heart is worth the most."""
    h = max(0.0, float(h))
    if h >= HP_FULL:
        return HP_WEIGHT + (h - HP_FULL)
    return HP_WEIGHT * _curve(h / HP_FULL)


def health_units(player):
    """Half hearts that absorb damage before death, as the bridge's lethal check counts them."""
    return (player['hearts'] + player.get('soul', 0) + player.get('eternal', 0) + 2 * player.get('bone', 0))


def clear_bonus(kind, blocking_points):
    return (ROOM_POINTS[kind] + float(blocking_points) + FLOOR_POINTS[kind]) / POINTS_PER_UNIT


class CombatV2:
    """Per-episode reward state. reset() with the first observation, step() after every step."""

    def reset(self, obs, kind):
        if kind not in ROOM_POINTS:
            raise ValueError(f'unknown task kind {kind!r}')
        c, p = obs['combat'], obs['players'][0]
        self.kind = kind
        self.health = health_units(p)
        self.damage = float(c['player_damage'])
        self.bombs = int(p['bombs'])
        self.blocking = float(c['blocking_hp'])
        self.bonus = clear_bonus(kind, c['blocking_points'])
        self.start = dict(health=self.health, blocking_hp=self.blocking, clear_bonus=self.bonus)
        self.totals = dict.fromkeys(COMPONENTS, 0.0)

    def step(self, obs, outcome, frames):
        """Components of one step. outcome: running/death/win/time_limit (error: the caller truncates)."""
        c, p = obs['combat'], obs['players'][0]
        r = dict.fromkeys(COMPONENTS, 0.0)
        counted = float(c['player_damage']) - self.damage
        if counted > 0:
            r['hurt'] = -(health_value(self.health) - health_value(self.health - counted))
        self.damage = float(c['player_damage'])
        self.health = health_units(p)
        if outcome == 'death':
            r['death'] = -DEATH
        r['time'] = -frames / FRAMES_PER_SECOND / SECONDS_PER_UNIT
        bombs = int(p['bombs'])
        if bombs < self.bombs:
            r['bomb'] = -BOMB * (self.bombs - bombs)
        self.bombs = bombs
        if outcome == 'win':
            r['clear'] = self.bonus
        elif outcome == 'time_limit':
            r['timeout'] = -TIMEOUT
        blocking = 0.0 if outcome in ('death', 'win', 'time_limit') else float(c['blocking_hp'])
        r['progress'] = (self.blocking - blocking) / PROGRESS_HP_PER_UNIT
        self.blocking = blocking
        for k, v in r.items():
            self.totals[k] += v
        return r

    def totals_array(self):
        return np.array([self.totals[k] for k in COMPONENTS], np.float32)


def describe():
    """Constants for run configs."""
    return dict(profile=PROFILE, units='half hearts (first half heart lost = 1 = ~200 daily-run points)',
                hurt=[round(health_value(h) - health_value(h - 1), 3) for h in range(HP_FULL, 0, -1)],
                death=-DEATH, time_per_second=-1 / SECONDS_PER_UNIT, bomb=-BOMB, timeout=-TIMEOUT,
                clear='(room points + start blocker kill points ceil(5*MaxHP^0.2) + floor points) / 200',
                room_points=ROOM_POINTS, floor_points=FLOOR_POINTS,
                progress=f'potential: blocking HP / {PROGRESS_HP_PER_UNIT:g}, remainder paid at death/timeout')


# ---------------------------------------------------------------------------------------------
# combat-v3 (user decisions 2026-09-25): clearing the room is the goal. combat-v2 made staying out
# of reach the best policy: an undamaged timeout cost 1.71, clearing a normal room at the price of
# two half hearts about 1.4-2.0 (greedy evaluation of abp-mix-01e u3900: 13 of 13 normal-room
# timeouts were the player parked in a corner; 35% of training game time went to timeouts).
#
#   hurt     the combat-v2 health curve (1.00 ... 4.09 per half heart), unchanged.
#   timeout  -20: an uncleared room ends the run. Worse than every clear, even one that costs
#            five of six half hearts (7.85), so no amount of damage justifies waiting.
#   death    -25 (timeout + combat-v2's 5) plus the time and idle penalties the rest of the 120 s
#            would have cost at the current state without another hit, so dying is never cheaper
#            than waiting for the timeout.
#   time     per second (1/120 + blocking NPCs alive / 60) * (1 + (t / 60)^2): every enemy left
#            alive costs, and waiting costs more the longer it lasts (x2 at 60 s, x5 at 120 s).
#   idle     20 s without any doors-blocking NPC losing HP, at any time (from the start or since the
#            last hit): -10 when the 20 s are reached and -1/6 per second (10 per minute) until the
#            next hit. (First version, abp-mix-02: only the first minute; the policy hit once at ~2 s
#            and then parked, 24.5% of episodes by 109 game hours.)
#   hit      +1 per 20 HP the doors-blocking NPCs lose (a 3.5-damage tear +0.175), paid when hit.
#   kill     +1 per doors-blocking NPC fewer.
#            hit and kill telescope: HP and NPCs that spawn or regrow count negative when they
#            appear, so an episode earns (start HP / 20 + start NPCs) for a clear, whatever spawned.
#   clear    +3 normal room, +6 boss room and arena.
#   bomb     -0.1 per bomb (the start bomb count is randomised in training: gpu_env.sample_bombs).
# The learner trains on the sum times SCALE (returns of about the combat-v2 range); components and
# totals are reported unscaled.
# ---------------------------------------------------------------------------------------------
PROFILE_V3 = 'combat-v3'
COMPONENTS_V3 = ('hurt', 'death', 'time', 'idle', 'bomb', 'hit', 'kill', 'clear', 'timeout')
V3 = dict(timeout=20.0, death=25.0, time_base=1 / 120, time_per_enemy=1 / 60, curve_s=60.0,
          stall_s=20.0, idle_lump=10.0, idle_rate=1 / 6, bomb=0.1, hit_hp=20.0, kill=1.0,
          clear={'normal': 3.0, 'boss': 6.0, 'arena': 6.0}, deadline_s=120.0, scale=0.25)


def time_cost(t0, t1, enemies, c=V3):
    """Integral of the time price over [t0, t1] seconds with `enemies` blocking NPCs alive."""
    if t1 <= t0:
        return 0.0
    rate = c['time_base'] + c['time_per_enemy'] * enemies
    return rate * ((t1 - t0) + (t1 ** 3 - t0 ** 3) / (3 * c['curve_s'] ** 2))


def idle_cost(s0, s1, c=V3):
    """Stall penalty while the seconds since the last hit (or the start) run from s0 to s1."""
    after = c['stall_s']
    lump = c['idle_lump'] if s0 < after <= s1 else 0.0
    return lump + c['idle_rate'] * max(0.0, s1 - max(s0, after))


class CombatV3:
    """Per-episode combat-v3 state; same interface as CombatV2."""
    components = COMPONENTS_V3
    scale = V3['scale']

    def reset(self, obs, kind):
        if kind not in V3['clear']:
            raise ValueError(f'unknown task kind {kind!r}')
        c, p = obs['combat'], obs['players'][0]
        self.kind = kind
        self.health = health_units(p)
        self.damage = float(c['player_damage'])
        self.bombs = int(p['bombs'])
        self.blocking = float(c['blocking_hp'])
        self.enemies = float(c['blocking_count'])
        self.t = self.last_hit = 0.0
        self.frames = 0                  # elapsed logic frames: t is exact (no summed 2/30 drift)
        self.start = dict(health=self.health, blocking_hp=self.blocking, blocking_count=self.enemies,
                          bombs=self.bombs)
        self.totals = dict.fromkeys(COMPONENTS_V3, 0.0)

    def remaining(self):
        """What the rest of the deadline costs from the current state without another hit (charged
        at a death)."""
        end = V3['deadline_s']
        return time_cost(self.t, end, self.enemies) + idle_cost(self.t - self.last_hit, end - self.last_hit)

    def step(self, obs, outcome, frames):
        """Components of one step (unscaled). outcome: running/death/win/time_limit."""
        c, p = obs['combat'], obs['players'][0]
        r = dict.fromkeys(COMPONENTS_V3, 0.0)
        counted = float(c['player_damage']) - self.damage
        if counted > 0:
            r['hurt'] = -(health_value(self.health) - health_value(self.health - counted))
        self.damage = float(c['player_damage'])
        self.health = health_units(p)
        blocking, enemies = float(c['blocking_hp']), float(c['blocking_count'])
        r['hit'] = (self.blocking - blocking) / V3['hit_hp']
        r['kill'] = (self.enemies - enemies) * V3['kill']
        self.frames += frames
        t0, t1 = self.t, self.frames / FRAMES_PER_SECOND
        r['time'] = -time_cost(t0, t1, self.enemies)
        if blocking < self.blocking - 1e-9:
            self.last_hit = t1           # a hit this step: no stall cost, the 20 s start again
        else:
            r['idle'] = -idle_cost(t0 - self.last_hit, t1 - self.last_hit)
        self.t, self.blocking, self.enemies = t1, blocking, enemies
        bombs = int(p['bombs'])
        if bombs < self.bombs:
            r['bomb'] = -V3['bomb'] * (self.bombs - bombs)
        self.bombs = bombs
        if outcome == 'win':
            r['clear'] = V3['clear'][self.kind]
        elif outcome == 'time_limit':
            r['timeout'] = -V3['timeout']
        elif outcome == 'death':
            r['death'] = -V3['death'] - self.remaining()
        for k, v in r.items():
            self.totals[k] += v
        return r

    def totals_array(self):
        return np.array([self.totals[k] for k in COMPONENTS_V3], np.float32)


CombatV2.components = COMPONENTS
CombatV2.scale = 1.0
REWARDS = {PROFILE: CombatV2, PROFILE_V3: CombatV3}


def describe_v3():
    """Constants for run configs."""
    return dict(profile=PROFILE_V3, units='half hearts (first half heart lost = 1)',
                hurt=[round(health_value(h) - health_value(h - 1), 3) for h in range(HP_FULL, 0, -1)],
                timeout=-V3['timeout'], death=f"-{V3['death']:g} - remaining time/idle cost of the 120 s",
                time='per s -(1/120 + blocking NPCs alive/60) * (1 + (t/60)^2)',
                idle='20 s without a blocking NPC losing HP (from the start or the last hit): -10, then -1/6 per s '
                     'until the next hit',
                hit=f"+1 per {V3['hit_hp']:g} blocking HP removed (telescoping)", kill='+1 per blocking NPC fewer (telescoping)',
                clear=V3['clear'], bomb=-V3['bomb'], training_scale=V3['scale'])


# ---------------------------------------------------------------------------------------------
# combat-v4: stage one of the user's two-stage plan (2026-09-25), learn to clear rooms. Stage two
# replaces the clear reward with the measured value of the room's end state (health, bombs, floor).
# combat-v3 kept a death strictly worse than a timeout (to rule out suicide), so wherever the
# policy's chance to clear was below ~25% hiding was the better bet: abp-mix-03 hid in 17.7% of
# the episodes of rooms it cleared < 10% of the time, 2.8% where it cleared > 70%. Here almost
# nothing is negative, so neither hiding nor dying early saves anything:
#   hit      +1 per 20 HP the doors-blocking NPCs lose (telescoping: spawns and regrowth count
#            negative when they appear).
#   kill     +0.25 per doors-blocking NPC fewer (telescoping), a small reward for finishing.
#   hurt     -0.1 per damage event, flat (credit assignment only, no health curve).
#   death    -0.5, nothing else charged.
#   clear    +30 - 0.5 per half heart lost - 0.5 per bomb used, settled at the clear only. 30 is
#            more than any room's hit + kill total (largest start: boss room 1047, 415 HP = 20.75)
#            and more than 6 half hearts and 3 bombs cost (4.5): every clear beats every non-clear.
#   timeout  none: the 120 s deadline truncates (the learner bootstraps the value), and the policy
#            gets no elapsed-time input, so the value is not asked to know about the deadline.
# The time price, the stall penalty and their observation inputs (engaged, time since the last
# hit) are gone; the room state input keeps the blocking NPC count and HP fraction. Training
# scales the reward by 0.1.
# ---------------------------------------------------------------------------------------------
PROFILE_V4 = 'combat-v4'
COMPONENTS_V4 = ('hit', 'kill', 'hurt', 'death', 'clear', 'clear_health', 'clear_bombs')
V4 = dict(hit_hp=20.0, kill=0.25, hurt_event=0.1, death=0.5, clear=30.0, lambda_health=0.5, lambda_bomb=0.5,
          scale=0.1)


class CombatV4:
    """Per-episode combat-v4 state; same interface as CombatV2."""
    components = COMPONENTS_V4
    scale = V4['scale']

    def reset(self, obs, kind):
        c, p = obs['combat'], obs['players'][0]
        self.kind = kind
        self.damage0 = self.damage = float(c['player_damage'])
        self.events = float(c['player_damage_events'])
        self.bombs0 = self.bombs = int(p['bombs'])
        self.blocking = float(c['blocking_hp'])
        self.enemies = float(c['blocking_count'])
        self.start = dict(blocking_hp=self.blocking, blocking_count=self.enemies, bombs=self.bombs0)
        self.totals = dict.fromkeys(COMPONENTS_V4, 0.0)

    def step(self, obs, outcome, frames):
        """Components of one step (unscaled). outcome: running/death/win/time_limit."""
        c, p = obs['combat'], obs['players'][0]
        r = dict.fromkeys(COMPONENTS_V4, 0.0)
        blocking, enemies = float(c['blocking_hp']), float(c['blocking_count'])
        r['hit'] = (self.blocking - blocking) / V4['hit_hp']
        r['kill'] = (self.enemies - enemies) * V4['kill']
        self.blocking, self.enemies = blocking, enemies
        events = float(c['player_damage_events'])
        if events > self.events:
            r['hurt'] = -V4['hurt_event'] * (events - self.events)
        self.events = events
        self.damage = float(c['player_damage'])
        self.bombs = int(p['bombs'])
        if outcome == 'win':
            r['clear'] = V4['clear']
            r['clear_health'] = -V4['lambda_health'] * (self.damage - self.damage0)
            r['clear_bombs'] = -V4['lambda_bomb'] * max(0, self.bombs0 - self.bombs)
        elif outcome == 'death':
            r['death'] = -V4['death']
        for k, v in r.items():
            self.totals[k] += v
        return r

    def totals_array(self):
        return np.array([self.totals[k] for k in COMPONENTS_V4], np.float32)


REWARDS[PROFILE_V4] = CombatV4


def describe_v4():
    """Constants for run configs."""
    return dict(profile=PROFILE_V4, stage='one: learn to clear (user plan 2026-09-25)',
                hit=f"+1 per {V4['hit_hp']:g} blocking HP removed (telescoping)",
                kill=f"+{V4['kill']:g} per blocking NPC fewer (telescoping)",
                hurt=f"-{V4['hurt_event']:g} per damage event (flat)", death=-V4['death'],
                clear=f"+{V4['clear']:g} - {V4['lambda_health']:g}/half heart lost - {V4['lambda_bomb']:g}/bomb used, at the clear",
                timeout='truncation (value bootstrap), no penalty; no elapsed-time input', training_scale=V4['scale'])


# ---------------------------------------------------------------------------------------------
# combat-v5 (user spec 2026-09-26): combat-v4's principles (hiding earns 0, within a room every
# clear beats every non-clear, nothing charged per step, the deadline truncates) with three
# changes. Spawns, regrowth and transformations no longer count negative; an alignment potential
# pays for moving to where a tear would hit; the hit reward doubles.
#   hit      +1 per 5 HP (a tear +0.7; round 1 had 10 HP, doubled by the user's rule: hit steps'
#            advantage stayed < 1 std above the rest) the roster lineage loses (bridge abp-0.2.3
#            combat.lineage_damage: the doors-blocking NPCs present when the room was set up, and
#            every NPC a lineage death leaves, lineage_mode 1, user decision 2026-09-26: a Nest's Big
#            Spider or Trite, a Big Spider's spiders, a Mulligan's flies). Each loss counts up to the
#            HP the NPC had when it joined, from any source (own bombs, other enemies' blasts
#            included). NPCs spawned by living ones (a Nest's spiders, a Portal's enemies) earn
#            nothing; they keep the doors shut and pay through the clear.
#   kill     +0.25 per lineage NPC death (combat.lineage_kills).
#   align    gamma * Phi(s') - Phi(s), Phi = -0.2 * d_fire / 40 (abplus_geometry.fire_geometry,
#            d_fire at most 600 px, so Phi in [-3, 0]). Phi(s') = 0 after a clear or a death; the
#            truncated deadline keeps it (the learner bootstraps). Potential-based: along any
#            episode it sums to gamma^T Phi(s_T) - Phi(s_0), so it changes no ranking of outcomes.
#            gamma must equal the learner's discount (train_abplus checks it).
#   hurt, death, clear, clear_health, clear_bombs, timeout: as combat-v4.
# Largest room: boss room 1047, 415 lineage HP, hit + kill up to about 83, more than the clear (30);
# the ranking within a room still holds because a non-clear gets at most the same hit and kill
# total. Training scales the reward by 0.1.
# ---------------------------------------------------------------------------------------------
PROFILE_V5 = 'combat-v5'
COMPONENTS_V5 = ('hit', 'kill', 'align', 'hurt', 'death', 'clear', 'clear_health', 'clear_bombs')
V5 = dict(hit_hp=5.0, kill=0.25, align=0.2, gamma=0.9995, hurt_event=0.1, death=0.5, clear=30.0,
          lambda_health=0.5, lambda_bomb=0.5, scale=0.1, lineage_mode=1)


class CombatV5:
    """Per-episode combat-v5 state; same interface as CombatV2 (hit_hp, align, gamma: overrides)."""
    components = COMPONENTS_V5
    scale = V5['scale']

    def __init__(self, hit_hp=None, align=None, gamma=None):
        self.hit_hp = float(V5['hit_hp'] if hit_hp is None else hit_hp)
        self.align = float(V5['align'] if align is None else align)
        self.gamma = float(V5['gamma'] if gamma is None else gamma)

    def potential(self, d_fire):
        return -self.align * d_fire / CELL

    def reset(self, obs, kind):
        c, p = obs['combat'], obs['players'][0]
        self.kind = kind
        self.damage0 = self.damage = float(c['player_damage'])
        self.events = float(c['player_damage_events'])
        self.bombs0 = self.bombs = int(p['bombs'])
        self.lineage_damage = float(c['lineage_damage'])
        self.lineage_kills = float(c['lineage_kills'])
        self.d_fire = fire_geometry(obs)['d_fire']
        self.phi = self.potential(self.d_fire)
        self.start = dict(lineage_hp=float(c['lineage_hp']), lineage_count=float(c['lineage_count']),
                          bombs=self.bombs0, d_fire=self.d_fire)
        self.totals = dict.fromkeys(COMPONENTS_V5, 0.0)

    def step(self, obs, outcome, frames):
        """Components of one step (unscaled). outcome: running/death/win/time_limit."""
        c, p = obs['combat'], obs['players'][0]
        r = dict.fromkeys(COMPONENTS_V5, 0.0)
        damage, kills = float(c['lineage_damage']), float(c['lineage_kills'])
        r['hit'] = max(0.0, damage - self.lineage_damage) / self.hit_hp
        r['kill'] = max(0.0, kills - self.lineage_kills) * V5['kill']
        self.lineage_damage, self.lineage_kills = damage, kills
        self.d_fire = fire_geometry(obs, self.d_fire)['d_fire']
        phi = 0.0 if outcome in ('win', 'death') else self.potential(self.d_fire)
        r['align'] = self.gamma * phi - self.phi
        self.phi = phi
        events = float(c['player_damage_events'])
        if events > self.events:
            r['hurt'] = -V5['hurt_event'] * (events - self.events)
        self.events = events
        self.damage = float(c['player_damage'])
        self.bombs = int(p['bombs'])
        if outcome == 'win':
            r['clear'] = V5['clear']
            r['clear_health'] = -V5['lambda_health'] * (self.damage - self.damage0)
            r['clear_bombs'] = -V5['lambda_bomb'] * max(0, self.bombs0 - self.bombs)
        elif outcome == 'death':
            r['death'] = -V5['death']
        for k, v in r.items():
            self.totals[k] += v
        return r

    def totals_array(self):
        return np.array([self.totals[k] for k in COMPONENTS_V5], np.float32)


REWARDS[PROFILE_V5] = CombatV5


def describe_v5(hit_hp=None, align=None, gamma=None):
    """Constants for run configs."""
    reward = CombatV5(hit_hp, align, gamma)
    return dict(profile=PROFILE_V5, stage='one, revised: learn to clear (user spec 2026-09-26)',
                hit=f"+1 per {reward.hit_hp:g} HP the roster lineage loses (damage events, bridge abp-0.2.3)",
                kill=f"+{V5['kill']:g} per lineage NPC death",
                lineage=f"roster at setup + the NPCs a lineage death leaves (lineage_mode {V5['lineage_mode']}); "
                        f"spawns of living NPCs earn 0",
                align=f"gamma*Phi(s')-Phi(s), Phi = -{reward.align:g} * d_fire / 40, d_fire <= 600 px, gamma {reward.gamma:g}",
                hurt=f"-{V5['hurt_event']:g} per damage event (flat)", death=-V5['death'],
                clear=f"+{V5['clear']:g} - {V5['lambda_health']:g}/half heart lost - {V5['lambda_bomb']:g}/bomb used, at the clear",
                timeout='truncation (value bootstrap), no penalty; no elapsed-time input', training_scale=V5['scale'])


# ---------------------------------------------------------------------------------------------
# The hit-rate test (user spec 2026-09-26, a temporary experiment). The player is invincible
# (bridge abp-0.2.4), starts without bombs, and the deadline is 180 s (truncated, the learner
# bootstraps). Only three terms remain, each paid at the step it happens; nothing is settled at the
# end of the episode:
#   hit   +1 per 5 HP the roster lineage loses (combat.lineage_damage, as combat-v5)
#   kill  +0.25 per lineage NPC death (as combat-v5)
#   time  -price(rate) per game second, linear in time. rate is the running tear hit rate
#         (hit_rate.HitRate: tear hits / tears fired over the last 3 s, 0 when no tear was fired),
#         price(rate) = 2 - 1.5 * rate: 2 per second at rate 0, 0.5 at rate 1.
# No clear bonus, alignment potential, hurt, death or bomb terms: a clear pays by ending the time
# price. The observation carries the same rate (transformer_obs.COMBAT_FIELDS_HITRATE). Training
# scales the reward by 0.1.
# ---------------------------------------------------------------------------------------------
PROFILE_HR = 'combat-hitrate'
COMPONENTS_HR = ('hit', 'kill', 'time')
HR = dict(hit_hp=5.0, kill=0.25, cost_hit=0.5, cost_miss=2.0, window_frames=WINDOW_FRAMES, deadline_s=180.0,
          scale=0.1, lineage_mode=1)


def time_price(rate, cost_hit=HR['cost_hit'], cost_miss=HR['cost_miss']):
    """Time penalty per game second at a hit rate in [0, 1] (linear between the two ends)."""
    return cost_miss + (cost_hit - cost_miss) * min(1.0, max(0.0, float(rate)))


class CombatHitRate:
    """Per-episode state of the hit-rate test; same interface as CombatV2 (overrides: hit_hp,
    cost_hit = price at rate 1, cost_miss = price at rate 0)."""
    components = COMPONENTS_HR
    scale = HR['scale']

    def __init__(self, hit_hp=None, cost_hit=None, cost_miss=None):
        self.hit_hp = float(HR['hit_hp'] if hit_hp is None else hit_hp)
        self.cost_hit = float(HR['cost_hit'] if cost_hit is None else cost_hit)
        self.cost_miss = float(HR['cost_miss'] if cost_miss is None else cost_miss)

    def reset(self, obs, kind):
        c = obs['combat']
        self.kind = kind
        self.lineage_damage = float(c['lineage_damage'])
        self.lineage_kills = float(c['lineage_kills'])
        self.hit_rate = HitRate(HR['window_frames'])
        self.rate = self.hit_rate.reset(obs)
        self.start = dict(lineage_hp=float(c['lineage_hp']), lineage_count=float(c['lineage_count']))
        self.totals = dict.fromkeys(COMPONENTS_HR, 0.0)

    def step(self, obs, outcome, frames):
        """Components of one step (unscaled). outcome: running/death/win/time_limit."""
        c = obs['combat']
        r = dict.fromkeys(COMPONENTS_HR, 0.0)
        damage, kills = float(c['lineage_damage']), float(c['lineage_kills'])
        r['hit'] = max(0.0, damage - self.lineage_damage) / self.hit_hp
        r['kill'] = max(0.0, kills - self.lineage_kills) * HR['kill']
        self.lineage_damage, self.lineage_kills = damage, kills
        self.rate = self.hit_rate.update(obs)
        r['time'] = -time_price(self.rate, self.cost_hit, self.cost_miss) * frames / FRAMES_PER_SECOND
        for k, v in r.items():
            self.totals[k] += v
        return r

    def totals_array(self):
        return np.array([self.totals[k] for k in COMPONENTS_HR], np.float32)


REWARDS[PROFILE_HR] = CombatHitRate


def describe_hitrate(hit_hp=None, cost_hit=None, cost_miss=None):
    """Constants for run configs."""
    reward = CombatHitRate(hit_hp, cost_hit, cost_miss)
    return dict(profile=PROFILE_HR, stage='temporary test: invincible, no bombs, hit-rate time price (user spec 2026-09-26)',
                hit=f"+1 per {reward.hit_hp:g} HP the roster lineage loses (bridge abp-0.2.3 lineage_damage)",
                kill=f"+{HR['kill']:g} per lineage NPC death",
                time=f"-({reward.cost_miss:g} + ({reward.cost_hit:g} - {reward.cost_miss:g}) * hit_rate) per game second, every step",
                hit_rate=f"tear hits on lineage NPCs / tears fired over the last {HR['window_frames'] / FRAMES_PER_SECOND:g} s "
                         f"(bridge abp-0.2.4 combat.tear_hits, events.tears), 0 when no tear was fired",
                player='invincible (bridge abp-0.2.4 AbpSetInvincible), 0 bombs',
                deadline=f"{HR['deadline_s']:g} s, truncation (value bootstrap), no penalty",
                removed='clear bonus, alignment potential, hurt, death, bomb terms', training_scale=HR['scale'])


# ---------------------------------------------------------------------------------------------
# The walking-distance potential on the hit-rate test (user decision 2026-09-26): combat-hitrate
# plus the potential shaping term of combat-v5, with d_fire measured as the distance the player has
# to walk to a firing position (abplus_geometry.fire_geometry(walk=True)), so it has no local
# minimum in front of rocks:
#   align  gamma * Phi(s') - Phi(s), Phi = -0.2 * d_walk / 40 (d_walk at most 600 px, Phi in [-3, 0]).
#          Phi(s') = 0 after a clear; the truncated deadline keeps it (the learner bootstraps).
#          Potential-based: along any episode it sums to gamma^T Phi(s_T) - Phi(s_0), so it changes
#          no ranking of outcomes; gamma must equal the learner's discount (train_abplus passes it).
# The observation's fire_distance and the auxiliary approach label use the same walking distance.
# ---------------------------------------------------------------------------------------------
PROFILE_HRW = 'combat-hitrate-walk'
COMPONENTS_HRW = ('hit', 'kill', 'time', 'align')
HRW = dict(HR, align=0.2, gamma=0.9995)


class CombatHitRateWalk(CombatHitRate):
    """combat-hitrate with the walking-distance potential (overrides: align, gamma)."""
    components = COMPONENTS_HRW

    def __init__(self, hit_hp=None, cost_hit=None, cost_miss=None, align=None, gamma=None):
        super().__init__(hit_hp, cost_hit, cost_miss)
        self.align = float(HRW['align'] if align is None else align)
        self.gamma = float(HRW['gamma'] if gamma is None else gamma)

    def potential(self, d_walk):
        return -self.align * d_walk / CELL

    def reset(self, obs, kind):
        super().reset(obs, kind)
        self.d_walk = fire_geometry(obs, walk=True)['d_fire']
        self.phi = self.potential(self.d_walk)
        self.start['d_walk'] = self.d_walk
        self.totals = dict.fromkeys(COMPONENTS_HRW, 0.0)

    def step(self, obs, outcome, frames):
        """Components of one step (unscaled). outcome: running/death/win/time_limit."""
        r = super().step(obs, outcome, frames)
        self.d_walk = fire_geometry(obs, self.d_walk, walk=True)['d_fire']
        phi = 0.0 if outcome in ('win', 'death') else self.potential(self.d_walk)
        r['align'] = self.gamma * phi - self.phi
        self.phi = phi
        self.totals['align'] += r['align']
        return r

    def totals_array(self):
        return np.array([self.totals[k] for k in COMPONENTS_HRW], np.float32)


REWARDS[PROFILE_HRW] = CombatHitRateWalk


def describe_hitrate_walk(hit_hp=None, cost_hit=None, cost_miss=None, align=None, gamma=None):
    """Constants for run configs."""
    reward = CombatHitRateWalk(hit_hp, cost_hit, cost_miss, align, gamma)
    base = describe_hitrate(hit_hp, cost_hit, cost_miss)
    return dict(base, profile=PROFILE_HRW,
                stage='temporary test: the hit-rate test plus the walking-distance potential, all normal rooms (user decision 2026-09-26)',
                align=f"gamma*Phi(s')-Phi(s), Phi = -{reward.align:g} * d_walk / 40, d_walk = walking distance "
                      f"to the nearest firing position (<= 600 px), gamma {reward.gamma:g}; Phi(s') = 0 after a clear",
                removed='clear bonus, hurt, death, bomb terms')


# ---------------------------------------------------------------------------------------------
# combat-hitrate-miss (user decisions 2026-09-26): combat-hitrate-walk with every enemy earning
# reward and a penalty for misses.
#   every enemy  lineage mode 3 (bridge abp-0.2.5): every doors-blocking NPC joins the lineage when it
#                is first seen, spawns of living NPCs included, so hit (+1 per 5 HP), kill (+0.25) and
#                the hit rate's tear hits count on all of them. Each NPC's hits count up to the HP it
#                had when it joined.
#   miss         -0.01 * min(k, 20) for the k-th miss in a row: a tear that is gone without having
#                damaged an enemy is a miss, and a hit resets the count (bridge miss_streak / miss_units:
#                the step's penalty is -0.01 * delta(miss_units)). Paid at the step the tear is gone.
#                The cap (bridge abp-0.2.6 AbpSetMissCap, user decision 2026-09-26) stops the growth at the
#                20th miss in a row: uncapped, n misses in a row cost 0.01 * n(n+1)/2 and the untrained
#                policy's miss term was as large as the time term (EXPERIMENTS.md C18).
# Spawners can now be farmed in principle; the time price (at least 0.5 per second) is the check.
# Three rooms where farming pays at a reachable hit rate are left out of the room set (C18).
# hurt_rest (user decision 2026-09-28, after C37): when the first damage ends the episode
# (train_abplus --hurt-ends-episode, outcome 'hurt'), that hurt also costs the rest of the deadline at
# the no-hit time price, cost_miss per second (component 'rest'; combat-hitrate-hurt's death term
# without its constant). C37 without it learned to walk into enemies that earned nothing (a Gaper's
# Gusher before bridge abp-0.2.8): a hit that ends the episode for free beats paying the time price.
# With it, ending the episode by a hit is never cheaper than standing until the deadline.
# ---------------------------------------------------------------------------------------------
PROFILE_HRM = 'combat-hitrate-miss'
COMPONENTS_HRM = ('hit', 'kill', 'time', 'align', 'miss')
COMPONENTS_HRM_REST = COMPONENTS_HRM + ('rest',)
HRM = dict(HRW, miss=0.01, miss_cap=20, lineage_mode=3)


class CombatHitRateMiss(CombatHitRateWalk):
    """combat-hitrate-walk plus the miss penalty (override: miss = price of the first miss in a row;
    hurt_rest: the hurt that ends the episode costs the rest of deadline_s at cost_miss per second).

    The cap on the growth is applied by the bridge (miss_units), so the reward only prices its delta.
    """
    components = COMPONENTS_HRM

    def __init__(self, hit_hp=None, cost_hit=None, cost_miss=None, align=None, gamma=None, miss=None,
                 deadline_s=None, hurt_rest=False):
        super().__init__(hit_hp, cost_hit, cost_miss, align, gamma)
        self.miss = float(HRM['miss'] if miss is None else miss)
        self.deadline_s = float(HR['deadline_s'] if deadline_s is None else deadline_s)
        self.hurt_rest = bool(hurt_rest)
        if self.hurt_rest:
            self.components = COMPONENTS_HRM_REST

    def reset(self, obs, kind):
        super().reset(obs, kind)
        self.miss_units = float(obs['combat'].get('miss_units', 0))
        self.played_s = 0.0
        self.totals = dict.fromkeys(self.components, 0.0)

    def step(self, obs, outcome, frames):
        """Components of one step (unscaled). outcome: running/death/win/time_limit/hurt."""
        r = super().step(obs, outcome, frames)
        units = float(obs['combat'].get('miss_units', 0))
        r['miss'] = -self.miss * max(0.0, units - self.miss_units)
        self.miss_units = units
        self.totals['miss'] += r['miss']
        self.played_s += frames / FRAMES_PER_SECOND
        if self.hurt_rest:
            r['rest'] = -self.cost_miss * max(0.0, self.deadline_s - self.played_s) if outcome == 'hurt' else 0.0
            self.totals['rest'] += r['rest']
        return r

    def totals_array(self):
        return np.array([self.totals[k] for k in self.components], np.float32)


REWARDS[PROFILE_HRM] = CombatHitRateMiss


def describe_hitrate_miss(hit_hp=None, cost_hit=None, cost_miss=None, align=None, gamma=None, miss=None,
                          miss_cap=None, deadline_s=None, hurt_rest=False):
    """Constants for run configs (miss_cap: the bridge's cap on the growth, 0 = none)."""
    reward = CombatHitRateMiss(hit_hp, cost_hit, cost_miss, align, gamma, miss, deadline_s, hurt_rest)
    cap = int(HRM['miss_cap'] if miss_cap is None else miss_cap)
    rest = (dict(rest=f"at the hurt that ends the episode (--hurt-ends-episode): -{reward.cost_miss:g} per second "
                      f"left of the {reward.deadline_s:g} s deadline") if reward.hurt_rest else {})
    return dict(describe_hitrate_walk(hit_hp, cost_hit, cost_miss, align, gamma), profile=PROFILE_HRM,
                stage='temporary test: every enemy rewarded, walking potential, linearly growing miss penalty (user decisions 2026-09-26)',
                hit=f"+1 per {reward.hit_hp:g} HP any doors-blocking NPC loses, spawns included (lineage mode 3, bridge abp-0.2.5)",
                kill=f"+{HRM['kill']:g} per doors-blocking NPC death",
                miss=(f"-{reward.miss:g} * min(k, {cap}) for the k-th miss in a row (bridge abp-0.2.6 cap)" if cap
                      else f"-{reward.miss:g} * k for the k-th miss in a row") +
                     " (a tear gone without damaging an enemy); a hit resets k",
                hit_rate='tear hits on any doors-blocking NPC / tears fired over the last 3 s, 0 when no tear was fired',
                **rest)


# ---------------------------------------------------------------------------------------------
# combat-hitrate-fire (user decision 2026-09-27): combat-hitrate-miss with the tear terms credited to
# the step the tear was fired in. A tear lands 0-14 steps after it is fired (C19's check), and the
# policy fires every step, so paid at landing a hit's credit reaches the firing decision only
# through GAE, spread over the steps in between.
#   The bridge (abp-0.2.7 combat.credits) charges each frame's lineage HP loss to the tears that hit
#   (oldest first, up to each tear's damage), a lineage death to the tear that hit it last, and a
#   miss to the missing tear, per fire frame. Here the fire frame becomes a step: transition s covers
#   the logic frames (L_s, L_s+1] of consecutive observations.
#   The step reward is unchanged (every term at the step it is observed); credit[j - 1] is the part
#   of it (hit, kill, miss of tears fired j steps back, 1 <= j <= CREDIT_STEPS) that the learner moves
#   to that step's reward before GAE. Totals per episode are unchanged; a credit whose step is before
#   the rollout stays where it was observed.
# ---------------------------------------------------------------------------------------------
PROFILE_HRF = 'combat-hitrate-fire'
COMPONENTS_HRF = COMPONENTS_HRM
HRF = dict(HRM)
CREDIT_STEPS = 32


class CombatHitRateFire(CombatHitRateMiss):
    """combat-hitrate-miss whose hit, kill and miss terms carry the step their tear was fired in."""
    components = COMPONENTS_HRF

    def reset(self, obs, kind):
        super().reset(obs, kind)
        self.marks = [int(obs['logic_frames'])]
        self.credit = np.zeros(CREDIT_STEPS, np.float32)
        self.credited = 0.0      # unscaled reward of this episode moved to earlier steps

    def step(self, obs, outcome, frames):
        r = super().step(obs, outcome, frames)
        self.marks.append(int(obs['logic_frames']))
        s = len(self.marks) - 2                      # the transition just played
        credit = np.zeros(CREDIT_STEPS, np.float64)
        for fire, damage, kills, miss_units in obs['combat'].get('credits', ()):
            i = bisect.bisect_left(self.marks, fire) - 1   # the transition whose frames hold the fire frame
            j = s - i
            if i >= 0 and 1 <= j <= CREDIT_STEPS:
                credit[j - 1] += damage / self.hit_hp + HRM['kill'] * kills - self.miss * miss_units
        self.credit = credit.astype(np.float32)
        self.credited += float(credit.sum())
        return r


REWARDS[PROFILE_HRF] = CombatHitRateFire


def describe_hitrate_fire(hit_hp=None, cost_hit=None, cost_miss=None, align=None, gamma=None, miss=None,
                          miss_cap=None, deadline_s=None, hurt_rest=False):
    """Constants for run configs."""
    return dict(describe_hitrate_miss(hit_hp, cost_hit, cost_miss, align, gamma, miss, miss_cap, deadline_s, hurt_rest),
                profile=PROFILE_HRF,
                stage='every enemy rewarded, walking potential, capped miss penalty, tear terms credited to the '
                      'firing step (user decision 2026-09-27)',
                credit=f'hit, kill and miss of a tear go to the step it was fired in (bridge abp-0.2.7 combat.credits, '
                       f'up to {CREDIT_STEPS} steps back); time and alignment stay at their step')


# ---------------------------------------------------------------------------------------------
# combat-hitrate-hurt (stage 2, user decisions 2026-09-27, EXPERIMENTS.md C33): combat-hitrate-miss with the
# player no longer invincible, plus
#   hurt   combat-v2's health curve: half hearts lost from h cost V(h) - V(h - n) (1.00 for the first of six,
#          up to 4.09 for the last), counted by the game's GetTotalDamageTaken (bridge combat.player_damage).
#   death  terminal: the rest of the deadline at the no-hit time price (cost_miss per second) plus 5, so a
#          death is never cheaper than standing until the deadline (no incentive to die early in a hard room;
#          combat-v3 charged the remaining time for the same reason). The walking potential is not paid out
#          at a death (Phi keeps its value; it is at a clear), which can only make a death worse.
# Hits, kills, misses, the time price and the potential are unchanged; the deadline is truncated.
# death=False (user decision 2026-09-28, C36) drops the death term: the lethal hit still ends the
# episode and its half hearts still cost the health curve, but a death costs nothing more (so it also
# ends the time price).
# ---------------------------------------------------------------------------------------------
PROFILE_HRH = 'combat-hitrate-hurt'
COMPONENTS_HRH = COMPONENTS_HRM + ('hurt', 'death')
HRH = dict(HRM, death=DEATH)


class CombatHitRateHurt(CombatHitRateMiss):
    """combat-hitrate-miss plus the health curve and a death that costs the rest of the deadline
    (overrides: deadline_s, the episode's truncation deadline in seconds; death=False drops the death
    term, C36)."""
    components = COMPONENTS_HRH

    def __init__(self, hit_hp=None, cost_hit=None, cost_miss=None, align=None, gamma=None, miss=None,
                 deadline_s=None, death=True):
        super().__init__(hit_hp, cost_hit, cost_miss, align, gamma, miss)
        self.deadline_s = float(HR['deadline_s'] if deadline_s is None else deadline_s)
        self.death_cost = bool(death)

    def reset(self, obs, kind):
        super().reset(obs, kind)
        self.health = health_units(obs['players'][0])
        self.damage = float(obs['combat']['player_damage'])
        self.elapsed_s = 0.0
        self.start['health'] = self.health
        self.totals = dict.fromkeys(COMPONENTS_HRH, 0.0)

    def step(self, obs, outcome, frames):
        """Components of one step (unscaled). outcome: running/death/win/time_limit."""
        r = super().step(obs, outcome, frames)
        c, p = obs['combat'], obs['players'][0]
        self.elapsed_s += frames / FRAMES_PER_SECOND
        counted = float(c['player_damage']) - self.damage
        r['hurt'] = -(health_value(self.health) - health_value(self.health - counted)) if counted > 0 else 0.0
        self.damage, self.health = float(c['player_damage']), health_units(p)
        r['death'] = 0.0
        if outcome == 'death':
            if self.death_cost:
                r['death'] = -self.cost_miss * max(0.0, self.deadline_s - self.elapsed_s) - HRH['death']
            # combat-hitrate-walk paid the potential out (Phi(s') = 0); a death keeps it instead.
            kept = self.gamma * self.potential(self.d_walk)
            r['align'] += kept
            self.totals['align'] += kept
            self.phi = self.potential(self.d_walk)
        self.totals['hurt'] += r['hurt']
        self.totals['death'] += r['death']
        return r

    def totals_array(self):
        return np.array([self.totals[k] for k in COMPONENTS_HRH], np.float32)


REWARDS[PROFILE_HRH] = CombatHitRateHurt


def describe_hitrate_hurt(hit_hp=None, cost_hit=None, cost_miss=None, align=None, gamma=None, miss=None,
                          miss_cap=None, deadline_s=None, death=True):
    """Constants for run configs."""
    reward = CombatHitRateHurt(hit_hp, cost_hit, cost_miss, align, gamma, miss, deadline_s, death)
    return dict(describe_hitrate_miss(hit_hp, cost_hit, cost_miss, align, gamma, miss, miss_cap), profile=PROFILE_HRH,
                stage='stage 2: combat-hitrate-miss without invincibility (user decisions 2026-09-27)',
                player='not invincible, 0 bombs',
                hurt=[round(health_value(h) - health_value(h - 1), 3) for h in range(HP_FULL, 0, -1)],
                death=(f"-{reward.cost_miss:g} per second left of the {reward.deadline_s:g} s deadline - {HRH['death']:g}; "
                       f"the walking potential is kept (not paid out) at a death" if reward.death_cost else
                       'none (C36): the lethal hit ends the episode and costs only the health curve'),
                removed='clear bonus, bomb terms')


# ---------------------------------------------------------------------------------------------
# combat-hp (user decisions 2026-09-28, EXPERIMENTS.md C39): trained from random weights on four room groups
# (real normal rooms, one Horf, 1-3 Horfs among rocks, spawners), no curriculum terms.
#   damage   +1 per 3.5 HP any monster in the room loses: bridge abp-0.2.8-hp combat.monster_damage, every
#            active-enemy NPC (the room's own and every spawn, doors-blocking or not), real HP loss only.
#   hurt     -0.5 per half heart the player loses (combat.player_damage, Entity_Player::GetTotalDamageTaken; the
#            lethal hit the bridge cancels is counted with its amount).
#   clear    +100 when the room is cleared (outcome win).
#   timeout  -100 at the deadline (180 s), a termination; the policy observes the remaining time.
#   death    -100, a termination (user decision: a death fails the room like the deadline, so dying early never
#            saves the deadline's cost).
# No time price, potential, miss or hit-rate term; the training scale is 0.1.
# ---------------------------------------------------------------------------------------------
PROFILE_HP = 'combat-hp'
COMPONENTS_HP = ('damage', 'hurt', 'clear', 'timeout', 'death')
HPR = dict(damage_hp=3.5, hurt=0.5, clear=100.0, timeout=100.0, death=100.0, scale=0.1, lineage_mode=3)


class CombatHp:
    """Per-episode combat-hp state (overrides: damage_hp, hurt, clear, timeout, death)."""
    components = COMPONENTS_HP
    scale = HPR['scale']

    def __init__(self, damage_hp=None, hurt=None, clear=None, timeout=None, death=None):
        self.damage_hp = float(HPR['damage_hp'] if damage_hp is None else damage_hp)
        self.hurt = float(HPR['hurt'] if hurt is None else hurt)
        self.clear = float(HPR['clear'] if clear is None else clear)
        self.timeout = float(HPR['timeout'] if timeout is None else timeout)
        self.death = float(HPR['death'] if death is None else death)

    def reset(self, obs, kind):
        c = obs['combat']
        if 'monster_damage' not in c:
            raise ValueError('combat-hp needs bridge abp-0.2.8-hp (combat.monster_damage)')
        self.monster_damage = float(c['monster_damage'])
        self.player_damage = float(c['player_damage'])
        self.start = dict(monster_damage=self.monster_damage, player_damage=self.player_damage)
        self.totals = dict.fromkeys(COMPONENTS_HP, 0.0)

    def step(self, obs, outcome, frames):
        """Components of one step (unscaled). outcome: running/death/win/time_limit."""
        c = obs['combat']
        r = dict.fromkeys(COMPONENTS_HP, 0.0)
        damage, hurt = float(c['monster_damage']), float(c['player_damage'])
        r['damage'] = max(0.0, damage - self.monster_damage) / self.damage_hp
        r['hurt'] = -self.hurt * max(0.0, hurt - self.player_damage)
        self.monster_damage, self.player_damage = damage, hurt
        if outcome == 'win':
            r['clear'] = self.clear
        elif outcome == 'time_limit':
            r['timeout'] = -self.timeout
        elif outcome == 'death':
            r['death'] = -self.death
        for k, v in r.items():
            self.totals[k] += v
        return r

    def totals_array(self):
        return np.array([self.totals[k] for k in COMPONENTS_HP], np.float32)


REWARDS[PROFILE_HP] = CombatHp

# combat-hp2 (user decisions 2026-09-29, EXPERIMENTS.md C41): combat-hp with -1 per half heart, on the 1x1 and the other
# Basement I room shapes and the Basement I boss rooms (the observation's 16x28 terrain canvas, abplus_worker).
PROFILE_HP2 = 'combat-hp2'
HPR2 = dict(HPR, hurt=1.0)


class CombatHp2(CombatHp):
    """combat-hp with HPR2's default cost per half heart."""

    def __init__(self, damage_hp=None, hurt=None, clear=None, timeout=None, death=None):
        super().__init__(damage_hp, HPR2['hurt'] if hurt is None else hurt, clear, timeout, death)


REWARDS[PROFILE_HP2] = CombatHp2
REWARDS['combat-hp2-camera'] = CombatHp2   # C41 as C39's continuation: the same reward, the camera's terrain view


def describe_hp(**options):
    """Constants for run configs."""
    reward = CombatHp(**options)
    return dict(profile=PROFILE_HP, stage='C39: random weights, four room groups (user decisions 2026-09-28)',
                damage=(f'+1 per {reward.damage_hp:g} HP any monster in the room loses: every active-enemy NPC (types '
                        '10-999 but shopkeepers, fire places, poop, movable TNT), the room own and every spawn, real HP '
                        'loss only (bridge abp-0.2.8-hp combat.monster_damage)'),
                hurt=f'-{reward.hurt:g} per half heart the player loses (GetTotalDamageTaken)',
                clear=f'+{reward.clear:g} at the clear', timeout=f'-{reward.timeout:g} at the deadline, terminal',
                death=f'-{reward.death:g}, terminal', training_scale=HPR['scale'],
                removed='time price, potentials, miss and hit-rate terms, bomb term')


def describe_hp2(**options):
    """Constants for run configs (C41)."""
    return {**describe_hp(**{'hurt': HPR2['hurt'], **options}), 'profile': PROFILE_HP2,
            'stage': 'C41: random weights, 1x1 and other-shape normal rooms, rocks with Horfs, boss rooms; stat, HP and '
                     'bomb start noise (user decisions 2026-09-29)'}
