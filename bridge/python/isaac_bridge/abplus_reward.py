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
import numpy as np

from .abplus_geometry import CELL, fire_geometry

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
