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
