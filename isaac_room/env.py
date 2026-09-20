"""Approximate combat, not extracted Isaac logic. No renderer needed for stepping."""
from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin, sqrt
from operator import index
import random

import numpy as np

from .geometry import circle_rect, move_circle, segment_circle, swept_circle_rect, wall_impact

DT = 1 / 60
WIDTH, HEIGHT = 640, 400
# 0 idle; clockwise from up. Shooting uses only the four cardinal directions.
MOVE = ((0, 0), (0, -1), (1 / sqrt(2), -1 / sqrt(2)), (1, 0),
        (1 / sqrt(2), 1 / sqrt(2)), (0, 1), (-1 / sqrt(2), 1 / sqrt(2)),
        (-1, 0), (-1 / sqrt(2), -1 / sqrt(2)))
SHOOT = ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0))
LAYOUTS = {'empty': (), 'pillars': ((220, 155, 42, 72), (378, 155, 42, 72))}


@dataclass(frozen=True)
class Config:
    layout: str = 'pillars'
    action_repeat: int = 4
    max_ticks: int = 3600

    def __post_init__(self):
        if self.layout not in LAYOUTS:
            raise ValueError(f'Unknown layout: {self.layout}')
        if self.action_repeat < 1 or self.max_ticks < 1:
            raise ValueError('action_repeat and max_ticks must be positive')


@dataclass
class Player:
    x: float = 320
    y: float = 310
    vx: float = 0
    vy: float = 0
    radius: float = 12
    hp: int = 6
    invulnerable: int = 0
    cooldown: int = 0


@dataclass
class Enemy:
    kind: int  # 0 pursuer, 1 stationary fan shooter (generic prototype archetypes)
    x: float
    y: float
    vx: float = 0
    vy: float = 0
    radius: float = 15
    hp: int = 6
    cooldown: int = 60
    visible: bool = True


@dataclass
class Bullet:
    team: int  # 0 player tear, 1 enemy projectile
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 5
    ttl: int = 120
    visible: bool = True


class RoomEnv:
    """reset -> (obs, info); step -> (obs, reward, terminated, truncated, info).

    Gymnasium-shaped API, intentionally not a gymnasium.Env subclass yet.
    Policy consumes observe() only; info/internal objects are evaluator channels.
    """

    def __init__(self, config=None):
        self.config = config or Config()
        self.rng = random.Random()
        self.ready = False

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('No reset options yet; use Config for layout and time limit')
        if seed is not None:
            self.rng.seed(seed)
        self.player = Player()
        self.blocks = LAYOUTS[self.config.layout]
        self.enemies = [Enemy(kind, x + self.rng.uniform(-18, 18), y + self.rng.uniform(-18, 18),
                              cooldown=self.rng.randrange(40, 100))
                        for kind, x, y in ((0, 145, 95), (0, 495, 125),
                                          (1, 105, 270), (1, 535, 65))]
        self.bullets = []
        self.ticks = self.kills = self.damage_taken = 0
        self.outcome = 'running'
        self.ready = True
        return self.observe(), self._info()

    def observe(self):
        p = self.player
        # No global entity IDs/order derived from hidden entities; no HP, timers,
        # RNG, bullet lifetime, or seed in enemy/projectile observations.
        enemies = sorted((e.kind, e.x, e.y, e.vx, e.vy, e.radius)
                         for e in self.enemies if e.visible and self._onscreen(e))
        bullets = sorted((b.team, b.x, b.y, b.vx, b.vy, b.radius)
                         for b in self.bullets if b.visible and self._onscreen(b))
        return {
            'player': np.array([p.x, p.y, p.vx, p.vy, p.radius, p.hp,
                                int(p.invulnerable > 0)], dtype=np.float32),
            'enemies': np.array(enemies, dtype=np.float32).reshape(-1, 6),
            'bullets': np.array(bullets, dtype=np.float32).reshape(-1, 6),
            'obstacles': np.array(self.blocks, dtype=np.float32).reshape(-1, 4),
            'room': np.array([WIDTH, HEIGHT], dtype=np.float32),
            'tick': self.ticks,
        }

    @staticmethod
    def _onscreen(entity):
        return (-entity.radius < entity.x < WIDTH + entity.radius
                and -entity.radius < entity.y < HEIGHT + entity.radius)

    def _info(self):
        return {'outcome': self.outcome, 'ticks': self.ticks,
                'kills': self.kills, 'damage_taken': self.damage_taken}

    def step(self, action):
        if not self.ready or self.outcome != 'running':
            raise RuntimeError('Call reset before stepping a new episode')
        if len(action) != 2:
            raise ValueError('Action is (move: 0..8, shoot: 0..4)')
        move, shoot = (index(value) for value in action)
        if not 0 <= move < 9 or not 0 <= shoot < 5:
            raise ValueError('Action is (move: 0..8, shoot: 0..4)')
        reward = 0.0
        for _ in range(self.config.action_repeat):
            reward += self._tick(move, shoot)
            if self.outcome != 'running':
                break
        return (self.observe(), reward, self.outcome in ('cleared', 'dead'),
                self.outcome == 'timeout', self._info())

    def _tick(self, move, shoot):
        self.ticks += 1
        p = self.player
        old_hp, old_kills = p.hp, self.kills
        p.invulnerable = max(0, p.invulnerable - 1)
        p.cooldown = max(0, p.cooldown - 1)
        dx, dy = MOVE[move]
        p.vx += (dx * 165 - p.vx) * 0.24
        p.vy += (dy * 165 - p.vy) * 0.24
        x, y = move_circle(p.x, p.y, p.vx * DT, p.vy * DT, p.radius,
                           self.blocks, WIDTH, HEIGHT)
        p.vx, p.vy = (x - p.x) / DT, (y - p.y) / DT
        p.x, p.y = x, y
        if shoot and p.cooldown == 0:
            sx, sy = SHOOT[shoot]
            self.bullets.append(Bullet(0, p.x + sx * 17, p.y + sy * 17,
                                       sx * 360, sy * 360, ttl=75))
            p.cooldown = 12

        for e in self.enemies:
            if e.kind == 0:
                heading = atan2(p.y - e.y, p.x - e.x)
                # Local steering around rocks, not the original NPC pathfinder.
                candidates = []
                for turn in (0, pi / 4, -pi / 4, pi / 2, -pi / 2):
                    vx, vy = cos(heading + turn) * 72, sin(heading + turn) * 72
                    ex, ey = move_circle(e.x, e.y, vx * DT, vy * DT, e.radius,
                                         self.blocks, WIDTH, HEIGHT)
                    if hypot(ex - e.x, ey - e.y) > 0.5:
                        candidates.append((hypot(p.x - ex, p.y - ey), ex, ey))
                if candidates:
                    _, ex, ey = min(candidates)
                    e.vx, e.vy = (ex - e.x) / DT, (ey - e.y) / DT
                    e.x, e.y = ex, ey
                else:
                    e.vx = e.vy = 0
            else:
                e.cooldown -= 1
                if e.cooldown <= 0:
                    heading = atan2(p.y - e.y, p.x - e.x)
                    for angle in (heading - 0.22, heading, heading + 0.22):
                        self.bullets.append(Bullet(1, e.x, e.y, cos(angle) * 125,
                                                   sin(angle) * 125, ttl=240))
                    e.cooldown = 100

        damage_dealt = 0
        surviving = []
        for b in self.bullets:
            bx, by = b.x + b.vx * DT, b.y + b.vy * DT
            hit_time = wall_impact(b.x, b.y, bx, by, b.radius, WIDTH, HEIGHT)
            first = 2.0 if hit_time is None else hit_time
            target = None
            for rect in self.blocks:
                t = swept_circle_rect(b.x, b.y, bx, by, b.radius, rect)
                if t is not None and t < first:
                    first = t
            targets = self.enemies if b.team == 0 else [p]
            for actor in targets:
                if actor.hp <= 0:
                    continue
                t = segment_circle(b.x, b.y, bx, by, actor.x, actor.y, actor.radius + b.radius)
                if t is not None and t < first:
                    first, target = t, actor
            if first <= 1:
                if target is p:
                    self._hurt_player()
                elif target is not None:
                    damage = min(2, target.hp)
                    target.hp -= damage
                    damage_dealt += damage
                continue
            b.x, b.y = bx, by
            b.ttl -= 1
            if b.ttl > 0:
                surviving.append(b)
        self.bullets = surviving
        self.kills += sum(e.hp <= 0 for e in self.enemies)
        self.enemies = [e for e in self.enemies if e.hp > 0]
        for e in self.enemies:
            if hypot(e.x - p.x, e.y - p.y) < e.radius + p.radius:
                self._hurt_player()
        self.damage_taken += old_hp - p.hp
        reward = 0.1 * damage_dealt + self.kills - old_kills - (old_hp - p.hp) - 0.001
        if p.hp <= 0:
            self.outcome = 'dead'
            reward -= 5
        elif not self.enemies:
            self.outcome = 'cleared'
            reward += 5
        elif self.ticks >= self.config.max_ticks:
            self.outcome = 'timeout'
        return reward

    def _hurt_player(self):
        if self.player.invulnerable == 0 and self.player.hp > 0:
            self.player.hp -= 1
            self.player.invulnerable = 45

    def close(self):
        self.ready = False
