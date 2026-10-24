"""Room-level task mixture for AB+ training: which room an episode of a given seed plays.

Tasks (user decision 2026-09-24: RL for combat only, room-level mixture):
  arena   the simulator's Monstro arena (abplus.sim_arena; plain Monstro, sim spawn points)
  normal  a Basement normal room with the enemies its layout spawns (goto d.<variant>)
  boss    a boss room with its own boss, champion roll included (goto s.boss.<variant>)
The rooms come from a catalog probed from the live engine (abplus_room_catalog.py), not from
the repository's Repentance room files. Selection is a pure function of the episode seed, so
training, evaluation and replays agree on the room of every seed.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TASKS = ('arena', 'normal', 'boss')


@dataclass(frozen=True)
class Task:
    kind: str       # 'arena' | 'normal' | 'boss'
    variant: int    # room variant (arena: the sim layout)
    entrance: int   # preferred door slot 0..3 (LEFT0/UP0/RIGHT0/DOWN0); arena: unused
    arm: int = -1   # target-arena arm (the tasks file's target['arms'], tier 6); -1: none


def target_arm(seed, arms):
    """Index of the target-arena arm an episode seed plays (tier 6, EXPERIMENTS.md C28): drawn by the
    arms' weights from the seed alone, so room retries keep it."""
    weights = np.array([float(a.get('weight', 1.0)) for a in arms], np.float64)
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x7A29])
    i = int(np.searchsorted(np.cumsum(weights / weights.sum()), rng.random(), side='right'))
    return min(i, len(arms) - 1)


class TaskSampler:
    def __init__(self, weights, normal=(), boss=(), arms=None):
        self.weights = {k: float(weights.get(k, 0)) for k in TASKS}
        self.rooms = {'normal': tuple(int(v) for v in normal), 'boss': tuple(int(v) for v in boss)}
        for kind in ('normal', 'boss'):
            if self.weights[kind] > 0 and not self.rooms[kind]:
                raise ValueError(f'task {kind} has weight but no rooms')
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError('all task weights are zero')
        self.cumulative = np.cumsum([self.weights[k] / total for k in TASKS])
        # Target arms (tier 6): a normal-room task first draws its arm, then a room of that arm's list.
        self.arms = list(arms) if arms else None
        if self.arms and not all(a.get('rooms') for a in self.arms):
            raise ValueError('every target arm needs rooms')

    @classmethod
    def from_file(cls, path, weights=None):
        spec = json.loads(Path(path).read_text(encoding='utf8'))
        return cls(weights or spec['weights'], spec['normal'], spec['boss'], (spec.get('target') or {}).get('arms'))

    def choose(self, seed, retry=0):
        """Task of an episode seed. retry > 0: the deterministic next candidate when a room turned out
        unusable for this seed (e.g. its random layout spawns produced no enemy that blocks the clear)."""
        key = [int(seed) & 0xFFFFFFFF, 0x7A5C] + ([int(retry)] if retry else [])
        rng = np.random.default_rng(key)
        kind = TASKS[int(np.searchsorted(self.cumulative, rng.random(), side='right'))]
        entrance = int(rng.integers(4))
        if kind == 'arena':
            return Task('arena', 0, entrance)
        if kind == 'normal' and self.arms:
            arm = target_arm(seed, self.arms)
            rooms = self.arms[arm]['rooms']
            return Task(kind, int(rooms[int(rng.integers(len(rooms)))]), entrance, arm)
        rooms = self.rooms[kind]
        return Task(kind, rooms[int(rng.integers(len(rooms)))], entrance)

    def describe(self):
        out = {'weights': self.weights, 'normal_rooms': len(self.rooms['normal']), 'boss_rooms': len(self.rooms['boss'])}
        if self.arms:
            out['arms'] = [{'name': a.get('name'), 'weight': a.get('weight', 1.0), 'rooms': len(a['rooms'])} for a in self.arms]
        return out
