"""Parallel task groups (user plan 2026-09-28): several room and arena families trained together in one run,
with the training budget split by environment steps instead of by episode starts.

A groups file lists the groups; each one is an abplus_tasks spec file (normal rooms, or an aiming / dodging
arena through its 'target'), its share of the budget and its episode deadline:

    {"groups": [{"name": "normal", "tasks": "mixture_positioning_rooms.json", "share": 0.5, "episode_seconds": 180},
                {"name": "obstacles", "tasks": "mixture_target_tier8.json", "share": 0.25, "episode_seconds": 30},
                {"name": "dodge", "tasks": "mixture_target_dodge.json", "share": 0.25, "episode_seconds": 30}]}

Task paths are relative to the groups file. Within a group the rooms come from the seed by the group's own
mixture (TaskSampler, target arms included); there is no room-level PLR.

Budget modes: which group an environment slot plays next. The choice is made at an episode boundary, so an
episode in progress is never cut. Each worker prepares its next episode on the standby instance when the
current one starts, so the choice for episode k + 1 is made when episode k begins.
  episodes  each episode's group is drawn from its seed with probability = share. A group's step share
            is then share x mean episode length / sum: 50 % of the starts in 30 s episodes next to 3 s ones
            take ~91 % of the steps.
  slots     every slot plays one group for the whole run. Slots are apportioned by share and interleaved
            (GroupScheduler.slot_groups); the slots step in lockstep, so the step shares equal the slot shares.
  steps     every slot counts the decisions it played per group and starts the group furthest below its
            share, counting the episode in progress at its group's running mean length (GroupScheduler).
"""
import json
from pathlib import Path

import numpy as np

BUDGETS = ('episodes', 'slots', 'steps')
GAME_FPS = 30


def load_groups(path):
    """The groups of a groups file, resolved: name, share (normalised), tasks file, spec {weights, normal, boss},
    target (or None), episode seconds and frames."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding='utf8'))
    groups = raw['groups'] if isinstance(raw, dict) else raw
    if not groups:
        raise ValueError(f'{path}: no groups')
    names = [str(g['name']) for g in groups]
    if len(set(names)) != len(names) or not all(n.replace('_', '').replace('-', '').isalnum() for n in names):
        raise ValueError(f'{path}: group names must be unique and alphanumeric ({names})')
    shares = np.array([float(g['share']) for g in groups], np.float64)
    if (shares <= 0).any():
        raise ValueError(f'{path}: every share must be positive')
    shares = shares / shares.sum()
    out = []
    for g, share in zip(groups, shares):
        tasks = (path.parent / g['tasks']).resolve()
        spec = json.loads(tasks.read_text(encoding='utf8'))
        seconds = float(g.get('episode_seconds', 180.0))
        if not 1 <= seconds <= 3600:
            raise ValueError(f"group {g['name']}: episode_seconds must be 1..3600")
        # C41: boss rooms too (a group's TaskSampler draws them; the reset is abplus._reset_room's goto s.boss); the
        # Monstro arena of the simulator's seeds is no group (the aiming arenas are normal rooms with a target).
        if spec.get('weights', {}).get('arena', 0):
            raise ValueError(f"group {g['name']}: no Monstro arena in a group (the aiming arenas are normal rooms with a target)")
        out.append(dict(name=str(g['name']), share=float(share), tasks=str(tasks),
                        spec={k: spec[k] for k in ('weights', 'normal', 'boss')}, target=spec.get('target'),
                        seconds=seconds, frames=int(round(seconds * GAME_FPS))))
    return out


def describe_groups(groups, budget):
    """Config entry of the groups (no room lists)."""
    return dict(budget=budget, groups=[dict(name=g['name'], share=round(g['share'], 6), tasks=g['tasks'],
                                            episode_seconds=g['seconds'],
                                            rooms=len(g['spec']['normal']) + len(g['spec']['boss']),
                                            has_target=bool(g['target']),
                                            target=(g['target'] or {}).get('name') if g['target'] else None,
                                            arms=len((g['target'] or {}).get('arms') or ()))
                                       for g in groups])


def slot_groups(shares, n):
    """Group of each of n slots: every slot takes the group furthest below share x slots so far, so the counts
    are within one of share x n and the groups interleave (the first slots, e.g. the near-greedy actors, see
    every group)."""
    shares = np.asarray(shares, np.float64) / np.sum(shares)
    counts = np.zeros(len(shares))
    out = []
    for i in range(n):
        g = int(np.argmax(shares * (i + 1) - counts))
        counts[g] += 1
        out.append(g)
    return out


class GroupScheduler:
    """One environment slot's choice of the group of its next episode (budget modes in the module docstring).

    start(g) when an episode of group g begins, finish(g, decisions) when it ends; choose(seed) for the episode
    after the ones in progress."""

    def __init__(self, shares, budget, slot_group=0):
        if budget not in BUDGETS:
            raise ValueError(f'budget must be one of {BUDGETS}')
        self.shares = np.asarray(shares, np.float64) / np.sum(shares)
        self.budget = budget
        self.slot_group = int(slot_group)
        k = len(self.shares)
        self.decisions = np.zeros(k)     # decisions of finished episodes, per group
        self.episodes = np.zeros(k)      # finished episodes per group
        self.inflight = []               # groups of the episodes that began and have not finished

    def mean_length(self, g):
        """Running mean episode length of group g in decisions (before any finished: the mean of the others,
        or 1)."""
        if self.episodes[g]:
            return self.decisions[g] / self.episodes[g]
        done = self.episodes > 0
        return float(self.decisions[done].sum() / self.episodes[done].sum()) if done.any() else 1.0

    def choose(self, seed):
        if self.budget == 'slots':
            return self.slot_group
        if self.budget == 'episodes':
            rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x6A09])
            return min(int(np.searchsorted(np.cumsum(self.shares), rng.random(), side='right')), len(self.shares) - 1)
        projected = self.decisions.copy()
        for g in self.inflight:
            projected[g] += self.mean_length(g)
        deficit = self.shares * projected.sum() - projected
        best = np.flatnonzero(deficit >= deficit.max() - 1e-9)
        if len(best) == 1:
            return int(best[0])
        # Ties (a slot's first episodes): drawn from the seed by share, so the slots do not all start alike.
        p = self.shares[best] / self.shares[best].sum()
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x6A0A])
        return int(best[min(int(np.searchsorted(np.cumsum(p), rng.random(), side='right')), len(best) - 1)])

    def start(self, g):
        self.inflight.append(int(g))

    def finish(self, g, decisions):
        g = int(g)
        if g in self.inflight:
            self.inflight.remove(g)
        self.decisions[g] += float(decisions)
        self.episodes[g] += 1

    def clear_inflight(self):
        """A new collection generation discards the prepared episodes (abplus_worker.Worker.reset)."""
        self.inflight = []

    def step_shares(self):
        total = self.decisions.sum()
        return self.decisions / total if total else np.zeros(len(self.shares))
