"""Prioritized Level Replay over the rooms of the AB+ mixture (user decision 2026-09-25).

Jiang, Grefenstette and Rocktaschel, "Prioritized Level Replay" (ICML 2021). A level is a room: the
Monstro arena, one Basement I normal room or one Basement I boss room. After every rollout the
learner scores each finished episode by its positive value loss, the mean over its steps of
max(GAE advantage, 0) (episodes that span rollouts accumulate over them), and replaces its room's
score with it.

The room kinds keep the mixture's weights (--tasks-file: arena 0.2, normal 0.45, boss 0.35); PLR only
decides which room of a kind is played:

    P(room) = w_kind * P_kind(room)
    P_kind  = (1 - floor) * [(1 - u) * ((1 - rho) * P_rank + rho * P_stale) + u * U_unseen] + floor * U_kind

  P_rank   (1 / rank of the score) ^ (1 / beta) over the kind's rooms played so far
  P_stale  episodes since the room last finished, normalised over them (rho: staleness weight)
  u        fraction of the kind's rooms not played yet (they are tried uniformly first, as in PLR)
  floor    a uniform share that keeps every room of the kind in the training distribution

One distribution over all 573 rooms, as first written, made the arena one room among 573: in
abp-mix-04 it got ~0.1% of the episodes and the 77 boss rooms ~8%, against the mixture's 20% and 35%.

beta = 1 (P_rank proportional to 1 / rank: the top 10 of the 495 normal rooms get ~43% of it) rather
than the paper's 0.1-0.3, which with 16 environments and 1-2 episodes per environment per rollout
would put most of every rollout into one or two rooms. The workers read P from shared memory at
every episode start (abplus_worker.PlrTaskChooser); evaluation keeps the seed's fixed room.
"""
import numpy as np


class PrioritizedLevels:
    def __init__(self, levels, weights, beta=1.0, staleness=0.3, floor=0.1):
        """levels: (kind, variant) rooms (mixture_levels); weights: the mixture's weight of each kind."""
        self.levels = [tuple(level) for level in levels]
        n = len(self.levels)
        if n == 0:
            raise ValueError('PLR needs at least one level')
        self.beta, self.staleness, self.floor = float(beta), float(staleness), float(floor)
        kinds = list(dict.fromkeys(kind for kind, _ in self.levels))
        unweighted = [kind for kind in kinds if float(weights.get(kind, 0)) <= 0]
        if unweighted:
            raise ValueError(f'PLR rooms of kinds without mixture weight: {unweighted}')
        total = sum(float(weights[kind]) for kind in kinds)
        self.weights = {kind: float(weights[kind]) / total for kind in kinds}
        self.groups = {kind: np.array([i for i, (k, _) in enumerate(self.levels) if k == kind]) for kind in kinds}
        self.scores = np.zeros(n)
        self.seen = np.zeros(n, bool)
        self.last = np.zeros(n)          # finished-episode count when the room last finished
        self.plays = np.zeros(n, np.int64)
        self.finished = 0
        self.partial = {}                # env -> [level, sum of positive advantages, steps]

    def update(self, advantages, starts, levels):
        """One rollout: advantages and episode starts (T, N), level id of every step (T, N)."""
        advantages = np.asarray(advantages, np.float64)
        starts = np.asarray(starts).astype(bool)
        levels = np.asarray(levels, np.int64)
        positive = np.maximum(advantages, 0.0)
        steps_total, envs = advantages.shape
        for i in range(envs):
            level, total, steps = self.partial.get(i, (-1, 0.0, 0))
            for t in range(steps_total):
                if starts[t, i] and steps:
                    self._finish(level, total / steps)
                    total, steps = 0.0, 0
                level = int(levels[t, i])
                total += positive[t, i]
                steps += 1
            self.partial[i] = (level, total, steps)

    def _finish(self, level, score):
        if level < 0:
            return
        self.finished += 1
        self.scores[level] = score
        self.seen[level] = True
        self.last[level] = self.finished
        self.plays[level] += 1

    def _within(self, rooms):
        """P_kind over the rooms (level indices) of one kind; sums to 1."""
        n = len(rooms)
        q = np.zeros(n)
        seen = np.flatnonzero(self.seen[rooms])
        unseen = np.flatnonzero(~self.seen[rooms])
        if len(seen):
            ranks = np.empty(len(seen))
            ranks[np.argsort(-self.scores[rooms[seen]], kind='stable')] = np.arange(1, len(seen) + 1)
            rank = (1.0 / ranks) ** (1.0 / self.beta)
            rank /= rank.sum()
            stale = self.finished - self.last[rooms[seen]]
            stale = stale / stale.sum() if stale.sum() > 0 else np.full(len(seen), 1.0 / len(seen))
            q[seen] = (1 - self.staleness) * rank + self.staleness * stale
        if len(unseen):
            u = len(unseen) / n
            q *= 1 - u
            q[unseen] += u / len(unseen)
        q = (1 - self.floor) * q + self.floor / n
        return q / q.sum()

    def probabilities(self):
        p = np.zeros(len(self.levels))
        for kind, rooms in self.groups.items():
            p[rooms] = self.weights[kind] * self._within(rooms)
        return p / p.sum()

    def summary(self, p=None):
        """Scalars for the training log: each kind's share and effective number of rooms, and totals."""
        p = self.probabilities() if p is None else p
        result = {}
        for kind, rooms in sorted(self.groups.items()):
            q = p[rooms]
            result[f'share_{kind}'] = float(q.sum())
            within = q[q > 0] / q.sum()
            result[f'effective_{kind}'] = float(np.exp(-(within * np.log(within)).sum()))
        nz = p[p > 0]
        result.update(seen=int(self.seen.sum()), finished=int(self.finished), max_p=float(p.max()),
                      effective_levels=float(np.exp(-(nz * np.log(nz)).sum())),
                      score_mean=float(self.scores[self.seen].mean()) if self.seen.any() else 0.0)
        return result

    def top(self, k=5, p=None):
        p = self.probabilities() if p is None else p
        order = np.argsort(-p)[:k]
        return [dict(level=list(self.levels[i]), p=round(float(p[i]), 4), score=round(float(self.scores[i]), 4),
                     plays=int(self.plays[i])) for i in order]

    def state_dict(self):
        return dict(levels=[list(level) for level in self.levels], weights=self.weights, beta=self.beta,
                    staleness=self.staleness, floor=self.floor, scores=self.scores.tolist(), seen=self.seen.tolist(),
                    last=self.last.tolist(), plays=self.plays.tolist(), finished=self.finished,
                    partial={str(k): list(v) for k, v in self.partial.items()})

    @classmethod
    def from_state(cls, state, weights=None):
        """weights: the kinds' mixture weights, by default the saved ones (states saved before the kinds
        kept their weights have none, and need them passed)."""
        weights = weights if weights is not None else state.get('weights')
        if weights is None:
            raise ValueError('this PLR state has no kind weights; pass the mixture weights')
        plr = cls(state['levels'], weights, state['beta'], state['staleness'], state['floor'])
        plr.scores = np.asarray(state['scores'], np.float64)
        plr.seen = np.asarray(state['seen'], bool)
        plr.last = np.asarray(state['last'], np.float64)
        plr.plays = np.asarray(state['plays'], np.int64)
        plr.finished = int(state['finished'])
        # Episodes in progress at the checkpoint restart fresh on resume.
        return plr


def mixture_levels(spec):
    """Levels of an abplus_tasks spec: the arena (if it has weight), then normal and boss rooms."""
    levels = [('arena', 0)] if spec['weights'].get('arena', 0) > 0 else []
    if spec['weights'].get('normal', 0) > 0:
        levels += [('normal', int(v)) for v in spec['normal']]
    if spec['weights'].get('boss', 0) > 0:
        levels += [('boss', int(v)) for v in spec['boss']]
    return levels
