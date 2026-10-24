"""Room Buffer (user design 2026-09-29, EXPERIMENTS.md C39): replay of seeds by their learning value, per parallel group.

A level is one seed: it fixes the room, the entrance, the enemies and where they stand, the player's start and the
game's RNG (an episode of the same seed replays exactly under the same actions, EXPERIMENTS.md A1). Each group keeps up to
`capacity` seeds. At every episode start a worker plays either a fresh seed (its own next training seed, never played)
or replays a buffer seed; fresh seeds take `fresh_share` of the group's steps.

Success is only the clear (outcome win); a death or the deadline is 0. Per seed:
    p_i  EMA of success                         (weight alpha)
    d_i  EMA of the half hearts lost in cleared episodes (weight alpha; only cleared episodes update it)
    L_i  EMA of the episode length in steps     (weight alpha)
A seed's first values start from its family's (group, room, target arm) current EMAs, else the group's, else p 0.5,
d d0, L the group's mean length.

Priority (user formula):
    P_i = 4 p_i (1 - p_i)  +  lam * p_i * clip(d_i / d0, 0, 1)  +  eta * S_i  +  eps
          learning frontier   cleared but badly               staleness    floor
    S_i = min(1, episodes of the group since seed i was last played / capacity)
P_i is the seed's share of the replay steps: a replay starts seed i with probability proportional to P_i / L'_i, where
L'_i = (n_i L_i + L_SHRINK L_family) / (n_i + L_SHRINK) shrinks the seed's length towards its family's (n_i plays): a length
known from one or two episodes is noisy, and dividing by it favours the seeds whose first episode happened to be short
(C39 smoke test: replays then took more steps than planned).

A fresh seed enters the buffer after its episode; when the buffer is full it replaces the seed of lowest value
V_i = 4 p_i (1 - p_i) + lam * p_i * clip(d_i / d0, 0, 1) if its own value is higher (staleness and floor keep a seed
visited, not in the buffer). Every episode trains (fresh ones too).

The workers read `table()` from shared memory (abplus_vec): per group the seeds and their start probabilities. Whether
an episode is fresh or a replay is each slot's own step budget (abplus_worker: abplus_groups.GroupScheduler over
(fresh, replay) with shares (f, 1 - f) in steps mode, the mechanism of the group shares): it needs no length estimate. A
start probability from the lengths of finished episodes was tried first (C39 smoke tests): early in a run the finished
replays are the short ones, so the fresh share fell to 6-26% of the steps instead of 30%. table()'s third array is the
start probability that model would give, kept for the log only.
"""
import numpy as np

DEFAULTS = dict(capacity=2000, fresh_share=0.3, alpha=0.25, lam=0.5, d0=2.0, eta=0.1, eps=0.02, family_alpha=0.1)
PRIOR_MIN = 3   # family episodes before its EMAs are a seed's prior (else the group's)
L_SHRINK = 4    # episodes' worth of the family length in a seed's start weight


class RoomBuffer:
    def __init__(self, names, capacity=2000, fresh_share=0.3, alpha=0.25, lam=0.5, d0=2.0, eta=0.1, eps=0.02,
                 family_alpha=0.1):
        if not 0 < fresh_share < 1 or capacity < 1 or not 0 < alpha <= 1 or d0 <= 0 or eps <= 0:
            raise ValueError('room buffer: need 0 < fresh_share < 1, capacity >= 1, 0 < alpha <= 1, d0 > 0, eps > 0')
        self.names = list(names)
        self.capacity, self.fresh_share, self.alpha = int(capacity), float(fresh_share), float(alpha)
        self.lam, self.d0, self.eta, self.eps, self.family_alpha = float(lam), float(d0), float(eta), float(eps), float(family_alpha)
        g, k = len(self.names), self.capacity
        self.seeds = np.full((g, k), -1, np.int64)
        self.p, self.d, self.L = np.zeros((g, k)), np.zeros((g, k)), np.zeros((g, k))
        self.plays = np.zeros((g, k), np.int64)
        self.last = np.zeros((g, k), np.int64)
        self.families = [dict() for _ in self.names]            # key -> [p, d, L, episodes]
        self.group = [dict(p=None, d=None, L=None, n=0) for _ in self.names]
        self.fresh_L = np.zeros(g)                               # EMA of the fresh episodes' length (0: none yet)
        self.replay_L = np.zeros(g)                              # EMA of the replayed episodes' length (0: none yet)
        self.family_of = [[None] * k for _ in self.names]        # slot -> family key
        self.episodes = np.zeros(g, np.int64)                    # finished episodes per group (the staleness clock)
        self.admitted = np.zeros(g, np.int64)
        self.evicted = np.zeros(g, np.int64)
        self.discarded = np.zeros(g, np.int64)
        self.where = [dict() for _ in self.names]                # seed -> slot

    # ------------------------------------------------------------------ statistics
    def _ema(self, old, new, weight):
        return new if old is None else old + weight * (new - old)

    def prior(self, g, family):
        """(p, d, L) a seed of this family starts from."""
        f = self.families[g].get(family)
        s = self.group[g]
        if f is not None and f[3] >= PRIOR_MIN:
            return f[0], f[1], f[2]
        if s['n']:
            return s['p'], s['d'], s['L']
        return 0.5, self.d0, 1.0

    def value(self, p, d):
        return 4 * p * (1 - p) + self.lam * p * np.clip(d / self.d0, 0.0, 1.0)

    def observe(self, g, seed, win, damage, steps, family, replay):
        """One finished episode of group g: seed, success, half hearts lost, length (steps), family key, replayed or fresh."""
        win, damage, steps = float(bool(win)), float(damage), float(steps)
        self.episodes[g] += 1
        p0, d0, L0 = self.prior(g, family)
        a = self.alpha
        slot = self.where[g].get(int(seed))
        if slot is None:
            # A fresh seed (or one evicted meanwhile): its prior, then this episode.
            p = p0 + a * (win - p0)
            d = d0 + a * (damage - d0) if win else d0
            L = steps if L0 <= 1.0 else L0 + a * (steps - L0)
            self._admit(g, int(seed), p, d, L, family)
        else:
            self.p[g, slot] += a * (win - self.p[g, slot])
            if win:
                self.d[g, slot] += a * (damage - self.d[g, slot])
            self.L[g, slot] += a * (steps - self.L[g, slot])
            self.plays[g, slot] += 1
            self.last[g, slot] = self.episodes[g]
        fam = self.families[g].setdefault(family, [p0, d0, L0 if L0 > 1.0 else steps, 0])
        w = self.family_alpha if fam[3] >= PRIOR_MIN else 1.0 / (fam[3] + 1)
        fam[0] += w * (win - fam[0])
        if win:
            fam[1] += w * (damage - fam[1])
        fam[2] += w * (steps - fam[2])
        fam[3] += 1
        s = self.group[g]
        w = self.family_alpha if s['n'] >= PRIOR_MIN else 1.0 / (s['n'] + 1)
        s['p'] = self._ema(s['p'], win, w)
        if win:
            s['d'] = self._ema(s['d'], damage, w)
        elif s['d'] is None:
            s['d'] = self.d0
        s['L'] = self._ema(s['L'], steps, w)
        s['n'] += 1
        lengths = self.replay_L if replay else self.fresh_L
        lengths[g] = steps if lengths[g] == 0 else lengths[g] + self.family_alpha * (steps - lengths[g])

    def _admit(self, g, seed, p, d, L, family=None):
        used = self.seeds[g] >= 0
        if not used.all():
            slot = int(np.flatnonzero(~used)[0])
        else:
            values = self.value(self.p[g], self.d[g])
            slot = int(np.argmin(values))
            if self.value(p, d) <= values[slot]:
                self.discarded[g] += 1
                return
            del self.where[g][int(self.seeds[g, slot])]
            self.evicted[g] += 1
        self.seeds[g, slot], self.p[g, slot], self.d[g, slot], self.L[g, slot] = seed, p, d, max(1.0, L)
        self.plays[g, slot], self.last[g, slot] = 1, self.episodes[g]
        self.family_of[g][slot] = family
        self.where[g][seed] = slot
        self.admitted[g] += 1

    # ------------------------------------------------------------------ sampling
    def priorities(self, g):
        """P_i of group g's slots (0 for empty ones)."""
        used = self.seeds[g] >= 0
        stale = np.minimum(1.0, (self.episodes[g] - self.last[g]) / self.capacity)
        P = self.value(self.p[g], self.d[g]) + self.eta * stale + self.eps
        return np.where(used, P, 0.0)

    def lengths(self, g):
        """L'_i of group g's slots: the seed's EMA length shrunk towards its family's (the group's without one)."""
        n = self.plays[g].astype(np.float64)
        family = np.array([self.prior(g, f)[2] if f is not None else (self.group[g]['L'] or 1.0)
                           for f in self.family_of[g]], np.float64)
        return np.maximum((n * self.L[g] + L_SHRINK * family) / (n + L_SHRINK), 1.0)

    def table(self):
        """(seeds (G, K), start probabilities (G, K), fresh start probability (G,)) for the workers."""
        g_count = len(self.names)
        probs = np.zeros(self.seeds.shape)
        fresh = np.ones(g_count)
        for g in range(g_count):
            P = self.priorities(g)
            if P.sum() <= 0:
                continue
            w = P / self.lengths(g)
            probs[g] = w / w.sum()
            # The realised mean lengths of replayed and fresh episodes (the model-based ones before any finished).
            replay_length = float(self.replay_L[g]) if self.replay_L[g] > 0 else float(P.sum() / w.sum())
            fresh_length = float(self.fresh_L[g]) if self.fresh_L[g] > 0 else (self.group[g]['L'] or replay_length)
            f = self.fresh_share
            fresh[g] = f * replay_length / (f * replay_length + (1 - f) * fresh_length)
        return self.seeds.copy(), probs, fresh

    def summary(self):
        """Scalars per group for the training log."""
        out = {}
        _, probs, fresh = self.table()
        for g, name in enumerate(self.names):
            used = self.seeds[g] >= 0
            n = int(used.sum())
            out[f'{name}/size'] = n
            out[f'{name}/fresh_start_p_model'] = float(fresh[g])
            out[f'{name}/admitted'] = int(self.admitted[g])
            out[f'{name}/evicted'] = int(self.evicted[g])
            out[f'{name}/discarded'] = int(self.discarded[g])
            if not n:
                continue
            p, d, L = self.p[g][used], self.d[g][used], self.L[g][used]
            P = self.priorities(g)[used]
            frontier = 4 * p * (1 - p)
            damage = self.lam * p * np.clip(d / self.d0, 0.0, 1.0)
            out[f'{name}/p_mean'] = float(p.mean())
            out[f'{name}/p_frontier_share'] = float(((p >= 0.2) & (p <= 0.8)).mean())
            out[f'{name}/p_solved_share'] = float((p > 0.9).mean())
            out[f'{name}/p_unsolved_share'] = float((p < 0.1).mean())
            out[f'{name}/d_mean_solved'] = float(d[p > 0.5].mean()) if (p > 0.5).any() else 0.0
            out[f'{name}/mass_frontier'] = float(frontier.sum() / P.sum())
            out[f'{name}/mass_damage'] = float(damage.sum() / P.sum())
            out[f'{name}/length_mean'] = float(L.mean())
            out[f'{name}/fresh_length'] = float(self.fresh_L[g])
            out[f'{name}/replay_length'] = float(self.replay_L[g])
            out[f'{name}/plays_mean'] = float(self.plays[g][used].mean())
            q = probs[g][used]
            q = q[q > 0]
            out[f'{name}/effective_seeds'] = float(np.exp(-(q * np.log(q)).sum())) if len(q) else 0.0
        return out

    # ------------------------------------------------------------------ checkpoints
    def state_dict(self):
        return dict(names=self.names, params=dict(capacity=self.capacity, fresh_share=self.fresh_share, alpha=self.alpha,
                                                  lam=self.lam, d0=self.d0, eta=self.eta, eps=self.eps,
                                                  family_alpha=self.family_alpha),
                    seeds=self.seeds.tolist(), p=self.p.tolist(), d=self.d.tolist(), L=self.L.tolist(),
                    plays=self.plays.tolist(), last=self.last.tolist(),
                    families=[{k: v for k, v in f.items()} for f in self.families], group=self.group,
                    fresh_L=self.fresh_L.tolist(), replay_L=self.replay_L.tolist(), family_of=self.family_of,
                    episodes=self.episodes.tolist(), admitted=self.admitted.tolist(),
                    evicted=self.evicted.tolist(), discarded=self.discarded.tolist())

    @classmethod
    def from_state(cls, state, **overrides):
        """overrides: parameters that replace the saved ones (a resume with other flags); the capacity must match."""
        params = {**state['params'], **overrides}
        if params['capacity'] != state['params']['capacity']:
            raise ValueError('the room buffer capacity of a resumed run must stay the same')
        b = cls(state['names'], **params)
        for key in ('seeds', 'plays', 'last'):
            setattr(b, key, np.asarray(state[key], np.int64))
        for key in ('p', 'd', 'L', 'fresh_L', 'replay_L'):
            setattr(b, key, np.asarray(state[key], np.float64))
        b.family_of = [list(row) for row in state['family_of']]
        for key in ('episodes', 'admitted', 'evicted', 'discarded'):
            setattr(b, key, np.asarray(state[key], np.int64))
        b.families = [{k: list(v) for k, v in f.items()} for f in state['families']]
        b.group = [dict(s) for s in state['group']]
        b.where = [{int(s): i for i, s in enumerate(row) if s >= 0} for row in b.seeds]
        return b


def family_key(group, seed, layout):
    """The prior family of an episode: the room, and for target arms the arm the seed draws (abplus_tasks.target_arm, a pure
    function of the seed)."""
    from .abplus_tasks import target_arm
    arms = (group.get('target') or {}).get('arms')
    return f'{int(layout)}:{target_arm(int(seed), arms)}' if arms else str(int(layout))
