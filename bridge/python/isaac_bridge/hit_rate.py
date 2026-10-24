"""Running hit rate of the player's tears (the hit-rate test, reward profile combat-hitrate).

Shots are the tears the player fired (obs events.tears, the bridge's MC_POST_FIRE_TEAR count). A
hit is a player tear that damaged a roster-lineage NPC (bridge abp-0.2.4 combat.tear_hits, one per
tear). The rate over the last `window` logic frames is hits / shots, capped at 1, and 0 when no
tear was fired in the window: not shooting is a hit rate of 0 (user spec 2026-09-26). Hits count
when they land and shots when they are fired, so under steady fire the ratio is the accuracy; it
lags by a tear's flight time (up to ~0.9 s at range 260) when firing starts or stops.

The reward (abplus_reward.CombatHitRate) and the observation (transformer_obs.VisibleHistory's
'hit_rate' field) each run one of these over the same observations, timed by obs logic_frames, so
both see the same value at every step.
"""
from collections import deque

WINDOW_FRAMES = 90   # 3 s at 30 logic frames/s


def counts(obs):
    """(tears fired, tear hits) so far in the episode's bridge counters."""
    return int(obs['events']['tears']), int(obs['combat'].get('tear_hits', 0))


class HitRate:
    def __init__(self, window_frames=WINDOW_FRAMES):
        self.window = int(window_frames)
        if self.window <= 0:
            raise ValueError('the hit-rate window must be at least one logic frame')

    def reset(self, obs):
        """Start an episode at obs; the rate is 0 until a tear is fired."""
        self.origin = int(obs['logic_frames'])
        self.shots0, self.hits0 = self.shots, self.hits = counts(obs)
        self.recent = deque()          # (logic frame, shots, hits) of each step still in the window
        self.window_shots = self.window_hits = 0
        self.rate = 0.0
        return self.rate

    def update(self, obs):
        """Rate after the step that produced obs."""
        t = int(obs['logic_frames']) - self.origin
        shots, hits = counts(obs)
        ds, dh = max(0, shots - self.shots), max(0, hits - self.hits)
        self.shots, self.hits = shots, hits
        self.recent.append((t, ds, dh))
        self.window_shots += ds
        self.window_hits += dh
        while self.recent and self.recent[0][0] <= t - self.window:
            _, s, h = self.recent.popleft()
            self.window_shots -= s
            self.window_hits -= h
        self.rate = min(1.0, self.window_hits / self.window_shots) if self.window_shots > 0 else 0.0
        return self.rate

    @property
    def episode_shots(self):
        return self.shots - self.shots0

    @property
    def episode_hits(self):
        return self.hits - self.hits0
