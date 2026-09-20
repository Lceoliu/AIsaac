"""Gymnasium adapter for the fixed, original-engine Monstro combat curriculum.

Actor tensors use only an explicit subset of the visible observation, never info.
Rewards use confirmed damage settlements and a clear bonus. A time limit
truncates (it is not a loss). Gym's seed seeds Python sampling, not engine RNG;
the Isaac arena seed is a separate, fixed string and does not promise replay.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .training import IsaacTrainingEnv
from .env import BridgeError


ENTITY_SLOTS = 128
TERRAIN_SHAPE = (5, 9, 15)
OBSERVATION_SCHEMA = 'monstro-combat-v2'


def terrain_observation(obs):
    """Inside, walkable, collision class / 5, pit, potential spike hazard.

    Cell-center passability for this 1x1 room, not a continuous path planner.
    Pits depend on the current player's flight; hazards can still be walkable.
    """
    raw = obs['terrain']
    if (raw['height'], raw['width']) != TERRAIN_SHAPE[1:]:
        raise ValueError('This curriculum requires a 15x9 room grid')
    terrain = np.zeros(TERRAIN_SHAPE, dtype=np.float32)
    for index, x, y, collision, inside, walkable, pit, hazard in raw['cells']:
        row, col = divmod(index, raw['width'])
        terrain[:, row, col] = [inside, walkable, collision / 5, pit, hazard]
    return terrain


def actor_observation(obs):
    """Fixed-size visible entity table. Padding is masked, overflow is an error.

    Excludes seeds, entity IDs, engine velocities, hidden NPCs/HP/AI timers,
    room alive counts, and damage-event counters. No recurrent reward feedback.
    Positions use room-relative units; these tensors are specific to this arena.
    """
    p = obs["players"][0]
    left, top = obs["room"]["top_left"]
    right, bottom = obs["room"]["bottom_right"]
    width, height = right - left, bottom - top
    player = np.asarray([
        (p["pos"][0] - left) / width, (p["pos"][1] - top) / height,
        p["size"] / width, p["hearts"] / 6, p["max_hearts"] / 6,
        p["soul"] / 6, p["bombs"], p["damage"], p["speed"],
        p["shot_speed"], p["fire_delay_max"], p["range"],
    ], dtype=np.float32)
    visible = sorted(obs["entities"], key=lambda e: (e["type"], e["variant"], *e["pos"]))
    if len(visible) > ENTITY_SLOTS:
        raise ValueError(f"Visible entity overflow: {len(visible)} > {ENTITY_SLOTS}")
    entities = np.zeros((ENTITY_SLOTS, 14), dtype=np.float32)
    mask = np.zeros(ENTITY_SLOTS, dtype=np.float32)
    for i, e in enumerate(visible):
        entities[i] = [
            e["type"], e["variant"], (e["pos"][0] - left) / width,
            (e["pos"][1] - top) / height, e["size"] / width,
            e["coll"], e["cdmg"], bool(e.get("enemy")),
            bool(e.get("projectile")), bool(e.get("boss")),
            e.get("boss_hp", 0), e.get("height", 0),
            e["size_multi"][0], e["size_multi"][1],
        ]
        mask[i] = 1
    return {"player": player, "entities": entities.reshape(-1), "mask": mask,
            "terrain": terrain_observation(obs)}


class MonstroGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, port=27015, max_episode_frames=5400, bridge=None,
                 hit_reward=0.05, damage_reward=1.0, clear_reward=1.0):
        super().__init__()
        if max_episode_frames < 1:
            raise ValueError("max_episode_frames must be positive")
        self.bridge = bridge if bridge is not None else IsaacTrainingEnv(port=port)
        self.max_episode_frames = max_episode_frames
        self.hit_reward, self.damage_reward, self.clear_reward = hit_reward, damage_reward, clear_reward
        self.action_space = spaces.MultiDiscrete([9, 5, 2, 2])
        self.observation_space = spaces.Dict({
            "player": spaces.Box(-np.inf, np.inf, (12,), np.float32),
            "entities": spaces.Box(-np.inf, np.inf, (ENTITY_SLOTS * 14,), np.float32),
            "mask": spaces.Box(0, 1, (ENTITY_SLOTS,), np.float32),
            "terrain": spaces.Box(0, 1, TERRAIN_SHAPE, np.float32),
        })
        self.connected = False
        self.finished = True
        self.raw_obs = None
        self.elapsed_frames = 0
        self.transport_failed = False
        self.cleanup_outcome = "not_connected"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not self.connected:
            self.bridge.connect()
            self.connected = True
        arena_seed = (options or {}).get("arena_seed", "9AM0 7PRP")
        try:
            obs, info = self.bridge.reset_monstro(arena_seed)
        except (OSError, BridgeError):
            self.transport_failed = True
            self.finished = True
            raise
        self.transport_failed = False
        self.cleanup_outcome = "connected"
        self.raw_obs = obs
        self.elapsed_frames = 0
        self.finished = False
        return actor_observation(obs), {**info, "arena_seed": arena_seed, "outcome": "running"}

    def step(self, action):
        if self.finished:
            raise RuntimeError("reset() is required before step() or after a terminal transition")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        repeat = min(self.bridge.action_repeat, self.max_episode_frames - self.elapsed_frames)
        previous = self.raw_obs["logic_frames"]
        previous_combat = self.raw_obs['combat']
        try:
            obs, _, _, _, info = self.bridge.step(action, repeat=repeat)
        except (OSError, BridgeError):
            self.transport_failed = True
            self.finished = True
            raise
        advanced = obs["logic_frames"] - previous
        if advanced != repeat:
            raise RuntimeError(f"Expected {repeat} logic frames, received {advanced}")
        self.raw_obs = obs
        self.elapsed_frames += advanced
        dead = obs["players"][0]["dead"]
        won = obs["room"]["clear"] and not dead
        terminated = bool(dead or won)
        truncated = self.elapsed_frames >= self.max_episode_frames and not terminated
        self.finished = terminated or truncated
        outcome = "death" if dead else "win" if won else "time_limit" if truncated else "running"
        combat = obs['combat']
        components = {
            'hurt': -float(combat['player_damage_events'] - previous_combat['player_damage_events']),
            'hit': self.hit_reward * (combat['enemy_damage_events'] - previous_combat['enemy_damage_events']),
            'damage': self.damage_reward * (combat['enemy_damage_fraction'] - previous_combat['enemy_damage_fraction']),
            'clear': self.clear_reward if won else 0.0,
        }
        reward = sum(components.values())
        return actor_observation(obs), reward, terminated, truncated, {
            **info, "outcome": outcome, "elapsed_frames": self.elapsed_frames,
            "logic_frames_advanced": advanced,
            "reward_components": components,
        }

    def close(self):
        if self.connected:
            try:
                if self.transport_failed:
                    # A dead worker cannot reset. Preserve the original step/reset error.
                    self.cleanup_outcome = "broken_connection_closed"
                else:
                    self.cleanup_outcome = "safe_reset_failed"
                    self.bridge.reset_safe()
                    self.cleanup_outcome = "safe_room_and_disconnected"
            finally:
                self.bridge.close()
                self.connected = False
                self.finished = True
