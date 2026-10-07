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
    for index, x, y, collision, inside, walkable, pit, hazard, *extra in raw['cells']:
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
        # Goal-conditioned line (rl/docs/GOAL_CONDITIONED_DESIGN.md): the option being played. 'combat' is every earlier
        # run's episode (its room's clear wins; a room change is an error unless combat_multi_room, below). 'goto': the clear does not end it; it ends
        # when the bridge reports the goal reached ('goal'), a room change ('room', the caller decides what that door
        # meant), a death or the option's frames. The bridge (abp-0.2.13) ends a step early at a reached goal or a room
        # change, so such a step may have fewer frames.
        self.option = "combat"
        # Goal line (user decision 2026-09-30, EXPERIMENTS.md A13): the player may leave the room during a COMBAT option
        # (a door a bomb blew open: on a real floor, or out of a goto room into the floor behind it) and the option goes
        # on; the history restarts at each room change (on_room_change). It is won only when its own room (target_room)
        # is clear: there by the room's clear, as before; from elsewhere by that room's descriptor (the engine clears a
        # room left during its clear countdown, A13). Another room's clear is not a win. Off: a room change during
        # COMBAT is an error (every earlier run).
        self.combat_multi_room = False
        self.target_room = None

    def begin_option(self, option, max_frames):
        """Start the next option in place, on the current observation (no reset): its kind and frame budget."""
        if option not in ("combat", "goto"):
            raise ValueError(f"unknown option {option!r}")
        self.option = option
        self.max_episode_frames = int(max_frames)
        self.elapsed_frames = 0
        self.finished = False
        self.target_room = self.raw_obs["room"]["room_idx"] if option == "combat" and self.raw_obs is not None else None

    def on_room_change(self):
        """A COMBAT option went on into another room (combat_multi_room), before its first observation is encoded."""

    def target_room_clear(self):
        """The COMBAT option's room is clear by its room descriptor (asked only while the player is elsewhere)."""
        return self.bridge.lua(f"return tostring(Game():GetLevel():GetRoomByIdx({int(self.target_room)}).Clear)") == "true"

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
        self.option = "combat"
        self.target_room = obs["room"]["room_idx"]
        return self.encode_observation(obs), {**info, "arena_seed": arena_seed, "outcome": "running"}

    def encode_observation(self, obs):
        return actor_observation(obs)

    def decode_action(self, action):
        return action

    def step(self, action):
        if self.finished:
            raise RuntimeError("reset() is required before step() or after a terminal transition")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        repeat = min(self.bridge.action_repeat, self.max_episode_frames - self.elapsed_frames)
        previous = self.raw_obs["logic_frames"]
        previous_room = self.raw_obs["room"]["room_idx"]
        previous_combat = self.raw_obs['combat']
        try:
            obs, _, _, _, info = self.bridge.step(self.decode_action(action), repeat=repeat)
        except (OSError, BridgeError):
            self.transport_failed = True
            self.finished = True
            raise
        advanced = obs["logic_frames"] - previous
        nav = obs.get("nav") or {}
        cut = bool(nav.get("goal_hit") or nav.get("room_changed")   # abp-0.2.13: the bridge ended the step early
                   or (getattr(self.bridge, "stop_clear", False) and obs["room"]["clear"]))
        if advanced != repeat and not (cut and 0 < advanced <= repeat):
            raise RuntimeError(f"Expected {repeat} logic frames, received {advanced}")
        changed = obs["room"]["room_idx"] != previous_room
        if changed and self.option == "combat" and not self.combat_multi_room:
            self.finished = True
            raise BridgeError(f"Monstro arena room changed: {previous_room} -> {obs['room']['room_idx']}")
        self.raw_obs = obs
        self.elapsed_frames += advanced
        dead = obs["players"][0]["dead"]
        if self.option == "goto":
            reached = bool(nav.get("goal_hit")) and not dead and not changed
            terminated = bool(dead or reached or changed)
            truncated = self.elapsed_frames >= self.max_episode_frames and not terminated
            self.finished = terminated or truncated
            won = False
            outcome = ("death" if dead else "room" if changed else "goal" if reached else "time_limit" if truncated
                       else "running")
        else:
            if changed:
                self.on_room_change()
            if obs["room"]["room_idx"] == self.target_room:
                won = obs["room"]["clear"] and not dead   # unchanged for every episode that never leaves its room
            else:
                won = not dead and self.target_room_clear()
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
        return self.encode_observation(obs), reward, terminated, truncated, {
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
