"""Afterbirth+ v1.06 (native Linux) backend for the synchronous bridge protocol.

The game side is rl/bridge/abplus/abp_bridge.lua (IsaacRLBridge 0.2.0 ported to AB+), loaded
through the analysis runtime copy's main.lua hook with --luadebug; the engine runs under the
abp_turbo LD_PRELOAD layer (virtual clock and the exactly-equivalent render-lite mode, see
analysis/docs/ABP_LINUX_REVERSE_ENGINEERING.md §7, §10.1). The protocol and observation schema
are the Repentance bridge's, so IsaacTrainingEnv / TransformerMonstroEnv / VisibleHistory are
reused; this module adds the AB+ launcher and the arena reset.

Arena: the Rust curriculum's (rl/sim/src/arena.rs). The same seed gives the same Boss-room
layout (1010/1012/1037/1038, identical rock sets in AB+ and the simulator), entrance and
Monstro position, so the two backends can be compared episode by episode.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from .abplus_obs import ObsDecoder, compare
from .env import BridgeError
from .training import IsaacTrainingEnv
from .transformer_obs import TransformerMonstroEnv, VisibleHistory

# rl/sim/src/layouts.rs (generated from the native special rooms): variant, rock cells, door slots.
LAYOUTS = (
    (1010, (), (True, True, True, True)),
    (1012, (16, 17, 27, 28, 31, 43, 91, 103, 106, 107, 117, 118), (True, True, True, True)),
    (1037, (16, 17, 18, 19, 20, 31, 32, 33, 34, 46, 47, 48, 61, 62, 63, 76, 77, 78, 91, 92, 93, 94,
            106, 107, 108, 109, 110), (False, True, True, True)),
    (1038, (16, 17, 20, 24, 27, 28, 31, 43, 76, 77, 87, 88, 91, 92, 93, 101, 102, 103, 106, 107,
            108, 109, 115, 116, 117, 118), (True, False, True, True)),
)
ENTRANCES = ((80.0, 280.0), (320.0, 160.0), (560.0, 280.0), (320.0, 400.0))
LAYOUT_ROCKS = {variant: sorted(rocks) for variant, rocks, _ in LAYOUTS}
GRID_ROCK, GRID_WALL, GRID_DOOR = 2, 15, 16
WIDTH, HEIGHT, TILE = 15, 9, 40.0
TOP_LEFT, BOTTOM_RIGHT = (60.0, 140.0), (580.0, 420.0)


class _XorShift32:
    """rl/sim/src/rng.rs, bit for bit."""

    def __init__(self, seed: int):
        seed &= 0xFFFFFFFF
        self.state = seed if seed else 0x9E3779B9

    def next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x


def _cell_center(index: int) -> Tuple[float, float]:
    return (index % WIDTH) * TILE + 40.0, (index // WIDTH) * TILE + 120.0


def _clear_circle(blocked: Sequence[int], p: Tuple[float, float], r: float) -> bool:
    if p[0] - r < TOP_LEFT[0] or p[0] + r > BOTTOM_RIGHT[0] or p[1] - r < TOP_LEFT[1] or p[1] + r > BOTTOM_RIGHT[1]:
        return False
    for i in blocked:
        cx, cy = _cell_center(i)
        dx, dy = abs(p[0] - cx) - 20.0, abs(p[1] - cy) - 20.0
        if max(dx, 0.0) ** 2 + max(dy, 0.0) ** 2 < r * r:
            return False
    return True


def sim_arena(seed: int) -> Dict[str, object]:
    """arena.rs `monstro(seed)`: layout, entrance (player spawn) and Monstro spawn."""
    rng = _XorShift32(seed ^ 0xA511E9B3)
    variant, rocks, doors = LAYOUTS[rng.next_u32() % len(LAYOUTS)]
    walls = [i for i in range(WIDTH * HEIGHT)
             if i // WIDTH in (0, HEIGHT - 1) or i % WIDTH in (0, WIDTH - 1)]
    blocked = walls + list(rocks)
    entries = [p for i, p in enumerate(ENTRANCES) if doors[i] and _clear_circle(blocked, p, 12.0)]
    player = entries[rng.next_u32() % len(entries)]
    candidates = [c for c in map(_cell_center, range(WIDTH * HEIGHT))
                  if _clear_circle(blocked, c, 42.0) and ((c[0] - player[0]) ** 2 + (c[1] - player[1]) ** 2) ** 0.5 >= 200.0]
    if not candidates:
        raise ValueError(f"layout {variant} has no safe Boss spawn")
    boss = candidates[rng.next_u32() % len(candidates)]
    return {"seed": seed, "variant": variant, "player": player, "boss": boss,
            "entrance": ENTRANCES.index(player)}


# Episode setup after `goto s.boss.<variant>`: same resource template and placement rules as the
# Repentance arena (rl/bridge/scenarios/monstro_empty.lua), positions from sim_arena.
ARENA_LUA = """
local game = Game(); local room = game:GetRoom(); local player = Isaac.GetPlayer(0)
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e.Type ~= EntityType.ENTITY_PLAYER then e:Remove() end
end
-- Collectible ids must stay below NUM_COLLECTIBLES (553 in AB+): larger ids crash the engine.
for id = 1, CollectibleType.NUM_COLLECTIBLES - 1 do
  while player:HasCollectible(id) do player:RemoveCollectible(id) end
end
player:AddHearts({player_hp} - player:GetHearts())
player:AddBombs({bombs} - player:GetNumBombs())
player:AddKeys(-player:GetNumKeys())
player:AddCoins(-player:GetNumCoins())
player.Position = Vector({px}, {py}); player.Velocity = Vector(0, 0)
-- Simulator arena: no curses (the restarted run may roll e.g. Darkness/Unknown/Maze) and plain rocks
-- (room generation turns some layout rocks into tinted/bomb/alt/super-special ones).
local level = game:GetLevel()
for _, curse in ipairs({{1, 2, 4, 8, 16, 32, 64, 128}}) do pcall(function() level:RemoveCurse(curse) end) end
local special = {{[4] = true, [5] = true, [6] = true, [22] = true}}
for i = 0, room:GetGridSize() - 1 do
  local g = room:GetGridEntity(i)
  if g ~= nil and special[g:GetType()] then g:SetType(GridEntityType.GRID_ROCK) end
end
-- Simulator start: shot ready, no invulnerability frames.
pcall(function() player.FireDelay = 0 end)
pcall(function() player:ResetDamageCooldown() end)
-- Monstro's AI and the render path draw from the global MT19937, which the engine seeds only at
-- startup. abp_turbo's getenv hook reseeds it (init_genrand) so every episode of a seed replays.
local reseeded = os.getenv("ABP_RESEED:{rng_seed}")
local monstro = Isaac.Spawn(EntityType.ENTITY_MONSTRO, 0, 0, Vector({bx}, {by}), Vector(0, 0), nil)
-- Training start randomisation (OpenAI Five's Roshan HP): a weakened Boss; 1.0 leaves it untouched.
if {boss_hp_fraction} < 1 then monstro.HitPoints = monstro.MaxHitPoints * {boss_hp_fraction} end
room:SetClear(false)
for slot = 0, 7 do
  local door = room:GetDoor(slot)
  if door then door:Close(true); door:Bar() end
end
-- abp-0.2.3: the doors-blocking NPCs now are the episode's roster (combat-v5 lineage).
local roster = AbpRosterMark({lineage_mode})
return tostring(player:GetCollectibleCount()) .. " " .. tostring(player:GetHearts()) .. " curses=" ..
  tostring(level:GetCurses()) .. " subtype=" .. tostring(monstro.SubType) .. " roster=" .. tostring(roster) ..
  " reseeded=" .. tostring(reseeded)
"""

# Mixture rooms (normal and boss rooms with their own enemies): the player/level template of the
# arena, Boss HP start randomisation, then the episode reseed. Doors are barred like the arena's.
ROOM_LUA = """
local game = Game(); local room = game:GetRoom(); local level = game:GetLevel(); local player = Isaac.GetPlayer(0)
for _, curse in ipairs({{1, 2, 4, 8, 16, 32, 64, 128}}) do pcall(function() level:RemoveCurse(curse) end) end
for id = 1, CollectibleType.NUM_COLLECTIBLES - 1 do
  while player:HasCollectible(id) do player:RemoveCollectible(id) end
end
player:AddHearts({player_hp} - player:GetHearts())
player:AddBombs({bombs} - player:GetNumBombs())
player:AddKeys(-player:GetNumKeys())
player:AddCoins(-player:GetNumCoins())
player.Velocity = Vector(0, 0)
pcall(function() player.FireDelay = 0 end)
pcall(function() player:ResetDamageCooldown() end)
if {boss_hp_fraction} < 1 then
  for _, e in ipairs(Isaac.GetRoomEntities()) do
    if e:IsBoss() and e.MaxHitPoints > 0 then e.HitPoints = e.MaxHitPoints * {boss_hp_fraction} end
  end
end
for slot = 0, 7 do
  local door = room:GetDoor(slot)
  if door then door:Close(true); door:Bar() end
end
local reseeded = os.getenv("ABP_RESEED:{rng_seed}")
-- abp-0.2.3: the doors-blocking NPCs now are the episode's roster (combat-v5 lineage).
local roster = AbpRosterMark({lineage_mode})
return "curses=" .. tostring(level:GetCurses()) .. " clear=" .. tostring(room:IsClear()) .. " roster=" .. tostring(roster) ..
  " reseeded=" .. tostring(reseeded)
"""

# Monstro subtypes present in the room (the goto room's own boss before the arena cleanup).
ROOM_MONSTROS_LUA = """
local s = {}
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e.Type == EntityType.ENTITY_MONSTRO then s[#s + 1] = tostring(e.SubType) end
end
return table.concat(s, ",")
"""
BRIDGE_VERSION = "abp-0.2.3"
GOTO_SETTLE_FRAMES = 8
MAX_ROOM_ATTEMPTS = 16
MAX_ROOM_RETRIES = 8  # mixture rooms tried per seed before giving up


class RoomUnusable(BridgeError):
    """A mixture room cannot host this episode (content, not a failure): pick the next candidate."""


def room_seed(seed: int, attempt: int) -> int:
    """Global-RNG seed for the goto of an arena seed's attempt-th room."""
    return ((seed ^ 0x51ED2701) + attempt * 0x9E3779B9) & 0xFFFFFFFF


class AbplusTrainingEnv(IsaacTrainingEnv):
    """IsaacTrainingEnv over abp_bridge.lua; reset rebuilds the simulator's arena for a seed."""

    require_reseed = True
    restart_run = True
    tasks = None          # abplus_tasks.TaskSampler: room-level mixture; None = Monstro arena only
    unusable_rooms = []   # (seed, kind, variant, reason) of rooms skipped for content reasons
    # abp-0.2.3 roster lineage: which death successors join (abp_bridge.lua LINEAGE_RADIUS): 0 none,
    # 1 all NPCs a lineage death leaves (the default, user decision 2026-09-26), 2 only a single one.
    lineage_mode = 1
    binary_obs = False    # format 2 (abp-0.2.1): binary observations decoded by abplus_obs.ObsDecoder
    validate_obs = False  # with binary_obs: receive the JSON observation too and compare every frame
    obs_format = 1

    def connect(self):
        hello = super().connect()
        if hello.get("version") != BRIDGE_VERSION:
            # The binary layout and combat-v2's blocking_hp/blocking_points need this exact bridge.
            raise BridgeError(f"abp_bridge.lua is {hello.get('version')}, this client needs {BRIDGE_VERSION}")
        self.obs_format = 1
        self.decoder = ObsDecoder()
        self.validation = {'frames': 0, 'mismatches': 0, 'first': None}
        self.last_pair = None
        self.unusable_rooms = []
        if self.binary_obs:
            self._send({"cmd": "format", "version": 2, "validate": bool(self.validate_obs)})
            reply = self._recv()
            if reply.get("type") != "ok" or reply.get("version") != 2:
                raise BridgeError(f"binary observations unavailable (bridge {hello.get('version')}): {reply}")
            self.obs_format = 2
        return hello

    def _read_line(self):
        while b"\n" not in self._buf:
            chunk = self._sock.recv(1 << 16)
            if not chunk:
                raise BridgeError("connection closed by game")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    def _read_exact(self, n):
        while len(self._buf) < n:
            chunk = self._sock.recv(max(1 << 16, n - len(self._buf)))
            if not chunk:
                raise BridgeError("connection closed by game")
            self._buf += chunk
        data, self._buf = self._buf[:n], self._buf[n:]
        return data

    def _recv(self):
        if self.obs_format != 2:
            return super()._recv()
        if not self._sock:
            raise BridgeError("not connected")
        line = self._read_line()
        if line.startswith(b"B "):
            _, size, event, seq = line.split(b" ")
            obs = self.decoder.decode(self._read_exact(int(size)))
            return {"type": "obs", "event": event.decode(), "seq": int(seq), "obs": obs}
        msg = json.loads(line.decode("utf-8"))
        if msg.get("type") == "error":
            raise BridgeError(msg.get("msg", "error"))
        if self.validate_obs and msg.get("type") == "obs":
            binary = self._recv()
            if binary.get("seq") != msg.get("seq"):
                raise BridgeError(f"validation lost pairing: {msg.get('seq')} vs {binary.get('seq')}")
            diff = compare(msg["obs"], binary["obs"])
            self.validation["frames"] += 1
            if diff is not None:
                self.validation["mismatches"] += 1
                if self.validation["first"] is None:
                    self.validation["first"] = [str(x)[:200] for x in diff]
            self.last_pair = (msg["obs"], binary["obs"])
        return msg

    def lua(self, code: str) -> Optional[str]:
        self._send({"cmd": "lua", "code": code})
        msg = self._recv()
        if msg.get("type") != "ok":
            raise BridgeError(f"lua failed: {msg}")
        return msg.get("result")

    def reset_monstro(self, seed=0, player_hp=6, boss_hp_fraction=1.0):
        """Arena of simulator seed `seed`. player_hp (half hearts 1..6) and boss_hp_fraction (0, 1]
        are the training start randomisation of gpu_env.sample_start; evaluation uses 6 / 1.0."""
        start = getattr(self, "pending_start", None)
        if start is not None:
            player_hp, boss_hp_fraction = start
            self.pending_start = None
        # Bombs the player starts with (training start randomisation); the game's own start is 1.
        pending, self.pending_bombs = getattr(self, "pending_bombs", None), None
        bombs = 1 if pending is None else int(pending)
        player_hp, boss_hp_fraction = int(round(player_hp)), float(boss_hp_fraction)
        if not (1 <= player_hp <= 6 and 0 < boss_hp_fraction <= 1 and 0 <= bombs <= 99):
            raise ValueError(f"start out of range: {player_hp}, {boss_hp_fraction}, {bombs} bombs")
        self.start_bombs = bombs
        if self.restart_run:
            # A new run recreates the player: goto keeps Entity_Player state across episodes (which eye
            # fires next, the player's RNG), and AB+ has no rewind. The global MT19937 seeds the run.
            reseeded = self.lua(f"return os.getenv('ABP_RESEED:{room_seed(int(seed), -1)}')")
            if self.require_reseed and reseeded != "1":
                raise BridgeError(f"global RNG reseed unavailable (abp_turbo getenv hook): {reseeded}")
            self.reset(phases=[["restart 0"]], settle=2)
        task = self.tasks.choose(seed) if self.tasks is not None else None
        for retry in range(1, MAX_ROOM_RETRIES + 1):
            if task is None or task.kind == "arena":
                return self._reset_arena(int(seed), player_hp, boss_hp_fraction)
            try:
                obs, info = self._reset_room(int(seed), task, player_hp, boss_hp_fraction)
                return obs, {**info, "room_retries": retry - 1}
            except RoomUnusable as exc:
                self.unusable_rooms.append((int(seed), task.kind, task.variant, str(exc)))
                task = self.tasks.choose(seed, retry)
        raise BridgeError(f"seed {seed}: no usable room in {MAX_ROOM_RETRIES} candidates")

    def _reset_arena(self, seed, player_hp, boss_hp_fraction):
        arena = sim_arena(seed)
        for attempt in range(MAX_ROOM_ATTEMPTS):
            # goto enters through the door opposite Level.LeaveDoor and creates only that door: leaving
            # by slot (e+2)%4 gives the simulator's single entrance door e (LEFT0/UP0/RIGHT0/DOWN0).
            # The new room descriptor's seeds come from the global MT19937, so reseeding first makes
            # the room (decorations, boss champion roll) a function of the arena seed.
            reseeded = self.lua(f"Game():GetLevel().LeaveDoor = {(arena['entrance'] + 2) % 4}; "
                                f"return os.getenv('ABP_RESEED:{room_seed(int(seed), attempt)}')")
            if self.require_reseed and reseeded != "1":
                raise BridgeError(f"global RNG reseed unavailable (abp_turbo getenv hook): {reseeded}")
            # The room spawns its own boss 3-4 frames after the goto; clean up only after that, or the
            # late spawn replaces the arena's Monstro.
            obs, info = self.reset(phases=[[f"goto s.boss.{arena['variant']}"]], settle=GOTO_SETTLE_FRAMES)
            if obs["room"].get("variant") != arena["variant"]:
                raise BridgeError(f"goto loaded room variant {obs['room'].get('variant')}, expected {arena['variant']}")
            if [d["slot"] for d in obs["doors"]] != [arena["entrance"]]:
                raise BridgeError(f"goto created doors {obs['doors']}, expected only slot {arena['entrance']}")
            # Entity_NPC::load_entity_config rolls boss champions (subtype = boss color, p = 0.3 with the
            # unlocks of this save) from RNG(room descriptor seed, 5): every boss of a room gets the same
            # roll. A room whose own Monstro is plain gives a plain arena Monstro, as in the simulator.
            if self.lua(ROOM_MONSTROS_LUA) == "0":
                break
        else:
            raise BridgeError(f"no champion-free {arena['variant']} room in {MAX_ROOM_ATTEMPTS} attempts")
        (px, py), (bx, by) = arena["player"], arena["boss"]
        result = self.lua(ARENA_LUA.format(px=px, py=py, bx=bx, by=by, rng_seed=int(seed) & 0xFFFFFFFF,
                                           player_hp=player_hp, boss_hp_fraction=repr(boss_hp_fraction),
                                           bombs=self.start_bombs, lineage_mode=int(self.lineage_mode)))
        if " curses=0 subtype=0 " not in str(result) or (self.require_reseed and not str(result).endswith("reseeded=1")):
            raise BridgeError(f"arena setup failed: {result}")
        for _ in range(90):
            obs, _, _, _, info = self.step({}, repeat=1)
            enemies = self.visible_enemies(obs)
            if len(enemies) == 1 and (enemies[0]["type"], enemies[0]["subtype"]) == (20, 0):
                if [round(v) for v in enemies[0]["pos"]] != [round(bx), round(by)]:
                    raise BridgeError(f"arena Monstro at {enemies[0]['pos']}, expected {(bx, by)}")
                player = obs["players"][0]
                if (player["ptype"] != 0 or player["active"] != 0 or player["hearts"] != player_hp
                        or player["bombs"] != self.start_bombs):
                    raise BridgeError(f"AB+ arena player initialization failed: {json.dumps(player, sort_keys=True)}")
                if abs(enemies[0].get("boss_hp", 0.0) - boss_hp_fraction) > 1e-3:
                    raise BridgeError(f"arena Monstro HP {enemies[0].get('boss_hp')} != {boss_hp_fraction}")
                if obs["room"]["clear"]:
                    raise BridgeError("AB+ arena room is already clear")
                rocks = sorted((g[0], g[1]) for g in obs["grid"] if g[1] not in (GRID_WALL, GRID_DOOR))
                if rocks != [(i, GRID_ROCK) for i in LAYOUT_ROCKS[arena["variant"]]]:
                    raise BridgeError(f"arena grid {rocks} differs from simulator layout {arena['variant']}")
                return obs, {**info, "arena": arena, "arena_setup": result, "reset_kind": "goto",
                             "room_attempts": attempt + 1, "task": "arena", "room_variant": arena["variant"]}
        seen = [(e["type"], e["variant"], e["subtype"], e["pos"], e.get("anim")) for e in obs["entities"]]
        raise BridgeError(f"arena Monstro did not become the only visible enemy within 90 frames: {seen}")

    def _reset_room(self, seed, task, player_hp, boss_hp_fraction):
        """A room of the mixture (abplus_tasks) with the enemies its layout spawns.

        Same determinism as the arena: the run was restarted from a reseeded MT19937, the room
        (layout spawns, champion rolls, decorations) comes from a reseed before the goto, and the
        episode's RNG from a reseed after the setup. The entrance is the task's preferred door
        slot, or the next slot the room accepts.
        """
        command = f"goto d.{task.variant}" if task.kind == "normal" else f"goto s.boss.{task.variant}"
        for attempt in range(4):
            slot = (task.entrance + attempt) % 4
            reseeded = self.lua(f"Game():GetLevel().LeaveDoor = {(slot + 2) % 4}; "
                                f"return os.getenv('ABP_RESEED:{room_seed(seed, attempt)}')")
            if self.require_reseed and reseeded != "1":
                raise BridgeError(f"global RNG reseed unavailable (abp_turbo getenv hook): {reseeded}")
            obs, info = self.reset(phases=[[command]], settle=GOTO_SETTLE_FRAMES)
            if obs["room"].get("variant") != task.variant:
                raise BridgeError(f"{command} loaded room variant {obs['room'].get('variant')}")
            if [d["slot"] for d in obs["doors"]] == [slot]:
                break
        else:
            raise RoomUnusable(f"{command}: no entrance slot accepted (doors {obs['doors']})")
        result = self.lua(ROOM_LUA.format(player_hp=player_hp, boss_hp_fraction=repr(boss_hp_fraction),
                                          rng_seed=seed & 0xFFFFFFFF, bombs=self.start_bombs,
                                          lineage_mode=int(self.lineage_mode)))
        if str(result).startswith("curses=0 clear=true"):
            raise RoomUnusable(f"{command} is clear at the start: {result}")
        if (not str(result).startswith("curses=0 clear=false") or
                (self.require_reseed and not str(result).endswith("reseeded=1"))):
            raise BridgeError(f"room setup failed ({command}): {result}")
        obs, _, _, _, info = self.step({}, repeat=1)
        player = obs["players"][0]
        if (player["ptype"] != 0 or player["active"] != 0 or player["hearts"] != player_hp
                or player["bombs"] != self.start_bombs):
            raise BridgeError(f"room player initialization failed: {json.dumps(player, sort_keys=True)}")
        if obs["room"]["clear"]:
            raise RoomUnusable(f"{command} is clear after the first frame")
        return obs, {**info, "task": task.kind, "room_variant": task.variant, "entrance": slot,
                     "room_attempts": attempt + 1, "room_setup": result, "reset_kind": "goto"}

    def reset_safe(self, seed=0):
        """Remove enemies so the instance idles harmlessly after the client disconnects."""
        self.lua("for _, e in ipairs(Isaac.GetRoomEntities()) do if e:IsEnemy() then e:Remove() end end")
        obs = self.query_obs()
        return obs, dict(self.last_info)


class AbplusTransformerEnv(TransformerMonstroEnv):
    """TransformerMonstroEnv on AB+. deadline=True adds combat-v1's remaining_time input (120 s task).

    reset(options={"arena_seed": seed}) takes the simulator's integer arena seed; an optional
    options["start"] = (player half hearts, Boss HP fraction) applies a training start and
    options["bombs"] the bombs the player starts with (default 1, the game's start).
    combat_state=True adds the combat-v3 room state input (VisibleHistory 'combat'); geometry and
    factored_actions are combat-v5's inputs (VisibleHistory). The action space stays the bridge's
    joint one; a factored policy converts with transformer_obs.factored_to_joint / factored_masks.
    """

    def __init__(self, port: int, max_episode_frames: int = 3600, deadline: bool = True,
                 combat_state: bool = False, geometry: bool = False, factored_actions: bool = False, **kwargs):
        super().__init__(port=port, max_episode_frames=max_episode_frames,
                         bridge=AbplusTrainingEnv(port=port), **kwargs)
        if deadline or combat_state or geometry or factored_actions:
            self.history = VisibleHistory(self.history.history, self.history.capacity, deadline=deadline,
                                          combat_state=combat_state, geometry=geometry,
                                          factored_actions=factored_actions)
            self.observation_space = self.history.space

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        start = options.pop("start", None)
        self.bridge.pending_start = tuple(start) if start is not None else None
        bombs = options.pop("bombs", None)
        self.bridge.pending_bombs = int(bombs) if bombs is not None else None
        return super().reset(seed=seed, options=options)


# Instance launcher (Linux host with ~/isaac-abplus from the analysis tooling).
ABP_HOME = Path(os.environ.get("ABP_HOME", Path.home() / "isaac-abplus"))
MODES = {
    # Exactly equivalent to real rendering (analysis doc §10.1): render path runs, pixels do not.
    "exact": {"ABP_RENDER_EVERY": "1", "ABP_NULLGL": "1", "ABP_NULLGL_LIST": "{tools}/null_render_e.txt",
              "ABP_X11CACHE": "1", "ABP_STUB_LIST": "{tools}/stub_render_f.txt"},
    # Render path skipped: faster, statistically but not frame-by-frame equivalent.
    "skip": {"ABP_RENDER_EVERY": "0", "ABP_NULLGL": "1", "ABP_NULLGL_LIST": "{tools}/null_render_e.txt",
             "ABP_X11CACHE": "1"},
    # Real rendering at turbo speed (visual inspection, reference traces).
    "render": {"ABP_RENDER_EVERY": "1", "__GL_SYNC_TO_VBLANK": "0"},
}


def launch_abplus(name: str, port: int, mode: str = "exact", bridge_lua: Optional[str] = None,
                  nice: int = 19, fixed_time: Optional[int] = None,
                  extra_env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    """Start one isolated AB+ instance (run_instance.sh) serving the bridge on 127.0.0.1:port."""
    tools = ABP_HOME / "tools"
    lua = Path(bridge_lua) if bridge_lua else ABP_HOME / "bridge" / "abp_bridge.lua"
    if not lua.is_file():
        raise FileNotFoundError(lua)
    env = dict(os.environ)
    env.update({"ABP_PRELOAD": str(tools / "libabp_turbo.so"), "ABP_TURBO": "1", "ABP_LUA": str(lua),
                "ISAAC_RL_PORT": str(port), "ISAAC_RL_PRIVILEGED": "0", "ISAAC_RL_ENGINE_VEL": "0"})
    env.update({k: v.format(tools=tools) for k, v in MODES[mode].items()})
    if fixed_time is not None:
        env["ABP_FIXED_TIME"] = str(fixed_time)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    return subprocess.Popen(["nice", "-n", str(nice), str(tools / "run_instance.sh"), name.lower(), "--luadebug",
                             "--set-stage=1", "--set-stage-type=0"], env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def stop_abplus(proc: subprocess.Popen, name: str) -> None:
    """Terminate the instance started by launch_abplus (the script execs the game)."""
    try:
        os.killpg(proc.pid, 15)
    except OSError:
        pass
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if open(f"/proc/{pid}/comm").read().strip() == "isaac.x64" and \
               f"instances/{name.lower()}/data".encode() in open(f"/proc/{pid}/environ", "rb").read():
                os.kill(int(pid), 15)
        except OSError:
            pass
