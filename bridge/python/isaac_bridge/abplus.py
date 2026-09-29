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

import numpy as np

from . import abplus_geometry as geometry
from .abplus_obs import ObsDecoder, compare
from .env import Action, BridgeError
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
-- abp-0.2.4: the hit-rate test's invincible player (reset turns it off).
local invincible = AbpSetInvincible({invincible})
-- abp-0.2.6: where the per-miss penalty stops growing (combat-hitrate-miss; 0 = no cap).
local miss_cap = AbpSetMissCap({miss_cap})
return tostring(player:GetCollectibleCount()) .. " " .. tostring(player:GetHearts()) .. " curses=" ..
  tostring(level:GetCurses()) .. " subtype=" .. tostring(monstro.SubType) .. " roster=" .. tostring(roster) ..
  " invincible=" .. tostring(invincible) .. " miss_cap=" .. tostring(miss_cap) .. " reseeded=" .. tostring(reseeded)
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
-- abp-0.2.11 (C41): the episode's stat offsets (all zero: the base stats, nothing re-evaluated).
local stats = AbpSetStats({stats})
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
-- abp-0.2.4: the hit-rate test's invincible player (reset turns it off).
local invincible = AbpSetInvincible({invincible})
-- abp-0.2.6: where the per-miss penalty stops growing (combat-hitrate-miss; 0 = no cap).
local miss_cap = AbpSetMissCap({miss_cap})
return "curses=" .. tostring(level:GetCurses()) .. " clear=" .. tostring(room:IsClear()) .. " roster=" .. tostring(roster) ..
  " stats=" .. stats .. " invincible=" .. tostring(invincible) .. " miss_cap=" .. tostring(miss_cap) ..
  " reseeded=" .. tostring(reseeded)
"""

# Single-enemy aiming arena (user decision 2026-09-27, EXPERIMENTS.md C22): a rock-free normal room
# loses everything but the player; the player stands on one interior cell and one target NPC is
# spawned on another (target_cells), after the episode reseed so the spawn replays. A champion roll
# is spawned again (at most 8 times). Doors are barred and the room is set uncleared, as the arena.
TARGET_LUA = """
local game = Game(); local room = game:GetRoom(); local level = game:GetLevel(); local player = Isaac.GetPlayer(0)
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e.Type ~= EntityType.ENTITY_PLAYER then e:Remove() end
end
for _, curse in ipairs({{1, 2, 4, 8, 16, 32, 64, 128}}) do pcall(function() level:RemoveCurse(curse) end) end
for id = 1, CollectibleType.NUM_COLLECTIBLES - 1 do
  while player:HasCollectible(id) do player:RemoveCollectible(id) end
end
player:AddHearts({player_hp} - player:GetHearts())
player:AddBombs({bombs} - player:GetNumBombs())
player:AddKeys(-player:GetNumKeys())
player:AddCoins(-player:GetNumCoins())
player.Position = room:GetGridPosition({player_cell}); player.Velocity = Vector(0, 0)
pcall(function() player.FireDelay = 0 end)
pcall(function() player:ResetDamageCooldown() end)
local stats = AbpSetStats({stats})   -- abp-0.2.11 (C41)
local reseeded = os.getenv("ABP_RESEED:{rng_seed}")
local target, rolls = nil, 0
repeat
  if target then target:Remove() end
  target = Isaac.Spawn({target_type}, {target_variant}, 0, room:GetGridPosition({target_cell}), Vector(0, 0), nil)
  rolls = rolls + 1
until not (target:ToNPC() and target:ToNPC():IsChampion()) or rolls >= 8
-- Tier 7 (C29): extra NPCs {{type, variant, grid index}}, each re-rolled like the target while a champion.
local extras, extra_champions = {extras}, 0
for _, x in ipairs(extras) do
  local npc, n = nil, 0
  repeat
    if npc then npc:Remove() end
    npc = Isaac.Spawn(x[1], x[2], 0, room:GetGridPosition(x[3]), Vector(0, 0), nil)
    n = n + 1
  until not (npc:ToNPC() and npc:ToNPC():IsChampion()) or n >= 8
  if npc:ToNPC() and npc:ToNPC():IsChampion() then extra_champions = extra_champions + 1 end
end
room:SetClear(false)
for slot = 0, 7 do
  local door = room:GetDoor(slot)
  if door then door:Close(true); door:Bar() end
end
local roster = AbpRosterMark({lineage_mode})
local invincible = AbpSetInvincible({invincible})
local miss_cap = AbpSetMissCap({miss_cap})
return "curses=" .. tostring(level:GetCurses()) .. " clear=" .. tostring(room:IsClear()) .. " roster=" .. tostring(roster) .. " stats=" .. stats ..
  " target=" .. tostring(target.Type) .. "." .. tostring(target.Variant) .. " champion=" ..
  tostring(target:ToNPC() ~= nil and target:ToNPC():IsChampion()) .. " rolls=" .. tostring(rolls) ..
  " extras=" .. tostring(#extras) .. " extra_champions=" .. tostring(extra_champions) ..
  " invincible=" .. tostring(invincible) .. " miss_cap=" .. tostring(miss_cap) .. " reseeded=" .. tostring(reseeded)
"""


STAT_BASE = (1.0, 3.5, 1.0, 10.0, 260.0)   # Isaac: MoveSpeed, Damage, ShotSpeed, MaxFireDelay, reported range (px)


def lua_stats(stats):
    """AbpSetStats' arguments: (speed, damage, shot_speed, tears, range) offsets; None: all zero."""
    return ', '.join(f'{float(v):.6f}' for v in (stats or (0.0,) * 5))


def expected_stats(stats):
    """The player's (speed, damage, shot_speed, fire_delay_max, range) the bridge reports after AbpSetStats(stats) on the
    base Isaac with no items (abp_bridge.lua MC_EVALUATE_CACHE)."""
    speed, damage, shot, tears, reach = (float(v) for v in stats)
    per_second = max(0.5, 30.0 / (STAT_BASE[3] + 1) + tears)
    return (max(0.1, STAT_BASE[0] + speed), max(0.5, STAT_BASE[1] + damage), max(0.6, STAT_BASE[2] + shot),
            float(max(1, int(30.0 / per_second - 1 + 0.5))), STAT_BASE[4] * (6.5 + reach) / 6.5)


def lua_extras(extras):
    """TARGET_LUA's extras table: {{type, variant, grid index}, ...}."""
    return '{' + ', '.join(f'{{{t}, {v}, {cell}}}' for cell, t, v in extras) + '}'


# The duel (bridge abp-0.2.9, EXPERIMENTS.md A6): the target arena's room preparation with the duel NPC as the only
# NPC, bound to the step command's duel_move / duel_shoot by AbpDuelAttach. The player has no bombs; a champion roll of
# the NPC is spawned again (at most 8 times).
DUEL_LUA = """
local game = Game(); local room = game:GetRoom(); local level = game:GetLevel(); local player = Isaac.GetPlayer(0)
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e.Type ~= EntityType.ENTITY_PLAYER then e:Remove() end
end
for _, curse in ipairs({{1, 2, 4, 8, 16, 32, 64, 128}}) do pcall(function() level:RemoveCurse(curse) end) end
for id = 1, CollectibleType.NUM_COLLECTIBLES - 1 do
  while player:HasCollectible(id) do player:RemoveCollectible(id) end
end
player:AddHearts({player_hp} - player:GetHearts())
player:AddBombs(-player:GetNumBombs())
player:AddKeys(-player:GetNumKeys())
player:AddCoins(-player:GetNumCoins())
player.Position = room:GetGridPosition({player_cell}); player.Velocity = Vector(0, 0)
pcall(function() player.FireDelay = 0 end)
pcall(function() player:ResetDamageCooldown() end)
local reseeded = os.getenv("ABP_RESEED:{rng_seed}")
local npc, rolls = nil, 0
repeat
  if npc then npc:Remove() end
  npc = Isaac.Spawn({npc_type}, {npc_variant}, 0, room:GetGridPosition({npc_cell}), Vector(0, 0), nil)
  rolls = rolls + 1
until not (npc:ToNPC() and npc:ToNPC():IsChampion()) or rolls >= 8
local attached = AbpDuelAttach(npc, {npc_hp})
room:SetClear(false)
for slot = 0, 7 do
  local door = room:GetDoor(slot)
  if door then door:Close(true); door:Bar() end
end
local roster = AbpRosterMark({lineage_mode})
local invincible = AbpSetInvincible(false)
local miss_cap = AbpSetMissCap({miss_cap})
return "curses=" .. tostring(level:GetCurses()) .. " clear=" .. tostring(room:IsClear()) .. " roster=" .. tostring(roster) ..
  " npc=" .. tostring(npc.Type) .. "." .. tostring(npc.Variant) .. " champion=" ..
  tostring(npc:ToNPC() ~= nil and npc:ToNPC():IsChampion()) .. " rolls=" .. tostring(rolls) .. " duel=" .. tostring(attached) ..
  " invincible=" .. tostring(invincible) .. " miss_cap=" .. tostring(miss_cap) .. " reseeded=" .. tostring(reseeded)
"""
DUEL_NPC = {'type': 11, 'variant': 1}   # Pacer: abp_bridge.lua DUEL_TYPE
DUEL_HP = 21.0                          # six 3.5-damage tears; the player: six half hearts
DUEL_ACTIVE_FRAMES = 60                 # the NPC's spawn phase (FLAG_APPEAR) takes about 20 frames


def duel_cells(seed, obs, spec):
    """(player cell, NPC cell) of a duel episode: two cells at least spec['min_cells'] apart (Chebyshev), on walkable
    cells from which the first can walk to a firing position at the second when spec['obstacles'] (target_cells_walkable),
    else anywhere inside the wall ring (target_cells); never on one row or column (user decision 2026-09-28: no shot at
    each other before either moves; spec['same_line'] = true allows it); swapped with probability 1/2, so neither side's
    start is drawn differently from the other's. A pure function of the seed and the room; None when no pair works."""
    same_line = bool(spec.get('same_line', False))
    if spec.get('obstacles'):
        cells = target_cells_walkable(seed, obs, int(spec.get('min_cells', 4)), int(spec.get('attempts', 64)),
                                      same_line=same_line)
    else:
        cells = target_cells(seed, int(obs['room']['gw']), int(obs['room']['gh']), int(spec.get('min_cells', 4)),
                             same_line=same_line)
    if cells is None:
        return None
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0xD0E1])
    return (cells[1], cells[0]) if rng.random() < 0.5 else cells


TARGET_SETTLE_FRAMES = 6


TARGET_RADIUS = 13.0   # collision radius of the targets (Horf, Gaper, Clotty: 13) for the reachability check
# Spikes and on/off spikes are walkable, but a mortal player placed on them is hurt before the episode starts
# (C36: seed 1711, room 1037); the player's start cell is never one of them.
HAZARD_GRID_TYPES = (8, 9)


def target_cells_walkable(seed, obs, min_cells, attempts=64, detour_min=0.0, trap_min=0.0, same_line=True):
    """(player cell, target cell) of a target-arena episode in a room with obstacles (tier 4, C26): two
    walkable interior cells at least min_cells apart (Chebyshev) such that the player can walk to a
    position from which a tear reaches the target (abplus_geometry's walking distance < D_FIRE_MAX), a
    pure function of the seed and the room. detour_min > 0 (tier 5, C27) also asks the walking distance to
    exceed the straight one by that many px (obstacles in the way); trap_min > 0 (tier 6, C28) asks heading
    straight for the target to run into a dead end at least that deep (abplus_geometry.trap_depth).
    A player cell with spikes (HAZARD_GRID_TYPES) is drawn again (C36), so only the seeds that drew one change.
    same_line=False (the duel) keeps the second cell off the first one's row and column.
    obs: the room right after the goto. None when no pair works."""
    grid = geometry.walk_grid(obs)
    if grid is None:
        return None
    (x0, y0), walkable, _ = grid
    gw = int(obs['room']['gw'])
    gh = int(obs['room'].get('gh') or len(obs['terrain']['cells']) // gw)
    interior = sorted((c, r) for c, r in walkable if 1 <= c <= gw - 2 and 1 <= r <= gh - 2)
    if not interior:
        return None
    hazards = {int(g[0]) for g in obs.get('grid', ()) if int(g[1]) in HAZARD_GRID_TYPES}
    stops, origin = geometry.tear_stops(obs)
    reach = float(obs['players'][0]['range'])
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x7A28])
    for _ in range(attempts):
        pc, pr = interior[int(rng.integers(len(interior)))]
        if pr * gw + pc in hazards:
            continue
        far = [(c, r) for c, r in interior
               if max(abs(c - pc), abs(r - pr)) >= min_cells and (same_line or (c != pc and r != pr))]
        if not far:
            continue
        tc, tr = far[int(rng.integers(len(far)))]
        target = (x0 + geometry.CELL * tc, y0 + geometry.CELL * tr, TARGET_RADIUS)
        limits = [geometry.target_limits(target, reach, stops, origin)]
        field = geometry.walk_field(grid, [target], limits)
        px, py = x0 + geometry.CELL * pc, y0 + geometry.CELL * pr
        walk = geometry.walk_distance(px, py, grid, field, [target], limits)
        if walk >= geometry.D_FIRE_MAX:
            continue
        if detour_min > 0 and walk - geometry.fire_distance(px, py, [target], limits) < detour_min:
            continue
        if trap_min > 0 and geometry.trap_depth(grid, field, (pc, pr), target) < trap_min:
            continue
        return pr * gw + pc, tr * gw + tc
    return None


def target_extras(seed, obs, extras, player_cell, target_cell):
    """Extra NPCs of a target-arena episode (tier 7, EXPERIMENTS.md C29): [(grid index, type, variant)], a pure
    function of the seed and the room. Their number is drawn from extras['counts'] and each type from
    extras['types'] (uniform); each goes on a walkable interior cell at least extras['min_cells'] (Chebyshev) from
    the player and 2 from the target and every NPC placed before it, from which the player can walk to a firing
    position (as target_cells_walkable). None when one cannot be placed in extras['attempts'] tries."""
    grid = geometry.walk_grid(obs)
    if grid is None:
        return None
    (x0, y0), walkable, _ = grid
    gw = int(obs['room']['gw'])
    gh = int(obs['room'].get('gh') or len(obs['terrain']['cells']) // gw)
    interior = sorted((c, r) for c, r in walkable if 1 <= c <= gw - 2 and 1 <= r <= gh - 2)
    stops, origin = geometry.tear_stops(obs)
    reach = float(obs['players'][0]['range'])
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x7A2A])
    counts, kinds = extras.get('counts', [1, 2]), extras['types']
    count = int(counts[int(rng.integers(len(counts)))])
    pc, pr = player_cell % gw, player_cell // gw
    px, py = x0 + geometry.CELL * pc, y0 + geometry.CELL * pr
    placed, out = [(target_cell % gw, target_cell // gw)], []
    for _ in range(count):
        kind = kinds[int(rng.integers(len(kinds)))]
        for _ in range(int(extras.get('attempts', 64))):
            c, r = interior[int(rng.integers(len(interior)))]
            if max(abs(c - pc), abs(r - pr)) < int(extras.get('min_cells', 3)):
                continue
            if any(max(abs(c - oc), abs(r - orow)) < 2 for oc, orow in placed):
                continue
            npc = (x0 + geometry.CELL * c, y0 + geometry.CELL * r, TARGET_RADIUS)
            limits = [geometry.target_limits(npc, reach, stops, origin)]
            field = geometry.walk_field(grid, [npc], limits)
            if geometry.walk_distance(px, py, grid, field, [npc], limits) >= geometry.D_FIRE_MAX:
                continue
            placed.append((c, r))
            out.append((r * gw + c, int(kind['type']), int(kind.get('variant', 0))))
            break
        else:
            return None
    return out


def target_kind(seed, target):
    """(type, variant) of a target-arena episode: one of target['types'] chosen from the episode seed
    (uniform), or the single target['type'] / ['variant']."""
    kinds = target.get('types')
    if not kinds:
        return int(target['type']), int(target.get('variant', 0))
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x7A27])
    kind = kinds[int(rng.integers(len(kinds)))]
    return int(kind['type']), int(kind.get('variant', 0))


def target_cells(seed, gw, gh, min_cells, same_line=True):
    """(player cell, target cell) grid indices of a target-arena episode: two interior cells (inside
    the wall ring) at least min_cells apart (Chebyshev), a pure function of the episode seed; same_line=False (the
    duel) keeps the second off the first one's row and column."""
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x7A26])
    interior = [(col, row) for row in range(1, gh - 1) for col in range(1, gw - 1)]
    pc, pr = interior[int(rng.integers(len(interior)))]
    far = [(c, r) for c, r in interior
           if max(abs(c - pc), abs(r - pr)) >= min_cells and (same_line or (c != pc and r != pr))]
    tc, tr = far[int(rng.integers(len(far)))]
    return pr * gw + pc, tr * gw + tc


# Monstro subtypes present in the room (the goto room's own boss before the arena cleanup).
ROOM_MONSTROS_LUA = """
local s = {}
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e.Type == EntityType.ENTITY_MONSTRO then s[#s + 1] = tostring(e.SubType) end
end
return table.concat(s, ",")
"""
BRIDGE_VERSION = "abp-0.2.12"
GOTO_SETTLE_FRAMES = 8
MAX_ROOM_ATTEMPTS = 16
MAX_ROOM_RETRIES = 8  # mixture rooms tried per seed before giving up


class RoomUnusable(BridgeError):
    """A mixture room cannot host this episode (content, not a failure): pick the next candidate."""


def room_seed(seed: int, attempt: int) -> int:
    """Global-RNG seed for the goto of an arena seed's attempt-th room."""
    return ((seed ^ 0x51ED2701) + attempt * 0x9E3779B9) & 0xFFFFFFFF


def action_code(move: int, shoot: int, bomb: int = 0, item: int = 0) -> int:
    """One action of the bridge's play command (abp-0.2.12): move 0-8, shoot 0-4, bomb and item 0/1."""
    return int(move) + 9 * int(shoot) + 45 * int(bomb) + 90 * int(item)


def joint_code(joint: int, bomb: int = 0, item: int = 0) -> int:
    """action_code of AbplusTransformerEnv's action [joint, bomb, item] (joint = 5 move + shoot)."""
    move, shoot = divmod(int(joint), 5)
    return action_code(move, shoot, bomb, item)


class AbplusTrainingEnv(IsaacTrainingEnv):
    """IsaacTrainingEnv over abp_bridge.lua; reset rebuilds the simulator's arena for a seed."""

    require_reseed = True
    restart_run = True
    tasks = None          # abplus_tasks.TaskSampler: room-level mixture; None = Monstro arena only
    unusable_rooms = []   # (seed, kind, variant, reason) of rooms skipped for content reasons
    # abp-0.2.3 roster lineage: which death successors join (abp_bridge.lua LINEAGE_RADIUS): 0 none,
    # 1 all NPCs a lineage death leaves (the default, user decision 2026-09-26), 2 only a single one;
    # abp-0.2.5 mode 3: every doors-blocking NPC joins when first seen, spawns included (every enemy
    # earns reward, user decision 2026-09-26).
    lineage_mode = 1
    # abp-0.2.4: the player takes no damage (the hit-rate test, combat-hitrate); set per episode.
    invincible = False
    # abp-0.2.6: a miss adds min(miss_streak, miss_cap) to combat.miss_units (0 = no cap); set per episode.
    miss_cap = 0
    # Single-enemy aiming arena (C22): dict(type, variant, min_cells) turns every normal-room task into
    # that room emptied, the player and one target NPC on interior cells chosen by target_cells;
    # dict(types=[{type, variant}, ...], min_cells) picks the target per episode (target_kind); obstacles=True
    # keeps the room's grid and places both on reachable walkable cells (target_cells_walkable, tier 4);
    # detour_min (px) keeps only starts whose walk is that much longer than the straight line (tier 5);
    # trap_min (px) only starts where heading straight for the target runs into a dead end that deep; arms
    # (tier 6): a list of such specs with weights and their own rooms, one drawn per seed (abplus_tasks);
    # extras (tier 7, on the target or an arm): dict(counts, types, min_cells) adds that many other NPCs
    # (target_extras).
    target = None
    # The duel (bridge abp-0.2.9): dict(npc, hp, min_cells, obstacles, arms=[{name, weight, rooms, obstacles, min_cells},
    # ...]) turns every normal-room task into that room emptied with the player and the duel NPC (duel_cells);
    # duel_action is the NPC's (move, shoot) sent with every step (None: no duel NPC action in the step command).
    duel = None
    duel_action = None
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

    def play(self, codes, repeat=None, repeats=None, stop_clear=True):
        """Batched replay (bridge abp-0.2.12): the actions (action_code values) back to back, each held for `repeat`
        logic frames (or its entry of `repeats`), exactly as the same step commands; one observation at the end. The
        batch stops early once the player is dead or (stop_clear) the room is clear. Returns (obs, played, stop) with
        stop 'done', 'dead' or 'clear'; the observation's credits cover the whole batch."""
        msg = {"cmd": "play", "actions": [int(c) for c in codes], "repeat": int(repeat or self.action_repeat),
               "stop_clear": bool(stop_clear)}
        if repeats is not None:
            msg["repeats"] = [int(r) for r in repeats]
        self._send(msg)
        ack = self._recv()
        if ack.get("type") != "ok" or ack.get("cmd") != "play":
            raise BridgeError(f"play failed: {ack}")
        obs = self._expect_obs()
        return obs, int(ack["played"]), str(ack["stop"])


    def step(self, action, repeat=None):
        """IsaacBridgeEnv.step; with duel_action set (abp-0.2.9) the duel NPC's move and shoot go with it."""
        if self.duel_action is None:
            return super().step(action, repeat)
        msg = {"cmd": "step", "repeat": int(repeat or self.action_repeat), **Action.from_any(action).__dict__,
               "duel_move": int(self.duel_action[0]), "duel_shoot": int(self.duel_action[1])}
        self._send(msg)
        obs = self._expect_obs()
        return obs, 0.0, self.is_terminal(obs), False, dict(self.last_info)

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
        # C41: the player's stat offsets (speed, damage, shot_speed, tears, range); None: the base stats.
        stats, self.pending_stats = getattr(self, "pending_stats", None), None
        self.start_stats = tuple(float(v) for v in stats) if stats is not None else None
        player_hp, boss_hp_fraction = int(round(player_hp)), float(boss_hp_fraction)
        if not (1 <= player_hp <= 6 and 0 < boss_hp_fraction <= 1 and 0 <= bombs <= 99):
            raise ValueError(f"start out of range: {player_hp}, {boss_hp_fraction}, {bombs} bombs")
        self.start_bombs = bombs
        if self.restart_run:
            # A new run recreates the player: goto keeps Entity_Player state across episodes (which eye
            # fires next, the player's RNG), and AB+ has no rewind. The global MT19937 seeds the run.
            # Two more things carried over from the instance's history (A7):
            # - The run starts a frame later (Manager::StartDebugGame only flags it) and then draws its start seed
            #   from the MT (Seeds::SetStartSeed(0) -> RandomU32), after the rest of this frame drew from it by an
            #   amount that depends on the previous room. abp_turbo's ABP_STARTSEED reseeds the MT at that draw.
            # - A sound replays (drawing Random for its variant) only once its stamp, set on its last play, has
            #   passed; the stamps outlived episodes. ABP_SOUND_RESET clears them.
            n = room_seed(int(seed), -1)
            reseeded = self.lua(f"return tostring(os.getenv('ABP_SOUND_RESET')) .. "
                                f"tostring(os.getenv('ABP_RESEED:{n}')) .. tostring(os.getenv('ABP_STARTSEED:{n}'))")
            if self.require_reseed and reseeded != "111":
                raise BridgeError(f"abp_turbo sound reset / reseed / start-seed override unavailable ({reseeded}): "
                                  f"rebuild tools/libabp_turbo.so from analysis/scripts/abplus/abp_turbo.c (A7)")
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
                                           bombs=self.start_bombs, lineage_mode=int(self.lineage_mode),
                                           invincible='true' if self.invincible else 'false',
                                           miss_cap=int(self.miss_cap)))
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
        if self.duel and task.kind == "normal":
            return self._reset_duel(seed, task, command, obs, slot, attempt)
        target = self.target if task.kind == "normal" else None
        # Tier 7 (C29): extra NPCs, from the arm's spec or else the whole target's.
        extras_spec = (target or {}).get("extras")
        if target and target.get("arms"):
            # Tier 6 (C28): the task's arm (abplus_tasks.target_arm, drawn with the room) is the target spec.
            if task.arm < 0:
                raise BridgeError("target arms need the tasks file's TaskSampler (--room-sampling mixture)")
            target = target["arms"][task.arm]
            extras_spec = target.get("extras", extras_spec)
        extras = []
        if target:
            if target.get("obstacles"):
                cells = target_cells_walkable(seed, obs, int(target.get("min_cells", 3)), int(target.get("attempts", 64)),
                                              float(target.get("detour_min", 0.0)), float(target.get("trap_min", 0.0)))
                if cells is None:
                    raise RoomUnusable(f"{command}: no reachable target placement")
                player_cell, target_cell = cells
            else:
                player_cell, target_cell = target_cells(seed, int(obs["room"]["gw"]), int(obs["room"]["gh"]),
                                                        int(target.get("min_cells", 3)))
            target_type, target_variant = target_kind(seed, target)
            if extras_spec:
                extras = target_extras(seed, obs, extras_spec, player_cell, target_cell)
                if extras is None:
                    raise RoomUnusable(f"{command}: no reachable cells for the extra NPCs")
            result = self.lua(TARGET_LUA.format(player_hp=player_hp, rng_seed=seed & 0xFFFFFFFF, bombs=self.start_bombs,
                                                stats=lua_stats(getattr(self, 'start_stats', None)),
                                                player_cell=player_cell, target_cell=target_cell,
                                                target_type=target_type, target_variant=target_variant,
                                                extras=lua_extras(extras), lineage_mode=int(self.lineage_mode),
                                                invincible='true' if self.invincible else 'false',
                                                miss_cap=int(self.miss_cap)))
            expected = f" target={target_type}.{target_variant} champion=false "
            if expected not in str(result) or f" extras={len(extras)} extra_champions=0 " not in str(result):
                raise BridgeError(f"target setup failed ({command}): {result}")
        else:
            result = self.lua(ROOM_LUA.format(player_hp=player_hp, boss_hp_fraction=repr(boss_hp_fraction),
                                              stats=lua_stats(getattr(self, 'start_stats', None)),
                                              rng_seed=seed & 0xFFFFFFFF, bombs=self.start_bombs,
                                              lineage_mode=int(self.lineage_mode),
                                              invincible='true' if self.invincible else 'false',
                                              miss_cap=int(self.miss_cap)))
        if str(result).startswith("curses=0 clear=true"):
            raise RoomUnusable(f"{command} is clear at the start: {result}")
        if (not str(result).startswith("curses=0 clear=false") or
                (self.require_reseed and not str(result).endswith("reseeded=1"))):
            raise BridgeError(f"room setup failed ({command}): {result}")
        # A spawned target is invisible for its first 4 frames (not in the observation): the target
        # arena's episode starts once it shows.
        obs, _, _, _, info = self.step({}, repeat=TARGET_SETTLE_FRAMES if target else 1)
        player = obs["players"][0]
        if target and player["hearts"] < player_hp:
            # C36: a mortal player hurt while the arena settles (the start is unsafe for this seed): next room.
            raise RoomUnusable(f"{command}: the player was hurt while the arena settled ({player['hearts']} of "
                               f"{player_hp} half hearts)")
        if (player["ptype"] != 0 or player["active"] != 0 or player["hearts"] != player_hp
                or player["bombs"] != self.start_bombs):
            raise BridgeError(f"room player initialization failed: {json.dumps(player, sort_keys=True)}")
        if obs["room"]["clear"]:
            raise RoomUnusable(f"{command} is clear after the first frame")
        stats = getattr(self, 'start_stats', None)
        if stats is not None and any(stats):   # C41: the offsets took effect (bridge abp-0.2.11)
            want = expected_stats(stats)
            got = (player["speed"], player["damage"], player["shot_speed"], player["fire_delay_max"], player["range"])
            if any(abs(float(a) - b) > 1e-3 * max(1.0, abs(b)) for a, b in zip(got, want)):
                raise BridgeError(f"player stats {got} != {want} for the offsets {stats} ({command})")
        return obs, {**info, "task": task.kind, "room_variant": task.variant, "entrance": slot,
                     "room_attempts": attempt + 1, "room_setup": result, "reset_kind": "goto", "target_arm": task.arm,
                     "target_extras": extras}

    def _reset_duel(self, seed, task, command, obs, slot, attempt):
        """The duel in the room the goto loaded (obs): the arm's placement (duel_cells), DUEL_LUA, then logic frames
        without input until the NPC's spawn phase is over (obs.duel.active and the NPC visible), so both sides play from
        the first step. Both start at full health, the player without bombs."""
        spec = {k: v for k, v in self.duel.items() if k != 'arms'}
        if self.duel.get('arms'):
            if task.arm < 0:
                raise BridgeError("duel arms need the tasks file's TaskSampler with the arms")
            spec.update(self.duel['arms'][task.arm])
        cells = duel_cells(seed, obs, spec)
        if cells is None:
            raise RoomUnusable(f"{command}: no duel placement")
        player_cell, npc_cell = cells
        npc = spec.get('npc') or DUEL_NPC
        hp = float(spec.get('hp', DUEL_HP))
        self.start_bombs = 0
        result = self.lua(DUEL_LUA.format(player_hp=6, rng_seed=seed & 0xFFFFFFFF, player_cell=player_cell,
                                          npc_cell=npc_cell, npc_type=int(npc['type']), npc_variant=int(npc['variant']),
                                          npc_hp=hp, lineage_mode=int(self.lineage_mode), miss_cap=int(self.miss_cap)))
        if (f" npc={npc['type']}.{npc['variant']} champion=false " not in str(result) or " duel=attached " not in str(result)
                or not str(result).startswith("curses=0 clear=false")
                or (self.require_reseed and not str(result).endswith("reseeded=1"))):
            raise BridgeError(f"duel setup failed ({command}): {result}")
        self.duel_action = (0, 0)
        for waited in range(1, DUEL_ACTIVE_FRAMES + 1):
            obs, _, _, _, info = self.step({}, repeat=1)
            d = obs.get('duel') or {}
            if d.get('active') and any(e['id'] == d['npc'] for e in obs['entities']):
                break
        else:
            raise BridgeError(f"duel NPC not active {DUEL_ACTIVE_FRAMES} frames after the setup ({command})")
        player = obs["players"][0]
        if player["hearts"] != 6 or player["bombs"] != 0 or player["ptype"] != 0 or player["active"] != 0:
            raise BridgeError(f"duel player initialization failed: {json.dumps(player, sort_keys=True)}")
        if obs["room"]["clear"] or d.get('dead') or abs(float(d.get('hp', -1)) - hp) > 1e-6:
            raise BridgeError(f"duel start failed ({command}): clear={obs['room']['clear']} duel={d}")
        return obs, {**info, "task": task.kind, "room_variant": task.variant, "entrance": slot,
                     "room_attempts": attempt + 1, "room_setup": result, "reset_kind": "goto", "target_arm": task.arm,
                     "duel_cells": [int(player_cell), int(npc_cell)], "duel_wait_frames": waited,
                     "duel_arm": spec.get('name')}

    def reset_safe(self, seed=0):
        """Remove enemies so the instance idles harmlessly after the client disconnects."""
        self.lua("for _, e in ipairs(Isaac.GetRoomEntities()) do if e:IsEnemy() then e:Remove() end end")
        obs = self.query_obs()
        return obs, dict(self.last_info)


class AbplusTransformerEnv(TransformerMonstroEnv):
    """TransformerMonstroEnv on AB+. deadline=True adds combat-v1's remaining_time input (the 120 s task, or
    deadline_s: combat-hp's per-group deadline, C39). frames_per_decision: logic frames one action is held for
    (default 2, the policy's 15 decisions per game second; C39 trains with 4).

    reset(options={"arena_seed": seed}) takes the simulator's integer arena seed; an optional
    options["start"] = (player half hearts, Boss HP fraction) applies a training start and
    options["bombs"] the bombs the player starts with (default 1, the game's start).
    combat_state=True adds the combat-v3 room state input (VisibleHistory 'combat'); geometry and
    factored_actions are combat-v5's inputs (VisibleHistory). The action space stays the bridge's
    joint one; a factored policy converts with transformer_obs.factored_to_joint / factored_masks.
    """

    def __init__(self, port: int, max_episode_frames: int = 3600, deadline: bool = True,
                 combat_state: bool = False, geometry: bool = False, factored_actions: bool = False,
                 deadline_s: float = 120.0, frames_per_decision: int = 2, terrain_shape=(9, 15),
                 room_scale: str = 'room', **kwargs):
        super().__init__(port=port, max_episode_frames=max_episode_frames,
                         bridge=AbplusTrainingEnv(port=port), **kwargs)
        if not 1 <= int(frames_per_decision) <= 30:
            raise ValueError('frames_per_decision must be 1..30')
        self.bridge.action_repeat = int(frames_per_decision)
        big = tuple(terrain_shape) != (9, 15) or room_scale != 'room'   # C41: rooms of every shape
        if deadline or combat_state or geometry or factored_actions or big:
            self.history = VisibleHistory(self.history.history, self.history.capacity, deadline=deadline,
                                          combat_state=combat_state, geometry=geometry,
                                          factored_actions=factored_actions, deadline_s=deadline_s,
                                          terrain_shape=terrain_shape, room_scale=room_scale)
            self.observation_space = self.history.space

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        start = options.pop("start", None)
        self.bridge.pending_start = tuple(start) if start is not None else None
        bombs = options.pop("bombs", None)
        self.bridge.pending_bombs = int(bombs) if bombs is not None else None
        stats = options.pop("stats", None)   # C41: the player's stat offsets (abplus_worker.sample_stats)
        self.bridge.pending_stats = tuple(stats) if stats is not None else None
        return super().reset(seed=seed, options=options)

    def play(self, actions):
        """Batched replay (bridge abp-0.2.12, Go-Explore returns) of step actions [joint, bomb, item], with step's
        frame budget (each action min(frames_per_decision, frames left)) and outcome, but one observation at the end.
        The game plays exactly the frames of the same step calls. Returns (encoded observation, actions played, outcome
        as step's info['outcome']); the observations between were never built, so the history window restarts with the
        last one, as after a reset."""
        if self.finished:
            raise RuntimeError("reset() is required before play() or after a terminal transition")
        actions = [tuple(int(v) for v in a) for a in actions]
        codes, repeats, left = [], [], self.max_episode_frames - self.elapsed_frames
        for joint, bomb, item in actions:
            if left <= 0:
                break
            codes.append(joint_code(joint, bomb, item))
            repeats.append(min(self.bridge.action_repeat, left))
            left -= repeats[-1]
        if not codes:
            raise ValueError("play needs an action within the episode's frames")
        previous, room = self.raw_obs["logic_frames"], self.raw_obs["room"]["room_idx"]
        uniform = all(r == self.bridge.action_repeat for r in repeats)
        try:
            obs, played, stop = self.bridge.play(codes, repeats=None if uniform else repeats)
        except (OSError, BridgeError):
            self.transport_failed = True
            self.finished = True
            raise
        advanced = obs["logic_frames"] - previous
        if advanced != sum(repeats[:played]):
            raise RuntimeError(f"Expected {sum(repeats[:played])} logic frames, received {advanced}")
        if obs["room"]["room_idx"] != room:
            self.finished = True
            raise BridgeError(f"room changed during play: {room} -> {obs['room']['room_idx']}")
        self.raw_obs = obs
        self.elapsed_frames += advanced
        self.history.clear()
        if played:
            self.history.set_previous_action(*(int(v) for v in actions[played - 1]))
        dead = obs["players"][0]["dead"]
        won = obs["room"]["clear"] and not dead
        truncated = self.elapsed_frames >= self.max_episode_frames and not (dead or won)
        self.finished = bool(dead or won or truncated)
        outcome = "death" if dead else "win" if won else "time_limit" if truncated else "running"
        return self.encode_observation(obs), played, outcome


# Instance launcher (Linux host with ~/isaac-abplus from the analysis tooling).
# Mesa's software OpenGL (llvmpipe) for the game instead of the NVIDIA driver (EXPERIMENTS.md B7): each instance
# then holds no GPU memory (about 80 MiB of the learner's GPU otherwise), with identical trajectories.
SOFTWARE_GL_ENV = {"__GLX_VENDOR_LIBRARY_NAME": "mesa", "LIBGL_ALWAYS_SOFTWARE": "1"}
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


OOM_SCORE_ADJ = 500   # AB+ instances go before the trainer (adj 0) when memory runs out (C39 t3)


def launch_abplus(name: str, port: int, mode: str = "exact", bridge_lua: Optional[str] = None,
                  nice: int = 19, fixed_time: Optional[int] = None,
                  extra_env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    """Start one isolated AB+ instance (run_instance.sh) serving the bridge on 127.0.0.1:port.

    The bridge script is bridge_lua, else $ABP_BRIDGE_LUA (inherited by worker processes, so a test copy
    can run next to the installed one), else $ABP_HOME/bridge/abp_bridge.lua."""
    tools = ABP_HOME / "tools"
    lua = Path(bridge_lua or os.environ.get("ABP_BRIDGE_LUA") or ABP_HOME / "bridge" / "abp_bridge.lua")
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
    # oom_score_adj 500 (C39 t3 OOM): under memory pressure the kernel kills an AB+ instance (a worker relaunches it) before
    # the trainer; an evaluation raises its own processes to 1000 (abplus_eval). The shell execs, so the pid stays the game's.
    return subprocess.Popen(["sh", "-c", f'echo {OOM_SCORE_ADJ} > /proc/self/oom_score_adj 2>/dev/null; exec "$@"', "sh",
                             "nice", "-n", str(nice), str(tools / "run_instance.sh"), name.lower(), "--luadebug",
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
