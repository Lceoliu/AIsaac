"""Room layout library (STB1 files) and the engine's room queries, translated from Afterbirth+ v1.06.

RoomConfig::Load (stages.xml)                 0x4551D0
RoomConfig::LoadStageBinary / read_room        0x452470 / 0x451E70
RoomConfig::GetRooms / GetRandomRoom           0x453100 / 0x453260
RoomConfig::ResetRoomWeights / GetRoom         0x4529A0 / 0x452B00
RoomConfig::GetStageID / GetStageAndTypeFromID 0x452C40 / 0x452C80

Weights are float32 as in the engine (initial weight at Room+0x1C, current weight at Room+0x20).
"""
from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .archive import ArchiveSet
from .rng import RNG

F32 = np.float32
EXACT_DOORS_FACTOR = F32(9.989999771118164)   # 0x9A5AD8
WEIGHT_DECAY = F32(0.10000000149011612)       # 0x9502A0
WEIGHT_FLOOR = F32(1.0000000116860974e-07)    # 0x9A5AD4

# Room flags set by LoadStageBinary (Room+0x40); their consumers are not translated yet.
FLAG_BASE_LAYOUT_ON_ALT_FLOOR = 8   # alt-floor file rooms with low variants (inherited layouts)
FLAG_SPECIAL_603_716 = 4            # variants 603 and 716 on stages 1-3

SHAPE_ANY = 13


@dataclass(slots=True)
class SpawnEntry:
    type: int
    variant: int
    subtype: int
    weight: float


@dataclass(slots=True)
class Spawn:
    x: int
    y: int
    entries: list[SpawnEntry]
    total_weight: float


@dataclass(slots=True, eq=False)
class Room:
    stage: int
    type: int
    variant: int
    subtype: int
    name: str
    difficulty: int
    initial_weight: np.float32
    weight: np.float32
    doors: int            # DoorSlot bit mask (bit 0 LEFT0 ... bit 7 DOWN1), from read_room
    width: int
    height: int
    shape: int
    spawns: list[Spawn] = field(repr=False)
    flags: int = 0
    door_list: list = field(default_factory=list, repr=False)   # (x, y, slot bit) of the STB doors that exist


def door_bit(x: int, y: int, shape: int) -> int:
    """read_room: map a door's wall coordinate to its DoorSlot bit (0 when it matches none)."""
    if x in (12, -1):
        if y == 3:
            return 0x01                     # LEFT0
        if y == 10:
            return 0x10                     # LEFT1
    if x == 6:
        if y == -1:
            return 0x02                     # UP0
        if y == 14:
            return 0x02 if shape == 11 else 0x08
        if y == 6:
            return 0x02 if shape == 9 else 0
        if y == 7:
            return 0x08                     # DOWN0
        return 0
    if x == 19:
        if y == -1:
            return 0x20                     # UP1
        if y == 14 and shape == 12:
            return 0x20
        if y == 6 and shape == 10:
            return 0x20
        if y in (14, 7):
            return 0x80                     # DOWN1
        return 0
    if x in (26, 13):
        if y == 3:
            return 0x04                     # RIGHT0
        if y == 10:
            return 0x40                     # RIGHT1
    return 0


def _post_flags(stage: int, variant: int) -> int:
    """LoadStageBinary: per-stage flags applied after read_room."""
    flags = 0
    if stage == 6:
        if variant < 0x39B:
            flags |= 8
    elif stage < 7:
        if stage == 3 and variant < 0x40B:
            flags |= 8
            if variant in (0x2CC, 0x25B):
                flags |= 4
        elif 1 <= stage <= 3 and variant in (0x2CC, 0x25B):
            flags |= 4
    elif stage == 9:
        if variant != 0x29 and variant < 899:
            flags |= 8
    elif stage == 12:
        if variant < 0x33D:
            flags |= 8
    return flags


def parse_stb(data: bytes, stage: int) -> list[Room]:
    if data[:4] != b'STB1':
        raise ValueError('not an STB1 room file')
    (count,) = struct.unpack_from('<I', data, 4)
    pos = 8
    rooms = []
    for _ in range(count):
        rtype, variant, subtype, difficulty, name_len = struct.unpack_from('<IIIBH', data, pos)
        pos += 15
        name = data[pos:pos + name_len].decode('latin-1')
        pos += name_len
        weight, width, height, shape, door_count, spawn_count = struct.unpack_from('<fBBBBH', data, pos)
        pos += 10
        doors = 0
        door_list = []
        for _ in range(door_count):
            x, y, exists = struct.unpack_from('<hhB', data, pos)
            pos += 5
            if exists:
                doors |= door_bit(x, y, shape)
                door_list.append((x, y, door_bit(x, y, shape)))
        spawns = []
        for _ in range(spawn_count):
            x, y, n = struct.unpack_from('<hhB', data, pos)
            pos += 5
            entries, total = [], F32(0)
            for _ in range(n):
                etype, evar, esub, ew = struct.unpack_from('<HHHf', data, pos)
                pos += 10
                entries.append(SpawnEntry(etype, evar, esub, ew))
                total = F32(total + F32(ew))
            spawns.append(Spawn(x, y, entries, float(total)))
        w = F32(weight)
        rooms.append(Room(stage, rtype, variant, subtype, name, difficulty, w, w, doors, width, height,
                          shape, spawns, _post_flags(stage, variant), door_list))
    if pos != len(data):
        raise ValueError(f'trailing bytes after {count} rooms: {len(data) - pos}')
    return rooms


def stage_id(level_stage: int, stage_type: int) -> int:
    """RoomConfig::GetStageID(eLevelStage, eStageType)."""
    if stage_type == 3:
        return level_stage + 0x12
    if level_stage > 8:
        if level_stage == 9:
            return 13
        if level_stage == 12:
            return 26
        return stage_type - 6 + level_stage * 2
    return stage_type + 1 + ((level_stage - 1) >> 1) * 3


class RoomConfig:
    def __init__(self, archives: ArchiveSet):
        self.archives = archives
        xml = archives.read('resources/stages.xml').decode('utf-8-sig')
        root = re.search(r'<stages[^>]*\broot="([^"]*)"', xml).group(1)
        self.paths: dict[int, str] = {}
        self.names: dict[int, str] = {}
        for m in re.finditer(r'<stage\s+([^>]*)/?>', xml):
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            sid = int(attrs['id'])
            self.paths[sid] = 'resources/' + root + attrs['path'][:-3] + 'stb'
            self.names[sid] = attrs['name']
        self.stages: dict[int, list[Room]] = {}

    def rooms(self, stage: int) -> list[Room]:
        if stage not in self.stages:
            self.stages[stage] = parse_stb(self.archives.read(self.paths[stage]), stage)
        return self.stages[stage]

    def reset_room_weights(self, stage: int) -> None:
        for room in self.rooms(stage):
            room.weight = room.initial_weight

    def get_room(self, stage: int, rtype: int, variant: int) -> Room | None:
        for room in self.rooms(stage):
            if room.type == rtype and room.variant == variant:
                return room
        return None

    def get_rooms(self, stage: int, rtype: int, shape: int = SHAPE_ANY, min_variant: int = 0,
                  max_variant: int = 0xFFFFFFFF, min_difficulty: int = 0, max_difficulty: int = 15,
                  required_doors: int = 0, subtype: int = -1) -> list[Room]:
        out = []
        for r in self.rooms(stage):
            if (r.type == rtype and (shape == SHAPE_ANY or r.shape == shape)
                    and min_variant <= r.variant <= max_variant
                    and min_difficulty <= r.difficulty <= max_difficulty
                    and required_doors & r.doors == required_doors
                    and (subtype == -1 or r.subtype == subtype)):
                out.append(r)
        return out

    def get_random_room(self, seed: int, reduce_weight: bool, stage: int, rtype: int,
                        shape: int = SHAPE_ANY, min_variant: int = 0, max_variant: int = 0xFFFFFFFF,
                        min_difficulty: int = 0, max_difficulty: int = 15, required_doors: int = 0,
                        subtype: int = -1, void_level: bool = False) -> Room | None:
        """RoomConfig::GetRandomRoom. `void_level` is the engine's check `*g_Game == 12`."""
        query = (rtype, shape, min_variant, max_variant, min_difficulty, max_difficulty, required_doors,
                 subtype)
        if void_level and stage != 0:
            candidates = []
            for sid in [s for s in range(1, 18) if s != 13]:
                candidates += self.get_rooms(sid, *query)
        else:
            candidates = self.get_rooms(stage, *query)
        if not candidates:
            return None
        total, exact_total, exact = F32(0), F32(0), []
        for r in candidates:
            if required_doors and required_doors == r.doors:
                exact.append(r)
                exact_total = F32(exact_total + r.weight)
            total = F32(total + r.weight)
        if total <= 0:
            return None
        rng = RNG(seed, 7)
        r1 = rng.random_float()
        if not exact or F32(F32(exact_total / total) * EXACT_DOORS_FACTOR) <= rng.random_float():
            chosen = _weighted_pick(candidates, F32(r1 * total))
        else:
            chosen = _weighted_pick(exact, F32(r1 * exact_total))
        if chosen is not None and reduce_weight:
            chosen.weight = max(F32(WEIGHT_DECAY * chosen.weight), WEIGHT_FLOOR)
        return chosen


def _weighted_pick(rooms: list[Room], target: np.float32) -> Room | None:
    acc = F32(rooms[0].weight + F32(0))
    i = 0
    while acc <= target:
        i += 1
        if i == len(rooms):
            return None
        acc = F32(acc + rooms[i].weight)
    return rooms[i]


def abplus_archives(rooms_a: str, afterbirth_a: str, afterbirthp_a: str) -> ArchiveSet:
    """AB+ load order (later wins): base, Afterbirth, Afterbirth+."""
    return ArchiveSet([rooms_a, afterbirth_a, afterbirthp_a])


def default_archive_path() -> str:
    """AB+ v1.06 afterbirthp.a: $ISAAC_ABPLUS_AFTERBIRTHP, else the copy under analysis/.

    That archive alone holds stages.xml and every room file of stages 0-24 (it overrides rooms.a
    and afterbirth.a for all of them), so it is enough for normal floors.
    """
    env = os.environ.get('ISAAC_ABPLUS_AFTERBIRTHP')
    if env:
        return env
    root = Path(__file__).resolve().parents[3]
    return str(root / 'analysis' / 'abplus-linux' / 'resources' / 'packed' / 'afterbirthp.a')


def default_room_config() -> RoomConfig:
    return RoomConfig(ArchiveSet([default_archive_path()]))
