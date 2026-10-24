"""J460 RoomConfig: stage ids, room queries and the weighted room pick.

RoomConfig::GetStageID      RVA 0x42D030   new stage types 4/5 (Repentance alt path), Home, Ascent
RoomConfig::LoadStageBinary RVA 0x42AA40   STB1 unchanged; post-load flags: stages 1-3 variants
                                           0x25B/0x2CC -> 4, stages 29/30 type 1 subtype 0x1F -> 2
RoomConfig::GetRooms        RVA 0x42CE40   same filter as AB+ (+ a mode argument: 0 normal, 1 Greed;
                                           rooms flagged 0x80 are only dropped under special seed
                                           effect 79, which this port does not model)
RoomConfig::GetRandomRoom   RVA 0x42C7D0   same weighted pick as AB+; the Void takes stage files
                                           1-17 except 13 with max difficulty forced to 20; with
                                           curse bit 0x80 (Curse of the Giant, never rolled in normal
                                           play) a coin flip may use generated rooms (0x42D880),
                                           which is not ported
RoomConfig::GetRoom         RVA 0x42C720   (REPENTOGON's ZHL scan mislabels it ResetRoomWeights)

Every active room file, stages.xml and the other config XMLs of Rep+ are in its afterbirthp.a, which
the engine mounts after rooms.a and afterbirth.a; repentance.a has no room files.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..archive import ArchiveSet
from ..roomconfig import EXACT_DOORS_FACTOR, F32, SHAPE_ANY, WEIGHT_DECAY, WEIGHT_FLOOR, Room, RoomConfig, \
    _weighted_pick, parse_stb
from ..rng import RNG

VOID_STAGE_FILES = [s for s in range(1, 18) if s != 13]
VOID_MAX_DIFFICULTY = 20
HOME_STAGE_ID, ASCENT_STAGE_ID, VOID_STAGE_ID = 0x23, 0x24, 0x1A
CURSE_OF_THE_GIANT = 0x80


def stage_id(stage: int, stage_type: int, greed: bool = False) -> int:
    """RoomConfig::GetStageID(eLevelStage, eStageType, mode)."""
    if greed:
        if stage == 7:
            return 0x19
        if stage == 6:
            return 0x18
        if stage == 5:
            return 0x0E
        if stage_type == 4:
            return stage * 2 + 0x19
        if stage_type == 5:
            return stage * 2 + 0x1A
        return stage_type + stage * 3 - 2
    if stage == 13:
        return HOME_STAGE_ID
    if stage > 8:
        if stage == 9:
            return ASCENT_STAGE_ID if stage_type == 4 else 0x0D
        if stage == 12:
            return VOID_STAGE_ID
        return stage_type + (stage - 3) * 2
    if stage_type == 4:
        return ((stage - 1) & ~1) + 0x1B
    if stage_type == 5:
        return ((stage - 1) >> 1) * 2 + 0x1C
    return stage_type + 1 + ((stage - 1) >> 1) * 3


def _post_flags_j460(room: Room) -> int:
    flags = 0
    if room.stage in (1, 2, 3) and room.variant in (0x25B, 0x2CC):
        flags |= 4
    if room.stage in (0x1D, 0x1E) and room.type == 1 and room.subtype == 0x1F:
        flags |= 2
    return flags


class RepRoomConfig(RoomConfig):
    def rooms(self, stage: int) -> list[Room]:
        if stage not in self.stages:
            rooms = parse_stb(self.archives.read(self.paths[stage]), stage)
            for r in rooms:
                r.flags = _post_flags_j460(r)
            self.stages[stage] = rooms
        return self.stages[stage]

    def get_random_room(self, seed: int, reduce_weight: bool, stage: int, rtype: int,
                        shape: int = SHAPE_ANY, min_variant: int = 0, max_variant: int = 0xFFFFFFFF,
                        min_difficulty: int = 0, max_difficulty: int = 15, required_doors: int = 0,
                        subtype: int = -1, void_level: bool = False, curses: int = 0) -> Room | None:
        """RoomConfig::GetRandomRoom. `void_level` is the engine's check that the current level is
        the Void (stage 12); `curses` are the level's curses (Level::GetCurses)."""
        if curses & CURSE_OF_THE_GIANT:
            raise NotImplementedError('Curse of the Giant: generated-room path (RVA 0x42D880)')
        if void_level and stage not in (0, HOME_STAGE_ID):
            candidates = []
            for sid in VOID_STAGE_FILES:
                candidates += self.get_rooms(sid, rtype, shape, min_variant, max_variant, min_difficulty,
                                             VOID_MAX_DIFFICULTY, required_doors, subtype)
        else:
            candidates = self.get_rooms(stage, rtype, shape, min_variant, max_variant, min_difficulty,
                                        max_difficulty, required_doors, subtype)
        total, exact_total, exact = F32(0), F32(0), []
        for r in candidates:
            total = F32(total + r.weight)
            if required_doors and required_doors == r.doors:
                exact.append(r)
                exact_total = F32(exact_total + r.weight)
        if not candidates or total <= 0:
            return None
        rng = RNG(seed, 7)
        r1 = rng.random_float()
        ratio = F32(F32(exact_total / total) * EXACT_DOORS_FACTOR)
        if not exact or ratio <= rng.random_float():
            chosen = _weighted_pick(candidates, F32(r1 * total))
        else:
            chosen = _weighted_pick(exact, F32(r1 * exact_total))
        if chosen is not None and reduce_weight:
            chosen.weight = max(F32(WEIGHT_DECAY * chosen.weight), WEIGHT_FLOOR)
        return chosen


def default_archive_path() -> str:
    """Rep+ afterbirthp.a: $ISAAC_REPPLUS_AFTERBIRTHP, else the local Steam install (read only)."""
    env = os.environ.get('ISAAC_REPPLUS_AFTERBIRTHP')
    if env:
        return env
    return str(Path('D:/Steam/steamapps/common/The Binding of Isaac Rebirth/resources/packed/afterbirthp.a'))


def default_room_config() -> RepRoomConfig:
    return RepRoomConfig(ArchiveSet([default_archive_path()]))
