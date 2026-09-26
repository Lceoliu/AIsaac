"""J460 run progression on the main route: Level::SetNextStage (RVA 0x3467C0) -> Level::Init.

On the main route J460 keeps the AB+ rules: the stage type of the next floor comes from its stage
seed (even -> the second variant if Unlocked(0x56/0x57/0x58) for chapters 1-3, always for the
Womb; divisible by 3 -> the third variant if Unlocked(0x156..0x159)), Curse of the Labyrinth skips
a floor (Womb I skips two), stage 8 leads to Sheol/Cathedral by run flag 19, and stage 11 keeps
its type. Not ported: the alternate path (secret exit rooms, knife pieces, Mom's door: stage types
4/5 with their own seed rule), Home (13), the Ascent, Greed mode and challenges.
"""
from __future__ import annotations

from typing import Iterator

from ..level import CURSE_LABYRINTH, LevelResult
from .level import RepGameContext, RepLevel
from .rng import Seeds
from .roomconfig import RepRoomConfig

NEXT_STAGE = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 10, 10: 11, 11: 11, 12: 12}
FLAG_CATHEDRAL = 19


def next_stage(ctx: RepGameContext, seeds: Seeds, stage: int, stage_type: int, curses: int,
               cathedral: bool | None = None) -> tuple[int, int]:
    if ctx.greed or ctx.challenge or ctx.ascent:
        raise NotImplementedError('Greed mode / challenges / Ascent')
    if stage_type in (4, 5):
        raise NotImplementedError('alternate path progression')
    nxt = NEXT_STAGE[stage]
    if curses & CURSE_LABYRINTH and stage in (1, 3, 5, 7):
        nxt += 2 if stage == 7 else 1
    if nxt == 10:
        flag = (ctx.state_flags >> FLAG_CATHEDRAL) & 1 if cathedral is None else int(cathedral)
        return 10, flag
    if nxt == 11:
        return 11, stage_type
    seed = seeds.stage_seed(nxt)
    st = 0
    if seed & 1 == 0 and ((nxt in (1, 2) and ctx.unlocked(0x56)) or (nxt in (3, 4) and ctx.unlocked(0x57))
                          or (nxt in (5, 6) and ctx.unlocked(0x58)) or nxt in (7, 8)):
        st = 1
    if seed % 3 == 0 and nxt < 10 and (
            (nxt in (1, 2) and ctx.unlocked(0x156)) or (nxt in (3, 4) and ctx.unlocked(0x157))
            or (nxt in (5, 6) and ctx.unlocked(0x158)) or (nxt in (7, 8) and ctx.unlocked(0x159))):
        st = 2
    return nxt, st


def stage_seed_for(seeds: Seeds, stage: int, stage_type: int) -> int:
    """Level::Init: stage types 4/5 use the next stage's seed."""
    return seeds.stage_seed(stage + 1 if stage_type in (4, 5) else stage)


def iter_run(room_config: RepRoomConfig, seeds: Seeds, ctx: RepGameContext | None = None, last_stage: int = 8,
             debug_start: bool = False, cathedral: bool | None = None, bosspool=None) -> Iterator[LevelResult]:
    """Floors of one run. The BossPool is per run: Game::Start and Game::StartDebug both call
    BossPool::Init with the game start seed, and its state (pool RNGs, bosses already seen) carries
    from floor to floor; pass one to continue a run, or leave None to start a new one."""
    from .bosspool import BossPool
    ctx = ctx or RepGameContext()
    if debug_start:
        ctx.achievements = None
        stage, stage_type = 1, 0
    else:
        stage, stage_type = next_stage(ctx, seeds, 0, 0, 0, cathedral)
    if bosspool is None:
        bosspool = BossPool.from_room_config(room_config, start_seed=seeds.start_seed)
    while stage <= last_stage:
        level = RepLevel(room_config, ctx, stage, stage_type, bosspool=bosspool).init(
            stage_seed_for(seeds, stage, stage_type))
        yield level
        if stage >= 11:
            break
        stage, stage_type = next_stage(ctx, seeds, stage, stage_type, level.curses, cathedral)
