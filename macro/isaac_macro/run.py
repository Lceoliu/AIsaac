"""A run's sequence of floors: Game::Start -> Level::SetNextStage -> Level::Init, repeated.

Game::Start (0x2E5800) calls Level::SetNextStage from stage 0, so a normal run can open in the Cellar
or the Burning Basement. Game::StartDebug (0x2DFDD0), which the engine instances of this project use
(their log.txt has no "RNG Start Seed" line), calls Level::SetStage with an explicit stage and type
instead and marks every achievement as unlocked (Game+0x215EA5, read by PersistentGameData::Unlocked).

Level::SetNextStage (0x332580) picks the next eLevelStage and its stage type from the next stage
seed and the save's achievements. Run flags (bosses and sins already seen, ...) live in
GameContext.state_flags and carry from floor to floor as in the engine; the player's resources are
whatever the caller puts in ctx.player before each floor.
"""
from __future__ import annotations

from typing import Iterator

from .level import CURSE_LABYRINTH, GameContext, LevelResult, generate_floor
from .rng import Seeds
from .roomconfig import RoomConfig

NEXT_STAGE = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 10, 10: 11, 11: 11, 12: 12}
FLAG_CATHEDRAL = 19    # Game+0x215E6A bit 3: after stage 8 go to the Cathedral (1) instead of Sheol (0)


def next_stage(ctx: GameContext, seeds: Seeds, stage: int, stage_type: int, curses: int,
               cathedral: bool | None = None) -> tuple[int, int]:
    """Level::SetNextStage for normal runs (no Greed mode, no challenge 0x1F, no special seeds).

    `curses` are the curses of the floor being left (Level::GetCurses). Leaving from the off-grid
    rooms -8 (Blue Womb trapdoor) or -9 (Void portal) is not modelled; those routes lead to floors
    this package does not generate exactly. `cathedral` overrides run flag 19; None reads it from
    ctx.state_flags.
    """
    if ctx.greed or ctx.challenge:
        raise NotImplementedError('Greed mode / challenges')
    nxt = NEXT_STAGE[stage]
    if curses & CURSE_LABYRINTH:
        if stage in (1, 3, 5):
            nxt += 1
        elif stage == 7:
            nxt += 2
    seed = seeds.stage_seed(nxt)
    st = 0
    if seed & 1 == 0:
        if nxt in (1, 2) and ctx.unlocked(0x56):
            st = 1
        elif nxt in (3, 4) and ctx.unlocked(0x57):
            st = 1
        elif nxt in (5, 6) and ctx.unlocked(0x58):
            st = 1
        else:
            st = int(nxt in (7, 8))
    if seed % 3 == 0 and nxt < 10 and (
            (nxt in (1, 2) and ctx.unlocked(0x156)) or (nxt in (3, 4) and ctx.unlocked(0x157))
            or (nxt in (5, 6) and ctx.unlocked(0x158)) or (nxt in (7, 8) and ctx.unlocked(0x159))):
        st = 2
    if nxt == 10:
        st = int(ctx.flag(FLAG_CATHEDRAL) if cathedral is None else cathedral)
    elif nxt == 11:
        st = stage_type
    return nxt, st


def iter_run(room_config: RoomConfig, seeds: Seeds, ctx: GameContext | None = None,
             last_stage: int = 8, debug_start: bool = False,
             cathedral: bool | None = None) -> Iterator[LevelResult]:
    """Yield the floors of one run from the first floor down to `last_stage`.

    Blue Womb (9), Greed mode and challenges are not translated. Between floors the caller may edit
    ctx.player (keys, coins, hearts, trinkets, items) to model the run so far; ctx.state_flags
    already carries the engine's own run flags. `debug_start` reproduces Game::StartDebug(1, 0):
    Basement I and all achievements unlocked, as in the RL engine instances.
    """
    ctx = ctx or GameContext()
    if debug_start:
        ctx.achievements = None
        stage, stage_type = 1, 0
    else:
        stage, stage_type = next_stage(ctx, seeds, 0, 0, 0, cathedral)
    while stage <= last_stage:
        floor = generate_floor(room_config, ctx, stage, stage_type, seeds.stage_seed(stage))
        yield floor
        if stage >= 11:
            break
        stage, stage_type = next_stage(ctx, seeds, stage, stage_type, floor.curses, cathedral)


def generate_run(room_config: RoomConfig, start_seed: int | str, ctx: GameContext | None = None,
                 last_stage: int = 8, debug_start: bool = False,
                 cathedral: bool | None = None) -> list[LevelResult]:
    """All floors of a run; `start_seed` is the numeric seed or the 'XXXX XXXX' string."""
    seeds = Seeds.from_string(start_seed) if isinstance(start_seed, str) else Seeds(start_seed)
    return list(iter_run(room_config, seeds, ctx, last_stage, debug_start, cathedral))
