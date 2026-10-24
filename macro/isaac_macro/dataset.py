"""Offline floor datasets from the translated generator.

Floors come from whole runs (Seeds -> SetNextStage -> Level::Init), so stage types, Curse of the
Labyrinth skips and run flags follow the engine. The player's resources, which decide some special
rooms (arcade: 5+ coins on even floors; challenge room: full health; dice room: 2+ keys; bedroom:
low health), are sampled per floor by `random_player`; the defaults are a rough guess, not measured.

Record fields (JSONL):
    run_seed, run_seed_str, stage, stage_type, stage_seed, curses, attempts, player{...},
    start, rooms[{index,x,y,shape,type,variant,subtype,doors,layout_doors,difficulty,name,depth}],
    boss, secret, super_secret   (cells; lists because Labyrinth floors have two boss rooms)
Array fields (NPZ, one row per floor, 13x13 grids, -1 = empty):
    room[N,13,13] int16 (index into the record's rooms), type, shape, depth int8,
    doors[N,13,13] uint8 (bit d set: the cell is connected to a different room in direction d,
    d = 0 left, 1 up, 2 right, 3 down), and stage, stage_type, curses, stage_seed per floor.
"""
from __future__ import annotations

import gzip
import json
import random
from dataclasses import asdict
from typing import Iterator

import numpy as np

from .floor import Floor
from .level import ROOM_BOSS, ROOM_SECRET, ROOM_SUPERSECRET, GameContext, Player
from .levelgen import GRID, TRAVEL, index
from .rng import Seeds, seed_to_string
from .roomconfig import RoomConfig
from .run import iter_run


def random_player(rng: random.Random, stage: int) -> Player:
    """A plausible resource state on arrival at `stage` (not calibrated on real runs)."""
    max_hearts = 2 * rng.choice([2, 3, 3, 3, 4, 4, 5, 6])
    hearts = rng.randint(1, max_hearts) if rng.random() < 0.6 else max_hearts
    return Player(hearts=hearts, max_hearts=max_hearts, soul_hearts=rng.choice([0, 0, 0, 1, 2, 3, 4]),
                  keys=rng.choice([0, 0, 1, 1, 1, 2, 2, 3, 5]),
                  coins=rng.choice([0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]))


def iter_floors(room_config: RoomConfig, n_runs: int, seed: int = 0, last_stage: int = 8,
                debug_start: bool = False, player_fn=random_player) -> Iterator[tuple[dict, Floor]]:
    """Yield (record, Floor) for every floor of `n_runs` runs drawn with random start seeds."""
    rng = random.Random(seed)
    for _ in range(n_runs):
        start_seed = rng.getrandbits(32) or 1
        ctx = GameContext()
        seeds = Seeds(start_seed)
        gen = iter_run(room_config, seeds, ctx, last_stage=last_stage, debug_start=debug_start,
                       cathedral=rng.random() < 0.5)
        ctx.player = player_fn(rng, 1)
        for level in gen:
            floor = Floor.from_level(level)
            yield floor_record(level, floor, ctx.player, start_seed), floor
            ctx.player = player_fn(rng, level.stage + 1)


def floor_record(level, floor: Floor, player: Player, run_seed: int | None = None) -> dict:
    cells = {t: [r.safe_grid_index for r in floor.rooms if r.type == t]
             for t in (ROOM_BOSS, ROOM_SECRET, ROOM_SUPERSECRET)}
    rec = dict(run_seed=run_seed, run_seed_str=seed_to_string(run_seed) if run_seed else None,
               stage=level.stage, stage_type=level.stage_type, stage_seed=level.stage_seed,
               curses=level.curses, attempts=level.attempts,
               player={k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in asdict(player).items()},
               start=floor.start, boss=cells[ROOM_BOSS], secret=cells[ROOM_SECRET],
               super_secret=cells[ROOM_SUPERSECRET])
    rec['rooms'] = floor.to_dict()['rooms']
    return rec


def floor_arrays(floor: Floor) -> dict[str, np.ndarray]:
    room = np.full((GRID, GRID), -1, np.int16)
    rtype = np.full((GRID, GRID), -1, np.int8)
    shape = np.full((GRID, GRID), -1, np.int8)
    depth = np.full((GRID, GRID), -1, np.int8)
    doors = np.zeros((GRID, GRID), np.uint8)
    pos = {r.index: k for k, r in enumerate(floor.rooms)}
    for r in floor.rooms:
        for c in r.cells:
            y, x = divmod(c, GRID)
            room[y, x], rtype[y, x], shape[y, x], depth[y, x] = pos[r.index], r.type, r.shape, r.depth
    for r in floor.rooms:
        for slot in range(8):
            if not r.doors >> slot & 1:
                continue
            t = r.slot_target(slot)
            d = slot & 3
            src = index(t % GRID - TRAVEL[d][0], t // GRID - TRAVEL[d][1])
            y, x = divmod(src, GRID)
            doors[y, x] |= 1 << d
    return dict(room=room, type=rtype, shape=shape, depth=depth, doors=doors)


def export(room_config: RoomConfig, jsonl_path: str | None, npz_path: str | None, n_runs: int,
           seed: int = 0, last_stage: int = 8, debug_start: bool = False) -> int:
    """Write the floors of `n_runs` runs; returns the number of floors."""
    out = gzip.open(jsonl_path, 'wt', encoding='utf-8') if jsonl_path else None
    arrays: dict[str, list] = {}
    count = 0
    try:
        for rec, floor in iter_floors(room_config, n_runs, seed, last_stage, debug_start):
            if out:
                out.write(json.dumps(rec, separators=(',', ':')) + '\n')
            if npz_path:
                for k, v in floor_arrays(floor).items():
                    arrays.setdefault(k, []).append(v)
                for k in ('stage', 'stage_type', 'curses', 'stage_seed', 'run_seed'):
                    arrays.setdefault(k, []).append(rec[k])
            count += 1
    finally:
        if out:
            out.close()
    if npz_path:
        np.savez_compressed(npz_path, **{k: np.asarray(v, dtype=np.uint32 if k.endswith('seed') else None)
                                         for k, v in arrays.items()})
    return count
