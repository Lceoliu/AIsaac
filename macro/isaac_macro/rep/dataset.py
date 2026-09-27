"""Floors of whole J460 runs (main route), as records like isaac_macro.dataset plus `ultra_secret`.

Same sampling as isaac_macro.dataset.iter_floors (random start seeds, a random Sheol / Cathedral
route, player resources from random_player), on the static J460 port (not engine-validated).
"""
from __future__ import annotations

import random
from typing import Iterator

from ..dataset import floor_record, random_player
from ..floor import Floor
from .level import RepGameContext
from .place_rooms import ROOM_ULTRASECRET
from .rng import Seeds
from .roomconfig import RepRoomConfig
from .run import iter_run


def iter_floors(room_config: RepRoomConfig, n_runs: int, seed: int = 0, last_stage: int = 8,
                debug_start: bool = False, player_fn=random_player) -> Iterator[tuple[dict, Floor]]:
    """Yield (record, Floor) for every floor of `n_runs` runs drawn with random start seeds."""
    rng = random.Random(seed)
    for _ in range(n_runs):
        start_seed = rng.getrandbits(32) or 1
        ctx = RepGameContext()
        gen = iter_run(room_config, Seeds(start_seed), ctx, last_stage=last_stage, debug_start=debug_start,
                       cathedral=rng.random() < 0.5)
        ctx.player = player_fn(rng, 1)
        for level in gen:
            floor = Floor.from_level(level)
            rec = floor_record(level, floor, ctx.player, start_seed)
            rec['ultra_secret'] = [r.safe_grid_index for r in floor.rooms if r.type == ROOM_ULTRASECRET]
            yield rec, floor
            ctx.player = player_fn(rng, level.stage + 1)
