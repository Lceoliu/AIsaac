"""The translated generator against the engine's own log.txt (AB+ v1.06 instances)."""
import json
import os
import random

import pytest

from isaac_macro.floor import Floor
from isaac_macro.level import GameContext, Level, generate_floor
from isaac_macro.levelgen import GRID, ROOM_SIZE

FIXTURE = os.path.join(os.path.dirname(__file__), 'data', 'engine_levels_abplus.json')


def replay(rc, stage, stage_type, seed):
    level = Level(rc, GameContext(), stage, stage_type)
    level.log = []
    res = level.init(seed)
    attempts, cur = [], None
    for ev in level.log:
        if ev[0] == 'generate':
            cur = [[], None, None, 0, None]
            attempts.append(cur)
        elif ev[0] == 'place_room' and cur is not None:
            cur[0].append(ev[1])
        elif ev[0] == 'rooms':
            cur[1], cur[2] = ev[1], ev[2]
        elif ev[0] == 'fail':
            cur[1], cur[2] = ev[1], ev[3]
        elif ev[0] == 'new_dead_end':
            cur[3] += 1
        elif ev[0] == 'dead_ends':
            cur[4] = f'dead_ends {ev[1]}/{ev[2]}'
        elif ev[0] in ('placing', 'unusable'):
            cur[4] = ev[0]
    start = [d for d in res.rooms if d.grid_index == 84 and d.type == 1 and d.config.stage == 0]
    return res, attempts, start[0].spawn_seed if start else None


def test_engine_log_regression(rc):
    data = json.load(open(FIXTURE))
    bad = []
    for stage, stage_type, seed, curses, map_loops, spawn, attempts in data['levels']:
        res, ours, our_spawn = replay(rc, stage, stage_type, seed)
        if (res.curses, res.attempts, ours, our_spawn) != (curses, map_loops, attempts, spawn):
            bad.append(seed)
    assert not bad, f'{len(bad)}/{len(data["levels"])} levels differ from the engine log, e.g. {bad[:5]}'


def test_engine_floor_dumps(rc):
    """Whole floors read from the engine (Level:GetRooms), stages 1-11, every stage type."""
    import gzip
    from isaac_macro.level import Player
    data = json.load(gzip.open(os.path.join(os.path.dirname(__file__), 'data', 'engine_floors_abplus.json.gz')))
    bad = []
    for stage, stage_type, seed, curses, flags_in, (hearts, max_hearts, soul, keys, coins), rooms in data['floors']:
        ctx = GameContext(state_flags=flags_in, player=Player(hearts=hearts, max_hearts=max_hearts, soul_hearts=soul,
                                                              keys=keys, coins=coins, active_item=105))
        res = generate_floor(rc, ctx, stage, stage_type, seed)
        ours = [[d.grid_index, d.type, d.variant, d.config.subtype, d.shape, d.spawn_seed, d.decoration_seed,
                 d.award_seed] for d in res.rooms]
        if ours != rooms or res.curses != curses:
            bad.append((stage, stage_type, seed))
    assert not bad, f'{len(bad)}/{len(data["floors"])} floors differ from the engine, e.g. {bad[:5]}'


def test_room_files_parse(rc):
    for sid in range(0, 18):
        rooms = rc.rooms(sid)
        assert rooms, sid
        for r in rooms:
            assert (r.width // 13, r.height // 7) == ROOM_SIZE[r.shape], (sid, r.variant)


@pytest.mark.parametrize('stage', [1, 2, 3, 4, 5, 6, 7, 8, 10, 11])
def test_floor_invariants(rc, stage):
    rnd = random.Random(stage)
    for _ in range(25):
        stage_type = rnd.choice([0, 1] if stage >= 10 else [0, 1, 2])
        level = generate_floor(rc, GameContext(), stage, stage_type, rnd.getrandbits(32) or 1)
        floor = Floor.from_level(level)
        types = [r.type for r in floor.rooms]
        assert floor.room_at(84) is not None and floor.room_at(84).type == 1
        assert 1 <= types.count(5) <= 2
        assert types.count(7) <= 1 and types.count(8) <= 1
        assert len(floor.distances()) == len(floor.rooms)       # every room is reachable
        for ss in floor.of_type(8):
            nbs = floor.neighbors(ss)
            assert len(nbs) == 1 and nbs[0][1].type == 1
        for r in floor.rooms:
            for c in r.cells:
                assert 0 <= c < GRID * GRID and floor.grid[c] == r.index
