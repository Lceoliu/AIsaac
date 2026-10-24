"""J460 seeds, stage ids and room queries (isaac_macro.rep)."""
import os

import pytest

from isaac_macro.rep.rng import Seeds
from isaac_macro.rep.roomconfig import default_archive_path, stage_id
from isaac_macro.rep.run import next_stage, stage_seed_for
from isaac_macro.rep.level import RepGameContext
from isaac_macro.rng import RNG, Seeds as AbplusSeeds


@pytest.fixture(scope='module')
def rep_rc():
    if not os.path.exists(default_archive_path()):
        pytest.skip('Rep+ afterbirthp.a not found')
    from isaac_macro.rep.roomconfig import default_room_config
    return default_room_config()


def test_seeds_have_fourteen_stages_and_keep_the_first_thirteen():
    for start in (1, 0x12345678, 322374056):
        rep, ab = Seeds(start), AbplusSeeds(start)
        assert len(rep.stage_seeds) == 14
        assert rep.stage_seeds[:13] == ab.stage_seeds
        assert rep.player_init_seed != ab.player_init_seed      # one draw later in J460
        assert rep.stage_seed(20) == rep.stage_seeds[13]


def test_forget_stage_seed_uses_triple_39():
    s = Seeds(99)
    before = s.stage_seeds[3]
    s.forget_stage_seed(3)
    x = before ^ (before >> 5)
    x ^= (x << 15) & 0xFFFFFFFF
    x ^= x >> 17
    assert s.stage_seeds[3] == x == RNG(before, 39).next()


def test_stage_ids():
    assert [stage_id(s, 0) for s in range(1, 9)] == [1, 1, 4, 4, 7, 7, 10, 10]
    assert [stage_id(s, 2) for s in (1, 3, 5, 7)] == [3, 6, 9, 12]
    assert [stage_id(s, 4) for s in (1, 2, 3, 5, 7)] == [27, 27, 29, 31, 33]
    assert [stage_id(s, 5) for s in (1, 3, 5)] == [28, 30, 32]
    assert (stage_id(9, 0), stage_id(9, 4), stage_id(12, 0), stage_id(13, 0)) == (13, 36, 26, 35)
    assert (stage_id(10, 0), stage_id(10, 1), stage_id(11, 0), stage_id(11, 1)) == (14, 15, 16, 17)


def test_alt_floors_use_the_next_stage_seed():
    s = Seeds(4242)
    assert stage_seed_for(s, 1, 4) == s.stage_seeds[2]
    assert stage_seed_for(s, 1, 0) == s.stage_seeds[1]


def test_main_route_matches_abplus_rules():
    from isaac_macro.level import GameContext
    from isaac_macro.run import next_stage as ab_next
    import random
    rnd = random.Random(3)
    for _ in range(200):
        seeds = Seeds(rnd.getrandbits(32) or 1)
        ab_seeds = AbplusSeeds(seeds.start_seed)
        for stage in range(0, 8):
            for curses in (0, 2):
                assert next_stage(RepGameContext(), seeds, stage, 0, curses) == \
                    ab_next(GameContext(), ab_seeds, stage, 0, curses)


def test_rooms_load_for_every_main_and_alt_floor(rep_rc):
    for stage in range(1, 9):
        for st in (0, 1, 2, 4, 5):
            if st == 5 and stage >= 7:
                continue
            assert rep_rc.rooms(stage_id(stage, st))


def test_get_random_room_decays_weight(rep_rc):
    room = rep_rc.get_random_room(12345, True, 1, 1, 1, 0, 0xFFFFFFFF, 1, 5, 0x0F, -1)
    assert room is not None and room.shape == 1 and room.doors & 0x0F == 0x0F
    assert room.weight < room.initial_weight
    rep_rc.reset_room_weights(1)
    assert room.weight == room.initial_weight
