"""J460 Level::place_rooms (isaac_macro.rep.place_rooms) through RepLevel: helper arithmetic and floor
invariants. Static port only; these tests check self-consistency, not agreement with the engine."""
import importlib.util
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest

from isaac_macro.floor import Floor
from isaac_macro.level import CURSE_LABYRINTH, Level as AbplusLevel, Player
from isaac_macro.rep.level import RepGameContext, RepLevel
from isaac_macro.rep.roomconfig import default_archive_path
from isaac_macro.rng import RNG

F32 = np.float32
HAVE_PORTS = all(importlib.util.find_spec(f'isaac_macro.rep.{m}') is not None for m in ('levelgen', 'bosspool'))
needs_ports = pytest.mark.skipif(not HAVE_PORTS, reason='rep/levelgen.py or rep/bosspool.py not ported yet')

GRID = 13
START = 0x54
LUNA, VOODOO_HEAD, BIRTHRIGHT, MOMS_BOX = 589, 599, 619, 439
SILVER_DOLLAR, BLOODY_CROWN, PAY_TO_WIN, FRAGMENTED_CARD, TELESCOPE_LENS = 110, 111, 112, 102, 152


@pytest.fixture(scope='module')
def pr():
    if not HAVE_PORTS:
        pytest.skip('rep/levelgen.py or rep/bosspool.py not ported yet')
    from isaac_macro.rep import place_rooms
    return place_rooms


@pytest.fixture(scope='module')
def rep_rc():
    if not os.path.exists(default_archive_path()):
        pytest.skip('Rep+ afterbirthp.a not found')
    if not HAVE_PORTS:
        pytest.skip('rep/levelgen.py or rep/bosspool.py not ported yet')
    from isaac_macro.rep.roomconfig import default_room_config
    return default_room_config()


@pytest.fixture(scope='module')
def pool_template(rep_rc):
    from isaac_macro.rep.bosspool import BossPool
    return BossPool.from_room_config(rep_rc)


def reset_all_weights(rc):
    """Room weights are game state that survives floors (Level::Init only resets its own files and
    stage 0); tests that compare two generations start both from initial weights."""
    for sid in list(rc.stages):
        rc.reset_room_weights(sid)


def generate(rc, pool_template, stage, stage_type, seed, start_seed, ctx=None):
    bp = pool_template.copy()
    bp.init(start_seed)
    level = RepLevel(rc, ctx or RepGameContext(), stage, stage_type, bosspool=bp)
    return level, level.init(seed)


def cells_of(desc):
    from isaac_macro.rep.levelgen import PLACEMENT, index
    x, y = desc.grid_index % GRID, desc.grid_index // GRID
    return [index(x + ox, y + oy) for ox, oy in PLACEMENT[desc.shape]]


def orthogonal(cell):
    x, y = cell % GRID, cell // GRID
    return [(x + dx) + GRID * (y + dy) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= x + dx < GRID and 0 <= y + dy < GRID]


def check_layout(level, res):
    """Structural invariants every generated floor must satisfy."""
    rooms = res.rooms
    assert [d.list_index for d in rooms] == list(range(len(rooms)))
    owner = {}
    for d in rooms:
        for c in cells_of(d):
            assert c >= 0, 'room outside the grid'
            assert c not in owner, 'overlapping rooms'
            owner[c] = d.list_index
    assert {c: i for c, i in enumerate(res.grid) if i >= 0} == owner
    start = res.room_at(START)
    assert start is not None and start.type == 1
    types = [d.type for d in rooms]
    assert types.count(8) >= 1 and types.count(29) <= 1
    floor = Floor.from_level(res)
    dist = floor.distances()
    for r in floor.rooms:
        if r.type != 29:                       # the ultra secret room touches no room
            assert r.index in dist, f'room {r.index} (type {r.type}) unreachable'
    for ss in floor.of_type(8):
        nbs = floor.neighbors(ss)
        assert len(nbs) == 1 and nbs[0][1].type == 1
    for d in rooms:
        if d.type == 29:
            assert all(res.grid[n] < 0 for n in orthogonal(d.grid_index))
            assert d.field_b4 == 99
    # the last boss room blocks its cells and their neighbours (0x5B1860)
    last = rooms[level.boss_list_index]
    assert last.type == 5
    blocked = {i for i, b in enumerate(res.generator.blocked) if b}
    area = set(cells_of(last))
    area |= {n for c in list(area) for n in orthogonal(c)}
    assert area <= blocked
    for d in rooms:
        if d.type in (7, 29):
            assert not set(cells_of(d)) & area
    return types


# ---------------------------------------------------------------------------------------- helpers
def test_init_seeds_three_draws_and_a_derived_fourth(pr):
    for seed in (1, 0xDEADBEEF, 123456789):
        rng = RNG(seed, 12)
        deco, spawn, award, extra = pr.init_seeds(rng)
        ref = RNG(seed, 12)
        assert (deco, spawn, award) == (ref.next(), ref.next(), ref.next())
        assert rng.seed == ref.seed                    # the fourth seed does not advance the RNG
        assert extra == RNG(award, 66).next()


def test_trinket_multiplier(pr):
    p = Player(trinkets=frozenset({SILVER_DOLLAR}))
    assert pr.trinket_multiplier(p, SILVER_DOLLAR) == 1 and not pr.has_golden_trinket(p, SILVER_DOLLAR)
    p = Player(trinkets=frozenset({SILVER_DOLLAR | 0x8000}))
    assert pr.trinket_multiplier(p, SILVER_DOLLAR) == 2 and pr.has_golden_trinket(p, SILVER_DOLLAR)
    p = Player(trinkets=frozenset({SILVER_DOLLAR | 0x8000}), active_item=MOMS_BOX)
    assert pr.trinket_multiplier(p, SILVER_DOLLAR) == 3
    assert pr.trinket_multiplier(Player(active_item=MOMS_BOX), SILVER_DOLLAR) == 0


def test_health_rules(pr):
    assert pr.full_health(Player())                                     # 6/6 red
    assert not pr.full_health(Player(hearts=4))
    assert pr.full_health(Player(hearts=4, soul_hearts=2))
    assert pr.full_health(pr.RepPlayer(hearts=1, lost_curse=True))
    assert pr.full_health(Player(player_type=39, hearts=0, max_hearts=0))
    assert pr.low_health(Player(hearts=1)) and not pr.low_health(Player())
    assert pr.low_health(Player(player_type=4, hearts=0, max_hearts=0, soul_hearts=2, bone_hearts=1))
    assert not pr.low_health(Player(player_type=16, hearts=0, max_hearts=0, soul_hearts=2, bone_hearts=1))


def _level_stub(stage, stage_type=0, **ctx):
    c = RepGameContext(**ctx)
    return SimpleNamespace(ctx=c, stage=stage, stage_type=stage_type, get_curses=lambda: 0)


def test_planetarium_chance(pr):
    base, step = F32(0.009999999776482582), F32(0.20000000298023224)
    assert pr.planetarium_chance(_level_stub(1)) == base
    assert pr.planetarium_chance(_level_stub(4)) == base                # every treasure room entered
    assert pr.planetarium_chance(_level_stub(4, treasure_rooms_visited=0)) == F32(F32(F32(3) * step) + base)
    assert pr.planetarium_chance(_level_stub(4, treasure_rooms_visited=0, planetarium_visits=1)) == base
    assert pr.planetarium_chance(_level_stub(2, 4, treasure_rooms_visited=0)) == F32(F32(F32(2) * step) + base)
    crystal = Player(collectibles=frozenset({158}))
    assert pr.planetarium_chance(_level_stub(4, treasure_rooms_visited=0, player=crystal)) == F32(1.0)
    assert pr.planetarium_chance(_level_stub(7)) == 0                    # past the Depths: Telescope Lens
    lens = Player(trinkets=frozenset({TELESCOPE_LENS}))
    expected = F32(F32(base + F32(0.15000000596046448)) + F32(0.09000000357627869))
    assert pr.planetarium_chance(_level_stub(7, player=lens)) == expected
    assert pr.planetarium_chance(_level_stub(9, player=lens)) == 0       # stage 9+: golden or Mom's Box
    golden = Player(trinkets=frozenset({TELESCOPE_LENS | 0x8000}))
    assert pr.planetarium_chance(_level_stub(10, player=golden)) > 0
    assert pr.planetarium_chance(_level_stub(11, player=golden)) == 0


def test_shop_levels(pr):
    def levels(stage=1, **ctx):
        out = set()
        for s in range(1, 4000):
            out.add(pr._shop_subtype(_level_stub(stage, **ctx), RNG(s * 2654435761 & 0xFFFFFFFF or 1, 35))[0])
        return out
    assert levels() == {4, 10, 11}                                     # everything unlocked: level 5 shops
    assert levels(achievements=frozenset()) == {0, 11}
    assert levels(difficulty=1) == {0, 1, 2, 3, 4, 10, 11}             # hard mode: random levels
    assert levels(player=Player(player_type=33)) == {104, 110, 111}     # Tainted Keeper
    dollar = Player(trinkets=frozenset({SILVER_DOLLAR | 0x8000}))
    assert levels(7, player=dollar) == {10, 0}                          # golden Silver Dollar: 4 -> 10


# ---------------------------------------------------------------------------------------- floors
@needs_ports
@pytest.mark.parametrize('stage', [1, 2, 3, 4, 5, 6, 7, 8])
def test_floor_invariants(rep_rc, pool_template, stage):
    rnd = random.Random(1000 + stage)
    for stage_type in (0, 1, 2):
        for _ in range(12):
            level, res = generate(rep_rc, pool_template, stage, stage_type, rnd.getrandbits(32) or 1,
                                  rnd.getrandbits(32) or 1)
            types = check_layout(level, res)
            labyrinth = bool(res.curses & CURSE_LABYRINTH)
            assert types.count(5) == (2 if labyrinth else 1)
            assert types.count(8) == 1 and types.count(7) <= 1
            start = res.room_at(START)
            assert (start.config.stage, start.variant) == (0, 2)
            if stage < 7:
                assert types.count(2) == 1 and types.count(4) == (2 if labyrinth else 1)
            else:
                assert types.count(2) == 0 and types.count(4) == 0
            assert types.count(9) == 0 and types.count(20) == 0       # no coins, no keys
            boss = res.rooms[level.first_boss_list_index]
            if stage == 6:
                assert boss.config.subtype == 6                         # Mom
            if stage == 8:
                assert boss.config.subtype == 25                        # It Lives (unlocked)
            sid = pr_stage_id(stage, stage_type)
            for d in res.rooms:
                if d.type == 1 and d.list_index != res.grid[START]:
                    # normal rooms: subtype 0 from the floor's file; Depths II extra room: subtype 1
                    # from the floor's file or the Depths file
                    assert (d.config.stage, d.config.subtype) == (sid, 0) or (
                        d.config.subtype == 1 and d.config.stage in (sid, 7))
            assert set(level.offgrid_rooms) == {-2, -4, -5, -6, -7, -8, -13, -18}


def pr_stage_id(stage, stage_type):
    from isaac_macro.rep.roomconfig import stage_id
    return stage_id(stage, stage_type)


@needs_ports
def test_secret_rooms_mostly_present(rep_rc, pool_template):
    rnd = random.Random(7)
    found = total = 0
    for stage in (1, 3, 5, 7):
        for _ in range(15):
            level, res = generate(rep_rc, pool_template, stage, 0, rnd.getrandbits(32) or 1, rnd.getrandbits(32) or 1)
            total += 1
            found += any(d.type == 7 for d in res.rooms)
    assert found >= total - 2


@needs_ports
def test_secret_blacklist_matches_abplus(rep_rc, pool_template, pr):
    """0x338D70 is the AB+ 0x337210 algorithm: compare on generated J460 floors."""
    rnd = random.Random(11)
    for stage in (2, 5, 11):
        for _ in range(5):
            st = rnd.choice([0, 1])
            level, res = generate(rep_rc, pool_template, stage, st, rnd.getrandbits(32) or 1, rnd.getrandbits(32) or 1)
            ab = AbplusLevel.__new__(AbplusLevel)
            ab.stage, ab.start_index, ab.rooms = stage, START, res.rooms
            assert pr.secret_room_blacklist(level) == ab.blacklist()


@needs_ports
def test_generation_is_deterministic(rep_rc, pool_template):
    rnd = random.Random(99)
    for stage in (1, 4, 6, 8):
        for st in (0, 2):
            seed, start = rnd.getrandbits(32) or 1, rnd.getrandbits(32) or 1
            out = []
            for _ in range(2):
                reset_all_weights(rep_rc)
                level, res = generate(rep_rc, pool_template, stage, st, seed, start)
                out.append(([(d.grid_index, d.type, d.variant, d.config.subtype, d.shape, d.decoration_seed,
                               d.spawn_seed, d.award_seed, d.boss_death_seed, d.flags, d.display_flags)
                             for d in res.rooms], res.grid, res.attempts, level.rng.seed,
                            sorted((i, d.config.type, d.config.variant) for i, d in level.offgrid_rooms.items()),
                            sorted(level.bosspool.removed)))
            assert out[0] == out[1]


@needs_ports
def test_run_shares_one_boss_pool(rep_rc, pool_template):
    """Floors of one run share the BossPool: committed bosses are removed for later floors."""
    from isaac_macro.rep.rng import Seeds
    seeds = Seeds(0x1234ABCD)
    bp = pool_template.copy()
    bp.init(seeds.start_seed)
    ctx = RepGameContext()
    removed = []
    for stage in range(1, 9):
        level = RepLevel(rep_rc, ctx, stage, 0, bosspool=bp)
        res = level.init(seeds.stage_seed(stage))
        check_layout(level, res)
        assert not bp.level_blacklist                         # committed after the successful attempt
        removed.append(len(bp.removed))
    assert removed == sorted(removed) and removed[-1] >= 1


@needs_ports
def test_labyrinth_floors(rep_rc, pool_template):
    rnd = random.Random(5)
    for stage in (1, 3, 5, 7):
        for _ in range(6):
            ctx = RepGameContext(extra_curses=CURSE_LABYRINTH)
            level, res = generate(rep_rc, pool_template, stage, 0, rnd.getrandbits(32) or 1,
                                  rnd.getrandbits(32) or 1, ctx)
            types = check_layout(level, res)
            assert types.count(5) == 2
            first, last = res.rooms[level.first_boss_list_index], res.rooms[level.boss_list_index]
            assert first.list_index != last.list_index
            assert set(cells_of(last)) & {n for c in cells_of(first) for n in orthogonal(c)}
            assert types.count(4) == (2 if stage < 7 else 0)


@needs_ports
def test_items_and_characters(rep_rc, pool_template):
    rnd = random.Random(21)
    counts = {'luna': 0, 'luna8': 0, 'card': 0, 'voodoo': 0}
    for _ in range(12):
        seed, start = rnd.getrandbits(32) or 1, rnd.getrandbits(32) or 1
        _, res = generate(rep_rc, pool_template, 3, 0, seed, start,
                          RepGameContext(player=Player(collectibles=frozenset({LUNA}))))
        types = [d.type for d in res.rooms]
        assert 1 <= types.count(8) <= 2 and types.count(7) <= 2
        counts['luna8'] += types.count(8) == 2              # a second dead end was left
        counts['luna'] += types.count(7) == 2
        _, res = generate(rep_rc, pool_template, 3, 0, seed, start,
                          RepGameContext(player=Player(trinkets=frozenset({FRAGMENTED_CARD | 0x8000}))))
        secrets = [d for d in res.rooms if d.type == 7]
        counts['card'] += len(secrets) == 2
        if secrets:
            assert secrets[0].display_flags & 4                  # golden Fragmented Card shows the first
        _, res = generate(rep_rc, pool_template, 4, 0, seed, start,
                          RepGameContext(player=Player(collectibles=frozenset({VOODOO_HEAD}), hearts=2)))
        counts['voodoo'] += sum(d.type == 10 for d in res.rooms) == 2
        _, res = generate(rep_rc, pool_template, 7, 0, seed, start,
                          RepGameContext(player=Player(trinkets=frozenset({SILVER_DOLLAR, BLOODY_CROWN}))))
        assert sum(d.type == 2 for d in res.rooms) == 1 and sum(d.type == 4 for d in res.rooms) == 1
        _, res = generate(rep_rc, pool_template, 2, 0, seed, start,
                          RepGameContext(player=Player(trinkets=frozenset({PAY_TO_WIN}))))
        assert [d.config.subtype for d in res.rooms if d.type == 4][0] in (2, 3)
        _, res = generate(rep_rc, pool_template, 2, 0, seed, start, RepGameContext(player=Player(player_type=33)))
        assert [d.config.subtype for d in res.rooms if d.type == 2][0] >= 100
        # Cain with Birthright: arcade layouts of subtype 1, allowed on odd floors and without coins
        _, res = generate(rep_rc, pool_template, 3, 0, seed, start,
                          RepGameContext(player=Player(player_type=2, collectibles=frozenset({BIRTHRIGHT}))))
        arcades = [d for d in res.rooms if d.type in (9, 20)]
        assert all(d.type == 9 and d.config.subtype == 1 for d in arcades)
        counts['cain'] = counts.get('cain', 0) + len(arcades)
    assert counts['luna8'] >= 8 and counts['luna'] >= 8 and counts['card'] >= 8 and counts['voodoo'] >= 6
    assert counts['cain'] >= 2      # dead ends usually run out first on stage 3 (curse + challenge rooms)


@needs_ports
def test_miniboss_sets_its_state_flag(rep_rc, pool_template):
    rnd = random.Random(3)
    seen = 0
    for _ in range(40):
        ctx = RepGameContext()
        _, res = generate(rep_rc, pool_template, 4, 0, rnd.getrandbits(32) or 1, rnd.getrandbits(32) or 1, ctx)
        mini = [d for d in res.rooms if d.type == 6]
        if mini:
            seen += 1
            sub = mini[0].config.subtype
            flag = {2: 9, 9: 9, 3: 10, 10: 10, 1: 11, 8: 11, 0: 12, 7: 12, 5: 13, 12: 13, 6: 14, 13: 14, 14: 32}[sub]
            assert ctx.flag(flag)
    assert seen >= 5


@needs_ports
def test_depths_ii_strange_door(rep_rc, pool_template):
    rnd = random.Random(8)
    for stage_type in (0, 1, 2):
        for strange in (True, False):
            for _ in range(4):
                ctx = RepGameContext(strange_door=strange)
                level, res = generate(rep_rc, pool_template, 6, stage_type, rnd.getrandbits(32) or 1,
                                      rnd.getrandbits(32) or 1, ctx)
                check_layout(level, res)
                special = [d for d in res.rooms if d.type == 1 and d.config.subtype == 1]
                assert len(special) == (1 if strange else 0)
                if strange:
                    # (6,5) is blocked for make_rooms (only a resized special room could take it)
                    above = res.room_at(6 + 5 * GRID)
                    assert above is None or above.type != 1
                    assert special[0].config.stage in (7, pr_stage_id(6, stage_type))


@needs_ports
def test_alt_path_and_late_floors(rep_rc, pool_template, pr):
    rnd = random.Random(13)
    corpse_flags = 0
    for stage, stage_type in ((5, 4), (5, 5), (6, 4), (6, 5), (7, 4), (8, 4), (10, 0), (10, 1), (11, 0),
                              (11, 1), (12, 0)):
        for _ in range(4):
            level, res = generate(rep_rc, pool_template, stage, stage_type, rnd.getrandbits(32) or 1,
                                  rnd.getrandbits(32) or 1)
            types = check_layout(level, res)
            if stage == 12:
                assert 6 <= types.count(5) <= 9
            if stage == 11 and stage_type == 0:
                assert any(d.type == 1 and d.config.stage == 0 and 3 <= d.variant <= 9 for d in res.rooms)
            if stage == 8 and stage_type == 4:
                assert level.offgrid_rooms[-10].config.subtype == 0x58      # Mother
            if stage_type == 4 and stage in (7, 8):
                corpse_flags += sum(bool(d.flags & 0x1000) for d in res.rooms)
    assert corpse_flags > 0


@needs_ports
def test_mirror_and_mineshaft_floors_stop_after_place_rooms(rep_rc, pool_template):
    ctx = RepGameContext(player=Player(collectibles=frozenset({626})))
    for stage, stage_type in ((2, 4), (4, 5)):
        bp = pool_template.copy()
        bp.init(77)
        level = RepLevel(rep_rc, ctx, stage, stage_type, bosspool=bp)
        with pytest.raises(NotImplementedError, match='mirror world / abandoned mineshaft'):
            level.init(12345)
        want = 0x22 if stage == 2 else 10
        assert any(d.type == 1 and d.config.subtype == want for d in level.rooms)


@needs_ports
def test_bosspool_is_required(rep_rc):
    level = RepLevel(rep_rc, RepGameContext(), 1, 0)
    with pytest.raises(ValueError, match='BossPool'):
        level.init(1)
