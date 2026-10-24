"""J460 boss selection (isaac_macro.rep.bosspool): MT19937, bosspools.xml, BossPool, boss-room picker."""
import os
import random

import numpy as np
import pytest

from isaac_macro.rep.bosspool import (BOSS_BITS, POOL_COUNT, BossContext, BossEntry, BossPool, MT19937, Pool,
                                      PoolSpec, parse_boss_achievements, parse_boss_pools, parse_xml,
                                      pick_boss_room)
from isaac_macro.rep.roomconfig import default_archive_path, stage_id
from isaac_macro.rng import RNG, SHIFTS

F32 = np.float32
HARBINGERS = {9, 10, 11, 12, 22, 38}
LOCKED = BossContext(achievements=frozenset())


@pytest.fixture(scope='module')
def rep_rc():
    """RoomConfig over the Rep+ afterbirthp.a; skips when the archive is absent."""
    path = default_archive_path()
    if not os.path.exists(path):
        pytest.skip(f'Rep+ archive not found: {path} (set ISAAC_REPPLUS_AFTERBIRTHP)')
    from isaac_macro.rep.roomconfig import default_room_config
    return default_room_config()


@pytest.fixture(scope='module')
def rep_data(rep_rc):
    return (parse_boss_pools(rep_rc.archives.read('resources/bosspools.xml')),
            parse_boss_achievements(rep_rc.archives.read('resources/bossportraits.xml')))


def fresh(rep_data, seed=0x1234ABCD, ctx=None):
    bp = BossPool(*rep_data, ctx=ctx)
    bp.init(seed)
    return bp


# ------------------------------------------------------------------------------------ MT19937
def test_mt19937_known_answers():
    mt = MT19937(5489)
    assert [mt.next() for _ in range(5)] == [3499211612, 581869302, 3890346734, 3586334585, 545404204]
    lazy = MT19937()                       # mti == 625: the first Next seeds with 5489
    out = [lazy.next() for _ in range(10000)]
    assert out[0] == 3499211612 and out[9999] == 4123659995     # std::mt19937 conformance value
    mt.init(5489)
    assert mt.next() == 3499211612


def test_mt19937_matches_numpy_init_genrand():
    bg = np.random.MT19937()
    if not hasattr(bg, '_legacy_seeding'):
        pytest.skip('numpy without MT19937._legacy_seeding')
    for seed in (1, 0x12345678, 0xFFFFFFFF, 0xAA17414F):
        bg._legacy_seeding(seed)
        ref = bg.random_raw(1300).tolist()
        mt = MT19937(seed)
        assert [mt.next() for _ in range(1300)] == ref, hex(seed)


# ------------------------------------------------------------------------------------ XML
def test_rapidxml_subset():
    nodes = parse_xml('<?xml version="1.0"?>\n<a x="1" y = \'2\'><!-- <b id="9"/> -->\n'
                      '\t<b id="3" w="&amp;"/><c></c>\n</wrong>')
    assert [n.name for n in nodes] == ['a']
    a = nodes[0]
    assert a.attrs == [('x', '1'), ('y', '2')]
    assert [(n.name, n.attrs) for n in a.children] == [('b', [('id', '3'), ('w', '&')]), ('c', [])]
    with pytest.raises(ValueError):
        parse_xml('<a>text</a>')             # would be a rapidxml data node
    with pytest.raises(ValueError):
        parse_xml('<a><b></b>')              # unexpected end of data


def test_parse_boss_pools_synthetic():
    xml = b'''<bosspools>
      <pool name="caves" doubletrouble="3700">
        <boss id="3" weight="0.25"/><boss id="4"/><boss id="7" weight="2" room="5000"/>
      </pool>
      <pool name="nowhere"><boss id="1" weight="1"/></pool>
      <pool name="sheol"/>
    </pool>'''
    specs = parse_boss_pools(xml)
    assert specs == [PoolSpec(4, 'caves', 3700, ((3, F32(0.25), 0), (4, F32(0.25), 0), (7, F32(2), 5000))),
                     PoolSpec(14, 'sheol', 0, ())]
    assert parse_boss_pools(b'<other/>') == []
    with pytest.raises(ValueError):
        parse_boss_pools(b'<bosspools><pool name="caves"><boss id="3"/></pool></bosspools>')
    ach = parse_boss_achievements(b'<bosses><boss id="9" achievement="5"/><boss id="1"/><boss id="200" '
                                  b'achievement="7"/></bosses>')
    assert ach == {9: 5, 1: -1}


EXPECTED_POOLS = {   # name: (stage id, bosses, total weight, double trouble)
    'basement': (1, 11, 8.75, 0), 'cellar': (2, 8, 8.0, 0), 'burning basement': (3, 12, 9.75, 0),
    'caves': (4, 12, 11.25, 3700), 'catacombs': (5, 13, 13.0, 3700), 'flooded caves': (6, 14, 13.25, 3700),
    'depths': (7, 8, 7.25, 3750), 'necropolis': (8, 7, 7.0, 3750), 'dank depths': (9, 8, 8.0, 3750),
    'womb': (10, 6, 5.25, 3800), 'utero': (11, 5, 5.0, 3800), 'scarred womb': (12, 7, 8.0, 3800),
    'sheol': (14, 0, 0.0, 0), 'cathedral': (15, 0, 0.0, 0), 'dark room': (16, 0, 0.0, 0),
    'chest': (17, 0, 0.0, 0), 'blue womb': (13, 0, 0.0, 0), 'void': (26, 0, 0.0, 0),
    'downpour': (27, 4, 4.0, 0), 'dross': (28, 5, 5.0, 0), 'mines': (29, 4, 4.0, 0), 'ashpit': (30, 5, 5.0, 0),
    'mausoleum': (31, 2, 2.0, 0), 'gehenna': (32, 2, 2.0, 0), 'corpse': (33, 3, 3.0, 0),
}


def test_bosspools_xml_counts(rep_rc, rep_data):
    raw = rep_rc.archives.read('resources/bosspools.xml')
    assert raw.rstrip().endswith(b'</pool>')      # J460 closes <bosspools> with </pool>: rapidxml accepts it
    specs, achievements = rep_data
    got = {s.name: (s.index, len(s.bosses), float(sum(F32(w) for _, w, _ in s.bosses)), s.double_trouble)
           for s in specs}
    assert got == EXPECTED_POOLS
    assert all(room == 0 for s in specs for _, _, room in s.bosses)
    assert [b for b, _, _ in next(s for s in specs if s.name == 'basement').bosses] == \
        [1, 17, 2, 44, 64, 56, 65, 20, 13, 60, 84]
    for bid, ach in {9: 5, 12: 5, 19: 18, 20: 16, 21: 17, 22: 5, 25: 34, 38: 66, 42: 68, 57: 346, 66: 346,
                     67: 347, 74: 347, 84: 347, 94: 347, 1: -1, 23: -1, 102: -1}.items():
        assert achievements[bid] == ach, bid


# ------------------------------------------------------------------------------------ Init
def test_init_seeds_and_shuffle(rep_data):
    specs, _ = rep_data
    seed = 0x0BADF00D
    bp = fresh(rep_data, seed)
    chain = RNG(seed, 26)
    for pool in bp.pools:
        assert pool.rng.seed == chain.next()
        assert (pool.rng.a, pool.rng.b, pool.rng.c) == SHIFTS[17] == (2, 21, 9)
    assert len(bp.pools) == POOL_COUNT and not bp.removed and not bp.level_blacklist
    for spec in specs:
        ids = [b for b, _, _ in spec.bosses]
        mt = MT19937(bp.pools[spec.index].rng.seed)
        for i in range(len(ids) - 1, 0, -1):
            j = mt.next() % (i + 1)
            ids[i], ids[j] = ids[j], ids[i]
        pool = bp.pools[spec.index]
        assert [e.id for e in pool.entries] == ids
        assert all(e.initial_weight == e.weight for e in pool.entries)
        assert pool.double_trouble == spec.double_trouble
    again = fresh(rep_data, seed)
    assert [[e.id for e in p.entries] for p in again.pools] == [[e.id for e in p.entries] for p in bp.pools]
    other = fresh(rep_data, seed + 1)
    assert [[e.id for e in p.entries] for p in other.pools] != [[e.id for e in p.entries] for p in bp.pools]
    with pytest.raises(ValueError):
        bp.init(0)


# ------------------------------------------------------------------------------------ PickBoss
def synthetic_pool(weights, ids=None):
    ids = ids or list(range(1, len(weights) + 1))
    entries = [BossEntry(i, F32(w), F32(w), -1, 0) for i, w in zip(ids, weights)]
    total = F32(0)
    for e in entries:
        total = F32(e.weight + total)
    return Pool(1, 'test', entries, total, RNG(1, 17), 0)


def test_pick_boss_intervals():
    pool = synthetic_pool([1, 0.25, 0, 2, 1])
    cums, c = [], F32(0)
    for e in pool.entries:
        c = F32(c + e.initial_weight)
        cums.append(c)
    for k in range(0, 4250, 7):
        r = F32(k / 1000)
        bp = BossPool([], {})
        chosen = bp.pick_boss(pool, r)
        want = next(i for i, cum in enumerate(cums) if r < cum)
        assert chosen is pool.entries[want] and chosen.id != 3        # the weight-0 entry never wins
        assert bp.level_blacklist == {chosen.id}


def test_pick_boss_repick_rule_and_probe():
    pool = synthetic_pool([1, 1])
    bp = BossPool([], {})
    bp.removed = {1}
    assert bp.pick_boss(pool, F32(0.25)).id == 2      # r -> ((1 - 0.25) / 1) * 2 = 1.5
    bp.level_blacklist.clear()
    assert bp.pick_boss(pool, F32(0.9)).id == 2       # 0.9 -> 0.2 -> 1.6
    bp.level_blacklist.clear()
    assert bp.pick_boss(pool, F32(0.5)).id == 2       # 0.5 -> 1.0: lands on B at once
    # r stuck on A: ((1 - 0.5) / 1) * 1 with a pool whose total is 1 (ids 1, 2 weights 1, 0.0)
    pool0 = synthetic_pool([1, 0])
    bp = BossPool([], {})
    bp.removed = {1}
    assert bp.pick_boss(pool0, F32(0.5)) is None      # probe skips the weight-0 entry
    pool3 = synthetic_pool([1, 1, 1])
    bp = BossPool([], {})
    bp.removed = {1, 2}
    assert bp.pick_boss(pool3, F32(0.5)).id == 3      # 0.5 -> 1.5, B rejected until the last scan, probe -> C
    assert bp.level_blacklist == {3}
    bp.level_blacklist.clear()
    assert bp.pick_boss(pool3, F32(0.0)) is None      # r == prev maps to r == total: no scan matches again
    bp = BossPool([], {})
    bp.removed = {1, 2, 3}
    assert bp.pick_boss(pool3, F32(1.5)) is None and not bp.level_blacklist
    assert BossPool([], {}).pick_boss(Pool(0), F32(0)) is None


def test_pick_boss_respects_unlocks_and_scat_man():
    pool = synthetic_pool([1, 1], ids=[74, 20])
    pool.entries[1].achievement = 16
    bp = BossPool([], {})
    for k in range(20):
        r = F32((k + 0.37) / 10)
        assert bp.pick_boss(pool, r, BossContext(challenge=36, achievements=frozenset({16}))).id == 20
        bp.level_blacklist.clear()
        assert bp.pick_boss(pool, r, BossContext(achievements=frozenset())).id == 74
        bp.level_blacklist.clear()
        assert bp.pick_boss(pool, r, BossContext(challenge=36, achievements=frozenset())) is None
    # the rescale can land exactly on the total weight: 1.25 -> 1.5 -> 1.0 -> 2.0, then no scan matches
    assert bp.pick_boss(pool, F32(1.25), BossContext(achievements=frozenset())) is None


# ------------------------------------------------------------------------------------ GetBossId
FIXED = [((6, 0), 6), ((6, 2), 6), ((8, 0), 25), ((8, 1), 25), ((10, 0), 24), ((10, 1), 39), ((11, 0), 54),
         ((11, 1), 40), ((12, 0), 70), ((6, 4), 89), ((6, 5), 89), ((8, 4), 88), ((8, 5), 88)]


def test_get_boss_id_fixed_bosses(rep_data):
    bp = fresh(rep_data)
    seeds = [p.rng.seed for p in bp.pools]
    for (stage, st), boss in FIXED:
        assert bp.get_boss_id(stage, st) == boss
    assert [p.rng.seed for p in bp.pools] == seeds and not bp.level_blacklist
    assert bp.get_boss_id(8, 0, ctx=LOCKED) == 8                       # It Lives needs achievement 34
    for override in (BossContext(achievements=frozenset(), daily_id=1),
                     BossContext(achievements=frozenset(), is_debug=True)):
        assert bp.get_boss_id(8, 0, ctx=override) == 25
    assert bp.get_boss_id(8, 0, ctx=BossContext(achievements=frozenset(), daily_id=1, manager_state=1)) == 8
    for stage in range(1, 13):
        assert bp.get_boss_id(stage, 0, ctx=BossContext(challenge=32)) == 30
    daily = BossContext(special_daily_id=13)
    assert [bp.get_boss_id(s, 0, ctx=daily) for s in (1, 2, 3, 4, 5, 7)] == [60, 43, 50, 23, 51, 15]
    assert [p.rng.seed for p in bp.pools] == seeds


def predict_shortcut(bp, stage, st, ctx):
    """The non-weighted outcomes of GetBossId from the pool RNG's 7 draws (documented order)."""
    pool = bp.pool_for(stage, st, ctx)
    rng = pool.rng.copy()
    r = [rng.next() for _ in range(7)]
    if r[1] % 10 == 0 and ctx.unlocked(5) and stage in (1, 3, 5, 7) and st not in (4, 5):
        if r[2] % 10 == 0 and not bp.was_boss_removed(22):
            return 22
        horseman = {1: 9, 3: 10, 5: 11, 7: 12}[stage]
        if not bp.was_boss_removed(horseman):
            return 38 if horseman == 12 and r[0] % 2 == 0 and ctx.unlocked(66) else horseman
    if pool.double_trouble:
        mod = 1 if ctx.challenge == 34 else {3: 50, 4: 50, 5: 25, 7: 40}.get(stage, 0)
        if mod and r[3] % mod == 0:
            return -pool.double_trouble
    if r[4] % 10 == 0 and st not in (4, 5) and ctx.state_flag(6) and not bp.was_boss_removed(23):
        return 23
    return None


@pytest.mark.parametrize('stage', [1, 2, 3, 4, 5, 7])
def test_get_boss_id_invariants(rep_data, stage):
    rnd = random.Random(stage)
    bp = fresh(rep_data)
    for trial in range(400):
        st = rnd.choice([0, 1, 2, 4] if stage == 7 else [0, 1, 2, 4, 5])
        ctx = BossContext(state_flags=(1 << 6) if trial % 2 else 0,
                          achievements=None if trial % 3 else frozenset(rnd.sample(range(1, 400), 200)))
        pool = bp.pool_for(stage, st, ctx)
        assert pool.index == stage_id(stage, st)
        pool.rng = RNG(rnd.getrandbits(32) or 1, 17)
        bp.removed = set(rnd.sample([e.id for e in pool.entries], rnd.randrange(len(pool.entries) + 1)))
        bp.level_blacklist = set()
        before = pool.rng.copy()
        expected = predict_shortcut(bp, stage, st, ctx)
        removed_before = set(bp.removed)
        boss = bp.get_boss_id(stage, st, ctx=ctx)
        for _ in range(7):
            before.next()
        assert pool.rng.seed == before.seed                    # exactly 7 draws, copy not written back
        if expected is not None:
            assert boss == expected
            if boss > 0:
                assert boss in bp.level_blacklist
            if boss == 38:
                assert bp.level_blacklist >= {12, 38}
            continue
        assert boss > 0 and boss not in HARBINGERS | {23}
        entry = next(e for e in pool.entries if e.id == boss)
        assert ctx.unlocked(entry.achievement)
        assert bp.level_blacklist == {boss}
        pool_ids = {e.id for e in pool.entries}
        eligible = {e.id for e in pool.entries if ctx.unlocked(e.achievement)}
        # PickBoss failing resets the pool's bits; it must when nothing eligible is left (it also can,
        # rarely, when a rescaled r lands exactly on the total weight)
        assert bp.removed in (removed_before, removed_before - pool_ids)
        if bp.removed == removed_before:
            assert boss not in removed_before
        if not eligible - removed_before:
            assert bp.removed == removed_before - pool_ids


def test_get_boss_id_empty_pool_defaults_to_monstro(rep_data):
    bp = fresh(rep_data)
    pool = bp.pool_for(7, 5)                                   # 'mortis' is never named in bosspools.xml
    assert pool.index == 34 and not pool.entries and pool.double_trouble == 0
    before = pool.rng.copy()
    assert bp.get_boss_id(7, 5) == 1
    for _ in range(7):
        before.next()
    assert pool.rng.seed == before.seed and not bp.level_blacklist


def test_get_boss_id_rates(rep_data):
    """Harbinger / Headless Horseman / double trouble / Fallen rates follow the draws' moduli."""
    bp = fresh(rep_data)
    rnd = random.Random(7)
    n = 4000
    counts = {'hh': 0, 'famine': 0, 'dt': 0, 'fallen': 0}
    flag6 = BossContext(state_flags=1 << 6)
    for _ in range(n):
        for stage, key_ctx in ((1, None), (4, None), (2, flag6)):
            bp.pools[stage_id(stage, 0)].rng = RNG(rnd.getrandbits(32) or 1, 17)
            bp.removed.clear()
            bp.level_blacklist.clear()
            b = bp.get_boss_id(stage, 0, ctx=key_ctx)
            counts['hh'] += b == 22
            counts['famine'] += b == 9
            counts['dt'] += b == -3700
            counts['fallen'] += b == 23
    assert 0.07 < counts['famine'] / n < 0.11 and 0.004 < counts['hh'] / n < 0.02
    assert 0.01 < counts['dt'] / n < 0.03 and 0.07 < counts['fallen'] / n < 0.13


def test_run_without_repeats_then_reset(rep_data):
    bp = fresh(rep_data, 0x5EED)
    # locked save: harbingers and 346/347/16 bosses are out; basement keeps 1, 2, 13, 17, 44, 56
    eligible = {1, 2, 13, 17, 44, 56}
    seen = []
    for _ in range(6):
        bp.clear_level_blacklist()
        seen.append(bp.get_boss_id(1, 0, ctx=LOCKED))
        bp.commit_level_blacklist()
    assert sorted(seen) == sorted(eligible) and bp.removed == eligible
    bp.clear_level_blacklist()
    again = bp.get_boss_id(1, 0, ctx=LOCKED)                  # pool exhausted -> reset -> pick
    bp.commit_level_blacklist()
    assert again in eligible and bp.removed == {again}


def test_failed_attempts_do_not_consume_bosses(rep_data):
    bp = fresh(rep_data, 99)
    bp.clear_level_blacklist()
    first = bp.get_boss_id(1, 0, ctx=LOCKED)
    bp.clear_level_blacklist()                                # generate_dungeon retries the floor
    second = bp.get_boss_id(1, 0, ctx=LOCKED)
    bp.commit_level_blacklist()
    assert bp.removed == {second} and first in {1, 2, 13, 17, 44, 56}


def test_scat_man_and_ultra_hard(rep_data):
    bp = fresh(rep_data, 42)
    rnd = random.Random(3)
    for _ in range(300):
        pool = bp.pools[stage_id(5, 0)]                        # depths: Reap Creep (74) is in the pool
        pool.rng = RNG(rnd.getrandbits(32) or 1, 17)
        bp.removed.clear()
        assert bp.get_boss_id(5, 0, ctx=BossContext(challenge=36)) != 74
        bp.pools[stage_id(4, 0)].rng = RNG(rnd.getrandbits(32) or 1, 17)
        assert bp.get_boss_id(4, 0, ctx=BossContext(challenge=34)) == -3700
        assert bp.get_boss_id(2, 1, ctx=BossContext(challenge=34)) > 0   # cellar has no double trouble


def test_determinism_copy_and_save_state(rep_data):
    def run(bp, floors):
        out = []
        for stage, st in floors:
            bp.clear_level_blacklist()
            out.append(bp.get_boss_id(stage, st))
            bp.commit_level_blacklist()
        return out
    floors = [(1, 0), (2, 1), (3, 2), (4, 0), (5, 1), (6, 0), (7, 2), (8, 0)] * 3
    a, b = fresh(rep_data, 777), fresh(rep_data, 777)
    assert run(a, floors) == run(b, floors)
    head = fresh(rep_data, 777)
    run(head, floors[:9])
    seeds, removed = head.store_state()
    branch = head.copy()
    restored = fresh(rep_data, 777)
    restored.restore_state(seeds, removed)
    tail = run(head, floors[9:])
    assert run(branch, floors[9:]) == tail == run(restored, floors[9:])
    assert all(0 <= x < BOSS_BITS for x in removed)


# ------------------------------------------------------------------------------------ 0x339080
def test_pick_boss_room(rep_rc, rep_data):
    specs, _ = rep_data
    ids = sorted({b for s in specs for b, _, _ in s.bosses} | {6, 8, 9, 22, 23, 25, 38, 70, 88})
    rnd = random.Random(11)
    try:
        for boss in ids:
            for _ in range(3):
                rep_rc.reset_room_weights(0)
                seed = rnd.getrandbits(32) or 1
                room = pick_boss_room(rep_rc, boss, seed)
                assert room is not None and room.stage == 0 and room.type == 5 and room.subtype == boss
                assert not 3700 <= room.variant < 3850
                rep_rc.reset_room_weights(0)
                first = rep_rc.get_random_room(RNG(seed, 39).next(), True, 0, 5, 13, 0, 0xFFFFFFFF, 1, 10, 0, boss)
                if not 3700 <= first.variant < 3850:
                    assert first is room                        # first draw: xorshift (5, 15, 17) of the seed
        for start in (3700, 3750, 3800):
            for _ in range(20):
                rep_rc.reset_room_weights(0)
                room = pick_boss_room(rep_rc, -start, rnd.getrandbits(32) or 1, LOCKED)
                assert start <= room.variant <= start + 49 and room.type == 5
                s = room.subtype
                assert s not in (19, 20, 21, 42) and not 57 <= s <= 72
        rep_rc.reset_room_weights(0)
        assert pick_boss_room(rep_rc, 89, 5).variant == 6030        # Mom (mausoleum), stage 0
        dads_note = BossContext(state_flags=1 << 47)
        for level_sid in (stage_id(6, 4), stage_id(6, 5)):             # Gehenna II falls back to stage 31
            rep_rc.reset_room_weights(level_sid)
            rep_rc.reset_room_weights(31)
            room = pick_boss_room(rep_rc, 89, 5, dads_note, level_stage_id=level_sid)
            assert room.stage == 31 and room.subtype == 89
        with pytest.raises(ValueError):
            pick_boss_room(rep_rc, 89, 5, dads_note)
        with pytest.raises(NotImplementedError):
            pick_boss_room(rep_rc, 1, 5, BossContext(difficulty=2))
    finally:
        for sid in (0, 31, 32):
            rep_rc.reset_room_weights(sid)
