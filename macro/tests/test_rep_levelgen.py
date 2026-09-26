"""J460 LevelGenerator port (isaac_macro.rep.levelgen): grid invariants, the J460-only behaviour, and
identity with the engine-validated AB+ translation on every path the port considers unchanged."""
import random

import pytest

from isaac_macro import levelgen as ab
from isaac_macro.rep import levelgen as rp
from isaac_macro.rep.levelgen import (GRID, PLACEMENT, RING2, TRAVEL, GenRoom, LevelGenerator, door_target,
                                      index)

SPECIAL_SHAPES = [1, 1, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12]   # 5/7 hit try_resize_endroom option 13


def room_state(r):
    return (r.index, r.x, r.y, r.shape, r.doors, r.dir, r.parent_slot, r.entry, sorted(r.neighbors),
            r.neighbor_count, r.dead_end, r.depth, r.flag74)


def state(g):
    return (g.rng.seed, (g.rng.a, g.rng.b, g.rng.c), list(g.grid), [room_state(r) for r in g.rooms],
            list(g.dead_ends), list(g.others), g.boss_idx, g.boss_count)


def ref(room):
    return None if room is None else room.index


def random_case(rnd):
    seed = rnd.getrandbits(32) or 1
    shapes = rnd.choice([0x1FFF, 0x1FFF, 0x1FFF, 0x0002 | rnd.getrandbits(13), rnd.getrandbits(13) & 0x1FFD])
    return (seed, rnd.randint(1, 60), rnd.random() < 0.2, rnd.random() < 0.3, rnd.random() < 0.15, shapes)


def pair(seed):
    a, b = ab.LevelGenerator(seed), rp.LevelGenerator(seed)
    a.log, b.log = [], []
    return a, b


def blocked_cells(g):
    return {i for i in range(GRID * GRID) if g.blocked[i]}


def check_layout(g, start=(6, 6)):
    """Invariants of a grid right after Generate."""
    cells = {}
    for r in g.rooms:
        for ox, oy in PLACEMENT[r.shape]:
            i = index(r.x + ox, r.y + oy)
            assert i >= 0 and i not in cells
            cells[i] = r.index
    assert all(g.grid[i] == cells.get(i, -1) for i in range(GRID * GRID))
    assert (g.rooms[0].x, g.rooms[0].y) == start
    for r in g.rooms[1:]:
        dx, dy = TRAVEL[(r.dir + 2) % 4]
        p = g.rooms[g.grid[index(r.entry[0] + dx, r.entry[1] + dy)]]
        assert p.index < r.index and r.depth == p.depth + 1
        assert door_target(p.x, p.y, p.shape, r.parent_slot) == r.entry
        assert g.grid[index(*r.entry)] == r.index
    for r in g.rooms:
        doors, nbs = 0, set()
        for slot in range(8):
            t = door_target(r.x, r.y, r.shape, slot)
            if t is not None and index(*t) >= 0 and g.grid[index(*t)] >= 0:
                doors |= 1 << slot
                nbs.add(g.grid[index(*t)])
        assert (r.doors, r.neighbors, r.neighbor_count) == (doors, nbs, len(nbs))
    assert sorted(g.dead_ends + g.others) == list(range(len(g.rooms)))
    assert all(g.rooms[i].dead_end for i in g.dead_ends) and not any(g.rooms[i].dead_end for i in g.others)
    depths = [g.rooms[i].depth for i in g.dead_ends]
    assert depths == sorted(depths, reverse=True)
    if g.dark_room:
        assert g.grid[index(start[0], start[1] - 1)] < 0


# ------------------------------------------------------------------ identity with AB+
def test_tables_and_geometry_match_abplus():
    assert rp.ROOM_SIZE == ab.ROOM_SIZE and rp.TRAVEL == ab.TRAVEL and rp.PLACEMENT == ab.PLACEMENT
    assert rp.CANDIDATE_WEIGHTS == ab.CANDIDATE_WEIGHTS
    for x, y in ((0, 0), (5, 7), (12, 12), (-1, 3), (13, 0)):
        assert rp.index(x, y) == ab.index(x, y)
        for shape in range(13):
            for slot in range(8):
                for nr in (False, True):
                    assert rp.has_shape_slot(shape, slot, nr) == ab.has_shape_slot(shape, slot, nr)
                    assert rp.door_target(x, y, shape, slot, nr) == ab.door_target(x, y, shape, slot, nr)
                assert rp.door_source(x, y, shape, slot) == ab.door_source(x, y, shape, slot, False)


def test_door_source_ignores_narrow_flag():
    for shape in range(13):
        for slot in range(8):
            assert rp.door_source(4, 4, shape, slot, True) == rp.door_source(4, 4, shape, slot, False)
    assert ab.door_source(4, 4, 3, 0, True) == (4, 4) and rp.door_source(4, 4, 3, 0, True) is None


@pytest.mark.parametrize('chunk', range(4))
def test_generate_matches_abplus(chunk):
    rnd = random.Random(1000 + chunk)
    for _ in range(250):
        seed, n, dark, lab, void, shapes = random_case(rnd)
        a, b = pair(seed)
        for _ in range(rnd.randint(1, 3)):          # generate_dungeon retries on the same generator
            a.generate(n, dark, lab, void, shapes)
            b.generate(n, dark, lab, void, shapes)
            assert state(a) == state(b) and a.log == b.log


def test_special_rooms_match_abplus():
    """The place_rooms-side calls on unchanged paths: same results, same grid, same RNG. As in
    place_rooms, only dead ends are resized or given back (resizing a parent orphans its children), and
    the secret room, which has no entry cell, comes last."""
    rnd = random.Random(7)
    ops = ['boss', 'boss', 'end', 'end', 'end', 'add', 'random_end', 'dead_end', 'resize', 'determine',
           'cands', 'remaining']
    for _ in range(500):
        seed, n, dark, lab, void, shapes = random_case(rnd)
        a, b = pair(seed)
        a.generate(n, dark, lab, void, shapes)
        b.generate(n, dark, lab, void, shapes)
        steps = rnd.randint(3, 14)
        for step in range(steps):
            op = 'secret' if step == steps - 1 else rnd.choice(ops)
            shape = rnd.choice(SPECIAL_SHAPES)
            doors = rnd.getrandbits(8) | rnd.choice([0, 0, 0xFF])
            k = rnd.randrange(len(a.rooms))
            linked = [r.index for r in a.rooms if r.entry is not None and r.dead_end]
            ra = rb = None
            if op == 'boss':
                force = rnd.random() < 0.3
                ra, rb = a.get_new_boss_room(shape, doors, force), b.get_new_boss_room(shape, doors, force)
            elif op == 'end':
                ra, rb = a.get_new_end_room(shape, doors), b.get_new_end_room(shape, doors)
            elif op == 'add' and linked:
                k = rnd.choice(linked)
                a.add_end_room(a.rooms[k])
                b.add_end_room(b.rooms[k])
            elif op == 'secret':
                bl = {i for i in range(GRID * GRID) if rnd.random() < 0.1}
                ra, rb = a.get_new_secret_room(bl), b.get_new_secret_room(bl)
            elif op == 'random_end':
                ra, rb = a.create_random_end_room(1), b.create_random_end_room()
            elif op == 'dead_end':
                assert a.make_new_dead_end() == b.make_new_dead_end()
                a.calc_required_doors()
                b.calc_required_doors()
            elif op == 'resize' and a.dead_ends:
                k = rnd.choice(a.dead_ends)
                assert (a.try_resize_endroom(a.rooms[k], shape, doors)
                        == b.try_resize_endroom(b.rooms[k], shape, doors))
            elif op == 'determine':
                a.determine_boss_room(shape, doors)
                b.determine_boss_room(shape, doors)
            elif op == 'cands':
                flag = rnd.random() < 0.5
                assert ([room_state(c) for c in a.get_neighbor_candidates(k, flag)]
                        == [room_state(c) for c in b.get_neighbor_candidates(k, flag)])
            elif op == 'remaining':
                assert ([r.index for r in a.get_remaining_rooms()]
                        == [r.index for r in b.get_remaining_rooms()])
            assert ref(ra) == ref(rb)
            assert state(a) == state(b)


def test_option13_raises_like_abplus():
    for seed in range(1, 200):
        a, b = pair(seed * 2654435761 & 0xFFFFFFFF or 1)
        a.generate(20, False, False, False, 0x1FFF)
        b.generate(20, False, False, False, 0x1FFF)
        for shape in (5, 7):
            ea = eb = None
            try:
                a.get_new_end_room(shape, 0xFF)
            except NotImplementedError as e:
                ea = e
            try:
                b.get_new_end_room(shape, 0xFF)
            except NotImplementedError as e:
                eb = e
            assert (ea is None) == (eb is None)
            assert state(a) == state(b)


# ------------------------------------------------------------------ J460 changes
@pytest.mark.parametrize('min_dead_ends', [0, 2, 5, 6, 7, 9])
def test_min_dead_ends_with_abplus_primitives(min_dead_ends):
    """Generate(0x5ADBB0) = AB+ Generate with the J460 loop condition, spelled out with AB+ parts."""
    rnd = random.Random(min_dead_ends)
    for _ in range(150):
        seed, n, dark, lab, void, shapes = random_case(rnd)
        a, b = pair(seed)
        a.shapes, a.labyrinth, a.void, a.dark_room = shapes, lab, void, dark
        a.make_rooms(n)
        a.calc_required_doors()
        tries = 5
        while len(a.dead_ends) < min_dead_ends and tries > 0:
            a.log.append(('new_dead_end',))
            a.make_new_dead_end()
            a.calc_required_doors()
            tries -= 1
        a.sort_dead_ends()
        b.generate(n, dark, lab, void, shapes, min_dead_ends)
        assert state(a) == state(b) and a.log == b.log
        check_layout(b)
        tried = b.log.count(('new_dead_end',))
        assert tried <= 5 and (len(b.dead_ends) >= min_dead_ends or tried == 5)


def test_start_room_argument():
    rnd = random.Random(5)
    for _ in range(100):
        seed, n, dark, lab, void, shapes = random_case(rnd)
        a, b = LevelGenerator(seed), LevelGenerator(seed)
        a.generate(n, dark, lab, void, shapes)
        b.generate(n, dark, lab, void, shapes, 5, GenRoom(6, 6, 1))
        assert state(a) == state(b)
        check_layout(a)
    g = LevelGenerator(99)
    g.generate(12, False, False, False, 0x1FFF, 0, GenRoom(4, 9, 8, depth=3))
    check_layout(g, start=(4, 9))
    assert (g.rooms[0].shape, g.rooms[0].depth) == (8, 3) and min(r.depth for r in g.rooms[1:]) == 4


def test_dark_closet_style_generation():
    """0x351650: start (2,6) shape 2, (1,6) (2,5) (2,7) blocked, 50 rooms, no dead-end minimum."""
    for seed in range(1, 80):
        g = LevelGenerator(seed * 7919)
        for x, y in ((1, 6), (2, 5), (2, 7)):
            g.block_position(x, y)
        g.generate(50, False, False, False, 0x1FFF, 0, GenRoom(2, 6, 2))
        assert (g.rooms[0].x, g.rooms[0].y, g.rooms[0].shape) == (2, 6, 2)
        check_layout(g, start=(2, 6))
        assert all(g.grid[index(x, y)] < 0 for x, y in ((1, 6), (2, 5), (2, 7)))
        assert g.rooms[0].doors == 1 << 2


def test_generate_avoids_and_keeps_blocked_cells():
    rnd = random.Random(12)
    for _ in range(200):
        seed, n, dark, lab, void, shapes = random_case(rnd)
        g = LevelGenerator(seed)
        cells = {rnd.randrange(GRID * GRID) for _ in range(rnd.randint(0, 15))} - {84}
        for i in cells:
            g.block_position(i % GRID, i // GRID)
        for _ in range(2):
            g.generate(n, dark, lab, void, shapes, rnd.randint(0, 8))
            check_layout(g)
            assert not any(g.grid[i] >= 0 for i in cells)
            assert blocked_cells(g) == cells


def test_is_pos_free():
    g = LevelGenerator(1)
    g.dark_room = True
    assert g.is_pos_free(6, 5, 1)                       # no rooms: nothing reserved
    g.create_room(6, 6, 1)
    assert not g.is_pos_free(6, 5, 1) and g.is_pos_free(7, 5, 1)
    g.block_position(8, 5)
    assert not g.is_pos_free(8, 5, 1) and not g.is_pos_free(7, 4, 8) and g.is_pos_free(9, 4, 8)


def test_secret_room_skips_blocked_cells_like_the_blacklist():
    rnd = random.Random(11)
    for _ in range(300):
        seed, n, dark, lab, void, shapes = random_case(rnd)
        a, b = pair(seed)
        a.generate(n, dark, lab, void, shapes)
        b.generate(n, dark, lab, void, shapes)
        blocked = {i for i in range(GRID * GRID) if rnd.random() < 0.15}
        blacklist = {i for i in range(GRID * GRID) if rnd.random() < 0.1}
        for i in blocked:
            b.block_position(i % GRID, i // GRID)
        ra, rb = a.get_new_secret_room(blacklist | blocked), b.get_new_secret_room(blacklist)
        assert ref(ra) == ref(rb) and state(a) == state(b)


def test_create_random_end_room_blacklist():
    """0x5ADDF0 against AB+ parts: shuffle, candidates room by room, 1x1 only, listed cells skipped."""
    rnd = random.Random(21)
    for _ in range(250):
        seed, n, dark, lab, void, shapes = random_case(rnd)
        a, b = pair(seed)
        a.generate(n, dark, lab, void, shapes)
        b.generate(n, dark, lab, void, shapes)
        bl = {i for i in range(GRID * GRID) if rnd.random() < 0.3}
        rb = b.create_random_end_room(rnd.choice([1, 4, 8]), bl)      # the shape argument is not read
        order = list(range(len(a.rooms)))
        ab.random_shuffle(order, a.rng)
        ra = None
        for i in order:
            for cand in a.get_neighbor_candidates(i, True):
                if (cand.shape == 1 and a.is_pos_free(cand.x, cand.y, 1) and a.count_neighbors(cand.entry) < 2
                        and index(cand.x, cand.y) not in bl):
                    ra = a.place_room(cand)
                    a.calc_required_doors()
                    break
            if ra is not None:
                break
        assert ref(ra) == ref(rb) and state(a) == state(b)
        if rb is not None:
            assert rb.shape == 1 and index(rb.x, rb.y) not in bl


def test_block_helpers():
    g = LevelGenerator(1)
    g.block_room_area(GenRoom(11, 0, 8))
    assert blocked_cells(g) == {index(x, y) for x, y in ((10, 0), (11, 0), (12, 0), (10, 1), (11, 1), (12, 1),
                                                         (11, 2), (12, 2))}
    g = LevelGenerator(1)
    g.block_position(13, 0)
    g.block_position(-1, 5)
    assert not blocked_cells(g)
    g.block_unused_door_targets(GenRoom(6, 6, 3), 0b0001)          # ignore_narrow: all four sides count
    assert blocked_cells(g) == {index(6, 5), index(7, 6), index(6, 7)}
    other = LevelGenerator(2)
    other.create_room(6, 6, 1)
    other.create_room(0, 0, 6)
    g = LevelGenerator(3)
    g.block_occupied_area(other)
    expect = set()
    for cell in (index(6, 6), index(0, 0), index(1, 0)):
        x, y = cell % GRID, cell // GRID
        expect |= {index(x + dx, y + dy) for dx, dy in rp.BLOCK_OFFSETS if index(x + dx, y + dy) >= 0}
    assert blocked_cells(g) == expect


# ------------------------------------------------------------------ CreateRoom and caller-made layouts
def test_create_room_blue_womb_layout():
    """The fixed layout Level::generate_blue_womb (0x34A500) builds; CreateRoom draws no RNG."""
    g = LevelGenerator(1)
    start = g.create_room(6, 6, 4)
    rooms = [g.create_room(6, 8, 1, 6, 8, 3), g.create_room(7, 7, 1, 7, 7, 2),
             g.create_room(5, 7, 1, 5, 7, 0), g.create_room(6, 4, 8, 6, 5, 1)]
    assert start.doors == (1 << 1) | (1 << 3) | (1 << 4) | (1 << 6) and start.neighbors == {1, 2, 3, 4}
    assert g.dead_ends == [1, 2, 3, 4] and g.others == [0]
    assert start.entry is None and (rooms[3].entry, rooms[3].dir) == ((6, 5), 1)
    assert g.rng.seed == 1


def test_caller_made_rooms_edge_cases():
    g = LevelGenerator(1)
    g.create_room(6, 6, 1)
    loose = g.create_room(6, 8, 1)                          # no entry, direction -1: always a dead end
    assert g.dead_ends == [1]
    with pytest.raises(NotImplementedError):
        g.try_resize_endroom(loose, 1, 0xFF)
    loose.dir = 3
    before = state(g)
    assert g.try_resize_endroom(loose, 1, 0xFF) is False    # XY::Invalid: every placement is off the grid
    assert state(g) == before
    with pytest.raises(AssertionError):
        g.create_room(9, 9, 1, 9, 9, 0)                     # its parent cell (10,9) is empty
    h = LevelGenerator(1)
    h.shapes = 1                                            # grid[-1] is the shape mask
    h.create_room(6, 6, 1)
    corner = h.create_room(0, 0, 1, 0, 0, 2)                # parent cell (-1,0) is off the grid
    assert not corner.dead_end and h.dead_ends == []
    h.shapes = 0x1FFF
    with pytest.raises(AssertionError):
        h.calc_required_doors()


# ------------------------------------------------------------------ ultra secret room (0x5AE640)
def check_ultra(g, room, blacklist, blocked):
    x, y = room.x, room.y
    assert room.shape == 1 and room.flag44 and room.doors == 0 and not room.neighbors
    assert index(x, y) not in blacklist and index(x, y) not in blocked
    for dx, dy in TRAVEL:
        j = index(x + dx, y + dy)
        assert j < 0 or (g.grid[j] < 0 and j not in blocked)
    near = [index(x + ox, y + oy) for ox, oy in RING2 if index(x + ox, y + oy) >= 0
            and g.grid[index(x + ox, y + oy)] >= 0]
    assert near
    for ox, oy in RING2:
        j = index(x + ox, y + oy)
        if j < 0 or g.grid[j] < 0:
            continue
        nb = g.rooms[g.grid[j]]
        for d, (dx, dy) in enumerate(TRAVEL):
            if dx * ox + dy * oy > 0:
                t = door_target(nb.x, nb.y, nb.shape, (d + 2) & 3)
                assert t is not None and index(*t) >= 0 and index(*t) not in blacklist


def test_ultra_secret_room_single_room():
    for seed in range(1, 40):
        g = LevelGenerator(seed)
        g.create_room(6, 6, 1)
        room = g.get_new_ultra_secret_room()
        assert abs(room.x - 6) + abs(room.y - 6) == 2
        check_ultra(g, room, set(), set())
        facing = {s for s in range(4)
                  if abs(door_target(6, 6, 1, s)[0] - room.x) + abs(door_target(6, 6, 1, s)[1] - room.y) == 1}
        assert g.rooms[0].doors == sum(1 << s for s in facing) and len(facing) in (1, 2)


def reference_ultra(g, blacklist):
    """Independent spelling of 0x5AE640's selection on a copy of the RNG: (cell, rng seed after)."""
    rng = g.rng.copy()
    scores = {}
    for i in range(GRID * GRID):
        if g.grid[i] >= 0 or g.blocked[i] or i in blacklist:
            continue
        x, y = i % GRID, i // GRID
        rng.next()
        score = rng.seed % 5 + 10
        touching = [index(x + dx, y + dy) for dx, dy in TRAVEL]
        if any(j >= 0 and (g.grid[j] >= 0 or g.blocked[j]) for j in touching):
            continue
        rooms_near = [(ox, oy, g.rooms[g.grid[index(x + ox, y + oy)]]) for ox, oy in RING2
                      if index(x + ox, y + oy) >= 0 and g.grid[index(x + ox, y + oy)] >= 0]
        bad = False
        for ox, oy, nb in rooms_near:
            for d, (dx, dy) in enumerate(TRAVEL):
                if dx * ox + dy * oy > 0:
                    t = door_target(nb.x, nb.y, nb.shape, (d + 2) & 3)
                    if t is None or index(*t) < 0 or index(*t) in blacklist:
                        bad = True
        if bad or not rooms_near:
            continue
        scores[i] = score - {1: 6, 2: 3}.get(len(rooms_near), 0)
    top = max(scores.values(), default=None)
    best = [i for i in sorted(scores) if top is not None and scores[i] == top and top >= 0]
    if not best:
        return None, rng.seed
    rng.next()
    return best[rng.seed % len(best)], rng.seed


def test_ultra_secret_room_on_generated_floors():
    rnd = random.Random(31)
    found = 0
    for _ in range(300):
        seed, n, dark, lab, void, shapes = random_case(rnd)
        g = LevelGenerator(seed)
        g.generate(n, dark, lab, void, shapes)
        g.get_new_boss_room(rnd.choice(SPECIAL_SHAPES), 0xFF, False)
        blocked = {i for i in range(GRID * GRID) if rnd.random() < 0.1}
        for i in blocked:
            g.block_position(i % GRID, i // GRID)
        bl = {i for i in range(GRID * GRID) if rnd.random() < 0.1} if rnd.random() < 0.7 else None
        want_cell, want_seed = reference_ultra(g, bl or set())
        doors_before = [r.doors for r in g.rooms]
        room = g.get_new_ultra_secret_room(bl)
        assert (None if room is None else index(room.x, room.y)) == want_cell
        assert g.rng.seed == want_seed
        if room is None:
            continue
        found += 1
        check_ultra(g, room, bl or set(), blocked)
        for r in g.rooms[:-1]:
            added = r.doors & ~doors_before[r.index]
            for slot in range(8):
                if added >> slot & 1:
                    t = door_target(r.x, r.y, r.shape, slot)
                    assert abs(t[0] - room.x) + abs(t[1] - room.y) == 1 and g.grid[index(*t)] < 0
    assert found > 100


def test_ultra_secret_room_none():
    g = LevelGenerator(5)
    g.generate(15, False, False, False, 0x1FFF)
    before = state(g)
    assert g.get_new_ultra_secret_room(set(range(GRID * GRID))) is None
    assert state(g) == before                                # no free cell, no draw
    for i in range(GRID * GRID):
        g.block_position(i % GRID, i // GRID)
    assert g.get_new_ultra_secret_room() is None and state(g) == before
