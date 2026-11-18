"""visible-v3 reference encoder (package C, 2026-11-17): visible_producer.observation_frame_v3 on a synthetic raw block
(abp_bridge.lua visible_v3_raw's JSON): the frame layout and size, positions in the canvas frame (world - origin, the
origin from the block), the cell rule floor(pos / 40) for doors and the local patch (clamped to the grid), kind + 1,
the flag words (AB+ TearFlags remapped, projectile / entity masks), Curse of the Unknown, the priority cap and its
overflow counts, the room words' bit packing, and the decorative list equal to the Lua one."""
import json
from types import SimpleNamespace
import re
from pathlib import Path

import numpy as np
import pytest

from isaac_bridge import schema_v3 as V3
from isaac_bridge import visible_producer as VP

BRIDGE = Path(__file__).resolve().parents[2]
O = (20.0, 100.0)


def body(**kw):
    b = dict(init_seed=1, x=O[0] + 100.0, y=O[1] + 60.0, vx=1.5, vy=-2.0, size=10, hearts=6, max_hearts=6, soul=2,
             bone=0, eternal=0, golden=0, broken=0, rotten=0, extra_lives=0, coins=7, bombs=1, keys=2, giga_bombs=0,
             golden_bomb=0, golden_key=0, poop_mana=0, soul_charge=0, blood_charge=0, damage=3.5, max_fire_delay=10,
             shot_speed=1, range=260, move_speed=1, luck=0, can_fly=0, invulnerable=0, controls=1, dead=0,
             head_dir=3, fire_dir=-1, move_dir=-1, sprite_frame=4, weapon_charge=0, weapon_charge_known=1, ptype=0,
             role=1, controller=0, controller_known=1, animation=1, animation_known=1, black=0b101, bone_bits=0,
             weapon_bits=0b10, bone_slots_known=1)
    b.update(kw)
    return b


def entity(**kw):
    e = dict(type=10, variant=0, subtype=0, x=O[0] + 300.0, y=O[1] + 200.0, vx=0, vy=0, size=13, size_mx=1, size_my=1,
             coll=4, gcoll=5, cdmg=1, sprite_frame=0, age=12, flip_x=0, identity_known=1, npc_flags=1 | 2 | 8,
             champion=1, boss_hp=0, height=0, fall=0, scale=0, l_cos=0, l_sin=0, l_len=0, l_ex=0, l_ey=0, l_radius=0,
             l_circle=0, price=0, tear=[0, 0, 0, 0], proj=[0, 0, 0, 0], flags_known=4, eflags=[0, 0, 0, 0], anim=0,
             kind=6)
    e.update(kw)
    return e


def raw_block(**kw):
    gw, gh = 15, 9
    cells = []
    for i in range(gw * gh):
        row, col = divmod(i, gw)
        inside = int(0 < row < gh - 1 and 0 < col < gw - 1)
        cells.append([inside, 0 if inside else 1, 0 if inside else 1, 0, 0, 0, 0 if inside else 15, 0, 0])
    cells[2 * gw + 3] = [1, 2, 1, 0, 1, 0, 2, 1, 0]          # a rock at (3, 2)
    cells[5 * gw + 7] = [1, 0, 0, 0, 0, 0, 17, 0, 0]          # a trapdoor (exit) at (7, 5)
    r = dict(schema='isaac-visible-raw-v3', version=V3.VERSION, game_version=1, control_scheme=0, origin=list(O),
             gw=gw, gh=gh, room_type=1, shape=1, stage=2, stage_type=1, difficulty=0, curses=0, here=60, frame=10,
             bodies=[body()], entities=[entity()], cells=cells,
             doors=[dict(slot=2, x=O[0] + 580.0, y=O[1] + 180.0, open=1, locked=0, target=4, visited=1, clear=1,
                         variant=1, state=2)],
             map=[[60, 1, 0, 1, 1, 1], [61, 4, 4, 0, 0, 0]],
             inventory=[[0, 105, 1]],
             slots=[dict(owner=0, family=1, slot=0, id=105, occupied=1, selected=1, charge=6, max_charge=6, battery=0)])
    r.update(kw)
    return r


def test_layout_and_positions():
    raw = raw_block()
    f = VP.unpack_frame_v3(VP.observation_frame_v3(json.dumps(raw), stall_share=0.25, stall_known=1))
    h = f['header']
    assert (int(h[0]), int(h[6]), int(h[7])) == (V3.MAGIC, V3.LAYOUT_GEN, 0)
    assert tuple(int(v) for v in h[2:6]) == (1, 1, 1, 1) and int(h[1]) == V3.frame_floats(1, 1, 1, 1)
    P, E = f['P'][0], f['E'][0]
    c = lambda cols, n: V3.col(cols, n)
    assert (P[c(V3.P_COLS, 'x')], P[c(V3.P_COLS, 'y')]) == (100.0, 60.0)          # world - origin, raw px
    assert (E[c(V3.E_COLS, 'x')], E[c(V3.E_COLS, 'y')]) == (300.0, 200.0)
    assert P[c(V3.P_COLS, 'vx')] == 1.5 and P[c(V3.P_COLS, 'stall_share')] == 0.25 and P[c(V3.P_COLS, 'stall_known')] == 1
    assert (P[41], P[42]) == (15, 9)
    assert P[c(V3.P_COLS, 'ptype')] == 2 and P[c(V3.P_COLS, 'controller')] == 2 and P[c(V3.P_COLS, 'stage_type')] == 2
    assert P[c(V3.P_COLS, 'difficulty')] == 1 and P[c(V3.P_COLS, 'game_version')] == 1
    assert [P[c(V3.P_COLS, f'black_slot{k}')] for k in range(3)] == [1, 0, 1] and P[c(V3.P_COLS, 'weapon1')] == 1
    assert E[c(V3.E_COLS, 'kind')] == 7 and E[c(V3.E_COLS, 'coll')] == 4 and E[c(V3.E_COLS, 'npc_flags')] == 11
    # the door at world (600, 280): canvas (580, 180) -> cell (14, 4); its words
    w0 = int(f['canvas'][4, 14, 0])
    bits = {}
    shift = 0
    for name, n in V3.ROOM_W0:
        bits[name] = (w0 >> shift) & ((1 << n) - 1)
        shift += n
    assert (bits['door'], bits['door_open'], bits['door_target'], bits['visited'], bits['clear']) == (1, 1, 4, 1, 1)
    w1 = int(f['canvas'][4, 14, 1])
    assert (w1 & 63, (w1 >> 6) & 15) == (1, 2)
    assert tuple(f['canvas'][2, 3, 2:]) == (2, 1, 0) and int(f['canvas'][2, 3, 0]) & 1 == 1
    assert (int(f['canvas'][5, 7, 0]) >> sum(n for name, n in V3.ROOM_W0[:-1])) & 1 == 1   # exit
    assert not f['canvas'][9:].any() and not f['canvas'][:, 15:].any()                    # outside the grid: 0
    # the local patch: centred on (floor(100 / 40), floor(60 / 40)) = (2, 1)
    assert np.array_equal(f['local'][0][4, 4], f['canvas'][1, 2]) and np.array_equal(f['local'][0][5, 5], f['canvas'][2, 3])
    assert not f['local'][0][:3, :].any()                                                 # rows above the canvas
    # minimap: room 60 current, room 61 (shape 4: cells 61, 74) icon 4
    m = f['minimap']
    assert int(m[60]) == 1 | 2 | 4 | 8 and int(m[61]) >> 4 == 4 and int(m[74]) >> 4 == 4


def test_local_patch_clamps_and_off_grid_entities():
    b = body(x=O[0] - 15.0, y=O[1] + 1000.0)     # left of the grid, far below: clamped to (0, 8)
    e = entity(x=O[0] + 500.0, y=O[1] + 367.0)   # 7 px below the grid (the Homunculus part of V1): sent, no cell
    f = VP.unpack_frame_v3(VP.observation_frame_v3(raw_block(bodies=[b], entities=[e])))
    assert np.array_equal(f['local'][0][4, 4], f['canvas'][8, 0])
    assert f['E'][0][V3.col(V3.E_COLS, 'y')] == 367.0
    assert VP.cell_of(500.0, 367.0, 15, 9) is None and VP.cell_of(-0.5, 10, 15, 9) is None
    assert VP.cell_of(0.0, 0.0, 15, 9) == (0, 0) and VP.cell_of(599.9, 359.9, 15, 9) == (14, 8)


def test_flags_and_laser():
    tear_ab = (1 << 3) | (1 << 30) | (1 << 60)                      # 3 -> 3, 30 -> 35, 60 -> 127 (LUDOVICO)
    words = lambda v: [(v >> (16 * k)) & 0xffff for k in range(4)]
    eflags = (1 << 5) | (1 << 4) | (1 << 29)                       # 4 is not a visible status
    proj = (1 << 1) | (1 << 32)                                     # 32 hidden
    laser = entity(type=7, kind=3, tear=words(tear_ab), eflags=words(eflags), proj=words(proj), l_ex=O[0] + 40.0,
                   l_ey=O[1] + 80.0, l_len=200, l_radius=0, l_cos=1.0)
    f = VP.unpack_frame_v3(VP.observation_frame_v3(raw_block(entities=[laser])))
    E = f['E'][0]
    tear = sum(int(E[V3.col(V3.E_COLS, f'tear_w{k}')]) << (16 * k) for k in range(8))
    assert tear == (1 << 3) | (1 << 35) | (1 << 127)
    assert sum(int(E[V3.col(V3.E_COLS, f'eflag_w{k}')]) << (16 * k) for k in range(4)) == (1 << 5) | (1 << 29)
    assert sum(int(E[V3.col(V3.E_COLS, f'proj_w{k}')]) << (16 * k) for k in range(4)) == 1 << 1
    assert (E[V3.col(V3.E_COLS, 'l_ex')], E[V3.col(V3.E_COLS, 'l_ey')]) == (40.0, 80.0) and E[V3.col(V3.E_COLS, 'kind')] == 4


def test_unknown_masks_health_and_cap_counts_overflow():
    ents = [entity(type=9, kind=2, npc_flags=0, champion=0, x=O[0] + i, y=O[1] + 50) for i in range(130)] + [entity(type=5, kind=5, npc_flags=0)]
    f = VP.unpack_frame_v3(VP.observation_frame_v3(raw_block(curses=8, entities=ents)))
    P = f['P'][0]
    for k in ('hearts', 'max_hearts', 'soul'):
        assert P[V3.col(V3.P_COLS, k)] == 0
    assert P[V3.col(V3.P_COLS, 'health_known')] == 0 and P[V3.col(V3.P_COLS, 'black_slot0')] == 0
    assert P[V3.col(V3.P_COLS, 'curse3')] == 1
    assert f['E'].shape[0] == V3.CAP_ENTITY and f['E'][0][V3.col(V3.E_COLS, 'type')] == 5   # the pickup first
    assert P[V3.col(V3.P_COLS, 'overflow_projectile')] == 3 and P[V3.col(V3.P_COLS, 'overflow_interactable')] == 0


def test_bits_that_do_not_fit_are_an_error():
    raw = raw_block()
    raw['doors'][0]['target'] = 64
    with pytest.raises(ValueError):
        VP.observation_frame_v3(raw)


def test_decorative_list_equals_the_bridges():
    lua = (BRIDGE / 'abplus' / 'abp_bridge.lua').read_text(encoding='utf-8')
    m = re.search(r'local V3_DECORATIVE_SPEC = \{(.*?)\n\}', lua.replace('\r\n', '\n'), re.S)
    lua_spec = {int(v): (n, why) for v, n, why in re.findall(r'\[(\d+)\] = "(\w+): (.*?)",', m.group(1))}
    assert lua_spec == VP.V3_DECORATIVE_WHY and tuple(sorted(lua_spec)) == VP.V3_DECORATIVE
    assert 41 in VP.V3_DECORATIVE and 114 in VP.V3_DECORATIVE      # the water droplet (V1), the Forgotten's chain
    assert 121 not in VP.V3_DECORATIVE and 138 not in VP.V3_DECORATIVE   # (uncertain: sent)
    assert all(why for _, why in VP.V3_DECORATIVE_WHY.values())


def test_a_body_without_a_room_position():
    """Schema batch 3: position_known 0 (the AB+ sub-player, never a room entity) -> x / y / vx / vy 0 and the local
    patch 0; the other fields as usual."""
    sub = body(init_seed=7, x=0.0, y=0.0, vx=3.0, vy=4.0, ptype=17, role=3, linked=1, position_known=0)
    main = body(linked=2, position_known=1)
    f = VP.unpack_frame_v3(VP.observation_frame_v3(raw_block(bodies=[main, sub])))
    P = f['P']
    c = lambda n: V3.col(V3.P_COLS, n)
    assert P[0][c('position_known')] == 1 and P[1][c('position_known')] == 0
    assert (P[1][c('x')], P[1][c('y')], P[1][c('vx')], P[1][c('vy')]) == (0, 0, 0, 0)
    assert not f['local'][1].any() and f['local'][0].any()
    assert P[1][c('ptype')] == 19 and P[1][c('body_role')] == 3 and P[1][c('hearts')] == 6


def test_engine_ranges_come_from_one_table():
    """The range convention (schema_v3.ABP_RANGES, the coordinator 2026-11-17): abp_bridge.lua's V3_ABP has the same
    values, and no loop of the v3 block has a bare-number bound (every one from V3_ABP or an engine count)."""
    lua = (BRIDGE / 'abplus' / 'abp_bridge.lua').read_text(encoding='utf-8').replace('\r\n', '\n')
    m = re.search(r'local V3_ABP = \{(.*?)\}', lua, re.S)
    table = {k: int(v) for k, v in re.findall(r'(\w+) = (\d+)', m.group(1))}
    assert table == {k: v for k, (v, _) in V3.ABP_RANGES.items()} and set(V3.REP_RANGES) == set(V3.ABP_RANGES)
    assert all(src for _, src in V3.ABP_RANGES.values()) and all('UNVERIFIED' in s for _, s in V3.REP_RANGES.values())
    start = lua.index('local V3_ABP = {')
    end = lua.index('\nend\n', lua.index('function visible_v3_raw()'))
    bare = re.findall(r'for \w+ = -?\d+, -?\d+ do', lua[start:end])
    assert not bare, bare


def test_priority_distance_ignores_bodies_without_a_position():
    """The 128-entity cap's distance (the coordinator, 2026-11-17): only bodies with a room position count. A Forgotten
    frame with its sub-player at world (0, 0) (position_known 0) and 140 projectiles: the kept 128 are the ones nearest
    the controlled body, whatever lies near world (0, 0)."""
    main = body(x=O[0] + 500.0, y=O[1] + 300.0, linked=2, position_known=1)
    sub = body(init_seed=7, x=0.0, y=0.0, ptype=17, role=3, linked=1, position_known=0)
    ents = [entity(type=9, kind=2, npc_flags=0, champion=0, x=float(i), y=float(i)) for i in range(70)]       # near (0, 0)
    ents += [entity(type=9, kind=2, npc_flags=0, champion=0, x=O[0] + 500.0 + i, y=O[1] + 300.0) for i in range(70)]
    kept, overflow = VP.v3_prioritize(ents, [main, sub])
    assert len(kept) == 128 and overflow[3] == 12
    near_main = [e for e in kept if e['x'] >= O[0] + 500.0]
    assert len(near_main) == 70 and kept[:70] == ents[70:]          # the main body's neighbours first, in their order
    only, _ = VP.v3_prioritize(ents, [main])
    assert kept == only                                              # the sub-player changes nothing
    none, _ = VP.v3_prioritize(ents, [sub])                          # no placed body: the bridge's order
    assert none == ents[:128]


def test_minimap_marks_only_the_rooms_cells():
    """schema_v3.MAP_SHAPE_CELLS (the engine's cells per shape, 184 mapcells1): an LTL room (shape 9) marks GridIndex +
    1, 13, 14 and not cell 0 (the missing quadrant); the table equals tok_obs.SHAPE_CELLS for every shape."""
    from isaac_bridge.tok_obs import SHAPE_CELLS
    assert {s: tuple(sorted(c)) for s, c in V3.MAP_SHAPE_CELLS.items()} == \
        {s: tuple(sorted(SHAPE_CELLS.get(s, (0,)))) for s in range(1, 13)}
    raw = raw_block(map=[[30, 9, 1, 1, 0, 0], [60, 1, 0, 1, 1, 1]])
    m = VP.unpack_frame_v3(VP.observation_frame_v3(raw))['minimap']
    assert m[30] == 0 and all(m[30 + d] != 0 for d in (1, 13, 14)) and m[60] != 0

def test_body_order_convention():
    """Batch 4: P row 0 is player 0's controlled body (position_known 1); each sub-player right after its own player.
    The reference encoder keeps the raw block's order (abp_bridge.lua v3_bodies makes it; abp_v3_native.c the same);
    v3_body_order finds a violation."""
    p0 = body(init_seed=1, linked=2)
    s0 = body(init_seed=2, x=0.0, y=0.0, ptype=17, role=3, linked=1, position_known=0)
    p1 = body(init_seed=3, ptype=3, controller=1)
    f = VP.unpack_frame_v3(VP.observation_frame_v3(raw_block(bodies=[p0, s0, p1])))
    assert VP.v3_body_order(f) == []
    c = lambda n: V3.col(V3.P_COLS, n)
    assert [int(r[c('body_role')]) for r in f['P']] == [1, 3, 1] and [int(r[c('body_index')]) for r in f['P']] == [0, 1, 2]
    # the old order (every player first, then the sub-players) is a violation
    old = [dict(p0, linked=3), dict(p1), dict(s0, linked=1)]
    assert VP.v3_body_order(VP.unpack_frame_v3(VP.observation_frame_v3(raw_block(bodies=old))))
    # row 0 without a room position is a violation
    assert VP.v3_body_order(VP.unpack_frame_v3(VP.observation_frame_v3(raw_block(bodies=[body(position_known=0)]))))


def test_the_bridge_and_the_native_encoder_put_the_sub_after_its_player():
    lua = (BRIDGE / 'abplus' / 'abp_bridge.lua').read_text(encoding='utf-8')
    fn = lua[lua.index('local function v3_bodies(game)'):lua.index('local function v3_entities')]
    assert fn.count('for i = 0, n - 1 do') == 1 and fn.index('Isaac.GetPlayer(i)') < fn.index('p:GetSubPlayer()')
    c = (BRIDGE / 'engine' / 'abp_v3_native.c').read_text(encoding='utf-8')
    fn = c[c.index('static int v3_bodies('):c.index('static void v3_body_row(')]
    assert fn.index('for (int i = 0; i < np; i++)') < fn.index('for (int pass = 0; pass < 2; pass++)')

def test_a_raw_block_of_another_version_is_refused():
    """observation_frame_v3 takes only a raw block of this tree's schema_v3.VERSION (2026-11-18, Codex round 1): a Lua
    / tool of another commit (e.g. batch 3, no position_known, which the encoder would read as 1) is an error naming
    both versions, and so is a block without one."""
    VP.observation_frame_v3(raw_block())
    old = raw_block(version=2026111703)
    with pytest.raises(ValueError, match='2026111703.*%d' % V3.VERSION):
        VP.observation_frame_v3(old)
    with pytest.raises(ValueError, match='2026111703'):
        VP.observation_frame_v3(json.dumps(old))
    none = raw_block()
    del none['version']
    with pytest.raises(ValueError, match='None.*%d' % V3.VERSION):
        VP.observation_frame_v3(none)
    with pytest.raises(ValueError, match='version'):     # (the worker's path: write_v3 of a handed-back raw block)
        from isaac_bridge.tok_obs import ROW
        VP.write_v3(np.zeros(1, ROW), SimpleNamespace(visible=old, visible_v3_frame=None))


def test_the_worker_checks_a_native_frames_header():
    """write_v3 (encode_row) checks a native frame (flag 64) before it goes into the ROW: magic, LAYOUT_GEN, the size
    word against the counts and exactly the bytes given (before 2026-11-18 it wrote the bytes unchecked)."""
    from isaac_bridge.tok_obs import ROW
    good = VP.observation_frame_v3(raw_block())
    row = np.zeros(1, ROW)
    assert VP.write_v3(row, SimpleNamespace(visible=None, visible_v3_frame=good))
    assert bytes(row['visible_frame'][0, :int(row['visible_bytes'][0])]) == good

    def edited(word, value):
        f = np.frombuffer(good, '<f4').copy()
        f[word] = value
        return f.tobytes()
    bad = {'magic': edited(V3.H_MAGIC, V3.MAGIC + 1), 'layout gen': edited(V3.H_GEN, V3.LAYOUT_GEN + 1),
           'size': edited(V3.H_SIZE, len(good) // 4 + 1), 'counts': edited(V3.H_E, 0 if VP.unpack_frame_v3(good)[
               'E'].shape[0] else 1), 'truncated': good[:-4], 'extra float': good + bytes(4),
           'partial float': good + bytes(2), 'header only part': good[:16]}
    for why, frame in bad.items():
        row = np.zeros(1, ROW)
        with pytest.raises(ValueError):
            VP.write_v3(row, SimpleNamespace(visible=None, visible_v3_frame=frame))
        assert int(row['visible_bytes'][0]) == 0, why
