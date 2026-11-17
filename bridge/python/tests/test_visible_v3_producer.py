"""visible-v3 reference encoder (package C, 2026-11-17): visible_producer.observation_frame_v3 on a synthetic raw block
(abp_bridge.lua visible_v3_raw's JSON): the frame layout and size, positions in the canvas frame (world - origin, the
origin from the block), the cell rule floor(pos / 40) for doors and the local patch (clamped to the grid), kind + 1,
the flag words (AB+ TearFlags remapped, projectile / entity masks), Curse of the Unknown, the priority cap and its
overflow counts, the room words' bit packing, and the decorative list equal to the Lua one."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from isaac_bridge import schema_v3 as V3
from isaac_bridge import visible_producer as VP

BRIDGE = Path(__file__).resolve().parents[2]
O = (20.0, 100.0)


def body(**kw):
    b = dict(entity_index=1, x=O[0] + 100.0, y=O[1] + 60.0, vx=1.5, vy=-2.0, size=10, hearts=6, max_hearts=6, soul=2,
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
    m = re.search(r'local V3_DECORATIVE = \{\}.*?ipairs\(\{(.*?)\}\)', lua, re.S)
    assert tuple(int(v) for v in re.findall(r'\d+', m.group(1))) == VP.V3_DECORATIVE
    assert 41 in VP.V3_DECORATIVE   # the water droplet (V1)


def test_a_body_without_a_room_position():
    """Schema batch 3: position_known 0 (the AB+ sub-player, never a room entity) -> x / y / vx / vy 0 and the local
    patch 0; the other fields as usual."""
    sub = body(entity_index=0, x=0.0, y=0.0, vx=3.0, vy=4.0, ptype=17, role=3, linked=1, position_known=0)
    main = body(linked=2, position_known=1)
    f = VP.unpack_frame_v3(VP.observation_frame_v3(raw_block(bodies=[main, sub])))
    P = f['P']
    c = lambda n: V3.col(V3.P_COLS, n)
    assert P[0][c('position_known')] == 1 and P[1][c('position_known')] == 0
    assert (P[1][c('x')], P[1][c('y')], P[1][c('vx')], P[1][c('vy')]) == (0, 0, 0, 0)
    assert not f['local'][1].any() and f['local'][0].any()
    assert P[1][c('ptype')] == 19 and P[1][c('body_role')] == 3 and P[1][c('hearts')] == 6
