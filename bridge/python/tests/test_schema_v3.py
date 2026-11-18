"""visible-v3's frozen artefacts (isaac_bridge.schema_v3; DESIGN.md F1 / F2 / F4): the numbers the design fixes, the
guards, the flag tables, the action code, the record formats; engine/abp_schema_v3.h equals its generator's output
(tools/gen_schema_v3_h.py) and compiles where a C compiler exists; the v2 path is untouched."""
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))

from isaac_bridge import schema_v3 as S   # noqa: E402
CRLF, LF = chr(13) + chr(10), chr(10)   # (a Windows checkout may hold the header with CRLF)


def test_f1_sizes():
    assert S.SCHEMA == 'isaac-visible-prlmens-g-v3' and S.VERSION == 2026111803
    assert (S.FIXED, S.PF, S.LOCAL, S.EF, S.NF, S.SF, S.BODY, S.RC) == (2417, 192, 405, 64, 8, 24, 597, 21)
    assert S.HEADER == 8 and S.MAGIC == 61107 and S.VERSION_WORD == 111803 and S.H_VERSION == 6
    assert not hasattr(S, 'LAYOUT_GEN') and not hasattr(S, 'H_GEN')   # (batch 5: one number, VERSION)
    assert S.frame_floats(1, 0, 0, 0) == 2417 + 597
    assert S.frame_floats(4, 128, 64, 32) * 4 < S.FRAME_BYTES   # (the design's worst case fits)
    for cols, width in ((S.P_COLS, S.PF), (S.E_COLS, S.EF), (S.N_COLS, S.NF), (S.S_COLS, S.SF)):
        assert len(cols) == width
        names = [c[0] for c in cols if c[1] != 'reserved']
        assert len(names) == len(set(names))
        assert all(c[1] == 'reserved' or c[3] > 0 for c in cols)


def test_f1_columns_the_design_names():
    P, E = S.P_COLS, S.E_COLS
    for name, c in (('x', 0), ('broken', 11), ('extra_lives', 13), ('giga_bombs', 17), ('damage', 23), ('dead', 32),
                    ('weapon_charge', 37), ('body_index', 38), ('linked_body', 39), ('stall_share', 40),
                    ('health_known', 48), ('weapon_charge_known', 51), ('stall_known', 53), ('ptype', 56),
                    ('stage', 60), ('stage_type', 61), ('game_version', 64), ('control_scheme', 65),
                    ('curse0', 72), ('black_slot0', 104), ('bone_slot0', 128), ('weapon0', 152),
                    ('overflow_interactable', 176), ('overflow_other', 180)):
        assert S.col(P, name) == c, name
    assert all(P[i][1] == 'reserved' for i in list(range(43, 48)) + [55] + list(range(66, 72)) + list(range(181, 192)))
    for name, c in (('type', 0), ('subtype', 2), ('kind', 15), ('npc_flags', 16), ('champion', 19), ('price', 31),
                    ('identity_known', 32), ('flags_known', 33), ('tear_w0', 34), ('tear_w7', 41), ('proj_w0', 42),
                    ('eflag_w0', 46), ('anim_id', 50)):
        assert S.col(E, name) == c, name
    assert all(E[i][1] == 'reserved' for i in range(51, 64))
    assert S.col(S.N_COLS, 'id') == 1 and S.N_COLS[1][1] == 'item:collectible'
    assert S.col(S.S_COLS, 'family') == 1 and S.col(S.S_COLS, 'pill_colour') == 11 and S.col(S.S_COLS, 'sub_charge_known') == 19
    assert sum(b for _, b in S.ROOM_W0) == 20 and sum(b for _, b in S.ROOM_W1) == 23
    assert sum(b for _, b in S.MAP_BITS) == 16
    assert max(sum(b for _, b in w) for w in (S.ROOM_W0, S.ROOM_W1, S.MAP_BITS)) <= 24   # (exact in float32)
    assert S.RC == 13 + 5 + 3 and S.ROOM_CHANNELS[9] == 'door_target'


def test_f1_guards():
    for v in (-1, -70000, 65535, 70000, float('nan')):
        assert S.identity_index(v) == S.IDENTITY_UNKNOWN
    assert S.identity_index(0) == 0 and S.identity_index(1023) == 1023
    for ns, (base, size) in S.ITEM_NS.items():
        assert S.item_index(ns, 0) == 0
        assert S.item_index(ns, 1) == base + 1 and S.item_index(ns, size - 1) == base + size - 1
        for v in (-1, size, 0x8000 + 3, float('nan')):
            assert S.item_index(ns, v) == S.ITEM_UNKNOWN, (ns, v)
    spans = sorted(S.ITEM_NS.values())
    assert all(a[0] + a[1] <= b[0] for a, b in zip(spans, spans[1:])) and spans[-1][0] + spans[-1][1] <= S.ITEM_UNKNOWN
    assert S.item_index('collectible', 732) == 732 and S.item_index('trinket', 189) == 2048 + 189
    assert S.item_index('card', 97) == 2560 + 97
    for ns, (base, size) in S.CAT_NS.items():
        assert S.cat_index(ns, 0) == 0 and S.cat_index(ns, 1) == base + 1
        assert S.cat_index(ns, 10 ** 6) == S.cat_index(ns, -1) == S.cat_index(ns, float('nan')) == base + size - 1
    spans = sorted(S.CAT_NS.values())
    assert all(a[0] + a[1] <= b[0] for a, b in zip(spans, spans[1:])) and spans[-1][0] + spans[-1][1] <= S.CAT_ROWS
    assert S.cat_index('ptype', 40 + 2) == 42   # (Rep's last PlayerType fits)
    assert S.map_icon_index(29) == 29 and S.map_icon_index(64) == 63 and S.map_icon_index(-1) == 63


def test_f1_flags():
    m = S.TEAR_AB_TO_CANON
    assert len(m) == 61 and len(set(m.values())) == 61 and all(m[b] == b for b in range(30))
    assert (m[30], m[31], m[33], m[34], m[42], m[43], m[44], m[59], m[60]) == (35, 30, 32, 36, 44, 33, 46, 61, 127)
    assert max(m.values()) < 16 * S.TEAR_WORDS
    assert S.tear_canonical(1 << 60) == 1 << 127 and S.tear_canonical(0b101) == 0b101
    assert S.flag_words(1 << 127, 8)[7] == 0x8000 and S.flag_words(0x12345, 2) == [0x2345, 0x1]
    assert not (S.PROJ_VISIBLE_MASK >> 32) & 0b11 and (S.PROJ_VISIBLE_MASK >> 31) & 1
    assert all((S.ENTITY_VISIBLE_MASK >> b) & 1 for b in (5, 12, 24, 29, 34, 58)) and not (S.ENTITY_VISIBLE_MASK >> 0) & 1


def test_f2_actions():
    assert S.HEADS == ('move', 'shoot', 'bomb', 'item', 'pill', 'drop') and S.HEAD_SIZES == (9, 5, 2, 2, 2, 2)
    assert (S.N_HEADS, S.N_LOGITS, S.CRN_COLS, S.N_ACTION_CODES) == (6, 22, 22, 720)
    seen = set()
    for m in range(9):
        for s in range(5):
            for b in range(2):
                for i in range(2):
                    for p in range(2):
                        for d in range(2):
                            c = S.action_code(m, s, b, i, p, d)
                            assert S.action_of_code(c) == (m, s, b, i, p, d)
                            seen.add(c)
    assert seen == set(range(720))
    assert S.action_code(3, 2, 1, 0, 1) == 3 + 18 + 45 + 180   # (drop 0: the v2 code)
    assert len(S.MOVE) == 9 and len(S.SHOOT) == 5 and S.BUTTON['drop'] == 'DROP' and S.BUTTON_ACTION_ABP['DROP'] == 11
    assert S.STEP_LINE_FIELDS[-1] == 'drop' and len(S.STEP_LINE_FIELDS) == 7


def test_f4_records():
    from isaac_bridge import tok_teacher2
    assert S.ACTION_F == 6 and S.V2_META == 31 and S.TEACH_META == 19
    assert S.V2_FIELDS[:12] == tuple(f'pend{k}' for k in range(6)) + tuple(f'act{k}' for k in range(6))
    # (v2's own fields after the action columns, unchanged and in order)
    assert S.V2_FIELDS[12:] == tok_teacher2.V2_FIELDS[10:]
    assert S.TEACH_FIELDS[6:15] == tuple(f'danger{k}' for k in range(9))


def test_header_is_the_generated_one():
    import gen_schema_v3_h as G
    assert G.OUT.read_bytes().decode().replace(CRLF, LF) == G.generate(), 'run: python tools/gen_schema_v3_h.py'


@pytest.mark.skipif(shutil.which('gcc') is None, reason='no C compiler here')
def test_header_compiles(tmp_path):
    import gen_schema_v3_h as G
    src = tmp_path / 'h.c'
    src.write_text('#include "abp_schema_v3.h"\nint main(void) { return ABP_V3_FIXED == 2417 && '
                   'abp_v3_head_sizes[5] == 2 && abp_v3_tear_ab_to_canon[60] == 127 ? 0 : 1; }\n')
    exe = tmp_path / 'h'
    subprocess.run(['gcc', '-Wall', '-Wextra', '-Werror', f'-I{G.OUT.parent}', '-o', str(exe), str(src)], check=True)
    assert subprocess.run([str(exe)]).returncode == 0


def test_v2_path_untouched():
    from isaac_bridge import structured_input as SI
    assert SI.SCHEMA == 'isaac-visible-prlmens-g-v2' and SI.VERSION == 2026110702
    assert (SI.FIXED, SI.PF, SI.EF, SI.NF, SI.SF, SI.RC) == (1969, 80, 48, 4, 13, 16)


def test_scale_vector():
    sv = S.scale_vector(S.P_COLS)
    assert len(sv) == S.PF and sv[0] == 520 and sv[S.col(S.P_COLS, 'stall_share')] == 1 and sv[S.col(S.P_COLS, 'ptype')] == 1
    assert all(math.isfinite(x) and x > 0 for x in sv)


def test_batch1_coordinate_contract():
    """Batch 1 (the coordinator, 2026-11-17; the coordinate contract, after V1): one canvas frame, isotropic scales,
    the laser's scales, P grid_w / grid_h, kind + 1."""
    P, E = S.P_COLS, S.E_COLS
    assert S.CELL == 40
    sp, se = S.scale_vector(P), S.scale_vector(E)
    for cols, sv, x, y in ((P, sp, 'x', 'y'), (E, se, 'x', 'y')):
        cx, cy = S.col(cols, x), S.col(cols, y)
        assert sv[cx] == sv[cy] == 520 and 'canvas-frame px' in cols[cx][2] and 'canvas-frame px' in cols[cy][2]
    assert sp[S.col(P, 'vx')] == sp[S.col(P, 'vy')] == se[S.col(E, 'vx')] == se[S.col(E, 'vy')] == 10
    assert (se[S.col(E, 'l_len')], se[S.col(E, 'l_ex')], se[S.col(E, 'l_ey')], se[S.col(E, 'l_radius')]) == (300, 520, 520, 100)
    assert 'canvas-frame px' in E[S.col(E, 'l_ex')][2] and 'canvas-frame px' in E[S.col(E, 'l_ey')][2]
    assert (S.col(P, 'grid_w'), S.col(P, 'grid_h')) == (41, 42) and (sp[41], sp[42]) == (28, 16)
    assert S.ENTITY_KIND == {'other': 1, 'tear': 2, 'projectile': 3, 'laser': 4, 'bomb': 5, 'pickup': 6, 'npc': 7,
                             'knife': 8, 'familiar': 9, 'effect': 10}
    assert 0 not in S.ENTITY_KIND.values() and max(S.ENTITY_KIND.values()) < S.CAT_NS['entity_kind'][1]
    assert len({S.cat_index('entity_kind', v) for v in S.ENTITY_KIND.values()}) == 10   # (no kind on the 'none' row)
    for cols in (P, E):   # (no column is relative to GetTopLeftPos any more)
        assert not any('top-left' in c[2] or 'TopLeft' in c[2] for c in cols)


def test_batch2_scales():
    """Batch 2 (the coordinator, 2026-11-17): the non-position scales B1-B4, NUM_CLAMP (B5), and NUM_RANGE (B6): one
    entry per 'num' column of every token kind, each None or an ordered (low, high)."""
    E = S.E_COLS
    se = S.scale_vector(E)
    for name, scale in (('size_mx', 1), ('size_my', 1), ('scale', 1), ('fall', 10), ('coll', 4), ('gcoll', 8),
                        ('cdmg', 2), ('anim', 8)):
        assert se[S.col(E, name)] == scale, name
    assert S.NUM_CLAMP == 8.0
    for key, cols in (('P', S.P_COLS), ('E', S.E_COLS), ('N', S.N_COLS), ('S', S.S_COLS)):
        nums = {c[0] for c in cols if c[1] == 'num'}
        assert set(S.NUM_RANGE[key]) == nums, (key, nums ^ set(S.NUM_RANGE[key]))
        for name, r in S.NUM_RANGE[key].items():
            assert r is None or (len(r) == 2 and r[0] < r[1] and -S.NUM_CLAMP <= r[0] and r[1] <= S.NUM_CLAMP), name
    P = S.P_COLS
    assert S.NUM_RANGE['P']['x'] == (0.0, 1120 / 520) and S.NUM_RANGE['P']['y'] == (0.0, 640 / 520)


def test_batch3_position_known():
    """Batch 3 (the coordinator, 2026-11-17): P col 54 position_known, a known flag; not a num column (no NUM_RANGE)."""
    P = S.P_COLS
    assert S.col(P, 'position_known') == 54 and P[54][1] == 'known' and S.VERSION >= 2026111703
    assert 'position_known' not in S.NUM_RANGE['P']

def test_batch4_categorical_direction_and_price():
    """Batch 4 (the coordinator, 2026-11-18; VERSION 2026111801): P 33-35 'cat:direction', E 31 'cat:price'; the column
    numbers stay; the four leave NUM_RANGE and the scale vector; the wire keeps the raw value (CAT_FROM_RAW: the decode
    makes the code)."""
    P, E = S.P_COLS, S.E_COLS
    for c, n in ((33, 'head_dir'), (34, 'fire_dir'), (35, 'move_dir')):
        assert S.col(P, n) == c and P[c][1] == 'cat:direction' and n not in S.NUM_RANGE['P']
        assert S.scale_vector(P)[c] == 1.0
    assert S.col(E, 'price') == 31 and E[31][1] == 'cat:price' and 'price' not in S.NUM_RANGE['E']
    assert S.scale_vector(E)[31] == 1.0
    assert S.CAT_NS['price'] == (400, 128) and S.CAT_NS['direction'] == (528, 8)
    spans = sorted(S.CAT_NS.values())
    assert all(a[0] + a[1] <= b[0] for a, b in zip(spans, spans[1:])) and spans[-1][0] + spans[-1][1] <= S.CAT_ROWS
    for kind in {c[1] for cols in (P, E, S.N_COLS, S.S_COLS) for c in cols if c[1].startswith('cat:')}:
        assert kind[4:] in S.CAT_NS, kind
    # direction: Direction -1..3 (enums.lua 1064) -> 1..5; anything else the last row
    assert [S.direction_code(v) for v in (-1, 0, 1, 2, 3)] == [1, 2, 3, 4, 5]
    assert S.direction_code(4) == S.direction_code(-2) == S.direction_code(float('nan')) == 7
    assert [S.wire_cat_index('direction', v) for v in (-1, 0, 3)] == [529, 530, 533]
    # price: AB+ PickupPrice (enums.lua 548-553) and the measured values (184 prices1)
    assert S.price_code(0) == 0
    for v in (2, 3, 5, 7, 15, 99):
        assert S.price_code(v) == v
    assert [S.price_code(v) for v in (-1, -2, -3, -1000)] == [100, 101, 102, 103]
    assert S.price_code(100) == S.price_code(-4) == S.price_code(float('nan')) == 127
    assert S.wire_cat_index('price', -1) == 500 and S.wire_cat_index('price', 15) == 415
    assert len({S.wire_cat_index('price', v) for v in (0, 2, 3, 5, 7, 15, -1, -2, -3, -1000, 100)}) == 11
    # every other namespace: the wire carries the code already
    assert S.wire_cat_index('ptype', 2) == S.cat_index('ptype', 2) and set(S.CAT_FROM_RAW) == {'price', 'direction'}


def test_batch4_rel_scale_python_only():
    """REL_SCALE (batch 4): the model's relative offsets / 200, distances / 300 (tok_obs' scales); not in the header."""
    assert S.REL_SCALE == {'offset': 200.0, 'distance': 300.0}
    header = (HERE.parents[1] / 'engine' / 'abp_schema_v3.h').read_text()
    assert 'REL_SCALE' not in header and f'#define ABP_V3_VERSION {S.VERSION}' in header
    assert '#define ABP_V3_CAT_NS_PRICE_BASE 400' in header and '#define ABP_V3_CAT_NS_DIRECTION_BASE 528' in header


def test_batch4_the_raw_block_carries_the_version():
    lua = (HERE.parents[1] / 'abplus' / 'abp_bridge.lua').read_text(encoding='utf-8')
    assert f'schema = "isaac-visible-raw-v3", version = {S.VERSION},' in lua


def test_batch5_the_header_word_is_the_version():
    """Batch 5 (the coordinator, 2026-11-18; VERSION 2026111802): header[6] = VERSION - 2026000000, exact in float32
    (integers to 2^24): the encoding holds through 2042 and is refused past it -- this test fails when VERSION outgrows
    it. The C header gets the word from the generator (abp_v3_native.c writes ABP_V3_VERSION_WORD, no number of its
    own)."""
    import numpy as np
    assert S.VERSION_WORD == S.VERSION - S.VERSION_BASE == S.version_word(S.VERSION)
    assert 0 < S.VERSION_WORD < 1 << 24 and float(np.float32(S.VERSION_WORD)) == S.VERSION_WORD
    assert S.version_word(2042123199) == 16123199 and float(np.float32(16123199)) == 16123199
    for past in (2043010101, 2026000000, 2025123101):
        with pytest.raises(ValueError):
            S.version_word(past)
    header = (HERE.parents[1] / 'engine' / 'abp_schema_v3.h').read_text()
    assert f'#define ABP_V3_VERSION_WORD {S.VERSION_WORD}' in header and '#define ABP_V3_H_VERSION 6' in header
    assert 'LAYOUT_GEN' not in header and 'H_GEN' not in header
    c = (HERE.parents[1] / 'engine' / 'abp_v3_native.c').read_text(encoding='utf-8')
    assert 'h[ABP_V3_H_VERSION] = ABP_V3_VERSION_WORD;' in c and str(S.VERSION_WORD) not in c


def test_price_coin_scale_is_model_side():
    """PRICE_COIN_SCALE (2026-11-18): the coins / 15 of the price number, model side only (no wire change: VERSION
    stays the batch's, nothing in the C header)."""
    assert S.PRICE_COIN_SCALE == 15.0 and S.VERSION == 2026111803
    header = (HERE.parents[1] / 'engine' / 'abp_schema_v3.h').read_text()
    assert 'PRICE_COIN' not in header
