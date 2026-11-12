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
    assert S.SCHEMA == 'isaac-visible-prlmens-g-v3' and S.VERSION == 2026111201
    assert (S.FIXED, S.PF, S.LOCAL, S.EF, S.NF, S.SF, S.BODY, S.RC) == (2417, 192, 405, 64, 8, 24, 597, 21)
    assert S.HEADER == 8 and S.MAGIC == 61107 and S.LAYOUT_GEN == 3
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
    assert all(P[i][1] == 'reserved' for i in list(range(41, 48)) + [54, 55] + list(range(66, 72)) + list(range(181, 192)))
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
