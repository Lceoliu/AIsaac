"""seq-v1 step 1 (DRAFT 2026-11-18): isaac_bridge.v3_input on the 19 golden frames (tests/fixtures/visible_v3) and on
synthetic frames. CPU only.

- the entry refuses what it must not read: another visible_schema than schema_v3.VERSION (row by row), a bad magic,
  another schema version in header[6] (batch 5), a size off the formula, P row 0 not the controlled body with a
  position;
- every golden frame decodes; numbers are value / scale clamped to +-NUM_CLAMP; the derived inputs too;
- the main body is P row 0 and equals the engine's player position (engine.json, read apart) minus the origin;
- relative offsets / distances (C4) to the main body, recomputed here from the frame's raw pixels and REL_SCALE;
- C13 (forgotten-soul / forgotten): a body with position_known 0 gets no relative quantity, no in-cell offset, a pad
  local patch;
- batch 4: price (E 31) and the directions (P 33-35) are categorical in schema_v3, decoded by wire_cat_index; the
  price number is coins / PRICE_COIN_SCALE, the special prices are categories only;
- pad_batch: bucketed shapes and masks."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from isaac_bridge import schema_v3 as V3
from isaac_bridge import v3_input as VI
from isaac_bridge import visible_producer as VP
from isaac_bridge.visible_producer import VisibleV3Fatal

FIX = Path(__file__).resolve().parent / 'fixtures' / 'visible_v3'
NAMES = [f['name'] for f in json.loads((FIX / 'manifest.json').read_text())['frames']]
PC = lambda n: V3.col(V3.P_COLS, n)   # noqa: E731
EC = lambda n: V3.col(V3.E_COLS, n)   # noqa: E731


def payload(name):
    return (FIX / f'{name}.v3.bin').read_bytes()


def decode(buf):
    return VI.decode_frame(buf, V3.VERSION, len(buf))


def edit(buf, table, row, col, value):
    """A copy of the frame bytes with one P / E value changed."""
    f = np.frombuffer(buf, '<f4').copy()
    p, e = int(f[V3.H_P]), int(f[V3.H_E])
    at = V3.FIXED + (row * V3.PF if table == 'P' else p * V3.BODY + row * V3.EF)
    f[at + col] = value
    return f.tobytes()


def test_the_entry_refuses_what_it_must_not_read():
    buf = payload(NAMES[0])
    decode(buf)
    with pytest.raises(VisibleV3Fatal, match='visible_schema'):
        VI.decode_frame(buf, V3.VERSION - 1, len(buf))
    with pytest.raises(VisibleV3Fatal, match='visible_schema'):
        VI.decode_frame(buf, 0, len(buf))
    f = np.frombuffer(buf, '<f4').copy()
    f[V3.H_MAGIC] += 1
    with pytest.raises(VisibleV3Fatal, match='header'):
        decode(f.tobytes())
    for old in (2026111801, 2026111703):   # (batch 5: header[6] = VERSION_WORD; a frame of another schema refused)
        f = np.frombuffer(buf, '<f4').copy()
        f[V3.H_VERSION] = old - V3.VERSION_BASE
        with pytest.raises(VisibleV3Fatal, match='version'):
            decode(f.tobytes())
    assert np.frombuffer(buf, '<f4')[V3.H_VERSION] == V3.VERSION_WORD
    with pytest.raises(VisibleV3Fatal, match='size'):
        VI.decode_frame(buf + b'\0' * 4, V3.VERSION, len(buf) + 4)
    with pytest.raises(VisibleV3Fatal, match='whole frame'):
        VI.decode_frame(buf, V3.VERSION, len(buf) - 2)
    f = np.frombuffer(buf, '<f4').copy()
    f[V3.H_SIZE] += 1
    with pytest.raises(VisibleV3Fatal, match='size'):
        decode(f.tobytes())


def test_p_row_0_must_be_the_controlled_body_with_a_position():
    buf = payload(NAMES[0])
    with pytest.raises(VisibleV3Fatal, match='row 0'):
        decode(edit(buf, 'P', 0, PC('position_known'), 0))
    with pytest.raises(VisibleV3Fatal, match='row 0'):
        decode(edit(buf, 'P', 0, PC('body_role'), V3.BODY_ROLE['sub']))


@pytest.mark.parametrize('name', NAMES)
def test_golden_frames_decode(name):
    buf = payload(name)
    d = decode(buf)
    raw = VP.unpack_frame_v3(buf)
    eng = json.loads((FIX / f'{name}.engine.json').read_text())
    ox, oy = eng['g0'][0] - V3.CELL / 2, eng['g0'][1] - V3.CELL / 2
    # the main body = the engine's player 0 (read apart), in canvas pixels
    assert d['main_xy'] == pytest.approx([eng['players'][0]['x'] - ox, eng['players'][0]['y'] - oy], abs=1e-3)
    # numbers: value / scale clamped
    pl = VI.PLAN['P']
    want = np.clip(raw['P'][:, pl['num']] / pl['scale'], -V3.NUM_CLAMP, V3.NUM_CLAMP)
    assert np.array_equal(d['P_num'], want.astype(np.float32))
    for k, v in d.items():
        if v.dtype.kind == 'f':
            assert np.isfinite(v).all(), k
            if k not in ('room', 'local', 'map_bits', 'main_xy'):   # (main_xy: raw pixels, diagnostics)
                assert np.abs(v).max(initial=0) <= V3.NUM_CLAMP, k
    # C4: entities relative to the main body, recomputed from the frame's pixels
    m = raw['P'][0, [PC('x'), PC('y')]].astype(np.float64)
    for r, e in enumerate(raw['E']):
        dx, dy = e[EC('x')] - m[0], e[EC('y')] - m[1]
        exp = np.clip([dx / V3.REL_SCALE['offset'], dy / V3.REL_SCALE['offset'], math.hypot(dx, dy) / V3.REL_SCALE['distance']],
                      -V3.NUM_CLAMP, V3.NUM_CLAMP)
        assert d['E_rel'][r] == pytest.approx(exp, abs=1e-5)
    # the categorical rows stay inside their namespaces
    for j, ns in enumerate(VI.PLAN['P']['cat_ns']):
        base, size = V3.CAT_NS[ns]
        rows = d['P_cat'][:, j]
        assert ((rows == 0) | ((rows >= base) & (rows < base + size))).all(), ns
    assert (d['E_id'] >= 0).all() and (d['E_id'] < V3.IDENTITY_ROWS).all()
    assert (d['map_icon'] >= 0).all() and (d['map_icon'] < V3.MAP_ICONS).all()


@pytest.mark.parametrize('name', ['forgotten', 'forgotten-soul'])
def test_c13_a_body_without_a_position_takes_no_part(name):
    buf = payload(name)
    d, raw = decode(buf), VP.unpack_frame_v3(buf)
    known = raw['P'][:, PC('position_known')] == 1
    assert known[0]
    for i in np.flatnonzero(~known):
        assert d['P_rel_valid'][i] == 0 and not d['P_rel'][i].any() and not d['P_off'][i].any() and d['local_pad'][i]
    assert d['P_rel'][0] == pytest.approx([0, 0, 0]) and d['P_rel_valid'][0] == 1


def test_c6_offsets_from_the_patch_centre():
    buf = payload(NAMES[0])
    raw = VP.unpack_frame_v3(buf)
    x, y = raw['P'][0, PC('x')], raw['P'][0, PC('y')]
    gw, gh = int(raw['P'][0, PC('grid_w')]), int(raw['P'][0, PC('grid_h')])
    for nx, ny in ((x, y), (3.0, 5.0), (-30.0, 7.0), (V3.CELL * gw + 50.0, V3.CELL * gh - 1.0)):
        b = edit(edit(buf, 'P', 0, PC('x'), nx), 'P', 0, PC('y'), ny)
        c = (min(max(math.floor(nx / V3.CELL), 0), gw - 1), min(max(math.floor(ny / V3.CELL), 0), gh - 1))
        exp = [(nx - V3.CELL * c[0] - V3.CELL / 2) / V3.CELL, (ny - V3.CELL * c[1] - V3.CELL / 2) / V3.CELL]
        assert decode(b)['P_off'][0] == pytest.approx(exp, abs=1e-5)


def test_derived_inputs_are_clamped():
    buf = payload(NAMES[0])
    far = edit(edit(buf, 'E', 0, EC('x'), 1e6), 'E', 0, EC('y'), -1e6) if np.frombuffer(buf, '<f4')[V3.H_E] else None
    if far is not None:
        r = decode(far)['E_rel'][0]
        assert list(r) == [V3.NUM_CLAMP, -V3.NUM_CLAMP, V3.NUM_CLAMP]
    big = edit(buf, 'P', 0, PC('sprite_frame'), 1e9)
    j = VI.PLAN['P']['num'].index(PC('sprite_frame'))
    assert decode(big)['P_num'][0, j] == V3.NUM_CLAMP


def test_batch4_price_and_directions():
    assert V3.P_COLS[PC('head_dir')][1] == 'cat:direction' and V3.E_COLS[EC('price')][1] == 'cat:price'
    for n in ('head_dir', 'fire_dir', 'move_dir'):
        assert PC(n) in VI.PLAN['P']['cat'] and PC(n) not in VI.PLAN['P']['num']
    assert EC('price') in VI.PLAN['E']['cat'] and EC('price') not in VI.PLAN['E']['num']
    buf = payload('devilprice')
    d, raw = decode(buf), VP.unpack_frame_v3(buf)
    j = VI.PLAN['E']['cat'].index(EC('price'))
    base = V3.CAT_NS['price'][0]
    pr = raw['E'][:, EC('price')]
    assert (pr < 0).any()   # (the fixture's point: a devil deal)
    for r, v in enumerate(pr):
        assert d['E_cat'][r, j] == V3.wire_cat_index('price', v)
        if v < 0:
            assert d['E_price'][r, 0] == 0 and d['E_cat'][r, j] in [base + c for c in V3.PRICE_CODES.values()] + [base + 127]
        elif v > 0:
            assert d['E_price'][r, 0] == pytest.approx(min(v / VI.PRICE_COIN_SCALE, V3.NUM_CLAMP))
    jd = VI.PLAN['P']['cat'].index(PC('fire_dir'))
    assert d['P_cat'][0, jd] == V3.cat_index('direction', V3.direction_code(raw['P'][0, PC('fire_dir')]))


def test_room_token_centres():
    c = VI.room_centres()
    assert c.shape == (28, 2) and tuple(c[0]) == (2 * V3.CELL, 2 * V3.CELL) and tuple(c[-1]) == (26 * V3.CELL, 14 * V3.CELL)
    assert VI.R_ROWS * VI.R_BLOCK == V3.CANVAS_H and VI.R_COLS * VI.R_BLOCK == V3.CANVAS_W


def test_pad_batch():
    fr = [decode(payload(n)) for n in NAMES[:5]]
    b = VI.pad_batch(fr)
    ne = max(len(f['E_num']) for f in fr)
    assert b['E_num'].shape[1] == VI.bucket(ne, VI.E_BUCKETS) and b['P_num'].shape[1] == VI.P_MAX
    assert b['local'].shape[:2] == (5, VI.P_MAX) and b['room'].shape[0] == 5
    for i, f in enumerate(fr):
        assert b['E_mask'][i].sum() == len(f['E_num']) and b['P_mask'][i].sum() == len(f['P_num'])
        assert np.array_equal(b['E_rel'][i, :len(f['E_rel'])], f['E_rel'])


def test_rows_are_checked_one_by_one():
    from isaac_bridge.tok_obs import ROW
    rows = np.zeros(3, ROW)
    for i, n in enumerate(NAMES[:3]):
        VP.write_frame_v3(rows[i:i + 1], payload(n))
    assert len(VI.decode_rows(rows)) == 3
    rows['visible_schema'][1] = 2026111801   # (a frame of another schema in a ROW the v3 path reads: refused)
    with pytest.raises(VisibleV3Fatal, match='visible_schema'):
        VI.decode_rows(rows)


# ------------------------------------------------------------------------------------- the device side (torch)
def _same(h, t):
    import torch
    assert set(h) == set(t), set(h) ^ set(t)
    for k in h:
        a = torch.as_tensor(np.asarray(h[k]))
        b = t[k].cpu()
        assert a.dtype == b.dtype or (a.dtype == torch.bool and b.dtype == torch.bool), (k, a.dtype, b.dtype)
        assert a.shape == b.shape, (k, a.shape, b.shape)
        assert torch.equal(a, b), (k, (a != b).nonzero()[:5])


def _devices():
    import torch
    return ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('name', NAMES)
def test_device_decode_equals_host_decode(name):
    buf = payload(name)
    for dev in _devices():
        _same(decode(buf), VI.decode_frame_torch(buf, V3.VERSION, len(buf), device=dev))


def test_device_decode_equals_host_on_out_of_range_codes():
    rich = max(NAMES, key=lambda n: tuple(int(v) for v in np.frombuffer(payload(n), '<f4')[V3.H_E:V3.H_S + 1]))
    buf = payload(rich)
    f = np.frombuffer(buf, '<f4').copy()
    p, e, n, s = (int(v) for v in f[V3.H_P:V3.H_S + 1])
    assert e >= 4
    eat = V3.FIXED + p * V3.BODY
    odd = [np.nan, -1.0, -2.5, 150.0, 0.5, -1000.0, 99.0, 1e9, -0.5, 3.7]
    for r in range(e):
        v = odd[r % len(odd)]
        for c in VI.PLAN['E']['cat'] + VI.PLAN['E']['idn'] + VI.PLAN['E']['anim'] + VI.PLAN['E']['num']:
            f[eat + r * V3.EF + c] = v
    for r in range(p):   # (P's categorical / numeric columns; row 0's main-body columns kept)
        for c in VI.PLAN['P']['cat'] + VI.PLAN['P']['num']:
            if c not in (PC('body_role'), PC('x'), PC('y'), PC('grid_w'), PC('grid_h')):
                f[V3.FIXED + r * V3.PF + c] = odd[(r + c) % len(odd)]
    nat = eat + e * V3.EF
    for r in range(n):
        for c in VI.PLAN['N']['item'] + VI.PLAN['N']['num']:
            f[nat + r * V3.NF + c] = odd[(r + c) % len(odd)]
    sat = nat + n * V3.NF
    for r in range(s):
        for c in VI.PLAN['S']['item'] + VI.PLAN['S']['cat'] + VI.PLAN['S']['num']:
            f[sat + r * V3.SF + c] = odd[(r + 2 * c) % len(odd)]
    b = f.tobytes()
    for dev in _devices():
        _same(decode(b), VI.decode_frame_torch(b, V3.VERSION, len(b), device=dev))


def test_device_decode_refuses_like_the_host():
    buf = payload(NAMES[0])
    with pytest.raises(VisibleV3Fatal, match='visible_schema'):
        VI.decode_frame_torch(buf, V3.VERSION - 1, len(buf))
    with pytest.raises(VisibleV3Fatal, match='row 0'):
        VI.decode_frame_torch(edit(buf, 'P', 0, PC('position_known'), 0), V3.VERSION, len(buf))


# ------------------------------------------------------------------------------------- the constants (grep gate)
def test_no_contract_constant_written_in_the_decoder():
    """Every scale / clamp / table size the decoder uses comes from schema_v3 (contract C7): none of them as a number in
    v3_input.py's code (comments and docstrings aside). Allowed: PRICE_COIN_SCALE's one definition (to move into
    schema_v3), the bit widths of the bits8 / word16 kinds, the padded layout's inventory / slot rows, the room-token block layout (4 x 7 blocks of 4 cells, checked against the canvas below), the actor's
    bucket sizes."""
    import ast
    import re
    src = (Path(VI.__file__)).read_text(encoding='utf-8')
    tree = ast.parse(src)
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node) is not None:
            d = node.body[0]
            doc_lines.update(range(d.lineno, d.end_lineno + 1))
    banned = {V3.CELL, V3.REL_SCALE['offset'], V3.REL_SCALE['distance'], POS := VI.POS_SCALE, V3.NUM_CLAMP,
              V3.IDENTITY_UNKNOWN, V3.ITEM_UNKNOWN, V3.ANIM_ROWS, V3.MAP_ICONS, V3.MAP_ICONS - 1, V3.CANVAS_W,
              V3.CANVAS_H, V3.PF, V3.EF, V3.NF, V3.SF, 100, 1000}
    allowed = ('PRICE_COIN_SCALE = 15.0', 'R_ROWS, R_COLS, R_BLOCK = 4, 7, 4', 'E_BUCKETS = (16, 32, 64, 128)',
               'W8, W16 = 8, V3.FLAG_WORD_BITS', 'N_MAX_V3, S_MAX_V3 = 64, 32',
               'N_BUCKETS, S_BUCKETS = (8, 16, 32, N_MAX_V3), (8, 16, S_MAX_V3)')
    found = []
    for i, line in enumerate(src.splitlines(), 1):
        code = line.split('#', 1)[0]
        if i in doc_lines or not code.strip() or any(a in code for a in allowed):
            continue
        for m in re.finditer(r'(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])', code):
            if float(m.group(1)) in {float(b) for b in banned}:
                found.append((i, line.strip()))
    assert not found, found
    assert POS == 520 and VI.R_ROWS * VI.R_BLOCK == V3.CANVAS_H and VI.R_COLS * VI.R_BLOCK == V3.CANVAS_W


# ------------------------------------------------------------------------------------- C4 / V5 on synthetic positions
def test_relative_quantities_are_translation_free_and_isotropic():
    buf = payload(max(NAMES, key=lambda n: int(np.frombuffer(payload(n), '<f4')[V3.H_E])))
    f = np.frombuffer(buf, '<f4').copy()
    p, e = int(f[V3.H_P]), int(f[V3.H_E])
    eat = V3.FIXED + p * V3.BODY
    mx, my = 300.0, 200.0
    f[V3.FIXED + PC('x')], f[V3.FIXED + PC('y')] = mx, my
    ring = [(mx + 120 * math.cos(a), my + 120 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, e, endpoint=False)]
    for r, (x, y) in enumerate(ring):
        f[eat + r * V3.EF + EC('x')], f[eat + r * V3.EF + EC('y')] = x, y
    d0 = decode(f.tobytes())['E_rel']
    assert np.allclose(d0[:, 2], 120 / V3.REL_SCALE['distance'], atol=1e-5)   # (the same distance in every direction)
    assert np.allclose(np.hypot(d0[:, 0], d0[:, 1]) * V3.REL_SCALE['offset'], 120, atol=1e-3)
    g = f.copy()   # every position moved by the same offset: the relative quantities do not change
    g[V3.FIXED + PC('x')] += 37.0
    g[V3.FIXED + PC('y')] -= 23.0
    for r in range(e):
        g[eat + r * V3.EF + EC('x')] += 37.0
        g[eat + r * V3.EF + EC('y')] -= 23.0
    assert np.allclose(decode(g.tobytes())['E_rel'], d0, atol=1e-5)


# ------------------------------------------------------------------------------------- round 1 of the review (11-18)
def _frame(name):
    return np.frombuffer(payload(name), '<f4').copy()


def test_strict_header_and_exact_length():
    buf = payload(NAMES[0])
    for word, value in ((V3.H_P, 1.5), (V3.H_E, np.nan), (V3.H_SIZE, np.inf), (V3.H_MAGIC, V3.MAGIC + 0.5)):
        f = np.frombuffer(buf, '<f4').copy()
        f[word] = value
        with pytest.raises(VisibleV3Fatal):
            decode(f.tobytes())
        with pytest.raises(VisibleV3Fatal):
            VI.decode_frame_torch(f.tobytes(), V3.VERSION)
    with pytest.raises(VisibleV3Fatal):   # (trailing data: the plain entry takes the whole buffer as the frame)
        VI.decode_frame(buf + b'\0' * 4, V3.VERSION)
    with pytest.raises(VisibleV3Fatal, match='visible_schema'):
        VI.decode_frame(buf, V3.VERSION + 0.5)
    # e = 129 with a self-consistent size: past CAP_ENTITY
    f = np.frombuffer(buf, '<f4').copy()
    p, e, n, s = (int(v) for v in f[V3.H_P:V3.H_S + 1])
    extra = (V3.CAP_ENTITY + 1 - e) * V3.EF
    eat = V3.FIXED + p * V3.BODY + e * V3.EF
    g = np.concatenate([f[:eat], np.zeros(extra, '<f4'), f[eat:]])
    g[V3.H_E] = V3.CAP_ENTITY + 1
    g[V3.H_SIZE] = V3.frame_floats(p, V3.CAP_ENTITY + 1, n, s)
    with pytest.raises(VisibleV3Fatal):
        decode(g.tobytes())


def test_main_body_flags_must_be_exactly_1():
    buf = payload(NAMES[0])
    for col, v in ((PC('body_role'), 1.5), (PC('body_role'), 0.999), (PC('position_known'), 1.25),
                   (PC('position_known'), np.nan)):
        with pytest.raises(VisibleV3Fatal, match='row 0'):
            decode(edit(buf, 'P', 0, col, v))


def test_a_bad_frame_fails_the_whole_batch():
    from isaac_bridge.tok_obs import ROW
    rows = np.zeros(3, ROW)
    for i, n in enumerate(NAMES[:3]):
        f = _frame(n)
        if i == 1:
            f[V3.H_P] = 1.5   # (written by hand below: write_frame_v3 does not read the header)
        b = f.tobytes()
        rows['visible_bytes'][i] = len(b)
        rows['visible_frame'][i, :len(b)] = np.frombuffer(b, np.uint8)
        rows['visible_schema'][i] = V3.VERSION
    with pytest.raises(VisibleV3Fatal):
        VI.decode_rows(rows)


def _spec_values(spec, salt):
    """Per 'num' column of a schema spec (col, name, kind, unit, scale): a wire value scale * q with q non-zero,
    unsaturated and distinct per column -- the expectation built from the schema alone, not from v3_input.PLAN."""
    cols = sorted((c, sc) for c, _n, kind, _u, sc in spec if kind == 'num')
    q = [round(0.11 + 0.017 * i + salt, 6) for i in range(len(cols))]
    return cols, q


def test_every_numeric_column_scaled_once_from_the_schema():
    import torch
    spec = {'P': V3.P_SPEC, 'E': V3.E_SPEC, 'N': V3.N_SPEC, 'S': V3.S_SPEC}
    width = {'P': V3.PF, 'E': V3.EF, 'N': V3.NF, 'S': V3.SF}
    seen = set()
    for name in ('forgotten', 'devilprice'):   # (2 bodies, 1 entity, 0 / 6 slots; 1 body, 5 entities, 1 / 3)
        frame = _frame(name)
        p, e, n, s = (int(v) for v in frame[V3.H_P:V3.H_S + 1])
        at = {'P': V3.FIXED, 'E': V3.FIXED + p * V3.BODY}
        at['N'] = at['E'] + e * V3.EF
        at['S'] = at['N'] + n * V3.NF
        rows = {'P': p, 'E': e, 'N': n, 'S': s}
        expect = {}
        for t in 'PENS':
            cols, q = _spec_values(spec[t], salt=0.001 * 'PENS'.index(t))
            for r in range(rows[t]):
                for (c, sc), qq in zip(cols, q):
                    frame[at[t] + r * width[t] + c] = np.float32(qq * sc)
            # (input revision 2: v3_input.WIDEN's columns in their wider unit -- the only scale not the schema's)
            widen = [VI.WIDEN.get((t, nm), 1.0) for c_, nm, kind, _u, _s in sorted(spec[t]) if kind == 'num']
            expect[t] = (np.asarray(q, np.float64) / np.asarray(widen)).astype(np.float32)
        d = decode(frame.tobytes())
        t_ = VI.decode_frame_torch(frame.tobytes(), V3.VERSION)
        for t in 'PENS':
            got = d[f'{t}_num']
            assert got.shape == (rows[t], len(expect[t])), t
            for r in range(rows[t]):
                assert np.allclose(got[r], expect[t], rtol=1e-5, atol=1e-6), (t, r, got[r] - expect[t])
                seen.add(t)
            assert torch.equal(t_[f'{t}_num'], torch.as_tensor(got)), t
    assert seen == set('PENS')


def test_bucket_bounds():
    for n, b in ((0, 16), (1, 16), (16, 16), (17, 32), (32, 32), (33, 64), (64, 64), (65, 128), (128, 128)):
        assert VI.bucket(n, VI.E_BUCKETS) == b
    with pytest.raises(ValueError):
        VI.bucket(129, VI.E_BUCKETS)


def test_c6_offset_outside_the_grid_is_from_the_edge_cell_centre():
    buf = payload(NAMES[0])
    b = edit(edit(buf, 'P', 0, PC('x'), -30.0), 'P', 0, PC('y'), 60.0)
    assert list(decode(b)['P_off'][0]) == [-1.25, 0.0]   # (centre cell 0: (-30 - 20) / 40; y in cell 1: (60 - 60) / 40)


def test_category_edges_same_on_both_sides():
    f = _frame('devilprice')
    p, e, n, s = (int(v) for v in f[V3.H_P:V3.H_S + 1])
    eat = V3.FIXED + p * V3.BODY
    nat, sat = eat + e * V3.EF, eat + e * V3.EF + n * V3.NF
    odd = [np.float32(2.0 ** 63), np.inf, -np.inf, np.nan, np.float32(-2.0 ** 63)]
    for r in range(e):
        for c in VI.PLAN['E']['cat'] + VI.PLAN['E']['idn'] + VI.PLAN['E']['anim']:
            f[eat + r * V3.EF + c] = odd[(r + c) % len(odd)]
    f[nat + V3.col(V3.N_COLS, 'id')] = np.inf
    for r in range(s):
        f[sat + r * V3.SF + V3.col(V3.S_COLS, 'family')] = odd[r % len(odd)]
        f[sat + r * V3.SF + V3.col(V3.S_COLS, 'id')] = 5.0
    b = f.tobytes()
    h = decode(b)
    for dev in _devices():
        _same(h, VI.decode_frame_torch(b, V3.VERSION, device=dev))
    jk = VI.PLAN['E']['cat'].index(EC('kind'))
    base, size = V3.CAT_NS['entity_kind']
    assert ((h['E_cat'][:, jk] >= base) & (h['E_cat'][:, jk] < base + size)).all()
    assert (h['E_id'] >= 0).all() and (h['E_id'] <= V3.IDENTITY_UNKNOWN).all()
    assert h['N_item'][0, 0] == V3.ITEM_UNKNOWN and (h['S_item'][:, 0] == V3.ITEM_UNKNOWN).all()
    assert (h['E_anim'] >= 0).all() and (h['E_anim'] < V3.ANIM_ROWS).all()


def test_finite_codes_follow_the_schema_functions():
    vals = np.concatenate([np.arange(-6, 140, 0.5), [999, 1000, -1000, 4095, 4096, 65534, 65535, 1e9]]).astype(np.float32)
    for ns in V3.CAT_NS:
        got = VI.n_wire_cat_index(ns, vals)
        assert got.tolist() == [V3.wire_cat_index(ns, float(v)) for v in vals], ns
    assert VI.n_identity_index(vals).tolist() == [V3.identity_index(float(v)) for v in vals]
    for ns, (base, size) in V3.ITEM_NS.items():
        assert VI.n_item_index(base, size, vals).tolist() == [V3.item_index(ns, float(v)) for v in vals], ns
    assert VI.n_map_icon_index(vals).tolist() == [V3.map_icon_index(float(v)) for v in vals]


def test_nonfinite_counted_per_section_and_replaced():
    f = _frame('devilprice')
    p, e, n, s = (int(v) for v in f[V3.H_P:V3.H_S + 1])
    eat = V3.FIXED + p * V3.BODY
    nat, sat = eat + e * V3.EF, eat + e * V3.EF + n * V3.NF
    f[V3.FIXED + VI.PLAN['P']['bit'][0]] = np.nan            # (P: a bit column)
    f[V3.FIXED + VI.PLAN['P']['known'][0]] = np.inf          # (P: a known column)
    f[eat + EC('vx')] = np.nan
    f[eat + V3.EF + VI.PLAN['E']['bits8'][0]] = np.inf      # (E: a bit word)
    f[nat + V3.col(V3.N_COLS, 'count')] = np.nan
    f[sat + V3.col(V3.S_COLS, 'charge')] = -np.inf
    f[V3.HEADER + 3] = np.nan                                # (canvas: cell 0, word 3 = grid_variant)
    f[V3.FIXED + p * V3.PF + 7] = np.inf                     # (body 0's local patch)
    b = f.tobytes()
    d = decode(b)
    assert dict(zip(VI.SECTIONS, d['nonfinite'].tolist())) == {'P': 2, 'E': 2, 'N': 1, 'S': 1, 'canvas': 2}
    for k, v in d.items():
        if v.dtype.kind == 'f':
            assert np.isfinite(v).all(), k
    for dev in _devices():
        _same(d, VI.decode_frame_torch(b, V3.VERSION, device=dev))


def test_padded_body_rows_are_pad():
    fr = [decode(payload(n)) for n in NAMES[:3]]
    b = VI.pad_batch(fr)
    for i, f in enumerate(fr):
        k = len(f['P_num'])
        assert b['local_pad'][i, k:].all() and np.array_equal(b['local_pad'][i, :k], f['local_pad'])
    assert b['nonfinite'].shape == (3, len(VI.SECTIONS))


# ------------------------------------------------------------------------------------- the batched device decode
def _frames_matrix(bufs):
    W = max(len(b) for b in bufs) // 4 + 7   # (trailing floats past every frame: never read)
    F = np.full((len(bufs), W), np.nan, np.float32)
    for i, b in enumerate(bufs):
        F[i, :len(b) // 4] = np.frombuffer(b, '<f4')
    return F


def test_batched_device_decode_equals_pad_batch_of_host_decodes():
    import torch
    bufs = [payload(n) for n in NAMES]
    odd = _frame('devilprice')   # (the category / non-finite edge frame of the tests above, built again)
    p, e = int(odd[V3.H_P]), int(odd[V3.H_E])
    eat = V3.FIXED + p * V3.BODY
    for r in range(e):
        for c in VI.PLAN['E']['cat'] + VI.PLAN['E']['idn'] + VI.PLAN['E']['anim']:
            odd[eat + r * V3.EF + c] = [np.float32(2.0 ** 63), np.inf, np.nan, -np.inf][(r + c) % 4]
    odd[eat + EC('vx')] = np.nan
    bufs.append(odd.tobytes())
    F = _frames_matrix(bufs)
    counts, trunc = VI.check_batch(F[:, :8], [V3.VERSION] * len(bufs), [len(b) for b in bufs], VI.P_MAX, 64, 32)
    assert trunc == dict(n=0, s=0)
    host = VI.pad_batch([VI.decode_frame(b, V3.VERSION) for b in bufs], e_bucket=64, n_max=64, s_max=32)
    for dev in _devices():
        got = VI.decode_frames_device(torch.from_numpy(F).to(dev), counts, 64, 64, 32)
        assert set(got) == set(host)
        for k in host:
            a = torch.as_tensor(host[k])
            assert got[k].dtype == a.dtype and got[k].shape == a.shape, k
            assert torch.equal(got[k].cpu(), a), (dev, k)


def test_device_decode_makes_no_host_copies_after_its_first_call(monkeypatch):
    """(184, 2026-11-18: GraphActor's capture failed -- 'operation not permitted when stream is capturing' -- on the room
    centres' as_tensor; a Python list as an index is such a copy too.) After one call per device, a call with the
    counts on the device creates no tensor from host data: as_tensor / tensor raise, and every column list goes
    through v3_input._ix (index_select of a cached index)."""
    import torch
    bufs = [payload(n) for n in NAMES[:3]]
    F = _frames_matrix(bufs)
    counts, _ = VI.check_batch(F[:, :8], [V3.VERSION] * len(bufs), [len(b) for b in bufs], VI.P_MAX, 64, 32)
    for dev in _devices():
        Ft, ct = torch.from_numpy(F).to(dev), torch.from_numpy(counts).to(dev)
        want = VI.decode_frames_device(Ft, ct, 64, 64, 32, check=False)

        def no(*a, **k):
            raise AssertionError('a tensor made from host data inside the device decode')
        with monkeypatch.context() as m:
            m.setattr(torch, 'as_tensor', no)
            m.setattr(torch, 'tensor', no)
            got = VI.decode_frames_device(Ft, ct, 64, 64, 32, check=False)
        for k in want:
            assert torch.equal(got[k], want[k]), (dev, k)
    import inspect
    src = inspect.getsource(VI.decode_frames_device) + ''.join(
        inspect.getsource(f) for f in (VI._t_flags, VI._t_num, VI._t_room, VI.t_item_index))
    assert '[_PC[' not in src.replace('_ix(P, [_PC[', '').replace('_ix(P[:, 0], [_PC[', '') and "[:, plan[" not in src


def test_check_batch_refuses_and_counts():
    bufs = [payload(n) for n in NAMES[:4]]
    F = _frames_matrix(bufs)
    sizes = [len(b) for b in bufs]
    with pytest.raises(VisibleV3Fatal, match='visible_schema'):
        VI.check_batch(F[:, :8], [V3.VERSION, V3.VERSION - 1, V3.VERSION, V3.VERSION], sizes)
    with pytest.raises(VisibleV3Fatal):
        VI.check_batch(F[:, :8], [V3.VERSION] * 4, [s + 4 for s in sizes])   # (the record's length off the header)
    g = F.copy()
    g[2, V3.H_P] = 1.5
    with pytest.raises(VisibleV3Fatal):
        VI.check_batch(g[:, :8], [V3.VERSION] * 4, sizes)
    counts, trunc = VI.check_batch(F[:, :8], [V3.VERSION] * 4, sizes, VI.P_MAX, 0 + 1, 2)
    assert trunc['n'] == int((counts[:, 2] > 1).sum()) and trunc['s'] == int((counts[:, 3] > 2).sum())


def test_widened_columns_are_the_listed_ones_only():
    """(input revision 2) v3_input.WIDEN names three numeric columns; every other column keeps the schema's scale."""
    assert set(VI.WIDEN) == {('E', 'age'), ('S', 'charge'), ('S', 'max_charge')} and VI.INPUT_REV == 2
    # (Codex round 12: the expectations independent of WIDEN -- age in game minutes: FrameCount / 1800, 30 logic frames
    #  a game second; charges in 120ths)
    for t, name, want in (('E', 'age', 1800.0), ('S', 'charge', 120.0), ('S', 'max_charge', 120.0)):
        cols = V3.E_COLS if t == 'E' else V3.S_COLS
        assert VI.PLAN[t]['scale'][VI.PLAN[t]['num'].index(V3.col(cols, name))] == np.float32(want), name
    for t, cols in (('P', V3.P_COLS), ('E', V3.E_COLS), ('N', V3.N_COLS), ('S', V3.S_COLS)):
        for j, i in enumerate(VI.PLAN[t]['num']):
            name, _kind, _u, sc = cols[i]
            assert VI.PLAN[t]['scale'][j] == np.float32((sc or 1.0) * VI.WIDEN.get((t, name), 1.0)), (t, name)
