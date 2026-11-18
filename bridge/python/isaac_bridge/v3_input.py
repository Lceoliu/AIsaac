"""visible-v3 decode for the model (package A, seq-v1 step 1). The only place where a v3 frame (schema_v3, the frozen
interface) becomes model inputs; the only place where the scales apply (contract C7). Host side (numpy): the actor
decodes outside its captured graph; device side (torch): the learner decodes on its own device. The two give the same
values (tests/test_v3_input.py).

Per frame (decode_frame / decode_frame_torch):
  - the record's visible_schema must equal schema_v3.VERSION exactly (row by row: the audit found a worker could write
    a v2 frame into a ROW configured for v3), and the header must pass visible_producer.check_frame_header_v3 with the
    exact length (every header word a finite whole number, magic, header[6] = VERSION_WORD, counts in range, the size
    formula, no trailing data); P row 0 must be the controlled body with a position (batch 4: body_role and
    position_known exactly 1) -- else VisibleV3Fatal (a protocol error, never retried, never a silent decode);
  - numeric ('num') columns: value / scale, clamped to [-NUM_CLAMP, NUM_CLAMP] (batch 2, B5);
  - 'known' / 'bit' columns: as they are (0 / 1); 'bits8' / 'word16': expanded into bits;
  - categorical columns -> rows of the shared tables, schema_v3's rules (cat_index / wire_cat_index -- price and
    direction coded from their raw wire values, batch 4 --, identity_index, item_index per family, map_icon_index, anim
    rows);
  - price (E col 31) also as a number: coins / PRICE_COIN_SCALE when > 0, else 0 (the coordinator, 11-18: the old
    model's /15; the special prices -1 / -2 / -3 / -1000 are categories only);
  - 'reserved' columns dropped.
Non-finite wire values (none by the wire rule; a producer bug otherwise turns every loss into NaN), the same on both
sides (the coordinator, 11-18): a numeric column NaN -> 0, +-inf -> +-NUM_CLAMP; a known / bit column, a bit word, a
canvas word NaN / inf -> 0; a categorical value NaN / inf -> its unknown row (cat: the namespace's last row; identity /
item: the unknown row; map icon: the last row; anim: 0); finite codes past +-2^31 are cut there before the integer
conversion (so 2^63 never wraps to a negative row). Each frame carries the count of its non-finite floats per section
(nonfinite: P, E, N, S, canvas = room canvas + minimap + local patches); V4's acceptance fails on any count above 0.
Derived inputs (the model's, never on the wire), every one clamped to [-NUM_CLAMP, NUM_CLAMP] (a NaN to 0):
  - C4 relative offsets / distances to the main body (P row 0): entities, other bodies, laser end points, the 28
    room-token centres; offsets / REL_SCALE['offset'], distances / REL_SCALE['distance'] (schema_v3, batch 4);
    computed from the raw canvas-frame pixels;
  - C6 P_off: each body's offset from the centre of its local patch's centre cell, in cells: (pos - CELL * c - CELL /
    2) / CELL with c = floor(pos / CELL) clamped into the grid (the producer's centre cell). Inside the grid within
    +-0.5; outside it larger (x = -30 px: -1.25) -- how far the body is from the patch's centre (the coordinator,
    11-18: not cut to +-0.5);
  - C13: a body with position_known 0 takes no part: its relative quantities and P_off are 0 with rel_valid 0, its
    local patch is marked pad.
Outputs are numpy arrays (see DecodedFrame); padded batches for the actor come from pad_batch."""
import numpy as np

from . import schema_v3 as V3
from .visible_producer import VisibleV3Fatal, check_frame_header_v3

# The coins scale of the price number (the coordinator 11-18: the old model's /15; to move into schema_v3 with the next
# batch -- until then the one model-side constant here, on the grep gate's whitelist).
PRICE_COIN_SCALE = 15.0
CLAMP = V3.NUM_CLAMP
W8, W16 = 8, V3.FLAG_WORD_BITS   # (the bits of a 'bits8' / a 'word16' column: schema_v3's kinds)
R_ROWS, R_COLS, R_BLOCK = 4, 7, 4   # the 28 room tokens: 4 x 7 blocks of 4 x 4 canvas cells (v2's down-sampling)
CODE_CUT = 2.0 ** 31   # a finite categorical / word value is cut to +-CODE_CUT before its integer conversion
SECTIONS = ('P', 'E', 'N', 'S', 'canvas')   # the nonfinite counts' order

_LASER_KIND = V3.ENTITY_KIND['laser']   # E col 15's code on the wire (raw kind + 1)
POS_SCALE = V3.P_COLS[V3.col(V3.P_COLS, 'x')][3]   # (the canvas-frame positions' scale, P / E x and y: 520)
_MAIN = V3.BODY_ROLE['main']


# ----------------------------------------------------------------------------------------------- column plans
def _plan(cols, table):
    """Per token type: the column indices of each decoded group, in column order."""
    g = dict(num=[], scale=[], known=[], bit=[], bits8=[], word16=[], cat=[], cat_ns=[], idn=[], item=[], item_ns=[],
             anim=[])
    for i, (name, kind, _unit, scale) in enumerate(cols):
        if kind == 'num':
            g['num'].append(i)
            g['scale'].append(scale or 1.0)
        elif kind in ('known', 'bit'):
            g[kind].append(i)
        elif kind == 'bits8':
            g['bits8'].append(i)
        elif kind == 'word16':
            g['word16'].append(i)
        elif kind.startswith('cat:'):
            g['cat'].append(i)
            g['cat_ns'].append(kind[4:])
        elif kind == 'id':
            g['idn'].append(i)
        elif kind.startswith('item:'):
            g['item'].append(i)
            g['item_ns'].append(kind[5:])
        elif kind == 'anim':
            g['anim'].append(i)
        elif kind != 'reserved':
            raise ValueError(f'v3_input: column {table} {i} {name}: unknown kind {kind}')
    g['scale'] = np.asarray(g['scale'], np.float32)
    return g


PLAN = {'P': _plan(V3.P_COLS, 'P'), 'E': _plan(V3.E_COLS, 'E'), 'N': _plan(V3.N_COLS, 'N'), 'S': _plan(V3.S_COLS, 'S')}
_PC = {n: V3.col(V3.P_COLS, n) for n in ('x', 'y', 'body_role', 'position_known', 'grid_w', 'grid_h')}
_EC = {n: V3.col(V3.E_COLS, n) for n in ('x', 'y', 'kind', 'l_ex', 'l_ey', 'price')}
_SC_FAMILY = V3.col(V3.S_COLS, 'family')
_FAMILY_NS = sorted((k, V3.ITEM_NS[ns]) for k, ns in V3.FAMILY_ITEM_NS.items())   # family code -> (base, size)


# ----------------------------------------------------------------------------------------------- frame bytes
def _refuse(msg):
    raise VisibleV3Fatal(f'v3_input: {msg}')


def split_frame(buf, visible_schema, nbytes=None):
    """The frame bytes of a record -> dict of float32 views (header, canvas [16, 28, 5], minimap [169], P [p, 192],
    local [p, 9, 9, 5], E [e, 64], N [n, 8], S [s, 24]). nbytes: the record's frame length (default: the whole
    buffer); the frame must fill it exactly. VisibleV3Fatal on another schema version, a bad header or a size that does
    not match."""
    vs = np.asarray(visible_schema).reshape(-1)
    if vs.size != 1 or not np.isfinite(float(vs[0])) or float(vs[0]) != V3.VERSION:
        _refuse(f'visible_schema {vs.tolist()} is not schema_v3.VERSION {V3.VERSION}')
    mv = memoryview(buf).cast('B')
    n = len(mv) if nbytes is None else int(nbytes)
    if n % 4 or n > len(mv) or n < 0:
        _refuse(f'{n} bytes is not a whole frame of floats ({len(mv)} in the buffer)')
    f = np.frombuffer(mv, '<f4', count=n // 4)
    p, e, nn, s = check_frame_header_v3(f, exact=True)
    at, out = V3.HEADER, {'header': f[:V3.HEADER]}
    for name, count, shape in (('canvas', 1, (V3.CANVAS_H, V3.CANVAS_W, V3.ROOM_WORDS)),
                               ('minimap', 1, (V3.MAP_SIDE * V3.MAP_SIDE,)), ('P', p, (V3.PF,)),
                               ('local', p, (V3.LOCAL_SIDE, V3.LOCAL_SIDE, V3.ROOM_WORDS)), ('E', e, (V3.EF,)),
                               ('N', nn, (V3.NF,)), ('S', s, (V3.SF,))):
        k = count * int(np.prod(shape))
        out[name] = f[at:at + k].reshape(shape if name in ('canvas', 'minimap') else (count,) + shape)
        at += k
    P = out['P']
    if not (P[0, _PC['body_role']] == _MAIN and P[0, _PC['position_known']] == 1):
        _refuse('P row 0 is not the controlled body with a position (schema_v3 batch 4: body_role and position_known '
                f'exactly 1, got {float(P[0, _PC["body_role"]])}, {float(P[0, _PC["position_known"]])})')
    return out


# ----------------------------------------------------------------------------------------------- host (numpy)
def _n_sane(v):
    """float64 values, cut to +-CODE_CUT, and the non-finite mask (those set 0)."""
    v = np.asarray(v, np.float64)
    bad = ~np.isfinite(v)
    return np.where(bad, 0.0, np.clip(v, -CODE_CUT, CODE_CUT)), bad


def _n_trunc(v):
    return np.trunc(v).astype(np.int64)


def n_cat_index(ns, code, bad):
    """schema_v3.cat_index on sanitised codes (bad: non-finite -> the namespace's last row)."""
    base, size = V3.CAT_NS[ns]
    c = _n_trunc(code)
    row = np.where(c == 0, 0, base + np.minimum(c, size - 1))
    return np.where(bad | (code < 0), base + size - 1, row)


def n_price_code(v, bad):
    """schema_v3.price_code."""
    size = V3.CAT_NS['price'][1]
    c = _n_trunc(v)
    out = np.full(c.shape, size - 1, np.int64)
    out = np.where(c == 0, 0, out)
    out = np.where((c >= 1) & (c <= 99), c, out)
    for raw, code in V3.PRICE_CODES.items():
        out = np.where(c == raw, code, out)
    return np.where(bad, size - 1, out).astype(np.float64)


def n_direction_code(v, bad):
    """schema_v3.direction_code."""
    size = V3.CAT_NS['direction'][1]
    return np.where(bad | (v < -1) | (v > 3), size - 1, _n_trunc(v) + 2).astype(np.float64)


_N_FROM_RAW = {'price': n_price_code, 'direction': n_direction_code}


def n_wire_cat_index(ns, wire):
    """schema_v3.wire_cat_index on an array (non-finite -> the last row)."""
    v, bad = _n_sane(wire)
    f = _N_FROM_RAW.get(ns)
    if f is None and ns in V3.CAT_FROM_RAW:
        raise KeyError(f'v3_input: no array form of schema_v3.CAT_FROM_RAW[{ns!r}]')
    if f is not None:
        return n_cat_index(ns, f(v, bad), np.zeros_like(bad))
    return n_cat_index(ns, v, bad)


def n_identity_index(wire):
    v, bad = _n_sane(wire)
    return np.where(bad | (v < 0) | (v >= V3.IDENTITY_UNKNOWN), V3.IDENTITY_UNKNOWN, _n_trunc(v))


def n_item_index(base, size, wire):
    """schema_v3.item_index with the namespace's (base, size) as arrays or ints (size 0: no namespace -> unknown)."""
    v, bad = _n_sane(wire)
    c = _n_trunc(v)
    size = np.asarray(size, np.int64)
    row = np.where((c > 0) & (c < size), np.asarray(base, np.int64) + c, V3.ITEM_UNKNOWN)
    row = np.where(c == 0, 0, row)
    return np.where(bad | (size == 0), V3.ITEM_UNKNOWN, row)


def n_map_icon_index(wire):
    v, bad = _n_sane(wire)
    return np.where(bad | (v < 0), V3.MAP_ICONS - 1, np.minimum(_n_trunc(v), V3.MAP_ICONS - 1))


def _n_words(w):
    """bit words as int64: non-finite -> 0, cut to [0, CODE_CUT]."""
    w = np.asarray(w, np.float64)
    return np.where(np.isfinite(w), np.clip(w, 0, CODE_CUT), 0).astype(np.int64)


def _bits(words, layout):
    """[..., ] float words -> [..., len(layout)] the unsigned fields (name, bits) LSB first, as float32."""
    w = _n_words(words)
    out, shift = [], 0
    for _name, bits in layout:
        out.append((w >> shift) & ((1 << bits) - 1))
        shift += bits
    return np.stack(out, -1).astype(np.float32)


_ROOM_DIV = np.asarray([V3.ROOM_DECODE_SCALE.get(n, 1) for n in V3.ROOM_CHANNELS], np.float32)


def _n_fin0(x):
    return np.where(np.isfinite(x), x, 0).astype(np.float32)


def room_channels(cells):
    """[..., 5] cell words -> [..., RC] the 21 decoded channels (word 0's 13 fields, word 1's 5, the raw words 2..4),
    divided by ROOM_DECODE_SCALE (codes / 64 etc.; others 1)."""
    ch = np.concatenate([_bits(cells[..., 0], V3.ROOM_W0), _bits(cells[..., 1], V3.ROOM_W1),
                         _n_fin0(cells[..., 2:5])], -1)
    return ch / _ROOM_DIV


def minimap(words):
    """[169] map words -> ([169, 4] shown / visited / clear / current, [169] icon rows)."""
    b = _bits(words, V3.MAP_BITS)
    return b[:, :4], n_map_icon_index(b[:, 4])


def _num(x, plan):
    return np.clip(np.nan_to_num(x[:, plan['num']] / plan['scale'], nan=0.0, posinf=CLAMP, neginf=-CLAMP),
                   -CLAMP, CLAMP).astype(np.float32)


def _flags(x, plan):
    n = len(x)
    parts = [_n_fin0(x[:, plan['known']]), _n_fin0(x[:, plan['bit']])]
    if plan['bits8']:
        parts.append(_bits(x[:, plan['bits8']], (('b', 1),) * W8).reshape(n, W8 * len(plan['bits8'])))
    if plan['word16']:
        parts.append(_bits(x[:, plan['word16']], (('b', 1),) * W16).reshape(n, W16 * len(plan['word16'])))
    return np.concatenate([np.asarray(q, np.float32) for q in parts], 1)


def _cats(x, plan):
    cols = [n_wire_cat_index(ns, x[:, c]) for c, ns in zip(plan['cat'], plan['cat_ns'])]
    return np.stack(cols, 1).astype(np.int64) if cols else np.zeros((len(x), 0), np.int64)


def _ids(x, plan):
    cols = [n_identity_index(x[:, c]) for c in plan['idn']]
    return np.stack(cols, 1).astype(np.int64) if cols else np.zeros((len(x), 0), np.int64)


def _anims(x, plan):
    v, bad = _n_sane(x[:, plan['anim']])
    return np.where(~bad & (v >= 0) & (v < V3.ANIM_ROWS), _n_trunc(v), 0).astype(np.int64)


def _family_ns(family):
    v, bad = _n_sane(family)
    fam = _n_trunc(v)
    base, size = np.zeros_like(fam), np.zeros_like(fam)
    for k, (b_, s_) in _FAMILY_NS:
        hit = (fam == k) & ~bad
        base, size = np.where(hit, b_, base), np.where(hit, s_, size)
    return base, size


def _items(x, plan, family=None):
    cols = []
    for c, ns in zip(plan['item'], plan['item_ns']):
        if ns == 'by_family':
            base, size = _family_ns(family)
        else:
            base, size = V3.ITEM_NS[ns]
        cols.append(n_item_index(base, size, x[:, c]))
    return np.stack(cols, 1).astype(np.int64) if cols else np.zeros((len(x), 0), np.int64)


def _clamp(v):
    return np.clip(np.nan_to_num(v, nan=0.0, posinf=CLAMP, neginf=-CLAMP), -CLAMP, CLAMP).astype(np.float32)


def _nonfinite(f):
    """The non-finite floats per section (SECTIONS order)."""
    c = lambda a: int(np.count_nonzero(~np.isfinite(a)))   # noqa: E731
    return np.asarray([c(f['P']), c(f['E']), c(f['N']), c(f['S']), c(f['canvas']) + c(f['minimap']) + c(f['local'])],
                      np.int64)


class DecodedFrame(dict):
    """One frame's model inputs (numpy). Keys:
      room [16, 28, RC]           canvas channels; room_xy [28, 2] the room tokens' centres / POS_SCALE; room_rel
                                  [28, 3] their offsets to the main body / REL offset, distance / REL distance
      map_bits [169, 4], map_icon [169]
      P_num, P_flag, P_cat        per body; P_off [p, 2] (C6, see the module doc); P_rel [p, 3]; P_rel_valid [p]
      local [p, 9, 9, RC], local_pad [p] (C13: position_known 0)
      E_num, E_flag, E_cat, E_id, E_anim; E_price [e, 1]; E_rel [e, 3]; E_laser_rel [e, 3] (0 for a non-laser)
      N_num, N_flag, N_item; S_num, S_flag, S_cat, S_item
      nonfinite [5] the non-finite wire floats per section (SECTIONS)
      main_xy [2] the main body's canvas-frame pixels (diagnostics)"""


def room_centres():
    """[28, 2] canvas-frame pixel centres of the 28 room tokens (block row-major): CELL * (R_BLOCK * col + R_BLOCK / 2,
    R_BLOCK * row + R_BLOCK / 2)."""
    r, c = np.divmod(np.arange(R_ROWS * R_COLS), R_COLS)
    return np.stack([V3.CELL * (R_BLOCK * c + R_BLOCK / 2), V3.CELL * (R_BLOCK * r + R_BLOCK / 2)], 1).astype(np.float64)


_ROOM_CENTRES = room_centres()


def _rel(xy, main):
    """[k, 2] canvas pixels -> [k, 3] (dx / REL offset, dy / REL offset, distance / REL distance), clamped."""
    d = np.asarray(xy, np.float64) - main
    dist = np.sqrt((d ** 2).sum(-1, keepdims=True))
    return _clamp(np.concatenate([d / V3.REL_SCALE['offset'], dist / V3.REL_SCALE['distance']], -1))


def _off(xy, gw, gh):
    """C6: [p, 2] offsets (in cells) from the centre of the producer's local-patch centre cell."""
    cell = np.stack([np.clip(np.floor(xy[:, 0] / V3.CELL), 0, gw - 1), np.clip(np.floor(xy[:, 1] / V3.CELL), 0, gh - 1)],
                    1)
    return _clamp((xy - V3.CELL * cell - V3.CELL / 2) / V3.CELL)


def _grid(P):
    g = np.nan_to_num(P[0, [_PC['grid_w'], _PC['grid_h']]].astype(np.float64), nan=1.0)
    return int(np.clip(g[0], 1, V3.CANVAS_W)), int(np.clip(g[1], 1, V3.CANVAS_H))


def decode_frame(buf, visible_schema, nbytes=None):
    """A record's v3 frame -> DecodedFrame (see the module doc). VisibleV3Fatal on a frame this decoder must not read."""
    f = split_frame(buf, visible_schema, nbytes)
    P, E, N, S = f['P'], f['E'], f['N'], f['S']
    main = P[0, [_PC['x'], _PC['y']]].astype(np.float64)
    known = P[:, _PC['position_known']] == 1
    out = DecodedFrame()
    out['main_xy'] = main.astype(np.float32)
    out['nonfinite'] = _nonfinite(f)
    out['room'] = room_channels(f['canvas'])
    out['room_xy'] = (_ROOM_CENTRES / POS_SCALE).astype(np.float32)
    out['room_rel'] = _rel(_ROOM_CENTRES, main)
    out['map_bits'], out['map_icon'] = minimap(f['minimap'])
    # bodies
    out['P_num'], out['P_flag'], out['P_cat'] = _num(P, PLAN['P']), _flags(P, PLAN['P']), _cats(P, PLAN['P'])
    xy = P[:, [_PC['x'], _PC['y']]].astype(np.float64)
    gw, gh = _grid(P)
    out['P_off'] = np.where(known[:, None], _off(xy, gw, gh), 0).astype(np.float32)
    out['P_rel'] = np.where(known[:, None], _rel(xy, main), 0).astype(np.float32)
    out['P_rel_valid'] = known.astype(np.float32)
    out['local'] = room_channels(f['local'])
    out['local_pad'] = ~known
    # entities
    out['E_num'], out['E_flag'], out['E_cat'] = _num(E, PLAN['E']), _flags(E, PLAN['E']), _cats(E, PLAN['E'])
    out['E_id'], out['E_anim'] = _ids(E, PLAN['E']), _anims(E, PLAN['E'])
    pr = E[:, _EC['price']]
    out['E_price'] = _clamp(np.where(pr > 0, pr / PRICE_COIN_SCALE, 0.0))[:, None]
    out['E_rel'] = _rel(E[:, [_EC['x'], _EC['y']]], main)
    laser = (E[:, _EC['kind']] == _LASER_KIND)[:, None]
    out['E_laser_rel'] = np.where(laser, _rel(E[:, [_EC['l_ex'], _EC['l_ey']]], main), 0).astype(np.float32)
    # inventory, slots
    out['N_num'], out['N_flag'], out['N_item'] = _num(N, PLAN['N']), _flags(N, PLAN['N']), _items(N, PLAN['N'])
    out['S_num'], out['S_flag'], out['S_cat'] = _num(S, PLAN['S']), _flags(S, PLAN['S']), _cats(S, PLAN['S'])
    out['S_item'] = _items(S, PLAN['S'], family=S[:, _SC_FAMILY])
    return out


def decode_rows(rows):
    """ROW records -> [DecodedFrame], each row's visible_schema checked on its own (decode_row); one bad row fails the
    whole call."""
    return [decode_row(rows[i:i + 1]) for i in range(len(rows))]


def decode_row(row):
    """A ROW record (length-1 view or a structured scalar) -> DecodedFrame."""
    n = int(np.asarray(row['visible_bytes']).reshape(-1)[0])
    buf = np.asarray(row['visible_frame']).reshape(-1)[:n]
    return decode_frame(buf.tobytes(), np.asarray(row['visible_schema']).reshape(-1)[0], n)


# ----------------------------------------------------------------------------------------------- device (torch)
# The same decode in torch, rule for rule (each t_* function mirrors the n_* one above). The header is read on the host
# (split_frame: the same checks); everything after it runs in torch on `device`.
def _t():
    import torch
    return torch


_DEVC = {}


def _dc(key, value, device, dtype=None):
    """A constant (a column list, a scale table, the room centres) as a tensor on `device`, made once per device and
    kept: inside a captured CUDA graph no host-to-device copy may run (a Python list as an index is one), and
    GraphActor's warm-up calls before each capture make every one of them. (184, 2026-11-18: the first capture failed
    with 'operation not permitted when stream is capturing' at the room centres' as_tensor.)"""
    torch = _t()
    k = (key, str(device), dtype)
    t = _DEVC.get(k)
    if t is None:
        t = _DEVC[k] = torch.as_tensor(np.asarray(value), device=device, dtype=dtype)
    return t


def _ix(x, cols):
    """x[..., cols] for a list of columns, the index from _dc (no copy from the host)."""
    torch = _t()
    return x.index_select(-1, _dc(('ix', tuple(int(c) for c in cols)), list(cols), x.device, torch.int64))


def _t_sane(v):
    torch = _t()
    v = v.to(torch.float64)
    bad = ~torch.isfinite(v)
    return torch.where(bad, torch.zeros_like(v), torch.clamp(v, -CODE_CUT, CODE_CUT)), bad


def _t_div(x, c):
    """x / c with c as a tensor of x's shape: a true division on every device (CUDA divides by a Python scalar as a
    multiplication by its reciprocal, which can differ from numpy's division in the last bit)."""
    return x / _t().full_like(x, float(c))


def _t_trunc(v):
    torch = _t()
    return torch.trunc(v).to(torch.int64)


def t_cat_index(ns, code, bad):
    torch = _t()
    base, size = V3.CAT_NS[ns]
    c = _t_trunc(code)
    row = torch.where(c == 0, torch.zeros_like(c), base + torch.clamp(c, max=size - 1))
    return torch.where(bad | (code < 0), torch.full_like(c, base + size - 1), row)


def t_price_code(v, bad):
    torch = _t()
    size = V3.CAT_NS['price'][1]
    c = _t_trunc(v)
    out = torch.full_like(c, size - 1)
    out = torch.where(c == 0, torch.zeros_like(c), out)
    out = torch.where((c >= 1) & (c <= 99), c, out)
    for raw, code in V3.PRICE_CODES.items():
        out = torch.where(c == raw, torch.full_like(c, code), out)
    return torch.where(bad, torch.full_like(c, size - 1), out).to(torch.float64)


def t_direction_code(v, bad):
    torch = _t()
    size = V3.CAT_NS['direction'][1]
    c = _t_trunc(v)
    return torch.where(bad | (v < -1) | (v > 3), torch.full_like(c, size - 1), c + 2).to(torch.float64)


_T_FROM_RAW = {'price': t_price_code, 'direction': t_direction_code}


def t_wire_cat_index(ns, wire):
    v, bad = _t_sane(wire)
    f = _T_FROM_RAW.get(ns)
    if f is None and ns in V3.CAT_FROM_RAW:
        raise KeyError(f'v3_input: no tensor form of schema_v3.CAT_FROM_RAW[{ns!r}]')
    if f is not None:
        return t_cat_index(ns, f(v, bad), _t().zeros_like(bad))
    return t_cat_index(ns, v, bad)


def t_identity_index(wire):
    torch = _t()
    v, bad = _t_sane(wire)
    c = _t_trunc(v)
    return torch.where(bad | (v < 0) | (v >= V3.IDENTITY_UNKNOWN), torch.full_like(c, V3.IDENTITY_UNKNOWN), c)


def t_item_index(base, size, wire):
    torch = _t()
    v, bad = _t_sane(wire)
    c = _t_trunc(v)
    if not torch.is_tensor(base):   # (a namespace's two ints: kernels, not a copy from the host)
        base, size = torch.full_like(c, int(base)), torch.full_like(c, int(size))
    unk = torch.full_like(c, V3.ITEM_UNKNOWN)
    row = torch.where((c > 0) & (c < size), base + c, unk)
    row = torch.where(c == 0, torch.zeros_like(c), row)
    return torch.where(bad | (size == 0), unk, row)


def t_map_icon_index(wire):
    torch = _t()
    v, bad = _t_sane(wire)
    c = _t_trunc(v)
    return torch.where(bad | (v < 0), torch.full_like(c, V3.MAP_ICONS - 1), torch.clamp(c, max=V3.MAP_ICONS - 1))


def _t_words(w):
    torch = _t()
    w = w.to(torch.float64)
    return torch.where(torch.isfinite(w), torch.clamp(w, 0, CODE_CUT), torch.zeros_like(w)).to(torch.int64)


def _t_bits(words, layout):
    torch = _t()
    w = _t_words(words)
    out, shift = [], 0
    for _name, bits in layout:   # (each field to float32 at once: no [..., fields] int64 stack -- 184's learner, 8 GiB)
        out.append(((w >> shift) & ((1 << bits) - 1)).to(torch.float32))
        shift += bits
    return torch.stack(out, -1)


def _t_fin0(x):
    torch = _t()
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x)).to(torch.float32)


def _t_room(cells):
    torch = _t()
    ch = torch.cat([_t_bits(cells[..., 0], V3.ROOM_W0), _t_bits(cells[..., 1], V3.ROOM_W1),
                    _t_fin0(cells[..., 2:5])], -1)
    return ch / _dc('room_div', _ROOM_DIV, cells.device)


def _t_flags(x, plan):
    torch = _t()
    n = x.shape[0]
    parts = [_t_fin0(_ix(x, plan['known'])), _t_fin0(_ix(x, plan['bit']))]
    if plan['bits8']:
        parts.append(_t_bits(_ix(x, plan['bits8']), (('b', 1),) * W8).reshape(n, W8 * len(plan['bits8'])))
    if plan['word16']:
        parts.append(_t_bits(_ix(x, plan['word16']), (('b', 1),) * W16).reshape(n, W16 * len(plan['word16'])))
    return torch.cat([q.to(torch.float32) for q in parts], 1)


def _t_clamp(v):
    torch = _t()
    return torch.clamp(torch.nan_to_num(v, nan=0.0, posinf=CLAMP, neginf=-CLAMP), -CLAMP, CLAMP).to(torch.float32)


def _t_num(x, plan):
    return _t_clamp(_ix(x, plan['num']) / _dc(('scale', tuple(plan['num']), tuple(map(float, plan['scale']))),
                                              plan['scale'], x.device))


def _t_stack(cols, n, device):
    torch = _t()
    return torch.stack(cols, 1) if cols else torch.zeros((n, 0), dtype=torch.int64, device=device)


def _t_cats(x, plan):
    return _t_stack([t_wire_cat_index(ns, x[:, c]) for c, ns in zip(plan['cat'], plan['cat_ns'])], x.shape[0], x.device)


def _t_family_ns(family):
    torch = _t()
    v, bad = _t_sane(family)
    fam = _t_trunc(v)
    base, size = torch.zeros_like(fam), torch.zeros_like(fam)
    for k, (b_, s_) in _FAMILY_NS:
        hit = (fam == k) & ~bad
        base = torch.where(hit, torch.full_like(fam, b_), base)
        size = torch.where(hit, torch.full_like(fam, s_), size)
    return base, size


def _t_items(x, plan, family=None):
    cols = []
    for c, ns in zip(plan['item'], plan['item_ns']):
        base, size = _t_family_ns(family) if ns == 'by_family' else V3.ITEM_NS[ns]
        cols.append(t_item_index(base, size, x[:, c]))
    return _t_stack(cols, x.shape[0], x.device)


def _t_rel(xy, main):
    torch = _t()
    d = xy.to(torch.float64) - main
    dist = torch.sqrt((d ** 2).sum(-1, keepdim=True))
    return _t_clamp(torch.cat([_t_div(d, V3.REL_SCALE['offset']), _t_div(dist, V3.REL_SCALE['distance'])], -1))


def decode_frame_torch(buf, visible_schema, nbytes=None, device='cpu'):
    """decode_frame on `device` in torch: the same keys, the same values (float32 / int64 / bool tensors)."""
    torch = _t()
    host = split_frame(buf, visible_schema, nbytes)
    nonfinite = _nonfinite(host)
    f = {k: torch.as_tensor(np.array(v), device=device) for k, v in host.items()}
    P, E, N, S = f['P'], f['E'], f['N'], f['S']
    main = P[0, [_PC['x'], _PC['y']]].to(torch.float64)
    known = P[:, _PC['position_known']] == 1
    centres = torch.as_tensor(_ROOM_CENTRES, device=device)
    out = {'main_xy': main.to(torch.float32), 'nonfinite': torch.as_tensor(nonfinite, device=device),
           'room': _t_room(f['canvas']), 'room_xy': _t_div(centres, POS_SCALE).to(torch.float32),
           'room_rel': _t_rel(centres, main)}
    mb = _t_bits(f['minimap'], V3.MAP_BITS)
    out['map_bits'], out['map_icon'] = mb[:, :4], t_map_icon_index(mb[:, 4])
    out['P_num'], out['P_flag'], out['P_cat'] = _t_num(P, PLAN['P']), _t_flags(P, PLAN['P']), _t_cats(P, PLAN['P'])
    xy = P[:, [_PC['x'], _PC['y']]].to(torch.float64)
    gw, gh = _grid(host['P'])
    cxy = torch.floor(_t_div(xy, V3.CELL))
    cell = torch.stack([torch.clamp(cxy[:, 0], 0, gw - 1), torch.clamp(cxy[:, 1], 0, gh - 1)], 1)
    off = _t_clamp(_t_div(xy - V3.CELL * cell - V3.CELL / 2, V3.CELL))
    out['P_off'] = torch.where(known[:, None], off, torch.zeros_like(off))
    rel = _t_rel(xy, main)
    out['P_rel'] = torch.where(known[:, None], rel, torch.zeros_like(rel))
    out['P_rel_valid'] = known.to(torch.float32)
    out['local'] = _t_room(f['local'])
    out['local_pad'] = ~known
    out['E_num'], out['E_flag'], out['E_cat'] = _t_num(E, PLAN['E']), _t_flags(E, PLAN['E']), _t_cats(E, PLAN['E'])
    out['E_id'] = _t_stack([t_identity_index(E[:, c]) for c in PLAN['E']['idn']], E.shape[0], device)
    v, bad = _t_sane(E[:, PLAN['E']['anim']])
    out['E_anim'] = torch.where(~bad & (v >= 0) & (v < V3.ANIM_ROWS), _t_trunc(v), torch.zeros_like(_t_trunc(v)))
    pr = E[:, _EC['price']]
    out['E_price'] = _t_clamp(torch.where(pr > 0, _t_div(pr, PRICE_COIN_SCALE), torch.zeros_like(pr)))[:, None]
    out['E_rel'] = _t_rel(E[:, [_EC['x'], _EC['y']]], main)
    lrel = _t_rel(E[:, [_EC['l_ex'], _EC['l_ey']]], main)
    out['E_laser_rel'] = torch.where((E[:, _EC['kind']] == _LASER_KIND)[:, None], lrel, torch.zeros_like(lrel))
    out['N_num'], out['N_flag'], out['N_item'] = _t_num(N, PLAN['N']), _t_flags(N, PLAN['N']), _t_items(N, PLAN['N'])
    out['S_num'], out['S_flag'], out['S_cat'] = _t_num(S, PLAN['S']), _t_flags(S, PLAN['S']), _t_cats(S, PLAN['S'])
    out['S_item'] = _t_items(S, PLAN['S'], family=S[:, _SC_FAMILY])
    return out


# ----------------------------------------------------------------------------------------------- batches (actor)
E_BUCKETS = (16, 32, 64, 128)
P_MAX = 4   # the actor's padded layout: bodies per frame (AB+ single player: 1-2); a frame with more is refused here
#            (the wire allows schema_v3.CAP_BODY; this is the first arm's graph shape, not a wire limit)


def bucket(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f'v3_input: {n} past the largest bucket {buckets[-1]}')


_PAD_TRUE = ('local_pad',)   # (a padded body row is pad: its local patch too)


def pad_batch(frames, e_bucket=None, n_max=None, s_max=None):
    """[DecodedFrame] -> dict of arrays [B, max, ...] with masks *_mask (True = a real token). e padded to its bucket
    (E_BUCKETS), bodies to P_MAX; n / s to the batch maximum unless given (their buckets wait for V4); a padded body's
    local_pad is True."""
    B = len(frames)
    e_bucket = e_bucket or bucket(max([len(f['E_num']) for f in frames] + [1]), E_BUCKETS)
    n_max = n_max or max([len(f['N_num']) for f in frames] + [1])
    s_max = s_max or max([len(f['S_num']) for f in frames] + [1])
    sizes = {'P': P_MAX, 'local': P_MAX, 'E': e_bucket, 'N': n_max, 'S': s_max}
    out = {}
    for k, v in frames[0].items():
        group = 'local' if k.startswith('local') else k.split('_')[0]
        if group in sizes:
            m = sizes[group]
            a = np.full((B, m) + v.shape[1:], k in _PAD_TRUE, v.dtype)
            for i, f in enumerate(frames):
                if len(f[k]) > m:
                    raise ValueError(f'v3_input: {k} has {len(f[k])} rows, the padded layout {m}')
                a[i, :len(f[k])] = f[k]
            out[k] = a
        else:
            out[k] = np.stack([f[k] for f in frames])
    for g in ('P', 'E', 'N', 'S'):
        lens = np.asarray([len(f[g + '_num']) for f in frames])
        out[g + '_mask'] = np.arange(sizes[g])[None, :] < lens[:, None]
    return out


# ----------------------------------------------------------------------------------------------- batches on the device
# (seq-v1 steps 3-4) The whole padded batch decoded on the device from the frames' floats -- the learner's batches and
# the actor's graphs. The headers are checked on the host first (check_batch: the same strict check per record, from
# the 8 header words and the record's byte count); the device then gathers each record's rows at its own offsets and
# decodes them with the t_* rules above. decode_frames_device(...) equals pad_batch([decode_frame(...) ...]) with the
# same e_bucket / n_max / s_max value for value (tests/test_v3_input.py).
def check_batch(headers, versions, sizes, p_max=None, n_max=None, s_max=None):
    """Host check of a batch: headers [B, >= 8] float32 (each record's first 8 words), versions [B] (visible_schema),
    sizes [B] (visible_bytes). Every record must pass split_frame's checks (VisibleV3Fatal otherwise). Returns
    (counts [B, 4] int64 p, e, n, s, truncated: dict(n=records with more inventory rows than n_max, s=... slots))."""
    headers = np.asarray(headers, np.float32)
    versions, sizes = np.asarray(versions), np.asarray(sizes)
    counts = np.zeros((len(headers), 4), np.int64)
    for i in range(len(headers)):
        vs = versions[i]
        if not np.isfinite(float(vs)) or float(vs) != V3.VERSION:
            _refuse(f'visible_schema {vs} is not schema_v3.VERSION {V3.VERSION} (record {i})')
        n = int(sizes[i])
        if n % 4 or n < V3.HEADER * 4:
            _refuse(f'{n} bytes is not a whole frame of floats (record {i})')
        f = np.zeros(n // 4, np.float32)
        f[:V3.HEADER] = headers[i, :V3.HEADER]
        f[1] = headers[i, 1]
        counts[i] = check_frame_header_v3(f, exact=True)
    if p_max is not None and (counts[:, 0] > p_max).any():
        _refuse(f'{int(counts[:, 0].max())} bodies, the padded layout holds {p_max}')
    trunc = dict(n=int((counts[:, 2] > n_max).sum()) if n_max else 0, s=int((counts[:, 3] > s_max).sum()) if s_max else 0)
    return counts, trunc


def _t_rows(F, base, stride, m, width, count):
    """[B, m, width] rows gathered from frame floats F [B, W] at base + i * stride (i < count; others 0) and the
    validity mask [B, m]."""
    torch = _t()
    dev = F.device
    i = torch.arange(m, device=dev)
    valid = i[None, :] < count[:, None]
    idx = base[:, None, None] + i[None, :, None] * stride + torch.arange(width, device=dev)[None, None, :]
    idx = torch.where(valid[..., None], idx, torch.zeros_like(idx)).clamp_(0, F.shape[1] - 1)
    rows = F.gather(1, idx.view(F.shape[0], -1)).view(F.shape[0], m, width)
    return torch.where(valid[..., None], rows, torch.zeros_like(rows)), valid


def decode_frames_device(frames, counts, e_bucket, n_max, s_max, p_max=None, check=True):
    """frames: float32 [B, W] on the device (each record's frame floats from its start; W >= every frame); counts [B, 4]
    from check_batch (numpy, or a device tensor: then nothing reads the device back -- check=False inside a captured
    graph; the caller has checked the bucket). Returns the pad_batch dict (tensors on frames' device)."""
    torch = _t()
    p_max = p_max or P_MAX
    F = frames.to(torch.float32)
    B, dev = F.shape[0], F.device
    c = (counts if torch.is_tensor(counts) else torch.as_tensor(np.asarray(counts))).to(dev, torch.int64)
    p, e = c[:, 0], torch.clamp(c[:, 1], max=e_bucket)
    n, s = torch.clamp(c[:, 2], max=n_max), torch.clamp(c[:, 3], max=s_max)
    if check and int(c[:, 1].max()) > e_bucket:
        raise ValueError(f'v3_input: {int(c[:, 1].max())} entities, the bucket {e_bucket}')
    fixed = torch.full((B,), V3.FIXED, dtype=torch.int64, device=dev)
    P, pv = _t_rows(F, fixed, V3.PF, p_max, V3.PF, p)
    Lr, _ = _t_rows(F, fixed + p * V3.PF, V3.LOCAL, p_max, V3.LOCAL, p)
    eat = fixed + p * V3.BODY
    E, ev = _t_rows(F, eat, V3.EF, e_bucket, V3.EF, e)
    nat = eat + c[:, 1] * V3.EF
    N, nv = _t_rows(F, nat, V3.NF, n_max, V3.NF, n)
    S, sv = _t_rows(F, nat + c[:, 2] * V3.NF, V3.SF, s_max, V3.SF, s)
    H, CW = V3.HEADER, V3.CANVAS_H * V3.CANVAS_W * V3.ROOM_WORDS
    canvas = F[:, H:H + CW].reshape(B, V3.CANVAS_H, V3.CANVAS_W, V3.ROOM_WORDS)
    mm = F[:, H + CW:H + CW + V3.MAP_SIDE * V3.MAP_SIDE]
    flat = lambda x: x.reshape(-1, x.shape[-1])   # noqa: E731

    def back(x, m):
        return x.reshape(B, m, *x.shape[1:])

    def keep(x, valid, fill=0):
        v = valid.view(*valid.shape, *([1] * (x.dim() - valid.dim())))
        return torch.where(v, x, torch.full_like(x, fill))
    nf = lambda x, valid: (~torch.isfinite(x) & valid.view(*valid.shape, *([1] * (x.dim() - valid.dim())))).flatten(1).sum(1)  # noqa: E731,E501
    L5 = Lr.view(B, p_max, V3.LOCAL_SIDE, V3.LOCAL_SIDE, V3.ROOM_WORDS)
    out = {'nonfinite': torch.stack([nf(P, pv), nf(E, ev), nf(N, nv), nf(S, sv),
                                     (~torch.isfinite(canvas)).flatten(1).sum(1) + (~torch.isfinite(mm)).sum(1)
                                     + nf(L5, pv)], 1).to(torch.int64)}
    main = _ix(P[:, 0], [_PC['x'], _PC['y']]).to(torch.float64)
    out['main_xy'] = main.to(torch.float32)
    out['room'] = _t_room(canvas)
    centres = _dc('room_centres', _ROOM_CENTRES, dev)
    out['room_xy'] = _t_div(centres, POS_SCALE).to(torch.float32)[None].expand(B, -1, -1).contiguous()
    out['room_rel'] = _t_rel(centres[None], main[:, None])
    mb = _t_bits(mm, V3.MAP_BITS)
    out['map_bits'], out['map_icon'] = mb[..., :4], t_map_icon_index(mb[..., 4])
    fp = flat(P)
    out['P_num'], out['P_flag'] = back(_t_num(fp, PLAN['P']), p_max), back(_t_flags(fp, PLAN['P']), p_max)
    out['P_cat'] = back(_t_cats(fp, PLAN['P']), p_max)
    known = (P[..., _PC['position_known']] == 1) & pv
    xy = _ix(P, [_PC['x'], _PC['y']]).to(torch.float64)
    g = torch.nan_to_num(_ix(P[:, 0], [_PC['grid_w'], _PC['grid_h']]).to(torch.float64), nan=1.0)
    gw = torch.trunc(torch.clamp(g[:, 0], 1, V3.CANVAS_W))[:, None]
    gh = torch.trunc(torch.clamp(g[:, 1], 1, V3.CANVAS_H))[:, None]
    cxy = torch.floor(_t_div(xy, V3.CELL))
    cell = torch.stack([torch.minimum(torch.clamp(cxy[..., 0], min=0), gw - 1),
                        torch.minimum(torch.clamp(cxy[..., 1], min=0), gh - 1)], -1)
    out['P_off'] = keep(_t_clamp(_t_div(xy - V3.CELL * cell - V3.CELL / 2, V3.CELL)), known)
    out['P_rel'] = keep(_t_rel(xy, main[:, None]), known)
    out['P_rel_valid'] = known.to(torch.float32)
    out['local'] = keep(_t_room(L5), pv)
    out['local_pad'] = ~known
    fe = flat(E)
    out['E_num'], out['E_flag'] = back(_t_num(fe, PLAN['E']), e_bucket), back(_t_flags(fe, PLAN['E']), e_bucket)
    out['E_cat'] = back(_t_cats(fe, PLAN['E']), e_bucket)
    out['E_id'] = back(_t_stack([t_identity_index(fe[:, c_]) for c_ in PLAN['E']['idn']], fe.shape[0], dev), e_bucket)
    v, bad = _t_sane(_ix(E, PLAN['E']['anim']))
    out['E_anim'] = torch.where(~bad & (v >= 0) & (v < V3.ANIM_ROWS), _t_trunc(v), torch.zeros_like(_t_trunc(v)))
    pr = E[..., _EC['price']]
    out['E_price'] = _t_clamp(torch.where(pr > 0, _t_div(pr, PRICE_COIN_SCALE), torch.zeros_like(pr)))[..., None]
    out['E_rel'] = _t_rel(_ix(E, [_EC['x'], _EC['y']]), main[:, None])
    lrel = _t_rel(_ix(E, [_EC['l_ex'], _EC['l_ey']]), main[:, None])
    out['E_laser_rel'] = torch.where((E[..., _EC['kind']] == _LASER_KIND)[..., None], lrel, torch.zeros_like(lrel))
    fn_ = flat(N)
    out['N_num'], out['N_flag'] = back(_t_num(fn_, PLAN['N']), n_max), back(_t_flags(fn_, PLAN['N']), n_max)
    out['N_item'] = back(_t_items(fn_, PLAN['N']), n_max)
    fs = flat(S)
    out['S_num'], out['S_flag'] = back(_t_num(fs, PLAN['S']), s_max), back(_t_flags(fs, PLAN['S']), s_max)
    out['S_cat'] = back(_t_cats(fs, PLAN['S']), s_max)
    out['S_item'] = back(_t_items(fs, PLAN['S'], family=fs[:, _SC_FAMILY]), s_max)
    for k in list(out):
        g_ = k.split('_')[0]
        valid = {'P': pv, 'E': ev, 'N': nv, 'S': sv}.get(g_)
        if valid is not None:
            out[k] = keep(out[k], valid)
    out['P_mask'], out['E_mask'], out['N_mask'], out['S_mask'] = pv, ev, nv, sv
    return out


# ----------------------------------------------------------------------------------------------- ROW records
# (seq-v1 step 3) The padded layout of the actor's graphs and the learner's batches: E in its bucket (E_BUCKETS, by the
# batch's largest e), bodies P_MAX, inventory / slot rows N_MAX_V3 / S_MAX_V3 (records with more are cut there and
# counted: STATS['trunc_n'] / ['trunc_s']). STATS (this process): the host side of the actor's calls -- checks, buckets,
# truncations, non-finite wire floats -- for the progress rows.
N_MAX_V3, S_MAX_V3 = 64, 32
STATS = dict(calls=0, records=0, host_s=0.0, trunc_n=0, trunc_s=0, nonfinite=0, **{f'bucket{b}': 0 for b in E_BUCKETS})


def rows_counts(rows, stats=True):
    """ROW records (numpy, each with its v3 frame) -> (counts [n, 4], e bucket): check_batch on the records' headers,
    visible_schema and visible_bytes; with stats the call's host time, bucket, truncations and non-finite frame floats
    added to STATS."""
    import time
    t0 = time.perf_counter()
    frames = rows['visible_frame'].view('<f4')
    counts, trunc = check_batch(frames[:, :V3.HEADER], rows['visible_schema'], rows['visible_bytes'], P_MAX, N_MAX_V3,
                                S_MAX_V3)
    eb = bucket(int(counts[:, 1].max()), E_BUCKETS)
    if stats:
        words = (np.asarray(rows['visible_bytes'], np.int64) // 4)
        w = int(words.max())
        live = np.arange(w)[None, :] < words[:, None]
        STATS['nonfinite'] += int((~np.isfinite(frames[:, :w]) & live).sum())
        STATS['calls'] += 1
        STATS['records'] += len(rows)
        STATS[f'bucket{eb}'] += 1
        STATS['trunc_n'] += trunc['n']
        STATS['trunc_s'] += trunc['s']
        STATS['host_s'] += time.perf_counter() - t0
    return counts, eb


V3_PREFIX = 'v3.'


def put_batch(b, d):
    """The decoded v3 tensors into a batch dict as top-level keys 'v3.<name>' (184, 2026-11-18: a nested dict b['v3']
    broke the learner's per-key row selection, tok_policy.take -- every per-key index / slice of a batch now covers the
    v3 tensors too, each with the records on its first axis). Returns b."""
    for k, v in d.items():
        b[V3_PREFIX + k] = v
    return b


def get_batch(b):
    """The v3 tensors of a batch dict (put_batch's keys) as a dict by name; {} when it has none."""
    n = len(V3_PREFIX)
    return {k[n:]: v for k, v in b.items() if k.startswith(V3_PREFIX)}


def batch_from_rows(rows, device, stats=False):
    """ROW records (numpy) -> the padded v3 batch on `device` (decode_frames_device)."""
    import torch
    counts, eb = rows_counts(rows, stats)
    w = int(np.asarray(rows['visible_bytes']).max()) // 4
    F = torch.from_numpy(np.ascontiguousarray(rows['visible_frame'].view('<f4')[:, :w])).to(device)
    return decode_frames_device(F, counts, eb, N_MAX_V3, S_MAX_V3)


def lazy_from_raw(raw, frame_at, schema_at, bytes_at):
    """batch_from_raw without the decode (184, 2026-11-18: the learner put a whole rollout through decode_rows at once;
    its padded v3 layout -- every record's 128-entity bucket -- did not fit the GPU): the records' frame bytes (a view
    of raw, no copy) and their counts (checked here, on the host: one read of the headers, schemas and sizes; kept on
    raw's device). put_batch them: per-key row selection (take) carries them, and _forward_v3 decodes only the rows a
    forward reads (materialize)."""
    import torch
    head = raw[:, frame_at:frame_at + 4 * V3.HEADER].contiguous().view(torch.float32).cpu().numpy()
    vs = raw[:, schema_at:schema_at + 4].contiguous().view(torch.int32)[:, 0].cpu().numpy()
    nb = raw[:, bytes_at:bytes_at + 4].contiguous().view(torch.int32)[:, 0].cpu().numpy()
    counts, _trunc = check_batch(head, vs, nb, P_MAX, N_MAX_V3, S_MAX_V3)
    w = int(nb.max())
    return {'raw': raw[:, frame_at:frame_at + w], 'counts': torch.from_numpy(counts).to(raw.device)}


def materialize(v):
    """get_batch's dict -> the decoded v3 tensors: a lazy one (lazy_from_raw: 'raw', 'counts') decoded now -- the E
    bucket from its rows' largest count (one read from the device) -- an already decoded one as it is."""
    if 'raw' not in v:
        return v
    import torch
    raw, c = v['raw'], v['counts']
    n, w = raw.shape
    eb = bucket(int(c[:, 1].max()), E_BUCKETS)
    F = raw.contiguous().view(torch.float32).view(n, w // 4)
    return decode_frames_device(F, c, eb, N_MAX_V3, S_MAX_V3, check=False)


def batch_from_raw(raw, frame_at, schema_at, bytes_at):
    """ROW records as bytes on the device (raw uint8 [n, >= ROW bytes]; the frame, visible_schema and visible_bytes at
    the given byte offsets) -> the padded v3 batch on raw's device. One host read of the headers, schemas and sizes."""
    import torch
    n = raw.shape[0]
    head = raw[:, frame_at:frame_at + 4 * V3.HEADER].contiguous().view(torch.float32).cpu().numpy()
    vs = raw[:, schema_at:schema_at + 4].contiguous().view(torch.int32)[:, 0].cpu().numpy()
    nb = raw[:, bytes_at:bytes_at + 4].contiguous().view(torch.int32)[:, 0].cpu().numpy()
    counts, trunc = check_batch(head, vs, nb, P_MAX, N_MAX_V3, S_MAX_V3)
    eb = bucket(int(counts[:, 1].max()), E_BUCKETS)
    w = int(nb.max())
    F = raw[:, frame_at:frame_at + w].contiguous().view(torch.float32).view(n, w // 4)
    return decode_frames_device(F, counts, eb, N_MAX_V3, S_MAX_V3)
