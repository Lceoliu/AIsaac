"""visible-v3 decode for the model (package A, seq-v1 step 1; DRAFT 2026-11-18, not committed: waits for main = the C
package). The only place where a v3 frame (schema_v3, the frozen interface) becomes model inputs; the only place where
the scales apply (contract C7). Host side (numpy): the actor decodes outside its captured graph; the device side (torch,
the learner) must give the same values (a test asserts it).

Per frame (decode_frame):
  - the record's visible_schema must equal schema_v3.VERSION (row by row: the audit found a worker could write a v2
    frame into a ROW configured for v3), and the frame header must pass visible_producer.check_frame_header_v3 (magic,
    header[6] = VERSION_WORD of this tree's schema, batch 5; counts in range; the size formula, exactly the record's
    byte count), else ValueError -- never a silent decode (main's 05f1622 interface; the hardening batch's
    VisibleV3Fatal replaces it when it lands);
  - numeric ('num') columns: value / scale, clamped to [-NUM_CLAMP, NUM_CLAMP] (batch 2, B5); a NaN (none on the
    wire by its rule) decodes to 0, an infinity to the clamp;
  - 'known' / 'bit' columns: 0 / 1; 'bits8' / 'word16': expanded into bits;
  - categorical columns -> rows of the shared tables (cat_index / wire_cat_index for the batch-4 columns whose wire
    value is raw: price, direction; identity_index; item_index per family; map_icon_index; anim rows);
  - price (E col 31) also as a number: coins / PRICE_COIN_SCALE when > 0, else 0 (the coordinator, 11-18: the old
    model's /15; the special prices -1 / -2 / -3 / -1000 are categories only);
  - 'reserved' columns dropped.
Derived inputs (the model's, never on the wire), every one clamped to [-NUM_CLAMP, NUM_CLAMP] (a NaN to 0):
  - C4 relative offsets / distances to the main body (P row 0, batch 4: body_role 1, position_known 1 -- asserted):
    entities, other bodies, laser end points, the 28 room-token centres; offsets / REL_SCALE['offset'], distances /
    REL_SCALE['distance'] (schema_v3, batch 4); computed from the raw canvas-frame pixels;
  - C6 in-cell offsets of each body with a position: (pos - CELL * clamp(floor(pos / CELL)) - CELL / 2) / CELL;
  - C13: a body with position_known 0 takes no part: its relative quantities and in-cell offset are 0 with
    rel_valid 0, its local patch is marked pad.
Outputs are numpy arrays (see DecodedFrame); padded batches for the actor come from pad_batch."""
import math

import numpy as np

from . import schema_v3 as V3
from .visible_producer import check_frame_header_v3

# The coins scale of the price number (the coordinator 11-18: the old model's /15; to move into schema_v3 with the next
# batch -- until then the one model-side constant here, on the grep gate's whitelist).
PRICE_COIN_SCALE = 15.0
CLAMP = V3.NUM_CLAMP
W8, W16 = 8, V3.FLAG_WORD_BITS   # (the bits of a 'bits8' / a 'word16' column: schema_v3's kinds)
R_ROWS, R_COLS, R_BLOCK = 4, 7, 4   # the 28 room tokens: 4 x 7 blocks of 4 x 4 canvas cells (v2's down-sampling)

_LASER_KIND = V3.ENTITY_KIND['laser']   # E col 15's code on the wire (raw kind + 1)
POS_SCALE = V3.P_COLS[V3.col(V3.P_COLS, 'x')][3]   # (the canvas-frame positions' scale, P / E x and y: 520)


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


# ----------------------------------------------------------------------------------------------- frame bytes
def split_frame(buf, visible_schema, nbytes=None):
    """The frame bytes of a record -> dict of float32 views (header, canvas [16, 28, 5], minimap [169], P [p, 192],
    local [p, 9, 9, 5], E [e, 64], N [n, 8], S [s, 24]). Raises ValueError on another schema version, a bad header or
    a size that does not match."""
    if int(visible_schema) != V3.VERSION:
        raise ValueError(f'v3_input: visible_schema {int(visible_schema)} is not schema_v3.VERSION {V3.VERSION}')
    mv = memoryview(buf).cast('B')
    if nbytes is not None and (nbytes % 4 or nbytes > len(mv)):
        raise ValueError(f'v3_input: {nbytes} bytes is not a whole frame of floats ({len(mv)} in the buffer)')
    f = np.frombuffer(mv, '<f4', count=(nbytes // 4) if nbytes is not None else -1)
    p, e, n, s = check_frame_header_v3(f, exact=nbytes is not None)
    if p < 1:
        raise ValueError('v3_input: a frame without a body')
    at, out = V3.HEADER, {'header': f[:V3.HEADER]}
    for name, count, shape in (('canvas', 1, (V3.CANVAS_H, V3.CANVAS_W, V3.ROOM_WORDS)),
                               ('minimap', 1, (V3.MAP_SIDE * V3.MAP_SIDE,)), ('P', p, (V3.PF,)),
                               ('local', p, (V3.LOCAL_SIDE, V3.LOCAL_SIDE, V3.ROOM_WORDS)), ('E', e, (V3.EF,)),
                               ('N', n, (V3.NF,)), ('S', s, (V3.SF,))):
        k = count * int(np.prod(shape))
        out[name] = f[at:at + k].reshape(shape if name in ('canvas', 'minimap') else (count,) + shape)
        at += k
    return out


# ----------------------------------------------------------------------------------------------- the decoders
def _bits(words, layout):
    """[..., ] float words -> [..., len(layout)] the unsigned fields (name, bits) LSB first, as float32."""
    w = words.astype(np.int64)
    out, shift = [], 0
    for _name, bits in layout:
        out.append((w >> shift) & ((1 << bits) - 1))
        shift += bits
    return np.stack(out, -1).astype(np.float32)


_ROOM_DIV = np.asarray([V3.ROOM_DECODE_SCALE.get(n, 1) for n in V3.ROOM_CHANNELS], np.float32)


def room_channels(cells):
    """[..., 5] cell words -> [..., RC] the 21 decoded channels (word 0's 13 fields, word 1's 5, the raw words 2..4),
    divided by ROOM_DECODE_SCALE (codes / 64 etc.; others 1)."""
    ch = np.concatenate([_bits(cells[..., 0], V3.ROOM_W0), _bits(cells[..., 1], V3.ROOM_W1),
                         cells[..., 2:5].astype(np.float32)], -1)
    return ch / _ROOM_DIV


def minimap(words):
    """[169] map words -> ([169, 4] shown / visited / clear / current, [169] icon rows)."""
    b = _bits(words, V3.MAP_BITS)
    icon = np.asarray([V3.map_icon_index(v) for v in b[:, 4]], np.int64)
    return b[:, :4], icon


def _num(x, plan):
    # (a NaN never reaches the model: 0, as a missing value -- the wire rule has none, a producer bug would otherwise
    #  turn every loss into NaN; the device side does the same)
    return np.clip(np.nan_to_num(x[:, plan['num']] / plan['scale'], nan=0.0, posinf=CLAMP, neginf=-CLAMP),
                   -CLAMP, CLAMP).astype(np.float32)


def _flags(x, plan):
    parts = [x[:, plan['known']], x[:, plan['bit']]]
    if plan['bits8']:
        parts.append(_bits(x[:, plan['bits8']], (('b', 1),) * W8).reshape(len(x), W8 * len(plan['bits8'])))
    if plan['word16']:
        parts.append(_bits(x[:, plan['word16']], (('b', 1),) * W16).reshape(len(x), W16 * len(plan['word16'])))
    return np.concatenate([np.asarray(q, np.float32) for q in parts], 1)


def _cats(x, plan):
    out = np.zeros((len(x), len(plan['cat'])), np.int64)
    for j, (c, ns) in enumerate(zip(plan['cat'], plan['cat_ns'])):
        out[:, j] = [V3.wire_cat_index(ns, v) for v in x[:, c]]
    return out


def _ids(x, plan):
    return np.asarray([[V3.identity_index(v) for v in r] for r in x[:, plan['idn']]], np.int64).reshape(len(x), len(plan['idn']))


def _anims(x, plan):
    a = x[:, plan['anim']]
    ok = (a == a) & (a >= 0) & (a < V3.ANIM_ROWS)
    return np.where(ok, np.nan_to_num(a), 0).astype(np.int64)


def _items(x, plan, family=None):
    out = np.zeros((len(x), len(plan['item'])), np.int64)
    for j, (c, ns) in enumerate(zip(plan['item'], plan['item_ns'])):
        for r in range(len(x)):
            n_ = ns
            if ns == 'by_family':
                n_ = V3.FAMILY_ITEM_NS.get(int(family[r]) if family[r] == family[r] else -1)
            out[r, j] = V3.ITEM_UNKNOWN if n_ is None else V3.item_index(n_, x[r, c])
    return out


def _clamp(v):
    return np.clip(np.nan_to_num(v, nan=0.0, posinf=CLAMP, neginf=-CLAMP), -CLAMP, CLAMP).astype(np.float32)


def _t_clamp(v):
    torch = _t()
    return torch.clamp(torch.nan_to_num(v, nan=0.0, posinf=CLAMP, neginf=-CLAMP), -CLAMP, CLAMP).to(torch.float32)


class DecodedFrame(dict):
    """One frame's model inputs (numpy). Keys:
      room [16, 28, RC]           canvas channels; room_xy [28, 2] the room tokens' centres / 520; room_rel [28, 3]
                                  their offsets to the main body / REL offset, distance / REL distance
      map_bits [169, 4], map_icon [169]
      P_num, P_flag, P_cat        per body; P_cell [p, 2] in-cell offset; P_rel [p, 3]; P_rel_valid [p]
      local [p, 9, 9, RC], local_pad [p] (C13: position_known 0)
      E_num, E_flag, E_cat, E_id, E_anim; E_price [e, 1]; E_rel [e, 3]; E_laser_rel [e, 3] (0 for a non-laser)
      N_num, N_flag, N_item; S_num, S_flag, S_cat, S_item
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


def decode_frame(buf, visible_schema, nbytes=None):
    """A record's v3 frame -> DecodedFrame (see the module doc). ValueError on a frame this decoder must not read."""
    f = split_frame(buf, visible_schema, nbytes)
    P, E, N, S = f['P'], f['E'], f['N'], f['S']
    # the main body (batch 4): P row 0, body_role 1, position_known 1
    if int(P[0, _PC['body_role']]) != V3.BODY_ROLE['main'] or int(P[0, _PC['position_known']]) != 1:
        raise ValueError('v3_input: P row 0 is not the controlled body with a position (schema_v3 batch 4)')
    main = P[0, [_PC['x'], _PC['y']]].astype(np.float64)
    known = P[:, _PC['position_known']] == 1
    out = DecodedFrame()
    out['main_xy'] = main.astype(np.float32)
    out['room'] = room_channels(f['canvas'])
    out['room_xy'] = (_ROOM_CENTRES / POS_SCALE).astype(np.float32)   # (the P / E positions' scale)
    out['room_rel'] = _rel(_ROOM_CENTRES, main)
    out['map_bits'], out['map_icon'] = minimap(f['minimap'])
    # bodies
    out['P_num'], out['P_flag'], out['P_cat'] = _num(P, PLAN['P']), _flags(P, PLAN['P']), _cats(P, PLAN['P'])
    xy = P[:, [_PC['x'], _PC['y']]].astype(np.float64)
    gw, gh = int(P[0, _PC['grid_w']]), int(P[0, _PC['grid_h']])
    cell = np.stack([np.clip(np.floor(xy[:, 0] / V3.CELL), 0, gw - 1), np.clip(np.floor(xy[:, 1] / V3.CELL), 0, gh - 1)], 1)
    out['P_cell'] = np.where(known[:, None], _clamp((xy - V3.CELL * cell - V3.CELL / 2) / V3.CELL), 0).astype(np.float32)
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
    """ROW records -> [DecodedFrame], each row's visible_schema checked on its own (decode_row)."""
    return [decode_row(rows[i:i + 1]) for i in range(len(rows))]


def decode_row(row):
    """A ROW record (length-1 view or a structured scalar) -> DecodedFrame."""
    n = int(np.asarray(row['visible_bytes']).reshape(-1)[0])
    buf = np.asarray(row['visible_frame']).reshape(-1)[:n]
    return decode_frame(buf.tobytes(), np.asarray(row['visible_schema']).reshape(-1)[0], n)


# ----------------------------------------------------------------------------------------------- the device side
# The same decode in torch (the learner decodes the frame bytes on its own device; contract C7: one decode, two
# implementations that give the same values -- tests/test_v3_input.py asserts it value for value on the golden frames
# and on frames with out-of-range codes). The header is read on the host (split_frame: the same checks); everything
# after it runs in torch on `device`. Every categorical rule mirrors schema_v3's scalar function named beside it.
def _t():
    import torch
    return torch


def _t_bits(w, layout):
    torch = _t()
    w = w.to(torch.int64)
    out, shift = [], 0
    for _name, bits in layout:
        out.append((w >> shift) & ((1 << bits) - 1))
        shift += bits
    return torch.stack(out, -1).to(torch.float32)


def _t_trunc(v):
    torch = _t()
    return torch.trunc(torch.nan_to_num(v.to(torch.float64), nan=0.0)).to(torch.int64)


def t_cat_index(ns, code):
    """schema_v3.cat_index on a tensor of codes."""
    torch = _t()
    base, size = V3.CAT_NS[ns]
    code = code.to(torch.float64)
    bad = torch.isnan(code) | (code < 0)
    c = _t_trunc(code)
    row = torch.where(c == 0, torch.zeros_like(c), base + torch.clamp(c, max=size - 1))
    return torch.where(bad, torch.full_like(c, base + size - 1), row)


def t_price_code(v):
    """schema_v3.price_code on a tensor."""
    torch = _t()
    size = V3.CAT_NS['price'][1]
    v = v.to(torch.float64)
    c = _t_trunc(v)
    out = torch.full_like(c, size - 1)
    out = torch.where(c == 0, torch.zeros_like(c), out)
    out = torch.where((c >= 1) & (c <= 99), c, out)
    for raw, code in V3.PRICE_CODES.items():
        out = torch.where(c == raw, torch.full_like(c, code), out)
    return torch.where(torch.isnan(v), torch.full_like(c, size - 1), out)


def t_direction_code(v):
    """schema_v3.direction_code on a tensor."""
    torch = _t()
    size = V3.CAT_NS['direction'][1]
    v = v.to(torch.float64)
    bad = torch.isnan(v) | (v < -1) | (v > 3)
    return torch.where(bad, torch.full_like(_t_trunc(v), size - 1), _t_trunc(v) + 2)


_T_FROM_RAW = {'price': t_price_code, 'direction': t_direction_code}


def t_wire_cat_index(ns, wire):
    """schema_v3.wire_cat_index on a tensor (the batch-4 raw columns coded first)."""
    f = _T_FROM_RAW.get(ns)
    if f is None and ns in V3.CAT_FROM_RAW:
        raise KeyError(f'v3_input: no tensor form of schema_v3.CAT_FROM_RAW[{ns!r}]')
    return t_cat_index(ns, f(wire) if f else wire)


def t_identity_index(v):
    """schema_v3.identity_index on a tensor."""
    torch = _t()
    v = v.to(torch.float64)
    bad = torch.isnan(v) | (v < 0) | (v >= V3.IDENTITY_UNKNOWN)
    return torch.where(bad, torch.full_like(_t_trunc(v), V3.IDENTITY_UNKNOWN), _t_trunc(v))


def t_item_index(base, size, v):
    """schema_v3.item_index with the namespace's (base, size) as tensors or ints (size 0: no namespace -> unknown)."""
    torch = _t()
    v = v.to(torch.float64)
    c = _t_trunc(v)
    base = torch.as_tensor(base, dtype=torch.int64, device=v.device)
    size = torch.as_tensor(size, dtype=torch.int64, device=v.device)
    unk = torch.full_like(c, V3.ITEM_UNKNOWN)
    row = torch.where((c > 0) & (c < size), base + c, unk)
    row = torch.where(c == 0, torch.zeros_like(c), row)
    return torch.where(torch.isnan(v) | (size == 0), unk, row)


def t_map_icon_index(v):
    """schema_v3.map_icon_index on a tensor."""
    torch = _t()
    v = v.to(torch.float64)
    bad = torch.isnan(v) | (v < 0)
    return torch.where(bad, torch.full_like(_t_trunc(v), V3.MAP_ICONS - 1),
                       torch.clamp(_t_trunc(v), max=V3.MAP_ICONS - 1))


def _t_room(cells):
    torch = _t()
    ch = torch.cat([_t_bits(cells[..., 0], V3.ROOM_W0), _t_bits(cells[..., 1], V3.ROOM_W1),
                    cells[..., 2:5].to(torch.float32)], -1)
    return ch / torch.as_tensor(_ROOM_DIV, device=cells.device)


def _t_flags(x, plan):
    torch = _t()
    n = x.shape[0]
    parts = [x[:, plan['known']], x[:, plan['bit']]]
    if plan['bits8']:
        parts.append(_t_bits(x[:, plan['bits8']], (('b', 1),) * W8).reshape(n, W8 * len(plan['bits8'])))
    if plan['word16']:
        parts.append(_t_bits(x[:, plan['word16']], (('b', 1),) * W16).reshape(n, W16 * len(plan['word16'])))
    return torch.cat([q.to(torch.float32) for q in parts], 1)


def _t_num(x, plan):
    torch = _t()
    v = x[:, plan['num']] / torch.as_tensor(plan['scale'], device=x.device)
    return torch.clamp(torch.nan_to_num(v, nan=0.0, posinf=CLAMP, neginf=-CLAMP), -CLAMP, CLAMP)


def _t_cats(x, plan):
    torch = _t()
    cols = [t_wire_cat_index(ns, x[:, c]) for c, ns in zip(plan['cat'], plan['cat_ns'])]
    return torch.stack(cols, 1) if cols else torch.zeros((x.shape[0], 0), dtype=torch.int64, device=x.device)


def _t_items(x, plan, family=None):
    torch = _t()
    cols = []
    for c, ns in zip(plan['item'], plan['item_ns']):
        if ns == 'by_family':
            fam = _t_trunc(family)
            base = torch.zeros_like(fam)
            size = torch.zeros_like(fam)
            for k, fns in V3.FAMILY_ITEM_NS.items():
                b_, s_ = V3.ITEM_NS[fns]
                hit = (fam == k) & ~torch.isnan(family.to(torch.float64))
                base = torch.where(hit, torch.full_like(fam, b_), base)
                size = torch.where(hit, torch.full_like(fam, s_), size)
            cols.append(t_item_index(base, size, x[:, c]))
        else:
            b_, s_ = V3.ITEM_NS[ns]
            cols.append(t_item_index(b_, s_, x[:, c]))
    return torch.stack(cols, 1) if cols else torch.zeros((x.shape[0], 0), dtype=torch.int64, device=x.device)


def _t_rel(xy, main):
    torch = _t()
    d = xy.to(torch.float64) - main
    dist = torch.sqrt((d ** 2).sum(-1, keepdim=True))
    return _t_clamp(torch.cat([d / V3.REL_SCALE['offset'], dist / V3.REL_SCALE['distance']], -1))


def decode_frame_torch(buf, visible_schema, nbytes=None, device='cpu'):
    """decode_frame on `device` in torch: the same keys, the same values (float32 / int64 / bool tensors)."""
    torch = _t()
    f = {k: torch.as_tensor(np.array(v), device=device) for k, v in
         split_frame(buf, visible_schema, nbytes).items()}
    P, E, N, S = f['P'], f['E'], f['N'], f['S']
    if int(P[0, _PC['body_role']]) != V3.BODY_ROLE['main'] or int(P[0, _PC['position_known']]) != 1:
        raise ValueError('v3_input: P row 0 is not the controlled body with a position (schema_v3 batch 4)')
    main = P[0, [_PC['x'], _PC['y']]].to(torch.float64)
    known = P[:, _PC['position_known']] == 1
    centres = torch.as_tensor(_ROOM_CENTRES, device=device)
    out = {'main_xy': main.to(torch.float32), 'room': _t_room(f['canvas']),
           'room_xy': (centres / POS_SCALE).to(torch.float32), 'room_rel': _t_rel(centres, main)}
    mb = _t_bits(f['minimap'], V3.MAP_BITS)
    out['map_bits'], out['map_icon'] = mb[:, :4], t_map_icon_index(mb[:, 4])
    out['P_num'], out['P_flag'], out['P_cat'] = _t_num(P, PLAN['P']), _t_flags(P, PLAN['P']), _t_cats(P, PLAN['P'])
    xy = P[:, [_PC['x'], _PC['y']]].to(torch.float64)
    gw, gh = int(P[0, _PC['grid_w']]), int(P[0, _PC['grid_h']])
    cell = torch.stack([torch.clamp(torch.floor(xy[:, 0] / V3.CELL), 0, gw - 1),
                        torch.clamp(torch.floor(xy[:, 1] / V3.CELL), 0, gh - 1)], 1)
    off = _t_clamp((xy - V3.CELL * cell - V3.CELL / 2) / V3.CELL)
    out['P_cell'] = torch.where(known[:, None], off, torch.zeros_like(off))
    rel = _t_rel(xy, main)
    out['P_rel'] = torch.where(known[:, None], rel, torch.zeros_like(rel))
    out['P_rel_valid'] = known.to(torch.float32)
    out['local'] = _t_room(f['local'])
    out['local_pad'] = ~known
    out['E_num'], out['E_flag'], out['E_cat'] = _t_num(E, PLAN['E']), _t_flags(E, PLAN['E']), _t_cats(E, PLAN['E'])
    if PLAN['E']['idn']:
        out['E_id'] = torch.stack([t_identity_index(E[:, c]) for c in PLAN['E']['idn']], 1)
    else:
        out['E_id'] = torch.zeros((E.shape[0], 0), dtype=torch.int64, device=device)
    a = E[:, PLAN['E']['anim']]
    ok = ~torch.isnan(a) & (a >= 0) & (a < V3.ANIM_ROWS)
    out['E_anim'] = torch.where(ok, _t_trunc(a), torch.zeros_like(_t_trunc(a)))
    pr = E[:, _EC['price']]
    out['E_price'] = _t_clamp(torch.where(pr > 0, pr / PRICE_COIN_SCALE, torch.zeros_like(pr)))[:, None]
    out['E_rel'] = _t_rel(E[:, [_EC['x'], _EC['y']]], main)
    lrel = _t_rel(E[:, [_EC['l_ex'], _EC['l_ey']]], main)
    out['E_laser_rel'] = torch.where((E[:, _EC['kind']] == _LASER_KIND)[:, None], lrel, torch.zeros_like(lrel))
    out['N_num'], out['N_flag'], out['N_item'] = _t_num(N, PLAN['N']), _t_flags(N, PLAN['N']), _t_items(N, PLAN['N'])
    out['S_num'], out['S_flag'], out['S_cat'] = _t_num(S, PLAN['S']), _t_flags(S, PLAN['S']), _t_cats(S, PLAN['S'])
    out['S_item'] = _t_items(S, PLAN['S'], family=S[:, _SC_FAMILY])
    return out


# ----------------------------------------------------------------------------------------------- batches (actor)
E_BUCKETS = (16, 32, 64, 128)
P_MAX = 4   # bodies per frame at most in the padded layout (AB+ single player: 1-2)


def bucket(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f'v3_input: {n} past the largest bucket {buckets[-1]}')


def pad_batch(frames, e_bucket=None, n_max=None, s_max=None):
    """[DecodedFrame] -> dict of arrays [B, max, ...] with masks *_mask (True = a real token). e padded to its bucket
    (E_BUCKETS), bodies to P_MAX; n / s to the batch maximum unless given (their buckets wait for V4)."""
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
            a = np.zeros((B, m) + v.shape[1:], v.dtype)
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
