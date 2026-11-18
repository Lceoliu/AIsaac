"""Frozen PyTorch decoder from 7ea9bd9 for bitwise and operator-count regression tests.

Only the device code is copied; schema and unmodified host parsing helpers are shared.
"""
import numpy as np

from isaac_bridge.v3_input import (
    CLAMP, CODE_CUT, PLAN, POS_SCALE, PRICE_COIN_SCALE, P_MAX, V3, W8, W16,
    _EC, _FAMILY_NS, _LASER_KIND, _PC, _ROOM_CENTRES, _ROOM_DIV, _SC_FAMILY,
    _grid, _nonfinite, _t, split_frame,
)


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
