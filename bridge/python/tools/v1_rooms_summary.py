"""V1 summary (2026-11-17): the room-shape table from abplus_probe_v1_rooms.py's raw files (rooms.jsonl, walk.jsonl,
sizes.jsonl). Standard library only (runs with the host's python3).

  python3 v1_rooms_summary.py <out dir>   ->  <out>/summary.md (the tables), <out>/summary_rows.json (one row per room)

Origin O = GetGridPosition(0) - (20, 20) (the contract's C1 candidate). Per room: the grid, whether every cell centre is
O + 40 * (col + 0.5, row + 0.5) (max deviation), GetTopLeftPos / GetBottomRightPos relative to O in pixels and cells,
the inside mask's bounding box in cells (IsPositionInRoom of the cell centre with size 0, and with the player's Size),
which quadrant of an L room has no inside cell, the first inside cell, the walk (player positions relative to O, the
cells reached, GetGridIndex vs the formula floor((pos - O) / 40): mismatches, off-grid samples and the GetGridIndex
values there), the clamp extents, the doors (slot position, its GetGridIndex and the formula's), the entities' formula
cells against GetGridIndex, the spawned coin's position against the one asked for, the cell-edge convention of
GetGridIndex around cell 0, and per character the inside mask's dependence on the IsPositionInRoom size.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

SHAPES = {1: '1x1', 2: 'IH', 3: 'IV', 4: '1x2', 5: 'IIV', 6: '2x1', 7: 'IIH', 8: '2x2', 9: 'LTL', 10: 'LTR', 11: 'LBL',
          12: 'LBR'}
SIZES = (0, 5, 10, 13, 15, 20, 25, 30, 40)


def cell_of(x, y, ox, oy, gw, gh):
    col, row = math.floor((x - ox) / 40.0), math.floor((y - oy) / 40.0)
    return (row * gw + col) if (0 <= col < gw and 0 <= row < gh) else -1


def bbox(cells, gw):
    if not cells:
        return None
    cols = [i % gw for i in cells]
    rows = [i // gw for i in cells]
    return [min(cols), min(rows), max(cols), max(rows)]


def size_bit(s):
    """The mask bit of the largest probed size <= s (the player's Size is one of them in practice)."""
    k = max(i for i, v in enumerate(SIZES) if v <= s + 1e-9)
    return k, SIZES[k]


def room_row(r):
    gw, gh, n = r['gw'], r['gh'], r['n']
    gp = r['gp']
    ox, oy = gp[0][0] - 20.0, gp[0][1] - 20.0
    dev = max(max(abs(gp[i][0] - (ox + 40 * (i % gw) + 20)), abs(gp[i][1] - (oy + 40 * (i // gw) + 20)))
              for i in range(n))
    k, ks = size_bit(r['psize'])
    in0 = [i for i in range(n) if r['inside'][i] & 1]
    inp = [i for i in range(n) if r['inside'][i] & (1 << k)]
    b0 = bbox(in0, gw)
    quads = {}
    if b0:
        for name, (c0, c1, r0, r1) in {'top-left': (0, gw // 2, 0, gh // 2), 'top-right': (gw // 2, gw, 0, gh // 2),
                                       'bottom-left': (0, gw // 2, gh // 2, gh),
                                       'bottom-right': (gw // 2, gw, gh // 2, gh)}.items():
            quads[name] = sum(1 for i in in0 if c0 <= i % gw < c1 and r0 <= i // gw < r1)
    missing = [q for q, v in quads.items() if v == 0]
    tl, br = r['tl'], r['br']
    w = r.get('walk') or {}
    doors = []
    for d in r['doors']:
        sp = d.get('slot_pos')
        if isinstance(sp, list):
            fg = cell_of(sp[0], sp[1], ox, oy, gw, gh)
            doors.append(dict(slot=d['slot'], allowed=d.get('allowed'), slot_pos=sp, slot_gi=d.get('slot_gi'),
                              formula=fg, door=d.get('pos') is not None, door_gi=d.get('gi'), own_gi=d.get('own_gi'),
                              rel=[sp[0] - ox, sp[1] - oy]))
    ents = (r.get('entities') or {}).get('entities') or []
    ent_mism = [e for e in ents if e.get('gi') != e.get('fgi')]
    sp = (r.get('entities') or {}).get('spawned')
    coin = None
    if sp:
        got = [e for e in ents if e.get('seed') == sp.get('seed')]
        if got:
            coin = dict(asked=sp['asked'], got=[got[0]['x'], got[0]['y']], gi=got[0]['gi'], fgi=got[0]['fgi'],
                        cell=sp['cell'])
    edge = None
    for row in r.get('edges') or []:
        if row['cell'] == 0:
            xs = sorted((d, g) for d, g in row['x'])
            edge = dict(x=[[d, g] for d, g in xs if abs(abs(d) - 20) < 0.05 or d in (-25, 25)],
                        y=[[d, g] for d, g in sorted(row['y']) if abs(abs(d) - 20) < 0.05 or d in (-25, 25)])
    clamp = [c for c in r.get('clamp') or []]
    return dict(key=r['key'], shape=r['shape'], shape_name=SHAPES.get(r['shape'], '?'), type=r['type'],
                variant=r.get('variant'), stage=r.get('stage'), stage_type=r.get('stage_type'), gw=gw, gh=gh, n=n,
                gp0=gp[0], origin=[ox, oy], corners=r.get('corner_cells'), centre_dev=dev,
                tl=tl, br=br, tl_rel=[tl[0] - ox, tl[1] - oy], br_rel=[br[0] - ox, br[1] - oy],
                tl_cells=[(tl[0] - ox) / 40, (tl[1] - oy) / 40], br_cells=[(br[0] - ox) / 40, (br[1] - oy) / 40],
                center=r.get('center'), first_inside=r.get('first_inside'),
                first_inside_pos=gp[r['first_inside']] if r.get('first_inside') is not None else None,
                inside_bbox0=b0, inside_bbox_psize=bbox(inp, gw), psize=r['psize'], psize_bit=ks,
                inside0_cells=len(in0), inside_psize_cells=len(inp), inside_same=(in0 == inp), quadrants=quads,
                l_missing=missing,
                walk=dict(samples=w.get('samples'), mismatch=w.get('mismatch'), off_grid=w.get('off_grid'),
                          left_room=w.get('left_room'), cells=w.get('cells'),
                          x_rel=[w.get('x_min', 0) - ox, w.get('x_max', 0) - ox],
                          y_rel=[w.get('y_min', 0) - oy, w.get('y_max', 0) - oy],
                          x=[w.get('x_min'), w.get('x_max')], y=[w.get('y_min'), w.get('y_max')],
                          off_grid_range=w.get('off_grid_range'), mismatch_examples=w.get('mismatch_examples')),
                clamp=clamp, doors=doors, entities=len(ents), entity_mismatch=len(ent_mism),
                entity_mismatch_examples=ent_mism[:5], coin=coin, edge0=edge,
                enemies_removed=r.get('enemies_removed'), payload_bytes=r.get('payload_bytes'))


def walk_transitions(path):
    """Off-grid / other-room samples in the walk file: the transition frames (entering, a door left through and back)."""
    t = defaultdict(lambda: dict(n=0, x=[1e9, -1e9], y=[1e9, -1e9], gi=set(), rooms=set()))
    for line in open(path):
        w = json.loads(line)
        if not w.get('transition'):
            continue
        s = t[w['key']]
        s['n'] += 1
        s['x'] = [min(s['x'][0], w['x']), max(s['x'][1], w['x'])]
        s['y'] = [min(s['y'][0], w['y']), max(s['y'][1], w['y'])]
        s['gi'].add(w['gi'])
        s['rooms'].add(w['room'])
    return {k: dict(v, gi=sorted(v['gi'])[:12], rooms=sorted(v['rooms'])) for k, v in t.items()}


def sizes_table(path):
    out = []
    for line in open(path):
        s = json.loads(line)
        if 'inside' not in s:
            out.append(dict(char=s.get('char'), error=s.get('error')))
            continue
        masks = s['inside']
        diff = {}
        for k in range(1, len(SIZES)):
            diff[SIZES[k]] = sum(1 for m in masks if bool(m & 1) != bool(m & (1 << k)))
        out.append(dict(char=s['char'], ptype=s.get('ptype'), psize=s.get('psize'), shape=s.get('shape'),
                        cells_differing_from_size0=diff))
    return out


def fmt(v, nd=1):
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    if isinstance(v, list):
        return '(' + ', '.join(fmt(x, nd) for x in v) + ')'
    return str(v)


def main():
    out = Path(sys.argv[1])
    rows = [room_row(json.loads(l)) for l in open(out / 'rooms.jsonl') if 'error' not in json.loads(l)]
    errs = [json.loads(l) for l in open(out / 'rooms.jsonl') if 'error' in json.loads(l)]
    trans = walk_transitions(out / 'walk.jsonl')
    for r in rows:
        r['transitions'] = trans.get(r['key'])
    sizes = sizes_table(out / 'sizes.jsonl') if (out / 'sizes.jsonl').exists() else []
    (out / 'summary_rows.json').write_text(json.dumps(dict(rooms=rows, errors=errs, sizes=sizes), indent=1))
    L = ['# V1 room-shape table', '', f'{len(rows)} rooms, {len(errs)} errors. Origin O = GetGridPosition(0) - (20, 20); '
         'rel = position - O (px); cells = rel / 40.', '',
         '## Per room', '',
         '| shape | room | stage | gw x gh | GetGridPosition(0) | centre dev | TopLeft rel (cells) | BottomRight rel (cells) '
         '| inside bbox (cells, size 0) | same at player Size | L missing | first inside | walk x rel | walk y rel '
         '| walk cells | GetGridIndex != formula | off-grid samples | entities gi != formula | coin got - asked |',
         '|' + '---|' * 19]
    for r in sorted(rows, key=lambda r: (r['shape'], r['key'])):
        w = r['walk']
        coin = r['coin']
        cd = fmt([coin['got'][0] - coin['asked'][0], coin['got'][1] - coin['asked'][1]], 2) if coin else '-'
        L.append(f"| {r['shape']} {r['shape_name']} | {r['key']} | {r['stage']}/{r['stage_type']} | {r['gw']} x {r['gh']} "
                 f"| {fmt(r['gp0'])} | {fmt(r['centre_dev'], 3)} | {fmt(r['tl_rel'])} ({fmt(r['tl_cells'], 2)}) "
                 f"| {fmt(r['br_rel'])} ({fmt(r['br_cells'], 2)}) | {fmt(r['inside_bbox0'])} | {r['inside_same']} "
                 f"| {','.join(r['l_missing']) or '-'} | {r['first_inside']} @ {fmt(r['first_inside_pos'])} "
                 f"| {fmt(w['x_rel'])} | {fmt(w['y_rel'])} | {w['cells']} | {w['mismatch']} / {w['samples']} "
                 f"| {w['off_grid']} | {r['entity_mismatch']} / {r['entities']} | {cd} |")
    L += ['', '## Per shape (all rooms of the shape)', '',
          '| shape | rooms | gw x gh | GetGridPosition(0) | TopLeft rel | BottomRight rel | inside bbox | walk x rel min..max '
          '| walk y rel min..max | mismatches | off-grid |', '|' + '---|' * 11]
    by = defaultdict(list)
    for r in rows:
        by[r['shape']].append(r)
    for s in sorted(SHAPES):
        rs = by.get(s, [])
        if not rs:
            L.append(f'| {s} {SHAPES[s]} | 0 | | | | | | | | | |')
            continue
        uniq = lambda f: ' / '.join(sorted({fmt(f(r)) for r in rs}))
        L.append(f"| {s} {SHAPES[s]} | {len(rs)} | {uniq(lambda r: [r['gw'], r['gh']])} | {uniq(lambda r: r['gp0'])} "
                 f"| {uniq(lambda r: r['tl_rel'])} | {uniq(lambda r: r['br_rel'])} | {uniq(lambda r: r['inside_bbox0'])} "
                 f"| {fmt(min(r['walk']['x_rel'][0] for r in rs))}..{fmt(max(r['walk']['x_rel'][1] for r in rs))} "
                 f"| {fmt(min(r['walk']['y_rel'][0] for r in rs))}..{fmt(max(r['walk']['y_rel'][1] for r in rs))} "
                 f"| {sum(r['walk']['mismatch'] or 0 for r in rs)} / {sum(r['walk']['samples'] or 0 for r in rs)} "
                 f"| {sum(r['walk']['off_grid'] or 0 for r in rs)} |")
    L += ['', '## Doors (slot position: GetGridIndex vs the formula; rel = position - O)', '',
          '| room | slot | allowed | slot pos rel | GetGridIndex | formula | door there | door GetGridIndex |',
          '|' + '---|' * 8]
    for r in sorted(rows, key=lambda r: (r['shape'], r['key'])):
        for d in r['doors']:
            if d['allowed'] is True or d['door']:
                L.append(f"| {r['key']} | {d['slot']} | {d['allowed']} | {fmt(d['rel'])} | {d['slot_gi']} | {d['formula']} "
                         f"| {d['door']} | {d['door_gi']} |")
    L += ['', '## Transition frames (entering, a door left through and the way back; unfiltered)', '',
          '| room | frames | x range | y range | GetGridIndex values | rooms |', '|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: (r['shape'], r['key'])):
        t = r['transitions']
        if t:
            L.append(f"| {r['key']} | {t['n']} | {fmt(t['x'])} | {fmt(t['y'])} | {t['gi']} | {t['rooms']} |")
    L += ['', '## GetGridIndex across the edges of cell 0 (offset from its centre: index)', '']
    for r in sorted(rows, key=lambda r: (r['shape'], r['key']))[:6]:
        L.append(f"- {r['key']}: x {r['edge0']['x'] if r['edge0'] else '-'}; y {r['edge0']['y'] if r['edge0'] else '-'}")
    L += ['', '## inside vs the IsPositionInRoom size (start room per character; cells whose inside differs from size 0)',
          '', '| char | PlayerType | Size | ' + ' | '.join(f's={s}' for s in SIZES[1:]) + ' |',
          '|' + '---|' * (3 + len(SIZES) - 1)]
    for s in sizes:
        if 'error' in s:
            L.append(f"| {s['char']} | error: {s['error']} |")
        else:
            L.append(f"| {s['char']} | {s['ptype']} | {s['psize']} | "
                     + ' | '.join(str(s['cells_differing_from_size0'][v]) for v in SIZES[1:]) + ' |')
    if errs:
        L += ['', '## Errors', ''] + [f"- {e.get('key')}: {e.get('error')}" for e in errs]
    (out / 'summary.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L[:8]))
    print(f'rooms {len(rows)}, errors {len(errs)}; {out / "summary.md"}')


if __name__ == '__main__':
    main()
