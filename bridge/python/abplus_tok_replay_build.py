"""HTML replay of recorded token-policy episodes (eval_tok.py --record): what the policy observed, decision by decision.

Reads the per-episode npz files of one or more record directories (tok_record.py), converts every record to the
viewer's compact JSON (positions in grid cells, the room grid and the minimap as per-cell bit masks, each distinct one
once), and writes
  <out>/episodes.json.gz   the data (the viewer's file picker also opens it, or a plain .json)
  <out>/index.html         isaac_bridge/tok_replay.html with the same data embedded (base64 of the gzip), so the page
                           opens from disk with no server

Geometry: ROW positions are room-normalised ((x - TopLeft) / (BottomRight - TopLeft)); the inner area spans grid cells
1 .. W-2, so a position in cells is 1 + n * (W - 2) (W, H: the grid's width and height, player features 26, 27). Door
positions are relative to the player (/ 200 px; 40 px a cell).

usage (from the bridge's python dir):
  python abplus_tok_replay_build.py --record <rec dir> [--record ...] --out <dir> --title "..." --subtitle "..."
"""
import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np

from abplus_replay_view import NAMES, VARIANT_NAMES

KIND_COLUMN = 16
EXIT_TYPE = 1023
MAP_CELLS = 169


def cells(v):
    """cells -> integer hundredths of a cell (JSON size)"""
    return int(round(float(v) * 100))


def convert(path, frames_per_decision, limit):
    z = np.load(path)
    meta = json.loads(str(z['meta']))
    P = z['player']
    T = len(P)
    W = np.rint(P[:, 26] * 28).astype(int)
    H = np.rint(P[:, 27] * 16).astype(int)
    grid_idx, map_idx = z['grid_idx'].astype(int), z['map_idx'].astype(int)
    grids = []
    for g, grid in enumerate(z['grids']):
        k = int(np.argmax(grid_idx == g))
        w, h = int(W[k]), int(H[k])
        bits = np.zeros((h, w), np.int64)
        for ch in range(grid.shape[0]):
            bits |= grid[ch, :h, :w].astype(np.int64) << ch
        grids.append(dict(w=w, h=h, c=bits.ravel().tolist()))
    maps, map_room, map_boss = [], [], []
    for m in z['maps']:
        bits = np.zeros(m.shape[1:], np.int64)
        for ch in range(m.shape[0]):
            bits |= m[ch].astype(np.int64) << ch
        maps.append(bits.ravel().tolist())
        here = np.flatnonzero(m[3].ravel())
        map_room.append(int(here.min()) if len(here) else -1)   # the player's room: its top-left map cell
        map_boss.append(bool((m[3] & m[4]).any()))
    px = 1 + P[:, 0] * (W - 2)
    py = 1 + P[:, 1] * (H - 2)
    # entities
    n_ent = z['n_ent'].astype(int)
    starts = np.concatenate([[0], np.cumsum(n_ent)])
    ent, ent_id = z['ent'], z['ent_id'].astype(int)
    types, type_index = [], {}
    E = []
    boss_hp = np.full(T, -1.0)
    for k in range(T):
        rows = []
        w2, h2 = W[k] - 2, H[k] - 2
        for j in range(starts[k], starts[k + 1]):
            e = ent[j]
            tvs = tuple(ent_id[j])
            kind = 7 if tvs[0] == EXIT_TYPE else int(np.argmax(e[KIND_COLUMN:KIND_COLUMN + 7]))
            key = '%d.%d.%d' % tvs
            if key not in type_index:
                type_index[key] = len(types)
                types.append(key)
            flags = int(e[23] > 0.5) | int(e[24] > 0.5) << 1 | int(e[25] > 0.5) << 2 | int(e[26] > 0.5) << 3
            hp = int(round(float(e[27]) * 100)) if flags & 4 else 0
            if flags & 4:
                boss_hp[k] = max(boss_hp[k], float(e[27]))
            r = [kind, cells(1 + e[5] * w2), cells(1 + e[6] * h2), cells(e[7] * 0.5), type_index[key], flags, hp]
            if kind in (2, 3):   # projectiles, lasers: velocity, cells per logic frame
                r += [cells(e[3] * 10 / 40), cells(e[4] * 10 / 40)]
            rows.append(r)
        E.append(rows)
    # doors: absolute cells, each distinct set once
    door_sets, door_index, door_idx = [], {}, []
    for k in range(T):
        d = z['doors'][k, :int(z['n_doors'][k])]
        s = [[int(round((px[k] + v[0] * 5) * 10)) * 10, int(round((py[k] + v[1] * 5) * 10)) * 10, int(v[2] > .5),
              int(v[3] > .5), int(round(v[4] * 30)), int(v[5] > .5), int(v[6] > .5)] for v in d]
        key = json.dumps(s)
        if key not in door_index:
            door_index[key] = len(door_sets)
            door_sets.append(s)
        door_idx.append(door_index[key])
    act = z['act'].astype(int)
    act_code = [-1 if a[0] < 0 else int(a[0] * 100 + a[1] * 10 + a[2]) for a in act]
    # rooms: number the player's room changes (minimap: the player's room)
    room_no, rooms_seen, cur, first_in = [], {}, None, []
    for k in range(T):
        rid = map_room[map_idx[k]]
        if rid not in rooms_seen:
            rooms_seen[rid] = len(rooms_seen)
        room_no.append(rooms_seen[rid])
        if rid != cur:
            first_in.append([k, rooms_seen[rid], int(map_boss[map_idx[k]])])
            cur = rid
    rooms_visited = len(rooms_seen)
    if int(meta['done']) == 1 and T > 1 and map_idx[-1] != map_idx[-2]:
        # a cleared floor's last record is already on the next floor (the stage changed): its start room
        if first_in[-1][0] == T - 1:
            first_in[-1][1:] = [-1, 0]
        else:
            first_in.append([T - 1, -1, 0])
        rooms_visited = len(set(room_no[:-1]))
    F = dict(
        gi=grid_idx.tolist(), mi=map_idx.tolist(), di=door_idx, px=[cells(v) for v in px], py=[cells(v) for v in py],
        hearts=np.rint(P[:, 4] * 12).astype(int).tolist(), soul=np.rint(P[:, 5] * 12).astype(int).tolist(),
        maxh=np.rint(P[:, 6] * 12).astype(int).tolist(), bombs=z['bombs'].astype(int).tolist(),
        keys=np.rint(P[:, 9] * 10).astype(int).tolist(), coins=np.rint(P[:, 10] * 50).astype(int).tolist(),
        inv=np.rint(P[:, 18]).astype(int).tolist(), clear=np.rint(P[:, 23]).astype(int).tolist(),
        enemies=np.rint(P[:, 28] * 10).astype(int).tolist(), blocking=np.rint(P[:, 29] * 10).astype(int).tolist(),
        stall=np.rint(P[:, 30] * 100).astype(int).tolist(), tshare=np.rint(P[:, 24] * 100).astype(int).tolist(),
        hurt=[round(float(v), 2) for v in z['hurt']], dmg=np.rint(z['damage'] * 1000).astype(int).tolist(),
        ev=z['events'].astype(int).tolist(), act=act_code, room=room_no,
        bosshp=[-1 if v < 0 else int(round(v * 100)) for v in boss_hp],
        size=[cells(v * 0.5) for v in P[:, 25]],
        stats=[[round(float(P[k, 11] * 10), 2), round(float(P[k, 12] * 20), 1), round(float(P[k, 13] * 2), 2),
                round(float(P[k, 14] * 500)), round(float(P[k, 15] * 2), 2), round(float(P[k, 16] * 5), 1),
                int(P[k, 17] > .5)] for k in range(T)])
    # stats change rarely: run-length (frame, values)
    st, last = [], None
    for k, s in enumerate(F.pop('stats')):
        if s != last:
            st.append([k] + s)
            last = s
    done = int(meta['done'])
    decisions = int(meta.get('decisions', T - 1))
    if done == 1:
        outcome = 'clear'
    elif done == 2:
        outcome = 'death'
    elif done == 3:
        outcome = 'timeout' if decisions >= limit else 'stall'
    else:
        outcome = 'aborted'
    step_s = frames_per_decision / 30
    return dict(id=f"{meta['seed']}", seed=int(meta['seed']), outcome=outcome, done=done, decisions=decisions,
                seconds=round(decisions * step_s, 1), rooms=meta.get('rooms'), entered=meta.get('entered'),
                boss=meta.get('boss'), hurt=meta.get('hurt'), step_s=step_s, records=T,
                rooms_visited=rooms_visited, boss_room_reached=any(b for _, _, b in first_in),
                grids=grids, maps=maps, doors=door_sets, types=types, F=F, E=E, stats=st, entries=first_in)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--record', action='append', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--title', default='整层回放（策略的观测）')
    p.add_argument('--subtitle', default='')
    p.add_argument('--template', default=str(Path(__file__).parent / 'isaac_bridge' / 'tok_replay.html'))
    args = p.parse_args()
    episodes, runs = [], []
    for d in args.record:
        d = Path(d)
        run = json.loads((d / 'run.json').read_text())
        runs.append(run)
        fpd = int(run.get('frames_per_decision', 4))
        limit = int(round(float(run.get('floor_seconds', 480)) * 30 / fpd))
        for f in sorted(d.glob('ep*.npz')):
            episodes.append(convert(f, fpd, limit))
    episodes.sort(key=lambda e: e['seed'])
    names = {str(k): v for k, v in NAMES.items()}
    variants = {f'{t}.{v}': n for (t, v), n in VARIANT_NAMES.items()}
    data = dict(title=args.title, subtitle=args.subtitle, runs=runs, names=names, variants=variants,
                episodes=episodes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(data, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    gz = gzip.compress(raw, 9)
    (out / 'episodes.json.gz').write_bytes(gz)
    page = Path(args.template).read_text(encoding='utf-8')
    page = page.replace('/*DATA*/', base64.b64encode(gz).decode('ascii'))
    (out / 'index.html').write_text(page, encoding='utf-8')
    print(f'{len(episodes)} episodes, json {len(raw) / 1e6:.1f} MB, gzip {len(gz) / 1e6:.1f} MB -> {out}')
    for e in episodes:
        print(e['seed'], e['outcome'], e['seconds'], 's', e['rooms'], '/', e['entered'], 'boss', e['boss'],
              'hurt', e['hurt'])


if __name__ == '__main__':
    main()
