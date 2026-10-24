"""Wall-trigger probe (B40, 2026-10-18; train_tok --cf-wall 1, isaac_bridge/tok_cf.py 'wall' / BombIntent).

Per seed (--seeds first:count, scanned until --want seeds with a secret room next to the starting room and --controls
seeds without one are found): the root's floor reset (Isaac, 1 bomb), the level's rooms from the engine (Lua:
GetRooms(), GridIndex / Data.Type / Data.Shape; ground truth for the probe only, never a policy or trigger input), the
template S, a lean fork W with the episode's reseed, its records encoded as a worker does (EpisodeState run mode,
items), --settle decisions of no action. Then on the starting room's record:
  detect      tok_cf.detect(wall=True) / wall_candidates: the candidates (side, neighbour cell, p_vis), the chosen one,
              whether the true secret (or super secret) room's wall is among them and its rank;
  branches    from a parked copy of W (W.fork(tag='snap')), lean forks running BombIntent closed-loop with the
              one-decision action lag of tok_teacher2.play_branch (the act decided at record j is applied in the step
              after record j + 1) and a stand-in policy (walk to the room's centre, else stand; no shots), for
              --seconds game seconds:
                secret : the bomb spot of the true secret room's wall (when it is a candidate);
                chosen : detect's choice (when different from secret);
              per branch: bombed at, bombs before / after (ROW 'bombs'), opened (the intent's rule), the neighbour
              cell shown on the minimap, doors before / after, other-special cells (map channel 7), half hearts lost.
Expected: secret: bombed, bombs 1 -> 0, opened (the cell shown or a door on that wall); a control seed's chosen wall:
bombed, not opened.

usage (bridge python dir, PYTHONPATH=., ABP_BRIDGE_LUA = the sandbox's bridge):
  python abplus_probe_wall.py --groups-file ../abplus/catalog/scaling2_groups.json --seeds 2147490000:60 --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.tok_cf import (BombIntent, MAP_N, SIDES, detect, inner_area, player_tiles, room_walls,
                                 wall_candidates)
from isaac_bridge.tok_obs import ROW, EpisodeState, encode_row
from isaac_bridge.tok_pickup import move_toward
from isaac_bridge.tok_sampler import lean_step

ROOMS_LUA = """
local level = Game():GetLevel()
local rooms = level:GetRooms()
local out = {}
for i = 0, rooms.Size - 1 do
  local d = rooms:Get(i)
  out[#out + 1] = tostring(d.GridIndex) .. ':' .. tostring(d.Data.Type) .. ':' .. tostring(d.Data.Shape)
end
local start = level.GetStartingRoomIndex and level:GetStartingRoomIndex() or level:GetCurrentRoomIndex()
return tostring(start) .. '|' .. table.concat(out, ',')
"""
FPD = 4


def parse_rooms(text):
    start, rest = text.split('|', 1)
    rooms = []
    for part in rest.split(','):
        bits = part.split(':')
        if len(bits) == 3:
            rooms.append(tuple(int(float(b)) for b in bits))
    return int(start), rooms


class Track:
    """One clone's records through encode_row (run mode, items), as a worker encodes them."""

    def __init__(self, clone):
        self.c, self.dec = clone, LeanDecoder()
        self.st = EpisodeState(10 ** 6, floor=True, run=True, items=True)
        self.row = np.zeros(1, ROW)
        self.t = 0
        clone._send({'cmd': 'obs'})
        self.o = read_lean(clone, self.dec)
        encode_row(self.o, self.st, self.row, 0)

    def step(self, a):
        self.o = lean_step(self.c, self.dec, a, FPD)
        self.t += 1
        return encode_row(self.o, self.st, self.row, self.t)


def stand_in(r):
    """The stand-in policy: walk to the room's centre (inner area), stand within a tile of it."""
    area = inner_area(r)
    if area is None:
        return (0, 0, 0, 0, 0)
    px, py = player_tiles(r, area)
    dx, dy = area[0] / 2 - px, area[1] / 2 - py
    return ((move_toward(dx, dy) if np.hypot(dx, dy) > 1.0 else 0), 0, 0, 0, 0)


def snapshot(r, cell):
    m = r['map']
    walls = room_walls(r)
    y, x = divmod(int(cell), MAP_N)
    return dict(bombs=int(r['bombs']), shown=int((m[0] > 0).sum()), cell_shown=bool(m[0, y, x] > 0),
                special=int((m[7] > 0).sum()), doors=int(r['n_doors']),
                wall_doors=sum(1 for w in walls if w['door'] and w['nb'] == (y, x)))


def run_branch(snap, target, seconds):
    """BombIntent closed-loop on a lean fork of snap (play_branch's action lag), the stand-in policy."""
    B = snap.fork(reseed=None, lean=True, alarm=600)
    try:
        tr = Track(B)
        it = BombIntent(target, h=int(round(20 * 30 / FPD)))
        cell = target[1]['cell']
        before = snapshot(tr.row[0], cell)
        under = (0, 0, 0, 0, 0)
        hurt, n, end = 0.0, int(round(seconds * 30 / FPD)), 'cap'
        for j in range(n):
            act = it.compose(j, stand_in(tr.row[0]), tr.row)
            done = tr.step(under)
            hurt += float(tr.row['hurt'][0])
            under = act
            if done or tr.o.dead:
                end = 'dead' if tr.o.dead else f'done {done}'
                break
        after = snapshot(tr.row[0], cell)
        return dict(it.summary(), before=before, after=after, hurt=hurt, end=end, records=tr.t)
    finally:
        try:
            B.close()
        except OSError:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147490000:60')
    p.add_argument('--want', type=int, default=4, help='seeds with a hidden room next to the start to probe')
    p.add_argument('--controls', type=int, default=2, help='seeds without one to probe')
    p.add_argument('--settle', type=int, default=3)
    p.add_argument('--seconds', type=float, default=30.0)
    p.add_argument('--port', type=int, default=49690)
    p.add_argument('--name', default='hb40p')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=FPD, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    first, count = (int(v) for v in args.seeds.split(':'))
    inst = Instance(args.name, args.port, gx, spec)
    report, n_pos, n_neg, scanned = [], 0, 0, 0
    try:
        for seed in range(first, first + count):
            if n_pos >= args.want and n_neg >= args.controls:
                break
            t0 = time.perf_counter()
            b = inst.env.bridge
            b.reset_mode = 'floor'
            b.character = 0
            inst.reset(seed)
            scanned += 1
            start, rooms = parse_rooms(b.lua(ROOMS_LUA) or '')
            sy, sx = divmod(start, MAP_N)
            near = {(sy - 1, sx): 'up', (sy + 1, sx): 'down', (sy, sx - 1): 'left', (sy, sx + 1): 'right'}
            hidden = [(gi, typ) for gi, typ, _ in rooms if typ in (7, 8) and divmod(gi, MAP_N) in near]
            if hidden and n_pos >= args.want or not hidden and n_neg >= args.controls:
                continue
            rec = dict(seed=seed, start=start, hidden=[dict(cell=gi, type=typ, side=near[divmod(gi, MAP_N)])
                                                       for gi, typ in hidden],
                       secret_cells=[gi for gi, typ, _ in rooms if typ == 7],
                       supersecret_cells=[gi for gi, typ, _ in rooms if typ == 8])
            S = b.fork(tag='template', alarm=0)
            clones = []
            try:
                R = seed % (2 ** 31 - 1) + 1
                W = S.fork(reseed=R, lean=True, alarm=900)
                clones.append(W)
                tw = Track(W)
                for _ in range(args.settle):
                    tw.step((0, 0, 0, 0, 0))
                r = tw.row[0]
                cands = wall_candidates(r)
                rec['room'] = dict(clear=float(r['player'][23]), bombs=int(r['bombs']), room_type=int(r['room_type']),
                                   area=inner_area(r), doors=int(r['n_doors']),
                                   walls=[(SIDES[w['side']], w['nb'][0] * MAP_N + w['nb'][1], w['door'], w['slots'])
                                          for w in room_walls(r)])
                rec['candidates'] = [dict(side=SIDES[w['side']], cell=w['nb'][0] * MAP_N + w['nb'][1],
                                          p=round(w['p'], 4)) for w in cands]
                found = detect(tw.row, set(), None, wall=True, walled=set())
                chosen = next((tg for t_, tg in found if tg is not None and tg[0] == 'wall'), None)
                rec['chosen'] = None if chosen is None else dict(side=SIDES[chosen[1]['side']], cell=chosen[1]['cell'],
                                                                 p=round(chosen[1]['p'], 4), x=chosen[1]['x'],
                                                                 y=chosen[1]['y'])
                cells = [c['cell'] for c in rec['candidates']]
                rec['hidden_rank'] = [cells.index(h['cell']) if h['cell'] in cells else None for h in rec['hidden']]
                snap = W.fork(tag='snap', alarm=0)
                clones.append(snap)
                runs = {}
                for h in rec['hidden']:
                    if h['cell'] in cells:
                        w = cands[cells.index(h['cell'])]
                        tg = ('wall', dict(x=w['target'][0], y=w['target'][1], side=w['side'], cell=h['cell'],
                                           p=w['p'], n=len(cands), room=(1, start), wall=w['wall']))
                        runs[f"secret-{h['type']}"] = run_branch(snap, tg, args.seconds)
                if chosen is not None and chosen[1]['cell'] not in [h['cell'] for h in rec['hidden']]:
                    runs['chosen'] = run_branch(snap, chosen, args.seconds)
                rec['runs'] = runs
            except Exception as e:
                rec['error'] = f'{type(e).__name__}: {e}'
            finally:
                for c in clones:
                    try:
                        c.close()
                    except OSError:
                        pass
                S.close()
            if hidden:
                n_pos += 1
                rs = [v for k, v in rec.get('runs', {}).items() if k.startswith('secret')]
                rec['ok'] = bool(rs) and all(v['bombed'] >= 0 and v['opened'] and v['after']['bombs'] < v['before']['bombs']
                                             for v in rs)
            else:
                n_neg += 1
                c = rec.get('runs', {}).get('chosen')
                rec['ok'] = c is None or (c['bombed'] >= 0 and not c['opened'])
            rec['ok'] = rec['ok'] and 'error' not in rec
            rec['seconds'] = round(time.perf_counter() - t0, 1)
            report.append(rec)
            print(json.dumps(rec), flush=True)
    finally:
        inst.kill()
    pos = [r for r in report if r['hidden']]
    summary = dict(scanned=scanned, positives=len(pos), controls=len(report) - len(pos),
                   ok=sum(r['ok'] for r in report), errors=sum('error' in r for r in report),
                   hidden_candidate=sum(all(k is not None for k in r.get('hidden_rank', [None])) for r in pos),
                   hidden_chosen=sum(bool(r.get('hidden_rank')) and 0 in r['hidden_rank'] for r in pos),
                   opened=sum(any(v['opened'] for k, v in r.get('runs', {}).items() if k.startswith('secret'))
                              for r in pos))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    (out / 'cases.json').write_text(json.dumps(report, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
