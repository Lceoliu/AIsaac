"""FastRow check (2026-10-04): does abp_turbo's abp_row_encode write the same ROW records as LeanDecoder.decode +
tok_obs.encode_row, byte for byte, on real episodes?

Per seed a lean clone plays (room mode: sticky random actions until the player dies or --steps, going on --after-clear
decisions after a clear, the doors open; --floor: legs through doors as abplus_probe_native_obs.py, room changes, map and
terrain blocks, trapdoors after a boss). Every payload goes through both encoders, each with its own decoder,
EpisodeState and two record slots used alternately as a sampler worker uses them (so the rows a record leaves untouched
are compared too); after every record the two slots must be equal byte for byte, and so must the done codes and the
hurt flags. Reported: records, how many FastRow did itself (fast) and how many it handed to encode_row (slow).

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_fast_row.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:24 --out <dir>
2026-10-08: --dump keeps the reference records (rows_<seed>.npz) for abplus_probe_laser_flags.py --compare.
2026-10-06 (Phase A): --floor --items --run: the root's bridge sends the inventory block (ISAAC_RL_LEAN_ITEMS=1), the
records are those of a run with items (EpisodeState run, items; ROW's item fields compared too), the random actions also
press the item and pill / card buttons (--item-prob), and after --legs legs a trapdoor is spawned in the room (cleared
first) and walked into: the run goes on on the next floor, --floors floors in all. Reported in addition: the EV_* item
and exit events seen, the stages reached.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from abplus_probe_fork import split_code, sticky_actions
from abplus_probe_native_obs import KILL_LUA, aim, move_to, read_raw
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder
from isaac_bridge.tok_obs import (EV_ACTIVE, EV_EXIT, EV_ITEM, EV_PILL, ROW, EpisodeState, FastRow, encode_row,
                                  fast_row_function)

# a trapdoor (GridEntityType 17) on a free tile near player 0; returns "x,y"
TRAPDOOR_LUA = ("local p = Isaac.GetPlayer(0) local room = Game():GetRoom() "
                "local pos = room:FindFreeTilePosition(p.Position + Vector({dx}, {dy}), 0) "
                "Isaac.GridSpawn(17, 0, pos, true) return tostring(pos.X) .. ',' .. tostring(pos.Y)")


class Pair:
    dump = False   # 2026-10-08 (--dump): keep every reference record (encode_row's) for a later comparison

    def __init__(self, fn, limit, floor, stall, run=False, items=False):
        self.kept = []
        self.dec_a, self.st_a = LeanDecoder(), EpisodeState(limit, floor=floor, stall=stall, run=run, items=items)
        self.rows_a, self.rows_b = np.zeros(2, ROW), np.zeros(2, ROW)
        self.addr_b = [self.rows_b[i:i + 1].ctypes.data for i in range(2)]
        dec_b = LeanDecoder()
        self.enc = FastRow(fn, EpisodeState(limit, floor=floor, stall=stall, run=run, items=items), dec_b, 7, 1234, 2)
        self.events = 0      # OR of the records' events
        self.counts = {}     # EV_* bit -> records with it
        self.stages = set()
        self.last_row = None
        self.dec_b = dec_b
        self.t = 0
        self.mismatch = None
        self.first = True

    def record(self, payload):
        t = self.t
        o = self.dec_a.decode(payload)
        ra = self.rows_a[t & 1:(t & 1) + 1]
        self.st_a.version = None   # the reference rebuilds the terrain channels at every record (no terrain cache)
        da = encode_row(o, self.st_a, ra, t)
        ra['episode'], ra['seed'], ra['group'], ra['first'] = 7, 1234, 2, t == 0
        ha = bool(ra['hurt'][0] > 0)
        rb = self.rows_b[t & 1:(t & 1) + 1]
        # the first record as the worker hands it over (decoded), every later one raw
        item = self.dec_b.decode(payload) if self.first else payload
        self.first = False
        db = self.enc.encode(item, rb, self.addr_b[t & 1], t)
        ev = int(ra['events'][0])
        self.events |= ev
        for bit in (1, 2, 4, EV_EXIT, EV_ITEM, EV_ACTIVE, EV_PILL):
            if ev & bit:
                self.counts[bit] = self.counts.get(bit, 0) + 1
        self.stages.add(int(o.room[5]))
        self.last_row = ra.copy()
        if Pair.dump:
            self.kept.append(ra.copy())
        if self.mismatch is None and (da != db or ha != self.enc.hurt or ra.tobytes() != rb.tobytes()):
            diff = [k for k in ROW.names if ra[k].tobytes() != rb[k].tobytes()]
            self.mismatch = dict(t=t, done=(da, db), hurt=(ha, bool(self.enc.hurt)), fields=diff)
            if 'ent' in diff:
                a, b = ra['ent'][0], rb['ent'][0]
                cells = np.argwhere(a.view(np.uint32) != b.view(np.uint32))[:4].tolist()
                self.mismatch['ent_cells'] = [(r, c, float(a[r, c]), float(b[r, c])) for r, c in cells]
            if 'player' in diff:
                a, b = ra['player'][0], rb['player'][0]
                self.mismatch['player_cols'] = [(int(i), float(a[i]), float(b[i]))
                                                for i in np.flatnonzero(a.view(np.uint32) != b.view(np.uint32))[:4]]
        self.t += 1
        return o, da


def step(c, action, rep):
    if len(action) > 4 and action[4]:   # the pill / card button (2026-10-06)
        c._sock.sendall(b"S %d %d %d %d %d %d\n" % (rep, action[0], action[1], action[2], action[3], action[4]))
    else:
        c._sock.sendall(b"S %d %d %d %d %d\n" % (rep, action[0], action[1], action[2], action[3]))
    return read_raw(c)


def enter_trapdoor(c, pair, o, rep, steps=200):
    """Spawn a trapdoor near the player (the room cleared first) and walk into it; (observation, entered)."""
    if not o.clear:
        if c.lua(KILL_LUA) != 'ok':
            raise RuntimeError('lua failed')
        for _ in range(40):
            o, _ = pair.record(step(c, (0, 0, 0, 0), rep))
            if o.clear:
                break
    stage = o.room[5]
    for dx, dy in ((0, 80), (80, 0), (0, -80), (-80, 0)):
        x, y = (float(v) for v in c.lua(TRAPDOOR_LUA.format(dx=dx, dy=dy)).split(','))
        pl = o.players[0]
        if abs(x - pl['x']) + abs(y - pl['y']) > 30:
            break
    for _ in range(30):   # the trapdoor opens once the room is clear and the player is off it
        o, _ = pair.record(step(c, (0, 0, 0, 0), rep))
    for _ in range(steps):
        pl = o.players[0]
        o, _ = pair.record(step(c, (move_to(pl['x'], pl['y'], x, y), 0, 0, 0), rep))
        if o.room[5] != stage or o.dead:
            break
    # the next floor's first frames (the fall animation): a few no-op steps
    for _ in range(10):
        if o.dead:
            break
        o, _ = pair.record(step(c, (0, 0, 0, 0), rep))
    return o, o.room[5] != stage


def room_seed(inst, seed, args, rep, fn):
    rng = np.random.default_rng(seed)
    warm, test = sticky_actions(rng, args.warm), sticky_actions(rng, args.steps)
    inst.reset(seed)
    _, _, stop = inst.env.bridge.play(warm, repeat=rep, stop_clear=True)
    if stop != 'done':
        return dict(seed=seed, skipped='episode ended in the warm-up')
    c = inst.env.bridge.fork(tag='fastrow', lean=True, alarm=900)
    pair = Pair(fn, 10 ** 6, False, 0)
    c._send({"cmd": "obs"})
    o, _ = pair.record(read_raw(c))
    after = None
    ents = 0
    for code in test:
        o, _ = pair.record(step(c, split_code(code), rep))
        ents = max(ents, len(o.entities))
        if o.dead:
            break
        if o.clear and after is None:
            after = args.after_clear
        if after is not None:
            after -= 1
            if after < 0:
                break
    c.close()
    dump_rows(args, seed, pair)
    return dict(seed=seed, records=pair.t, fast=pair.enc.fast, slow=pair.enc.slow, reasons=pair.enc.reasons,
                max_entities=ents, mismatch=pair.mismatch)


def dump_rows(args, seed, pair):
    """--dump: the seed's reference records as rows_<seed>.npz, one array per ROW field (abplus_probe_laser_flags.py
    --compare compares two such dumps field by field, e.g. before and after a change of the entity columns)."""
    if not Pair.dump or not pair.kept:
        return
    rows = np.concatenate(pair.kept)
    np.savez_compressed(Path(args.out) / f'rows_{seed}.npz', **{k: rows[k] for k in rows.dtype.names})


def floor_seed(inst, seed, args, rep, fn):
    inst.env.bridge.reset_mode = 'floor'
    inst.reset(seed)
    rng = np.random.default_rng(seed)
    c = inst.env.bridge.fork(tag='fastrow', lean=True, alarm=900)
    pair = Pair(fn, 10 ** 6, True, 450, run=args.run, items=args.items)
    c._send({"cmd": "obs"})
    o, _ = pair.record(read_raw(c))
    legs, rooms, ents, floors, floor_legs = 0, set(), 0, 1, 0
    while not o.dead:
        if args.run and floor_legs >= args.legs:   # this floor's legs done (or a leg failed): through a trapdoor
            if floors >= args.floors:
                break
            o, entered = enter_trapdoor(c, pair, o, rep)
            if not entered or o.dead:
                break
            floors += 1
            floor_legs = 0
        floor_legs += 1
        if not args.run and floor_legs > args.legs:
            break
        doors = [d for d in o.doors if d['open'] and not d['locked']]
        if not doors:
            floor_legs = args.legs
            continue
        d = doors[int(rng.integers(len(doors)))]
        here = o.room[4]
        for _ in range(150):
            pl = o.players[0]
            o, _ = pair.record(step(c, (move_to(pl['x'], pl['y'], float(d['x']), float(d['y'])), 0, 0, 0), rep))
            if o.room[4] != here or o.dead:
                break
        if o.dead:
            break
        if o.room[4] == here:
            floor_legs = args.legs
            continue
        legs += 1
        rooms.add(o.room[4])
        after = None
        for code in sticky_actions(rng, args.steps):
            move, _, bomb, item = split_code(code)
            if args.items:   # the item and pill / card buttons now and then
                item, pill = int(rng.random() < args.item_prob), int(rng.random() < args.item_prob)
            else:
                pill = 0
            o, _ = pair.record(step(c, (move, aim(o), bomb, item, pill), rep))
            ents = max(ents, len(o.entities))
            if o.dead:
                break
            if o.clear and after is None:
                after = args.after_clear
            if after is not None:
                after -= 1
                if after < 0:
                    break
        if not o.clear and not o.dead:
            if c.lua(KILL_LUA) != 'ok':
                raise RuntimeError('lua failed')
            for _ in range(args.after_clear):
                o, _ = pair.record(step(c, (0, 0, 0, 0), rep))
    c.close()
    dump_rows(args, seed, pair)
    return dict(seed=seed, legs=legs, rooms=len(rooms), records=pair.t, fast=pair.enc.fast, slow=pair.enc.slow,
                reasons=pair.enc.reasons, max_entities=ents, mismatch=pair.mismatch, floors=floors,
                stages=sorted(pair.stages), events={str(k): v for k, v in pair.counts.items()},
                inv=pair.last_row['inv'][0][:6].tolist(), pitem=pair.last_row['pitem'][0].tolist(),
                pinv=[round(float(v), 3) for v in pair.last_row['pinv'][0]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:16')
    p.add_argument('--floor', action='store_true')
    p.add_argument('--items', action='store_true', help='2026-10-06: the inventory block and ROW item fields')
    p.add_argument('--run', action='store_true', help='2026-10-06: run records; with --floors > 1 through trapdoors')
    p.add_argument('--floors', type=int, default=1)
    p.add_argument('--item-prob', type=float, default=0.05)
    p.add_argument('--legs', type=int, default=6)
    p.add_argument('--warm', type=int, default=10)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--after-clear', type=int, default=30)
    p.add_argument('--fpd', type=int, default=4)
    p.add_argument('--port', type=int, default=34200)
    p.add_argument('--name', default='stpfast')
    p.add_argument('--dump', action='store_true',
                   help='2026-10-08: save every reference record per seed (rows_<seed>.npz in --out)')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    Pair.dump = args.dump
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    if args.items:
        os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
    if args.run and args.floors > 1 and not args.floor:
        raise SystemExit('--floors needs --floor')
    fn = fast_row_function(default_preload())
    if fn is None:
        raise SystemExit('abp_row_encode not available')
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                   stub_list=stub if Path(stub).is_file() else '')
    inst = Instance(args.name, args.port, cfg, spec)
    rows = []
    t0 = time.perf_counter()
    try:
        for seed in parse_seeds(args.seeds):
            row = (floor_seed if args.floor else room_seed)(inst, seed, args, args.fpd, fn)
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
    finally:
        inst.close()
        done = [r for r in rows if 'skipped' not in r]
        summary = dict(group='floor' if args.floor else args.group, seeds=len(rows), tested=len(done),
                       records=sum(r['records'] for r in done), fast=sum(r['fast'] for r in done),
                       slow=sum(r['slow'] for r in done), equal=sum(r['mismatch'] is None for r in done),
                       max_entities=max([r['max_entities'] for r in done] or [0]),
                       floors=sum(r.get('floors', 1) for r in done),
                       stages=sorted({s for r in done for s in r.get('stages', [])}),
                       events={k: sum(r.get('events', {}).get(k, 0) for r in done)
                               for k in sorted({k for r in done for k in r.get('events', {})})},
                       seconds=round(time.perf_counter() - t0, 1))
        (out / 'rows.json').write_text(json.dumps(rows, indent=1, default=str))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
