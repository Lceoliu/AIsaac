"""Probe of the counterfactual branches (isaac_bridge/tok_branch.py, 2026-10-07): restore, prefixes, pickup_block,
common random numbers. One instance in the bridge's floor mode with items (ISAAC_RL_LEAN_ITEMS=1) and the workers'
instance defaults. Per seed:

item     a pedestal (The Sad Onion) is spawned next to the player in the start room, the state parked (base); an
         "episode" clone walks into it (scripted moves) until the record that shows the item gained (t). A Point of kind
         item is built as the worker builds it (d = t - 1 - item_back) and
           restore     tok_branch.restore: the observation at d equals the episode's record d (players, entities, room,
                       totals, inventory);
           take        prefix(option 0): the observation after the prefix equals the episode's record t, the item taken;
           CRN         two take branches with the same reseed play the same --steps random actions: every observation
                       equal, and the hidden-state digest (ABPGX_DIGEST) at the end; a third with another reseed: where
                       it first differs (it may not: an empty room draws little);
           skip        prefix(option 1): the item not taken, the pedestal still holds it; then --push decisions held
                       toward the pedestal: still not taken; then out through a door and back in (the room rebuilt):
                       still not taken;
           take vs skip from the same reseed and the same actions: where they first differ (the item's effect).
door     an episode clone walks from the start room through its first open door (record t: the room change); a Point of
         kind door (d = t - 1) is built: restore equals record d; option 0's prefix equals record t; each other door's
         walk (Walker) enters a room (which one, how many decisions); two branches of option 1 with the same reseed and
         the same random actions are equal.
Also the cost of each part (wall seconds).

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_branch.py --groups-file ../abplus/catalog/scaling2_groups.json --seeds 2147600000:3 --out <dir>
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.tok_branch import (CI, KIND_DOOR, KIND_ITEM, PedestalTracker, Point, Walker, inv_counts, move_to, options_of, prefix,
                                     restore, walk_to)
from isaac_bridge.tok_obs import EV_ITEM, ROW, EpisodeState, encode_row
from isaac_bridge.tok_sampler import apply_instance_defaults, default_stub_list, lean_step

SPAWN_LUA = ("local p = Isaac.GetPlayer(0) local room = Game():GetRoom() "
             "local pos = room:FindFreeTilePosition(p.Position + Vector({dx}, {dy}), 0) "
             "Isaac.Spawn(5, 100, {sub}, pos, Vector(0, 0), nil) return tostring(pos.X) .. ',' .. tostring(pos.Y)")
FPD = 4


class Held:
    """Stands in for tok_floor.Parked (tok_branch.restore reads .clone)."""

    def __init__(self, clone):
        self.clone = clone


def digest(o):
    return (o.players.tobytes(), o.entities.tobytes(), tuple(o.room), o.damage_taken, o.monsters_hp, o.clear,
            tuple(sorted(inv_counts(o).items())))


def hidden(c):
    """md5 of the clone's hidden-state digest (ABPGX_DIGEST: players, entities, RNG states, room records)."""
    return hashlib.md5(str(c.lua('return ABPGX_DIGEST(false)')).encode()).hexdigest()


def first_diff(a, b):
    for j, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return j
    return None if len(a) == len(b) else min(len(a), len(b))


def play_random(c, dec, o, acts):
    trace = [digest(o)]
    for a in acts:
        if o.dead:
            break
        o = lean_step(c, dec, a, FPD)
        trace.append(digest(o))
    return trace, o


def pedestal_items(o):
    e = o.entities
    sel = (e['type'] == 5) & (e['variant'] == 100)
    return [int(v) for v in e['subtype'][sel]]


def item_seed(template, seed, args, rng):
    rec = dict(seed=seed)
    t0 = time.perf_counter()
    E = template.fork(lean=True, alarm=600, reseed=seed % 1000 + 1)
    dec = LeanDecoder()
    E._send({"cmd": "obs"})
    read_lean(E, dec)
    x, y = (float(v) for v in E.lua(SPAWN_LUA.format(sub=1, dx=0, dy=120)).split(','))
    o = lean_step(E, dec, (0, 0, 0, 0, 0), FPD)
    base = Held(E.fork(tag='base', alarm=0))   # the room's restore point (reseed None, offset 0)
    st = EpisodeState(10 ** 6, floor=True, run=True, items=True)
    row = np.zeros((1,), ROW)
    encode_row(o, st, row, 0)
    tracker = PedestalTracker()   # as the worker: the last record with the item on a pedestal next to the player
    tracker.update(row, 0)
    obs_at, applied, t, emptied = [o], [], None, None
    for k in range(1, 120):
        pl = o.players[0]
        a = (move_to(float(pl['x']), float(pl['y']), x, y), 0, 0, 0, 0)
        applied.append(a)
        o = lean_step(E, dec, a, FPD)
        obs_at.append(o)
        encode_row(o, st, row, k)
        if emptied is None and 1 not in pedestal_items(o):
            emptied = k
        if row['events'][0] & EV_ITEM:
            t = k
            break
        tracker.update(row, k)
    rec['pedestal'] = (x, y)
    rec['pickup_record'] = t
    rec['pedestal_emptied_record'] = emptied
    if t is None or tracker.touch(1) is None:
        rec['error'] = 'the scripted walk did not take the item'
        return rec
    rec['touch_record'] = tracker.touch(1)[0]
    back = args.item_back
    d = max(0, tracker.touch(1)[0] - back)
    point = Point(kind=KIND_ITEM, base=base, reseed=None, offset=0, applied=applied, d=d, t=t, item=1, visited=(),
                  row=None, key=('item', 1), stage=1, seed=seed, episode=0, room=0, n_alt=0, pid=1, order=1)
    t1 = time.perf_counter()
    S, _, oS = restore(point, FPD, 600)
    rec['restore_s'] = round(time.perf_counter() - t1, 4)
    rec['restore_equal'] = digest(oS) == digest(obs_at[d])
    opts = options_of(point, oS, rng)
    rec['options'] = [(h, round(p[0]), round(p[1]), j) for h, p, j in opts]
    rec['pickups_at_d'] = [(int(e['type']), int(e['variant']), int(e['subtype'])) for e in oS.entities
                           if e['type'] == 5]
    rec['pickups_at_d_episode'] = [(int(e['type']), int(e['variant']), int(e['subtype'])) for e in obs_at[d].entities
                                   if e['type'] == 5]
    if len(opts) < 2:
        rec['error'] = 'options not found'
        S.close()
        base.clone.close()
        E.close()
        return rec
    R = int(rng.integers(1, 2 ** 31 - 1))
    acts = [(int(rng.integers(9)), int(rng.integers(5)), 0, 0, 0) for _ in range(args.steps)]
    runs = {}
    for name, opt, reseed in (('take1', opts[0], R), ('take2', opts[0], R), ('take_other', opts[0], R + 1),
                              ('skip', opts[1], R)):
        t2 = time.perf_counter()
        B, bdec, o0, comp, status, notes = prefix(S, oS, point, opt, reseed, FPD, 600, 75)
        prefix_s = time.perf_counter() - t2
        try:
            info = dict(status=status, took=bool(comp[CI['took']]), prefix_s=round(prefix_s, 4),
                        block_found=notes.get('block_found'),
                        prefix_equal_record_t=digest(notes['o_prefix']) == digest(obs_at[t]))
            if o0 is None:
                runs[name] = info
                continue
            trace, o_end = play_random(B, bdec, o0, acts)
            info.update(steps=len(trace) - 1, hidden_digest=hidden(B),
                        inventory=sorted(inv_counts(o_end).items()))
            if name == 'skip':   # hold toward the pedestal, then out through a door and back in
                pk = []
                o = o_end
                for _ in range(args.push):
                    pl = o.players[0]
                    o = lean_step(B, bdec, (move_to(float(pl['x']), float(pl['y']), x, y), 0, 0, 0, 0), FPD)
                pk.append(dict(after='push', taken=inv_counts(o).get(1, 0) > 0, pedestal=pedestal_items(o)))
                doors = [dd for dd in o.doors if dd['open'] and not dd['locked']]
                if doors:
                    room0 = o.room[4]
                    dd = doors[0]
                    o, n1, ok1 = walk_to(B, bdec, o, (float(dd['x']), float(dd['y'])),
                                         lambda q: q.room[4] != room0, 150, FPD, lean_step)
                    for _ in range(10):
                        o = lean_step(B, bdec, (0, 0, 0, 0, 0), FPD)
                    room1 = o.room[4]
                    pl = o.players[0]
                    back_door = min(o.doors, key=lambda q: (q['x'] - pl['x']) ** 2 + (q['y'] - pl['y']) ** 2)
                    o, n2, ok2 = walk_to(B, bdec, o, (float(back_door['x']), float(back_door['y'])),
                                         lambda q: q.room[4] != room1, 150, FPD, lean_step)
                    for _ in range(10):
                        o = lean_step(B, bdec, (0, 0, 0, 0, 0), FPD)
                    back_in = o.room[4] == room0
                    o, n3, got = walk_to(B, bdec, o, (x, y), lambda q: inv_counts(q).get(1, 0) > 0, 60, FPD,
                                         lean_step)
                    pk.append(dict(after='out_and_back', out=ok1, back=ok2 and back_in, taken=got,
                                   pedestal=pedestal_items(o)))
                info['skip_checks'] = pk
            runs[name] = dict(info, trace=trace)
        finally:
            B.close()
    S.close()
    rec['runs'] = {k: {kk: vv for kk, vv in v.items() if kk != 'trace'} for k, v in runs.items()}
    tr = {k: v.get('trace') for k, v in runs.items()}
    if tr.get('take1') and tr.get('take2'):
        rec['crn_same_reseed_equal'] = tr['take1'] == tr['take2'] and \
            runs['take1']['hidden_digest'] == runs['take2']['hidden_digest']
        rec['crn_other_reseed_first_diff'] = first_diff(tr['take1'], tr.get('take_other') or [])
        rec['take_vs_skip_first_diff'] = first_diff(tr['take1'], tr.get('skip') or [])
    # the episode itself is unchanged by its clones: it takes the same random actions as take1 from record t
    E._send({"cmd": "obs"})
    rec['seconds'] = round(time.perf_counter() - t0, 2)
    base.clone.close()
    E.close()
    return rec


def door_seed(template, seed, args, rng):
    rec = dict(seed=seed)
    reseed0 = seed % 1000 + 7
    E = template.fork(lean=True, alarm=600, reseed=reseed0)
    dec = LeanDecoder()
    E._send({"cmd": "obs"})
    o = read_lean(E, dec)
    doors = [dd for dd in o.doors if dd['open'] and not dd['locked']]
    rec['doors'] = len(doors)
    if len(doors) < 2:
        rec['error'] = 'fewer than two open doors'
        E.close()
        return rec
    target = (float(doors[0]['x']), float(doors[0]['y']))
    room0 = o.room[4]
    obs_at, applied, t = [o], [], None
    walker = Walker(o.terrain)
    for k in range(1, 150):
        pl = o.players[0]
        a = (walker.move(float(pl['x']), float(pl['y']), *target), 0, 0, 0, 0)
        applied.append(a)
        o = lean_step(E, dec, a, FPD)
        obs_at.append(o)
        if o.room[4] != room0:
            t = k
            break
    rec['room_change_record'] = t
    if t is None:
        rec['error'] = 'the walk did not leave the room'
        E.close()
        return rec
    point = Point(kind=KIND_DOOR, base=Held(template), reseed=reseed0, offset=0, applied=applied, d=t - 1, t=t,
                  item=0, visited=(room0,), row=None, key=('door', 0, 1), stage=1, seed=seed, episode=0, room=room0,
                  n_alt=len(doors) - 1, pid=2, order=2)
    t1 = time.perf_counter()
    S, _, oS = restore(point, FPD, 600)
    rec['restore_s'] = round(time.perf_counter() - t1, 4)
    rec['restore_equal'] = digest(oS) == digest(obs_at[t - 1])
    opts = options_of(point, oS, rng)
    rec['options'] = [(h, round(p[0]), round(p[1]), j) for h, p, j in opts]
    R = int(rng.integers(1, 2 ** 31 - 1))
    acts = [(int(rng.integers(9)), int(rng.integers(5)), 0, 0, 0) for _ in range(args.steps)]
    rec['branches'] = []
    traces = {}
    plan = [(j, opt) for j, opt in enumerate(opts)] + ([(1, opts[1])] if len(opts) > 1 else [])
    for n, (j, opt) in enumerate(plan):
        t2 = time.perf_counter()
        B, bdec, o0, comp, status, notes = prefix(S, oS, point, opt, R, FPD, 600, 75)
        info = dict(option=j, how=opt[0], slot=opt[2], status=status, prefix_decisions=int(comp[CI['prefix_decisions']]),
                    prefix_s=round(time.perf_counter() - t2, 4), walk_fail=notes['walk_fail'],
                    room=int(notes['o_prefix'].room[4]))
        if j == 0:
            info['prefix_equal_record_t'] = digest(notes['o_prefix']) == digest(obs_at[t])
        if o0 is not None and j == 1:
            trace, _ = play_random(B, bdec, o0, acts)
            traces.setdefault(1, []).append((trace, hidden(B)))
        B.close()
        rec['branches'].append(info)
    S.close()
    E.close()
    if len(traces.get(1, [])) == 2:
        (a1, d1), (a2, d2) = traces[1]
        rec['crn_same_reseed_equal'] = a1 == a2 and d1 == d2
        rec['crn_steps'] = len(a1) - 1
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147600000:3')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--push', type=int, default=40)
    p.add_argument('--item-back', type=int, default=1)
    p.add_argument('--port', type=int, default=43950)
    p.add_argument('--name', default='brnprobe')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    os.environ.update(FORK_ENV)
    apply_instance_defaults()   # the workers' instance settings (ABP_FAST, PU skip, lite forks)
    os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=FPD, start_hp=6, bombs=1, stub_list=default_stub_list(default_preload()))
    first, count = (int(v) for v in args.seeds.split(':'))
    inst = Instance(args.name, args.port, gx, spec)
    rng = np.random.default_rng(11)
    report = []
    try:
        for seed in range(first, first + count):
            inst.env.bridge.reset_mode = 'floor'
            inst.reset(seed)
            template = inst.env.bridge.fork(tag='template', alarm=0)
            try:
                rec = dict(seed=seed, item=item_seed(template, seed, args, rng), door=door_seed(template, seed, args,
                                                                                                rng))
            finally:
                template.close()
            report.append(rec)
            print(json.dumps(rec, default=str), flush=True)
        items = [r['item'] for r in report if 'error' not in r['item']]
        doors = [r['door'] for r in report if 'error' not in r['door']]
        summary = dict(
            seeds=count,
            item_restore_equal=sum(r['restore_equal'] for r in items),
            item_take_equal_record_t=sum(r['runs']['take1']['prefix_equal_record_t'] for r in items),
            item_take_took=sum(r['runs']['take1']['took'] for r in items),
            item_skip_took=sum(r['runs']['skip']['took'] for r in items),
            item_skip_later_taken=sum(any(c['taken'] for c in r['runs']['skip'].get('skip_checks', [])) for r in items),
            item_crn_equal=sum(bool(r.get('crn_same_reseed_equal')) for r in items),
            items=len(items),
            door_restore_equal=sum(r['restore_equal'] for r in doors),
            door_option0_equal_record_t=sum(bool(r['branches'][0].get('prefix_equal_record_t')) for r in doors),
            door_walks=sum(len(r['branches']) - 1 for r in doors),
            door_walk_fail=sum(b['walk_fail'] for r in doors for b in r['branches']),
            door_crn_equal=sum(bool(r.get('crn_same_reseed_equal')) for r in doors), doors=len(doors))
        print('SUMMARY', json.dumps(summary))
        (out / 'probe_branch.json').write_text(json.dumps(dict(summary=summary, seeds=report), default=str, indent=1))
    finally:
        inst.close()


if __name__ == '__main__':
    main()
