"""Phase C probe (2026-10-08): is a transplanted build (tok_lab.transplant_lua: AddCollectible(id, charge, true)) the same
as the build a pedestal pickup gives?

One instance in the bridge's floor mode (lean_items on). Per seed the floor's start is parked (template S). Per collectible
X, two lean clones of S with the same reseed:
  P (pickup)     a pedestal holding X is spawned next to the player (Isaac.Spawn: the same Lua in both clones, so the
                 same random draws and position) and the player walks into it (moves toward it, no shots) until the
                 inventory shows X, then stands until the item queue is empty and the hold-up animation is over (settled).
  T (transplant) the same pedestal spawned, made untakeable (bridge pickup_block: the player bumps into it), the same
                 actions as P decision by decision (the moves are P's); at P's settled decision transplant_lua([X]).
Compared:
  - EvaluateItems: the stats the transplant's Lua chunk returns right after AddCollectible (no frame in between) against
    T's stats before and its next observation;
  - the settled builds: player 0's lean fields that a build sets (hearts of every kind, consumables, damage, fire delay,
    shot speed, range, speed, luck, flight, active item and its charge, weapon types, size), the inventory block and the
    familiars / pickups in the room, P against T;
  - the game after both: in both clones the pedestals are removed, T's player is put on P's position with P's velocity,
    both reseed the global MT with the same value, then the same --steps decisions of sticky random moves and shots: per
    decision the lean payload (players without position-free fields, entities without ids, sorted) P against T, and at
    the end the hidden-state digest (ABPGX_DIGEST) line by line: which lines differ.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_transplant.py --groups-file ../abplus/catalog/scaling2_groups.json --seeds 2147600000:1 \
      --items all --steps 60 --out <dir>
"""
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import PLAYER_FIELDS, LeanDecoder, read_lean
from isaac_bridge.tok_lab import load_items, transplant_lua
from isaac_bridge.tok_sampler import lean_step

MOVES = {(0, 0): 0, (0, -1): 1, (1, -1): 2, (1, 0): 3, (1, 1): 4, (0, 1): 5, (-1, 1): 6, (-1, 0): 7, (-1, -1): 8}
SPAWN_LUA = ("local p = Isaac.GetPlayer(0) local room = Game():GetRoom() "
             "local pos = room:FindFreeTilePosition(p.Position + Vector(0, 80), 0) "
             "Isaac.Spawn(5, 100, {sub}, pos, Vector(0, 0), nil) return tostring(pos.X) .. ',' .. tostring(pos.Y)")
SETTLED_LUA = ("local p = Isaac.GetPlayer(0) local q = true pcall(function() q = p:IsItemQueueEmpty() end) "
               "local a = true pcall(function() a = p:IsExtraAnimationFinished() end) "
               "return tostring(q) .. ',' .. tostring(a)")
# the pedestals, and (--clear-effects) the room's effects (EntityType 1000: dust, sparkles; they move with the global MT
# and so already differ between P and T: P's pickup drew random numbers T's blocked bump did not)
REMOVE_LUA = ("for _, e in ipairs(Isaac.GetRoomEntities()) do if (e.Type == 5 and e.Variant == 100) or "
              "(e.Type == 1000 and {effects}) then e:Remove() end end return 'ok'")
PLACE_LUA = "local p = Isaac.GetPlayer(0) p.Position = Vector({x!r}, {y!r}) p.Velocity = Vector({vx!r}, {vy!r}) return 'ok'"
# the lean player fields a build sets (position, velocity, directions, animation frame and timers left out)
BUILD_FIELDS = ('hearts', 'max_hearts', 'soul', 'black', 'bone', 'eternal', 'golden', 'lives', 'coins', 'bombs', 'keys',
                'damage', 'fire_delay_max', 'shot_speed', 'range', 'speed', 'luck', 'can_fly', 'active',
                'active_charge', 'active_ready', 'size', 'weapons')
STAT_FIELDS = ('damage', 'fire_delay_max', 'shot_speed', 'range', 'speed')


def move_to(px, py, x, y):
    sx = 0 if abs(x - px) < 6 else (1 if x > px else -1)
    sy = 0 if abs(y - py) < 6 else (1 if y > py else -1)
    return MOVES[(sx, sy)]


def inv(o):
    return {int(i): int(c) for i, c in zip(o.inv['id'], o.inv['count'])} if o.inv_head is not None else {}


def room_set(o, types=(3,)):
    e = o.entities
    return sorted((int(t), int(v), int(s)) for t, v, s in zip(e['type'], e['variant'], e['subtype']) if int(t) in types)


def ent_key(o):
    e = o.entities
    cols = [n for n in e.dtype.names if n != 'id']
    rows = sorted(tuple(np.round(float(x), 6) if isinstance(x, (float, np.floating)) else int(x) for x in r[cols])
                  for r in e)
    return rows


def player_diff(a, b, fields=PLAYER_FIELDS):
    pa, pb = a.players[0], b.players[0]
    return [f for f in fields if abs(float(pa[f]) - float(pb[f])) > 1e-9]


def digest_lines(c):
    return c.lua('return ABPGX_DIGEST(false)').split('\n')


def digest_diff(lp, lt, keep=6):
    """The digest lines only one side has (as multisets; at most `keep` each, cut to 240 characters), their counts."""
    a, b = Counter(lp), Counter(lt)
    only_p, only_t = list((a - b).elements()), list((b - a).elements())
    return dict(n_p=len(only_p), n_t=len(only_t), only_p=[[x[:240]] for x in only_p[:keep]],
                only_t=[[x[:240]] for x in only_t[:keep]])


def one_item(S, item, seed, args, rng):
    rec = dict(item=item)
    P = S.fork(lean=True, alarm=600, reseed=seed % 1000 + 1)
    T = S.fork(lean=True, alarm=600, reseed=seed % 1000 + 1)
    try:
        dp, dt = LeanDecoder(), LeanDecoder()
        P._send({"cmd": "obs"})
        T._send({"cmd": "obs"})
        op, ot = read_lean(P, dp), read_lean(T, dt)
        inv0 = inv(op)
        pos = P.lua(SPAWN_LUA.format(sub=item))
        if T.lua(SPAWN_LUA.format(sub=item)) != pos:
            rec['error'] = 'pedestal positions differ'
            return rec
        x, y = (float(v) for v in pos.split(','))
        blk = T.pickup_block(subtype=item, x=x, y=y)
        rec['blocked'] = int(blk.get('found', 0))
        base_t = {f: float(ot.players[0][f]) for f in BUILD_FIELDS}
        got = settled = None
        k = 0
        for k in range(1, 160):
            pl = op.players[0]
            a = (move_to(float(pl['x']), float(pl['y']), x, y) if got is None else 0, 0, 0, 0, 0)
            op = lean_step(P, dp, a, 4)
            ot = lean_step(T, dt, a, 4)
            if got is None and inv(op).get(item, 0) > inv0.get(item, 0):
                got = k
            if got is not None:
                q, an = P.lua(SETTLED_LUA).split(',')
                T.lua(SETTLED_LUA)   # (the same command in both: a lua command marks the terrain block for resend)
                if q == 'true' and an == 'true':
                    settled = k
                    break
        rec.update(taken_at=got, settled_at=settled, same_pos_before=player_diff(op, ot, ('x', 'y', 'vx', 'vy')) == [])
        if settled is None:
            rec['error'] = 'not taken' if got is None else 'not settled'
            return rec
        # the transplant at P's settled decision
        lua = T.lua(transplant_lua([item]))
        parts = lua.split('|')
        rec['lua_stats'] = [float(v) for v in parts[1:6]]
        T._send({"cmd": "obs"})
        ot = read_lean(T, dt)
        P._send({"cmd": "obs"})
        op = read_lean(P, dp)
        obs_t = [float(ot.players[0][f]) for f in ('damage', 'fire_delay_max', 'shot_speed', 'range', 'speed')]
        rec['obs_stats'] = obs_t
        rec['evaluate_in_add'] = bool(np.allclose([rec['lua_stats'][k] for k in (0, 1, 2, 4)],
                                                  [obs_t[k] for k in (0, 1, 2, 4)], atol=1e-5))
        rec['stats_changed'] = any(abs(base_t[f] - float(ot.players[0][f])) > 1e-9 for f in STAT_FIELDS)
        # one decision standing (familiars and pickups the transplant spawned show from the next frame on)
        op = lean_step(P, dp, (0, 0, 0, 0, 0), 4)
        ot = lean_step(T, dt, (0, 0, 0, 0, 0), 4)
        rec['build_diff'] = player_diff(op, ot, BUILD_FIELDS)
        rec['build_diff_values'] = {f: [float(op.players[0][f]), float(ot.players[0][f])] for f in rec['build_diff']}
        rec['inv_equal'] = inv(op) == inv(ot) and tuple(op.inv_head[1:]) == tuple(ot.inv_head[1:])
        rec['familiars_p'], rec['familiars_t'] = room_set(op), room_set(ot)
        rec['familiars_equal'] = rec['familiars_p'] == rec['familiars_t']
        rec['pickups_equal'] = room_set(op, (5,)) == room_set(ot, (5,)) or \
            [r for r in room_set(op, (5,)) if r[1] != 100] == [r for r in room_set(ot, (5,)) if r[1] != 100]
        # the game after both: pedestals removed, T on P's place, the same reseed, the same actions
        P.lua(REMOVE_LUA.format(effects='true' if args.clear_effects else 'false'))
        T.lua(REMOVE_LUA.format(effects='true' if args.clear_effects else 'false'))
        if args.clear_effects:   # Remove() takes effect at the next update: one frame standing, before the reseed
            op = lean_step(P, dp, (0, 0, 0, 0, 0), 1)
            ot = lean_step(T, dt, (0, 0, 0, 0, 0), 1)
        pl = op.players[0]
        T.lua(PLACE_LUA.format(x=float(pl['x']), y=float(pl['y']), vx=float(pl['vx']), vy=float(pl['vy'])))
        R = int(rng.integers(1, 2 ** 31 - 1))
        P.reseed(R)
        T.reseed(R)
        P._send({"cmd": "obs"})
        T._send({"cmd": "obs"})
        op, ot = read_lean(P, dp), read_lean(T, dt)
        rec['start_player_diff'] = player_diff(op, ot)
        l0p, l0t = digest_lines(P), digest_lines(T)
        rec['start_digest_diff'] = digest_diff(l0p, l0t)
        first, fields, same_ents = None, Counter(), 0
        move, shoot = 0, 0
        steps = 0
        for j in range(args.steps):
            if rng.random() < 0.15:
                move, shoot = int(rng.integers(9)), int(rng.integers(5))
            op = lean_step(P, dp, (move, shoot, 0, 0, 0), 4)
            ot = lean_step(T, dt, (move, shoot, 0, 0, 0), 4)
            steps += 1
            d = player_diff(op, ot)
            ents = ent_key(op) == ent_key(ot)
            same_ents += ents
            fields.update(d)
            if (d or not ents) and first is None:
                first = j
            if op.dead or ot.dead:
                break
        rec.update(steps=steps, first_diff=first, player_fields=dict(fields), same_entities=same_ents)
        lp, lt = digest_lines(P), digest_lines(T)
        rec['digest_equal'] = lp == lt
        rec['digest_diff'] = digest_diff(lp, lt)
        rec['digest_diff_kinds'] = dict(Counter(x[0].split('|')[0] for x in rec['digest_diff']['only_p']) +
                                        Counter(x[0].split('|')[0] for x in rec['digest_diff']['only_t']))
        rec['digest_lines'] = [len(lp), len(lt)]
        return rec
    finally:
        for c in (P, T):
            try:
                c.close()
            except OSError:
                pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147600000:1')
    p.add_argument('--items', default='all', help="'all' (catalog/lab_items.json) or ids '1,2,3'")
    p.add_argument('--items-file', default='../abplus/catalog/lab_items.json')
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--clear-effects', type=int, default=1,
                   help='1: the effects (type 1000) of both rooms removed before the common reseed (see REMOVE_LUA)')
    p.add_argument('--port', type=int, default=44100)
    p.add_argument('--name', default='labtp')
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
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    pool = load_items(args.items_file)
    items = sorted(pool) if args.items == 'all' else [int(v) for v in args.items.split(',')]
    first, count = (int(v) for v in args.seeds.split(':'))
    inst = Instance(args.name, args.port, gx, spec)
    rng = np.random.default_rng(11)
    report = []
    fh = open(out / 'transplant.jsonl', 'w')
    try:
        for seed in range(first, first + count):
            inst.env.bridge.reset_mode = 'floor'
            inst.reset(seed)
            S = inst.env.bridge.fork(tag='template', alarm=0)
            t0 = time.perf_counter()
            for item in items:
                try:
                    rec = one_item(S, item, seed, args, rng)
                except Exception as exc:   # an item that breaks a clone: recorded, the next one
                    rec = dict(item=item, error=f'{type(exc).__name__}: {str(exc)[:200]}')
                rec.update(seed=seed, name=pool.get(item, {}).get('name'), kind=pool.get(item, {}).get('kind'))
                report.append(rec)
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
            S.close()
            print('seed', seed, 'items', len(items), 'seconds', round(time.perf_counter() - t0, 1), flush=True)
        ok = [r for r in report if 'error' not in r]
        summary = dict(
            items=len(report), compared=len(ok), errors=Counter(r['error'] for r in report if 'error' in r),
            evaluate_in_add=sum(r['evaluate_in_add'] for r in ok),
            stats_changed=sum(r['stats_changed'] for r in ok),
            build_equal=sum(not r['build_diff'] for r in ok),
            build_diff_fields=dict(Counter(f for r in ok for f in r['build_diff'])),
            inv_equal=sum(r['inv_equal'] for r in ok), familiars_equal=sum(r['familiars_equal'] for r in ok),
            pickups_equal=sum(r['pickups_equal'] for r in ok),
            start_player_diff=dict(Counter(f for r in ok for f in r['start_player_diff'])),
            trajectory_equal=sum(r['first_diff'] is None for r in ok),
            digest_equal=sum(r['digest_equal'] for r in ok),
            digest_diff_kinds=dict(sum((Counter(r['digest_diff_kinds']) for r in ok), Counter())),
            start_digest_equal=sum(r['start_digest_diff']['n_p'] + r['start_digest_diff']['n_t'] == 0 for r in ok),
            start_digest_kinds=dict(sum((Counter('|'.join(x[0].split('|')[:3]) for x in
                                                 r['start_digest_diff']['only_p']) for r in ok), Counter())),
            trajectory_equal_by_kind=dict(Counter(f"{r['kind']}:{r['first_diff'] is None}" for r in ok)),
            player_fields_after=dict(sum((Counter({k: 1 for k in r['player_fields']}) for r in ok), Counter())),
            settle_decisions=[int(np.percentile([r['settled_at'] for r in ok], q)) for q in (0, 50, 100)] if ok else [],
            same_pos_before=sum(r['same_pos_before'] for r in ok))
        print('SUMMARY', json.dumps(summary, default=str), flush=True)
        (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    finally:
        fh.close()
        inst.close()


if __name__ == '__main__':
    main()
