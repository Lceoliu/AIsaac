"""Lasers and tear flags in the token policy's observation (2026-10-08, abp-0.2.16): are the new entity columns
(tok_obs FLAG_COLUMN .. ENT_F: the flag word's bits, a laser's geometry) what the game has, in both encoders?

Per seed the floor's start room is parked (template S, floor mode, lean_items on). Per build (collectibles given with
tok_lab.transplant_lua: none, Ipecac 149, Brimstone 118, Technology 68, Tech X 395) a lean clone of S plays --steps
decisions: fire right held for --hold decisions, then released for --release (a charged weapon fires on the release),
with a slow random walk. Every payload goes through LeanDecoder + encode_row and through abp_row_encode (FastRow; the
pair of abplus_probe_fast_row.py: both records must be equal byte for byte). Checked per record:
  - every laser token (kind 3): its laser columns against the Lua laser record of the same entity (the payload's laser
    block, the bridge's laser_record): cos / sin of AngleDegrees, LaserLength / 300, end point relative to the player /
    200 (a circle laser: its position), circle, Radius / 100;
  - every tear / projectile / laser token: its flag columns against the masks applied to the flag word, and (every
    --lua-every decisions, a Lua command on the same state) the flag word against the entity's TearFlags /
    ProjectileFlags read in Lua;
  - tears with the explosive column (TEAR_EXPLOSIVE, Ipecac) per build.
Every --spawn-every decisions a projectile with ProjectileFlags SMART | EXPLODE is spawned 200 px from the player (its
columns: homing and explosive). With --native (default on) the Ipecac build is also played by three clones in native
obs modes lua / native / check (abplus_probe_native_obs.Trio): the native C entity records must carry the same flag word.

--compare OLD NEW (no game): two directories of abplus_probe_fast_row.py --dump runs on the same seeds, before and after
the change: every ROW field equal, entity columns 0 .. 32 equal byte for byte (the old records have 33 columns), and how
many new-column cells are not zero.

usage (bridge python dir, PYTHONPATH=.):
  python abplus_probe_laser_flags.py --groups-file ../abplus/catalog/scaling2_groups.json --seeds 2147600000:2 \
      --out <dir>
  python abplus_probe_laser_flags.py --compare <old dump dir> <new dump dir>
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from abplus_probe_fast_row import Pair, step
from abplus_probe_native_obs import Trio, read_raw
from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.tok_lab import transplant_lua
from isaac_bridge.tok_obs import (ENT_F, ENT_F0, FLAG_COLUMN, FLAG_F, LASER_COLUMN, PROJ_FLAG_MASKS, TEAR_FLAG_MASKS,
                                  fast_row_function)

BUILDS = (('plain', []), ('ipecac', [149]), ('brimstone', [118]), ('technology', [68]), ('techx', [395]))
EXPLOSIVE = 11        # flag column of TEAR_EXPLOSIVE / EXPLODE
HOMING = 2            # flag column of TEAR_HOMING / SMART
FLAGS_LUA = ("local s = {} for _, e in ipairs(Isaac.GetRoomEntities()) do local w = nil "
             "if e.Type == 2 then w = e:ToTear().TearFlags elseif e.Type == 9 then w = e:ToProjectile().ProjectileFlags "
             "elseif e.Type == 7 then w = e:ToLaser().TearFlags end "
             "if w ~= nil then s[#s + 1] = e.Index .. ':' .. math.type(w) .. ':' .. w end end return table.concat(s, ',')")
SPAWN_LUA = ("local p = Isaac.GetPlayer(0) local room = Game():GetRoom() "
             "local pos = room:FindFreeTilePosition(p.Position + Vector(0, {dy}), 0) "
             "local pr = Isaac.Spawn(9, 0, 0, pos, Vector(0, 0), nil):ToProjectile() "
             "pr.ProjectileFlags = 3 return 'ok'")


def actions(rng, steps, hold, release):
    """(move, shoot) per decision: fire right held for `hold`, released for `release`; a slow random walk."""
    out, move = [], 0
    for t in range(steps):
        if t % 12 == 0:
            move = int(rng.choice([0, 1, 5, 3, 7, 0]))
        shoot = 2 if (t % (hold + release)) < hold else 0
        out.append((move, shoot))
    return out


def check_record(o, row, stats, lua_flags=None):
    """The checks of one record (see the module docstring), counted into stats."""
    e = o.entities
    n = len(e)
    if n > 64:
        stats['skipped_records'] += 1
        return
    out = row['ent'][0]
    px, py = float(o.players[0]['x']), float(o.players[0]['y'])
    lasers = {int(rec['id']): rec for rec in o.lasers}
    for j in range(n):
        kind = int(e['kind'][j])
        word = int(e['tflags'][j])
        masks = PROJ_FLAG_MASKS if kind == 2 else TEAR_FLAG_MASKS
        want = np.array([(word & int(m)) != 0 for m in masks], np.float32)
        got = out[j, FLAG_COLUMN:FLAG_COLUMN + FLAG_F]
        stats['flag_tokens'] += kind in (1, 2, 3)
        if not np.array_equal(got, want):
            stats['flag_mismatch'] += 1
        if kind == 1:
            stats['tears'] += 1
            stats['tears_explosive'] += bool(got[EXPLOSIVE])
            stats['tears_flagged'] += word != 0
        if kind == 2:
            stats['projectiles'] += 1
            stats['projectiles_homing_explode'] += bool(got[HOMING] and got[EXPLOSIVE])
        if kind != 3 and np.any(out[j, LASER_COLUMN:ENT_F]):
            stats['laser_columns_on_non_laser'] += 1
        if lua_flags is not None and kind in (1, 2, 3):
            stats['lua_flag_checks'] += 1
            lw = lua_flags.get(int(e['id'][j]))
            if lw is None or lw[1] != word:
                stats['lua_flag_mismatch'] += 1
                stats.setdefault('lua_flag_first', dict(id=int(e['id'][j]), kind=kind, word=word, lua=lw))
            elif lw[0] != 'integer':
                stats['lua_flag_not_integer'] += 1
        if kind != 3:
            continue
        stats['laser_tokens'] += 1
        rec = lasers.get(int(e['id'][j]))
        if rec is None:
            stats['laser_without_record'] += 1
            continue
        lc = out[j, LASER_COLUMN:ENT_F]
        rad = math.radians(rec['angle'])
        ex, ey = (float(e['x'][j]), float(e['y'][j])) if rec['circle'] else rec['end']
        want = [math.cos(rad), math.sin(rad), rec['length'] / 300, (ex - px) / 200, (ey - py) / 200,
                1.0 if rec['circle'] else 0.0, rec['radius'] / 100]
        err = max(abs(float(a) - b) for a, b in zip(lc, want))
        stats['laser_max_err'] = max(stats['laser_max_err'], err)
        stats['laser_ok'] += err < 1e-4
        stats['laser_nonzero'] += bool(np.any(lc != 0))
        stats['laser_circle'] += bool(rec['circle'])
        if len(stats['laser_examples']) < 6:
            stats['laser_examples'].append(dict(id=int(e['id'][j]), variant=int(e['variant'][j]), angle=rec['angle'],
                                                length=rec['length'], end=[round(v, 2) for v in rec['end']],
                                                circle=rec['circle'], radius=rec['radius'],
                                                columns=[round(float(v), 4) for v in lc], word=word))


def lua_flag_words(c):
    text = c.lua(FLAGS_LUA)
    out = {}
    for part in filter(None, text.split(',')):
        idx, typ, w = part.split(':')
        out[int(idx)] = (typ, int(w))
    return out


def new_stats():
    s = {k: 0 for k in ('records', 'skipped_records', 'flag_tokens', 'flag_mismatch', 'tears', 'tears_explosive',
                        'tears_flagged', 'projectiles', 'projectiles_homing_explode', 'laser_columns_on_non_laser',
                        'lua_flag_checks', 'lua_flag_mismatch', 'lua_flag_not_integer', 'laser_tokens',
                        'laser_without_record', 'laser_ok', 'laser_nonzero', 'laser_circle', 'spawned')}
    s['laser_max_err'] = 0.0
    s['laser_examples'] = []
    return s


def one_build(S, seed, name, items, args, fn, rng):
    C = S.fork(lean=True, alarm=600, reseed=seed % 1000 + 1)
    stats = new_stats()
    try:
        if items:
            C.lua(transplant_lua(items))
        pair = Pair(fn, 10 ** 6, True, 10 ** 6, items=True)
        C._send({"cmd": "obs"})
        o, _ = pair.record(read_raw(C))
        for t, (move, shoot) in enumerate(actions(rng, args.steps, args.hold, args.release)):
            if args.spawn_every and t % args.spawn_every == args.spawn_every - 1:
                if C.lua(SPAWN_LUA.format(dy=-200 if t % 2 else 200)) == 'ok':
                    stats['spawned'] += 1
            o, _ = pair.record(step(C, (move, shoot, 0, 0), 4))
            stats['records'] += 1
            lua_flags = lua_flag_words(C) if args.lua_every and t % args.lua_every == 0 else None
            check_record(o, pair.last_row, stats, lua_flags)
            if o.dead:
                break
        stats['fast_row_equal'] = pair.mismatch is None
        stats['fast_row_mismatch'] = pair.mismatch
        stats['fast'], stats['slow'] = pair.enc.fast, pair.enc.slow
    finally:
        C.close()
    return stats


def native_build(S, seed, items, args, rng):
    """The build played by three clones in native obs modes lua / native / check (Trio): payloads equal byte for byte."""
    C = S.fork(lean=True, alarm=600, reseed=seed % 1000 + 1)
    try:
        C.lua(transplant_lua(items))
        trio = Trio(C, 25)
        flagged = 0
        for t, (move, shoot) in enumerate(actions(rng, args.steps, args.hold, args.release)):
            if args.spawn_every and t % args.spawn_every == args.spawn_every - 1:
                trio.lua(SPAWN_LUA.format(dy=-200 if t % 2 else 200))
            o = trio.step((move, shoot, 0, 0), 4)
            flagged += int(np.count_nonzero(o.entities['tflags']))
            if o.dead:
                break
        res = trio.finish()
        res['flagged_entity_records'] = flagged
        return res
    finally:
        C.close()


def compare(old_dir, new_dir):
    old_dir, new_dir = Path(old_dir), Path(new_dir)
    report = []
    for f in sorted(old_dir.glob('rows_*.npz')):
        g = new_dir / f.name
        if not g.is_file():
            report.append(dict(file=f.name, missing=True))
            continue
        a, b = np.load(f), np.load(g)
        rec = dict(file=f.name, records=(len(a['t']), len(b['t'])), fields_differ=[])
        if len(a['t']) != len(b['t']):
            report.append(rec)
            continue
        for k in a.files:
            x, y = a[k], b[k]
            if k == 'ent':
                rec['ent_columns'] = (x.shape[-1], y.shape[-1])
                y0 = y[..., :x.shape[-1]]
                if x.shape[-1] != ENT_F0 or x.tobytes() != np.ascontiguousarray(y0).tobytes():
                    rec['fields_differ'].append('ent[:33]')
                new = y[..., ENT_F0:]
                live = np.arange(y.shape[1])[None, :] < b['n_ent'][:, None]   # the records' own entity rows
                rec['new_cells_nonzero'] = int(np.count_nonzero(new[live]))
                rec['new_flag_cells_nonzero'] = int(np.count_nonzero(new[live][:, :FLAG_F]))
                rec['new_laser_cells_nonzero'] = int(np.count_nonzero(new[live][:, FLAG_F:]))
            elif x.tobytes() != y.tobytes():
                rec['fields_differ'].append(k)
        rec['equal'] = not rec['fields_differ']
        report.append(rec)
    summary = dict(files=len(report), equal=sum(bool(r.get('equal')) for r in report),
                   records=sum(r['records'][0] for r in report if 'records' in r),
                   new_cells_nonzero=sum(r.get('new_cells_nonzero', 0) for r in report))
    print(json.dumps(report, indent=1))
    print('SUMMARY', json.dumps(summary))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--compare', nargs=2, metavar=('OLD', 'NEW'))
    p.add_argument('--groups-file', default='')
    p.add_argument('--seeds', default='2147600000:2')
    p.add_argument('--builds', default='plain,ipecac,brimstone,technology,techx')
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--hold', type=int, default=10)
    p.add_argument('--release', type=int, default=4)
    p.add_argument('--lua-every', type=int, default=3)
    p.add_argument('--spawn-every', type=int, default=20)
    p.add_argument('--native', type=int, default=1)
    p.add_argument('--port', type=int, default=44330)
    p.add_argument('--name', default='obslaser')
    p.add_argument('--out', default='')
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    fn = fast_row_function(default_preload())
    if fn is None:
        raise SystemExit('abp_row_encode not available (or of another version)')
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    builds = [b for b in BUILDS if b[0] in args.builds.split(',')]
    first, count = (int(v) for v in args.seeds.split(':'))
    inst = Instance(args.name, args.port, gx, spec)
    report = []
    try:
        for seed in range(first, first + count):
            inst.env.bridge.reset_mode = 'floor'
            inst.reset(seed)
            S = inst.env.bridge.fork(tag='template', alarm=0)
            t0 = time.perf_counter()
            for name, items in builds:
                rec = dict(seed=seed, build=name, items=items,
                           **one_build(S, seed, name, items, args, fn, np.random.default_rng(seed + 7)))
                report.append(rec)
                print(json.dumps(rec), flush=True)
            if args.native:
                rec = dict(seed=seed, build='ipecac', native=native_build(S, seed, [149], args,
                                                                          np.random.default_rng(seed + 7)))
                report.append(rec)
                print(json.dumps(rec, default=str), flush=True)
            S.close()
            print('seed', seed, 'seconds', round(time.perf_counter() - t0, 1), flush=True)
    finally:
        inst.kill()
    builds_rep = [r for r in report if 'native' not in r]
    natives = [r['native'] for r in report if 'native' in r]
    summary = dict(
        runs=len(builds_rep),
        fast_row_equal=sum(r['fast_row_equal'] for r in builds_rep),
        records=sum(r['records'] for r in builds_rep),
        flag_mismatch=sum(r['flag_mismatch'] for r in builds_rep),
        lua_flag_checks=sum(r['lua_flag_checks'] for r in builds_rep),
        lua_flag_mismatch=sum(r['lua_flag_mismatch'] for r in builds_rep),
        lua_flag_not_integer=sum(r['lua_flag_not_integer'] for r in builds_rep),
        laser_columns_on_non_laser=sum(r['laser_columns_on_non_laser'] for r in builds_rep),
        by_build={name: dict(tears=sum(r['tears'] for r in builds_rep if r['build'] == name),
                             tears_explosive=sum(r['tears_explosive'] for r in builds_rep if r['build'] == name),
                             tears_flagged=sum(r['tears_flagged'] for r in builds_rep if r['build'] == name),
                             projectiles=sum(r['projectiles'] for r in builds_rep if r['build'] == name),
                             projectiles_homing_explode=sum(r['projectiles_homing_explode'] for r in builds_rep
                                                            if r['build'] == name),
                             laser_tokens=sum(r['laser_tokens'] for r in builds_rep if r['build'] == name),
                             laser_ok=sum(r['laser_ok'] for r in builds_rep if r['build'] == name),
                             laser_nonzero=sum(r['laser_nonzero'] for r in builds_rep if r['build'] == name),
                             laser_circle=sum(r['laser_circle'] for r in builds_rep if r['build'] == name),
                             laser_max_err=max([r['laser_max_err'] for r in builds_rep if r['build'] == name] or [0]))
                  for name, _ in builds},
        native=[dict(payload_mismatch=n['payload_mismatch'], digests_equal=n['digests_equal'],
                     check_checks=n['check_checks'], check_mismatches=n['check_mismatches'],
                     native_calls=n['native_calls'], native_fallbacks=n['native_fallbacks'],
                     flagged_entity_records=n['flagged_entity_records'], payloads=n['payloads']) for n in natives])
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print('SUMMARY', json.dumps(summary, default=str), flush=True)


if __name__ == '__main__':
    main()
