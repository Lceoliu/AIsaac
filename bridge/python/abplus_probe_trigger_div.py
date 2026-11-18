"""How often does the trigger-once switch change a training episode? (the coordinator, 2026-11-18, before deciding
from which arm the new model turns ISAAC_RL_TRIGGER_ONCE on.)

Per episode (seed k): the training's setting -- a floor start (the bridge's floor reset, Isaac), run mode (EpisodeState
run=True, items=True), a start build with probability --build-prob (1 .. --build-max collectibles of
catalog/lab_items.json, at most one active, spawned as pedestals: tok_pickup.spawn_lua, as --start-build-mode
pedestal), forced pickups (tok_pickup.PickupForcer, every episode forced: --pickup-force 1:0 at the run's start) --
two lean clones of the same start (same reseed) stepped in lockstep with the same action sequence: one with the
bridge as it is (ITEM / PILLCARD / BOMB "pressed this frame" answered twice), one with ABP_SET_TRIGGER_ONCE(true).
Actions: move and shoot uniform (0 .. 8, 0 .. 4), bomb / item / pill each with probability --p (an early policy: a
-4 head bias is about 2 %); the forcer's move replaces the drawn one where it gives one (the same in both clones until
they part). Until death (either clone) or --steps decisions.

Divergence: the first decision whose ROW bytes (tok_obs.encode_row) differ, or, every --digest-every decisions and at
the end, the hidden-state digests (ABPGX_DIGEST) differ (then the first such check). Cause: the presses up to the
first divergence, with the player's state before each press read from the off clone (identical until then): pill
with two pocket items -> 'two_pockets'; item with a battery charge (charge over the item's maximum) -> 'battery';
else 'other' (the last presses listed). Background, per episode (the off clone, every 25 decisions and at each
press): Polydactyly (454) held with two pocket items at once; The Battery (63) held; a battery charge.

usage (bridge python dir, PYTHONPATH=.): python abplus_probe_trigger_div.py --groups-file ../abplus/catalog/scaling2_groups.json --p 0.02 --out <dir>
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

SNAP_LUA = r"""
local json = require("json")
local p = Isaac.GetPlayer(0)
local function try(f) local ok, v = pcall(f) if ok then return v end return -1 end
local cfg = Isaac.GetItemConfig():GetCollectible(p:GetActiveItem())
return json.encode({ c0 = p:GetCard(0), c1 = p:GetCard(1), p0 = p:GetPill(0), p1 = p:GetPill(1),
  poly = p:GetCollectibleNum(454), purse = p:GetCollectibleNum(139), battery_item = p:GetCollectibleNum(63),
  active = p:GetActiveItem(), charge = p:GetActiveCharge(), battery = try(function() return p:GetBatteryCharge() end),
  maxc = cfg and cfg.MaxCharges or 0, bombs = p:GetNumBombs() })
"""


def two_pockets(s):
    return sum(1 for k in ('c0', 'c1', 'p0', 'p1') if s[k] > 0) >= 2


def worker(args, seeds, wi, out_path):
    from abplus_probe_fork import digest
    from abplus_probe_native_obs import read_raw
    from goexplore_abplus import default_bridge_lua, default_preload, load_spec
    from isaac_bridge.abplus import FORK_ENV
    from isaac_bridge.abplus_goexplore import GxConfig, Instance
    from isaac_bridge.abplus_lean import LeanDecoder
    from isaac_bridge.tok_lab import load_items
    from isaac_bridge.tok_obs import EV_ITEM, ROW, EpisodeState, encode_row
    from isaac_bridge.tok_pickup import PickupForcer, spawn_lua

    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    inst = Instance(f'tdiv{wi}', args.port + 10 * wi, gx, spec)
    items = load_items(str(Path(__file__).resolve().parent.parent / 'abplus' / 'catalog' / 'lab_items.json'))
    build_ids = np.array(sorted(items), np.int64)
    active = {i for i, r in items.items() if r['kind'] == 'active'}
    f = open(out_path, 'w')
    try:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            rec = dict(seed=seed, p=args.p)
            try:
                b = inst.env.bridge
                b.reset_mode, b.character = 'floor', 0
                inst.reset(seed)
                b = inst.env.bridge
                build = []
                if rng.random() < args.build_prob:
                    k = 1 + int(rng.integers(max(1, args.build_max)))
                    while True:
                        build = sorted(int(v) for v in rng.choice(build_ids, size=k, replace=False))
                        if sum(i in active for i in build) <= 1:
                            break
                    rec['spawned'] = b.lua(spawn_lua(build))
                rec['build'] = build
                reseed = int(rng.integers(1, 2 ** 31 - 1))
                acts = [(int(rng.integers(9)), int(rng.integers(5)), int(rng.random() < args.p),
                         int(rng.random() < args.p), int(rng.random() < args.p)) for _ in range(args.steps)]
                cl, dec, st, row, frc = {}, {}, {}, {}, {}
                for mode in ('off', 'on'):
                    c = b.fork(alarm=1800, reseed=reseed, lean=True)
                    c.lua(f"return tostring(ABP_SET_TRIGGER_ONCE({'true' if mode == 'on' else 'false'}))")
                    cl[mode], dec[mode] = c, LeanDecoder()
                    st[mode] = EpisodeState(10 ** 6, floor=True, run=True, items=True)
                    row[mode] = np.zeros(1, ROW)
                    frc[mode] = PickupForcer()
                    frc[mode].begin(True)
                    c._send({"cmd": "obs"})
                payload = {m: read_raw(cl[m]) for m in cl}
                counts = {m: dict(steps=0, forced=0, free=0) for m in cl}
                first, presses, bg = None, [], dict(two_pockets_poly=False, two_pockets=False, battery_item=False,
                                                    battery_charge=False, poly=False)
                t = 0
                dead = False
                for t in range(args.steps + 1):
                    moves = {}
                    for m in cl:
                        o = dec[m].decode(payload[m])
                        row[m][:] = 0
                        done = encode_row(o, st[m], row[m], t)
                        r = row[m]
                        moves[m] = frc[m].step(r, (int(r['stage'][0]), int(o.room[4])),
                                               t > 0 and bool(r['events'][0] & EV_ITEM), done, counts[m])
                        dead = dead or o.dead
                    if first is None and row['off'].tobytes() != row['on'].tobytes():
                        first = dict(t=t, by='row')
                    if first is None and (t % args.digest_every == 0 and t > 0):
                        if digest(cl['off'])[0] != digest(cl['on'])[0]:
                            first = dict(t=t, by='digest')
                    if dead or t == args.steps:
                        break
                    mv, sh, bo, it, pi = acts[t]
                    if bo or it or pi or t % 25 == 0:   # (background: the whole episode; presses: until they part)
                        s = json.loads(cl['off'].lua(SNAP_LUA))
                        bg['poly'] |= s['poly'] > 0
                        bg['two_pockets'] |= two_pockets(s)
                        bg['two_pockets_poly'] |= s['poly'] > 0 and two_pockets(s)
                        bg['battery_item'] |= s['battery_item'] > 0
                        bg['battery_charge'] |= s['battery'] > 0
                        if (bo or it or pi) and first is None:
                            presses.append(dict(t=t, bomb=bo, item=it, pill=pi, two=two_pockets(s),
                                                over=s['battery'] > 0, active=s['active'], charge=s['charge'],
                                                battery=s['battery']))
                    for m in cl:
                        mm = moves[m] or mv
                        cl[m]._sock.sendall(b'S 4 %d %d %d %d %d\n' % (mm, sh, bo, it, pi) if pi else
                                            b'S 4 %d %d %d %d\n' % (mm, sh, bo, it))
                    for m in cl:
                        payload[m] = read_raw(cl[m])
                if first is None and digest(cl['off'])[0] != digest(cl['on'])[0]:
                    first = dict(t=t, by='digest_end')
                for c in cl.values():
                    c.close()
                cause = None
                if first is not None:
                    before = [q for q in presses if q['t'] < first['t']]
                    if any(q['pill'] and q['two'] for q in before):
                        cause = 'two_pockets'
                    elif any(q['item'] and q['over'] for q in before):
                        cause = 'battery'
                    else:
                        cause = 'other'
                    rec['last_presses'] = before[-4:]
                rec.update(steps=t, dead=dead, first=first, cause=cause, background=bg, n_presses=len(presses),
                           pickups=counts['off'])
            except Exception as e:   # reported; the instance relaunched
                rec['error'] = f'{type(e).__name__}: {e}'
                try:
                    inst.relaunch(hard=True)
                except Exception as e2:
                    rec['relaunch_error'] = str(e2)
            f.write(json.dumps(rec) + '\n')
            f.flush()
    finally:
        f.close()
        inst.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--seeds', default='2148600000:200')
    ap.add_argument('--p', type=float, default=0.02)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--build-prob', type=float, default=0.5)
    ap.add_argument('--build-max', type=int, default=3)
    ap.add_argument('--digest-every', type=int, default=50)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--port', type=int, default=44700)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from isaac_bridge.abplus import FORK_ENV
    os.environ.update(FORK_ENV)
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    os.environ.pop('ISAAC_RL_TRIGGER_ONCE', None)
    first, count = (int(v) for v in args.seeds.split(':'))
    seeds = list(range(first, first + count))
    procs = []
    for w in range(args.workers):
        p = mp.Process(target=worker, args=(args, seeds[w::args.workers], w, str(out / f'eps-{w}.jsonl')))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    rows = [json.loads(l) for w in range(args.workers) for l in open(out / f'eps-{w}.jsonl')]
    ok = [r for r in rows if 'error' not in r]
    div = [r for r in ok if r['first'] is not None]
    causes = {}
    for r in div:
        causes[r['cause']] = causes.get(r['cause'], 0) + 1
    bgk = ('poly', 'two_pockets_poly', 'two_pockets', 'battery_item', 'battery_charge')
    summary = dict(p=args.p, episodes=len(rows), errors=len(rows) - len(ok), diverged=len(div),
                   diverged_share=round(len(div) / max(1, len(ok)), 4), causes=causes,
                   first_t=sorted(r['first']['t'] for r in div), first_by={k: sum(r['first']['by'] == k for r in div)
                                                                           for k in ('row', 'digest', 'digest_end')},
                   background={k: sum(bool(r['background'][k]) for r in ok) for k in bgk},
                   steps_mean=round(float(np.mean([r['steps'] for r in ok])) if ok else 0, 1),
                   deaths=sum(bool(r['dead']) for r in ok), builds=sum(bool(r['build']) for r in ok),
                   pickups=sum(r['pickups']['forced'] + r['pickups']['free'] for r in ok),
                   presses=sum(r['n_presses'] for r in ok))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
