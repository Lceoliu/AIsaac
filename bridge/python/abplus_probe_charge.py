"""Charged weapons in the lean observation (2026-10-07): how the charge counter (abplus_lean PLAYER 'charge',
Entity_Player +0x2634) and the weapon-type bits behave, and what a hold of the shoot button for N decisions does.

Per weapon (Brimstone 118, Mom's Knife 114, Chocolate Milk 69, Monstro's Lung 229, Cursed Eye 316 by default): a
room of the group is reset, its enemies killed, the collectible given (AddCollectible) and a dummy enemy spawned to the
player's right on the same row (a Gaper, frozen, 10,000 HP); the state is parked. For each frames-per-decision value
(--fpd, e.g. 1,2,4) and N = 1 .. --max-frames // fpd: a lean clone of the parked state holds "shoot right" for N
decisions of fpd frames (N x fpd frames), then releases (no input) for --after frames. Recorded per run: the charge
counter at the release (the last held observation) and its maximum, the weapon bits, the first decision a laser
(type 7) / knife (type 8) / player tear (type 2) shows after the release, the knife's farthest distance from the player,
the largest tear CollisionDamage, and the dummy's HP lost (Lua at the end). --native-check runs the instance in native
obs check mode (ISAAC_RL_NATIVE_OBS=2): the Lua and C observations (with the charge fields) are compared every
decision; the counters are reported.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_charge.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal --seed 2147500000 \
      --fpd 1,2,4 --out <dir>
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
from isaac_bridge.abplus_lean import LeanDecoder, read_lean_raw
from isaac_bridge.tok_sampler import apply_instance_defaults, default_stub_list

WEAPONS = {118: 'Brimstone', 114: "Mom's Knife", 69: 'Chocolate Milk', 229: "Monstro's Lung", 316: 'Cursed Eye',
           395: 'Tech X', 52: 'Dr. Fetus', 168: 'Epic Fetus'}
KILL_LUA = ("for _, e in ipairs(Isaac.GetRoomEntities()) do "
            "if e:IsActiveEnemy(false) then e:Remove() end end return 'ok'")
GIVE_LUA = "Isaac.GetPlayer(0):AddCollectible({item}, 0, false) return 'ok'"
DUMMY_LUA = ("local p = Isaac.GetPlayer(0) local room = Game():GetRoom() "
             "local pos = room:FindFreeTilePosition(Vector(p.Position.X + {dx}, p.Position.Y), 0) "
             "local e = Isaac.Spawn(10, 0, 0, pos, Vector(0, 0), nil) "
             "{freeze} e.MaxHitPoints = 10000 e.HitPoints = 10000 AbpSetInvincible(true) "
             "ABP_DUMMY = e return tostring(pos.X) .. ',' .. tostring(pos.Y)")
DUMMY_HP_LUA = "return tostring(ABP_DUMMY and ABP_DUMMY.HitPoints or -1)"
# the grid entities on the player's row (rocks, poops, ...) destroyed: a tear stops at a rock, a laser does not
CLEAR_ROW_LUA = ("local p = Isaac.GetPlayer(0) local room = Game():GetRoom() local n = 0 "
                 "for i = 0, room:GetGridSize() - 1 do local g = room:GetGridEntity(i) "
                 "if g and math.abs(g.Position.Y - p.Position.Y) < 30 and g:GetType() ~= 15 and g:GetType() ~= 16 "
                 "then g:Destroy(true) n = n + 1 end end return tostring(n)")
PLACE_LUA = ("local p = Isaac.GetPlayer(0) local room = Game():GetRoom() "
             "p.Position = room:GetCenterPos() + Vector(-160, 0) p.Velocity = Vector(0, 0) return 'ok'")


def step(c, dec, move, shoot, rep):
    c._sock.sendall(b"S %d %d %d 0 0\n" % (rep, move, shoot))
    return dec.decode(read_lean_raw(c))


FAST_FN = None   # abp_row_encode (tok_obs.fast_row_function): every payload also through FastRow vs encode_row


def run(parked, hold_frames, fpd, after):
    """Hold shoot-right for hold_frames // fpd decisions, release for `after` frames; what happened."""
    c = parked.fork(lean=True, alarm=300)
    try:
        from abplus_probe_fast_row import Pair
        pair = Pair(FAST_FN, 10 ** 6, False, 0) if FAST_FN is not None else None
        dec = LeanDecoder()

        def get(move, shoot):
            if move is None:
                c._send({"cmd": "obs"})
            else:
                c._sock.sendall(b"S %d %d %d 0 0\n" % (fpd, move, shoot))
            raw = read_lean_raw(c)
            return pair.record(raw)[0] if pair is not None else dec.decode(raw)

        o = get(None, None)
        n = hold_frames // fpd
        charges = []
        for _ in range(n):
            o = get(0, 2)
            charges.append(float(o.players[0]['charge']))
        px0 = float(o.players[0]['x'])
        rec = dict(hold=n * fpd, decisions=n, charge_release=charges[-1] if charges else 0.0,
                   charge_max=max(charges) if charges else 0.0, weapons=int(o.players[0]['weapons']),
                   fire_delay_max=float(o.players[0]['fire_delay_max']), laser_at=None, knife_far=0.0, tear_at=None,
                   tear_damage=0.0, tears=0, charges_after=[])
        seen = set()
        for k in range((after + fpd - 1) // fpd):
            o = get(0, 0)
            pl = o.players[0]
            rec['charges_after'].append(float(pl['charge']))
            e = o.entities
            if (len(e) and (e['type'] == 7).any()) or o.lasers:
                if rec['laser_at'] is None:
                    rec['laser_at'] = (k + 1) * fpd
            kn = e[e['type'] == 8] if len(e) else e
            if len(kn):
                rec['knife_far'] = max(rec['knife_far'], float(np.hypot(kn['x'] - pl['x'], kn['y'] - pl['y']).max()))
            tears = e[e['type'] == 2] if len(e) else e
            if len(tears):
                if rec['tear_at'] is None:
                    rec['tear_at'] = (k + 1) * fpd
                rec['tear_damage'] = max(rec['tear_damage'], float(tears['cdmg'].max()))
                seen.update(int(i) for i in tears['id'])
                rec['tears'] = len(seen)
        rec['charges_after'] = rec['charges_after'][:6]
        rec['dummy_hp_lost'] = 10000.0 - float(c.lua(DUMMY_HP_LUA))
        rec['player_x_release'] = px0
        if pair is not None:
            rec['fastrow'] = dict(records=pair.t, fast=pair.enc.fast, slow=pair.enc.slow, mismatch=pair.mismatch)
        c._send({"cmd": "native_obs"})
        m = c._recv()
        rec['obs_checks'], rec['obs_mismatches'] = int(m.get('checks') or 0), int(m.get('mismatches') or 0)
        if m.get('first'):
            rec['obs_first'] = m.get('first')
        return rec
    finally:
        c.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=2147500000)
    p.add_argument('--items', default='118,114,69,229,316')
    p.add_argument('--fpd', default='1,2,4')
    p.add_argument('--max-frames', type=int, default=80)
    p.add_argument('--after', type=int, default=60)
    p.add_argument('--dummy-dx', type=float, default=240.0)
    p.add_argument('--native-check', action='store_true')
    p.add_argument('--walking-dummy', action='store_true', help='the dummy is not frozen (a frozen one took no tear '
                   'damage); the player is invincible either way')
    p.add_argument('--port', type=int, default=42400)
    p.add_argument('--name', default='bmbchg')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    if args.native_check:
        os.environ['ISAAC_RL_NATIVE_OBS'] = '2'
    apply_instance_defaults()
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  stub_list=default_stub_list(default_preload()))
    inst = Instance(args.name, args.port, gx, spec)
    global FAST_FN
    from isaac_bridge.tok_obs import fast_row_function
    FAST_FN = fast_row_function(default_preload())
    rows, t0 = [], time.perf_counter()
    try:
        inst.reset(args.seed)
        base = inst.env.bridge.fork(tag='base', alarm=0, lean=True)
        for item in [int(x) for x in args.items.split(',') if x]:
            w = base.fork(tag='weapon', alarm=0, lean=True)
            try:
                assert w.lua(KILL_LUA) == 'ok'
                assert w.lua(PLACE_LUA) == 'ok'
                cleared = w.lua(CLEAR_ROW_LUA)
                assert w.lua(GIVE_LUA.format(item=item)) == 'ok'
                dec = LeanDecoder()
                for _ in range(40):   # the pickup / evaluate frames, the killed enemies gone
                    o = step(w, dec, 0, 0, 1)
                freeze = '' if args.walking_dummy else \
                    'pcall(function() e:AddEntityFlags(EntityFlag.FLAG_FREEZE) end)'
                dummy = w.lua(DUMMY_LUA.format(dx=args.dummy_dx, freeze=freeze))
                for _ in range(20):   # the dummy's spawn frames
                    o = step(w, dec, 0, 0, 1)
                pl = o.players[0]
                head = dict(item=item, name=WEAPONS.get(item, str(item)), weapons=int(pl['weapons']),
                            charge_idle=float(pl['charge']), fire_delay_max=float(pl['fire_delay_max']),
                            player=(float(pl['x']), float(pl['y'])), dummy=dummy, row_cleared=cleared)
                print(json.dumps(head), flush=True)
                parked = w.fork(tag='parked', alarm=0)
                try:
                    for fpd in [int(x) for x in args.fpd.split(',') if x]:
                        for hold in range(fpd, args.max_frames + 1, fpd):
                            rec = dict(head, fpd=fpd, **run(parked, hold, fpd, args.after))
                            rows.append(rec)
                            print(json.dumps({k: v for k, v in rec.items() if k != 'charges_after'}), flush=True)
                finally:
                    parked.close()
            finally:
                w.close()
        base.close()
    finally:
        inst.close()
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
        fr = [r['fastrow'] for r in rows if 'fastrow' in r]
        print('SUMMARY', json.dumps(dict(runs=len(rows), seconds=round(time.perf_counter() - t0, 1),
                                         fastrow_records=sum(f['records'] for f in fr),
                                         fastrow_fast=sum(f['fast'] for f in fr),
                                         fastrow_mismatches=sum(f['mismatch'] is not None for f in fr),
                                         obs_checks=sum(r.get('obs_checks', 0) for r in rows),
                                         obs_mismatches=sum(r.get('obs_mismatches', 0) for r in rows))))


if __name__ == '__main__':
    main()
