"""visible-v3 batch 4 evidence (the coordinator, 2026-11-18): the pickup Price values AB+ produces, for E col 31's
categorical code. Per character (Isaac 0, ??? 4, Keeper 14, the Lost 10, the Forgotten 16) and seed: a devil room
(goto s.devil.0), the floor's shop (ChangeRoom), the shop again with Steam Sale (64); every pickup's (variant,
subtype, Price) and the player's hearts. -> <out>/prices.jsonl, a summary of the distinct prices.

usage (bridge python dir, PYTHONPATH=.): python abplus_probe_v3_prices.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
"""
import argparse
import json
import os
from pathlib import Path

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FLOOR_LUA, FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance

READ_LUA = r"""
local json = require("json")
local p = Isaac.GetPlayer(0)
local out = { room_type = Game():GetRoom():GetType(), ptype = p:GetPlayerType(), hearts = p:GetHearts(),
  max_hearts = p:GetMaxHearts(), soul = p:GetSoulHearts(), coins = p:GetNumCoins(), pickups = {} }
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e.Type == 5 then
    local k = e:ToPickup()
    out.pickups[#out.pickups + 1] = { e.Variant, e.SubType, k and k.Price or 0 }
  end
end
return json.encode(out)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--seeds', default='2148300000:4')
    ap.add_argument('--chars', default='0,4,14,10,16')
    ap.add_argument('--port', type=int, default=44560)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    inst = Instance('prices', args.port, gx, spec)
    first, count = (int(v) for v in args.seeds.split(':'))
    seen = {}
    with open(out / 'prices.jsonl', 'w') as f:
        try:
            for seed in range(first, first + count):
                for char in (int(v) for v in args.chars.split(',')):
                    b = inst.env.bridge
                    b.reset_mode, b.character = 'floor', char
                    try:
                        inst.reset(seed)
                    except Exception as e:   # reported
                        f.write(json.dumps(dict(seed=seed, char=char, error=str(e))) + '\n')
                        inst.relaunch(hard=True)
                        continue
                    shops = [int(r.split(',')[0]) for r in b.lua(FLOOR_LUA).split(';')[2:] if int(r.split(',')[1]) == 2]
                    for case, setup in (('devil', "Isaac.ExecuteCommand('goto s.devil.0') return 'ok'"),
                                        ('shop', f"Game():ChangeRoom({shops[0]}) return 'ok'" if shops else None),
                                        ('shop+steam_sale', f"Isaac.GetPlayer(0):AddCollectible(64, 0, false) "
                                                            f"Game():ChangeRoom({shops[0]}) return 'ok'" if shops else None)):
                        if setup is None:
                            continue
                        c = b.fork(tag='prices', alarm=600)
                        try:
                            c.lua(setup)
                            for _ in range(40):
                                c.step({}, repeat=1)
                            rec = dict(seed=seed, char=char, case=case, **json.loads(c.lua(READ_LUA)))
                        finally:
                            c.close()
                        f.write(json.dumps(rec) + '\n'); f.flush()
                        for v, s, price in rec['pickups']:
                            seen.setdefault(str(price), set()).add(f'{case}/char{char}')
        finally:
            inst.kill()
    summary = {k: sorted(v) for k, v in sorted(seen.items(), key=lambda kv: int(kv[0]))}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
