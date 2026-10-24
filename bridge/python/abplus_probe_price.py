"""Probe of the item_left take filter's price read (Phase B2, 2026-10-07; tok_branch.pedestal_prices: Lua
EntityPickup.Price, the engine's Entity_Pickup::GetPrice at +0xC30). One instance in the bridge's floor mode with items:
a pedestal is spawned in the start room and read (price 0); its Price set to 15 (a shop item) and read again; then the
take filter (options_of for an item_left point) with the player holding 0 coins: the priced pedestal is passed over,
and a second, free pedestal is chosen instead; a scripted walk into the priced one does not take it.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_price.py --groups-file ../abplus/catalog/scaling2_groups.json
"""
import argparse
import json
import os

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.tok_branch import KIND_ITEM_LEFT, Point, inv_counts, options_of, pedestal_prices, walk_to
from isaac_bridge.tok_sampler import apply_instance_defaults, default_stub_list, lean_step

SPAWN = ("local p = Isaac.GetPlayer(0) local room = Game():GetRoom() "
         "local pos = room:FindFreeTilePosition(p.Position + Vector({dx}, {dy}), 0) "
         "local e = Isaac.Spawn(5, 100, {sub}, pos, Vector(0, 0), nil) {extra} "
         "return tostring(pos.X) .. ',' .. tostring(pos.Y)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seed', type=int, default=2147600000)
    p.add_argument('--port', type=int, default=43910)
    args = p.parse_args()
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    os.environ.update(FORK_ENV)
    apply_instance_defaults()
    os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=default_stub_list(default_preload()))
    inst = Instance('brnprice', args.port, gx, spec)
    rec = {}
    try:
        inst.env.bridge.reset_mode = 'floor'
        inst.reset(args.seed)
        tpl = inst.env.bridge.fork(tag='template', alarm=0)
        E = tpl.fork(lean=True, alarm=300, reseed=3)
        dec = LeanDecoder()
        E._send({"cmd": "obs"})
        read_lean(E, dec)
        x1, y1 = (float(v) for v in E.lua(SPAWN.format(sub=1, dx=0, dy=120, extra='')).split(','))
        rec['after_spawn'] = pedestal_prices(E)
        rec['set_price'] = E.lua("for _, e in ipairs(Isaac.GetRoomEntities()) do if e.Type == 5 and e.Variant == 100 "
                                 "and e.SubType == 1 then local pk = e:ToPickup() pk.Price = 15 "
                                 "return tostring(pk.Price) .. ' shop=' .. tostring(pk:IsShopItem()) end end "
                                 "return 'none'")
        o = lean_step(E, dec, (0, 0, 0, 0, 0), 4)
        rec['after_price'] = pedestal_prices(E)
        rec['coins'] = E.lua("return tostring(Isaac.GetPlayer(0):GetNumCoins())")
        x2, y2 = (float(v) for v in E.lua(SPAWN.format(sub=2, dx=-120, dy=0, extra='')).split(','))
        o = lean_step(E, dec, (0, 0, 0, 0, 0), 4)
        rec['two'] = pedestal_prices(E)
        point = Point(kind=KIND_ITEM_LEFT, item=1, base=None, d=0, t=1, offset=0, applied=[], key=('item', 1),
                      visited=(), stage=1, seed=0, episode=0, room=0, n_alt=0, row=None, reseed=None, pid=1, order=1)
        notes = {}
        opts = options_of(point, o, None, E, notes)
        rec['options'] = [(h, round(t[0]), round(t[1]), j) for h, t, j in opts]
        rec['notes'], rec['chosen_item'] = notes, point.item
        W = E.fork(lean=True, alarm=300)
        d2 = LeanDecoder()
        W._send({"cmd": "obs"})
        o2 = read_lean(W, d2)
        o2, k, got = walk_to(W, d2, o2, (x1, y1), lambda q: inv_counts(q).get(1, 0) > 0, 75, 4, lean_step)
        pl = o2.players[0]
        rec['walk_into_priced'] = dict(decisions=k, taken=got, dist=round(((pl['x'] - x1) ** 2 + (pl['y'] - y1) ** 2)
                                                                          ** 0.5, 1))
        W.close()
        E.close()
        tpl.close()
    finally:
        inst.close()
    print(json.dumps(rec, default=str, indent=1))


if __name__ == '__main__':
    main()
