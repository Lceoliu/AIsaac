"""visible-v3 range audit (the coordinator, 2026-11-17): the AB+ values behind the producer's argument ranges, read on
the game (the static ones -- bounds checks in the getters -- come from the disassembly, see schema_v3.ABP_RANGES):

  hearts      a run as Isaac given black / soul / bone hearts, then some taken away: GetBlackHearts' mask and
              IsBoneHeart(0..31) against the heart counts (are there bits past the heart slots, stale or not?)
  items       Isaac.GetItemConfig():GetCollectibles().Size against CollectibleType.NUM_COLLECTIBLES (the counts the
              producer reads with GetCollectibleNum(id), id < Size, have no bounds check in the engine)
  pockets     GetMaxPoketItems / GetMaxTrinkets bare, with Polydactyly (454) and Mom's Purse (139), and the pocket
              contents per slot 0..1

usage (bridge python dir, PYTHONPATH=.): python abplus_probe_v3_ranges.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
"""
import argparse
import json
import os
from pathlib import Path

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance

HEARTS_LUA = r"""
local json = require("json")
local p = Isaac.GetPlayer(0)
local function read(tag)
  local bone = 0
  for i = 0, 31 do if p:IsBoneHeart(i) then bone = bone | (1 << i) end end
  return { tag = tag, max = p:GetMaxHearts(), hearts = p:GetHearts(), soul = p:GetSoulHearts(),
    black_mask = p:GetBlackHearts(), bone_hearts = p:GetBoneHearts(), bone_bits = bone }
end
local out = { read('start') }
p:AddBlackHearts(6) p:AddSoulHearts(2) p:AddBoneHearts(2)
out[#out + 1] = read('+3 black, +1 soul, +2 bone')
p:AddSoulHearts(-6)
out[#out + 1] = read('-3 soul hearts')
p:AddBoneHearts(-1)
out[#out + 1] = read('-1 bone heart')
p:AddBlackHearts(4) p:AddBoneHearts(3)
out[#out + 1] = read('+2 black, +3 bone')
p:AddMaxHearts(-6) p:AddSoulHearts(-20) p:AddBoneHearts(-10) p:AddMaxHearts(2) p:AddHearts(2)
out[#out + 1] = read('everything taken, one red heart back')
return json.encode(out)
"""
ITEMS_LUA = r"""
local json = require("json")
local p = Isaac.GetPlayer(0)
local function pockets(tag)
  local s = {}
  for i = 0, 1 do s[#s + 1] = { p:GetCard(i), p:GetPill(i) } end
  return { tag = tag, max_pockets = p:GetMaxPoketItems(), max_trinkets = p:GetMaxTrinkets(), slots = s,
    trinkets = { p:GetTrinket(0), p:GetTrinket(1), p:GetTrinket(2) } }
end
local out = { size = Isaac.GetItemConfig():GetCollectibles().Size, num = CollectibleType.NUM_COLLECTIBLES,
  pockets = { pockets('bare') } }
p:AddCard(1) p:AddPill(3)
out.pockets[#out.pockets + 1] = pockets('a card then a pill, one pocket')
p:AddCollectible(454, 0, false) p:AddCard(2)
out.pockets[#out.pockets + 1] = pockets('Polydactyly, a card')
p:AddCollectible(139, 0, false) p:AddTrinket(1) p:AddTrinket(2)
out.pockets[#out.pockets + 1] = pockets("Mom's Purse, two trinkets")
return json.encode(out)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--seed', type=int, default=2147800000)
    ap.add_argument('--port', type=int, default=44520)
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
    inst = Instance('ranges', args.port, gx, spec)
    res = {}
    try:
        b = inst.env.bridge
        b.reset_mode, b.character = 'floor', 0
        inst.reset(args.seed)
        for name, code in (('hearts', HEARTS_LUA), ('items', ITEMS_LUA)):
            c = b.fork(tag='ranges', alarm=600)
            try:
                res[name] = json.loads(c.lua(code))
            finally:
                c.close()
    finally:
        inst.kill()
    (out / 'ranges.json').write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
