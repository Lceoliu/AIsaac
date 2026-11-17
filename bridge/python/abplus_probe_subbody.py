"""visible-v3 batch 3 evidence (the coordinator, 2026-11-17): when has a sub-player body (the Forgotten / the Soul) a real
position in the current room? For the Forgotten (PlayerType 16): at the start (the Forgotten controlled), after one
drop press (the Soul controlled), after a second press (back), a few frames apart, per body (the player and its
GetSubPlayer): PlayerType, InitSeed, Index, Position, Velocity, Visible, Exists, IsDead, ControlsEnabled,
EntityCollisionClass, GridCollisionClass, whether it is among Isaac.GetRoomEntities() and Isaac.FindByType(1)
(matched by InitSeed), the sprite's current frame and file. -> <out>/subbody.jsonl, printed.

usage (bridge python dir, PYTHONPATH=.): python abplus_probe_subbody.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
"""
import argparse
import json
import os
from pathlib import Path

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance

BODIES_LUA = r"""
local json = require("json")
local function info(p)
  if p == nil then return false end
  local in_room, in_find = false, false
  for _, e in ipairs(Isaac.GetRoomEntities()) do if e.InitSeed == p.InitSeed then in_room = true end end
  for _, e in ipairs(Isaac.FindByType(1, -1, -1, false, false)) do if e.InitSeed == p.InitSeed then in_find = true end end
  local spr = p:GetSprite()
  return { ptype = p:GetPlayerType(), seed = p.InitSeed, index = p.Index, x = p.Position.X, y = p.Position.Y,
    vx = p.Velocity.X, vy = p.Velocity.Y, visible = p.Visible, exists = p:Exists(), dead = p:IsDead(),
    controls = p.ControlsEnabled, coll = p.EntityCollisionClass, gcoll = p.GridCollisionClass,
    in_room_entities = in_room, in_find_by_type = in_find, frame = spr:GetFrame(), file = spr:GetFilename(),
    num_players = Game():GetNumPlayers() }
end
local p = Isaac.GetPlayer(0)
local ok, sub = pcall(function() return p:GetSubPlayer() end)
return json.encode({ game_frame = Game():GetFrameCount(), player = info(p), sub = ok and info(sub) or false })
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--seeds', default='2147800000:3')
    ap.add_argument('--port', type=int, default=44510)
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
    inst = Instance('subbody', args.port, gx, spec)
    first, count = (int(v) for v in args.seeds.split(':'))
    with open(out / 'subbody.jsonl', 'w') as f:
        try:
            for seed in range(first, first + count):
                b = inst.env.bridge
                b.reset_mode, b.character = 'floor', 16
                inst.reset(seed)
                c = b.fork(tag='subbody', alarm=600)
                try:
                    def rec(stage):
                        r = dict(seed=seed, stage=stage, **json.loads(c.lua(BODIES_LUA)))
                        f.write(json.dumps(r) + '\n'); f.flush()
                        print(json.dumps(r)[:700], flush=True)
                    rec('start')
                    c.step({'move': 3}, repeat=8)          # walk right a little: the controlled body moves
                    rec('start+walk')
                    c.step({'drop': 1}, repeat=4)
                    for _ in range(3):
                        c.step({}, repeat=4)
                    rec('after drop 1 (the Soul)')
                    c.step({'move': 5}, repeat=8)          # the Soul walks down: does the Forgotten stay?
                    rec('after drop 1 + walk')
                    c.step({'drop': 1}, repeat=4)
                    for _ in range(3):
                        c.step({}, repeat=4)
                    rec('after drop 2 (back)')
                finally:
                    c.close()
        finally:
            inst.kill()


if __name__ == '__main__':
    main()
