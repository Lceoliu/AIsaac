"""visible-v3 minimap cells per RoomShape from the engine (the coordinator, 2026-11-17): for every room of many floors
(seeds x stages, as the V1 probe scans them), the 13 x 13 map cells g whose level:GetRoomByIdx(g) is that room (same
SafeGridIndex and GridIndex), as offsets from the room's GridIndex (the minimap's base cell). Per shape: the offset
sets seen (one expected), against tok_obs.SHAPE_CELLS and (0,) + SHAPE_CELLS (what the v2 / v3 reference writes).
-> <out>/mapcells.json (every room), printed summary.

usage (bridge python dir, PYTHONPATH=.): python abplus_probe_v3_mapcells.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
"""
import argparse
import json
import os
from pathlib import Path

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.tok_obs import SHAPE_CELLS

CELLS_LUA = r"""
local json = require("json")
local level = Game():GetLevel()
local rooms = level:GetRooms()
local out = {}
for i = 0, rooms.Size - 1 do
  local r = rooms:Get(i)
  local gi, safe = r.GridIndex, r.SafeGridIndex
  local ok, shape = pcall(function() return r.Data.Shape end)
  if gi >= 0 and gi < 169 and ok then
    local cells = {}
    for g = 0, 168 do
      local d = level:GetRoomByIdx(g)
      if d and d.SafeGridIndex == safe and d.GridIndex == gi then cells[#cells + 1] = g - gi end
    end
    out[#out + 1] = { gi, safe, shape, cells }
  end
end
return json.encode(out)
"""
SHAPES = {1: '1x1', 2: 'IH', 3: 'IV', 4: '1x2', 5: 'IIV', 6: '2x1', 7: 'IIH', 8: '2x2', 9: 'LTL', 10: 'LTR', 11: 'LBL',
          12: 'LBR'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--seeds', default='2147700000:40')
    ap.add_argument('--stages', default='1,2,3,4,5,6,7,8')
    ap.add_argument('--port', type=int, default=44550)
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
    inst = Instance('mapcells', args.port, gx, spec)
    first, count = (int(v) for v in args.seeds.split(':'))
    stages = [int(v) for v in args.stages.split(',')]
    rooms, seen = [], {}
    try:
        b = inst.env.bridge
        for seed in range(first, first + count):
            b = inst.env.bridge
            b.reset_mode, b.character = 'floor', 0
            inst.reset(seed)
            for stage in stages:
                if stage != 1:
                    b.reset(phases=[[f'stage {stage}']], settle=2)
                for gi, safe, shape, cells in json.loads(b.lua(CELLS_LUA)):
                    rooms.append(dict(seed=seed, stage=stage, gi=gi, safe=safe, shape=shape, cells=sorted(cells)))
                    seen.setdefault(shape, {}).setdefault(tuple(sorted(cells)), 0)
                    seen[shape][tuple(sorted(cells))] += 1
    finally:
        inst.kill()
    table = []
    for shape in sorted(seen):
        sets = seen[shape]
        engine = sorted(sets, key=lambda k: -sets[k])[0]
        st = tuple(sorted(SHAPE_CELLS.get(shape, (0,))))
        ref = tuple(sorted(set((0,) + SHAPE_CELLS.get(shape, ()))))
        table.append(dict(shape=shape, name=SHAPES.get(shape, '?'), engine=list(engine), rooms=sum(sets.values()),
                          variants={str(list(k)): v for k, v in sets.items()}, shape_cells=list(st),
                          reference=list(ref), equal_shape_cells=engine == st, equal_reference=engine == ref))
    (out / 'mapcells.json').write_text(json.dumps(dict(table=table, rooms=rooms), indent=1, sort_keys=True))
    for t in table:
        print(json.dumps({k: t[k] for k in ('shape', 'name', 'engine', 'rooms', 'shape_cells', 'reference',
                                             'equal_shape_cells', 'equal_reference', 'variants')}), flush=True)


if __name__ == '__main__':
    main()
