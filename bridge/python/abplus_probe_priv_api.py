"""What AB+'s Lua offers for the privileged critic's fields (rl/docs/PREDICTIVE_MEMORY_DESIGN.md section 4.1), read from a
real floor: the level's room list (are secret and super secret rooms in it before they are found?), a room config's
spawn list (can an unvisited room's pickups be counted?), and each call's time in Lua. Prints JSON lines. Read-only.
usage (bridge's python dir, PYTHONPATH=.): python abplus_probe_priv_api.py --groups-file ../abplus/catalog/scaling2_groups.json --seeds 2147600000:3 --out <dir>
"""
import argparse
import json
from pathlib import Path

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus_goexplore import GxConfig, Instance

ROOMS_LUA = r"""
local out = {}
local level = Game():GetLevel()
local rooms = level:GetRooms()
for i = 0, rooms.Size - 1 do
  local r = rooms:Get(i)
  local d = r.Data
  out[#out + 1] = table.concat({r.GridIndex, r.SafeGridIndex, d.Shape, d.Type, r.DisplayFlags, r.VisitedCount, tostring(r.Clear)}, ":")
end
return "rooms=" .. rooms.Size .. " stage=" .. level:GetStage() .. " start=" .. level:GetStartingRoomIndex() .. " " .. table.concat(out, " ")
"""

KEYS_LUA = r"""
local out = {}
local level = Game():GetLevel()
local r = level:GetRooms():Get(0)
local function try(name, f)
  local ok, v = pcall(f)
  out[#out + 1] = name .. "=" .. (ok and tostring(v) or ("ERR " .. tostring(v):sub(1, 80)))
end
try("Data", function() return r.Data end)
try("Spawns", function() return r.Data.Spawns end)
try("SpawnCount", function() return r.Data.SpawnCount end)
try("Spawns.Size", function() return r.Data.Spawns.Size end)
try("Spawns:Get(0)", function() return r.Data.Spawns:Get(0) end)
try("Spawn.EntryCount", function() return r.Data.Spawns:Get(0).EntryCount end)
try("Spawn.X", function() return r.Data.Spawns:Get(0).X end)
try("Spawn:PickEntry", function() return r.Data.Spawns:Get(0):PickEntry(0) end)
try("Entry.Type", function() return r.Data.Spawns:Get(0):PickEntry(0).Type end)
try("Entry.Variant", function() return r.Data.Spawns:Get(0):PickEntry(0).Variant end)
try("Entry.Subtype", function() return r.Data.Spawns:Get(0):PickEntry(0).Subtype end)
try("SavedEntities", function() return r.SavedEntities end)
try("getmetatable(Data)", function()
  local mt = getmetatable(r.Data)
  local ks = {}
  for k in pairs(mt) do ks[#ks + 1] = tostring(k) end
  local g = mt.__propget or {}
  for k in pairs(g) do ks[#ks + 1] = "get:" .. tostring(k) end
  return table.concat(ks, ",")
end)
try("getmetatable(desc)", function()
  local mt = getmetatable(r)
  local ks = {}
  for k in pairs(mt) do ks[#ks + 1] = tostring(k) end
  local g = mt.__propget or {}
  for k in pairs(g) do ks[#ks + 1] = "get:" .. tostring(k) end
  return table.concat(ks, ",")
end)
return table.concat(out, " | ")
"""

COUNT_LUA = r"""
local level = Game():GetLevel()
local rooms = level:GetRooms()
local counts, total, rc = {}, 0, 0
for i = 0, rooms.Size - 1 do
  local d = rooms:Get(i).Data
  local sp = d.Spawns
  for j = 0, sp.Size - 1 do
    local s = sp:Get(j)
    local e = s:PickEntry(0)
    total = total + 1
    if e.Type == 5 then
      counts[e.Variant] = (counts[e.Variant] or 0) + 1
      rc = rc + 1
    end
  end
end
local o = {}
for k, v in pairs(counts) do o[#o + 1] = k .. ":" .. v end
return "spawns=" .. total .. " pickups=" .. rc .. " " .. table.concat(o, " ")
"""

TIME_LUA = r"""
local n = 2000
local t0 = os.clock()
for i = 1, n do
  local rooms = Game():GetLevel():GetRooms()
  for j = 0, rooms.Size - 1 do local r = rooms:Get(j); local d = r.Data; local a = d.Type + r.VisitedCount end
end
local t1 = os.clock()
for i = 1, n do
  for _, e in ipairs(Isaac.GetRoomEntities()) do
    local t = e.Type
    if t >= 10 and t < 1000 and e:ToNPC() then local a = e.HitPoints + e.MaxHitPoints end
  end
end
local t2 = os.clock()
return string.format("rooms %.1f us, entities %.1f us (%d entities)", (t1 - t0) / n * 1e6, (t2 - t1) / n * 1e6, #Isaac.GetRoomEntities())
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147600000:3')
    p.add_argument('--out', required=True)
    p.add_argument('--port', type=int, default=33700)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=a.groups_file, group='normal', tasks='', seconds=0.0))
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0)
    inst = Instance('privapi', a.port, gx, spec)
    s0, n = (int(v) for v in a.seeds.split(':'))
    try:
        inst.env.bridge.reset_mode = 'floor'
        for seed in range(s0, s0 + n):
            inst.reset(seed)
            c = inst.env.bridge.fork(tag='api', lean=True, alarm=300)
            for name, code in (('rooms', ROOMS_LUA), ('keys', KEYS_LUA), ('count', COUNT_LUA), ('time', TIME_LUA)):
                try:
                    print(json.dumps(dict(seed=seed, what=name, out=c.lua(code))), flush=True)
                except Exception as exc:
                    print(json.dumps(dict(seed=seed, what=name, error=str(exc)[:300])), flush=True)
            c.close()
    finally:
        inst.close()


if __name__ == '__main__':
    main()
