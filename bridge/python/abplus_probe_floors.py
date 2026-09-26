"""Dump whole AB+ floors for an exact comparison with the offline generator (rl/macro).

Each run: reseed the global MT19937, `restart 0` (the instance starts with --set-stage=1
--set-stage-type=0, i.e. Game::StartDebug: new run seed, Basement I) and dump the floor. Then set
the run's player resources (coins / keys / red hearts vary by run, so the resource-gated special
rooms are covered) and run one set of console `stage` commands, dumping after each. The console
command calls Level::SetStage and Level::Init synchronously (submit_input 0x12B2D1/0x12B2DD), so
the dumped Seeds:GetStageSeed is the seed the level was generated from.

A dump holds the seeds, curses, player resources, run state flags and every room descriptor (grid
position, layout, and the three per-room seeds, which pin the level RNG chain room by room).

usage: python abplus_probe_floors.py <out.jsonl> [runs] [port]
Compare: python rl/macro/tools/compare_engine_floors.py <out.jsonl>
Run on the Linux host only (it starts one extra engine instance at nice 19).
"""
import json
import sys

from isaac_bridge.abplus import AbplusTrainingEnv, launch_abplus, stop_abplus

LUA = r"""
local g = Game(); local level = g:GetLevel(); local seeds = g:GetSeeds(); local p = Isaac.GetPlayer(0)
local st = level:GetStage()
local out = {stage = st, stage_type = level:GetStageType(), curses = level:GetCurses(),
  start_seed = seeds:GetStartSeed(), start_seed_str = seeds:GetStartSeedString(),
  stage_seed = seeds:GetStageSeed(st), start = level:GetStartingRoomIndex(),
  player = {hearts = p:GetHearts(), max_hearts = p:GetMaxHearts(), soul_hearts = p:GetSoulHearts(),
            keys = p:GetNumKeys(), coins = p:GetNumCoins(), bombs = p:GetNumBombs()},
  flags = {}, rooms = {}}
for f = 0, 63 do
  local ok, v = pcall(function() return g:GetStateFlag(f) end)
  if ok and v then out.flags[#out.flags + 1] = f end
end
local function field(obj, name)   -- the AB+ docs list no RoomConfig.Room members; probe each one
  if obj == nil then return -1 end
  local ok, v = pcall(function() return obj[name] end)
  if ok and v ~= nil then return v end
  return -1
end
local rooms = level:GetRooms()
for i = 0, rooms.Size - 1 do
  local r = rooms:Get(i); local d = r.Data
  out.rooms[#out.rooms + 1] = {list = r.ListIndex, grid = r.GridIndex, safe = r.SafeGridIndex,
    type = field(d, 'Type'), variant = field(d, 'Variant'), subtype = field(d, 'Subtype'),
    shape = field(d, 'Shape'), spawn_seed = r.SpawnSeed, decoration_seed = r.DecorationSeed,
    award_seed = r.AwardSeed}
end
local function enc(v)
  if type(v) == 'table' then
    local parts, n = {}, #v
    if n > 0 or next(v) == nil then
      for i = 1, n do parts[i] = enc(v[i]) end
      return '[' .. table.concat(parts, ',') .. ']'
    end
    for k, x in pairs(v) do parts[#parts + 1] = string.format('%q:%s', tostring(k), enc(x)) end
    return '{' .. table.concat(parts, ',') .. '}'
  elseif type(v) == 'string' then return string.format('%q', v)
  else return tostring(v) end
end
return enc(out)
"""

COMMAND_SETS = [
    [f'stage {n}' for n in (2, 3, 4, 5, 6, 7, 8, 10, 11)],
    [f'stage {n}a' for n in (1, 2, 3, 4, 5, 6, 7, 8, 10, 11)],
    [f'stage {n}b' for n in (1, 2, 3, 4, 5, 6, 7, 8)],
]
COINS, KEYS, HEART_LOSS = (0, 5, 12), (0, 2), (0, 2)


def resources(i):
    return COINS[i % 3], KEYS[(i // 3) % 2], HEART_LOSS[(i // 6) % 2]


out_path = sys.argv[1]
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 150
port = int(sys.argv[3]) if len(sys.argv) > 3 else 27193
proc = launch_abplus('floorprobe', port, 'exact')
env = AbplusTrainingEnv(port=port)
try:
    env.connect()
    with open(out_path, 'w') as out:
        for i in range(runs):
            env.lua(f"return os.getenv('ABP_RESEED:{0x52000000 + i}')")
            env.reset(phases=[['restart 0']], settle=2)
            commands = COMMAND_SETS[i % len(COMMAND_SETS)]
            for step, command in enumerate([None] + commands):
                if command is not None:
                    env.reset(phases=[[command]], settle=2)
                rec = json.loads(env.lua(LUA))
                rec.update(run=i, step=step, command=command or 'restart 0')
                out.write(json.dumps(rec) + '\n')
                if step == 0:
                    coins, keys, loss = resources(i)
                    env.lua(f"local p = Isaac.GetPlayer(0); p:AddCoins({coins}); p:AddKeys({keys}); "
                            f"p:AddHearts(-{loss}); return 'ok'")
            out.flush()
finally:
    try:
        env.close()
    finally:
        stop_abplus(proc, 'floorprobe')
print('wrote', out_path)
