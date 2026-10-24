"""Which boss rooms does the AB+ level generator put on Basement I? Empirical pool by restarts.

Each run: reseed the global MT19937, `restart 0` (new run seed, Basement I), then read the boss
room of the generated level (Level:GetRooms, RoomDescriptor.Data). Output: variant/name counts.
usage: python abplus_probe_boss_pool.py <out.json> [runs]
"""
import collections, json, sys
from isaac_bridge.abplus import AbplusTrainingEnv, launch_abplus, stop_abplus

LUA = """
local level = Game():GetLevel(); local rooms = level:GetRooms()
local out = tostring(level:GetStage()) .. '|' .. tostring(level:GetStageType())
for i = 0, rooms.Size - 1 do
  local d = rooms:Get(i).Data
  if d and d.Type == RoomType.ROOM_BOSS then out = out .. '|' .. tostring(d.Variant) .. '|' .. tostring(d.Name) .. '|' .. tostring(d.Shape) end
end
return out
"""

out, runs = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300
proc = launch_abplus('bosspool', 27192, 'exact')
env = AbplusTrainingEnv(port=27192)
counts = collections.Counter()
stages = collections.Counter()
try:
    env.connect()
    for i in range(runs):
        env.lua(f"return os.getenv('ABP_RESEED:{0x51000000 + i}')")
        env.reset(phases=[['restart 0']], settle=2)
        parts = env.lua(LUA).split('|')
        stages[tuple(parts[:2])] += 1
        for j in range(2, len(parts), 3):
            counts[(int(parts[j]), parts[j + 1], int(parts[j + 2]))] += 1
finally:
    try:
        env.close()
    finally:
        stop_abplus(proc, 'bosspool')
rows = [dict(variant=v, name=n, shape=s, count=c) for (v, n, s), c in counts.most_common()]
json.dump(dict(runs=runs, stages={'|'.join(k): v for k, v in stages.items()}, rooms=rows), open(out, 'w'), indent=1)
names = collections.Counter()
for r in rows: names[r['name'].replace(' (copy)', '')] += r['count']
print(json.dumps(dict(runs=runs, stages={"|".join(k): v for k, v in stages.items()}, rooms=len(rows), by_name=names.most_common())))
