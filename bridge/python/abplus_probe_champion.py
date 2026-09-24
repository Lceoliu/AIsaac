"""When does the arena Monstro's SubType change? Read it in the spawning Lua call and per frame after."""
import json, sys
from isaac_bridge.abplus import AbplusTrainingEnv, launch_abplus, stop_abplus, sim_arena, GOTO_SETTLE_FRAMES

seed = int(sys.argv[1]); runs = int(sys.argv[2])
arena = sim_arena(seed)
(px, py), (bx, by) = arena['player'], arena['boss']
SPAWN = f"""
local room = Game():GetRoom()
for _, e in ipairs(Isaac.GetRoomEntities()) do if e.Type ~= EntityType.ENTITY_PLAYER then e:Remove() end end
local reseeded = os.getenv("ABP_RESEED:{seed & 0xFFFFFFFF}")
local m = Isaac.Spawn(EntityType.ENTITY_MONSTRO, 0, 0, Vector({bx}, {by}), Vector(0, 0), nil)
room:SetClear(false)
return tostring(m.SubType) .. ' seed=' .. tostring(m.InitSeed) .. ' reseeded=' .. tostring(reseeded) .. ' bosscount=' .. tostring(room:GetAliveBossesCount())
"""
proc = launch_abplus('subprobe', 27193, 'exact')
env = AbplusTrainingEnv(port=27193)
try:
    env.connect()
    for run in range(runs):
        env.lua(f"Game():GetLevel().LeaveDoor = {(arena['entrance'] + 2) % 4}")
        obs, info = env.reset(phases=[[f"goto s.boss.{arena['variant']}"]], settle=GOTO_SETTLE_FRAMES)
        pre = [(e['type'], e['subtype']) for e in obs['entities'] if e['type'] == 20]
        at_spawn = env.lua(SPAWN)
        seq = []
        for f in range(12):
            obs, _, _, _, info = env.step({}, repeat=1)
            seq.append([(e['id'], e['subtype'], e.get('anim'), e.get('aframe')) for e in obs['entities'] if e['type'] == 20])
        print(json.dumps({'run': run, 'layout_boss_before_cleanup': pre, 'at_spawn': at_spawn, 'frames': seq}), flush=True)
finally:
    try:
        env.close()
    finally:
        stop_abplus(proc, 'subprobe')
