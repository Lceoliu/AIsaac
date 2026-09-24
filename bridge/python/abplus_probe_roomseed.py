"""Is the goto room's descriptor seed (champion roll, decorations) a function of the global MT19937?"""
import json, sys
from isaac_bridge.abplus import AbplusTrainingEnv, launch_abplus, stop_abplus, GOTO_SETTLE_FRAMES

proc = launch_abplus('seedprobe', 27194, 'exact')
env = AbplusTrainingEnv(port=27194)
DESC = ("local d = Game():GetLevel():GetCurrentRoomDesc(); "
        "local b = {} for _, e in ipairs(Isaac.GetRoomEntities()) do if e.Type == 20 then b[#b+1] = e.SubType end end "
        "return tostring(d.SpawnSeed) .. ' ' .. tostring(d.AwardSeed) .. ' ' .. tostring(d.DecorationSeed) .. ' boss=' .. table.concat(b, ',')")
try:
    env.connect()
    for reseed in (None, 12345, 12345, 777, 12345, None):
        if reseed is not None:
            env.lua(f'return os.getenv("ABP_RESEED:{reseed}")')
        env.lua("Game():GetLevel().LeaveDoor = 1")
        obs, info = env.reset(phases=[["goto s.boss.1010"]], settle=GOTO_SETTLE_FRAMES)
        grid = [g[:5] for g in obs['grid'] if g[1] not in (15, 16)][:6]
        print(json.dumps({'reseed': reseed, 'desc': env.lua(DESC), 'grid': grid}), flush=True)
finally:
    try:
        env.close()
    finally:
        stop_abplus(proc, 'seedprobe')
