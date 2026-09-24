"""Stage/stage type after the per-episode restart, and reset wall time."""
import json, sys, time
from isaac_bridge.abplus import AbplusTransformerEnv, launch_abplus, stop_abplus

proc = launch_abplus('resetprobe', 27195, 'exact')
env = AbplusTransformerEnv(port=27195)
try:
    times = []
    for seed in range(2147483728, 2147483728 + 12):
        t = time.perf_counter()
        obs, info = env.reset(options={'arena_seed': seed})
        times.append(time.perf_counter() - t)
        st = env.bridge.lua("local l = Game():GetLevel(); return tostring(l:GetStage()) .. ' ' .. tostring(l:GetStageType()) .. ' ' .. tostring(l:GetCurses()) .. ' ' .. Game():GetSeeds():GetStartSeedString()")
        print(json.dumps({'seed': seed, 'stage': st, 'attempts': info.get('room_attempts'), 'reset_s': round(times[-1], 3),
                          'grid': sorted(set((g[1], g[2]) for g in env.raw_obs['grid']))}), flush=True)
    print('mean reset s', sum(times[1:]) / len(times[1:]))
finally:
    try:
        env.close()
    finally:
        stop_abplus(proc, 'resetprobe')
