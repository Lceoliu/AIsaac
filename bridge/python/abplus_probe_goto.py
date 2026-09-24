"""What does `goto s.boss.<v>` spawn, and when? Entities/grid per frame after the goto, before any cleanup."""
import json, sys
from isaac_bridge.abplus import AbplusTrainingEnv, launch_abplus, stop_abplus, sim_arena

seed = int(sys.argv[1]); runs = int(sys.argv[2]); frames = int(sys.argv[3])
arena = sim_arena(seed)
proc = launch_abplus('gotoprobe', 27192, 'exact')
env = AbplusTrainingEnv(port=27192)
try:
    env.connect()
    for run in range(runs):
        env.lua(f"Game():GetLevel().LeaveDoor = {(arena['entrance'] + 2) % 4}")
        obs, info = env.reset(phases=[[f"goto s.boss.{arena['variant']}"]], settle=0)
        seen = []
        for f in range(frames):
            ents = [(e['id'], e['type'], e['variant'], e['subtype'], e.get('anim'), e.get('aframe'), e.get('age'),
                     [round(v) for v in e['pos']]) for e in obs['entities'] if e['type'] not in (1000,)]
            key = [(x[0], x[1], x[3]) for x in ents]
            if not seen or seen[-1][1] != key:
                seen.append((f, key, ents))
            obs, _, _, _, info = env.step({}, repeat=1)
        extra = env.lua("local r = Game():GetRoom(); local l = Game():GetLevel(); "
                        "return tostring(r:GetFrameCount()) .. ' clear=' .. tostring(r:IsClear()) .. ' first=' .. "
                        "tostring(r:IsFirstVisit()) .. ' curses=' .. tostring(l:GetCurses()) .. ' stage=' .. tostring(l:GetStage())")
        print(json.dumps({'run': run, 'changes': [(f, ents) for f, _, ents in seen], 'room': extra}), flush=True)
finally:
    try:
        env.close()
    finally:
        stop_abplus(proc, 'gotoprobe')
