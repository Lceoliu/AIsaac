"""Which door does `goto s.boss.<v>` create? Probe Level.LeaveDoor / EnterDoor before the goto."""
import json, sys
from isaac_bridge.abplus import AbplusTrainingEnv, launch_abplus, stop_abplus

proc = launch_abplus("doorprobe", 27190, 'exact')
env = AbplusTrainingEnv(port=27190)
try:
    env.connect()
    for field in ('LeaveDoor', 'EnterDoor'):
        for slot in (0, 1, 2, 3, 0, 3):
            before = env.lua(f"local l = Game():GetLevel(); l.{field} = {slot}; return tostring(l.LeaveDoor) .. ',' .. tostring(l.EnterDoor)")
            obs, info = env.reset(phases=[["goto s.boss.1010"]], settle=2)
            after = env.lua("local l = Game():GetLevel(); return tostring(l.LeaveDoor) .. ',' .. tostring(l.EnterDoor)")
            doors = [(d['slot'], d['pos']) for d in obs['doors']]
            print(json.dumps({'set': field, 'value': slot, 'before': before, 'after': after, 'doors': doors,
                              'player': obs['players'][0]['pos']}), flush=True)
finally:
    try:
        env.close()
    finally:
        stop_abplus(proc, 'doorprobe')
