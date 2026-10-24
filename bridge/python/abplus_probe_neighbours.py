"""Companion of abplus_probe_combat_exit.py: (1) what lies behind each door of a chain's room A (room type and shape of
the neighbour, by Level:GetRoomByIdx of the cell beyond the door; big rooms included), over many chain resets; (2) whether
a room loaded by `goto d.<variant>` can be re-entered once the player walked out (and GetRoomByIdx(-3).Clear).
usage: python abplus_probe_neighbours.py [port] [first_seed] [chains]   (Linux host only)"""
import collections
import json
import sys

from abplus_probe_combat_exit import bomb_door, doors, walk_out, STATE
from isaac_bridge.abplus import AbplusTrainingEnv, launch_abplus, stop_abplus
from isaac_bridge.abplus_groups import load_groups

NEIGHBOURS = r"""
local l = Game():GetLevel(); local r = Game():GetRoom(); local t = {}
for s = 0, 7 do
  local d = r:GetDoor(s)
  if d then
    local desc = l:GetRoomByIdx(d.TargetRoomIndex)
    t[#t + 1] = string.format('%d:%d:%d:%d', s, d.TargetRoomType, desc.Data and desc.Data.Shape or -1, d.TargetRoomIndex)
  end
end
return table.concat(t, ' ')
"""


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 27892
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 2147602000
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    group = {g['name']: g for g in load_groups('../abplus/catalog/goal_m2_groups.json')}['chain']
    proc = launch_abplus('probenb', port, 'exact')
    env = AbplusTrainingEnv(port=port)
    try:
        env.connect()
        env.chain_rooms = frozenset(int(v) for v in group['spec']['normal'])
        env.start_bombs, env.invincible = 1, True
        kinds = collections.Counter()
        doors_per_a = collections.Counter()
        for seed in range(first, first + count):
            obs, info = env._reset_chain(seed, 6)
            parts = env.lua(NEIGHBOURS).split()
            doors_per_a[len(parts)] += 1
            for p in parts:
                s, ttype, shape, tidx = (int(v) for v in p.split(':'))
                kinds[(ttype, shape)] += 1
        print(json.dumps(dict(case='neighbours', chains=count, doors_per_a=dict(doors_per_a),
                              by_type_shape={f'{t}/{s}': n for (t, s), n in sorted(kinds.items())})), flush=True)
        # (2) goto room: walk out, then try to come back through the door opposite the entry
        variant = int(group['spec']['normal'][0])
        env.reset(phases=[[f"goto d.{variant}"]], settle=4)
        for _ in range(10):
            env.step({}, repeat=1)
        clear_debug = env.lua("return tostring(Game():GetLevel():GetRoomByIdx(-3).Clear)")
        slot = next(iter(doors(env)))
        f, door = bomb_door(env, slot)
        out = walk_out(env, slot)
        where = env.lua(STATE)
        back = (slot + 2) % 4
        back_door = doors(env).get(back)
        rec = dict(case='goto_return', variant=variant, debug_room_clear_flag=clear_debug, left_by=slot, walk_out=out,
                   now=where, door_back=back_door)
        if back_door:
            for _ in range(200):   # kill the room so its doors open, then walk back
                env.lua("for _, e in ipairs(Isaac.GetRoomEntities()) do if e:IsActiveEnemy(false) and e:IsVulnerableEnemy() then e:Kill() end end return 'ok'")
                env.step({}, repeat=1)
                if doors(env).get(back, {}).get('open'):
                    break
            rec['door_back_after_clear'] = doors(env).get(back)
            rec['walk_back'] = walk_out(env, back)
            rec['after_walk_back'] = env.lua(STATE)
        print(json.dumps(rec), flush=True)
    finally:
        try:
            env.close()
        finally:
            stop_abplus(proc, 'probenb')


if __name__ == '__main__':
    main()
