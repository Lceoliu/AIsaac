"""Goal-conditioned line M0, P5 follow-up (EXPERIMENTS.md A9): what an undiscovered secret-room door looks like before
and after a bomb reveals it, in the engine (GetDoor), in the bridge observation's doors and in its grid cells.

Per seed (floors whose room A, as abplus_probe_chain.py picks it, has a secret / super-secret neighbour): fresh
`restart 0` with the A7 hooks, Lua StartRoomTransition into A, A cleared by killing its enemies, then a bomb spawned
30 px inside the secret door; every logic frame for 150 frames the door (variant, State, IsOpen, TargetRoomType), the
bridge's door list and grid cells of type 16 (door) near it are recorded; finally the player walks through the door
(abplus_probe_chain.walk_through) to see which room it leads to.

usage: python abplus_probe_secret_door.py <out.jsonl> <seed> [<seed> ...]   (port 27850)
Run on the Linux host only (one extra engine instance at nice 19).
"""
import json
import sys

from abplus_probe_chain import (ENC, FLOOR, KILL, UNIT, clear_room, fresh_run, pick_chain, secret_neighbours, step,
                                walk_through)
from isaac_bridge.abplus import AbplusTrainingEnv, launch_abplus, stop_abplus

DOOR = ENC + r"""
local room = Game():GetRoom(); local d = room:GetDoor(SLOT)
if d == nil then return 'null' end
return enc({variant = d:GetVariant(), state = d.State, open = d:IsOpen(), locked = d:IsLocked(), ttype = d.TargetRoomType,
  target = d.TargetRoomIndex, x = d.Position.X, y = d.Position.Y, grid = d:GetGridIndex(),
  collision = room:GetGridCollision(d:GetGridIndex()),
  anim = (function() local ok, v = pcall(function() return d:GetSprite():GetAnimation() end); return ok and v or -1 end)()})
"""


def door_view(env, obs, slot):
    door = json.loads(env.lua(DOOR.replace('SLOT', str(slot))))
    bridge = [d for d in obs['doors'] if d['slot'] == slot]
    cells = [c for c in obs['grid'] if door != 'null' and c[0] == door['grid']] if door != 'null' else []
    return dict(lf=obs['logic_frames'], engine=door, bridge=bridge, grid_cell=cells)


def probe(env, seed):
    hooks, _, _, floor = fresh_run(env, seed)
    chain = pick_chain(floor)
    if chain is None:
        return dict(seed=seed, skipped='no chain')
    slot_a, a, _, _ = chain
    secrets = secret_neighbours(floor, a)
    if not secrets:
        return dict(seed=seed, skipped='A has no secret neighbour')
    slot, kind = next(iter(secrets.items()))
    slot = int(slot)
    env.lua(f"local l = Game():GetLevel(); l.LeaveDoor = {slot_a}; Game():StartRoomTransition({a}, {slot_a}, 0); return 'ok'")
    obs = step(env, 0)
    for _ in range(10):
        if obs['room']['room_idx'] == a:
            break
        obs = step(env, 0)
    out = dict(seed=seed, hooks=hooks, a=a, secret_slot=slot, secret_type=kind, before_clear=door_view(env, obs, slot))
    out['clear'] = clear_room(env, [])
    obs = step(env, 0)
    out['after_clear'] = door_view(env, obs, slot)
    d = out['after_clear']['engine']
    dx, dy = UNIT[slot]
    env.lua(f"Isaac.Spawn(4, 0, 0, Vector({d['x'] - 30 * dx}, {d['y'] - 30 * dy}), Vector(0, 0), nil); return 'ok'")
    trace, last = [], None
    for _ in range(150):
        obs = step(env, 0)
        view = door_view(env, obs, slot)
        key = json.dumps([view['engine'], view['bridge'], view['grid_cell']], sort_keys=True)
        if key != last:
            trace.append(view)
            last = key
    out['bomb_trace'] = trace
    walk = []
    out['walk'] = walk_through(env, slot, walk)
    out['walk_room'] = walk[-1]['ri'] if walk else None
    out['walk_room_type'] = next((r['type'] for r in floor['rooms'] if r['safe'] == out['walk_room']), None)
    return out


def main():
    out_path, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
    proc = launch_abplus('secretprobe', 27850, 'exact')
    env = AbplusTrainingEnv(port=27850)
    try:
        env.connect()
        with open(out_path, 'w') as f:
            for seed in seeds:
                try:
                    rec = probe(env, seed)
                except Exception as exc:
                    rec = dict(seed=seed, error=f'{type(exc).__name__}: {exc}')
                f.write(json.dumps(rec) + '\n')
                f.flush()
                print(seed, rec.get('skipped') or rec.get('error') or f"walk -> {rec.get('walk_room')} type {rec.get('walk_room_type')}",
                      flush=True)
    finally:
        try:
            env.close()
        finally:
            stop_abplus(proc, 'secretprobe')


if __name__ == '__main__':
    main()
