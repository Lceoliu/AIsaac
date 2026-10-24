"""Live probe for COMBAT options that leave the room (user decision 2026-09-30: leaving does not end COMBAT; the design must
be sound). Establishes on the engine what the static reading of the AB+ decompile says or leaves open:

P1 a bomb next to a combat-closed door between two normal rooms blows it open (GridEntity_Door::CanBlowOpen /
   TryBlowOpen: variant 8, Busted, state open), and the player can walk out through it;
P2 on leaving room A uncleared its monsters are not saved (Room::SaveState / ShouldSaveEntity keeps bombs, pickups,
   slots, fires, shopkeepers, TNT only); on returning, which monsters are there, with what HP, and whether the busted
   door is still open while A has monsters again; also a monster damaged before leaving;
P3 the door of B back to A when B has monsters (can the player walk back without clearing B or bombing?);
P4 respawns are a function of the seed (the same chain twice);
P5 leaving within the clear delay (Room::SaveState marks a room whose clear countdown runs as cleared);
P6 a room loaded by the debug goto (the single-room training rooms): do its doors blow open at all.

Chain rooms as the training's chain reset (abplus.AbplusTrainingEnv._reset_chain, invincible player). One AB+ instance.
usage: python abplus_probe_combat_exit.py [port] [first_seed] [chains] > out.jsonl   (Linux host only)
"""
import json
import sys

from abplus_probe_chain import KILL, SLOTS, UNIT
from isaac_bridge.abplus import AbplusTrainingEnv, launch_abplus, stop_abplus
from isaac_bridge.abplus_groups import load_groups

DOORS = r"""
local r = Game():GetRoom(); local t = {}
for s = 0, 7 do
  local d = r:GetDoor(s)
  if d then
    local ok, busted = pcall(function() return d:IsBusted() end)
    t[#t + 1] = string.format('%d:%d:%d:%s:%s:%d:%d', s, d.State, d:GetVariant(), tostring(d:IsOpen()),
      ok and tostring(busted) or 'na', d.TargetRoomType, d.TargetRoomIndex)
  end
end
return table.concat(t, ' ')
"""
ENEMIES = r"""
local t = {}
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e:IsActiveEnemy(false) then
    t[#t + 1] = string.format('%d.%d.%d:%.1f/%.1f@%d,%d', e.Type, e.Variant, e.SubType, e.HitPoints, e.MaxHitPoints,
      math.floor(e.Position.X), math.floor(e.Position.Y))
  end
end
table.sort(t)
return table.concat(t, ' ')
"""
STATE = ("local l = Game():GetLevel(); local r = Game():GetRoom(); return tostring(l:GetCurrentRoomIndex()) .. ' clear=' .. "
         "tostring(r:IsClear()) .. ' alive=' .. tostring(r:GetAliveEnemiesCount())")


def doors(env):
    out = {}
    for part in env.lua(DOORS).split():
        s, state, variant, is_open, busted, ttype, tidx = part.split(':')
        out[int(s)] = dict(state=int(state), variant=int(variant), open=is_open == 'true', busted=busted,
                           ttype=int(ttype), tidx=int(tidx))
    return out


def enemies(env):
    text = env.lua(ENEMIES)
    return text.split() if text else []


def room_clear(env, idx):
    return env.lua(f"return tostring(Game():GetLevel():GetRoomByIdx({idx}).Clear)")


def frames(env, n=1, move=0):
    obs = None
    for _ in range(n):
        obs, *_ = env.step({'move': move}, repeat=1)
    return obs


def bomb_door(env, slot, wait=120):
    """A bomb 30 px inside door `slot`, the player moved away to the room's centre; frames until the bomb is gone.
    Returns (frames, the door after)."""
    env.lua(f"local r = Game():GetRoom(); local d = r:GetDoor({slot}); local c = r:GetCenterPos(); "
            f"local p = Isaac.GetPlayer(0); p.Position = c; p.Velocity = Vector(0, 0); "
            f"local v = (c - d.Position):Normalized(); "
            f"Isaac.Spawn(EntityType.ENTITY_BOMBDROP, 0, 0, d.Position + v * 30, Vector(0, 0), p); return 'ok'")
    for f in range(1, wait + 1):
        frames(env)
        left = env.lua("local n = 0; for _, e in ipairs(Isaac.GetRoomEntities()) do if e.Type == 4 then n = n + 1 end end; "
                       "return tostring(n)")
        if left == '0':
            return f, doors(env).get(slot)
    return wait, doors(env).get(slot)


def walk_out(env, slot, max_frames=90):
    """The player 60 px inside door `slot`, holding the move into it; the frames until the room changed (or None)."""
    d = env.lua(f"local d = Game():GetRoom():GetDoor({slot}); return tostring(d.Position.X) .. ',' .. tostring(d.Position.Y)")
    x, y = (float(v) for v in d.split(','))
    dx, dy = UNIT[slot]
    env.lua(f"local p = Isaac.GetPlayer(0); p.Position = Vector({x - 60 * dx}, {y - 60 * dy}); p.Velocity = Vector(0, 0); "
            "return 'ok'")
    start = env.lua("return tostring(Game():GetLevel():GetCurrentRoomIndex())")
    for f in range(1, max_frames + 1):
        frames(env, 1, SLOTS[slot][1])
        if env.lua("return tostring(Game():GetLevel():GetCurrentRoomIndex())") != start:
            return f
    return None


def transition(env, target, slot, settle=6):
    """Enter room `target` as through this room's door `slot` (A9: the same landing, doors and spawns as walking)."""
    env.lua(f"local l = Game():GetLevel(); l.LeaveDoor = {slot}; Game():StartRoomTransition({target}, {slot}, 0); return 'ok'")
    for _ in range(settle):
        frames(env)
    return env.lua(STATE)


def chain_case(env, seed):
    """P1-P3 on one chain: returns a record."""
    obs, info = env._reset_chain(seed, 6)
    slot_a, a, slot_b, b = info['chain'][:4]
    rec = dict(seed=seed, chain=list(info['chain'][:4]), attempts=info['room_attempts'])
    frames(env, 10)
    rec['a_enter'] = dict(state=env.lua(STATE), doors=doors(env), enemies=enemies(env))
    # damage one monster to half before leaving (P2: does the damage survive the respawn?)
    rec['damaged'] = env.lua("for _, e in ipairs(Isaac.GetRoomEntities()) do if e:IsActiveEnemy(false) and "
                             "e:IsVulnerableEnemy() then local hp = e.HitPoints; e.HitPoints = hp / 2; "
                             "return string.format('%d.%d %.1f->%.1f', e.Type, e.Variant, hp, e.HitPoints) end end return 'none'")
    f, door = bomb_door(env, slot_b)
    rec['bomb'] = dict(frames=f, door=door, state=env.lua(STATE))
    if not (door and door['open']):
        rec['result'] = 'door not blown open'
        return rec
    rec['walk_out_frames'] = walk_out(env, slot_b)
    frames(env, 10)
    rec['b_enter'] = dict(state=env.lua(STATE), doors=doors(env), enemies=enemies(env), a_clear=room_clear(env, a))
    back = (slot_b + 2) % 4
    rec['b_door_back'] = rec['b_enter']['doors'].get(back)
    rec['back_state'] = transition(env, a, back)
    frames(env, 10)
    rec['a_return'] = dict(state=env.lua(STATE), doors=doors(env), enemies=enemies(env))
    rec['a_return_door_b'] = rec['a_return']['doors'].get(slot_b)
    # can the player leave again through the busted door while A has monsters?
    again = walk_out(env, slot_b)
    rec['walk_out_again_frames'] = again
    if again is not None:
        frames(env, 6)
        rec['back_state_2'] = transition(env, a, back)
        frames(env, 10)
        rec['a_return_2'] = dict(enemies=enemies(env), door_b=doors(env).get(slot_b))
    return rec


def clear_delay_case(env, seed):
    """P5: kill A's monsters, leave by transition k frames later, then A's descriptor clear flag."""
    out = []
    for k in (0, 1, 3, 12):
        obs, info = env._reset_chain(seed, 6)
        slot_a, a, slot_b, b = info['chain'][:4]
        frames(env, 10)
        killed = 0
        for _ in range(30):
            killed += int(env.lua(KILL) or 0)
            frames(env)
            if env.lua("return tostring(Game():GetRoom():GetAliveEnemiesCount())") == '0':
                break
        clear_now = env.lua(STATE)
        frames(env, k)
        before = env.lua(STATE)
        after = transition(env, b, slot_b)
        out.append(dict(k=k, killed=killed, at_kill=clear_now, before_leave=before, in_b=after,
                        a_clear=room_clear(env, a)))
    return out


def goto_room_case(env, variant):
    """P6: a normal room loaded by `goto d.<variant>` (the single-room training reset): its doors and a bomb at each."""
    env.reset(phases=[[f"goto d.{variant}"]], settle=4)
    frames(env, 10)
    rec = dict(variant=variant, state=env.lua(STATE), level_room=env.lua("return tostring(Game():GetLevel():GetCurrentRoomIndex())"),
               doors=doors(env), enemies=len(enemies(env)))
    rec['bombed'] = {}
    for slot in list(rec['doors'])[:2]:
        f, door = bomb_door(env, slot)
        rec['bombed'][slot] = dict(frames=f, door=door)
        if door and door['open']:
            rec['walk_out_frames'] = walk_out(env, slot)
            rec['after_walk'] = env.lua(STATE)
            break
    return rec


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 27890
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 2147601000
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    group = {g['name']: g for g in load_groups('../abplus/catalog/goal_m2_groups.json')}['chain']
    proc = launch_abplus('probexit', port, 'exact')
    env = AbplusTrainingEnv(port=port)
    try:
        env.connect()
        env.chain_rooms = frozenset(int(v) for v in group['spec']['normal'])
        env.start_bombs = 1
        env.invincible = True
        for seed in range(first, first + count):
            for rep in range(2 if seed == first else 1):   # P4: the first chain twice
                try:
                    print(json.dumps(dict(case='chain', rep=rep, **chain_case(env, seed))), flush=True)
                except Exception as exc:
                    print(json.dumps(dict(case='chain', seed=seed, error=repr(exc))), flush=True)
        try:
            print(json.dumps(dict(case='clear_delay', seed=first, rows=clear_delay_case(env, first))), flush=True)
        except Exception as exc:
            print(json.dumps(dict(case='clear_delay', error=repr(exc))), flush=True)
        for variant in [int(v) for v in group['spec']['normal'][:3]]:
            try:
                print(json.dumps(dict(case='goto_room', **goto_room_case(env, variant))), flush=True)
            except Exception as exc:
                print(json.dumps(dict(case='goto_room', variant=variant, error=repr(exc))), flush=True)
    finally:
        try:
            env.close()
        finally:
            stop_abplus(proc, 'probexit')


if __name__ == '__main__':
    main()
