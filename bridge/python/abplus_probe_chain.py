"""Goal-conditioned line M0 probes P2/P3/P5/P6 (rl/docs/GOAL_CONDITIONED_DESIGN.md): a real Basement I floor, room A
(a 1x1 normal neighbour of the start room) and room B (a 1x1 normal neighbour of A), entered through real doors.

Per seed, three fresh runs (lua, walk, lua again for reproducibility) (`restart 0` with the A7 start-seed / sound / reseed hooks, as reset_monstro does):
- method 'lua': Level.LeaveDoor = slot, Game():StartRoomTransition(A, slot, 0) from the start room (P2);
- method 'walk': the player is put inside the start room's door to A and walks through it (engine's own door check).
Both then: A's first frames without input; A cleared by killing its active enemies (Lua) until the doors open; the doors
of A as the engine and the bridge observation list them, next to the floor's secret-room neighbours (P5); the player
put inside A's door to B, walking through it one logic frame per step, and in B switching to a perpendicular move to see
from which frame input acts (P3). Stage type and curses are read right after the restart, before anything removes a
curse (P6). Every step is one logic frame (repeat=1); each record keeps the bridge observation's frame counters, room
and doors next to a Lua snapshot (velocity, ControlsEnabled, LeaveDoor/EnterDoor, door states and targets).

usage: python abplus_probe_chain.py <out.jsonl> [seeds] [port] [first_seed]
Run on the Linux host only (one extra engine instance at nice 19).
"""
import json
import sys
import time

from isaac_bridge.abplus import AbplusTrainingEnv, launch_abplus, room_seed, stop_abplus

ENC = r"""
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
local function field(obj, name)
  if obj == nil then return -1 end
  local ok, v = pcall(function() return obj[name] end)
  if ok and v ~= nil then return v end
  return -1
end
"""

FLOOR = ENC + r"""
local g = Game(); local level = g:GetLevel()
local out = {stage = level:GetStage(), stage_type = level:GetStageType(), curses = level:GetCurses(),
  start = level:GetStartingRoomIndex(), current = level:GetCurrentRoomIndex(), rooms = {}}
local rooms = level:GetRooms()
for i = 0, rooms.Size - 1 do
  local r = rooms:Get(i); local d = r.Data
  out.rooms[#out.rooms + 1] = {grid = r.GridIndex, safe = r.SafeGridIndex, type = field(d, 'Type'),
    variant = field(d, 'Variant'), shape = field(d, 'Shape'), doors = field(d, 'Doors'), clear = r.Clear}
end
return enc(out)
"""

SNAP = ENC + r"""
local g = Game(); local level = g:GetLevel(); local room = g:GetRoom(); local p = Isaac.GetPlayer(0)
local doors = {}
for slot = 0, 7 do
  local d = room:GetDoor(slot)
  if d then
    doors[#doors + 1] = {slot = slot, open = d:IsOpen(), locked = d:IsLocked(), state = d.State,
      variant = d:GetVariant(), target = d.TargetRoomIndex, ttype = d.TargetRoomType, x = d.Position.X, y = d.Position.Y}
  end
end
local npc = 0
for _, e in ipairs(Isaac.GetRoomEntities()) do if e:IsActiveEnemy(false) then npc = npc + 1 end end
return enc({gf = g:GetFrameCount(), ri = level:GetCurrentRoomIndex(), rf = room:GetFrameCount(),
  px = p.Position.X, py = p.Position.Y, vx = p.Velocity.X, vy = p.Velocity.Y, ctrl = p.ControlsEnabled,
  ld = level.LeaveDoor, ed = level.EnterDoor, clear = room:IsClear(), npc = npc, doors = doors,
  hearts = p:GetHearts(), bombs = p:GetNumBombs(), cooldown = p:GetDamageCooldown()})
"""

ENTS = ENC + r"""
local out = {}
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e.Type ~= 1 and e.Type ~= 1000 then
    out[#out + 1] = {e.Type, e.Variant, e.SubType, math.floor(e.Position.X * 10 + 0.5), math.floor(e.Position.Y * 10 + 0.5),
      e.InitSeed, e.HitPoints}
  end
end
return enc(out)
"""

KILL = r"""
local n = 0
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e:IsActiveEnemy(false) and e:IsVulnerableEnemy() then e:Kill(); n = n + 1 end
end
return tostring(n)
"""

# 1x1 door slots LEFT0 UP0 RIGHT0 DOWN0: grid offset, the move that walks through the door, a perpendicular move.
SLOTS = {0: (-1, 7, 1), 1: (-13, 1, 3), 2: (1, 3, 5), 3: (13, 5, 7)}
UNIT = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}


def neighbours(idx):
    col = idx % 13
    for slot, (offset, _, _) in SLOTS.items():
        if (slot == 0 and col == 0) or (slot == 2 and col == 12):
            continue
        yield slot, idx + offset


def pick_chain(floor):
    """(slot start->A, A, slot A->B, B) for 1x1 normal rooms whose layouts have the doors, or None."""
    by_grid = {r['safe']: r for r in floor['rooms']}

    def ok(r, slot):
        return r is not None and r['type'] == 1 and r['shape'] == 1 and r['doors'] != -1 and (r['doors'] >> slot) & 1

    for slot, a in neighbours(floor['start']):
        room_a = by_grid.get(a)
        if not ok(room_a, (slot + 2) % 4):
            continue
        for slot_b, b in neighbours(a):
            room_b = by_grid.get(b)
            if b != floor['start'] and ok(room_b, (slot_b + 2) % 4) and (room_a['doors'] >> slot_b) & 1:
                return slot, a, slot_b, b
    return None


def secret_neighbours(floor, idx):
    """The floor's secret (7) / super secret (8) rooms on a 1x1 room's four sides: {slot: type}."""
    by_grid = {r['safe']: r for r in floor['rooms']}
    return {slot: by_grid[n]['type'] for slot, n in neighbours(idx) if n in by_grid and by_grid[n]['type'] in (7, 8)}


def record(env, obs):
    snap = json.loads(env.lua(SNAP))
    room = obs['room']
    return dict(lf=obs['logic_frames'], gf_obs=obs['game_frame'], room_idx=room['room_idx'], room_frame=room['frame'],
                controls=obs['players'][0].get('controls'), bridge_doors=[(d['slot'], d['open'], d.get('target_type'))
                                                                           for d in obs['doors']], **snap)


def step(env, move, frames=1):
    obs, *_ = env.step({'move': move}, repeat=frames)
    return obs


def walk_through(env, slot, trace, max_frames=90):
    """Put the player 60 px inside this room's door `slot`, hold the move into it until the room changes."""
    door = env.lua(f"local d = Game():GetRoom():GetDoor({slot}); if d == nil then return 'none' end; "
                   f"return tostring(d.Position.X) .. ',' .. tostring(d.Position.Y)")
    if door == 'none':
        return 'no door'
    dx, dy = UNIT[slot]
    x, y = (float(v) for v in door.split(','))
    env.lua(f"local p = Isaac.GetPlayer(0); p.Position = Vector({x - 60 * dx}, {y - 60 * dy}); "
            f"p.Velocity = Vector(0, 0); return 'ok'")
    start = env.lua("return tostring(Game():GetLevel():GetCurrentRoomIndex())")
    move = SLOTS[slot][1]
    for _ in range(max_frames):
        rec = record(env, step(env, move))
        trace.append(rec)
        if str(rec['ri']) != start:
            return 'changed'
    return 'no change'


def after_entry(env, trace, move, frames):
    for _ in range(frames):
        trace.append(record(env, step(env, move)))


def clear_room(env, trace, max_frames=900):
    """Kill the room's vulnerable active enemies every 2 frames until the room is clear and every door is open."""
    killed = 0
    for f in range(0, max_frames, 2):
        killed += int(env.lua(KILL) or 0)
        rec = record(env, step(env, 0, 2))
        if f % 20 == 0 or rec['clear']:
            trace.append(rec)
        if rec['clear'] and all(d['open'] for d in rec['doors'] if d['ttype'] not in (7, 8)):
            return dict(killed=killed, frames=f + 2, clear=True)
    return dict(killed=killed, frames=max_frames, clear=False)


def fresh_run(env, seed):
    n = room_seed(seed, -1)
    hooks = env.lua(f"return tostring(os.getenv('ABP_SOUND_RESET')) .. tostring(os.getenv('ABP_RESEED:{n}')) .. "
                    f"tostring(os.getenv('ABP_STARTSEED:{n}'))")
    t = time.perf_counter()
    obs, _ = env.reset(phases=[['restart 0']], settle=2)
    reset_s = time.perf_counter() - t
    return hooks, reset_s, obs, json.loads(env.lua(FLOOR))


def probe(env, seed, method):
    hooks, reset_s, obs, floor = fresh_run(env, seed)
    out = dict(seed=seed, method=method, hooks=hooks, reset_s=round(reset_s, 3), stage=floor['stage'],
               stage_type=floor['stage_type'], curses=floor['curses'], start=floor['start'], current=floor['current'],
               rooms=len(floor['rooms']))
    chain = pick_chain(floor)
    if chain is None:
        return dict(out, skipped='no 1x1 normal A/B chain')
    slot_a, a, slot_b, b = chain
    room_a = next(r for r in floor['rooms'] if r['safe'] == a)
    room_b = next(r for r in floor['rooms'] if r['safe'] == b)
    out.update(slot_a=slot_a, a=a, a_variant=room_a['variant'], slot_b=slot_b, b=b, b_variant=room_b['variant'],
               a_secret_neighbours=secret_neighbours(floor, a), start_doors=record(env, obs)['doors'])
    entry = []
    t = time.perf_counter()
    if method == 'lua':
        env.lua(f"local l = Game():GetLevel(); l.LeaveDoor = {slot_a}; Game():StartRoomTransition({a}, {slot_a}, 0); "
                f"return 'ok'")
        for _ in range(90):
            rec = record(env, step(env, 0))
            entry.append(rec)
            if rec['ri'] == a:
                break
        out['enter'] = 'changed' if entry and entry[-1]['ri'] == a else 'no change'
    else:
        out['enter'] = walk_through(env, slot_a, entry)
    out['enter_s'] = round(time.perf_counter() - t, 3)
    if out['enter'] != 'changed':
        return dict(out, entry=entry)
    out['a_entities'] = json.loads(env.lua(ENTS))
    after_entry(env, entry, 0, 30)
    out['entry'] = entry
    cleared = []
    out['clear_a'] = clear_room(env, cleared)
    out['clear_trace'] = cleared
    out['a_doors_after_clear'] = cleared[-1]['doors'] if cleared else None
    out['a_bridge_doors_after_clear'] = cleared[-1]['bridge_doors'] if cleared else None
    to_b = []
    out['enter_b'] = walk_through(env, slot_b, to_b)
    if out['enter_b'] == 'changed':
        out['b_entities'] = json.loads(env.lua(ENTS))
        after_entry(env, to_b, SLOTS[slot_b][1], 3)       # keep holding A's last move for 3 frames
        after_entry(env, to_b, SLOTS[slot_b][2], 20)      # then a perpendicular move
    out['to_b'] = to_b
    return out


def main():
    out_path = sys.argv[1]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 27840
    first = int(sys.argv[4]) if len(sys.argv) > 4 else 2147600100
    proc = launch_abplus('chainprobe', port, 'exact')
    env = AbplusTrainingEnv(port=port)
    try:
        env.connect()
        with open(out_path, 'w') as f:
            for seed in range(first, first + seeds):
                for method in ('lua', 'walk', 'lua'):
                    try:
                        rec = probe(env, seed, method)
                    except Exception as exc:   # keep probing the other seeds; the error is the finding
                        rec = dict(seed=seed, method=method, error=f'{type(exc).__name__}: {exc}')
                    f.write(json.dumps(rec) + '\n')
                    f.flush()
                    print(seed, method, rec.get('enter'), rec.get('enter_b'), rec.get('skipped') or rec.get('error') or '',
                          flush=True)
    finally:
        try:
            env.close()
        finally:
            stop_abplus(proc, 'chainprobe')
    print('wrote', out_path)


if __name__ == '__main__':
    main()
