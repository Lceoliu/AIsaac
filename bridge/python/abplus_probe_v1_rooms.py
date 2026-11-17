"""V1: the engine's room-shape table (2026-11-17, SINGLE_TRANSFORMER_DESIGN.md section 9.3 V1; the coordinate contract C1's
premise). Measured on AB+ (isaac.x64 sha256 b9bfef32...) for every RoomShape 1-12, at least --need distinct rooms each:

  per room (one JSON line in <out>/rooms.jsonl)
    the room     shape, type, variant, SafeGridIndex, stage / stage type, GetGridWidth / Height / Size, GetTopLeftPos,
                 GetBottomRightPos, GetCenterPos, GetGridPosition(i) of every cell
    cells        per cell: IsPositionInRoom(GetGridPosition(i), s) for every s of SIZES as a bit mask, the grid entity type
                 (-1 none), GetGridCollision(i)
    boundaries   GetGridIndex of points stepped across the cell edges around cell 0, the first inside cell and the last
                 (which cell a position belongs to: the edge convention behind "origin = GetGridPosition(0) - (20, 20)")
    clamping     GetClampedPosition of far points (the four corners) for size 0 and the player's Size
    doors        per slot 0-7: IsDoorSlotAllowed, GetDoorSlotPosition, the door's Position, GetGridIndex(Position), the
                 door entity's own GetGridIndex, target room type
    the player   its Size, PlayerType, position at entry, and over a scripted walk (every direction held to the walls,
                 the four corners, along each wall; the room's enemies removed, the player invincible): every sample's
                 position and GetGridIndex(position) -> <out>/walk.jsonl, the min / max in rooms.jsonl
    the payload  the lean observation at entry (raw bytes) -> <out>/payloads/<key>.lean (the V2 golden frames' inputs)
  per character of --chars (a floor reset as it): its Size and the inside mask of the start room for its Size
  (<out>/sizes.jsonl) -- does a cell's inside change with the player's size.

Rooms are entered by Game():ChangeRoom(SafeGridIndex) in a clone of the floor's template (the template is never
changed); floors from --seeds, then the stages of --stages ("stage N" on the template) until every shape has --need
rooms. Read only on the game except the clone's room change, enemy removal and walking.

usage (from the bridge's python dir, PYTHONPATH=.; CPU only, the game's instances as the other probes):
  python abplus_probe_v1_rooms.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

from abplus_probe_native_obs import read_raw
from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FLOOR_LUA, FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance

SHAPES = {1: '1x1', 2: 'IH', 3: 'IV', 4: '1x2', 5: 'IIV', 6: '2x1', 7: 'IIH', 8: '2x2', 9: 'LTL', 10: 'LTR', 11: 'LBL',
          12: 'LBR'}
SIZES = (0, 5, 10, 13, 15, 20, 25, 30, 40)   # IsPositionInRoom sizes, bit k of a cell's mask = SIZES[k]

ROOM_LUA = r"""
local json = require("json")
local game, level, room = Game(), Game():GetLevel(), Game():GetRoom()
local p = Isaac.GetPlayer(0)
local SIZES = {%s}
local function v(a) return { a.X, a.Y } end
local function try(f, ...) local ok, r = pcall(f, ...) if ok then return r end return "ERR:" .. tostring(r) end
local gw, gh, n = room:GetGridWidth(), room:GetGridHeight(), room:GetGridSize()
local desc = level:GetCurrentRoomDesc()
local out = { shape = room:GetRoomShape(), type = room:GetType(), gw = gw, gh = gh, n = n,
  safe = desc.SafeGridIndex, grid_index = desc.GridIndex, variant = try(function() return desc.Data.Variant end),
  stage = level:GetStage(), stage_type = try(function() return level:GetStageType() end), curses = level:GetCurses(),
  tl = v(room:GetTopLeftPos()), br = v(room:GetBottomRightPos()), center = v(room:GetCenterPos()),
  psize = p.Size, ptype = p:GetPlayerType(), ppos = v(p.Position), pgi = room:GetGridIndex(p.Position),
  gp = {}, inside = {}, gtype = {}, gcoll = {}, edges = {}, clamp = {}, doors = {} }
local first, last
for i = 0, n - 1 do
  local pos = room:GetGridPosition(i)
  out.gp[#out.gp + 1] = { pos.X, pos.Y }
  local m = 0
  for k, s in ipairs(SIZES) do if room:IsPositionInRoom(pos, s) then m = m | (1 << (k - 1)) end end
  out.inside[#out.inside + 1] = m
  if m & 1 ~= 0 then first = first or i; last = i end
  local g = room:GetGridEntity(i)
  out.gtype[#out.gtype + 1] = g and g:GetType() or -1
  out.gcoll[#out.gcoll + 1] = try(function() return room:GetGridCollision(i) end)
end
out.first_inside, out.last_inside = first, last
-- the cell a position belongs to, across the edges of cell 0, the first and the last inside cell (offsets from the
-- cell's GetGridPosition, -30..+30 px in 5 px steps, x and y apart)
for _, c in ipairs({ 0, first or 0, last or 0 }) do
  local base = room:GetGridPosition(c)
  local row = { cell = c, x = {}, y = {} }
  for d = -30, 30, 5 do
    row.x[#row.x + 1] = { d, room:GetGridIndex(Vector(base.X + d, base.Y)) }
    row.y[#row.y + 1] = { d, room:GetGridIndex(Vector(base.X, base.Y + d)) }
  end
  for _, d in ipairs({ -20.01, -20, -19.99, 19.99, 20, 20.01 }) do
    row.x[#row.x + 1] = { d, room:GetGridIndex(Vector(base.X + d, base.Y)) }
    row.y[#row.y + 1] = { d, room:GetGridIndex(Vector(base.X, base.Y + d)) }
  end
  out.edges[#out.edges + 1] = row
end
for _, s in ipairs({ 0, p.Size }) do
  for _, far in ipairs({ { -5000, -5000 }, { 5000, -5000 }, { -5000, 5000 }, { 5000, 5000 } }) do
    local c = room:GetClampedPosition(Vector(far[1], far[2]), s)
    out.clamp[#out.clamp + 1] = { s, far[1], far[2], c.X, c.Y, room:GetGridIndex(c) }
  end
end
for slot = 0, 7 do
  local d = { slot = slot, allowed = try(function() return room:IsDoorSlotAllowed(slot) end),
    slot_pos = try(function() return v(room:GetDoorSlotPosition(slot)) end) }
  if type(d.slot_pos) == "table" then d.slot_gi = room:GetGridIndex(Vector(d.slot_pos[1], d.slot_pos[2])) end
  local door = room:GetDoor(slot)
  if door then
    d.pos = v(door.Position); d.gi = room:GetGridIndex(door.Position)
    d.own_gi = try(function() return door:GetGridIndex() end)
    d.target_type = try(function() return door.TargetRoomType end)
  end
  out.doors[#out.doors + 1] = d
end
return json.encode(out)
""" % ', '.join(str(s) for s in SIZES)

ENTER_LUA = """
local idx = %d
Game():ChangeRoom(idx)
return 'ok'
"""
HERE_LUA = "return tostring(Game():GetLevel():GetCurrentRoomDesc().SafeGridIndex)"
CALM_LUA = """
AbpSetInvincible(true)
local n = 0
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e:IsActiveEnemy(false) or e.Type == 9 then e:Remove(); n = n + 1 end
end
return tostring(n)
"""
POS_LUA = ("local p = Isaac.GetPlayer(0) local r = Game():GetRoom() "
           "return string.format('%.3f,%.3f,%d', p.Position.X, p.Position.Y, r:GetGridIndex(p.Position))")
# the walk: (move code, decisions of 6 frames); move 1 up, 2 up-right, 3 right, ... 8 up-left (abp_bridge MOVE)
WALK = ([(m, 15) for m in (1, 3, 5, 7)] + [(m, 15) for m in (2, 4, 6, 8)]
        + [(8, 25), (3, 30), (5, 25), (7, 30), (1, 25)]      # the corners and along each wall, clockwise
        + [(4, 25), (7, 30), (1, 25), (3, 30), (5, 25)])     # and from the opposite corner, the other way round


def lua_json(c, code):
    text = c.lua(code)
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return dict(error=str(text)[:400])


def payload(c):
    cc = c.fork(lean=True, alarm=120)
    try:
        cc._send({"cmd": "obs"})
        return read_raw(cc)
    finally:
        cc.close()


def walk(c, key, wf):
    xs, ys, cells = [], [], set()
    for move, n in WALK:
        for _ in range(n):
            c.step({'move': move}, repeat=6)
            x, y, gi = c.lua(POS_LUA).split(',')
            x, y, gi = float(x), float(y), int(gi)
            xs.append(x); ys.append(y); cells.add(gi)
            wf.write(json.dumps(dict(key=key, move=move, x=x, y=y, gi=gi)) + '\n')
    wf.flush()
    return dict(x_min=min(xs), x_max=max(xs), y_min=min(ys), y_max=max(ys), samples=len(xs), cells=len(cells))


def measure(b, idx, key, out, rf, wf, meta):
    c = b.fork(tag='v1room', alarm=900)
    try:
        if idx != int(c.lua(HERE_LUA)):
            c.lua(ENTER_LUA % idx)
            for _ in range(20):
                c.step({}, repeat=6)
                if int(c.lua(HERE_LUA)) == idx:
                    break
        here = int(c.lua(HERE_LUA))
        if here != idx:
            return dict(key=key, idx=idx, error=f'ChangeRoom: in room {here}, not {idx}', **meta)
        removed = int(c.lua(CALM_LUA))
        c.step({}, repeat=6)
        rec = lua_json(c, ROOM_LUA)
        raw = payload(c)
        (out / 'payloads' / f'{key}.lean').write_bytes(raw)
        rec.update(key=key, idx=idx, enemies_removed=removed, payload_bytes=len(raw), **meta)
        rec['walk'] = walk(c, key, wf)
        rec['after_walk'] = lua_json(c, "local p = Isaac.GetPlayer(0) return require('json').encode({p.Position.X, "
                                        "p.Position.Y, p:GetPlayerType()})")
        return rec
    finally:
        c.close()


def sizes_pass(inst, chars, seed, sf):
    b = inst.env.bridge
    for char in chars:
        b.reset_mode, b.character = 'floor', char
        try:
            inst.reset(seed)
            rec = lua_json(b, ROOM_LUA)
            sf.write(json.dumps(dict(char=char, seed=seed, psize=rec.get('psize'), ptype=rec.get('ptype'),
                                     shape=rec.get('shape'), inside=rec.get('inside'))) + '\n')
        except Exception as e:   # reported
            sf.write(json.dumps(dict(char=char, seed=seed, error=f'{type(e).__name__}: {e}')) + '\n')
            inst.relaunch(hard=True)
        sf.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147700000:40')
    p.add_argument('--stages', default='1,2,3,4,5,6,7,8')
    p.add_argument('--need', type=int, default=3, help='rooms per shape')
    p.add_argument('--max-per-floor', type=int, default=2, help='rooms per shape taken from one floor')
    p.add_argument('--chars', default='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17')
    p.add_argument('--fpd', type=int, default=4)
    p.add_argument('--port', type=int, default=44480)
    p.add_argument('--name', default='v1rooms')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    (out / 'payloads').mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=args.fpd, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    first, count = (int(v) for v in args.seeds.split(':'))
    stages = [int(v) for v in args.stages.split(',') if v.strip()]
    chars = [int(v) for v in args.chars.split(',') if v.strip()]
    inst = Instance(args.name, args.port, gx, spec)
    got = {s: 0 for s in SHAPES}
    seen = set()
    errors = 0
    t0 = time.perf_counter()
    with open(out / 'rooms.jsonl', 'w') as rf, open(out / 'walk.jsonl', 'w') as wf, \
            open(out / 'sizes.jsonl', 'w') as sf, open(out / 'floors.jsonl', 'w') as ff:
        try:
            for seed in range(first, first + count):
                if all(n >= args.need for n in got.values()):
                    break
                b = inst.env.bridge
                b.reset_mode, b.character = 'floor', 0
                try:
                    inst.reset(seed)
                except Exception as e:   # reported; the instance relaunched
                    ff.write(json.dumps(dict(seed=seed, error=f'{type(e).__name__}: {e}')) + '\n'); ff.flush()
                    inst.relaunch(hard=True)
                    continue
                for stage in stages:
                    if all(n >= args.need for n in got.values()):
                        break
                    try:
                        if stage != 1:
                            b.reset(phases=[[f'stage {stage}']], settle=2)
                        parts = b.lua(FLOOR_LUA).split(';')
                        lv = lua_json(b, "local l = Game():GetLevel() return require('json').encode({l:GetStage(), "
                                         "l:GetStageType(), l:GetCurses()})")
                    except Exception as e:   # reported
                        ff.write(json.dumps(dict(seed=seed, stage=stage, error=f'{type(e).__name__}: {e}')) + '\n')
                        ff.flush()
                        break
                    rooms = []
                    for row in parts[2:]:
                        idx, t, var, shape, doors = (int(x) for x in row.split(','))
                        rooms.append((idx, t, var, shape))
                    ff.write(json.dumps(dict(seed=seed, stage=stage, level=lv, rooms=rooms)) + '\n'); ff.flush()
                    taken = {}
                    for idx, t, var, shape in rooms:
                        if shape not in got or got[shape] >= args.need or taken.get(shape, 0) >= args.max_per_floor:
                            continue
                        if (shape, var, t) in seen:   # (a distinct room: its layout variant)
                            continue
                        key = f'{SHAPES[shape]}-s{seed}-st{stage}-r{idx}'
                        try:
                            rec = measure(b, idx, key, out, rf, wf, dict(seed=seed, stage_cmd=stage))
                        except Exception as e:   # reported; the template's clone is gone, the template kept
                            rec = dict(key=key, idx=idx, seed=seed, stage_cmd=stage, error=f'{type(e).__name__}: {e}')
                        rf.write(json.dumps(rec) + '\n'); rf.flush()
                        if 'error' in rec:
                            errors += 1
                            print('ERROR', key, rec['error'][:300], flush=True)
                            continue
                        seen.add((shape, var, t))
                        taken[shape] = taken.get(shape, 0) + 1
                        got[shape] += 1
                        print(key, 'shape', rec.get('shape'), 'gw', rec.get('gw'), 'gh', rec.get('gh'), 'tl', rec.get('tl'),
                              'gp0', rec['gp'][0] if rec.get('gp') else None, 'walk', rec.get('walk'), flush=True)
            sizes_pass(inst, chars, first, sf)
        finally:
            inst.kill()
    summary = dict(got={SHAPES[s]: n for s, n in got.items()}, errors=errors, seconds=round(time.perf_counter() - t0, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
