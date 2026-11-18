"""visible-v3 alignment dataset for package A (the coordinator, 2026-11-18): real AB+ steps in every RoomShape 1-12,
--per-shape frames at least each, as the training path makes them (lean clones, step lines, ISAAC_RL_VISIBLE_V3=2: the
native v3 frame, flag 64), each with the engine's own values read apart in the same game state:
  bodies    per P row (the frame's order: player 0's controlled body first, each sub-player right after its player):
            InitSeed, Position, room:GetGridIndex(Position), in_room (position_known), the frame's x / y
  entities  per E row (the sent entities, after the decorative filter and the 128 cap, in the frame's order): the engine
            entity it is (same type / variant / subtype, nearest canvas position; match error kept), its InitSeed,
            Position, room:GetGridIndex(Position)
  doors     every DoorSlot 0-7 with a door: Position, its GetGridIndex(), room:GetGridIndex(Position)
plus the formula cell floor((pos - O) / 40) of each (None when outside the grid: "no cell"), O = GetGridPosition(0) -
(20, 20). The stall columns (P 40 / 53) are 0 in the native frame (the worker backfills them; here they stay 0).

Rooms: per seed (stage 1 + k mod --stages), every room of the floor whose shape still needs frames is entered
(ChangeRoom) in a fresh lean clone and stepped --per-room decisions (4 frames each): the player walks holding a
direction 10 decisions at a time (into the walls), shoots, now and then bombs; at decision --door-at it is put one cell
inside a door slot of the room, that door opened (Open(): a room with monsters keeps them shut), and walks into the
door frame (through it the next room: the frames count for the shape they show). A record without a native frame (the bridge's fallback, e.g. after a lethal hit) ends
the room.

Writes <out>/frames.bin (the frames back to back), <out>/index.jsonl (one line per frame: where its bytes are, the room,
the engine values), <out>/summary.json, <out>/FORMAT.md.

usage (bridge python dir, PYTHONPATH=.): python abplus_probe_v3_align.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('ISAAC_RL_VISIBLE_V3', '2')

from abplus_probe_native_obs import read_raw   # noqa: E402
from goexplore_abplus import default_bridge_lua, default_preload, load_spec   # noqa: E402
from isaac_bridge import schema_v3 as V3, visible_producer as VP   # noqa: E402
from isaac_bridge.abplus import FLOOR_LUA, FORK_ENV   # noqa: E402
from isaac_bridge.abplus_goexplore import GxConfig, Instance   # noqa: E402
from isaac_bridge.abplus_lean import LeanDecoder   # noqa: E402

SHAPES = {1: '1x1', 2: 'IH', 3: 'IV', 4: '1x2', 5: 'IIV', 6: '2x1', 7: 'IIH', 8: '2x2', 9: 'LTL', 10: 'LTR',
          11: 'LBL', 12: 'LBR'}
ENTER_LUA = "Game():ChangeRoom(%d) return 'ok'"
# the engine's values (bodies in the frame's order: abp_bridge.lua v3_bodies, schema_v3 batch 4)
ENGINE_LUA = r"""
local json = require("json")
local function X(v) return string.format("%.17g", v) end   -- (exact: json.encode writes %.14g)
local room = Game():GetRoom()
local g0 = room:GetGridPosition(0)
local level = Game():GetLevel()
local out = { g0 = { X(g0.X), X(g0.Y) }, gw = room:GetGridWidth(), gh = room:GetGridHeight(), shape = room:GetRoomShape(),
  room_type = room:GetType(), here = level:GetCurrentRoomIndex(), frame = Game():GetFrameCount(),
  bodies = {}, entities = {}, doors = {} }
local in_room, seen = {}, {}
for _, e in ipairs(Isaac.GetRoomEntities()) do if e.Type == 1 then in_room[e.InitSeed] = true end end
local function body(p, role)
  if seen[p.InitSeed] then return end
  seen[p.InitSeed] = true
  out.bodies[#out.bodies + 1] = { seed = p.InitSeed, role = role, ptype = p:GetPlayerType(), x = X(p.Position.X),
    y = X(p.Position.Y), gi = room:GetGridIndex(p.Position), in_room = in_room[p.InitSeed] and 1 or 0 }
end
for i = 0, Game():GetNumPlayers() - 1 do
  local p = Isaac.GetPlayer(i)
  body(p, 1)
  local ok, sub = pcall(function() return p:GetSubPlayer() end)
  if ok and sub then body(sub, 3) end
end
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e.Type ~= 1 then
    out.entities[#out.entities + 1] = { e.Type, e.Variant, e.SubType, X(e.Position.X), X(e.Position.Y),
      room:GetGridIndex(e.Position), e.InitSeed }
  end
end
for slot = 0, 7 do
  local d = room:GetDoor(slot)
  if d then
    out.doors[#out.doors + 1] = { slot = slot, x = X(d.Position.X), y = X(d.Position.Y), door_gi = d:GetGridIndex(),
      gi = room:GetGridIndex(d.Position), open = d:IsOpen() and 1 or 0 }
  end
end
return json.encode(out)
"""
# one cell inside door slot `slot` (0 left, 1 up, 2 right, 3 down; the wide rooms' 4-7 by their side)
DOOR_START = """
local room = Game():GetRoom()
local p = Isaac.GetPlayer(0)
local inward = { [0] = { 40, 0 }, [1] = { 0, 40 }, [2] = { -40, 0 }, [3] = { 0, -40 }, [4] = { 40, 0 }, [5] = { 0, 40 },
  [6] = { -40, 0 }, [7] = { 0, -40 } }
local slot = %d
local d = room:GetDoor(slot)
if room:IsDoorSlotAllowed(slot) and d then
  if not d:IsOpen() then pcall(function() d:Open() end) end
  local q = room:GetDoorSlotPosition(slot)
  p.Position = Vector(q.X + inward[slot][1], q.Y + inward[slot][2]); p.Velocity = Vector(0, 0)
  return 'ok'
end
return 'none'
"""
INTO = {0: 7, 1: 1, 2: 3, 3: 5, 4: 7, 5: 1, 6: 3, 7: 5}   # the move code toward each door slot
HOLD = (1, 3, 5, 7, 2, 4, 6, 8)   # up, right, down, left, the diagonals
HOLD_FOR = 10                     # decisions per held direction (40 frames: across a 1x1 room, into its wall)


def set_mode(c, mode):
    c._send({"cmd": "visible_v3", "mode": mode})
    ack = c._recv()
    assert ack.get('type') == 'ok' and ack.get('mode') == mode, ack
    return ack


def exact(eng):
    """The %.17g strings back to numbers (doubles, exact)."""
    eng['g0'] = [float(v) for v in eng['g0']]
    for b in eng['bodies']:
        b['x'], b['y'] = float(b['x']), float(b['y'])
    for e in eng['entities']:
        e[3], e[4] = float(e[3]), float(e[4])
    for d in eng['doors']:
        d['x'], d['y'] = float(d['x']), float(d['y'])
    return eng


def cell(x, y, gw, gh):
    """The formula cell of a canvas-frame position: (col, row, index), None outside the grid ("no cell")."""
    c, r = math.floor(x / V3.CELL), math.floor(y / V3.CELL)
    return [c, r, r * gw + c] if 0 <= c < gw and 0 <= r < gh else None


def align(frame, eng):
    """The engine values per P / E row of the frame (see the module text). Returns (bodies, entities, doors, worst)."""
    ox, oy = eng['g0'][0] - 20, eng['g0'][1] - 20
    gw, gh = eng['gw'], eng['gh']
    f32 = lambda v: float(np.float32(v))
    pc = lambda n: V3.col(V3.P_COLS, n)
    worst = dict(body=0.0, entity=0.0, unmatched=0, body_count=0)
    bodies = []
    if len(frame['P']) != len(eng['bodies']):
        worst['body_count'] = 1
    for row, b in zip(frame['P'], eng['bodies']):
        x, y = f32(b['x'] - ox), f32(b['y'] - oy)
        known = int(row[pc('position_known')])
        if known:
            worst['body'] = max(worst['body'], abs(float(row[pc('x')]) - x), abs(float(row[pc('y')]) - y))
        bodies.append(dict(seed=b['seed'], role=b['role'], ptype=b['ptype'], in_room=b['in_room'],
                           position_known=known, world=[b['x'], b['y']], canvas=[x, y], engine_gi=b['gi'],
                           cell=cell(x, y, gw, gh) if known else None,
                           frame_xy=[float(row[pc('x')]), float(row[pc('y')])]))
    pool = list(eng['entities'])
    used = set()
    ents = []
    for row in frame['E']:
        t, v, s, fx, fy = int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4])
        best, bd = None, None
        for k, e in enumerate(pool):
            if k in used or e[0] != t or e[1] != v or e[2] != s:
                continue
            d = max(abs(f32(e[3] - ox) - fx), abs(f32(e[4] - oy) - fy))
            if bd is None or d < bd:
                best, bd = k, d
        if best is None:
            worst['unmatched'] += 1
            ents.append(dict(type=t, variant=v, subtype=s, frame_xy=[fx, fy], matched=False))
            continue
        used.add(best)
        e = pool[best]
        worst['entity'] = max(worst['entity'], bd)
        x, y = f32(e[3] - ox), f32(e[4] - oy)
        ents.append(dict(type=t, variant=v, subtype=s, seed=e[6], world=[e[3], e[4]], canvas=[x, y], engine_gi=e[5],
                         cell=cell(x, y, gw, gh), frame_xy=[fx, fy], match_error=bd, matched=True))
    doors = []
    for d in eng['doors']:
        x, y = f32(d['x'] - ox), f32(d['y'] - oy)
        doors.append(dict(slot=d['slot'], world=[d['x'], d['y']], canvas=[x, y], door_gi=d['door_gi'],
                          engine_gi=d['gi'], cell=cell(x, y, gw, gh), open=d['open']))
    return bodies, ents, doors, worst


FORMAT = """# visible-v3 alignment dataset (abplus_probe_v3_align.py)

- `frames.bin`: the native v3 frames (ISAAC_RL_VISIBLE_V3=2, abp_native_v3) back to back, little-endian float32,
  exactly as the bridge sent them (flag 64). Stall columns P 40 / 53 are 0 (the worker backfills them in training).
  Read one with `visible_producer.unpack_frame_v3(frames.bin[offset:offset + length])`.
- `index.jsonl`: one JSON object per frame, in the order of `frames.bin`:
  - `i`, `offset`, `length` (bytes); `seed`, `stage`, `room` (the room entered), `here` (the room shown),
    `shape` (RoomShape 1-12), `shape_name`, `room_type`, `t` (decision in the room), `frame` (game frame count),
    `door_walk` (true from the decision the player was put inside a door until the room's end)
  - `origin` O = GetGridPosition(0) - (20, 20) (world px), `g0` GetGridPosition(0), `gw`, `gh`
  - `bodies`: one per P row, same order. `seed` (InitSeed), `role` (1 player, 3 sub-player), `ptype`, `in_room`,
    `position_known` (the frame's P col 54), `world` (engine Position), `canvas` (= float32(world - O)), `engine_gi`
    (room:GetGridIndex(Position)), `cell` ([col, row, row * gw + col] = floor(canvas / 40), null = no cell: outside
    the grid or no position), `frame_xy` (the frame's P cols 0 / 1)
  - `entities`: one per E row, same order. `type`, `variant`, `subtype`, `frame_xy` (E cols 3 / 4), `matched`; when
    matched: `seed` (InitSeed), `world`, `canvas`, `engine_gi`, `cell` (null = no cell), `match_error` (max abs px
    between the frame's x / y and float32(world - O); 0 expected)
  - (positions: `world` comes from the engine as %.17g text, exact; `canvas` is float32(world - O), the frame's
    arithmetic)
  - `doors`: every DoorSlot with a door: `slot`, `world` (door Position), `canvas`, `door_gi` (the door's own
    GetGridIndex()), `engine_gi` (room:GetGridIndex(Position)), `cell`, `open`
- `summary.json`: frames per shape / room type, per shape the frames with the player in an edge cell (row / column
  0 or last: a wall or door-frame cell), in a wall-side cell (row / column 1 or last - 1; in L rooms the inner
  corner's walls are not counted), with an enemy (an NPC type 10-999), on a door walk, the worst body / entity match error, unmatched rows, fallbacks.
- Engine GetGridIndex: truncation toward 0 per axis, -1 outside the grid except a 40 px band left / top (V1); the
  formula cell is the contract (C2); both are given so the two can be compared.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--seeds', default='2148400000:200')
    ap.add_argument('--stages', type=int, default=8)
    ap.add_argument('--per-shape', type=int, default=220)
    ap.add_argument('--per-room', type=int, default=50)
    ap.add_argument('--door-at', type=int, default=38)
    ap.add_argument('--port', type=int, default=44580)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'FORMAT.md').write_text(FORMAT)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    inst = Instance('v3align', args.port, gx, spec)
    counts = {s: 0 for s in SHAPES}
    types, errors = {}, []
    worst = dict(body=0.0, entity=0.0, unmatched=0, body_count=0, stall_nonzero=0, order=0)
    where = {s: dict(wall_side=0, edge=0, enemies=0, door_walk=0) for s in SHAPES}
    fallbacks, rooms_done, n = 0, 0, 0
    first, count = (int(v) for v in args.seeds.split(':'))
    t0 = time.perf_counter()
    fb = open(out / 'frames.bin', 'wb')
    fi = open(out / 'index.jsonl', 'w')
    pc = lambda name: V3.col(V3.P_COLS, name)
    try:
        for seed in range(first, first + count):
            if min(counts.values()) >= args.per_shape:
                break
            stage = 1 + (seed - first) % args.stages
            try:
                b = inst.env.bridge
                b.reset_mode, b.character = 'floor', 0
                inst.reset(seed)
                b = inst.env.bridge
                if stage > 1:
                    b.reset(phases=[[f'stage {stage}']], settle=2)
                rooms = [tuple(int(x) for x in row.split(',')[:4]) for row in b.lua(FLOOR_LUA).split(';')[2:]]
            except Exception as e:   # reported; the instance relaunched
                errors.append(dict(seed=seed, error=f'{type(e).__name__}: {e}'))
                inst.relaunch(hard=True)
                continue
            for idx, rtype, _var, shape in rooms:
                if shape not in SHAPES or counts[shape] >= args.per_shape:
                    continue
                try:
                    c = b.fork(tag='v3align', lean=True, alarm=1800)
                except Exception as e:
                    errors.append(dict(seed=seed, room=idx, error=f'fork: {type(e).__name__}: {e}'))
                    inst.relaunch(hard=True)
                    break
                try:
                    set_mode(c, 2)
                    dec = LeanDecoder()
                    c.lua(ENTER_LUA % idx)
                    c._sock.sendall(b'S 4 0 0 0 0\n')   # (one step: the room drawn)
                    payload = read_raw(c)
                    door_walk, into = False, 0
                    for t in range(args.per_room):
                        o = dec.decode(payload)
                        if o.visible_v3_frame is None:
                            fallbacks += 1
                            break
                        frame = bytes(o.visible_v3_frame)
                        eng = exact(json.loads(c.lua(ENGINE_LUA)))
                        f = VP.unpack_frame_v3(frame)
                        bodies, ents, doors, w = align(f, eng)
                        if VP.v3_body_order(f):
                            worst['order'] += 1
                        worst['stall_nonzero'] += int(np.any(f['P'][:, [pc('stall_share'), pc('stall_known')]] != 0))
                        worst['body'] = max(worst['body'], w['body'])
                        worst['entity'] = max(worst['entity'], w['entity'])
                        worst['unmatched'] += w['unmatched']
                        worst['body_count'] += w['body_count']
                        sh = int(eng['shape'])
                        rec = dict(i=n, offset=fb.tell(), length=len(frame), seed=seed, stage=stage, room=idx,
                                   here=eng['here'], shape=sh, shape_name=SHAPES.get(sh, str(sh)),
                                   room_type=eng['room_type'], t=t, frame=eng['frame'], door_walk=door_walk,
                                   origin=[eng['g0'][0] - 20, eng['g0'][1] - 20], g0=eng['g0'], gw=eng['gw'],
                                   gh=eng['gh'], bodies=bodies, entities=ents, doors=doors)
                        fb.write(frame)
                        fi.write(json.dumps(rec) + '\n')
                        n += 1
                        if sh in counts:
                            counts[sh] += 1
                            pc0 = bodies[0]['cell'] if bodies else None
                            if pc0:
                                gw, gh = eng['gw'], eng['gh']
                                where[sh]['edge'] += pc0[0] in (0, gw - 1) or pc0[1] in (0, gh - 1)
                                where[sh]['wall_side'] += pc0[0] in (1, gw - 2) or pc0[1] in (1, gh - 2)
                            where[sh]['enemies'] += any(10 <= e['type'] < 1000 for e in ents)
                            where[sh]['door_walk'] += door_walk
                        types[str(eng['room_type'])] = types.get(str(eng['room_type']), 0) + 1
                        if t == args.door_at:   # put inside a door of the room, then walk into its frame
                            for slot in sorted(range(8), key=lambda s: (s * 5 + seed + t) % 8):
                                if c.lua(DOOR_START % slot) == 'ok':
                                    door_walk, into = True, INTO[slot]
                                    break
                        move = into if door_walk else HOLD[(t // HOLD_FOR + seed) % len(HOLD)]
                        shoot = (t // 3 + seed) % 5
                        c._sock.sendall(b'S 4 %d %d %d 0\n' % (move, shoot, int(t % 37 == 20)))
                        payload = read_raw(c)
                    rooms_done += 1
                except Exception as e:
                    errors.append(dict(seed=seed, room=idx, error=f'{type(e).__name__}: {e}'))
                finally:
                    try:
                        c.close()
                    except Exception:
                        pass
                if min(counts.values()) >= args.per_shape:
                    break
            fb.flush()
            fi.flush()
            (out / 'PROGRESS').write_text(json.dumps(dict(seed=seed, frames=n, counts=counts)) + '\n')
    finally:
        fb.close()
        fi.close()
        inst.kill()
    summary = dict(frames=n, bytes=os.path.getsize(out / 'frames.bin'), per_shape={SHAPES[s]: v for s, v in counts.items()},
                   room_types=types, rooms=rooms_done,
                   player_cells={SHAPES[s]: v for s, v in where.items()}, fallbacks=fallbacks, worst=worst, errors=errors[:50],
                   n_errors=len(errors), seconds=round(time.perf_counter() - t0, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True))
    print('SUMMARY', json.dumps(summary, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
