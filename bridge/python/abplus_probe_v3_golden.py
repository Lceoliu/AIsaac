"""visible-v3 golden frames V2 (SINGLE_TRANSFORMER_DESIGN 9.3 V2, the coordinator 2026-11-17; package C): real AB+
frames as abp_bridge.lua visible_v3_raw() gives them, each with the reference encoder's v3 bytes and the engine's own
world values read apart from the block (the player's and every entity's Position, GetGridPosition(0), GetGridIndex)
for an independent check of the model-unit values.

Cases (every one a clone of a floor template; nothing written back):
  shape-<name>    one room of every RoomShape 1-12 (IH / IV narrow, the four L), entered by ChangeRoom, 10 frames on
  offgrid         the 2x1 room of V1 with the Homunculus part 228.10 below the grid (seed 2147700000, stage 7, room 89)
  doorframe       the player put on an allowed door slot's position of the start room, 2 frames on
  longlived       a coin spawned in the start room, 600 frames on (age 600: 10 after the scale, clamped at 8)
  devilprice      a collectible pedestal with Price -1 (one heart container, a devil deal's code)
  shooting        the player shooting right for 3 decisions (fire_dir set)
  forgotten       the Forgotten (PlayerType 16) at the start: the Soul as a sub-player body

Writes <out>/<name>.raw.json (the block, with its stall extras), <out>/<name>.v3.bin, <out>/<name>.engine.json (the
engine values read apart) and <out>/manifest.json (per frame: shape, counts, sha256 of the bytes, what it covers).

usage (bridge python dir, PYTHONPATH=.): python abplus_probe_v3_golden.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import abplus_probe_v1_rooms as R
from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge import visible_producer as VP
from isaac_bridge.abplus import FLOOR_LUA, FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance

ENGINE_LUA = r"""
local json = require("json")
local room = Game():GetRoom()
local g0 = room:GetGridPosition(0)
local out = { g0 = { g0.X, g0.Y }, gw = room:GetGridWidth(), gh = room:GetGridHeight(), shape = room:GetRoomShape(),
  players = {}, entities = {} }
for i = 0, Game():GetNumPlayers() - 1 do
  local p = Isaac.GetPlayer(i)
  out.players[#out.players + 1] = { index = p.Index, x = p.Position.X, y = p.Position.Y, gi = room:GetGridIndex(p.Position),
    ptype = p:GetPlayerType(), fire_dir = p:GetFireDirection() }
end
for _, e in ipairs(Isaac.GetRoomEntities()) do
  out.entities[#out.entities + 1] = { type = e.Type, variant = e.Variant, subtype = e.SubType, x = e.Position.X,
    y = e.Position.Y, gi = room:GetGridIndex(e.Position), age = e.FrameCount, visible = e.Visible,
    price = (e.Type == 5 and e:ToPickup()) and e:ToPickup().Price or nil }
end
return json.encode(out)
"""
SPAWN_COIN = """
local room = Game():GetRoom()
for i = 0, room:GetGridSize() - 1 do
  local q = room:GetGridPosition(i)
  if room:IsPositionInRoom(q, 0) and room:GetGridCollision(i) == 0 and i > 2 * room:GetGridWidth() then
    Isaac.Spawn(5, 20, 1, q, Vector(0, 0), nil) return tostring(i)
  end
end
return 'none'
"""
SPAWN_DEVIL = """
local room = Game():GetRoom()
local p = Isaac.GetPlayer(0)
local q = room:FindFreePickupSpawnPosition(p.Position + Vector(80, 0), 0, true)
local e = Isaac.Spawn(5, 100, 1, q, Vector(0, 0), nil):ToPickup()
e.Price = -1
return tostring(e.Price)
"""
DOOR_LUA = """
local room = Game():GetRoom()
for slot = 0, 7 do
  if room:IsDoorSlotAllowed(slot) and room:GetDoor(slot) then
    local q = room:GetDoorSlotPosition(slot)
    Isaac.GetPlayer(0).Position = q
    return string.format('%d,%.3f,%.3f', slot, q.X, q.Y)
  end
end
return 'none'
"""
STALL = dict(stall_share=0.5, stall_known=1)


def capture(c, name, out, manifest, covers):
    text = c.lua("return visible_v3_raw()")
    raw = json.loads(text)
    engine = json.loads(c.lua(ENGINE_LUA))
    payload = VP.observation_frame_v3(raw, **STALL)
    (out / f'{name}.raw.json').write_text(json.dumps(dict(raw=raw, **STALL)))
    (out / f'{name}.engine.json').write_text(json.dumps(engine))
    (out / f'{name}.v3.bin').write_bytes(payload)
    f = VP.unpack_frame_v3(payload)
    manifest.append(dict(name=name, covers=covers, shape=raw['shape'], gw=raw['gw'], gh=raw['gh'],
                         origin=raw['origin'], counts=[int(v) for v in f['header'][2:6]], bytes=len(payload),
                         sha256=hashlib.sha256(payload).hexdigest()))
    print(name, manifest[-1]['counts'], raw['origin'], flush=True)


def idle(c, frames):
    for _ in range(frames):
        c.step({}, repeat=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147700000:150')
    p.add_argument('--stages', default='1,2,3,4,5,6,7,8')
    p.add_argument('--port', type=int, default=44500)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    inst = Instance('v3golden', args.port, gx, spec)
    manifest, errors, t0 = [], [], time.perf_counter()
    first, count = (int(v) for v in args.seeds.split(':'))
    stages = [int(v) for v in args.stages.split(',')]
    try:
        b = inst.env.bridge

        def floor(seed, char=0):
            b.reset_mode, b.character = 'floor', char
            inst.reset(seed)

        def case(name, covers, fn, seed=first, char=0):
            try:
                floor(seed, char)
                c = b.fork(tag='v3golden', alarm=900)
                try:
                    fn(c)
                    capture(c, name, out, manifest, covers)
                finally:
                    c.close()
            except Exception as e:   # reported; the instance relaunched
                errors.append(dict(name=name, error=f'{type(e).__name__}: {e}'))
                print('ERROR', name, errors[-1]['error'], flush=True)
                inst.relaunch(hard=True)

        def enter(c, idx):
            c.lua(R.ENTER_LUA % idx)
            for _ in range(120):
                c.step({}, repeat=1)
                if int(c.lua(R.HERE_LUA)) == idx:
                    break
            idle(c, 10)

        case('doorframe', 'the player on a door slot position (the door frame)',
             lambda c: (c.lua(DOOR_LUA), idle(c, 2)))
        case('longlived', 'a coin 600 frames old (age clamp)', lambda c: (c.lua(SPAWN_COIN), idle(c, 600)))
        case('devilprice', 'a pedestal with Price -1', lambda c: (c.lua(SPAWN_DEVIL), idle(c, 2)))
        case('shooting', 'the player shooting right', lambda c: [c.step({'shoot': 2}, repeat=4) for _ in range(3)])
        case('forgotten', 'the Forgotten and the Soul as bodies', lambda c: idle(c, 2), char=16)

        try:   # (the stage commands on the template as the V1 probe ran them: 2, 3, .., 7 in turn)
            floor(first)
            for st in range(2, 8):
                b.reset(phases=[[f'stage {st}']], settle=2)
            c = b.fork(tag='v3golden', alarm=900)
            try:
                enter(c, 89)
                capture(c, 'offgrid', out, manifest, 'V1 room 89 (2x1): the Homunculus part 228.10 below the grid')
            finally:
                c.close()
        except Exception as e:
            errors.append(dict(name='offgrid', error=f'{type(e).__name__}: {e}'))
            inst.relaunch(hard=True)
        # one room per shape
        got = set()
        for seed in range(first, first + count):
            if len(got) == len(R.SHAPES):
                break
            try:
                floor(seed)
            except Exception as e:
                errors.append(dict(name=f'floor {seed}', error=f'{type(e).__name__}: {e}'))
                inst.relaunch(hard=True)
                continue
            for stage in stages:
                if len(got) == len(R.SHAPES):
                    break
                try:
                    if stage != 1:
                        b.reset(phases=[[f'stage {stage}']], settle=2)
                    rooms = [tuple(int(x) for x in row.split(',')) for row in b.lua(FLOOR_LUA).split(';')[2:]]
                except Exception as e:
                    errors.append(dict(name=f'floor {seed} stage {stage}', error=f'{type(e).__name__}: {e}'))
                    break
                for idx, t, var, shape, _ in rooms:
                    if shape in got or shape not in R.SHAPES:
                        continue
                    try:
                        c = b.fork(tag='v3golden', alarm=900)
                        try:
                            enter(c, idx)
                            if int(c.lua(R.HERE_LUA)) != idx:
                                raise RuntimeError('ChangeRoom did not arrive')
                            capture(c, f'shape-{R.SHAPES[shape]}', out, manifest,
                                    f'RoomShape {shape} (seed {seed}, stage {stage}, room {idx}, type {t})')
                            got.add(shape)
                        finally:
                            c.close()
                    except Exception as e:
                        errors.append(dict(name=f'shape {shape} room {idx}', error=f'{type(e).__name__}: {e}'))
    finally:
        inst.kill()
    (out / 'manifest.json').write_text(json.dumps(dict(frames=manifest, errors=errors, stall=STALL,
                                                       seconds=round(time.perf_counter() - t0, 1)), indent=1))
    print('SUMMARY frames', len(manifest), 'errors', len(errors), flush=True)


if __name__ == '__main__':
    main()
