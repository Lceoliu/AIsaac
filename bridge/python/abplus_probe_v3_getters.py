"""visible-v3 getter probe (2026-11-12, package C; DESIGN.md rep-compat-schema-20261112 sections 2.2-2.7): the AB+ values
behind the design's UNVERIFIED items, read on the running game, plus real lean payloads for the golden fixtures (F3).

Static analysis of isaac.x64 (sha256 b9bfef32..., LuaEngine::register_classes) already found the getters exist:
Level:GetStageType (Level +4), Game().Difficulty (Game::GetDifficulty, +0x296878), Entity_Player:IsBoneHeart(i) (bone
count +0x38f8 > 0 and bit i of the mask +0x38fc), Entity_Player:GetSubPlayer (GetSecondaryPlayer, +0x3908),
Entity:GetEntityFlags (+0x540), Entity_Bomb.Flags (+0x758), Entity_Knife.TearFlags (+0x740), GridEntity:GetVariant (+0xc)
and .State (+0x10) on doors; no Sprite:GetAnimation (only IsPlaying / IsFinished by name, GetDefaultAnimationName).
This probe reads what they return:

  per character (--chars) and seed: a floor reset as that PlayerType, a lean clone of the start, then
    getters()    stage, stage type, difficulty, curses; per player and its sub-player: type, index, controller, hearts of
                 every kind, IsBoneHeart(0..15) as bits, the black-heart mask; per door slot: variant, State,
                 TargetRoomType, open / locked
    the payload  the lean observation (raw bytes, with the v2 visible block) -> <out>/f3/<char>-<seed>-start.lean
  then an equipped variant (Schoolbag 534 + The Battery's active item, a card and a pill): getters + payload (-equip);
  for the Forgotten (16): one drop press (JSON step, drop 1): getters + payload (-drop), the switch seen in PlayerType
  and the sub-player; a bomb placed: its Flags word, and GetEntityFlags of every NPC in the room.

Read only on the game state except the equipping and the drop press (clones; the template untouched). Writes
<out>/getters.json (every record), <out>/summary.json and the payloads with a meta line each (<out>/f3/index.jsonl).

usage (from the bridge's python dir, PYTHONPATH=.; the game's instances as the other probes):
  python abplus_probe_v3_getters.py --groups-file ../abplus/catalog/scaling2_groups.json --seeds 2147600000:3 \
      --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

from abplus_probe_native_obs import read_raw
from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.tok_floor import CHARACTER_NAMES

GETTERS_LUA = r"""
local json = require("json")
local function try(f, ...) local ok, v = pcall(f, ...) if ok then return v end return "ERR:" .. tostring(v) end
local game, level, room = Game(), Game():GetLevel(), Game():GetRoom()
local out = { stage = try(function() return level:GetStage() end),
  stage_type = try(function() return level:GetStageType() end),
  difficulty = try(function() return game.Difficulty end),
  curses = try(function() return level:GetCurses() end),
  room_type = try(function() return room:GetType() end),
  num_players = try(function() return game:GetNumPlayers() end), players = {}, doors = {} }
local function body(p)
  local b = { ptype = try(function() return p:GetPlayerType() end), index = p.Index,
    controller = try(function() return p.ControllerIndex end),
    hearts = try(function() return p:GetHearts() end), max_hearts = try(function() return p:GetMaxHearts() end),
    soul = try(function() return p:GetSoulHearts() end), black_mask = try(function() return p:GetBlackHearts() end),
    bone = try(function() return p:GetBoneHearts() end), eternal = try(function() return p:GetEternalHearts() end),
    golden = try(function() return p:GetGoldenHearts() end),
    x = p.Position.X, y = p.Position.Y, visible = try(function() return p.Visible end),
    active = try(function() return p:GetActiveItem() end),
    secondary = try(function() local s = p.SecondaryActiveItem return { s.Item, s.Charge, s.BatteryCharge } end),
    card = try(function() return p:GetCard(0) end), pill = try(function() return p:GetPill(0) end),
    pockets = try(function() return p:GetMaxPoketItems() end),
    entity_flags = try(function() return p:GetEntityFlags() end) }
  local bits = 0
  for i = 0, 15 do
    local ok, v = pcall(function() return p:IsBoneHeart(i) end)
    if ok and v then bits = bits | (1 << i) end
  end
  b.bone_bits = bits
  return b
end
for i = 0, game:GetNumPlayers() - 1 do
  local p = Isaac.GetPlayer(i)
  local rec = body(p)
  local ok, sub = pcall(function() return p:GetSubPlayer() end)
  if ok and sub then rec.sub = body(sub) elseif not ok then rec.sub = "ERR:" .. tostring(sub) else rec.sub = false end
  out.players[#out.players + 1] = rec
end
for slot = 0, 7 do
  local d = room:GetDoor(slot)
  if d then
    out.doors[#out.doors + 1] = { slot = slot, variant = try(function() return d:GetVariant() end),
      state = try(function() return d.State end), target_type = try(function() return d.TargetRoomType end),
      current_type = try(function() return d.CurrentRoomType end),
      open = try(function() return d:IsOpen() end), locked = try(function() return d:IsLocked() end),
      target_index = try(function() return d.TargetRoomIndex end) }
  end
end
local ents = {}
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e.Type == 4 then
    ents[#ents + 1] = { type = 4, variant = e.Variant, flags = try(function() return e:ToBomb().Flags end),
      entity_flags = try(function() return e:GetEntityFlags() end) }
  elseif e.Type >= 10 and e.Type < 1000 then
    ents[#ents + 1] = { type = e.Type, variant = e.Variant, entity_flags = try(function() return e:GetEntityFlags() end) }
  end
end
out.entities = ents
return json.encode(out)
"""
EQUIP_LUA = ("local p = Isaac.GetPlayer(0) p:AddCollectible(534, 0, false) p:AddCollectible(63, 0, false) "
             "p:AddCollectible(45, 0, false) p:AddCard(1) p:AddPill(3) return 'ok'")
BOMB_LUA = "local p = Isaac.GetPlayer(0) Isaac.Spawn(4, 0, 0, p.Position, Vector(0, 0), p) return 'ok'"
DROP_STEP = b'{"cmd":"step","repeat":4,"move":0,"shoot":0,"bomb":0,"item":0,"pill":0,"drop":1}\n'
IDLE_STEP = b'S 4 0 0 0 0\n'


def getters(c):
    text = c.lua(GETTERS_LUA)
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return dict(error=str(text)[:400])


def save(out, name, payload, meta, index):
    (out / 'f3' / f'{name}.lean').write_bytes(payload)
    index.write(json.dumps(dict(name=name, bytes=len(payload), **meta), default=str) + '\n')
    index.flush()


def obs(c):
    c._send({"cmd": "obs"})
    return read_raw(c)


def one(inst, seed, char, out, index):
    b = inst.env.bridge
    b.reset_mode = 'floor'
    b.character = char
    _, info = inst.reset(seed)
    rec = dict(char=char, name=CHARACTER_NAMES.get(char), seed=seed, curses=info.get('floor_curses'))
    S = b.fork(tag='template', alarm=0)
    try:
        c = S.fork(lean=True, alarm=600)
        try:
            payload = obs(c)
            rec['start'] = getters(c)
            save(out, f'{char}-{seed}-start', payload, dict(char=char, seed=seed, kind='start'), index)
            if c.lua(EQUIP_LUA) == 'ok':
                c._sock.sendall(IDLE_STEP)
                payload = read_raw(c)
                rec['equip'] = getters(c)
                save(out, f'{char}-{seed}-equip', payload, dict(char=char, seed=seed, kind='equip'), index)
            if char == 16:   # the Forgotten: one drop press switches to the Soul (and back on the next)
                c._sock.sendall(DROP_STEP)
                read_raw(c)
                for _ in range(3):
                    c._sock.sendall(IDLE_STEP)
                    payload = read_raw(c)
                rec['drop'] = getters(c)
                save(out, f'{char}-{seed}-drop', payload, dict(char=char, seed=seed, kind='drop'), index)
            if c.lua(BOMB_LUA) == 'ok':
                c._sock.sendall(IDLE_STEP)
                payload = read_raw(c)
                rec['bomb'] = getters(c)
                save(out, f'{char}-{seed}-bomb', payload, dict(char=char, seed=seed, kind='bomb'), index)
        finally:
            c.close()
    finally:
        S.close()
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147600000:3')
    p.add_argument('--chars', default='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17')
    p.add_argument('--fpd', type=int, default=4)
    p.add_argument('--port', type=int, default=44470)
    p.add_argument('--name', default='v3probe')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    (out / 'f3').mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=args.fpd, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    first, count = (int(v) for v in args.seeds.split(':'))
    chars = [int(v) for v in args.chars.split(',') if v.strip()]
    inst = Instance(args.name, args.port, gx, spec)
    report = []
    t0 = time.perf_counter()
    with open(out / 'f3' / 'index.jsonl', 'w') as index:
        try:
            for seed in range(first, first + count):
                for char in chars:
                    try:
                        rec = one(inst, seed, char, out, index)
                    except Exception as e:   # reported; the instance relaunched
                        rec = dict(char=char, seed=seed, error=f'{type(e).__name__}: {e}')
                        inst.relaunch(hard=True)
                    report.append(rec)
                    print(json.dumps(rec, default=str)[:2000], flush=True)
        finally:
            inst.kill()
    ok = [r for r in report if 'error' not in r]
    errs = sorted({k for r in ok for part in ('start', 'equip', 'drop', 'bomb') if isinstance(r.get(part), dict)
                   for k, v in r[part].items() if isinstance(v, str) and v.startswith('ERR:')})
    summary = dict(records=len(report), errors=len(report) - len(ok), getter_errors=errs,
                   stage_types=sorted({r['start'].get('stage_type') for r in ok if isinstance(r.get('start'), dict)},
                                      key=str),
                   difficulties=sorted({r['start'].get('difficulty') for r in ok if isinstance(r.get('start'), dict)},
                                       key=str),
                   sub_players={str(r['char']): [pl.get('sub') not in (False, None) for pl in
                                                 r['start'].get('players', [])] for r in ok
                                if isinstance(r.get('start'), dict)},
                   seconds=round(time.perf_counter() - t0, 1))
    (out / 'getters.json').write_text(json.dumps(report, indent=1, default=str))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print('SUMMARY', json.dumps(summary, default=str), flush=True)


if __name__ == '__main__':
    main()
