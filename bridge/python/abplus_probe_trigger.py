"""ITEM / PILLCARD / BOMB: what one press does on the bridge as it is (IsActionTriggered answered true in both player
updates of the press's first game frame: AB+ updates the player at 60 Hz, the bridge clears "pressed this frame" at
30 Hz) and with ABP_SET_TRIGGER_ONCE(true) (true for the first query of the press only, as a real key press: one input
poll per player update). The coordinator, 2026-11-18, before deciding which the new model's arms use.

Cases (per seed; the items given, 20 idle decisions for the pickup animation, the press, 10 idle decisions):
  pill1          one pill (pocket slot 0)
  card1          one card (The Lovers, 7: two red hearts spawn)
  poly_pillcard  Polydactyly (454) + a card (7) + a pill: two pocket items, does one press use both?
  poly_cards     Polydactyly + two cards (The Lovers 7, Justice 9)
  item_free      an active item without charges (the first of IV Bag 135, Kamikaze 40, Razor Blade 126 whose
                 MaxCharges is 0): used twice?
  item_full      Yum Heart (45) fully charged, the player hurt by one heart first
  item_battery   The Battery (63) + Yum Heart charged twice over (SetActiveCharge(2 x MaxCharges)): both charges at once?
  item_timed     an active item with a timed charge (MaxCharges > 12 among the actives; none: skipped), fully charged
  bomb           three bombs, one press: placed / spent once or twice?
  bomb_twice     three bombs, the press held on two decisions in a row (the second inside the bomb's cooldown)
Each case: mode double (the bridge as it is) and once (ABP_SET_TRIGGER_ONCE(true) in the clone); path line (a lean
clone with step lines: abp_turbo's native input, the training path) and json (a JSON clone: the bridge's Lua callback,
with a logging MC_INPUT_ACTION callback after it: the IsActionTriggered queries of the case's action and the player
updates of the press, with Isaac.GetFrameCount / Game():GetFrameCount). Recorded before / after: bombs, coins, keys,
hearts, soul hearts, the active item, its charge and battery charge, the pocket slots 0 / 1 (card, pill), the
entities new in the room (type.variant.subtype). -> <out>/trigger.jsonl, summary.json.

usage (bridge python dir, PYTHONPATH=.): python abplus_probe_trigger.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
"""
import argparse
import json
import os
from pathlib import Path

from abplus_probe_native_obs import read_raw
from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance

# (action per case: 'pill' -> PILLCARD, 'item' -> ITEM, 'bomb' -> BOMB)
CASES = {
    'pill1': ('pill', "p:AddPill(3)"),
    'card1': ('pill', "p:AddCard(7)"),
    'poly_pillcard': ('pill', "p:AddCollectible(454, 0, false) p:AddCard(7) p:AddPill(3)"),
    'poly_cards': ('pill', "p:AddCollectible(454, 0, false) p:AddCard(7) p:AddCard(9)"),
    'item_free': ('item', "p:AddCollectible(__FREE__, 0, false)"),
    'item_full': ('item', "p:AddCollectible(45, 0, false) p:TakeDamage(2, 0, EntityRef(p), 0) p:FullCharge()"),
    'item_battery': ('item', "p:AddCollectible(63, 0, false) p:AddCollectible(45, 0, false) "
                             "p:TakeDamage(2, 0, EntityRef(p), 0) p:SetActiveCharge(8)"),
    'item_timed': ('item', "p:AddCollectible(__TIMED__, 0, false) p:FullCharge()"),
    'bomb': ('bomb', "p:AddBombs(2)"),
    'bomb_twice': ('bomb', "p:AddBombs(2)"),
}
ACTION = {'pill': 'ACTION_PILLCARD', 'item': 'ACTION_ITEM', 'bomb': 'ACTION_BOMB'}
PICK_LUA = r"""
local json = require("json")
local cfg = Isaac.GetItemConfig()
local free, timed, maxc = nil, {}, {}
for _, id in ipairs({ 135, 40, 126 }) do
  local c = cfg:GetCollectible(id)
  if c and c.MaxCharges == 0 and free == nil then free = id end
end
for id = 1, 600 do
  local c = cfg:GetCollectible(id)
  if c and c.Type == ItemType.ITEM_ACTIVE then
    if c.MaxCharges > 12 then timed[#timed + 1] = { id, c.MaxCharges, c.Name } end
    maxc[#maxc + 1] = { id, c.MaxCharges }
  end
end
return json.encode({ free = free, timed = timed, n_active = #maxc })
"""
STATE_LUA = r"""
local json = require("json")
local p = Isaac.GetPlayer(0)
local ents = {}
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e.Type ~= 1 then ents[#ents + 1] = { e.InitSeed, e.Type .. "." .. e.Variant .. "." .. e.SubType } end
end
local function try(f) local ok, v = pcall(f) if ok then return v end return "ERR" end
return json.encode({ bombs = p:GetNumBombs(), coins = p:GetNumCoins(), keys = p:GetNumKeys(), hearts = p:GetHearts(),
  soul = p:GetSoulHearts(), active = p:GetActiveItem(), charge = p:GetActiveCharge(),
  battery = try(function() return p:GetBatteryCharge() end), card0 = p:GetCard(0), card1 = p:GetCard(1),
  pill0 = p:GetPill(0), pill1 = p:GetPill(1), frame = Game():GetFrameCount(), ents = ents })
"""
LOGGER_LUA = r"""
ABPT = { press = false, seq = {}, action = ButtonAction.__ACTION__ }
local m = RegisterMod("abpt_trigger", 1)
m:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, ent, hook, action)
  if ABPT.press and action == ABPT.action and hook == InputHook.IS_ACTION_TRIGGERED and #ABPT.seq < 80 then
    ABPT.seq[#ABPT.seq + 1] = string.format("trig(%s) i%d g%d", ent == nil and "nil" or tostring(ent.Type),
      Isaac.GetFrameCount(), Game():GetFrameCount())
  end
  return nil
end)
m:AddCallback(ModCallbacks.MC_POST_PLAYER_UPDATE, function(_, p)
  if ABPT.press and #ABPT.seq < 80 then
    ABPT.seq[#ABPT.seq + 1] = string.format("player_update i%d g%d", Isaac.GetFrameCount(), Game():GetFrameCount())
  end
end)
return "ok"
"""


def step_line(c, b=0, i=0, p=0):
    if p:
        c._sock.sendall(b'S 4 0 0 %d %d %d\n' % (b, i, p))
    else:
        c._sock.sendall(b'S 4 0 0 %d %d\n' % (b, i))
    read_raw(c)


def step_json(c, b=0, i=0, p=0):
    c._send({"cmd": "step", "repeat": 4, "move": 0, "shoot": 0, "bomb": b, "item": i, "pill": p})
    while True:
        msg = c._recv()
        if msg.get('type') == 'error':
            raise RuntimeError(msg.get('msg'))
        if msg.get('type') != 'ok' or 'cmd' not in msg:
            return


def diff(before, after):
    keys = ('bombs', 'coins', 'keys', 'hearts', 'soul', 'active', 'charge', 'battery', 'card0', 'card1', 'pill0',
            'pill1')
    out = {k: [before[k], after[k]] for k in keys if before[k] != after[k]}
    old = {s for s, _ in before['ents']}
    new = {}
    for s, name in after['ents']:
        if s not in old:
            new[name] = new.get(name, 0) + 1
    out['new_entities'] = new
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--seeds', default='2147800000:3')
    ap.add_argument('--cases', default=','.join(CASES))
    ap.add_argument('--port', type=int, default=44640)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    os.environ.pop('ISAAC_RL_TRIGGER_ONCE', None)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    inst = Instance('trig', args.port, gx, spec)
    first, count = (int(v) for v in args.seeds.split(':'))
    rows, picks = [], None
    with open(out / 'trigger.jsonl', 'w') as f:
        try:
            for seed in range(first, first + count):
                for case in args.cases.split(','):
                    kind, setup = CASES[case]
                    b = inst.env.bridge
                    b.reset_mode, b.character = 'floor', 0
                    inst.reset(seed)
                    b = inst.env.bridge
                    if picks is None:
                        picks = json.loads(b.lua(PICK_LUA))
                        f.write(json.dumps(dict(picks=picks)) + '\n')
                    timed = picks['timed'][0][0] if picks['timed'] else None
                    if case == 'item_timed' and timed is None or case == 'item_free' and picks['free'] is None:
                        continue
                    code = setup.replace('__FREE__', str(picks['free'])).replace('__TIMED__', str(timed))
                    assert b.lua(f"local p = Isaac.GetPlayer(0) {code} return 'ok'") == 'ok'
                    S = b.fork(tag='template', alarm=0)
                    try:
                        for mode in ('double', 'once'):
                            for path in ('line', 'json'):
                                c = S.fork(lean=path == 'line', alarm=600)
                                try:
                                    if path == 'line':
                                        c._send({"cmd": "obs"})
                                        read_raw(c)
                                    stp = step_line if path == 'line' else step_json
                                    mask = c.lua(f"return tostring(ABP_SET_TRIGGER_ONCE({'true' if mode == 'once' else 'false'}))")
                                    if path == 'json':
                                        assert c.lua(LOGGER_LUA.replace('__ACTION__', ACTION[kind])) == 'ok'
                                    for _ in range(20):
                                        stp(c)
                                    before = json.loads(c.lua(STATE_LUA))
                                    if path == 'json':
                                        c.lua("ABPT.press = true return 'ok'")
                                    press = dict(b=int(kind == 'bomb'), i=int(kind == 'item'), p=int(kind == 'pill'))
                                    for _ in range(2 if case == 'bomb_twice' else 1):
                                        stp(c, **press)
                                    if path == 'json':
                                        c.lua("ABPT.press = false return 'ok'")
                                    for _ in range(10):
                                        stp(c)
                                    after = json.loads(c.lua(STATE_LUA))
                                    seq = json.loads(c.lua("return require('json').encode(ABPT.seq)")) \
                                        if path == 'json' else None
                                finally:
                                    c.close()
                                row = dict(seed=seed, case=case, mode=mode, path=path, once_mask=mask,
                                           change=diff(before, after), seq=seq,
                                           before={k: v for k, v in before.items() if k != 'ents'})
                                rows.append(row)
                                f.write(json.dumps(row) + '\n')
                                f.flush()
                                print(json.dumps({k: row[k] for k in ('seed', 'case', 'mode', 'path', 'change')}),
                                      flush=True)
                    finally:
                        S.close()
        finally:
            inst.kill()
    summary = {}
    for r in rows:
        summary.setdefault(r['case'], {}).setdefault(f"{r['mode']}/{r['path']}", []).append(r['change'])
    (out / 'summary.json').write_text(json.dumps(dict(picks=picks, cases=summary), indent=1))
    print('SUMMARY', json.dumps(dict(picks=picks)), flush=True)


if __name__ == '__main__':
    main()
