"""Why drop does not swap the Schoolbag or cycle the pockets (drop2, 2026-11-17: the Forgotten switch and the held
trinket drop work, these two not). Hypothesis: the engine asks for ACTION_DROP there without a player entity (an
Input query by controller index), and the bridge answers only queries for a player entity (MC_INPUT_ACTION: entity nil
-> nil; abp_turbo's native input the same), so the game reads its devices: nothing pressed.

Per case (schoolbag: Schoolbag + Yum Heart over the D6; pockets: Polydactyly + a card + a pill; forgotten) and variant,
on a JSON-mode clone (the bridge's Lua callback answers, native input off), a logging MC_INPUT_ACTION callback (a mod
registered at run time, after the bridge's) counts every ACTION_DROP query by (entity type or nil, hook) and frame:
  plain          it only logs (returns nil: the bridge's answers stand)
  nil_triggered  during the press it also answers the queries without an entity: IS_ACTION_TRIGGERED true on the
                 press's first frame, IS_ACTION_PRESSED true, GET_ACTION_VALUE 1.0 while drop is held
  trig_all       during the press IS_ACTION_TRIGGERED true for the player on every frame (not only the first)
  hold<N>        the press held N decisions (4N frames) instead of one
20 idle decisions, one drop press (one decision, repeat 4), 6 idle decisions; the state before / after (active,
secondary active, cards / pills in slots 0 and 1, PlayerType) and the query log -> <out>/drop_input.jsonl.

usage (bridge python dir, PYTHONPATH=.): python abplus_probe_drop_input.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
"""
import argparse
import json
import os
from pathlib import Path

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance

SETUP = {
    'schoolbag': "local p = Isaac.GetPlayer(0) p:AddCollectible(534, 0, false) p:AddCollectible(45, 0, false) return 'ok'",
    'pockets': "local p = Isaac.GetPlayer(0) p:AddCollectible(454, 0, false) p:AddCard(1) p:AddPill(3) return 'ok'",
    'forgotten': None,
}
CHAR = {'forgotten': 16}
STATE_LUA = r"""
local json = require("json")
local p = Isaac.GetPlayer(0)
local function try(f) local ok, v = pcall(f) if ok then return v end return "ERR" end
return json.encode({ ptype = p:GetPlayerType(), active = p:GetActiveItem(),
  secondary = try(function() return p.SecondaryActiveItem.Item end),
  card0 = p:GetCard(0), card1 = p:GetCard(1), pill0 = p:GetPill(0), pill1 = p:GetPill(1),
  frame = Game():GetFrameCount() })
"""
LOGGER_LUA = r"""
ABPX = { log = {}, frames = {}, press = false, mode = "%s", first = nil }
local m = RegisterMod("abpx_drop_input", 1)
m:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, ent, hook, action)
  if action ~= ButtonAction.ACTION_DROP then return nil end
  local who = ent == nil and "nil" or tostring(ent.Type)
  local key = who .. ":" .. tostring(hook) .. (ABPX.press and ":press" or "")
  ABPX.log[key] = (ABPX.log[key] or 0) + 1
  if ABPX.press then ABPX.frames[key .. "@" .. tostring(Game():GetFrameCount())] = true end
  if ABPX.press and ent == nil and ABPX.mode == "nil_triggered" then
    local f = Game():GetFrameCount()
    if hook == InputHook.IS_ACTION_TRIGGERED then
      if ABPX.first == nil then ABPX.first = f end
      return f == ABPX.first
    elseif hook == InputHook.IS_ACTION_PRESSED then return true
    elseif hook == InputHook.GET_ACTION_VALUE then return 1.0 end
  end
  if ABPX.press and ent ~= nil and ABPX.mode == "trig_all" and hook == InputHook.IS_ACTION_TRIGGERED then
    return true
  end
  return nil
end)
return "ok"
"""
LOG_LUA = r"""
local json = require("json")
local fr = {}
for k in pairs(ABPX.frames) do fr[#fr + 1] = k end
table.sort(fr)
return json.encode({ log = ABPX.log, frames = fr, first = ABPX.first })
"""


def step(c, drop=0):
    c._send({"cmd": "step", "repeat": 4, "move": 0, "shoot": 0, "bomb": 0, "item": 0, "pill": 0, "drop": drop})
    while True:
        msg = c._recv()
        if msg.get('type') == 'error':
            raise RuntimeError(msg.get('msg'))
        if msg.get('type') != 'ok' or 'cmd' not in msg:
            return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--seeds', default='2147800000:2')
    ap.add_argument('--cases', default='schoolbag,pockets,forgotten')
    ap.add_argument('--variants', default='plain,nil_triggered')
    ap.add_argument('--port', type=int, default=44620)
    ap.add_argument('--mode', default='exact', help='the instance launch mode (abplus.MODES: exact / skip / render)')
    ap.add_argument('--stubs', default='h', help="the stub list: h (stub_render_h.txt), '' the mode's own")
    ap.add_argument('--fast', default='3', help="ABP_FAST ('' unset)")
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    args.stubs = '' if args.stubs == 'none' else args.stubs   # ('none': an empty value through the runner's $*)
    args.fast = '' if args.fast == 'none' else args.fast
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    for k, v in {'ABP_FAST': args.fast, 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        if v:
            os.environ.setdefault(k, v)
    stub = str(Path(default_preload()).parent / f'stub_render_{args.stubs}.txt') if args.stubs else ''
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0, mode=args.mode,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if stub and Path(stub).is_file() else '')
    inst = Instance('dropin', args.port, gx, spec)
    first, count = (int(v) for v in args.seeds.split(':'))
    rows = []
    with open(out / 'drop_input.jsonl', 'w') as f:
        try:
            for seed in range(first, first + count):
                for case in args.cases.split(','):
                    b = inst.env.bridge
                    b.reset_mode, b.character = 'floor', CHAR.get(case, 0)
                    inst.reset(seed)
                    b = inst.env.bridge
                    if SETUP.get(case):
                        assert b.lua(SETUP[case]) == 'ok'
                    S = b.fork(tag='template', alarm=0)
                    try:
                        for variant in args.variants.split(','):
                            c = S.fork(alarm=600)
                            try:
                                assert c.lua(LOGGER_LUA % variant) == 'ok'
                                for _ in range(20):
                                    step(c)
                                before = json.loads(c.lua(STATE_LUA))
                                c.lua("ABPX.press = true return 'ok'")
                                hold = int(variant[4:]) if variant.startswith('hold') else 1
                                for _ in range(hold):
                                    step(c, drop=1)
                                c.lua("ABPX.press = false return 'ok'")
                                mid = json.loads(c.lua(STATE_LUA))
                                for _ in range(6):
                                    step(c)
                                after = json.loads(c.lua(STATE_LUA))
                                log = json.loads(c.lua(LOG_LUA))
                            finally:
                                c.close()
                            keys = ('ptype', 'active', 'secondary', 'card0', 'card1', 'pill0', 'pill1')
                            row = dict(seed=seed, case=case, variant=variant, before=before, after_press=mid,
                                       after=after, changed=[k for k in keys if before[k] != after[k]], log=log)
                            rows.append(row)
                            f.write(json.dumps(row) + '\n')
                            f.flush()
                            print(json.dumps(dict(seed=seed, case=case, variant=variant, changed=row['changed'],
                                                  log=log['log'])), flush=True)
                    finally:
                        S.close()
        finally:
            inst.kill()
    summary = {f"{r['case']}/{r['variant']}": [] for r in rows}
    for r in rows:
        summary[f"{r['case']}/{r['variant']}"].append(r['changed'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
