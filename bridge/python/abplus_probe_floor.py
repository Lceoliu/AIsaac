"""Floor-runner probe (floor-clear mandate, 2026-10-01): on one AB+ instance, a whole Basement I floor from its start room
(bridge reset_mode 'floor'): the room descriptors, the start room's doors and where they lead, a scripted GOTO_DOOR into a
neighbour, then the boss room by the Lua transition, the boss killed by Lua, the trapdoor, and a scripted walk into it
(does the stage change, does the bridge survive the level transition).

  python abplus_probe_floor.py --checkpoint CKPT [--seed 1001] [--port 27900]
"""
import argparse
import numpy as np
import json
import time
from pathlib import Path

from abplus_eval_options import act, play_option, rewards_of
from isaac_bridge.abplus import AbplusTransformerEnv, launch_abplus, stop_abplus
from isaac_bridge.abplus_options import OptionSequence
from isaac_bridge.abplus_worker import observation_options

DOORS_LUA = """
local room = Game():GetRoom(); local level = Game():GetLevel(); local out = {}
for slot = 0, 7 do
  local d = room:GetDoor(slot)
  if d and d:GetVariant() ~= 7 then
    local t = d.TargetRoomIndex; local desc = level:GetRoomByIdx(t)
    local ty, safe = -1, -999
    pcall(function() ty = desc.Data.Type; safe = desc.SafeGridIndex end)
    out[#out + 1] = table.concat({slot, t, safe, ty, d:IsLocked() and 1 or 0, d:IsOpen() and 1 or 0}, ",")
  end
end
return table.concat(out, ";")
"""

TRAPDOOR_LUA = """
local room = Game():GetRoom(); local out = {}
for i = 0, room:GetGridSize() - 1 do
  local g = room:GetGridEntity(i)
  if g ~= nil and g:GetType() == 17 then
    local p = room:GetGridPosition(i)
    out[#out + 1] = table.concat({i, p.X, p.Y, g.State, g.CollisionClass or -1}, ",")
  end
end
return table.concat(out, ";")
"""


def make_env(port, config):
    env = AbplusTransformerEnv(port=port, max_episode_frames=30 * 600, **observation_options(config['reward_profile']),
                               deadline_s=180.0, frames_per_decision=int(config.get('frames_per_decision', 4)),
                               history=config['history'], entity_capacity=config['entity_capacity'])
    env.bridge.binary_obs = True
    env.combat_multi_room = True
    env.bridge.reset_mode = 'floor'
    return env


class Args:
    scripted_goto = True
    expert_actions = False
    expert_unstick = False
    check_expert = False
    stochastic = False


def say(*parts):
    print(time.strftime('%H:%M:%S'), *parts, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--seed', type=int, default=1001)
    p.add_argument('--port', type=int, default=27900)
    args = p.parse_args()
    config = json.loads((args.checkpoint / 'state.json').read_text())['config']
    proc = launch_abplus('flrprobe', args.port, 'exact')
    env = make_env(args.port, config)
    try:
        obs, info = env.reset(options={'arena_seed': args.seed})
        raw = env.raw_obs
        floor = info['floor']
        types = {}
        for r in floor.values():
            types[r['type']] = types.get(r['type'], 0) + 1
        boss = [i for i, r in floor.items() if r['type'] == 5]
        say('reset', 'start', info['start_room'], 'curses', info['floor_curses'], 'attempts', info['room_attempts'],
            'rooms', len(floor), 'types', types, 'boss', boss, 'stage', raw['room']['stage'],
            'hp', raw['players'][0]['hearts'], 'bombs', raw['players'][0]['bombs'], 'clear', raw['room']['clear'])
        say('obs doors', [(d['slot'], d['open'], d['locked'], d.get('target_type')) for d in raw['doors']])
        doors = [tuple(int(float(x)) for x in row.split(',')) for row in env.bridge.lua(DOORS_LUA).split(';') if row]
        say('lua doors (slot, target, safe, type, locked, open)', doors)
        group = dict(name='floor', mode='combat', goto_seconds=10, goto_seconds_big=15, goal_radius=20.0, seconds=90)
        seq = OptionSequence(group, args.seed, info, rewards_of(config))
        normal = [d for d in doors if d[3] == 1 and not d[4]]
        if normal:
            slot, target = normal[0][0], normal[0][2]
            seq.plan, seq.index = [dict(task='goto_door', source='goto', slot=slot, room=target)], 0
            started = seq.start(env, first=True)
            say('goto_door start', slot, '->', target, 'started', started is not None)
            rec = play_option(env, None, seq, Args, False)
            raw = env.raw_obs
            say('goto_door', rec['outcome'], 'frames', rec['frames'], 'room now', raw['room']['room_idx'], 'clear',
                raw['room']['clear'], 'alive', raw['room']['alive'], 'entities', len(raw['entities']))
            res = env.bridge.lua("for _, e in ipairs(Isaac.GetRoomEntities()) do if e:IsEnemy() then e:Kill() end end "
                                 "return 'ok'")
            for _ in range(30):
                obs, _, _, _, info2 = env.bridge.step({}, repeat=4)
                if obs['room']['clear']:
                    break
            raw = env.bridge.query_obs() if hasattr(env.bridge, 'query_obs') else obs
            say('after kill', res, 'clear', obs['room']['clear'], 'doors', [(d['slot'], d['open']) for d in obs['doors']])
        if boss:
            b = boss[0]
            res = env.bridge.lua(f"local l = Game():GetLevel(); l.LeaveDoor = -1; Game():StartRoomTransition({b}, -1, 0); "
                                 "return 'ok'")
            t0 = time.time()
            for k in range(40):
                obs, _, _, _, info2 = env.bridge.step({}, repeat=4)
                if obs['room']['room_idx'] == b and k > 3:
                    break
            paused = env.bridge.lua("return tostring(Game():IsPaused())")
            say('boss transition', res, 'room', obs['room']['room_idx'], 'type', obs['room']['type'], 'clear',
                obs['room']['clear'], 'alive', obs['room']['alive'], 'paused', paused, 'secs', round(time.time() - t0, 1),
                'bosses', [(e['type'], e['variant']) for e in obs['entities'] if e.get('boss')][:4])
            res = env.bridge.lua("for _, e in ipairs(Isaac.GetRoomEntities()) do if e:IsEnemy() then e:Kill() end end "
                                 "return 'ok'")
            trap = ''
            for k in range(120):
                obs, _, _, _, info2 = env.bridge.step({}, repeat=4)
                trap = env.bridge.lua(TRAPDOOR_LUA)
                if trap and obs['room']['clear']:
                    break
            say('boss killed', res, 'clear', obs['room']['clear'], 'trapdoor (index, x, y, state, coll)', trap,
                'pickups', [(e['type'], e['variant'], e.get('subtype')) for e in obs['entities'] if e['type'] == 5])
            if trap:
                env.raw_obs = obs
                ti, tx, ty = trap.split(';')[0].split(',')[:3]
                cells = obs['terrain']['cells']
                width = obs['terrain']['width']
                cell = next((c for c in cells if c[0] == int(ti)), None)
                say('trapdoor cell record', cell)
                seq.plan, seq.index = [dict(task='goto_position', source='goto', target=(float(tx), float(ty)))], 0
                started = seq.start(env, room_changed=True)
                say('goto trapdoor start', started is not None, 'field cells', len(seq.option['state']['nav']['field'])
                    if seq.option and seq.option['state'].get('nav') else None)
                if started is not None:
                    from isaac_bridge.abplus_nav import descent_move
                    nav = seq.option['state']['nav']
                    for k in range(40):
                        px, py = env.raw_obs['players'][0]['pos']
                        mv = descent_move(px, py, nav)
                        o2, _, term, trunc, inf = env.step(np.array([mv * 5, 0, 0]))
                        if k % 4 == 0 or term or trunc:
                            say('  step', k, 'move', mv, 'pos', [round(v) for v in env.raw_obs['players'][0]['pos']],
                                'room', env.raw_obs['room']['room_idx'], 'stage', env.raw_obs['room']['stage'],
                                'outcome', inf.get('outcome'), 'term', term, trunc,
                                'controls', env.raw_obs['players'][0].get('controls'))
                        if term or trunc:
                            break
                    rec = dict(outcome='debug', frames=0)
                    say('lua around player', env.bridge.lua(
                        "local room = Game():GetRoom(); local p = Isaac.GetPlayer(0); local out = {'pos=' .. p.Position.X .. ',' .. p.Position.Y}; "
                        "for dy = -2, 0 do local q = p.Position + Vector(0, 40 * dy); local i = room:GetGridIndex(q); local g = room:GetGridEntity(i); "
                        "out[#out + 1] = 'cell' .. i .. ':' .. (g and (g:GetType() .. '/' .. g.State .. '/coll' .. tostring(g.CollisionClass)) or 'none') .. ':coll' .. room:GetGridCollision(i) end; "
                        "for _, e in ipairs(Isaac.GetRoomEntities()) do if e.Position:Distance(p.Position) < 90 and e.Index ~= p.Index then "
                        "out[#out + 1] = e.Type .. '.' .. e.Variant .. '.' .. e.SubType .. '@' .. math.floor(e.Position.X) .. ',' .. math.floor(e.Position.Y) .. ' coll' .. e.EntityCollisionClass .. ' grid' .. e.GridCollisionClass end end; "
                        "return table.concat(out, ' | ')"))
                    say('obs entities', [(e['type'], e['variant'], [round(v) for v in e['pos']]) for e in env.raw_obs['entities']][:12])
                    raw = env.raw_obs
                    say('goto trapdoor', rec['outcome'], 'frames', rec['frames'], 'room', raw['room']['room_idx'], 'stage',
                        raw['room']['stage'], 'type', raw['room']['type'])
                for k in range(60):
                    obs, _, _, _, info2 = env.bridge.step({}, repeat=4)
                    if obs['room']['stage'] != 1:
                        break
                say('after waiting', 'stage', obs['room']['stage'], 'room', obs['room']['room_idx'], 'frame',
                    obs['room']['frame'], 'player', [round(v) for v in obs['players'][0]['pos']])
    finally:
        try:
            env.close()
        finally:
            stop_abplus(proc, 'flrprobe')


if __name__ == '__main__':
    main()
