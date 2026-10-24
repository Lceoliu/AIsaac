"""Live check of COMBAT options that leave their room (combat_multi_room, user decision 2026-09-30, EXPERIMENTS.md A13) on
the real environment and option sequence of the goal line (abplus_eval_options.make_env, OptionSequence), with scripted
actions and Lua help (bombs, kills; invincible player):

  case 'b'     A's door to B blown open, walk into B, kill B's monsters (B turns clear: no win), walk back into A (its
               monsters are back), kill them: win; the plan goes on (GOTO_POSITION in A)
  case 'start' A's door to the start room blown open, walk into the (clear) start room, a few decisions there: no win

Per step it records the outcome, the room, the history window length and the time feature (the option's clock).
usage: python abplus_probe_combat_rooms.py <checkpoint_dir> [port] [first_seed] [chains]   (Linux host only)"""
import json
import sys
from pathlib import Path

import numpy as np

import abplus_eval_options as E
from abplus_probe_chain import KILL, SLOTS, UNIT
from isaac_bridge.abplus import launch_abplus, stop_abplus
from isaac_bridge.abplus_groups import load_groups
from isaac_bridge.abplus_options import OptionSequence

NOOP = np.array([0, 0, 0])


def lua_bomb(env, slot):
    env.bridge.lua(f"local r = Game():GetRoom(); local d = r:GetDoor({slot}); local c = r:GetCenterPos(); "
                   f"local p = Isaac.GetPlayer(0); p.Position = c; p.Velocity = Vector(0, 0); "
                   f"Isaac.Spawn(EntityType.ENTITY_BOMBDROP, 0, 0, d.Position + (c - d.Position):Normalized() * 30, Vector(0, 0), p); "
                   f"return 'ok'")


def bombs_left(env):
    return env.bridge.lua("local n = 0; for _, e in ipairs(Isaac.GetRoomEntities()) do if e.Type == 4 then n = n + 1 end end; "
                          "return tostring(n)")


def door_open(env, slot):
    return env.bridge.lua(f"local d = Game():GetRoom():GetDoor({slot}); return tostring(d ~= nil and d:IsOpen())") == 'true'


class Run:
    def __init__(self, env, seq):
        self.env, self.seq, self.trace, self.done = env, seq, [], None

    def step(self, action=NOOP, note=''):
        obs, _, terminated, truncated, info = self.env.step(action)
        raw = self.env.raw_obs
        out = self.seq.outcome(raw, info['outcome'])
        reward, parts, _ = self.seq.step(raw, out, 0)
        h = self.env.history
        self.trace.append(dict(note=note, outcome=out, room=raw['room']['room_idx'], clear=raw['room']['clear'],
                               window=len(h.frames), time=round(float(h.frames[-1]['time']), 3),
                               reward=round(reward, 4)))
        if terminated or truncated:
            self.done = out
        return out

    def until(self, test, limit, action=NOOP, note=''):
        for _ in range(limit):
            if self.done or test():
                return True
            self.step(action, note)
        return test()

    def walk(self, slot, note):
        d = self.env.bridge.lua(f"local d = Game():GetRoom():GetDoor({slot}); return tostring(d.Position.X) .. ',' .. tostring(d.Position.Y)")
        x, y = (float(v) for v in d.split(','))
        dx, dy = UNIT[slot]
        self.env.bridge.lua(f"local p = Isaac.GetPlayer(0); p.Position = Vector({x - 60 * dx}, {y - 60 * dy}); "
                            "p.Velocity = Vector(0, 0); return 'ok'")
        room = self.env.raw_obs['room']['room_idx']
        return self.until(lambda: self.env.raw_obs['room']['room_idx'] != room, 30, np.array([SLOTS[slot][1] * 5, 0, 0]), note)


def case(env, config, group, seed, kind):
    obs, info = env.reset(options={'arena_seed': int(seed), 'bombs': 1})
    slot_a, a, slot_b, b = info['chain'][:4]
    seq = OptionSequence(dict(group), seed, info, E.rewards_of(config))
    seq.start(env, first=True)
    run = Run(env, seq)
    rec = dict(seed=seed, kind=kind, chain=[slot_a, a, slot_b, b], target_room=env.target_room)
    run.until(lambda: False, 3, note='settle')
    slot = slot_b if kind == 'b' else (slot_a + 2) % 4
    lua_bomb(env, slot)
    run.until(lambda: bombs_left(env) == '0', 40, note='bomb')
    rec['door_open'] = door_open(env, slot)
    if not rec['door_open'] or run.done:
        return dict(rec, result='door closed' if not rec['door_open'] else run.done, trace=run.trace)
    rec['left'] = run.walk(slot, 'walk out')
    if kind == 'start':
        run.until(lambda: False, 6, note='in start room')
        rec['result'] = run.done or 'running'
    else:
        for _ in range(40):   # kill B's monsters: B turns clear, which is no win
            env.bridge.lua(KILL)
            run.step(note='kill B')
            if run.done or (env.raw_obs['room']['clear'] and door_open(env, (slot_b + 2) % 4)):
                break
        rec['b_clear'] = env.raw_obs['room']['clear']
        run.until(lambda: False, 2, note='B cleared')
        if not run.done:
            rec['back'] = run.walk((slot_b + 2) % 4, 'walk back')
            rec['a_monsters_back'] = env.bridge.lua("return tostring(Game():GetRoom():GetAliveEnemiesCount())")
            for _ in range(60):
                env.bridge.lua(KILL)
                run.step(note='kill A')
                if run.done:
                    break
        rec['result'] = run.done or 'running'
        rec['plan_goes_on'] = bool(run.done == 'win' and seq.next('win', env.raw_obs['room']['room_idx']))
        rec['next_task'] = seq.plan[seq.index]['task'] if rec['plan_goes_on'] else None
    rec['totals'] = [round(float(v), 3) for v in seq.totals()] if seq.totals() is not None else None
    rec['trace'] = run.trace
    return rec


def main():
    ck = Path(sys.argv[1])
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 27894
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 2147601000
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    config = json.loads((ck / 'state.json').read_text())['config']
    group = {g['name']: g for g in load_groups('../abplus/catalog/goal_m2_groups.json')}['chain']
    group.pop('goal_strata', None)
    proc = launch_abplus('probecr', port, 'exact')
    env = E.make_env(port, config, group, 'chain')
    env.bridge.invincible = True
    try:
        for seed in range(first, first + count):
            for kind in ('b', 'start'):
                try:
                    print(json.dumps(case(env, config, group, seed, kind)), flush=True)
                except Exception as exc:
                    print(json.dumps(dict(seed=seed, kind=kind, error=repr(exc))), flush=True)
    finally:
        try:
            env.close()
        finally:
            stop_abplus(proc, 'probecr')


if __name__ == '__main__':
    main()
