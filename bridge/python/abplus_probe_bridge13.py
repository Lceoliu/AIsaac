"""Live check of bridge abp-0.2.13 (goal-conditioned M0): binary and JSON observations agree (validate mode, nav record
included); an undiscovered secret door is not in the observation and appears once a bomb opens it; a step with a goal
ends at the frame the player gets within the radius; a 4-frame step that walks through a door ends at the new room's
first logic frame; a play batch stops at a room change. Each test starts from a fresh `restart 0` (A7 hooks) with the
Lua transition into room A (abplus_probe_chain), A cleared by killing its enemies.

usage: python abplus_probe_bridge13.py [seed_with_secret_A] [port]   (defaults 2147600218, 27870)
Run on the Linux host only (one extra engine instance at nice 19); ABP_BRIDGE_LUA selects the bridge copy under test.
"""
import json
import math
import sys

from abplus_probe_chain import SLOTS, UNIT, clear_room, fresh_run, pick_chain, secret_neighbours
from isaac_bridge.abplus import AbplusTrainingEnv, BRIDGE_VERSION, action_code, launch_abplus, stop_abplus

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2147600218
port = int(sys.argv[2]) if len(sys.argv) > 2 else 27870
checks = {}


def check(name, ok, detail=''):
    checks[name] = bool(ok)
    print(('PASS' if ok else 'FAIL'), name, detail, flush=True)


def door_pos(env, slot):
    pos = env.lua(f"local d = Game():GetRoom():GetDoor({slot}); return d and (tostring(d.Position.X) .. ',' .. "
                  f"tostring(d.Position.Y)) or 'nil'")
    return None if pos == 'nil' else tuple(float(v) for v in pos.split(','))


def place(env, x, y):
    env.lua(f"local p = Isaac.GetPlayer(0); p.Position = Vector({x}, {y}); p.Velocity = Vector(0, 0); return 'ok'")


def move_towards(dx, dy):
    """Move code nearest (dx, dy), screen y down: 0 deg (right) = 3, then clockwise 4 5 6 7 8 1 2."""
    return (3, 4, 5, 6, 7, 8, 1, 2)[round(math.degrees(math.atan2(dy, dx)) / 45) % 8]


def start_in_a(env, run_seed=None):
    """Fresh run, Lua transition into A (not cleared). Returns (obs, floor, (slot_a, a, slot_b, b))."""
    _, _, obs, floor = fresh_run(env, seed if run_seed is None else run_seed)
    slot_a, a, slot_b, b = pick_chain(floor)
    env.lua(f"local l = Game():GetLevel(); l.LeaveDoor = {slot_a}; Game():StartRoomTransition({a}, {slot_a}, 0); return 'ok'")
    for _ in range(6):
        obs, *_ = env.step({}, repeat=1)
        if obs['room']['room_idx'] == a:
            break
    return obs, floor, (slot_a, a, slot_b, b)


def test_secret(env):
    obs, floor, (_, a, _, _) = start_in_a(env)
    check('entered A (Lua transition)', obs['room']['room_idx'] == a, (a, obs['room']['room_idx']))
    secrets = secret_neighbours(floor, a)
    slot = int(next(iter(secrets))) if secrets else None
    variant = env.lua(f"local d = Game():GetRoom():GetDoor({slot}); return d and tostring(d:GetVariant()) or 'nil'") \
        if slot is not None else 'nil'
    check('A has an undiscovered secret door in the engine', variant == '7', (slot, variant))
    check('secret door absent from obs.doors', all(d['slot'] != slot for d in obs['doors']),
          [(d['slot'], d['open'], d.get('target_type')) for d in obs['doors']])
    check('no DOOR_HIDDEN grid cell in obs.grid', not [c for c in obs['grid'] if c[1] == 16 and c[2] == 7])
    check('A cleared', clear_room(env, [])['clear'])
    x, y = door_pos(env, slot)
    dx, dy = UNIT[slot]
    env.lua(f"Isaac.Spawn(4, 0, 0, Vector({x - 30 * dx}, {y - 30 * dy}), Vector(0, 0), nil); return 'ok'")
    seen = None
    for f in range(80):
        obs, *_ = env.step({}, repeat=1)
        mine = [d for d in obs['doors'] if d['slot'] == slot]
        if mine:
            seen = (f, mine[0], [c for c in obs['grid'] if c[1] == 16 and abs(c[5] - x) < 1 and abs(c[6] - y) < 1])
            break
    check('revealed secret door appears (open) with its door cell', seen is not None and seen[1]['open'] and seen[2], seen)


def test_goal(env):
    obs, _, _ = start_in_a(env)
    clear_room(env, [])
    obs, *_ = env.step({}, repeat=1)
    # a straight run of three walkable interior cells: the player on the first, the goal on the third (80 px away)
    walk = {(round(c[1]), round(c[2])) for c in obs['terrain']['cells'] if c[4] and c[5]}
    start = next((c, (c[0] + 2 * dx, c[1] + 2 * dy)) for c in sorted(walk) for dx, dy in ((40, 0), (0, 40), (-40, 0), (0, -40))
                 if (c[0] + dx, c[1] + dy) in walk and (c[0] + 2 * dx, c[1] + 2 * dy) in walk)
    (sx, sy), (gx, gy) = start
    place(env, sx, sy)
    obs, *_ = env.step({}, repeat=1)
    env.goal = (gx, gy, 20.0)
    trace, hit = [], None
    for i in range(60):
        px, py = obs['players'][0]['pos']
        before = obs['logic_frames']
        obs, *_ = env.step({'move': move_towards(gx - px, gy - py)}, repeat=4)
        px, py = obs['players'][0]['pos']
        trace.append((obs['logic_frames'] - before, round(px), round(py), obs['nav']['goal_hit'], round(obs['nav']['goal_min_dist'], 1)))
        if obs['nav']['goal_hit']:
            hit = dict(step=i, frames=trace[-1][0], dist=round(math.hypot(px - gx, py - gy), 2),
                       min=round(obs['nav']['goal_min_dist'], 2))
            break
    env.goal = None
    check('goal step ends at the frame the player is within 20 px', hit is not None and hit['dist'] <= 20 and hit['min'] <= 20,
          hit if hit else dict(goal=(gx, gy), trace=trace[-8:]))
    obs, *_ = env.step({}, repeat=4)
    check('step without a goal: no goal check', obs['nav']['goal_hit'] is False and obs['nav']['goal_min_dist'] == -1, obs['nav'])


def test_cross(env):
    obs, _, (_, a, slot_b, b) = start_in_a(env)
    clear_room(env, [])
    x, y = door_pos(env, slot_b)
    dx, dy = UNIT[slot_b]
    place(env, x - 70 * dx, y - 70 * dy)
    cross = None
    for i in range(30):
        before = obs['logic_frames']
        obs, *_ = env.step({'move': SLOTS[slot_b][1]}, repeat=4)
        if obs['room']['room_idx'] != a:
            cross = dict(step=i, frames=obs['logic_frames'] - before, room=obs['room']['room_idx'], room_frame=obs['room']['frame'],
                         nav=obs['nav'])
            break
    check("crossing step ends at the new room's first frame", cross is not None and cross['room'] == b and cross['room_frame'] == 1
          and cross['nav']['room_changed'] and cross['frames'] <= 4, cross)
    obs, *_ = env.step({}, repeat=4)
    check('next step is a normal 4-frame step', obs['nav']['room_changed'] is False and obs['room']['frame'] == 5,
          (obs['nav'], obs['room']['frame']))


def test_stop_clear(env):
    """stop_clear: a step ends at the frame the room becomes clear. The enemies are killed between steps (splitting ones
    leave children; the room turns clear a few frames after the last death, i.e. inside a later step)."""
    from abplus_probe_chain import KILL
    for candidate in range(2147600100, 2147600140):   # a floor whose room A has enemies (A9: most do)
        _, _, _, floor = fresh_run(env, candidate)
        if pick_chain(floor) is None:
            continue
        obs, _, (_, a, _, _) = start_in_a(env, candidate)
        if not obs['room']['clear']:
            break
    check('A has enemies before the stop_clear test', not obs['room']['clear'], candidate)
    env.stop_clear = True
    ended = None
    for i in range(12):
        env.lua(KILL)
        before = obs['logic_frames']
        obs, *_ = env.step({}, repeat=8)
        if obs['room']['clear']:
            ended = dict(step=i, frames=obs['logic_frames'] - before)
            break
    env.stop_clear = False
    check('stop_clear ends the step at the clear frame', ended is not None and ended['frames'] < 8, ended)


def test_play(env):
    obs, _, (_, a, slot_b, b) = start_in_a(env)
    clear_room(env, [])
    x, y = door_pos(env, slot_b)
    dx, dy = UNIT[slot_b]
    place(env, x - 70 * dx, y - 70 * dy)
    obs, played, stop = env.play([action_code(SLOTS[slot_b][1], 0)] * 30, repeat=4, stop_clear=False)
    check('play stops at a room change', stop == 'room' and obs['room']['room_idx'] == b and obs['room']['frame'] == 1,
          (played, stop, obs['room']['room_idx'], obs['room']['frame']))


proc = launch_abplus('bridge13', port, 'exact')
env = AbplusTrainingEnv(port=port)
env.binary_obs, env.validate_obs = True, True
try:
    hello = env.connect()
    check('version', hello.get('version') == BRIDGE_VERSION == 'abp-0.2.13', hello.get('version'))
    _, _, obs, _ = fresh_run(env, seed)
    check('nav in reset obs', obs.get('nav', {}).get('room_changed') is False, obs.get('nav'))
    for test in (test_secret, test_goal, test_cross, test_stop_clear, test_play):
        try:
            test(env)
        except Exception as exc:
            check(test.__name__, False, f'{type(exc).__name__}: {exc}')
    check('binary == JSON on every frame', env.validation['frames'] > 0 and env.validation['mismatches'] == 0, env.validation)
finally:
    try:
        env.close()
    finally:
        stop_abplus(proc, 'bridge13')
print(json.dumps({'passed': sum(checks.values()), 'failed': [k for k, v in checks.items() if not v]}))
