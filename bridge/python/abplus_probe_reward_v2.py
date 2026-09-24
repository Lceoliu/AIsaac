"""combat-v2 on the live engine: bridge counters, reward invariants and component scale per task.

Plays mixture rooms and the Monstro arena through AbplusTransformerEnv with a naive policy (shoot
along the dominant axis at the nearest vulnerable enemy, step toward alignment, some random moves,
rare bombs), every other episode from 3 half hearts, and computes combat-v2 as the training worker
does (isaac_bridge/abplus_reward.py). Checks:
  * blocking_hp / blocking_points of the reset observation equal a separate Lua enumeration of the
    NPCs with CanShutDoors (HP sum, sum of ceil(5 * MaxHP^0.2));
  * the progress total of every finished episode equals blocking_hp(start) / 70;
  * without a death, the hurt total equals -(V(h_start) - V(h_end)) (all health lost was counted).
Reports per task: outcomes, clear bonus and mean components.
usage: python abplus_probe_reward_v2.py <mixture.json> [episodes per task] [port]
"""
import collections
import json
import sys

import numpy as np

from isaac_bridge.abplus import AbplusTransformerEnv, launch_abplus, stop_abplus
from isaac_bridge.abplus_reward import COMPONENTS, PROGRESS_HP_PER_UNIT, CombatV2, health_units, health_value
from isaac_bridge.abplus_tasks import TaskSampler

spec = json.load(open(sys.argv[1]))
per_task = int(sys.argv[2]) if len(sys.argv) > 2 else 8
port = int(sys.argv[3]) if len(sys.argv) > 3 else 27390
BLOCKING_LUA = """local hp, pts = 0, 0
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e:ToNPC() and e:CanShutDoors() and not e:IsDead() then
    hp = hp + math.max(0, e.HitPoints); pts = pts + math.ceil(5 * math.max(0, e.MaxHitPoints) ^ 0.2)
  end
end
return string.format('%.6f,%d', hp, pts)"""


class Env(AbplusTransformerEnv):
    def encode_observation(self, obs):
        return self.history.encode(obs)


def naive(raw, rng, state):
    px, py = raw['players'][0]['pos']
    targets = [e for e in raw['entities'] if e.get('enemy') and e.get('vulnerable') and e['type'] != 33]
    shoot = move = 0
    if targets:
        t = min(targets, key=lambda e: (e['pos'][0] - px) ** 2 + (e['pos'][1] - py) ** 2)
        dx, dy = t['pos'][0] - px, t['pos'][1] - py
        if abs(dx) >= abs(dy):
            shoot = 2 if dx > 0 else 4
            move = (5 if dy > 0 else 1) if abs(dy) > 12 else 0
            if abs(dx) < 70:
                move = 7 if dx > 0 else 3
        else:
            shoot = 3 if dy > 0 else 1
            move = (3 if dx > 0 else 7) if abs(dx) > 12 else 0
            if abs(dy) < 70:
                move = 1 if dy > 0 else 5
    if state['k'] > 0 or rng.random() < 0.15:
        if state['k'] <= 0:
            state['move'], state['k'] = int(rng.integers(9)), int(rng.integers(3, 8))
        state['k'] -= 1
        move = state['move']
    return move * 5 + shoot


def episode(env, seed, low_hp, rng):
    options = {'arena_seed': seed, 'start': (3, 1.0)} if low_hp else {'arena_seed': seed}
    _, info = env.reset(options=options)
    task = info.get('task', 'arena')
    raw = env.raw_obs
    hp, pts = (float(x) for x in env.bridge.lua(BLOCKING_LUA).split(','))
    counters_ok = (abs(raw['combat']['blocking_hp'] - hp) < 1e-3 and int(raw['combat']['blocking_points']) == int(pts))
    reward = CombatV2()
    reward.reset(raw, task)
    h_start = health_units(raw['players'][0])
    state, frames, outcome = {'k': 0, 'move': 0}, 0, 'running'
    while True:
        mask = env.action_masks()
        bomb = int(mask[46] and rng.random() < 0.01)
        _, _, term, trunc, info = env.step(np.array([naive(env.raw_obs, rng, state), bomb, 0]))
        reward.step(env.raw_obs, info['outcome'], info['elapsed_frames'] - frames)
        frames, outcome = info['elapsed_frames'], info['outcome']
        if term or trunc:
            break
    h_end = health_units(env.raw_obs['players'][0])
    t = reward.totals
    progress_ok = abs(t['progress'] - reward.start['blocking_hp'] / PROGRESS_HP_PER_UNIT) < 1e-6
    hurt_expected = None if outcome == 'death' else -(health_value(h_start) - health_value(h_end))
    hurt_ok = hurt_expected is None or abs(t['hurt'] - hurt_expected) < 1e-6
    return dict(seed=seed, task=task, outcome=outcome, seconds=round(frames / 30, 1), h_start=h_start, h_end=h_end,
                start=reward.start, totals={k: round(v, 4) for k, v in t.items()}, total=round(sum(t.values()), 4),
                counters_ok=counters_ok, lua_blocking=[hp, pts], progress_ok=progress_ok, hurt_ok=hurt_ok)


proc = launch_abplus('rv2', port, 'exact', nice=19)
env = Env(port=port, max_episode_frames=3600, deadline=True)
env.bridge.binary_obs = True
rows = []
try:
    rng = np.random.default_rng(7)
    for kind in ('arena', 'normal', 'boss'):
        env.bridge.tasks = None if kind == 'arena' else TaskSampler({kind: 1.0}, spec['normal'], spec['boss'])
        for i in range(per_task):
            row = episode(env, 6000 + 97 * i + {'arena': 0, 'normal': 1, 'boss': 2}[kind], i % 2 == 1, rng)
            row['task'] = kind
            rows.append(row)
            print(json.dumps(row), flush=True)
finally:
    try:
        env.close()
    finally:
        stop_abplus(proc, 'rv2')

summary = {}
for kind in ('arena', 'normal', 'boss'):
    mine = [r for r in rows if r['task'] == kind]
    summary[kind] = dict(
        episodes=len(mine), outcomes=collections.Counter(r['outcome'] for r in mine),
        clear_bonus=[round(min(r['start']['clear_bonus'] for r in mine), 3), round(max(r['start']['clear_bonus'] for r in mine), 3)],
        mean_components={k: round(float(np.mean([r['totals'][k] for r in mine])), 3) for k in COMPONENTS},
        mean_total=round(float(np.mean([r['total'] for r in mine])), 3))
checks = {k: sum(not r[k] for r in rows) for k in ('counters_ok', 'progress_ok', 'hurt_ok')}
print(json.dumps({'summary': summary, 'check_failures': checks, 'episodes': len(rows)}, indent=1))
