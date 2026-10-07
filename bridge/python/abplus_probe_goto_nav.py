"""Goal-conditioned line M0, probe P7 (rl/docs/GOAL_CONDITIONED_DESIGN.md): does walking down the d_geo field
(abplus_nav.goal_field / descent_move) reach the sampled GOTO goals in the engine? Also the scripted GOTO reference for the
E-GOTO evaluation (success, decisions, walked px per decision against d_geo).

Per seed: an emptied C39 1x1 normal room (reset_mode goto_room: the room's layout, every entity removed, the player on a
random cell of the largest walkable component), then GOALS goals in a row, each drawn by abplus_nav.sample_goal from the
player's position (the stratum cycles straight / detour / dead_end); the bridge checks the 20 px radius every logic frame
(abp-0.2.13 step goal); 4 frames per decision, at most 60 decisions (8 s) per goal; a room change ends the room.

usage: python abplus_probe_goto_nav.py <out.jsonl> <tasks_file> [rooms] [port] [first_seed]
Run on the Linux host only (one extra engine instance at nice 19); ABP_BRIDGE_LUA selects the bridge copy under test.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from isaac_bridge.abplus import AbplusTrainingEnv, launch_abplus, stop_abplus
from isaac_bridge.abplus_nav import STRATA, descent_move, goal_distance, goal_field, nav_key, sample_goal
from isaac_bridge.abplus_tasks import TaskSampler

GOALS, RADIUS, DECISIONS, REPEAT = 6, 20.0, 60, 4


def main():
    out_path, tasks_file = sys.argv[1], sys.argv[2]
    rooms = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 27890
    first = int(sys.argv[5]) if len(sys.argv) > 5 else 2147496000
    spec = json.loads(Path(tasks_file).read_text(encoding='utf8'))
    proc = launch_abplus('gotonav', port, 'exact')
    env = AbplusTrainingEnv(port=port)
    env.binary_obs = True
    env.lineage_mode = 3
    env.tasks = TaskSampler(spec['weights'], spec['normal'], spec['boss'], None)
    env.reset_mode = 'goto_room'
    try:
        env.connect()
        with open(out_path, 'w') as f:
            for r in range(rooms):
                seed = first + r
                try:
                    obs, info = env.reset_monstro(seed, 6, 1.0)
                except Exception as exc:
                    f.write(json.dumps(dict(seed=seed, error=f'{type(exc).__name__}: {exc}')) + '\n')
                    continue
                rng = np.random.default_rng([seed, 0x9A7])
                for k in range(GOALS):
                    got = sample_goal(obs, rng, STRATA[k % len(STRATA)])
                    if got is None:
                        break
                    target, stratum, d0 = got
                    nav = goal_field(obs, target)
                    env.goal = (target[0], target[1], RADIUS)
                    start = tuple(obs['players'][0]['pos'])
                    walked, outcome, steps = 0.0, 'time_limit', 0
                    prev = start
                    for steps in range(1, DECISIONS + 1):
                        if nav_key(obs) != nav['key']:
                            nav = goal_field(obs, target)
                        px, py = obs['players'][0]['pos']
                        obs, *_ = env.step({'move': descent_move(px, py, nav)}, repeat=REPEAT)
                        pos = tuple(obs['players'][0]['pos'])
                        walked += math.dist(prev, pos)
                        prev = pos
                        if obs['nav']['room_changed']:
                            outcome = 'exit'
                            break
                        if obs['nav']['goal_hit']:
                            outcome = 'goal'
                            break
                        if obs['players'][0].get('dead'):
                            outcome = 'death'
                            break
                    env.goal = None
                    left = goal_distance(*obs['players'][0]['pos'], nav)
                    f.write(json.dumps(dict(seed=seed, room=info.get('room_variant'), goal=k, stratum=stratum, d_geo=round(d0, 1),
                                            straight=round(math.dist(start, target), 1), outcome=outcome, decisions=steps,
                                            walked=round(walked, 1), left=round(left, 1) if left is not None else None,
                                            end=[round(v, 1) for v in obs['players'][0]['pos']])) + '\n')
                    f.flush()
                    if outcome in ('exit', 'death'):
                        break
                print(seed, info.get('room_variant'), flush=True)
    finally:
        try:
            env.close()
        finally:
            stop_abplus(proc, 'gotonav')


if __name__ == '__main__':
    main()
