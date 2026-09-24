"""Where does the Lua observation cost go? Per-component timings of abp_bridge.lua's obs builder.

Plays random actions in the Monstro arena and asks the bridge ({"cmd": "profile"}) to time each
component (player, entities, room, grid/doors, terrain, JSON) in several game states.
usage: python abplus_probe_obs_cost.py [seed]
"""
import json, sys
import numpy as np
from isaac_bridge.abplus import AbplusTransformerEnv, launch_abplus, stop_abplus

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2147483730
proc = launch_abplus('obscost', 27196, 'exact')
env = AbplusTransformerEnv(port=27196)
rng = np.random.default_rng(0)
try:
    env.reset(options={'arena_seed': seed})
    for step in range(301):
        if step % 60 == 0:
            env.bridge._send({'cmd': 'profile', 'n': 30})
            reply = env.bridge._recv()
            print(json.dumps({'step': step, 'entities': reply.get('entities'), 'bytes': reply.get('bytes'),
                              'ms': {k: round(v, 3) for k, v in reply['ms'].items()}}), flush=True)
        obs, r, term, trunc, info = env.step(np.array([int(rng.integers(45)), 0, 0]))
        if term or trunc:
            env.reset(options={'arena_seed': seed + step})
finally:
    try:
        env.close()
    finally:
        stop_abplus(proc, 'obscost')
