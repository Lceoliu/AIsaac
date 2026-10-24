"""Room-mixture resets: success, entrance attempts, reset time, enemies, determinism.

For each task kind (normal, boss) plays N seeds with random actions for a few seconds, then
replays the first seeds and compares trajectory hashes.
usage: python abplus_probe_rooms.py <mixture.json> [seeds per kind] [steps]
"""
import collections, hashlib, json, sys, time
import numpy as np
from isaac_bridge.abplus import AbplusTransformerEnv, launch_abplus, stop_abplus
from isaac_bridge.abplus_tasks import TaskSampler

spec = json.load(open(sys.argv[1]))
per_kind = int(sys.argv[2]) if len(sys.argv) > 2 else 20
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 60


class Env(AbplusTransformerEnv):
    def encode_observation(self, obs):
        return self.history.encode(obs)


def episode(env, seed):
    t = time.perf_counter()
    env.reset(options={'arena_seed': seed})
    reset_ms = 1000 * (time.perf_counter() - t)
    info = dict(env.bridge.last_info)
    raw = env.raw_obs
    enemies = [e for e in raw['entities'] if e.get('enemy')]
    h = hashlib.sha256()
    rng = np.random.default_rng(seed)
    outcome = 'running'
    for _ in range(steps):
        _, _, term, trunc, info2 = env.step(np.array([int(rng.integers(45)), 0, 0]))
        r = env.raw_obs
        h.update(json.dumps([r['players'][0]['pos'], sorted((e['type'], e['pos']) for e in r['entities'])]).encode())
        if term or trunc:
            outcome = info2['outcome']
            break
    return dict(seed=seed, reset_ms=round(reset_ms), enemies=len(enemies),
                kinds=sorted({(e['type'], e['variant']) for e in enemies}), outcome=outcome, hash=h.hexdigest()[:12])


proc = launch_abplus('rooms', 27193, 'exact')
env = Env(port=27193)
env.bridge.binary_obs = True
try:
    for kind in ('normal', 'boss'):
        env.bridge.tasks = TaskSampler({kind: 1.0}, spec['normal'], spec['boss'])
        rows = []
        for seed in range(1000, 1000 + per_kind):
            task = env.bridge.tasks.choose(seed)
            row = episode(env, seed)
            row.update(variant=task.variant, entrance=task.entrance)
            rows.append(row)
        # the bridge's reset info of the last episode shows the accepted entrance and attempts
        repeat = [episode(env, r['seed'])['hash'] == r['hash'] for r in rows[:4]]
        print(json.dumps({'kind': kind, 'episodes': len(rows),
                          'reset_ms_median': float(np.median([r['reset_ms'] for r in rows])),
                          'enemies_median': float(np.median([r['enemies'] for r in rows])),
                          'distinct_enemy_kinds': len({k for r in rows for k in map(tuple, r['kinds'])}),
                          'outcomes': collections.Counter(r['outcome'] for r in rows),
                          'replay_identical': repeat}), flush=True)
        for r in rows[:5]:
            print('   ', json.dumps(r))
finally:
    try:
        env.close()
    finally:
        stop_abplus(proc, 'rooms')
