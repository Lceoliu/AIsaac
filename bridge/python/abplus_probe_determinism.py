"""Arena reset determinism probe: same seed, fixed actions; diff raw observations; log extra Monstros."""
import json, sys
from isaac_bridge.abplus import AbplusTransformerEnv, launch_abplus, stop_abplus
import numpy as np

seed = int(sys.argv[1]); runs = int(sys.argv[2]); steps = int(sys.argv[3])
proc = launch_abplus('arenaprobe', 27191, 'exact')
env = AbplusTransformerEnv(port=27191)
IGN = {'logic_frames', 'game_frame', 'id', 'frame'}

def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in IGN}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o

def first_diff(a, b, path=''):
    if type(a) != type(b):
        return path, a, b
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return path + '.' + k, a.get(k), b.get(k)
            d = first_diff(a[k], b[k], path + '.' + k)
            if d: return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return path + '#len', len(a), len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, path + f'[{i}]')
            if d: return d
        return None
    return None if a == b else (path, a, b)

try:
    traces = []
    rng = np.random.default_rng(1)
    actions = [np.array([int(rng.integers(45)), 0, 0]) for _ in range(steps)]
    for run in range(runs):
        obs, info = env.reset(options={'arena_seed': seed})
        tr = [strip(env.raw_obs)]
        extra = []
        for t, a in enumerate(actions):
            obs, r, term, trunc, info = env.step(a)
            raw = env.raw_obs
            m = [(e['id'], e['subtype'], e.get('anim'), e.get('aframe'), e.get('age'), [round(v, 1) for v in e['pos']])
                 for e in raw['entities'] if e['type'] == 20]
            if len(m) != 1: extra.append((t, m))
            tr.append(strip(raw))
            if term or trunc: break
        traces.append(tr)
        print(json.dumps({'run': run, 'arena_setup': info.get('arena_setup'), 'len': len(tr), 'extra_monstro': extra[:6]}), flush=True)
    for run in range(1, runs):
        for t, (x, y) in enumerate(zip(traces[0], traces[run])):
            d = first_diff(x, y)
            if d:
                print(json.dumps({'compare': [0, run], 'step': t, 'path': d[0], 'a': d[1], 'b': d[2]})[:600], flush=True)
                break
        else:
            print(json.dumps({'compare': [0, run], 'identical_steps': min(len(traces[0]), len(traces[run]))}))
finally:
    try:
        env.close()
    finally:
        stop_abplus(proc, 'arenaprobe')
