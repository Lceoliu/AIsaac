"""Memory and time growth of one AB+ instance per operation (training instances grew ~0.76 MiB per reset).

Runs each operation N times on a fresh instance and reports the resident-memory growth of the game
process and of the Lua heap per operation, and the operation time at the start and at the end:
  arena    the full training reset (reseed, restart 0, reseed, goto, Lua setup, wait for Monstro)
  restart  "restart 0" alone          goto  "goto s.boss.1010" alone (no restart)
  lua      the room setup chunk alone  steps  20 two-frame steps, no reset
  mixture_reset / arena_play / mixture_play   training-like episodes: room-mixture resets, and
           up to 300 steps of random moves, shots and bombs until death or clear
usage: python abplus_probe_leak.py [N] [port] [ops,comma,separated] [mixture.json] [mode] [KEY=VALUE,...]
  mode: exact (default) / skip / render (abplus.MODES); KEY=VALUE pairs override the instance
  environment, an empty VALUE removes the key (e.g. ABP_STUB_LIST= runs exact mode without stubs).
"""
import hashlib
import json
import sys
import time

import numpy as np

from isaac_bridge.abplus import MODES, ROOM_LUA, AbplusTrainingEnv, launch_abplus, stop_abplus
from isaac_bridge.abplus_tasks import TaskSampler

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 27700
ONLY = sys.argv[3].split(',') if len(sys.argv) > 3 else None
MIXTURE = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != '-' else '../abplus/catalog/mixture_basement1.json'
MODE = sys.argv[5] if len(sys.argv) > 5 else 'exact'
ENV = dict(kv.split('=', 1) for kv in sys.argv[6].split(',')) if len(sys.argv) > 6 and sys.argv[6] else {}


def rss_mib(pid):
    with open(f'/proc/{pid}/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return float('nan')


if ENV:
    # launch_abplus applies MODES[mode] after the caller's environment, so overrides go into a copy.
    MODES[MODE + '-probe'] = {k: v for k, v in {**MODES[MODE], **ENV}.items() if v != ''}
    MODE = MODE + '-probe'
print(json.dumps({'mode': MODE, 'env': MODES[MODE]}), flush=True)
proc = launch_abplus('leak', PORT, MODE, nice=19)
bridge = AbplusTrainingEnv(port=PORT)
bridge.binary_obs = True
try:
    bridge.connect()
    bridge.reset_monstro(1)
    mixture = TaskSampler.from_file(MIXTURE)
    trajectory = hashlib.sha256()   # game state along every played episode, to compare modes

    def play(seed, tasks):
        rng = np.random.default_rng(seed)
        bridge.tasks = tasks
        obs, _ = bridge.reset_monstro(seed)
        for _ in range(300):
            obs = bridge.step({'move': int(rng.integers(9)), 'shoot': int(rng.integers(5)),
                               'bomb': int(rng.random() < 0.02)}, repeat=2)[0]
            trajectory.update(json.dumps([obs['players'][0]['pos'], obs['players'][0]['hearts'],
                                          sorted((e['type'], e['variant'], e['pos']) for e in obs['entities'])]).encode())
            if obs['players'][0]['dead'] or obs['room']['clear']:
                break
        bridge.tasks = None

    def mixture_reset(seed):
        bridge.tasks = mixture
        bridge.reset_monstro(seed)
        bridge.tasks = None

    ops = {
        'mixture_reset': lambda i: mixture_reset(1000 + i),
        'arena_play': lambda i: play(2000 + i, None),
        'mixture_play': lambda i: play(3000 + i, mixture),
        'mixture_play2': lambda i: play(4000 + i, mixture),   # measured after mixture_play as warm-up
        'arena': lambda i: bridge.reset_monstro(100 + i),
        'restart': lambda i: bridge.reset(phases=[['restart 0']], settle=2),
        'goto': lambda i: bridge.reset(phases=[['goto s.boss.1010']], settle=8),
        'lua': lambda i: bridge.lua(ROOM_LUA.format(player_hp=6, boss_hp_fraction='1.0', rng_seed=i)),
        'steps': lambda i: [bridge.step({}, repeat=2) for _ in range(20)],
    }
    for name, op in ops.items():
        if ONLY and name not in ONLY:
            continue
        lua0 = float(bridge.lua("return tostring(collectgarbage('count'))"))
        rss0 = rss_mib(proc.pid)
        times = []
        for i in range(N):
            t = time.perf_counter()
            op(i)
            times.append(1000 * (time.perf_counter() - t))
        lua1 = float(bridge.lua("return tostring(collectgarbage('count'))"))
        rss1 = rss_mib(proc.pid)
        head, tail = sorted(times[:20]), sorted(times[-20:])
        first, last = head[len(head) // 2], tail[len(tail) // 2]
        print(json.dumps({'op': name, 'n': N, 'rss_mib_per_op': round((rss1 - rss0) / N, 4),
                          'lua_kib_per_op': round((lua1 - lua0) / N, 3), 'rss_mib': [round(rss0), round(rss1)],
                          'ms_median_first20': round(first, 1), 'ms_median_last20': round(last, 1),
                          'ms_mean': round(sum(times) / len(times), 1),
                          'trajectory': trajectory.hexdigest()[:16]}), flush=True)
finally:
    try:
        bridge.close()
    finally:
        stop_abplus(proc, 'leak')
