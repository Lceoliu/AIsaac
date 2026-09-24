"""Environment-side capacity of the AB+ training workers, without the learner or CUDA.

Spawns N abplus_worker processes (two AB+ instances each) exactly as AbplusFrameVecEnv does,
steps them in lockstep with random actions (13-byte messages, frames into shared memory) and
reports decisions/s and game time. This is the rollout ceiling the learner can at most consume.
With combat-v2 it also reports the mean reward components of the finished episodes.
usage: python abplus_bench_workers.py <workers> <seconds> [mixture.json|none] [nice] [combat-v2|combat-v1]
                                      [recycle episodes]
"""
import json
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing.connection import wait
import struct
import sys
import time

import numpy as np

from isaac_bridge import abplus_worker as W

GAME_FPS, FRAMES_PER_DECISION = 30, 2


def exchange(conns, messages):
    for conn, msg in zip(conns, messages):
        conn.send_bytes(msg)
    pending = set(conns)
    while pending:
        for conn in wait(list(pending)):
            reply = conn.recv_bytes()
            if reply[:1] != b'k':
                raise RuntimeError(reply[1:].decode('utf8', 'replace'))
            pending.discard(conn)


def main():
    n, seconds = int(sys.argv[1]), float(sys.argv[2])
    tasks = None if len(sys.argv) < 4 or sys.argv[3] == 'none' else json.load(open(sys.argv[3]))
    nice = int(sys.argv[4]) if len(sys.argv) > 4 else 19
    profile = sys.argv[5] if len(sys.argv) > 5 else 'combat-v2'
    recycle = int(sys.argv[6]) if len(sys.argv) > 6 else W.RECYCLE_EPISODES
    if tasks:
        tasks = {k: tasks[k] for k in ('weights', 'normal', 'boss')}
    frame_bytes = 2 * n * W.FRAME_DTYPE.itemsize
    shm = shared_memory.SharedMemory(create=True, size=frame_bytes + n * W.META_DTYPE.itemsize)
    meta = np.ndarray((n,), W.META_DTYPE, buffer=shm.buf, offset=frame_bytes)
    frames = np.ndarray((2, n), W.FRAME_DTYPE, buffer=shm.buf)
    config = dict(num_envs=n, base_seed=1, mode='exact', name='bw', port=27600, nice=nice,
                  start_randomization=None, tasks=tasks, binary_obs=True, reward_profile=profile,
                  recycle_episodes=recycle)
    ctx = mp.get_context('spawn')
    conns, procs = [], []
    try:
        for i in range(n):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=W.worker_main, args=(i, config, shm.name, child))
            proc.start()
            child.close()
            conns.append(parent)
            procs.append(proc)
        for conn in conns:
            if conn.recv_bytes()[:1] != b'r':
                raise RuntimeError('worker start failed')
        t = time.perf_counter()
        exchange(conns, [b'R' + struct.pack('<qqff', 1 + i, 1, float('nan'), float('nan')) for i in range(n)])
        print(json.dumps({'workers': n, 'instances': 2 * n, 'reset_s': round(time.perf_counter() - t, 1)}), flush=True)
        rng = np.random.default_rng(0)
        steps, episodes, t0 = 0, 0, time.perf_counter()
        returns, finished = np.zeros(n), []
        while time.perf_counter() - t0 < seconds:
            actions = [b'S' + struct.pack('<3i', int(rng.integers(45)), 0, 0) for _ in range(n)]
            exchange(conns, actions)
            steps += 1
            returns += frames[0]['reward']
            for i in np.flatnonzero(frames[0]['done']):
                finished.append((W.OUTCOMES[frames[0]['outcome'][i]], returns[i], meta['components'][i].copy()))
                returns[i] = 0
            episodes += int(frames[0]['done'].sum())
        elapsed = time.perf_counter() - t0
        rate = steps * n / elapsed
        print(json.dumps({'workers': n, 'lockstep_steps': steps, 'decisions_per_s': round(rate),
                          'game_speed_x_realtime': round(rate * FRAMES_PER_DECISION / GAME_FPS, 1),
                          'game_hours_per_hour': round(rate * FRAMES_PER_DECISION / GAME_FPS, 1),
                          'episodes_finished': episodes, 'worker_step_ms_mean': round(float(meta['step_ms'].mean()), 2),
                          'switch_wait_ms_max': round(float(meta['switch_wait_ms'].max()), 2),
                          'errors': int(meta['errors'].sum()), 'instance_recycles': int(meta['recycles'].sum())}),
              flush=True)
        if finished:
            parts = np.array([c for _, _, c in finished])
            print(json.dumps({'reward_profile': profile, 'outcomes': dict(zip(*np.unique([o for o, _, _ in finished],
                                                                                      return_counts=True))),
                              'mean_return': round(float(np.mean([r for _, r, _ in finished])), 3),
                              'return_equals_components': bool(np.allclose([r for _, r, _ in finished], parts.sum(1),
                                                                           atol=1e-3)) if profile == 'combat-v2' else None,
                              'mean_components': ({k: round(float(v), 3) for k, v in zip(W.COMPONENTS, parts.mean(0))}
                                                  if profile == 'combat-v2' else None)},
                             default=int), flush=True)
    finally:
        for conn in conns:
            try:
                conn.send_bytes(b'Q')
            except OSError:
                pass
        for proc in procs:
            proc.join(timeout=90)
            if proc.is_alive():
                proc.terminate()
        del meta, frames
        shm.close()
        shm.unlink()


if __name__ == '__main__':
    main()
