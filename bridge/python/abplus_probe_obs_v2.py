"""Bridge v2 (binary observations) against v1 (JSON) on the same game states, then speed.

Validation: the bridge sends every observation twice (JSON, then binary). Each pair must decode
to the same dict (abplus_obs.compare), and the policy frames VisibleHistory builds from the two
must be bit-identical. Random play with bombs covers bomb/explosion entities and terrain updates.
Speed: per-step wall time of step + frame encoding with JSON and with binary observations.
usage: python abplus_probe_obs_v2.py [episodes] [speed steps]
"""
import json
import sys
import time

import numpy as np

from isaac_bridge.abplus import AbplusTransformerEnv, launch_abplus, stop_abplus
from isaac_bridge.transformer_obs import VisibleHistory

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SPEED_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
SEED = 2147483740


class Env(AbplusTransformerEnv):
    def encode_observation(self, obs):
        return self.history.encode(obs)


def play(env, rng, steps):
    """Seconds spent in step() (bridge round trip + frame encoding), resets excluded."""
    done, spent = True, 0.0
    for i in range(steps):
        if done:
            env.reset(options={'arena_seed': SEED + i})
        bomb = int(rng.random() < 0.05 and env.action_masks()[46])
        t = time.perf_counter()
        _, _, term, trunc, _ = env.step(np.array([int(rng.integers(45)), bomb, 0]))
        spent += time.perf_counter() - t
        done = term or trunc
    return spent


def main():
    proc = launch_abplus('obsv2', 27198, 'exact')
    try:
        # 1. validation
        env = Env(port=27198)
        env.bridge.binary_obs, env.bridge.validate_obs = True, True
        hj, hb = VisibleHistory(64, 256, deadline=True), VisibleHistory(64, 256, deadline=True)
        stats = {'frames': 0, 'frame_mismatches': 0, 'max_abs_diff': {}, 'terrain_blocks': 0}
        last_version = [None]

        def check(reset):
            json_obs, bin_obs = env.bridge.last_pair
            if reset:
                hj.clear(), hb.clear()
            hj.previous_action[:] = env.history.previous_action
            hb.previous_action[:] = env.history.previous_action
            fj, fb = hj.encode(json_obs), hb.encode(bin_obs)
            stats['frames'] += 1
            if bin_obs['terrain']['version'] != last_version[0]:
                stats['terrain_blocks'] += 1
                last_version[0] = bin_obs['terrain']['version']
            for key in fj:
                a, b = np.asarray(fj[key], np.float64), np.asarray(fb[key], np.float64)
                if not np.array_equal(a, b):
                    diff = float(np.abs(a - b).max())
                    stats['max_abs_diff'][key] = max(stats['max_abs_diff'].get(key, 0.0), diff)
                    stats['frame_mismatches'] += 1

        rng = np.random.default_rng(0)
        steps = 0
        for episode in range(EPISODES):
            env.reset(options={'arena_seed': SEED + episode})
            check(reset=True)
            done = False
            while not done:
                bomb = int(rng.random() < 0.05 and env.action_masks()[46])
                _, _, term, trunc, _ = env.step(np.array([int(rng.integers(45)), bomb, 0]))
                check(reset=False)
                steps += 1
                done = term or trunc
        print(json.dumps({'validation': env.bridge.validation, 'frames': stats, 'episodes': EPISODES, 'steps': steps}),
              flush=True)
        env.close()
        # 2. speed: JSON (v1) vs binary (v2), same seeds and actions
        for binary in (False, True):
            env = Env(port=27198)
            env.bridge.binary_obs = binary
            seconds = play(env, np.random.default_rng(1), SPEED_STEPS)
            print(json.dumps({'format': 2 if binary else 1, 'steps': SPEED_STEPS,
                              'ms_per_step': round(1000 * seconds / SPEED_STEPS, 3)}), flush=True)
            env.close()
    finally:
        stop_abplus(proc, 'obsv2')


if __name__ == '__main__':
    main()
