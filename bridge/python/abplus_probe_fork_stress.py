"""A19 (gate 0), part 3: many forks in a row: how often does a clone fail to answer, and where is it stuck?

One instance; per seed a reset and `warm` decisions, then `forks` times: fork a clone (watchdog `alarm` real seconds),
play `steps` decisions in it, close it; the parent plays one decision every `parent_every` forks, so the forks fall on
different moments of the game's other threads. A clone that does not answer within the watchdog is counted as stuck; with
ABP_FORK_HANG_DIR abp_turbo writes its main thread's stack to <out>/hang-<pid>.txt.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_fork_stress.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:8 --forks 250 --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from abplus_probe_fork import sticky_actions
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.env import BridgeError


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:8')
    p.add_argument('--forks', type=int, default=250)
    p.add_argument('--warm', type=int, default=10)
    p.add_argument('--steps', type=int, default=6)
    p.add_argument('--alarm', type=int, default=4)
    p.add_argument('--parent-every', type=int, default=5)
    p.add_argument('--env', action='append', default=[], help="NAME=VALUE for the game (repeatable)")
    p.add_argument('--port', type=int, default=27980)
    p.add_argument('--name', default='fkstress')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    os.environ['ABP_FORK_HANG_DIR'] = str(out)
    os.environ.update(kv.split('=', 1) for kv in args.env)
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0)
    inst = Instance(args.name, args.port, cfg, spec)
    total = dict(forks=0, ok=0, stuck=0, fork_errors=0, parent_dead=0)
    t0 = time.perf_counter()
    try:
        for seed in parse_seeds(args.seeds):
            rng = np.random.default_rng(seed)
            inst.reset(seed)
            bridge = inst.env.bridge
            bridge.play(sticky_actions(rng, args.warm), repeat=4, stop_clear=True)
            for i in range(args.forks):
                plan = sticky_actions(rng, args.steps, repeat_prob=0.5)
                total['forks'] += 1
                try:
                    clone = bridge.fork(tag=f'{seed}-{i}', alarm=args.alarm)
                except (BridgeError, OSError) as exc:
                    total['fork_errors'] += 1
                    print('fork error', seed, i, repr(exc), flush=True)
                    continue
                try:
                    clone.play(plan, repeat=4, stop_clear=True)
                    total['ok'] += 1
                    counters = clone.lua("return tostring(os.getenv('ABP_FORK_COUNTERS'))")
                    passed = int(counters.split('clone_passed=')[1].split()[0]) if 'clone_passed=' in counters else 0
                    total['clone_passed'] = total.get('clone_passed', 0) + passed
                    total['clones_that_passed'] = total.get('clones_that_passed', 0) + (passed > 0)
                except (BridgeError, OSError) as exc:
                    total['stuck'] += 1
                    print('stuck', seed, i, clone.pid, repr(exc), flush=True)
                finally:
                    try:
                        clone.close()
                    except OSError:
                        pass
                if i % args.parent_every == 0:
                    _, _, stop = bridge.play(plan[:1], repeat=4, stop_clear=True)
                    if stop != 'done':   # the parent's episode ended: a fresh state
                        total['parent_dead'] += 1
                        inst.reset(seed)
                        bridge = inst.env.bridge
                        bridge.play(sticky_actions(rng, args.warm), repeat=4, stop_clear=True)
            total['counters'] = bridge.lua("return tostring(os.getenv('ABP_FORK_COUNTERS'))")
            print(json.dumps(dict(seed=seed, **total, seconds=round(time.perf_counter() - t0, 1))), flush=True)
    finally:
        inst.close()
        hangs = {}
        for f in sorted(out.glob('hang-*.txt')):
            hangs[f.name] = f.read_text().splitlines()[:40]
        total['hang_files'] = len(hangs)
        (out / 'summary.json').write_text(json.dumps(dict(total=total, hangs=hangs), indent=1))
        print('SUMMARY', json.dumps(total))


if __name__ == '__main__':
    main()
