"""A19 (gate 0), part 2: many clones of one state at once: what branching costs in time and memory.

Per seed: reset, `warm` decisions, then for each fan-out M of --fans:
  - M clones are forked one after another from the parent (time to make them all);
  - each clone plays its own `steps` decisions (one batched play per clone, all at once, one Python thread per clone);
    clone 0 and clone 1 get the same actions and must end in the same hidden state (digest), the others get different
    ones (how many distinct end states);
  - before the clones are closed: each clone's private memory (Private_Dirty of /proc/<pid>/smaps_rollup: the pages it
    had to copy) and the parent's;
  - the parent, which did not move, must still have the digest it had before the fan-out.
Reported per M: fork time per clone, logic frames per second of all clones together, private MiB per clone.
With --reseed the clones other than 0 and 1 also reseed the global MT (ABP_RESEED) first: branches then differ in the
game's random draws, not only in the actions.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_fork_fan.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:4 --fans 1,2,4,8,16,32 --out <dir>
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from abplus_probe_fork import digest, sticky_actions
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance


def private_mib(pid):
    try:
        with open(f'/proc/{pid}/smaps_rollup') as f:
            rows = dict(line.split(':', 1) for line in f if ':' in line)
        return (int(rows['Private_Dirty'].split()[0]) + int(rows.get('Private_Clean', '0 kB').split()[0])) / 1024, \
            int(rows['Pss'].split()[0]) / 1024
    except (OSError, KeyError, ValueError):
        return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:4')
    p.add_argument('--fans', default='1,2,4,8,16,32')
    p.add_argument('--warm', type=int, default=20)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--reseed', action='store_true')
    p.add_argument('--port', type=int, default=27960)
    p.add_argument('--name', default='fkfan')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0)
    inst = Instance(args.name, args.port, cfg, spec)
    rep = cfg.frames_per_decision
    rows = []
    try:
        for seed in parse_seeds(args.seeds):
            rng = np.random.default_rng(seed)
            inst.reset(seed)
            bridge = inst.env.bridge
            _, n, stop = bridge.play(sticky_actions(rng, args.warm), repeat=rep, stop_clear=True)
            if stop != 'done':
                continue
            d0 = digest(bridge)[0]
            for m in [int(x) for x in args.fans.split(',')]:
                plans = [sticky_actions(np.random.default_rng([seed, max(i, 1)]), args.steps, repeat_prob=0.7)
                         for i in range(m)]
                t = time.perf_counter()
                clones = [bridge.fork(tag=f'{seed}-{m}-{i}') for i in range(m)]
                fork_s = time.perf_counter() - t
                if args.reseed:
                    for i, c in enumerate(clones[2:], 2):
                        c.lua(f"return os.getenv('ABP_RESEED:{1000003 * seed % 2147483647 + i}')")
                results = [None] * m

                def run(i):
                    obs, played, why = clones[i].play(plans[i], repeat=rep, stop_clear=True)
                    results[i] = (played, why)

                threads = [threading.Thread(target=run, args=(i,)) for i in range(m)]
                t = time.perf_counter()
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()
                play_s = time.perf_counter() - t
                ends = [digest(c)[0] for c in clones]
                mem = [private_mib(c.pid) for c in clones]
                parent_mem = private_mib(inst.proc.pid)
                for c in clones:
                    c.close()
                frames = sum(r[0] for r in results) * rep
                row = dict(seed=seed, fan=m, fork_ms_each=1000 * fork_s / m, frames=frames,
                           frames_per_s=frames / play_s, play_s=play_s,
                           twins_equal=(ends[0] == ends[1]) if m > 1 else None, distinct_ends=len(set(ends)),
                           clone_private_mib=float(np.mean([x[0] for x in mem if x[0] is not None])),
                           clone_pss_mib=float(np.mean([x[1] for x in mem if x[1] is not None])),
                           parent_private_mib=parent_mem[0], parent_pss_mib=parent_mem[1],
                           parent_unchanged=digest(bridge)[0] == d0,
                           stops=sorted(set(r[1] for r in results)))
                rows.append(row)
                print(json.dumps(row), flush=True)
            row = dict(seed=seed, counters=bridge.lua("return tostring(os.getenv('ABP_FORK_COUNTERS'))"))
            print(json.dumps(row), flush=True)
    finally:
        inst.close()
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
        by = {}
        for r in rows:
            by.setdefault(r['fan'], []).append(r)
        summary = {m: dict(fork_ms_each=float(np.mean([r['fork_ms_each'] for r in v])),
                           frames_per_s=float(np.mean([r['frames_per_s'] for r in v])),
                           clone_private_mib=float(np.mean([r['clone_private_mib'] for r in v])),
                           twins_equal=sum(r['twins_equal'] is True for r in v),
                           parent_unchanged=sum(r['parent_unchanged'] for r in v), n=len(v))
                   for m, v in by.items()}
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
