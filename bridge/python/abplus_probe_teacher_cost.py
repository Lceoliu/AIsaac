"""Teacher search cost outside training (2026-10-05, diagnostic): per-search CPU breakdown of tok_sampler.hindsight
(the prof records of ISAAC_RL_TEACH_PROF) on recorded hurts of sticky-random episodes, and the memory of the processes
a search forks (root, template, a restore clone after its replay: /proc status and the largest smaps entries).

Run several copies at once (other --name / --port) to see the cost under contention.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_teacher_cost.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:4 --out <file.jsonl> [--smaps]
"""
import argparse
import json
import os
import time

import numpy as np

from abplus_probe_fork import sticky_actions
from abplus_probe_teacher import record_episode
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder
from isaac_bridge.tok_sampler import TokSamplerConfig, hindsight, play_actions

STATUS_KEYS = ('VmSize', 'VmRSS', 'RssAnon', 'RssFile', 'RssShmem', 'VmPTE', 'VmData')


def status(pid):
    out = {}
    try:
        for line in open(f'/proc/{pid}/status'):
            k, _, v = line.partition(':')
            if k in STATUS_KEYS:
                out[k] = v.strip()
    except OSError:
        pass
    return out


def smaps_top(pid, n=25):
    """The largest mappings by Rss: (rss kB, anonymous kB, size kB, flags, name)."""
    rows, cur = [], None
    try:
        for line in open(f'/proc/{pid}/smaps'):
            parts = line.split()
            if '-' in parts[0] and len(parts) >= 5 and ':' not in parts[0]:
                cur = dict(range=parts[0], perms=parts[1], name=parts[5] if len(parts) > 5 else '[anon]', rss=0,
                           anon=0, size=0, thp=0)
                rows.append(cur)
            elif parts[0] == 'Rss:':
                cur['rss'] = int(parts[1])
            elif parts[0] == 'Anonymous:':
                cur['anon'] = int(parts[1])
            elif parts[0] == 'Size:':
                cur['size'] = int(parts[1])
            elif parts[0] == 'AnonHugePages:':
                cur['thp'] = int(parts[1])
    except OSError:
        return []
    total = dict(n=len(rows), rss=sum(r['rss'] for r in rows), anon=sum(r['anon'] for r in rows),
                 thp=sum(r['thp'] for r in rows))
    return total, sorted(rows, key=lambda r: -r['rss'])[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:4')
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--hurts', type=int, default=2)
    p.add_argument('--port', type=int, default=38500)
    p.add_argument('--name', default='tchcost')
    p.add_argument('--stub-list', default='')
    p.add_argument('--smaps', action='store_true')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  stub_list=args.stub_list)
    cfg = TokSamplerConfig()
    fpd = cfg.frames_per_decision
    inst = Instance(args.name, args.port, gx, spec)
    fh = open(args.out, 'a')
    try:
        for seed in parse_seeds(args.seeds):
            rng = np.random.default_rng(seed)
            inst.reset(seed)
            template = inst.env.bridge.fork(tag='template', alarm=0)
            try:
                if args.smaps:
                    root_pid = inst.proc.pid
                    st = template.fork(reseed=12345, lean=True, alarm=120)
                    play_actions(st, LeanDecoder(), [(1, 1, 0)] * 100, fpd)
                    mem = {name: dict(status=status(pid), smaps=smaps_top(pid))
                           for name, pid in (('root', root_pid), ('template', template.pid), ('restore', st.pid))}
                    st.close()
                    fh.write(json.dumps(dict(seed=seed, memory=mem)) + '\n')
                for e in range(args.episodes):
                    reseed = int(rng.integers(1, 2 ** 31 - 1))
                    applied, hurts = record_episode(template, reseed, sticky_actions(rng, args.steps), fpd)
                    for hurt_at in hurts[:args.hurts]:
                        prof = []
                        t0 = time.perf_counter()
                        c0 = time.thread_time()
                        found = hindsight(template, reseed, applied, hurt_at, cfg, 10 ** 6, prof=prof)
                        fh.write(json.dumps(dict(t=time.time(), wall=time.perf_counter() - t0, encode=0.0,
                                                 worker_cpu=time.thread_time() - c0, held=1, held_end=1,
                                                 applied=len(applied), hurt_at=hurt_at, records=len(found),
                                                 queue=0, running=int(open('/proc/loadavg').read().split()[3]
                                                                      .split('/')[0]), depths=prof)) + '\n')
                        fh.flush()
            finally:
                template.close()
    finally:
        inst.close()
        fh.close()


if __name__ == '__main__':
    main()
