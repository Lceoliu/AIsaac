"""Teacher search throughput (2026-10-05): how many hindsight searches the machine does per second when N workers
search at once, as during the trainer's update hold, and what one costs.

Each worker launches an instance (fork mode, the frame fast paths of the environment), takes start states of its group
(seed + worker + N k, as the sampler), records episodes of sticky random actions on lean clones of the template and
keeps their hurts; then all workers start together and search their hurts one after another (tok_sampler.hindsight,
the sampler's TokSamplerConfig with --cfg overrides) for --seconds, at most --slots at once (tok_sampler.SearchSlots;
0: no limit). Reported: searches and teacher records per second over the machine, CPU seconds per search from
/proc/stat over the timed window (every process on the machine: run it on a quiet host) and from the searches' own
processes (/proc schedstat, as ISAAC_RL_TEACH_PROF: restore clone, its clones up to their report, the base's side of
the fork; not the clones' exit), depths and decisions replayed per search.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_bench_teacher.py --groups-file ../abplus/catalog/scaling2_groups.json --groups normal:8,boss:5,normal_big:3 \
      --workers 16 --seconds 60 --out <file.json> [--slots 4] [--cfg '{"teacher_skip_known": true}']
"""
import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import time

import numpy as np

from abplus_probe_fork import sticky_actions
from abplus_probe_teacher import record_episode
from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.tok_sampler import SearchSlots, TokSamplerConfig, hindsight, hindsight_shared


def parse_groups(text, n):
    names, weights = [], []
    for part in text.split(','):
        name, w = part.split(':')
        names.append(name)
        weights.append(float(w))
    total = sum(weights)
    assign, acc = [], 0.0
    for i in range(n):   # worker i -> group by cumulative share
        x = (i + 0.5) / n * total
        acc, g = 0.0, 0
        for g, w in enumerate(weights):
            acc += w
            if x < acc:
                break
        assign.append(g)
    return names, assign


def cpu_ticks():
    with open('/proc/stat') as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return sum(v) - v[3] - v[4], sum(v)


def worker(i, args, group, barrier, out_q):
    os.environ.update(FORK_ENV)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group=group, tasks='', seconds=0.0))
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=args.nice,
                  stub_list=args.stub_list)
    cfg = dataclasses.replace(TokSamplerConfig(name=args.name, port=args.port, teacher_slots=args.slots),
                              **json.loads(args.cfg))
    gate = SearchSlots(cfg)
    os.nice(args.nice)
    inst = None
    items, templates = [], []
    res = dict(worker=i, group=group, searches=0, records=0, depths=0, replayed=0, errors=0, wall=0.0, cpu_ns=0,
               hurts=0, base_ns=0, replay_ns=0, kids_ns=0, restore_ns=0, reaped_ns=0, skipped=0, launch_errors=0)
    for attempt in range(4):   # an instance sometimes fails to come up (a launch crash): try again
        try:
            inst = Instance(f'{args.name}{i}', args.port + i, gx, spec)
            break
        except Exception:
            res['launch_errors'] += 1
            time.sleep(2.0)
    try:
        try:   # whatever happens here, the worker reaches the barrier (the others wait for it)
            rng = np.random.default_rng(args.seed + i)
            k = 0
            while inst is not None and sum(len(it[3]) for it in items) < args.items and k < 50:
                seed = args.seed + i + args.workers * k
                k += 1
                inst.reset(seed)
                template = inst.env.bridge.fork(tag='template', alarm=0)
                templates.append(template)
                for _ in range(args.episodes):
                    reseed = int(rng.integers(1, 2 ** 31 - 1))
                    applied, hurts = record_episode(template, reseed, sticky_actions(rng, args.steps), 4)
                    res['hurts'] += len(hurts)
                    hurts = [h for h in hurts if h - 1 - 2 >= 1]
                    if not hurts:
                        continue
                    # as the worker: up to teacher_per_episode hurts of an episode, in time order
                    picks = [hurts[i] for i in sorted(rng.permutation(len(hurts))[:cfg.teacher_per_episode])]
                    if cfg.teacher_pair:
                        items.append((template, reseed, applied, picks))
                    else:
                        items.extend((template, reseed, applied, [h]) for h in picks)
            rng.shuffle(items)
        except Exception:
            res['launch_errors'] += 1
        barrier.wait()
        t_end = time.perf_counter() + args.seconds
        j = 0
        while items and time.perf_counter() < t_end:
            if args.idle:   # the baseline: instances and parked clones, no search
                time.sleep(0.05)
                continue
            slot = gate.take()
            if slot is None:
                time.sleep(0.0005)
                continue
            try:
                tpl, reseed, applied, hs = items[j % len(items)]
                j += 1
                prof = []
                t0 = time.perf_counter()
                try:
                    if len(hs) > 1:
                        founds = hindsight_shared(tpl, reseed, applied, hs, cfg, 10 ** 6, prof=prof)
                    else:
                        founds = [hindsight(tpl, reseed, applied, hs[0], cfg, 10 ** 6, prof=prof)]
                except Exception:
                    res['errors'] += 1
                    continue
                res['wall'] += time.perf_counter() - t0
            finally:
                gate.give(slot)
            res['searches'] += len(hs)
            res['records'] += sum(len(f) for f in founds)
            for dp in prof:
                res['depths'] += 1
                res['skipped'] += dp['nb'] < 9
                res['replayed'] += dp['replayed']
                own = dp['fm'].get(dp['nb'] - 1)
                kids = sum(v[8] for b, v in dp['fm'].items() if b != dp['nb'] - 1)
                res['base_ns'] += dp['base_cpu']
                res['replay_ns'] += dp['st_hello'][0] + dp['st_replay'][0]
                res['kids_ns'] += kids
                if own is not None and len(own) > 13:   # the restore clone, and its clones reaped with their exits
                    res['restore_ns'] += own[8]
                    res['reaped_ns'] += own[13]
                    res['cpu_ns'] += dp['base_cpu'] + own[8] + own[13]
                else:
                    res['cpu_ns'] += dp['base_cpu'] + sum(v[8] for v in dp['fm'].values())
        res['items'] = len(items)
    finally:
        for t in templates:
            try:
                t.close()
            except OSError:
                pass
        if inst is not None:
            inst.close()
        gate.close()
        out_q.put(res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='../abplus/catalog/scaling2_groups.json')
    p.add_argument('--groups', default='normal:8,boss:5,normal_big:3')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seconds', type=float, default=60.0)
    p.add_argument('--slots', type=int, default=0)
    p.add_argument('--items', type=int, default=24, help='hurts recorded per worker before the timed part')
    p.add_argument('--episodes', type=int, default=3, help='episodes per start state')
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--seed', type=int, default=2147520000)
    p.add_argument('--nice', type=int, default=10)
    p.add_argument('--cfg', default='{}')
    p.add_argument('--idle', action='store_true', help='no searches in the timed part (the machine baseline)')
    p.add_argument('--stub-list', default='')
    p.add_argument('--name', default='tchb')
    p.add_argument('--port', type=int, default=38800)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    names, assign = parse_groups(args.groups, args.workers)
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(args.workers + 1)
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(i, args, names[assign[i]], barrier, q)) for i in range(args.workers)]
    for pr in procs:
        pr.start()
    barrier.wait()
    c0, t0 = cpu_ticks(), time.perf_counter()
    time.sleep(args.seconds)
    c1, t1 = cpu_ticks(), time.perf_counter()
    rows = [q.get() for _ in procs]
    for pr in procs:
        pr.join()
    tot = {k: sum(r[k] for r in rows) for k in ('searches', 'records', 'depths', 'replayed', 'errors', 'wall',
                                                 'cpu_ns', 'hurts', 'base_ns', 'replay_ns', 'kids_ns', 'restore_ns',
                                                 'reaped_ns', 'skipped', 'launch_errors')}
    hz = os.sysconf('SC_CLK_TCK')
    busy_s = (c1[0] - c0[0]) / hz
    n = max(tot['searches'], 1)
    summary = dict(args=vars(args), seconds=t1 - t0, searches=tot['searches'], records=tot['records'],
                   errors=tot['errors'], launch_errors=tot['launch_errors'],
                   searches_per_s=tot['searches'] / (t1 - t0),
                   records_per_s=tot['records'] / (t1 - t0), busy_cpus=busy_s / (t1 - t0),
                   machine_cpu_per_search=busy_s / n, machine_cpu_per_record=busy_s / max(tot['records'], 1),
                   own_cpu_per_search=tot['cpu_ns'] / 1e9 / n, wall_per_search=tot['wall'] / n,
                   parts_per_search={k: tot[k] / 1e9 / n for k in ('base_ns', 'replay_ns', 'restore_ns', 'kids_ns',
                                                                    'reaped_ns')},
                   skipped_per_depth=tot['skipped'] / max(tot['depths'], 1),
                   depths_per_search=tot['depths'] / n, replayed_per_depth=tot['replayed'] / max(tot['depths'], 1),
                   env={k: v for k, v in os.environ.items() if k.startswith(('ABP_', 'ISAAC_RL_'))}, workers=rows)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()
                      if k not in ('workers', 'args', 'env')}))


if __name__ == '__main__':
    main()
