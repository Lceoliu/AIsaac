"""Phase C probe (2026-10-08): the build lab's panel (tok_lab.parse_panel / build_panel_state) in one root instance.

Per panel state: the room-mode reset of its seed in the root (floor mode otherwise, as a training worker's root), the
parked clone, its build time and its hidden-state digest; then the root builds --floors floors (as it would between lab
jobs) and the parked clones' memory is read (Pss / Private_Dirty, /proc/<pid>/smaps_rollup); then every state is
built again from its seed and its digest compared with the parked one's (are rebuilt states the same states?); then per
state a lab start: fork, transplant (--build), reseed, first observation, timed.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_lab_panel.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.tok_lab import build_panel_state, parse_panel, state_reseed, stats_of, transplant_lua
from isaac_bridge.tok_sampler import STATS, trim_parked


def smaps(pid):
    out = {}
    try:
        for line in open(f'/proc/{pid}/smaps_rollup'):
            k, _, v = line.partition(':')
            if k in ('Pss', 'Private_Dirty', 'Rss'):
                out[k] = int(v.split()[0]) / 1024
    except OSError:
        pass
    return out


def digest(c):
    return hashlib.blake2b(c.lua('return ABPGX_DIGEST(false)').encode(), digest_size=12).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--panel', default='normal:24,boss:6,normal_big:6')
    p.add_argument('--seed', type=int, default=2147700000)
    p.add_argument('--floors', type=int, default=3)
    p.add_argument('--build', default='118,2', help='the collectibles of the timed lab starts')
    p.add_argument('--port', type=int, default=44200)
    p.add_argument('--name', default='labpn')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names, panel = parse_panel(args.panel, args.seed)
    specs = [load_spec(argparse.Namespace(groups_file=args.groups_file, group=n, tasks='', seconds=0.0)) for n in names]
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    spec0 = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    inst = Instance(args.name, args.port, gx, spec0)
    cfg = argparse.Namespace(trim_parked=True)
    parked, recs = [], []
    try:
        inst.env.bridge.reset_mode = 'floor'
        inst.reset(1)   # connect (as the worker's first floor build)
        for k, (g, seed) in enumerate(panel):
            t0 = time.perf_counter()
            clone, info = build_panel_state(inst, specs[g], seed)
            dt = time.perf_counter() - t0
            trim_parked(clone, cfg, np.zeros(len(STATS)))
            recs.append(dict(state=k, group=names[g], seed=seed, build_s=round(dt, 3), task=info.get('task'),
                             variant=info.get('variant'), digest=digest(clone), pid=clone.pid))
            parked.append(clone)
        root_after_panel = smaps(inst.proc.pid)
        t0 = time.perf_counter()
        for f in range(args.floors):   # the root moves on (as between lab jobs)
            inst.env.bridge.reset_mode = 'floor'
            inst.reset(100 + f)
        floors_s = time.perf_counter() - t0
        for r, c in zip(recs, parked):
            r.update({f'mem_{k}': round(v, 1) for k, v in smaps(c.pid).items()})
        # rebuilt from the seed: the same state?
        for r, (g, seed) in zip(recs, panel):
            t0 = time.perf_counter()
            clone, _ = build_panel_state(inst, specs[g], seed)
            r['rebuild_s'] = round(time.perf_counter() - t0, 3)
            r['rebuild_same'] = digest(clone) == r['digest']
            clone.close()
        # lab starts: fork, transplant, reseed, first observation
        build = [int(v) for v in args.build.split(',') if v]
        for r, c, (g, seed) in zip(recs, parked, panel):
            t0 = time.perf_counter()
            B = c.fork(lean=True, alarm=300)
            t1 = time.perf_counter()
            B.lua(transplant_lua(build))
            t2 = time.perf_counter()
            B.reseed(state_reseed(seed, 0))
            B._send({"cmd": "obs"})
            o = read_lean(B, LeanDecoder())
            t3 = time.perf_counter()
            st, fam, _ = stats_of(o)
            r.update(fork_ms=round(1000 * (t1 - t0), 2), transplant_ms=round(1000 * (t2 - t1), 2),
                     reseed_obs_ms=round(1000 * (t3 - t2), 2), monsters_hp=round(float(o.monsters_hp), 1),
                     n_entities=len(o.entities), damage_after=st[0], clear=bool(o.clear))
            B.close()
        for r in recs:
            print(json.dumps(r), flush=True)
        mem = [r.get('mem_Pss', 0) for r in recs]
        summary = dict(states=len(recs), build_s=[round(float(np.percentile([r['build_s'] for r in recs], q)), 3)
                                                  for q in (0, 50, 100)],
                       rebuild_same=sum(r['rebuild_same'] for r in recs),
                       pss_mib=[round(float(np.percentile(mem, q)), 1) for q in (0, 50, 100)],
                       pss_total_mib=round(float(sum(mem)), 1),
                       private_dirty_mib=round(float(sum(r.get('mem_Private_Dirty', 0) for r in recs)), 1),
                       root_after_panel=root_after_panel, root_now=smaps(inst.proc.pid), floors_s=round(floors_s, 2),
                       fork_ms=float(np.median([r['fork_ms'] for r in recs])),
                       transplant_ms=float(np.median([r['transplant_ms'] for r in recs])),
                       reseed_obs_ms=float(np.median([r['reseed_obs_ms'] for r in recs])),
                       tasks={t: sum(r['task'] == t for r in recs) for t in {r['task'] for r in recs}})
        print('SUMMARY', json.dumps(summary), flush=True)
        (out / 'panel.json').write_text(json.dumps(dict(summary=summary, states=recs), indent=1))
    finally:
        for c in parked:
            try:
                c.close()
            except OSError:
                pass
        inst.close()


if __name__ == '__main__':
    main()
