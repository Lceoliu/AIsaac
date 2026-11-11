"""Probe (2026-11-11) of abp_turbo's ABP_CLONE_STATS: one instance with the counters on, clones of known purposes and
lengths, then the counters checked against what was done (bridge/ops/clone_stats.py prints them).

From a lean clone S of the instance (purpose hurt): K clones each of purposes hurt, fatal and cf, each stepping N lean
steps of 4 frames and closed (ABP_EXIT); one fork_many of B batches of M codes (purpose fork_many); S closed; the root
steps a few frames (its loop ticks reap S). Checks per purpose: forks (in S / the root), clones, exits, frames per clone
(4 N for the stepped ones), reaped with a teardown CPU, the parent-side RSS / PTE / map count, the CPU parts positive.
Also ABP_CLONE_SELF of S before and after N steps (its frames and CPU: the per-frame CPU of a clone that is not new).
  usage: python abplus_probe_clone_stats.py --out <dir> [--k 4] [--steps 25]
The instance's preload: bridge/tools/libabp_turbo.so of this copy when built there (default_preload)."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from abplus_probe_fork import split_code, sticky_actions
from abplus_probe_native_obs import read_raw
from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance

HERE = Path(__file__).resolve().parent


def steps(c, codes):
    for code in codes:
        m, s, b, i = split_code(code)
        c._send({"cmd": "step", "repeat": 4, "move": m, "shoot": s, "bomb": b, "item": i})
        read_raw(c)


def self_line(c):
    return [int(x) for x in c.lua('return os.getenv("ABP_CLONE_SELF")').split()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='../abplus/catalog/scaling2_groups.json')
    p.add_argument('--group', default='normal')
    p.add_argument('--seed', type=int, default=2147500000)
    p.add_argument('--port', type=int, default=34670)
    p.add_argument('--k', type=int, default=4)
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group=args.group, tasks='', seconds=0.0))
    os.environ.update(FORK_ENV)
    os.environ['ABP_CLONE_STATS'] = str(out)
    pre = default_preload()
    stub = str(Path(pre).parent / 'stub_render_g.txt') if pre else ''
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=pre, al_stopped=True, nice=0,
                   stub_list=stub if stub and Path(stub).is_file() else '')
    print('preload', pre or '(ABP_HOME tools)', flush=True)
    inst = Instance('clonestats', args.port, cfg, spec)
    rng = np.random.default_rng(args.seed)
    K, N = args.k, args.steps
    res = {}
    try:
        inst.reset(args.seed)
        inst.env.bridge.play(sticky_actions(rng, 10), repeat=4, stop_clear=True)
        S = inst.env.bridge.fork(tag='S', lean=True, alarm=600, purpose='hurt')
        S._send({"cmd": "obs"}); read_raw(S)
        s0 = self_line(S)
        steps(S, sticky_actions(rng, N))
        s1 = self_line(S)
        res['S_self'] = dict(frames=s1[0]-s0[0], cpu_ms=(s1[1]-s0[1])/1e6, minflt=s1[2]-s0[2])
        t = {}
        for purpose in ('hurt', 'fatal', 'cf'):
            t0 = time.perf_counter()
            for _ in range(K):
                c = S.fork(lean=True, alarm=300, purpose=purpose)
                c._send({"cmd": "obs"}); read_raw(c)
                steps(c, sticky_actions(rng, N))
                c.close()
            t[purpose] = time.perf_counter()-t0
        res['wall_s'] = t
        ends = S.fork_many([sticky_actions(rng, 10) for _ in range(K)], repeat=4)
        res['fork_many'] = [list(e) if e else None for e in ends]
        steps(S, sticky_actions(rng, 3))   # (S's loop ticks: its last clones reaped)
        time.sleep(0.5)
        steps(S, sticky_actions(rng, 3))
        S.close()
        time.sleep(0.5)
        inst.env.bridge.play(sticky_actions(rng, 5), repeat=4)   # (the root's loop ticks: S reaped)
    finally:
        inst.close()
    files = sorted(out.glob('clone-stats-*.bin'))
    sys.path.insert(0, str(HERE.parent / 'ops'))
    import clone_stats
    tot, meta = clone_stats.summed(files)
    res['tot'] = tot
    (out / 'probe.json').write_text(json.dumps(res, indent=1))
    clone_stats.report(tot, meta)
    print(json.dumps({k: v for k, v in res.items() if k != 'tot'}))
    bad = []
    for purpose in ('hurt', 'fatal', 'cf'):
        d = tot[purpose]
        want_clones = K+(1 if purpose == 'hurt' else 0)   # (S itself is a hurt clone)
        if d['clones'] != want_clones or d['forks'] != want_clones:
            bad.append(f'{purpose}: clones {d["clones"]} forks {d["forks"]} != {want_clones}')
        if d['exits'] != want_clones or d['reaped'] != want_clones:
            bad.append(f'{purpose}: exits {d["exits"]} reaped {d["reaped"]} != {want_clones}')
        if not (d['cpu_start_ns'] > 0 and d['cpu_sim_ns'] > 0 and d['cpu_teardown_ns'] > 0 and d['rss_kb'] > 0 and d['pte_kb'] > 0
                and d['maps_n'] > 0):
            bad.append(f'{purpose}: a part is zero {d}')
    for purpose in ('fatal', 'cf'):
        d = tot[purpose]
        if abs(d['frames']-4*N*K) > 2*K:   # (4 frames per step; a clone's obs command adds none)
            bad.append(f'{purpose}: frames {d["frames"]} vs {4*N*K}')
    fm = tot['fork_many']
    if fm['clones'] != K or fm['exits'] != K:
        bad.append(f'fork_many: clones {fm["clones"]} exits {fm["exits"]} != {K}')
    print('CHECK', 'ok' if not bad else 'FAILED: ' + '; '.join(bad), flush=True)
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
