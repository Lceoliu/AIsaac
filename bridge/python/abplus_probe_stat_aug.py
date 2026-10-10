"""Stat augmentation probe (2026-10-08, user: random multipliers of the base stats for generalisation): in a floor-mode
start state's lean clone, tok_floor.stat_offsets(multipliers) -> AbpSetStats (abp_bridge.lua MC_EVALUATE_CACHE) ->
the player's lean fields; per draw: the multipliers, the offsets, the stats the bridge reports, the stats the next lean
observation shows, and the expected ones (abplus.expected_stats); then --steps decisions of sticky random moves and
shots to see the clone plays on (no error, the stats stay). Also: a clone forked from an augmented clone (what an
archive entry would be) keeps the stats, and a fresh clone of the start state has the base stats (the offsets live in
the clone's Lua state only).

usage (bridge python dir, PYTHONPATH=.):
  python abplus_probe_stat_aug.py --groups-file ../abplus/catalog/scaling2_groups.json --seeds 2147600000:2 --draws 6 \
      --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV, expected_stats, lua_stats
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.tok_floor import STAT_AUG_KEYS, parse_stat_aug, stat_offsets
from isaac_bridge.tok_sampler import lean_step

STAT_FIELDS = ('speed', 'damage', 'shot_speed', 'fire_delay_max', 'range')


def stats_of(obs):
    pl = obs.players[0]
    return [round(float(pl[k]), 4) for k in STAT_FIELDS]


def play(clone, decoder, steps, rng):
    move, shoot = 0, 0
    for t in range(steps):
        if t % 8 == 0:
            move, shoot = int(rng.integers(0, 9)), int(rng.integers(0, 5))
        obs = lean_step(clone, decoder, (move, shoot, 0, 0, 0), 4)
        if obs.dead:
            break
    return obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147600000:2')
    p.add_argument('--draws', type=int, default=6)
    p.add_argument('--stat-aug', default='')
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--port', type=int, default=44200)
    p.add_argument('--name', default='staug')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    ranges = parse_stat_aug(args.stat_aug)
    first, count = (int(v) for v in args.seeds.split(':'))
    inst = Instance(args.name, args.port, gx, spec)
    rng = np.random.default_rng(17)
    report = []
    try:
        for seed in range(first, first + count):
            inst.env.bridge.reset_mode = 'floor'
            inst.reset(seed)
            S = inst.env.bridge.fork(tag='template', alarm=0)
            t0 = time.perf_counter()
            for d in range(args.draws):
                mult = {k: float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) for k, (lo, hi) in ranges.items()}
                offs = stat_offsets(mult)
                C = S.fork(lean=True, alarm=600, reseed=seed % 1000 + d + 1)
                dec = LeanDecoder()
                C._send({'cmd': 'obs'})
                before = stats_of(read_lean(C, dec))
                reported = [round(float(v), 4) for v in C.lua('return AbpSetStats(%s)' % lua_stats(offs)).split(',')]
                C._send({'cmd': 'obs'})
                after = stats_of(read_lean(C, dec))
                exp = [round(float(v), 4) for v in expected_stats(offs)]
                obs = play(C, dec, args.steps, rng)
                played = stats_of(obs)
                # an archive-like clone of the augmented clone keeps the stats; a fresh clone of S has the base ones
                A = C.fork(lean=True, alarm=600, reseed=seed % 1000 + 50)
                da = LeanDecoder()
                A._send({'cmd': 'obs'})
                archived = stats_of(read_lean(A, da))
                A.close()
                F = S.fork(lean=True, alarm=600, reseed=seed % 1000 + 60)
                df = LeanDecoder()
                F._send({'cmd': 'obs'})
                fresh = stats_of(read_lean(F, df))
                F.close()
                C.close()
                rec = dict(seed=seed, draw=d, mult={k: round(v, 4) for k, v in mult.items()},
                           offsets=[round(v, 4) for v in offs], before=before, reported=reported, after=after,
                           expected=exp, played=played, dead=bool(obs.dead), archived=archived, fresh=fresh,
                           ok_expected=all(abs(a - e) < 1e-3 for a, e in zip(after, exp)),
                           ok_kept=played == after and archived == after, ok_fresh=fresh == before)
                report.append(rec)
                print(json.dumps(rec), flush=True)
            S.close()
            print('seed', seed, 'draws', args.draws, 'seconds', round(time.perf_counter() - t0, 1), flush=True)
    finally:
        inst.kill()
    summary = dict(draws=len(report), expected_ok=sum(r['ok_expected'] for r in report),
                   kept_ok=sum(r['ok_kept'] for r in report), fresh_ok=sum(r['ok_fresh'] for r in report),
                   deaths=sum(r['dead'] for r in report), keys=list(STAT_AUG_KEYS))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    (out / 'draws.json').write_text(json.dumps(report, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
