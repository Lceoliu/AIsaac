"""Death teacher probe (2026-10-09, SCALING_THESIS.md S5): random-action episodes from a floor's start until the player
dies; the fatal hurt is then searched with tok_sampler.hindsight at the death depths (2, 4, 8, 16, 32 decisions before
the fatal step, margin 15) exactly as tok_floor's search does for a fatal hurt (restore = the floor's start clone,
the whole episode replayed), and reports per depth the nine held moves' outcomes (half hearts lost, +100 dead), whether
a safe move exists, and the seconds. The same search at the ordinary depths (2, 4, 8, margin 4) for comparison.

usage (bridge python dir, PYTHONPATH=.):
  python abplus_probe_death_teacher.py --groups-file ../abplus/catalog/scaling2_groups.json --seeds 2147600000:4 \
      --out <dir>
"""
import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.tok_sampler import TokSamplerConfig, hindsight, lean_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147600000:4')
    p.add_argument('--max-decisions', type=int, default=900)
    p.add_argument('--death-depths', default='2,4,8,16,32')
    p.add_argument('--death-margin', type=int, default=15)
    p.add_argument('--port', type=int, default=44500)
    p.add_argument('--name', default='dth')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    cfg = TokSamplerConfig(workers=1, specs=[spec], assign=[0], frames_per_decision=4, mode='run', teacher=True)
    cfg_death = dataclasses.replace(cfg, teacher_depths=tuple(int(v) for v in args.death_depths.split(',')),
                                    teacher_margin=args.death_margin)
    first, count = (int(v) for v in args.seeds.split(':'))
    inst = Instance(args.name, args.port, gx, spec)
    rng = np.random.default_rng(23)
    report = []
    try:
        for seed in range(first, first + count):
            inst.env.bridge.reset_mode = 'floor'
            inst.reset(seed)
            S = inst.env.bridge.fork(tag='template', alarm=0)
            # a random episode until the player dies (sticky random moves, always shooting somewhere)
            reseed = seed % 1000 + 1
            E = S.fork(lean=True, alarm=600, reseed=reseed)
            dec = LeanDecoder()
            E._send({'cmd': 'obs'})
            obs = read_lean(E, dec)
            applied, hurt_at, t = [], None, 0
            move, shoot = 0, 1
            while t < args.max_decisions and not obs.dead:
                if t % 6 == 0:
                    move, shoot = int(rng.integers(0, 9)), int(rng.integers(1, 5))
                action = (move, shoot, 0)
                applied.append(action)
                obs = lean_step(E, dec, action, 4)
                t += 1
                if obs.dead:
                    hurt_at = t
            E.close()
            rec = dict(seed=seed, decisions=t, died=bool(obs.dead))
            if hurt_at is not None:
                for name, c in (('death', cfg_death), ('ordinary', cfg)):
                    t0 = time.perf_counter()
                    found = hindsight(S, reseed, applied, hurt_at, c, limit=10 ** 6, offset=0)
                    rec[name] = dict(seconds=round(time.perf_counter() - t0, 2),
                                     depths=[dict(k=k, lost=[round(float(v), 2) for v in lost],
                                                  safe=[m for m in range(9) if lost[m] == 0], took=int(took))
                                             for k, at, under_way, lost, took in found],
                                     avoidable=bool(found) and min(found[-1][3]) == 0)
            S.close()
            report.append(rec)
            print(json.dumps(rec), flush=True)
    finally:
        inst.kill()
    died = [r for r in report if r['died']]
    summary = dict(episodes=len(report), deaths=len(died),
                   death_avoidable=sum(r['death']['avoidable'] for r in died),
                   ordinary_avoidable=sum(r['ordinary']['avoidable'] for r in died),
                   death_seconds=round(sum(r['death']['seconds'] for r in died), 1),
                   ordinary_seconds=round(sum(r['ordinary']['seconds'] for r in died), 1),
                   death_depths_tried=[len(r['death']['depths']) for r in died])
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    (out / 'episodes.json').write_text(json.dumps(report, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
