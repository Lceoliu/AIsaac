"""Teacher v2 sampler probe (2026-10-10): tok_floor's workers with the search thread and teacher v2 on, the lanes and the
V2 rings, without the trainer: a stand-in actor (numpy: random actions with sticky moves and shots, value 0.5) answers
the workers' records and their lanes the way train_tok's actor does (main_value before the reply; the lanes' actions,
value, log-probability). Runs --seconds of wall time, then reports the workers' totals (errors, episodes, the teacher's
and teacher v2's counters), the lanes answered, the imitation records collected (checks: each record's step j and
length n, the weight in (margin, cap], the first record's action under way present, records per point = n) and closes
the sampler. --v2 0: the same run without teacher v2 (the flag-off path of the worker).

usage (bridge python dir, PYTHONPATH=.):
  python abplus_probe_teacher_v2_sampler.py --groups-file ../abplus/catalog/scaling2_groups.json --workers 3 \
      --seconds 240 --out <dir> [--port 44620] [--name tvs] [--v2 1]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.tok_sampler import STATS, TokSampler, TokSamplerConfig, default_stub_list

FLOOR_REWARD = 'damage=1,hurt=0.75,room=1,explore=0.5,boss=2,exit=3,death=2,timeout=1,time=0.002'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--seconds', type=float, default=240.0)
    p.add_argument('--v2', type=int, default=1)
    p.add_argument('--random', type=float, default=120.0, help='teacher_v2_random (points per game hour)')
    p.add_argument('--port', type=int, default=44620)
    p.add_argument('--name', default='tvs')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    cfg = TokSamplerConfig(workers=args.workers, specs=[spec], assign=[0] * args.workers, frames_per_decision=4,
                           mode='run', seed=4242, name=args.name, port=args.port, teacher=True, teacher_share=0.5,
                           teacher_thread=1, teacher_death_depths=(2, 4, 8, 16, 32), teacher_death_margin=15,
                           teacher_v2=bool(args.v2), teacher_v2_random=args.random, branch_reward=FLOOR_REWARD,
                           run_seconds=300.0, bridge_lua=default_bridge_lua(), preload=default_preload(),
                           stub_list=default_stub_list(default_preload()))
    sampler = TokSampler(cfg)
    n = args.workers
    rng = np.random.default_rng(3)
    move, shoot = np.zeros(n, np.int64), np.ones(n, np.int64)
    lanes, lane_rows_seen, recs = 0, 0, []
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds and sampler.live:
            if sampler.lane_ready:   # the lanes, as train_tok's serve_lanes
                idx = sampler.take_lanes()
                k = len(idx)
                sampler.lane_io[idx, 5] = rng.integers(0, 9, k)
                sampler.lane_io[idx, 6] = rng.integers(1, 5, k)
                sampler.lane_io[idx, 7:10] = 0
                sampler.lane_out[idx, 0], sampler.lane_out[idx, 1], sampler.lane_out[idx, 2] = 0.5, -2.0, 0
                sampler.reply_lanes(idx)
                lanes += 1
                lane_rows_seen += k
            idx, slots = sampler.poll_ready(idle=0.05)
            if not len(idx):
                continue
            ch = rng.random(len(idx)) < 1 / 6
            move[idx[ch]] = rng.integers(0, 9, int(ch.sum()))
            shoot[idx[ch]] = rng.integers(1, 5, int(ch.sum()))
            sampler.actions[idx, 0], sampler.actions[idx, 1], sampler.actions[idx, 2] = move[idx], shoot[idx], 0
            if sampler.main_value is not None:
                sampler.main_value[idx] = 0.5
            sampler.reply(idx)
            got = sampler.v2_records()
            if got is not None and len(got[0]):
                recs.append(got)
        got = sampler.v2_records()
        if got is not None and len(got[0]):
            recs.append(got)
        tot = dict(zip(STATS, sampler.stats.sum(axis=0).tolist()))
    finally:
        sampler.close()
    from isaac_bridge.tok_teacher2 import VI
    rep = dict(seconds=round(time.time() - t0, 1), lanes_calls=lanes, lane_rows=lane_rows_seen,
               **{k: tot[k] for k in ('errors', 'episodes', 'deaths', 'decisions', 'hurts', 'searches', 'teach',
                                      'death_searches', 'teach_s')},
               **{k: tot[k] for k in STATS if k.startswith('v2_')})
    if recs:
        meta = np.concatenate([m for _, m in recs])
        rows = np.concatenate([r for r, _ in recs])
        firsts = meta[:, VI['j']] == 0
        pts = np.unique(meta[:, VI['point']])
        per_point = [int((meta[:, VI['point']] == q).sum()) for q in pts]
        n_of = [int(meta[meta[:, VI['point']] == q, VI['n']][0]) for q in pts]
        rep.update(records=len(meta), points_with_records=len(pts), firsts=int(firsts.sum()),
                   records_per_point_ok=per_point == n_of,
                   weight_ok=bool(np.all((meta[:, VI['weight']] > cfg.teacher_v2_margin) &
                                         (meta[:, VI['weight']] <= cfg.teacher_v2_cap))),
                   j_lt_n=bool(np.all(meta[:, VI['j']] < meta[:, VI['n']])),
                   t_consistent=bool(np.all(rows['t'] >= 1)),
                   value_seen=float(meta[:, VI['value']].mean()), mean_gain=float(meta[firsts, VI['gain']].mean()),
                   kinds={int(k): int((meta[firsts, VI['kind']] == k).sum()) for k in (1, 2, 3)})
    (out / 'summary.json').write_text(json.dumps(rep, indent=1))
    print('SUMMARY', json.dumps(rep), flush=True)


if __name__ == '__main__':
    main()
