"""End-to-end environment throughput of tok_sampler / tok_floor workers (2026-10-04): TokSampler with N workers and a
server that answers every record at once with a uniformly random action (no network; B10's `uniform`). Measures, after
--warm seconds, the logic frames, decisions and episodes over --seconds, the workers' own split (step_s, encode_s,
wait_s per decision) and with --teacher-share the hindsight searches (seconds per search). Runs with whatever code is
next to it, so the same script measures an older copy of the sampler too.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_bench_tok_sampler.py --groups-file ../abplus/catalog/scaling2_groups.json --workers 2 --seconds 60 \
      --out <dir> [--mode floor] [--teacher-share 0.3]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.tok_sampler import TokSampler, TokSamplerConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='../abplus/catalog/scaling2_groups.json')
    p.add_argument('--group', default='normal')
    p.add_argument('--mode', default='room')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--fpd', type=int, default=4)
    p.add_argument('--warm', type=float, default=25.0)
    p.add_argument('--seconds', type=float, default=60.0)
    p.add_argument('--teacher-share', type=float, default=0.0)
    p.add_argument('--episodes-per-state', type=int, default=16)
    p.add_argument('--seed', type=int, default=4242)
    p.add_argument('--port', type=int, default=34500)
    p.add_argument('--name', default='stpbt')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group=args.group, tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    teacher = args.teacher_share > 0
    cfg = TokSamplerConfig(workers=args.workers, specs=[spec], assign=[0] * args.workers,
                           frames_per_decision=args.fpd, episodes_per_state=args.episodes_per_state,
                           start_hp=(6, 6) if args.mode == 'floor' else (2, 6), seed=args.seed, name=args.name,
                           port=args.port, teacher=teacher, teacher_share=args.teacher_share, mode=args.mode,
                           bridge_lua=default_bridge_lua(), preload=default_preload(),
                           stub_list=stub if Path(stub).is_file() else '')
    sampler = TokSampler(cfg)
    rng = np.random.default_rng(args.seed)
    marks = []
    t0 = time.perf_counter()
    try:
        phase = 0
        while True:
            now = time.perf_counter() - t0
            if phase == 0 and now >= args.warm:
                marks.append((time.perf_counter(), sampler.totals(), os.getloadavg()))
                phase = 1
            if phase == 1 and now >= args.warm + args.seconds:
                marks.append((time.perf_counter(), sampler.totals(), os.getloadavg()))
                break
            idx, _ = sampler.poll_ready(idle=0.05)
            if len(idx):
                n = len(idx)
                sampler.actions[idx, 0] = rng.integers(0, 9, n)
                sampler.actions[idx, 1] = rng.integers(0, 5, n)
                sampler.actions[idx, 2] = 0
                sampler.actions[idx, 3] = 0
                sampler.reply(idx)
            if teacher:
                sampler.teacher_records()
    finally:
        sampler.close()
    (ta, a, load_a), (tb, b, load_b) = marks
    dt = tb - ta
    d = {k: b[k] - a[k] for k in b}
    n = max(d['decisions'], 1)
    summary = dict(mode=args.mode, group=args.group, workers=args.workers, fpd=args.fpd, seconds=round(dt, 1),
                   load=(load_a, load_b), frames_per_s=round(d['frames'] / dt, 1),
                   x_real_time=round(d['frames'] / dt / 30, 1), decisions_per_s=round(d['decisions'] / dt, 1),
                   episodes=d['episodes'], forks=d['forks'], errors=d['errors'],
                   outcomes={k: d.get(k, 0) for k in ('wins', 'deaths', 'timeouts', 'empty', 'states', 'extra_episodes')},
                   decisions_per_episode=round(d['decisions'] / max(d['episodes'], 1), 1),
                   step_ms=round(1000 * d['step_s'] / n, 4), encode_ms=round(1000 * d['encode_s'] / n, 4),
                   wait_ms=round(1000 * d['wait_s'] / n, 4),
                   fork_ms_per_episode=round(1000 * d['fork_s'] / max(d['forks'], 1), 2),
                   searches=d.get('searches', 0), teach_records=d.get('teach', 0),
                   s_per_search=round(d.get('teach_s', 0) / max(d.get('searches', 0), 1), 4),
                   teach_share=round(d.get('teach_s', 0) / (dt * args.workers), 4),
                   env={k: os.environ.get(k) for k in ('ISAAC_RL_STEP_LINE', 'ISAAC_RL_NATIVE_STEP', 'ISAAC_RL_FAST_ROW',
                                                       'ISAAC_RL_TERRAIN_CACHE', 'ISAAC_RL_FORK_MANY')})
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
