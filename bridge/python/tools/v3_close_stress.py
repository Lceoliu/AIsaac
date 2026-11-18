"""Close-race stress check (2026-11-18, the coordinator, after A1's mode-0 smoke on c80c9f2 ended with a worker exit 86
at the close): start a TokSampler (floor workers, mode 0 -- every visible / trigger switch at its default), answer its
records with random actions for a while, close it; again and again with the close at varied moments (from before the
first connect is done to well into the episodes), so that some closes find a worker's start-state builder launching /
connecting / querying the bridge's mode. Per round: every worker's exit code, whether close() raised VisibleV3Fatal,
and the fatal files left. Pass: no worker exit 86, no VisibleV3Fatal, no fatal file, in any round.

usage (bridge python dir, PYTHONPATH=.; the CPU runner adds --groups-file / --out):
  python tools/v3_close_stress.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
         [--rounds 36] [--workers 4] [--delays 0.5,1,2,3,5,8,13,20] [--mode floor]
Writes <out>/rounds.jsonl and <out>/summary.json; prints SUMMARY; exit 0 on a pass, 1 otherwise.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
for k in ('ISAAC_RL_VISIBLE_V3', 'ISAAC_RL_NATIVE_VISIBLE', 'ISAAC_RL_TRIGGER_ONCE'):   # (the old arm: mode 0)
    os.environ.pop(k, None)

import numpy as np   # noqa: E402

from goexplore_abplus import default_bridge_lua, default_preload, load_spec   # noqa: E402
from isaac_bridge import visible_producer as VP   # noqa: E402
from isaac_bridge.tok_sampler import TokSampler, TokSamplerConfig   # noqa: E402


def one_round(k, args, spec, out):
    fatal_dir = out / f'fatal-{k}'
    os.environ['ISAAC_RL_FATAL_DIR'] = str(fatal_dir)
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    cfg = TokSamplerConfig(workers=args.workers, specs=[spec], assign=[0] * args.workers, frames_per_decision=4,
                           episodes_per_state=2, mode=args.mode, floor_seconds=60.0, name=f'cs{k % 4}',
                           port=args.port + 40 * (k % 4), seed=1000 + 97 * k, bridge_lua=default_bridge_lua(),
                           preload=default_preload(), stub_list=stub if Path(stub).is_file() else '', nice=0)
    delay = args.delays[k % len(args.delays)]
    rng = np.random.default_rng(k)
    t0 = time.perf_counter()
    sampler = TokSampler(cfg)
    records, raised, during = 0, None, None
    try:
        while time.perf_counter() - t0 < delay and sampler.live:
            idx, slots = sampler.poll(want=1, max_wait=0.0005, idle=0.05)
            for i in idx:   # (move, shoot; the rest 0)
                sampler.actions[i] = 0
                sampler.actions[i, 0], sampler.actions[i, 1] = rng.integers(0, 9), rng.integers(0, 5)
            records += len(idx)
            sampler.reply(idx)
    except VP.VisibleV3Fatal as exc:
        during = str(exc)[:400]
    t_close = time.perf_counter()
    try:
        sampler.close(timeout=60.0)
    except VP.VisibleV3Fatal as exc:
        raised = str(exc)[:400]
    codes = [p.exitcode for p in sampler.procs]
    files = sorted(str(p.name) for p in fatal_dir.glob('fatal-worker-*.txt')) if fatal_dir.is_dir() else []
    return dict(round=k, delay=delay, records=records, close_s=round(time.perf_counter() - t_close, 2),
                exit_codes=codes, fatal_on_close=raised, fatal_during=during, fatal_files=files,
                seconds=round(time.perf_counter() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rounds', type=int, default=36)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--delays', default='0.5,1,2,3,5,8,13,20')
    ap.add_argument('--mode', default='floor', choices=('room', 'floor'))
    ap.add_argument('--port', type=int, default=46100)
    args = ap.parse_args()
    args.delays = [float(v) for v in args.delays.split(',')]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    rows, t0 = [], time.perf_counter()
    with open(out / 'rounds.jsonl', 'w') as f:
        for k in range(args.rounds):
            try:
                r = one_round(k, args, spec, out)
            except Exception as exc:   # (a round that could not run: reported, counted, the next)
                r = dict(round=k, error=f'{type(exc).__name__}: {str(exc)[:300]}')
            rows.append(r)
            f.write(json.dumps(r) + '\n')
            f.flush()
            print(json.dumps(r)[:500], flush=True)
    done = [r for r in rows if 'error' not in r]
    codes = {}
    for r in done:
        for c in r['exit_codes']:
            codes[str(c)] = codes.get(str(c), 0) + 1
    summary = dict(rounds=len(rows), ran=len(done), errors=len(rows) - len(done), workers=args.workers,
                   mode=args.mode, exit_codes=codes, exit_86=codes.get(str(VP.V3_FATAL_EXIT), 0),
                   fatal_on_close=sum(bool(r['fatal_on_close']) for r in done),
                   fatal_during=sum(bool(r['fatal_during']) for r in done),
                   fatal_files=sum(len(r['fatal_files']) for r in done),
                   delays=args.delays, seconds=round(time.perf_counter() - t0, 1))
    ok = (summary['ran'] >= 30 and summary['exit_86'] == 0 and summary['fatal_on_close'] == 0 and
          summary['fatal_during'] == 0 and summary['fatal_files'] == 0)
    summary['ok'] = ok
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)
    print('PASS' if ok else 'FAIL', flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
