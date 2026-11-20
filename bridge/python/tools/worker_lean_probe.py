"""(2026-11-20, the memory ledger: ISAAC_RL_WORKER_LEAN) Real floor workers (their games, the training's teacher config:
death teaching, the search thread and its forks) played with random actions on the CPU, the switch off or on: each
worker's private memory and whether it maps libtorch, sampled while they play; the records; the workers' torch_held
count; any fatal (with the switch on, a worker that confirms torch at its start ends the run). No learner, no GPU,
nothing written but <out>.

usage (bridge python dir, PYTHONPATH=.; the CPU runner adds --groups-file / --out):
  python tools/worker_lean_probe.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir> --lean 0|1
         [--workers 2] [--seconds 120] [--port 46600]
Prints and writes <out>/lean-<0|1>.json."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())


def worker_memory(pid):
    """(private MB, maps libtorch) of a live process, from /proc (None, None when it is gone)."""
    try:
        priv = 0
        for line in open(f'/proc/{pid}/smaps_rollup'):
            if line.startswith(('Private_Clean', 'Private_Dirty')):
                priv += int(line.split()[1])
        maps = open(f'/proc/{pid}/maps').read()
        return round(priv / 1024.0, 1), ('libtorch' in maps)
    except OSError:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lean', type=int, required=True, choices=(0, 1))
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seconds', type=float, default=120.0)
    ap.add_argument('--port', type=int, default=46600)
    args = ap.parse_args()
    os.environ['ISAAC_RL_WORKER_LEAN'] = str(args.lean)   # (before the bridge modules are imported: theirs and the
    out = Path(args.out)                                    #  workers' environment)
    out.mkdir(parents=True, exist_ok=True)
    os.environ['ISAAC_RL_FATAL_DIR'] = str(out / f'fatal-{args.lean}')
    import numpy as np
    from goexplore_abplus import default_bridge_lua, default_preload, load_spec
    from isaac_bridge import tok_obs as TO
    from isaac_bridge import visible_producer as VP
    from isaac_bridge.tok_sampler import TokSampler, TokSamplerConfig
    assert TO.WORKER_LEAN == bool(args.lean)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    cfg = TokSamplerConfig(workers=args.workers, specs=[spec], assign=[0] * args.workers, frames_per_decision=4,
                           episodes_per_state=2, mode='floor', floor_seconds=20.0, name=f'wl{args.lean}',
                           port=args.port + 40 * args.lean, seed=4242, bridge_lua=default_bridge_lua(),
                           preload=default_preload(), stub_list=stub if Path(stub).is_file() else '', nice=0,
                           teacher=True, teacher_share=0.5, teacher_death_depths=(2, 4, 8), teacher_thread=True,
                           teacher_fork=True)
    rng = np.random.default_rng(7)
    t0 = time.perf_counter()
    res = dict(lean=args.lean, workers=args.workers, records=0, fatal=None, samples=[])
    sampler = TokSampler(cfg)
    try:
        next_sample = time.perf_counter() + 15.0
        while time.perf_counter() - t0 < args.seconds and sampler.live:
            try:
                idx, _ = sampler.poll(want=1, max_wait=0.0005, idle=0.02)
                sampler.check_fatal()
            except VP.VisibleV3Fatal as exc:
                res['fatal'] = str(exc)[:600]
                break
            for i in idx:
                sampler.actions[i] = 0
                sampler.actions[i, 0], sampler.actions[i, 1] = rng.integers(0, 9), rng.integers(0, 5)
            res['records'] += len(idx)
            sampler.reply(idx)
            if time.perf_counter() >= next_sample:
                next_sample += 15.0
                res['samples'].append([worker_memory(p.pid) for p in sampler.procs])
        res['teach_records'] = int(sampler.stats[:, sampler_st('teach')].sum())
        res['torch_held'] = int(sampler.stats[:, sampler_st('torch_held')].sum())
    finally:
        try:
            sampler.close(timeout=20.0)
        except VP.VisibleV3Fatal as exc:
            res['fatal'] = res['fatal'] or str(exc)[:600]
    last = [s for s in res['samples'] if all(m[0] is not None for m in s)]
    if last:
        res['private_mb_last'] = [m[0] for m in last[-1]]
        res['libtorch_mapped'] = [m[1] for m in last[-1]]
    res['seconds'] = round(time.perf_counter() - t0, 1)
    (out / f'lean-{args.lean}.json').write_text(json.dumps(res, indent=1))
    print('LEAN_RESULT ' + json.dumps({k: v for k, v in res.items() if k != 'samples'}), flush=True)


def sampler_st(name):
    from isaac_bridge.tok_sampler import ST
    return ST[name]


if __name__ == '__main__':
    main()
