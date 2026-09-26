"""Export generated AB+ floors for offline training (see isaac_macro/dataset.py for the fields).

Each worker generates its own runs from its own seed and writes one shard:
    <out>/floors-<w>.jsonl.gz   one JSON record per floor
    <out>/floors-<w>.npz        13x13 grids per floor (room index, type, shape, depth, doors)
usage: python tools/export_floors.py --out DIR --runs 10000 [--workers 8] [--seed 0] [--last-stage 8]
                                     [--debug-start] [--no-jsonl] [--no-npz]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, __file__.rsplit('tools', 1)[0])


def work(args):
    w, runs, seed, out, last_stage, debug_start, jsonl, npz = args
    from isaac_macro.dataset import export
    from isaac_macro.roomconfig import default_room_config
    t = time.time()
    n = export(default_room_config(), os.path.join(out, f'floors-{w:02d}.jsonl.gz') if jsonl else None,
               os.path.join(out, f'floors-{w:02d}.npz') if npz else None, runs, seed, last_stage, debug_start)
    return w, runs, n, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--last-stage', type=int, default=8)
    ap.add_argument('--debug-start', action='store_true', help='Basement I type 0, as the RL instances')
    ap.add_argument('--no-jsonl', action='store_true')
    ap.add_argument('--no-npz', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    per = [a.runs // a.workers + (w < a.runs % a.workers) for w in range(a.workers)]
    jobs = [(w, per[w], a.seed * 1000 + w, a.out, a.last_stage, a.debug_start, not a.no_jsonl, not a.no_npz)
            for w in range(a.workers) if per[w]]
    t = time.time()
    with mp.Pool(len(jobs)) as pool:
        results = pool.map(work, jobs)
    floors = sum(r[2] for r in results)
    meta = dict(runs=a.runs, floors=floors, workers=len(jobs), seed=a.seed, last_stage=a.last_stage,
                debug_start=a.debug_start, seconds=round(time.time() - t, 1),
                shards=[dict(worker=w, runs=r, floors=n, seconds=round(s, 1)) for w, r, n, s in results])
    with open(os.path.join(a.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f"{floors} floors from {a.runs} runs in {meta['seconds']} s "
          f"({floors / max(meta['seconds'], 1e-9):.0f} floors/s, {len(jobs)} workers) -> {a.out}")


if __name__ == '__main__':
    main()
