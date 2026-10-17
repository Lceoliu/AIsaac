"""B38 step 3a, a measurement aid: replay one dumped learner job (train_tok ISAAC_RL_B38_DUMP) to the learner
process(es) of train_tok --learner-process 1, without an actor, a sampler or games, so that one learner configuration and another
can be timed (and compared) on the same job. This script plays the actor's process: it puts the job's host arrays in
shared memory (train_tok.SharedArrays), its temporal rows in shared CPU tensors (no actor GPU: the learner copies
them host -> device), keeps a CPU weights buffer for the learner to publish into, spawns the learner
(train_tok.learner_child_entry with bench=True) and sends the same job --reps times, waiting for each update's
announcement. The learner processes write progress.csv and last.pt into --out as in a run.

usage: python abplus_b38_replay.py --job <dir>/job.pt --reps 8 -- <train_tok arguments, incl. --learner-process 1
       --learner-device cuda:0 --out <new dir> and the run's --workers / --rollout / model flags>
Prints the wall seconds per replayed update and, from progress.csv, the medians of update_s and its parts."""
import argparse
import csv
import os
import queue
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np


def main():
    if '--' not in sys.argv:
        raise SystemExit(__doc__)
    cut = sys.argv.index('--')
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--reps', type=int, default=8)
    a = p.parse_args(sys.argv[1:cut])
    targs = sys.argv[cut + 1:]
    import torch
    import train_tok as TT
    from isaac_bridge import tok_policy as TP
    d = torch.load(a.job, map_location='cpu', weights_only=False)
    scal, arrays, tens, meta = d['scal'], d['arrays'], d['tensors'], d['meta']
    n, cap = scal['n'], scal['cap']
    sa = TT.SharedArrays()
    obj = types.SimpleNamespace()
    for k, v in arrays.items():   # (full shapes: per worker capacity, the t_* / rooms_* prefixes)
        shape = list(v.shape)
        if v.ndim >= 2 and v.shape[0] == n and k.startswith('rooms_') and k != 'rooms_n':
            shape[1] = scal['tR'] + cap
        elif v.ndim >= 2 and v.shape[0] == n and k.startswith('t_'):
            shape[1] = scal['tP'] + cap
        elif v.ndim >= 2 and v.shape[0] == n:
            shape[1] = cap
        full = sa(tuple(shape), v.dtype)
        if k in ('t_row',):
            full[...] = -1
        elif k == 't_ep':
            full[...] = -2
        elif k == 'rooms_ep':
            full[...] = -3
        full[tuple(slice(0, s_) for s_ in v.shape)] = v
        setattr(obj, k, full)
    tdev = None
    if tens is not None:
        kv = torch.zeros((tens['rows'],) + tuple(tens['kv'].shape[1:]), dtype=tens['kv'].dtype)
        kv[:len(tens['kv'])] = tens['kv']
        o = torch.zeros((tens['orows'],) + tuple(tens['o'].shape[1:]), dtype=tens['o'].dtype)
        o[:len(tens['o'])] = tens['o']
        tdev = (kv.share_memory_(), o.share_memory_())
    # the weights buffer: the size of the learner's parameters and buffers (its checkpoint's model)
    ck_ = targs[targs.index('--resume') + 1]
    model = TP.TokPolicy(**torch.load(ck_, map_location='cpu', weights_only=False)['config'])
    sizes = [t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers())]
    fa = torch.empty(sum(sizes), dtype=torch.uint8).share_memory_()
    world = 1   # (this branch: one learner process; the B38 step-3 branch replays several ranks)
    mpc = torch.multiprocessing.get_context('spawn')
    jobs = [mpc.Queue()]
    results, fresh = mpc.Queue(), mpc.Event()
    rolls = {0: dict(arrays=sa.describe(obj), scal=scal, tdev=tdev)}
    proc = mpc.Process(target=TT.learner_child_entry, name='replay', daemon=True, args=(
        targs, dict(pid=os.getpid(), stats=np.asarray(meta['stats']), v2_on=False, rolls=rolls, fa=fa,
                    sizes=[s_ for s_ in sizes], jobs=jobs[0], results=results, fresh=fresh, bench=True)))
    procs = types.SimpleNamespace(start=proc.start, is_alive=proc.is_alive, join=proc.join,
                                  terminate=lambda: proc.is_alive() and proc.terminate())
    procs.start()
    out = Path(targs[targs.index('--out') + 1])
    walls = []
    t_start = time.perf_counter()
    try:
        for rep in range(a.reps):
            job = dict(meta, roll=None, bid=0, rscal=scal, t_start=t_start, t_put=time.perf_counter())
            t0 = time.perf_counter()
            for q in jobs:
                q.put(job)
            while not fresh.wait(1.0):
                if not procs.is_alive():
                    raise SystemExit('a learner process ended')
            fresh.clear()
            kind, info = results.get(timeout=60)
            if kind == 'error':
                raise SystemExit('learner error:\n' + info)
            walls.append(time.perf_counter() - t0)
            print(f'replay {rep + 1}/{a.reps}: {walls[-1]:.3f} s (update {info["update"]})', flush=True)
    finally:
        for q in jobs:
            q.put(None)
        t_lim = time.perf_counter() + 300
        while time.perf_counter() < t_lim:
            try:
                kind, info = results.get(timeout=1.0)
            except queue.Empty:
                if not procs.is_alive():
                    break
                continue
            if kind == 'done':
                break
        procs.join(60)
        procs.terminate()
    rows = list(csv.DictReader(open(out / 'progress.csv')))[1:]   # (the first: allocation and warm-up)
    cols = ['update_s', 'u_recompute', 'u_ppo', 'u_opt', 'u_gae', 'u_batch', 'u_temporal', 'gpu_reserved_gb',
            'handoff_gpu_ms', 'c_learner_cores', 'kl', 'clip_frac']
    med = {c: statistics.median(float(r[c]) for r in rows) for c in cols if rows and rows[0].get(c) not in (None, '')}
    print(f'ranks {world}: wall per update (median of {len(walls) - 1} after the first) '
          f'{statistics.median(walls[1:]) if len(walls) > 1 else walls[0]:.3f} s | '
          + ' '.join(f'{k}={v:.4g}' for k, v in med.items()), flush=True)


if __name__ == '__main__':
    main()
