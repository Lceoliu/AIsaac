"""P package, phase 0 (2026-11-19, the coordinator): the actor's GraphActor calls with --actor-fold 0 (seq1's setting)
against --actor-fold 1 (every per-call operation inside the graph: two H2D copies, the replay, one D2H copy), on real
records and the model of a train_tok --dump-rollouts directory (D1's: seq1's state), on one GPU.

Calls as the training loop makes them: records of one decision index t of the rollout across a random subset of the
workers (--rows records a call, seq1: 7.8 on average), each worker its own temporal slot, its pending actions, the
episode starts from the records' 'first'; the K / V, s and o written into rollout-shaped tensors (tdest).
Per mode, the median over --calls calls of:
  host_submit_ms   submit()'s host time (rows_counts, staging, the enqueued operations),
  wait_ms          result()'s wait after submit (the GPU work left + the D2H copy),
  call_ms          submit -> result, one call in flight,
  gpu_ms           CUDA events around the call on the actor's stream (the stream's busy span),
  replay_ms        the call's graph replayed alone (CUDA events: the network's GPU time),
and pipelined_calls_per_s: two GraphActors in turn (train_tok --actor-slots 2), submit of one while the other runs.
Prints one JSON object (and --out <file>).

   usage: python tools/actor_fold_speed.py <rollouts dir> [--rows 8] [--calls 400] [--out f.json]"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def calls_of(z, n_calls, rows, rng):
    """[(worker indices, t)] -- each call a random subset of the workers that have record t."""
    length = z['length']
    W = len(length)
    out, t = [], 0
    while len(out) < n_calls:
        have = np.flatnonzero(length > t)
        if not len(have):
            t = 0
            continue
        k = min(len(have), max(1, int(rng.poisson(rows - 1)) + 1))
        out.append((np.sort(rng.choice(have, k, replace=False)), t))
        t += 1
    return out, W


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('rollouts')
    ap.add_argument('--rows', type=float, default=7.8, help='records a call on average (seq1: 7.8)')
    ap.add_argument('--calls', type=int, default=400)
    ap.add_argument('--modes', default='0,1,1:8', help='fold[:row buckets] per mode')
    ap.add_argument('--fp32', action='store_true', help='graphs in float32 (default bf16, as --actor-bf16 1)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    import torch
    from isaac_bridge import tok_policy as TP
    dev = torch.device('cuda')
    paths = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.npz')))
    pts = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.pt')))
    ck = torch.load(pts[0], map_location='cpu', weights_only=False)
    model = TP.TokPolicy(**ck['config']).to(dev).eval()
    model.load_state_dict(ck['model'])
    z = np.load(paths[0], allow_pickle=False)
    rows_all, pend_all = z['rows'], z['pending'].astype(np.int64)
    calls, W = calls_of(z, a.calls, a.rows, np.random.default_rng(1))
    ta = model.tatt
    R = sum(len(w) for w, _ in calls) + W + 8
    out = dict(rollout=os.path.basename(paths[0]), workers=W, calls=len(calls),
               rows_mean=float(np.mean([len(w) for w, _ in calls])), gpu=torch.cuda.get_device_name(0),
               bf16=not a.fp32, config={k: ck['config'].get(k) for k in ('frame', 'temporal', 'memory', 'width')})
    for spec in a.modes.split(','):
        fold, _, rb = spec.partition(':')
        fold = bool(int(fold))
        row_sizes = [int(x) for x in rb.split('+') if x] or None
        res = {}
        ts = TP.TemporalState(model, W, dev, torch.bfloat16)
        td = dict(kv=torch.zeros((R, ta.L, 2, ta.D), dtype=ts.dtype, device=dev),
                  s=torch.zeros((R, ta.D), dtype=ts.dtype, device=dev),
                  o=torch.zeros((R, ta.D), device=dev), obase=0)
        st = torch.cuda.Stream(priority=-5)
        mem = int(getattr(model, 'memory', 0) or 0)
        table = torch.zeros((W + 1, mem), device=dev) if mem else None
        actors = [TP.GraphActor(model, W, 32, stream=st, crn=False, memory=table, temporal=ts, bf16=not a.fp32,
                                fold=fold, row_sizes=row_sizes) for _ in range(2)]
        torch.cuda.synchronize()
        used = [0]

        def args_of(i):
            w, t = calls[i % len(calls)]
            sub = np.ascontiguousarray(rows_all[w, t])
            k = len(w)
            dest = dict(td, rows=np.arange(used[0], used[0] + k, dtype=np.int64) % (R - 1))
            used[0] += k
            return sub, np.ascontiguousarray(pend_all[w, t]), w.astype(np.int64), sub['first'] != 0, dest

        # warm-up: every graph the calls reach
        for i in range(min(len(calls), 60)):
            sub, p, w, first, dest = args_of(i)
            actors[i % 2].submit(sub, p, w, first, tdest=dest)
            actors[i % 2].result()
        torch.cuda.synchronize()
        hs, ws, cs, gs = [], [], [], []
        g = actors[0]
        for i in range(len(calls)):
            sub, p, w, first, dest = args_of(i)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(st)
            t0 = time.perf_counter()
            ok = g.submit(sub, p, w, first, tdest=dest)
            t1 = time.perf_counter()
            assert ok, 'a call did not fit'
            e1.record(st)
            g.result()
            t2 = time.perf_counter()
            e1.synchronize()
            hs.append((t1 - t0) * 1e3)
            ws.append((t2 - t1) * 1e3)
            cs.append((t2 - t0) * 1e3)
            gs.append(e0.elapsed_time(e1))
        res.update(host_submit_ms=float(np.median(hs)), wait_ms=float(np.median(ws)), call_ms=float(np.median(cs)),
                   gpu_ms=float(np.median(gs)), host_submit_p90_ms=float(np.percentile(hs, 90)),
                   call_p90_ms=float(np.percentile(cs, 90)))
        # the network's GPU time: the graph of the most common bucket replayed alone
        rp = []
        for gr in g.graphs[:len(g.sizes)]:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(st):
                for _ in range(3):
                    gr.replay()
                e0.record(st)
                for _ in range(50):
                    gr.replay()
                e1.record(st)
            e1.synchronize()
            rp.append(e0.elapsed_time(e1) / 50)
        res['replay_ms_by_e_bucket'] = dict(zip([str(s) for s in g.sizes], rp))
        # pipelined: two actors in turn, as --actor-slots 2 (the next call submitted while the other runs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        busy = []
        for i in range(len(calls)):
            g = actors[i % 2]
            if len(busy) == 2:
                busy.pop(0).result()
            sub, p, w, first, dest = args_of(i)
            g.submit(sub, p, w, first, tdest=dest)
            busy.append(g)
        for g in busy:
            g.result()
        torch.cuda.synchronize()
        res['pipelined_calls_per_s'] = len(calls) / (time.perf_counter() - t0)
        out[f'fold{int(fold)}' + (f'_rows{"+".join(map(str, row_sizes))}' if row_sizes else '')] = res
        actors = table = None
        torch.cuda.empty_cache()
    s = json.dumps(out, indent=1)
    print(s)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(s)


if __name__ == '__main__':
    main()
