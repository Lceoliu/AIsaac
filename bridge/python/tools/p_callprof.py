"""P package (2026-11-20, the coordinator's breakdown of the sampling side): what one actor call costs, through the
production middle (isaac_bridge.tok_driver.Middle: the C++ check, staging, three H2D copies, the graph replay, the D2H
copy, the K / V / s / o rows into the output ring, the stream synchronised) on GraphActor(fold=True)'s graphs, with
real records and the model of a train_tok --dump-rollouts directory (D1's), one GPU.

For each row count k (--rows, e.g. 1,4,8,16): --calls calls of exactly k records (random subsets of the workers,
consecutive decision indices, each worker its own temporal slot), the median of
  wall_ms      Middle.call's wall time (what the driver thread waits per call),
  gpu_ms       CUDA events around the call on its stream (the stream's busy span),
  replay_ms    the call's graph replayed alone, back to back (the network's GPU time without the host);
and at k = --profile, a torch.profiler trace over --prof-calls calls: kernels per call, the kernels' summed device time
per call (the GPU computing), H2D / D2H copies per call and their time, the busy share (kernel time / wall time), the
kernels by time and by count, grouped by name.
--buckets: the graphs' row buckets (train_tok --actor-row-buckets; '' = off: every call runs the 16-row graph).
--compile: GraphActor's compile mode (torch.compile of the step; '' = none).

   usage: python tools/p_callprof.py <rollouts dir> [--rows 1,4,8,16] [--calls 300] [--buckets ''] [--profile 8]
          [--prof-calls 50] [--compile ''] [--out f.json]"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def category(name):
    n = name.lower()
    for key, pats in (('gemm', ('gemm', 'cutlass', 'sm80_xmma', 'sm86_xmma', 'ampere_', 'cublas', 'matmul')),
                      ('attention', ('flash', 'fmha', 'attention', 'sdpa', 'efficient_attention')),
                      ('softmax', ('softmax',)), ('norm', ('layer_norm', 'layernorm', 'rms', 'norm')),
                      ('reduce', ('reduce', 'sum_', 'max_', 'argmax', 'cumsum', 'scan')),
                      ('index/gather', ('index', 'gather', 'scatter', 'take', 'masked')),
                      ('copy/cast', ('copy', 'cast', 'convert', 'fill', 'memset')),
                      ('elementwise', ('elementwise', 'vectorized', 'unrolled', 'pointwise', 'triton'))):
        if any(p in n for p in pats):
            return key
    return 'other'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('rollouts')
    ap.add_argument('--rows', default='1,4,8,16')
    ap.add_argument('--calls', type=int, default=300)
    ap.add_argument('--buckets', default='')
    ap.add_argument('--profile', type=int, default=8)
    ap.add_argument('--prof-calls', type=int, default=50)
    ap.add_argument('--compile', default='')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    import torch
    from isaac_bridge import tok_policy as TP
    from isaac_bridge.tok_driver import Middle
    dev = torch.device('cuda')
    paths = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.npz')))
    pts = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.pt')))
    ck = torch.load(pts[0], map_location='cpu', weights_only=False)
    model = TP.TokPolicy(**ck['config']).to(dev).eval()
    model.load_state_dict(ck['model'])
    zf = np.load(paths[0], allow_pickle=False)
    length = zf['length']
    W = len(length)
    T = int(min(length.max(), 400))
    rows_all = np.ascontiguousarray(zf['rows'][:, :T])
    pend_all = zf['pending'][:, :T].astype(np.int64)
    del zf
    ta = model.tatt
    rs = [int(x) for x in a.buckets.split(',') if x.strip()] or None
    ts = TP.TemporalState(model, W, dev, torch.bfloat16)
    K = 8192
    ring = dict(kv=torch.zeros((K, ta.L, 2, ta.D), dtype=ts.dtype, device=dev),
                s=torch.zeros((K, ta.D), dtype=ts.dtype, device=dev), o=torch.zeros((K, ta.D), device=dev), obase=0)
    st = torch.cuda.Stream()
    t_c = time.perf_counter()
    ga = TP.GraphActor(model, W, 32, stream=st, crn=False, memory=None, temporal=ts, bf16=True, fold=True,
                       row_sizes=rs, compile_mode=a.compile or None)
    mid = Middle(ga, ring=ring)
    ts.reset_()
    torch.cuda.synchronize()
    out = dict(rollout=os.path.basename(paths[0]), workers=W, gpu=torch.cuda.get_device_name(0), buckets=rs,
               compile=a.compile or None, graphs=len(ga.graphs), e_buckets=list(map(int, ga.sizes)),
               row_buckets=list(map(int, ga.rsizes)), build_s=time.perf_counter() - t_c, by_rows={})
    rng = np.random.default_rng(7)
    nxt = np.zeros(W, np.int64)
    used = [0]

    def call_args(k):
        w = np.sort(rng.choice(np.flatnonzero(length > 0), k, replace=False))
        t = nxt[w] % np.minimum(length[w], T)
        nxt[w] += 1
        sub = np.ascontiguousarray(rows_all[w, t])
        dest = (np.arange(used[0], used[0] + k, dtype=np.int64)) % K
        used[0] += k
        return sub, np.ascontiguousarray(pend_all[w, t]), w.astype(np.int64), dest

    for k in [int(x) for x in a.rows.split(',') if x.strip()]:
        for _ in range(20):   # (warm: every graph this k reaches)
            mid.call(*call_args(k))
        torch.cuda.synchronize()
        wall, gpu = [], []
        for _ in range(a.calls):
            args_ = call_args(k)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(st)
            t0 = time.perf_counter()
            mid.call(*args_)
            wall.append((time.perf_counter() - t0) * 1e3)
            e1.record(st)
            e1.synchronize()
            gpu.append(e0.elapsed_time(e1))
        out['by_rows'][str(k)] = dict(wall_ms=float(np.median(wall)), wall_p90_ms=float(np.percentile(wall, 90)),
                                      gpu_ms=float(np.median(gpu)))
    # the graphs replayed alone, back to back: the network's GPU time per (row bucket, E bucket)
    rp = {}
    with torch.cuda.stream(st), torch.inference_mode():
        for gi, gr in enumerate(ga.graphs):
            for _ in range(3):
                gr.replay()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(st)
            for _ in range(50):
                gr.replay()
            e1.record(st)
            e1.synchronize()
            r_ = ga.rsizes[gi // len(ga.sizes)]
            t_ = ga.sizes[gi % len(ga.sizes)]
            rp[f'r{r_}_e{t_}'] = e0.elapsed_time(e1) / 50
    torch.cuda.synchronize()
    out['replay_ms'] = rp
    # the profile
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        k = a.profile
        for _ in range(10):
            mid.call(*call_args(k))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(a.prof_calls):
                mid.call(*call_args(k))
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        ker = collections.Counter()
        ker_t = collections.Counter()
        cat_n, cat_t = collections.Counter(), collections.Counter()
        cp_n, cp_t = collections.Counter(), collections.Counter()
        for e in prof.events():
            if str(getattr(e, 'device_type', '')).endswith('CUDA'):
                name = e.name
                dt = float(getattr(e, 'device_time', None) or getattr(e, 'cuda_time', 0.0) or 0.0) / 1e3   # (ms)
                if re.search(r'memcpy|Memcpy', name):
                    kind = 'HtoD' if 'HtoD' in name or 'Host -> Device' in name else (
                        'DtoH' if 'DtoH' in name or 'Device -> Host' in name else 'DtoD')
                    cp_n[kind] += 1
                    cp_t[kind] += dt
                    continue
                if re.search(r'memset|Memset', name):
                    cp_n['memset'] += 1
                    cp_t['memset'] += dt
                    continue
                ker[name] += 1
                ker_t[name] += dt
                c = category(name)
                cat_n[c] += 1
                cat_t[c] += dt
        n = a.prof_calls
        k_ms = sum(ker_t.values()) / n
        out['profile'] = dict(
            rows=k, calls=n, wall_ms_per_call=wall_s * 1e3 / n, kernels_per_call=sum(ker.values()) / n,
            kernel_ms_per_call=k_ms, busy_share=k_ms / (wall_s * 1e3 / n),
            copies_per_call={x: cp_n[x] / n for x in cp_n}, copy_ms_per_call={x: cp_t[x] / n for x in cp_t},
            by_category={c: dict(per_call=cat_n[c] / n, ms_per_call=cat_t[c] / n) for c in cat_n},
            top_by_time=[(nm[:90], ker[nm] / n, ker_t[nm] / n) for nm, _ in ker_t.most_common(15)],
            top_by_count=[(nm[:90], ker[nm] / n, ker_t[nm] / n) for nm, _ in ker.most_common(15)])
    s = json.dumps(out, indent=1, default=str)
    print(s)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(s)


if __name__ == '__main__':
    main()
