"""(2026-11-22, the coordinator: the learner's two remaining real paths, a / b -- measurement only) the v3 frame
encoder's shapes, host time and compile cost in one real update of the run's configuration.

Run by abplus_bench_learner.run_bench when ISAAC_RL_V3_PROBE=1 (the learner in the trainer's process, the job in
process, path 0). Nothing of the run's training changes: the probe wraps the learner model's frame encoder for its
census and leaves it as it was.

  census: every frame_v3.forward_packed call of one update -- grad or not, records, the padded v3 tensors' shapes, the
          attention buckets' layout (records, width), real tokens, the call's wall time on the host (its own pad.cpu()
          sync included) and its GPU time (CUDA events). Distinct shape signatures: raw, and with the rows rounded up to
          a multiple of ISAAC_RL_V3_PROBE_ROWS (256) and every token width to a multiple of ISAAC_RL_V3_PROBE_TOK (8)
          -- the number of CUDA graphs / compiled shapes a per-bucket scheme would need, and the padding it costs.
  profile (a): one update under torch.profiler (CPU + CUDA) with the v3 marks on: kernels launched, the GPU's busy
          time (the kernels' intervals merged), and inside the 'v3:frame' regions their CPU time and launches. The
          update's wall time without the profiler minus the GPU's busy time bounds what CUDA graphs could take off.
  compile (b): torch.compile(forward_packed, dynamic=False) on up to ISAAC_RL_V3_PROBE_KEEP (5) of the census' training
          inputs of distinct shapes, forward + backward under the run's autocast: each first call (its compile), then
          the steady compiled and eager times; then one dynamic=True compile over the same inputs. Dynamo's frame and
          graph counters with each.

Output: lines 'probe ...' on stdout and <out>/v3_probe.json."""
import collections
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def _sync():
    torch.cuda.synchronize()


def _up(x, m):
    return -(-int(x) // m) * m


def rounded_signature(call, rows_m, tok_m):
    """A call's signature with the rows rounded up to rows_m and every other dimension of the v3 tensors and the bucket
    widths up to tok_m (bucket record counts as the rows)."""
    sig = tuple((k, (_up(s[0], rows_m),) + tuple(_up(d, tok_m) for d in s[1:2]) + tuple(s[2:]))
                for k, s in call['sig'] if len(s))
    lay = tuple((_up(n, rows_m), _up(w, tok_m)) for n, w in (call['layout'] or ()))
    return (call['grad'], sig, lay)


def census_summary(calls, rows_m=256, tok_m=8):
    """Counts of distinct signatures (raw / rounded), per grad mode, the padding the rounding costs."""
    out = {}
    for grad in (True, False):
        cs = [c for c in calls if c['grad'] == grad]
        if not cs:
            continue
        raw = {(c['sig'], c['layout']) for c in cs}
        inp = {c['sig'] for c in cs}
        lay = {c['layout'] for c in cs}
        rnd = {rounded_signature(c, rows_m, tok_m) for c in cs}
        rlay = {rounded_signature(c, rows_m, tok_m)[2] for c in cs}
        real = sum(sum(n * w for n, w in c['layout'] or ()) for c in cs)
        padded = sum(sum(_up(n, rows_m) * _up(w, tok_m) for n, w in c['layout'] or ()) for c in cs)
        out['train' if grad else 'no_grad'] = dict(
            calls=len(cs), distinct=len(raw), distinct_inputs=len(inp), distinct_layouts=len(lay),
            distinct_rounded=len(rnd), distinct_rounded_layouts=len(rlay),
            bucket_slots=sum(len(c['layout'] or ()) for c in cs),
            attention_padding_rounded=round(padded / max(real, 1), 4),
            host_s=round(sum(c['host'] for c in cs), 4), gpu_s=round(sum(c.get('gpu', 0.0) for c in cs), 4),
            rows=sum(c['B'] for c in cs))
    return out


def _merged(intervals):
    total, end = 0.0, -1.0
    for s, e in sorted(intervals):
        if s >= end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def probe(ctx, one, out_dir):
    import isaac_bridge.tok_policy as tp
    model, args = ctx['model'], ctx['args']
    fe = model.frame_v3
    if fe is None:
        print('probe: not a v3 model: nothing to probe', flush=True)
        return
    rows_m = int(os.environ.get('ISAAC_RL_V3_PROBE_ROWS', '256'))
    tok_m = int(os.environ.get('ISAAC_RL_V3_PROBE_TOK', '8'))
    n_keep = int(os.environ.get('ISAAC_RL_V3_PROBE_KEEP', '5'))
    res = dict(rows_m=rows_m, tok_m=tok_m)
    orig_fp = fe.forward_packed      # (the bound method: restored at the end)
    orig_ab = tp._attention_buckets
    cur, calls, keep = {}, [], {}
    rec = [False]

    def ab(cnt, buckets):
        base, layout = orig_ab(cnt, buckets)
        cur['layout'] = tuple((int(len(c)), int(w)) for c, w in layout)
        cur['tokens'] = int(np.asarray(cnt).sum())
        return base, layout

    def fp(v, pending, gtok, buckets=3):
        if not rec[0]:
            return orig_fp(v, pending, gtok, buckets)
        cur.clear()
        sig = tuple(sorted((k, tuple(t.shape)) for k, t in v.items()))
        grad = torch.is_grad_enabled()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        e0.record()
        out = orig_fp(v, pending, gtok, buckets)
        e1.record()
        t1 = time.perf_counter()
        calls.append(dict(grad=grad, B=int(pending.shape[0]), sig=sig, layout=cur.get('layout'),
                          tokens=cur.get('tokens'), host=t1 - t0, ev=(e0, e1)))
        if grad and sig not in keep and len(keep) < n_keep:
            keep[sig] = ({k: t.detach().clone() for k, t in v.items()}, pending.detach().clone())
        return out

    tp._attention_buckets = ab
    fe.forward_packed = fp
    try:
        one(0)                      # (warm-up)
        rec[0] = True
        r = one(0)
        rec[0] = False
        _sync()
        for c in calls:
            e0_, e1_ = c.pop('ev')
            c['gpu'] = e0_.elapsed_time(e1_) / 1000
    finally:
        rec[0] = False
        tp._attention_buckets = orig_ab
        del fe.forward_packed       # (the instance attribute: the class's method again)
    for c in calls:
        c.pop('ev', None)
    res['update_s'] = round(r['s'], 4)
    res['U'] = {k: round(v, 4) for k, v in r['U'].items()}
    res['census'] = census_summary(calls, rows_m, tok_m)
    by_lay = collections.Counter(len(c['layout'] or ()) for c in calls)
    res['buckets_per_call'] = dict(by_lay)
    res['rows_per_call'] = sorted(collections.Counter(c['B'] for c in calls).items())
    print(f'probe census: update {r["s"]:.3f} s; {len(calls)} frame-encoder calls; '
          + '; '.join(f'{k}: {json.dumps(v)}' for k, v in res['census'].items()), flush=True)
    print(f'probe census: rows per call {res["rows_per_call"][:12]}; buckets per call {dict(by_lay)}', flush=True)

    # (a) one update under the profiler: launches, the GPU's busy time, the v3:frame regions
    if os.environ.get('ISAAC_RL_V3_PROBE_PROFILE', '1') == '1':
        from torch.profiler import ProfilerActivity, profile as tprof
        marks0 = tp.PROF_MARKS
        tp.PROF_MARKS = True
        try:
            with tprof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                rp = one(0)
                _sync()
        finally:
            tp.PROF_MARKS = marks0
        ev = list(prof.events())
        kern = [e for e in ev if e.device_type == torch.autograd.DeviceType.CUDA]
        launch = [e for e in ev if e.device_type == torch.autograd.DeviceType.CPU and
                  ('LaunchKernel' in e.name or 'cudaGraphLaunch' in e.name or 'cuLaunchKernel' in e.name)]
        busy = _merged([(e.time_range.start, e.time_range.end) for e in kern]) / 1e6
        frames = [e for e in ev if e.name == 'v3:frame']
        win = sorted((e.time_range.start, e.time_range.end, e.thread) for e in frames)
        in_frame = 0
        for e in launch:
            s_ = e.time_range.start
            for a, b, th in win:
                if a <= s_ <= b and th == e.thread:
                    in_frame += 1
                    break
        fr_cpu = sum(b - a for a, b, _ in win) / 1e6
        res['profile'] = dict(update_s_profiled=round(rp['s'], 4), update_s=res['update_s'], kernels=len(kern),
                              launches=len(launch), gpu_busy_s=round(busy, 4),
                              host_exposed_s=round(res['update_s'] - busy, 4),
                              v3_frame_regions=len(frames), v3_frame_cpu_s=round(fr_cpu, 4),
                              v3_frame_launches=in_frame)
        print(f'probe profile: {json.dumps(res["profile"])}', flush=True)

    # (b) compile: forward_packed, fixed shapes, then dynamic, forward + backward under the run's autocast
    if os.environ.get('ISAAC_RL_V3_PROBE_COMPILE', '1') == '1' and keep:
        import importlib   # (not 'import torch._dynamo': that binds torch as a local name of probe(), and the
        #                     census's wrapper, a closure, then finds it unassigned)
        dynamo = importlib.import_module('torch._dynamo')
        counters = importlib.import_module('torch._dynamo.utils').counters
        cfg = dynamo.config
        for name_ in ('cache_size_limit', 'recompile_limit', 'accumulated_cache_size_limit',
                      'accumulated_recompile_limit'):
            if hasattr(cfg, name_):
                setattr(cfg, name_, max(int(getattr(cfg, name_)), 1024))
        amp = getattr(args, 'amp', 'off') == 'bf16'
        gtok = model.gtok.tok
        params = [p for p in fe.parameters() if p.requires_grad]

        def step(fn, v, pend):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                x, _pad, _g = fn(v, pend, gtok)
            x.float().square().mean().backward()
            _sync()

        def timed(fn, v, pend, n=3):
            ts = []
            for _ in range(n):
                t0 = time.perf_counter()
                step(fn, v, pend)
                ts.append(time.perf_counter() - t0)
            return float(np.median(ts))

        items = list(keep.values())
        comp = []
        for dyn in (False, True):
            dynamo.reset()
            counters.clear()
            cfn = torch.compile(orig_fp, dynamic=dyn)
            rows_ = []
            for v, pend in items:
                t0 = time.perf_counter()
                try:
                    step(cfn, v, pend)
                    first = time.perf_counter() - t0
                    rows_.append(dict(rows=int(pend.shape[0]), first_s=round(first, 3),
                                      compiled_s=round(timed(cfn, v, pend), 4),
                                      eager_s=round(timed(orig_fp, v, pend), 4)))
                except Exception as exc:   # (a measurement: the failure is the result)
                    rows_.append(dict(rows=int(pend.shape[0]), error=f'{type(exc).__name__}: {str(exc)[:300]}'))
                    break
                finally:
                    for p_ in params:
                        p_.grad = None
            st = {k: dict(v_) for k, v_ in counters.items() if k in ('stats', 'graph_break', 'recompiles')}
            comp.append(dict(dynamic=dyn, runs=rows_,
                             stats={k: {kk: int(vv) for kk, vv in list(v_.items())[:12]} for k, v_ in st.items()}))
            print(f'probe compile dynamic={dyn}: {json.dumps(comp[-1])}', flush=True)
        res['compile'] = comp
        dynamo.reset()
    for p_ in model.parameters():
        p_.grad = None
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / 'v3_probe.json').write_text(json.dumps(dict(res, calls=[
            dict(c, sig=[list(x) for x in c['sig']]) for c in calls]), default=str))
        print(f'probe: written to {Path(out_dir) / "v3_probe.json"}', flush=True)
    except Exception as exc:
        print('probe: no json', exc, flush=True)
