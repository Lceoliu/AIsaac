"""B52 condition 6: which learner kernels run longer than 10 ms (they block the actor's high-priority stream).

usage: python3 b52_kernels.py <torch profiler trace.json> [threshold_ms]

Reads the Chrome trace torch.profiler wrote for one learner update (b52_learner_kernels.py, or train_tok's
ISAAC_RL_B38_TORCHPROF), takes the GPU kernel events (cat 'kernel' / 'gpu_memcpy' / 'gpu_memset') and prints: the total
GPU time and the share in kernels over the threshold, the duration distribution, the GPU time per region (the
record_function ranges on the GPU timeline, cat 'gpu_user_annotation': fwd_grad / bwd / opt / fwd_nograd), and the
kernels over the threshold by region and name (count, total ms, max ms). Standard library only."""
import bisect
import json
import sys
from collections import defaultdict


def main():
    path = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    with open(path) as f:
        ev = json.load(f)
    ev = ev['traceEvents'] if isinstance(ev, dict) else ev
    kern = [e for e in ev if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset', 'Kernel', 'Memcpy', 'Memset')]
    if not kern:
        print('no GPU kernel events in the trace'); return
    ann = sorted((e['ts'], e['ts']+e['dur'], e['name']) for e in ev
                 if e.get('ph') == 'X' and e.get('cat') == 'gpu_user_annotation' and e.get('name') in ('fwd_grad', 'bwd', 'opt', 'fwd_nograd'))
    starts = [a[0] for a in ann]

    def region(ts):
        i = bisect.bisect_right(starts, ts)-1
        return ann[i][2] if i >= 0 and ts <= ann[i][1] else '-'

    total = sum(e['dur'] for e in kern)/1e3
    by = defaultdict(lambda: [0, 0.0, 0.0])
    big = defaultdict(lambda: [0, 0.0, 0.0])
    reg = defaultdict(lambda: [0, 0.0, 0, 0.0])
    durs = sorted(e['dur']/1e3 for e in kern)
    for e in kern:
        d = e['dur']/1e3
        name = e['name'][:100]
        r = region(e['ts'])
        b = by[name]; b[0] += 1; b[1] += d; b[2] = max(b[2], d)
        g = reg[r]; g[0] += 1; g[1] += d
        if d >= thr:
            g[2] += 1; g[3] += d
            k = big[(r, name)]; k[0] += 1; k[1] += d; k[2] = max(k[2], d)
    over = sum(v[1] for v in big.values())
    n = len(durs)
    print(f'{n} GPU kernels, {total/1e3:.2f} s GPU time; >= {thr:.0f} ms: {sum(v[0] for v in big.values())} kernels, '
          f'{over/1e3:.2f} s = {100*over/total:.1f}% of GPU time')
    print(f'duration ms: p50 {durs[n//2]:.3f}, p90 {durs[int(n*.9)]:.3f}, p99 {durs[int(n*.99)]:.3f}, p99.9 {durs[int(n*.999)]:.2f}, max {durs[-1]:.1f}')
    print(f'\n== GPU time by region: kernels, ms, kernels >= {thr:.0f} ms, their ms')
    for r, (c, t, cb, tb) in sorted(reg.items(), key=lambda kv: -kv[1][1]):
        print(f'  {r:11s} {c:7d} {t:10.1f} {cb:5d} {tb:9.1f}')
    print(f'\n== kernels >= {thr:.0f} ms by region and name: count, total ms, max ms')
    for (r, name), (c, t, mx) in sorted(big.items(), key=lambda kv: -kv[1][1])[:40]:
        print(f'  {r:11s} {c:5d} {t:9.1f} {mx:8.1f}  {name}')
    print('\n== top kernels by total time: count, total ms, max ms')
    for name, (c, t, mx) in sorted(by.items(), key=lambda kv: -kv[1][1])[:20]:
        print(f'  {c:6d} {t:9.1f} {mx:8.1f}  {name}')


if __name__ == '__main__':
    main()
