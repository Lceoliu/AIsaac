"""B52 condition 6: which learner kernels run longer than 10 ms (they block the actor's high-priority stream).

usage: python3 b52_kernels.py <torch profiler trace.json from ISAAC_RL_B38_TORCHPROF> [threshold_ms]

Reads the Chrome trace torch.profiler wrote for one learner update (train_tok profile_learn), takes the GPU kernel
events (cat 'kernel' / 'gpu_memcpy' / 'gpu_memset') and prints: kernels over the threshold by name (count, total ms,
max ms), the total GPU time, the share of GPU time in kernels over the threshold, and the duration distribution.
Standard library only."""
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
    total = sum(e['dur'] for e in kern)/1e3
    by = defaultdict(lambda: [0, 0.0, 0.0])
    big = defaultdict(lambda: [0, 0.0, 0.0])
    durs = sorted(e['dur']/1e3 for e in kern)
    for e in kern:
        d = e['dur']/1e3
        name = e['name'][:110]
        b = by[name]; b[0] += 1; b[1] += d; b[2] = max(b[2], d)
        if d >= thr:
            g = big[name]; g[0] += 1; g[1] += d; g[2] = max(g[2], d)
    over = sum(v[1] for v in big.values())
    n = len(durs)
    print(f'{n} GPU kernels, {total/1e3:.2f} s GPU time; >= {thr:.0f} ms: {sum(v[0] for v in big.values())} kernels, '
          f'{over/1e3:.2f} s = {100*over/total:.1f}% of GPU time')
    print(f'duration ms: p50 {durs[n//2]:.3f}, p90 {durs[int(n*.9)]:.3f}, p99 {durs[int(n*.99)]:.3f}, max {durs[-1]:.1f}')
    print(f'\n== kernels >= {thr:.0f} ms by name: count, total ms, max ms')
    for name, (c, t, mx) in sorted(big.items(), key=lambda kv: -kv[1][1])[:30]:
        print(f'  {c:5d} {t:9.1f} {mx:8.1f}  {name}')
    print('\n== top kernels by total time: count, total ms, max ms')
    for name, (c, t, mx) in sorted(by.items(), key=lambda kv: -kv[1][1])[:20]:
        print(f'  {c:6d} {t:9.1f} {mx:8.1f}  {name}')


if __name__ == '__main__':
    main()
