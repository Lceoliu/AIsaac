#!/usr/bin/env python3
"""A run's start-up switch lines against a reference run's (2026-11-12, the coordinator: before the rehearsal, every
optional native component's line of each rank against long2's; any line from on to off, missing, or a failed build is
blocking). Read-only; the host's python3.

  python3 switch_cmp.py <reference stdout> <stdout> [<stdout> ...]

The switch lines: B45 visible frame encoder, B52 cpp actor driver, B58 native consumer, B58c native lineage, B58d native
lanes, B58e batched records, B42 compiled Spatial warm-up (B56 fx cache), B44 Flex Reader warm-up, B42 actor / lane
pre-capture, B62 CUDA scheduling, BM1 graph capture mode, learner device (B38), devices (B38), driver CPUs, learner
CPUs. Each line printed whole (reference, then each candidate); BLOCK when a reference key is missing from a candidate,
or a candidate's line says off / build failed / unavailable / reference (the Python path) where the reference's does
not. Exit 1 on any BLOCK."""
import re
import sys

KEYS = ('B45 visible frame encoder', 'B52 cpp actor driver', 'B58 native consumer', 'B58c native lineage',
        'B58d native lanes', 'B58e batched records', 'B42 compiled Spatial warm-up', 'B44 Flex Reader warm-up',
        'B42 actor pre-capture', 'B42 lane pre-capture', 'B62 CUDA scheduling', 'BM1 graph capture mode',
        'learner device (B38)', 'devices (B38)', 'driver CPUs (ISAAC_DRIVER_CPUS)', 'learner CPUs (--learner-cpus)')
BAD = re.compile(r'\boff\b|build failed|unavailable|reference \(|disabled|load failed', re.I)


def lines(path):
    out = {}
    with open(path, errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            for k in KEYS:
                if line.startswith(k) and k not in out:
                    out[k] = line
    return out


def main():
    ref, cands = sys.argv[1], sys.argv[2:]
    R = lines(ref)
    C = {c: lines(c) for c in cands}
    block = 0
    for k in KEYS:
        if k not in R:
            continue
        print(f'== {k}')
        print(f'   ref   {R[k][:300]}')
        for c in cands:
            got = C[c].get(k)
            bad = got is None or (BAD.search(got) is not None and BAD.search(R[k]) is None)
            block += bad
            print(f'   {"BLOCK" if bad else "ok   "} {(got or "(missing)")[:300]}   [{c}]')
    print(f'switch_cmp: {block} blocking line(s)')
    sys.exit(1 if block else 0)


if __name__ == '__main__':
    main()
