"""(2026-11-20, ISAAC_RL_RING_COMPACT) The teacher and V2 rings' resident memory once full, whole slots against compact
slots, measured: real records (a --dump-rollouts directory: their fixed fields and frames, so the real frame sizes)
written into shared-memory rings of the sampler's shape (workers x TEACH_RING / V2_RING ROW slots, after stale
whole-slot contents never: fresh rings, as a run starts), read back as the trainer reads them, and the rings' resident
pages read from /proc/self/smaps (Linux). Prints MB per ring and mode for each worker count.

   usage: python tools/ring_mem_probe.py <rollouts dir> [--workers 16,24] [--records 4000]"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def shm_rss_kb(name):
    want = '/dev/shm/' + name.lstrip('/')
    rss, cur = 0, False
    for line in open('/proc/self/smaps'):
        f = line.split()
        if len(f) >= 5 and '-' in f[0] and ':' in f[3]:
            cur = (f[-1] if len(f) > 5 else '') == want
        elif cur and f[0] == 'Rss:':
            rss += int(f[1])
    return rss


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('rollouts')
    ap.add_argument('--workers', default='16,24')
    ap.add_argument('--records', type=int, default=4000)
    a = ap.parse_args(argv)
    from multiprocessing import shared_memory
    from isaac_bridge import tok_obs as TO
    from isaac_bridge.tok_obs import ROW
    from isaac_bridge.tok_sampler import TEACH_RING
    from isaac_bridge.tok_teacher2 import V2_RING
    recs = []
    for p in sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.npz'))):
        z = np.load(p, allow_pickle=False)
        rows, length = z['rows'], z['length']
        for w in range(rows.shape[0]):
            for t in range(int(length[w])):
                r = np.zeros(1, ROW)
                src = rows[w, t:t + 1]
                if src.dtype == ROW:
                    r[:] = src
                else:   # (a compact dump: expand_rows' layout has its frames in the record already)
                    for name in src.dtype.names:
                        if name in ROW.names:
                            r[name] = src[name]
                recs.append(r)
                if len(recs) >= a.records:
                    break
            if len(recs) >= a.records:
                break
        if len(recs) >= a.records:
            break
    vb = np.array([int(r['visible_bytes'][0]) for r in recs])
    print(f'records {len(recs)}: visible_bytes mean {vb.mean():.0f} p50 {np.median(vb):.0f} p99 '
          f'{np.percentile(vb, 99):.0f} max {vb.max()} (frame budget {ROW.itemsize - TO.ROW_FRAME_AT})', flush=True)
    for W in [int(x) for x in a.workers.split(',')]:
        for ring_name, R in (('teacher', TEACH_RING), ('v2', V2_RING)):
            for compact in (False, True):
                TO.RING_COMPACT = compact
                shm = shared_memory.SharedMemory(create=True, size=ROW.itemsize * W * R)
                try:
                    ring = np.ndarray((W, R), ROW, buffer=shm.buf)
                    k = 0
                    for w in range(W):
                        for j in range(R):   # (a full ring: every slot written, then the trainer's read)
                            TO.ring_put(ring[w, j:j + 1], recs[k % len(recs)])
                            k += 1
                    for w in range(W):
                        for j in range(R):
                            TO.ring_take(ring[w, j:j + 1])
                    rss = shm_rss_kb(shm.name) / 1024.0
                    print(f'workers {W:3d} {ring_name:8s} ring ({R} slots / worker) '
                          f'{"compact" if compact else "whole  "}: resident {rss:8.1f} MB '
                          f'(reserved {ROW.itemsize * W * R / 2 ** 20:8.1f} MB)', flush=True)
                    del ring
                finally:
                    shm.close()
                    shm.unlink()


if __name__ == '__main__':
    main()
