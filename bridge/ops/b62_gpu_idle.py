"""B62 B2 upper bound: how long the GPU sits idle on the learner's critical path, from an nsys report with the cuda and
nvtx traces of train_tok (ISAAC_BLOCK_NVTX=1, B50window capture).

  python3 b62_gpu_idle.py <report.nsys-rep | .sqlite> [--nsys /usr/local/cuda/bin/nsys]

Over every learner.update range in the window, with the actor main thread's h.wait_learner ranges (the actor waiting
for the learner: nothing else feeds the GPU then):
  * wall, GPU busy (union of every kernel / memcpy / memset of the process), GPU fully idle;
  * the same restricted to the h.wait_learner parts: the GPU idle there is time the learner's host side (Python
    dispatch, host preparation, synchronisations) keeps the whole GPU waiting while the actor waits for the learner --
    the most any learner-side host speed-up (B2's Spatial dispatch included) can recover per cycle;
  * the idle inside the learner's forward_packed ranges (Spatial / Writer / Reader dispatch) of those parts.
Standard library only (sqlite3)."""
import os
import sqlite3
import subprocess
import sys


def export(path, nsys):
    if path.endswith('.sqlite'):
        return path
    out = os.path.splitext(path)[0]+'.sqlite'
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(path):
        subprocess.run([nsys, 'export', '--type', 'sqlite', '--force-overwrite', 'true', '-o', out, path], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def covered(u, lo, hi):
    """Length of the union u (sorted, disjoint) inside [lo, hi]."""
    s = 0
    for a, b in u:
        if b <= lo:
            continue
        if a >= hi:
            break
        s += min(b, hi)-max(a, lo)
    return s


def intersect(xs, ys):
    """Pairwise intersections of two interval lists (each sorted, disjoint)."""
    out, i, j = [], 0, 0
    while i < len(xs) and j < len(ys):
        a, b = max(xs[i][0], ys[j][0]), min(xs[i][1], ys[j][1])
        if a < b:
            out.append((a, b))
        if xs[i][1] < ys[j][1]:
            i += 1
        else:
            j += 1
    return out


def main():
    nsys = sys.argv[sys.argv.index('--nsys')+1] if '--nsys' in sys.argv else '/usr/local/cuda/bin/nsys'
    c = sqlite3.connect(export(sys.argv[1], nsys))
    T = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    names = dict(c.execute('select id, value from StringIds'))

    def ranges(text):
        rows = c.execute('select start, end, text, textId from NVTX_EVENTS where end is not null').fetchall()
        return sorted((a, b) for a, b, t, tid in rows if (t or names.get(tid)) == text)
    upd = ranges('learner.update') or ranges('B60bench')   # (the bench: its captured rep)
    wait = union([list(x) for x in ranges('h.wait_learner')])
    fwd = union([list(x) for x in ranges('learner.forward_packed')])
    gpu = []
    for t in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
        if t in T:
            gpu += [list(r) for r in c.execute(f'select start, end from {t}')]
    gpu = union(gpu)
    tot = dict(updates=len(upd), wall=0, busy=0, wait=0, wait_busy=0, wait_fwd=0, wait_fwd_busy=0)
    for lo, hi in upd:
        tot['wall'] += hi-lo
        tot['busy'] += covered(gpu, lo, hi)
        for a, b in intersect([(lo, hi)], [tuple(x) for x in wait]):
            tot['wait'] += b-a
            tot['wait_busy'] += covered(gpu, a, b)
            for a2, b2 in intersect([(a, b)], [tuple(x) for x in fwd]):
                tot['wait_fwd'] += b2-a2
                tot['wait_fwd_busy'] += covered(gpu, a2, b2)
    n = max(tot['updates'], 1)
    s = 1e-9
    print(f"learner.update ranges: {tot['updates']} (per update, s)")
    print(f"  wall {tot['wall']*s/n:.2f}  GPU busy {tot['busy']*s/n:.2f}  GPU idle {(tot['wall']-tot['busy'])*s/n:.2f}")
    print(f"  while the actor waits for the learner: {tot['wait']*s/n:.2f}  GPU busy {tot['wait_busy']*s/n:.2f}  "
          f"GPU idle {(tot['wait']-tot['wait_busy'])*s/n:.2f}  <- upper bound of what learner host speed-ups recover")
    print(f"    of which inside forward_packed: {tot['wait_fwd']*s/n:.2f}  GPU idle "
          f"{(tot['wait_fwd']-tot['wait_fwd_busy'])*s/n:.2f}")
    # measure 2: inside every 'learner.spatial' range (a Spatial segment: the forward on the learner thread, the
    # backward's recomputation on the autograd thread), the kernels / copies that thread launched, in order; between
    # two of them, the time the GPU had finished the first before the thread even called the launch of the second
    # (launch call start - previous end, when positive): the segment's own stream starved by host dispatch, whatever
    # else the GPU was doing meanwhile
    if 'CUPTI_ACTIVITY_KIND_RUNTIME' not in T:
        print('measure 2: no runtime API trace (nsys -t cuda)')
        return
    rt = dict(c.execute('select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME'))
    rt_tid = dict(c.execute('select correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME'))
    work = []
    for t in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
        if t in T:
            work += [(rt[k], a, b, rt_tid[k]) for a, b, k in c.execute(f'select start, end, correlationId from {t}')
                     if k in rt]
    by_tid = {}
    for launch, a, b, tid in work:
        by_tid.setdefault(tid, []).append((launch, a, b))
    for v in by_tid.values():
        v.sort()
    rows = c.execute('select start, end, globalTid, text, textId from NVTX_EVENTS where end is not null').fetchall()
    sp = [(a, b, tid) for a, b, tid, t, i in rows if (t or names.get(i)) == 'learner.spatial']
    import bisect
    gap = segs = launches = 0
    for a, b, tid in sp:
        v = by_tid.get(tid, [])
        lo = bisect.bisect_left(v, (a, -1, -1))
        hi = bisect.bisect_right(v, (b, 1 << 62, 1 << 62))
        seg = v[lo:hi]
        segs += 1
        launches += len(seg)
        prev_end = None
        for launch, ka, kb in seg:
            if prev_end is not None and launch > prev_end:
                gap += launch-prev_end
            prev_end = kb if prev_end is None else max(prev_end, kb)
    print(f"measure 2 (Spatial segments): {segs} ranges, {launches} launches; dispatch gaps "
          f"{gap*s/n:.2f} s per update (window {n} update(s))")


if __name__ == '__main__':
    main()
