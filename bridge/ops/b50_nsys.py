"""B50 phase 0: per-thread and per-phase breakdown of an Nsight Systems report of train_tok (block memory).

usage: python3 b50_nsys.py <report.nsys-rep | report.sqlite> [--nsys /usr/local/cuda/bin/nsys]

The report comes from `nsys profile -t nvtx,python-gil[,cuda] --capture-range=cudaProfilerApi` of train_tok.py run
with ISAAC_BLOCK_NVTX=1 and ISAAC_NSYS_RANGE=a:b (isaac_bridge/block_nvtx.py). Prints, for the actor (main) thread
and the learner thread of the train_tok process over the capture window: GIL held / GIL waited (python-gil ranges),
CUDA synchronisation waits (CUPTI runtime API, when cuda was traced), running / blocked time (sched events, when
available), the NVTX ranges of block_nvtx (inclusive and exclusive sums, per-call ms, percentiles) and the GIL
waits attributed to the range they start in. Only the standard library is used (sqlite3)."""
import os
import subprocess
import sqlite3
import sys
from collections import defaultdict

SYNC_NAMES = ('cudaStreamSynchronize', 'cudaEventSynchronize', 'cudaDeviceSynchronize', 'cuStreamSynchronize',
              'cuEventSynchronize', 'cuCtxSynchronize')


def export(path, nsys):
    if path.endswith('.sqlite'):
        return path
    out = os.path.splitext(path)[0]+'.sqlite'
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(path):
        subprocess.run([nsys, 'export', '--type', 'sqlite', '--force-overwrite', 'true', '-o', out, path], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def tables(c):
    return {r[0] for r in c.execute("select name from sqlite_master where type='table'")}


def strings(c):
    return dict(c.execute('select id, value from StringIds'))


def ms(ns):
    return ns/1e6


def pct(ds, q):
    return ds[min(len(ds)-1, int(len(ds)*q))]


def main():
    nsys = '/usr/local/cuda/bin/nsys'
    args = [a for a in sys.argv[1:]]
    if '--nsys' in args:
        i = args.index('--nsys'); nsys = args[i+1]; del args[i:i+2]
    db = export(args[0], nsys)
    c = sqlite3.connect(db)
    T = tables(c)
    S = strings(c)
    # ---- the train_tok process: the one with NVTX block_nvtx ranges (domain 0, text 'actor.poll')
    rows = c.execute("select globalTid, count(*) from NVTX_EVENTS where text='actor.poll' group by globalTid").fetchall()
    if not rows:   # (B52: the C++ driver polls; the Python main thread is the consumer with the lane service's ranges)
        rows = c.execute("select globalTid, count(*) from NVTX_EVENTS where text like 'lane.%' or text like 'actor.%' "
                         "group by globalTid").fetchall()
    if not rows:
        print('no actor.poll NVTX ranges: was ISAAC_BLOCK_NVTX=1 set?'); return
    actor_tid = max(rows, key=lambda r: r[1])[0]
    pid = actor_tid >> 24   # globalTid = (pid << 24) | tid on Linux targets
    rows = c.execute("select globalTid, count(*) from NVTX_EVENTS where text like 'learner.%' group by globalTid").fetchall()
    learner_tid = max(rows, key=lambda r: r[1])[0] if rows else None
    # ---- the capture window: the B50window range (block_nvtx.profiler_mark) when present, else every event
    row = c.execute("select start, end from NVTX_EVENTS where text = 'B50window' and end is not null order by start limit 1").fetchone()
    if row:
        lo, hi = row
        print('window: the B50window range')
    else:
        lo, hi = c.execute("select min(start), max(coalesce(end, start)) from NVTX_EVENTS where (globalTid >> 24) = ?",
                           (pid,)).fetchone()
        print('window: every event of the process (no B50window range)')
    win = hi-lo
    print(f'process pid {pid}: capture window {win/1e9:.2f} s; actor tid {actor_tid & 0xffffff}, '
          f'learner tid {learner_tid & 0xffffff if learner_tid else None}')
    threads = [('actor', actor_tid)] + ([('learner', learner_tid)] if learner_tid else [])
    n_calls = c.execute("select count(*) from NVTX_EVENTS where globalTid = ? and text = 'actor.step_main' and start >= ? and end <= ?",
                        (actor_tid, lo, hi)).fetchone()[0]
    n_steps = c.execute("select count(*) from NVTX_EVENTS where globalTid = ? and text = 'g.sync' and start >= ? and end <= ?",
                        (actor_tid, lo, hi)).fetchone()[0]
    n_upd = c.execute("select count(*) from NVTX_EVENTS where text = 'learner.update' and start >= ? and end <= ?",
                      (lo, hi)).fetchone()[0]
    n_rounds = c.execute("select count(*) from NVTX_EVENTS where globalTid = ? and text = 'lane.eager' and start >= ? and end <= ?",
                         (actor_tid, lo, hi)).fetchone()[0]
    n_coll = c.execute("select count(*) from NVTX_EVENTS where globalTid = ? and text = 'actor.weights' and start >= ? and end <= ?",
                       (actor_tid, lo, hi)).fetchone()[0]
    print(f'in the window: {n_calls} main calls, {n_steps} step graph syncs, {n_rounds} lane rounds '
          f'(lane-only steps {n_steps-n_calls}, rounds merged into a main call {n_rounds-(n_steps-n_calls)}), '
          f'{n_upd} complete learner updates, {n_coll} weight takes (collects)')
    per = max(n_coll, 1)

    def clipped_sum(table, where, params):
        """sum of (min(end,hi)-max(start,lo)) over ranges of a table."""
        q = f"select coalesce(sum(min(end, ?) - max(start, ?)), 0), count(*) from {table} where end is not null and {where}"
        return c.execute(q, (hi, lo)+params).fetchone()

    # ---- GIL (python-gil trace: NVTX_EVENTS, text ids 'Holding GIL' / 'Waiting for GIL')
    gil_ids = {v: k for k, v in S.items() if v in ('Holding GIL', 'Waiting for GIL')}
    print('\n== per thread, over the window (s)')
    print(f'{"thread":8s} {"GIL held":>9s} {"GIL wait":>9s} {"cuda sync":>10s} {"running":>9s} {"blocked":>9s} {"window":>8s}')
    sync_ids = {k for k, v in S.items() if v in SYNC_NAMES}
    for name, tid in threads:
        held = wait = 0.0
        if gil_ids:
            held, _ = clipped_sum('NVTX_EVENTS', 'globalTid = ? and textId = ?', (tid, gil_ids['Holding GIL']))
            wait, _ = clipped_sum('NVTX_EVENTS', 'globalTid = ? and textId = ?', (tid, gil_ids['Waiting for GIL']))
        sync = 0.0
        if 'CUPTI_ACTIVITY_KIND_RUNTIME' in T and sync_ids:
            sync, _ = clipped_sum('CUPTI_ACTIVITY_KIND_RUNTIME',
                                  f"globalTid = ? and nameId in ({','.join(map(str, sync_ids))})", (tid,))
        run = blk = 0.0
        if 'SCHED_EVENTS' in T:
            ev = c.execute("select start, isSchedIn from SCHED_EVENTS where globalTid = ? order by start", (tid,)).fetchall()
            last_in = None
            for t, sin in ev:
                if sin:
                    last_in = t
                elif last_in is not None:
                    a, b = max(last_in, lo), min(t, hi)
                    if b > a:
                        run += b-a
                    last_in = None
            blk = win-run
        print(f'{name:8s} {held/1e9:9.2f} {wait/1e9:9.2f} {sync/1e9:10.2f} {run/1e9:9.2f} {blk/1e9:9.2f} {win/1e9:8.2f}')

    # ---- CUDA runtime API per thread (top calls)
    if 'CUPTI_ACTIVITY_KIND_RUNTIME' in T:
        print('\n== CUDA runtime API per thread (window): name, calls, total s, mean us')
        for name, tid in threads:
            rows = c.execute("select nameId, count(*), sum(min(end, ?) - max(start, ?)) from CUPTI_ACTIVITY_KIND_RUNTIME "
                             "where globalTid = ? and start < ? and end > ? group by nameId order by 3 desc limit 12",
                             (hi, lo, tid, hi, lo)).fetchall()
            for nid, n, tot in rows:
                print(f'  {name:8s} {S.get(nid, nid):40s} {n:8d} {tot/1e9:8.3f} {tot/max(n,1)/1e3:9.1f}')

    # ---- NVTX ranges of block_nvtx: inclusive, exclusive (children of the same thread subtracted), per call
    print('\n== NVTX ranges (block_nvtx) per thread: name, calls, inclusive s, exclusive s, ms/call (inclusive)')
    for name, tid in threads:
        ev = c.execute("select start, end, text from NVTX_EVENTS where globalTid = ? and text is not null and end is not null "
                       "and domainId = 0 and start >= ? and end <= ? order by start, end desc", (tid, lo, hi)).fetchall()
        incl, excl, cnt = defaultdict(int), defaultdict(int), defaultdict(int)
        ctx_sum, ctx_cnt = defaultdict(int), defaultdict(int)   # g.sync by context: under lane.eager or not
        stack = []
        for s, e, t in ev:
            while stack and stack[-1][1] <= s:
                stack.pop()
            d = e-s
            incl[t] += d; cnt[t] += 1
            excl[t] += d
            if stack:
                excl[stack[-1][2]] -= d
            if t == 'g.sync':
                k = 'lane round (lane-only or merged)' if any(x[2] == 'lane.eager' for x in stack) else 'main-only call'
                ctx_sum[k] += d; ctx_cnt[k] += 1
            stack.append((s, e, t))
        for k in ctx_sum:
            print(f'  {name:8s} g.sync in {k:36s} {ctx_cnt[k]:8d} {ctx_sum[k]/1e9:9.3f} s  {ms(ctx_sum[k])/ctx_cnt[k]:7.3f} ms/step'
                  f'   per collect {ctx_sum[k]/1e9/per:7.3f}')
        print(f'  ({name}: inclusive / exclusive also per collect, over {per} collects)')
        for t in sorted(incl, key=lambda k: -incl[k]):
            print(f'  {name:8s} {t:26s} {cnt[t]:8d} {incl[t]/1e9:9.3f} {excl[t]/1e9:9.3f} {ms(incl[t])/max(cnt[t],1):8.3f}'
                  f'   per collect {incl[t]/1e9/per:7.3f} / {excl[t]/1e9/per:7.3f}')

    # ---- GIL waits attributed to the innermost block_nvtx range they start in (actor thread), GIL percentiles
    if gil_ids:
        print('\n== GIL waits by the block_nvtx range they start in (actor thread): range, waits, total s, mean us, p90 us')
        rng = c.execute("select start, end, text from NVTX_EVENTS where globalTid = ? and text is not null and end is not null "
                        "and domainId = 0 and start >= ? and end <= ? order by start, end desc", (actor_tid, lo, hi)).fetchall()
        waits = c.execute("select start, end from NVTX_EVENTS where globalTid = ? and textId = ? and end is not null "
                          "and start >= ? and end <= ? order by start", (actor_tid, gil_ids['Waiting for GIL'], lo, hi)).fetchall()
        by = defaultdict(list)
        stack, j = [], 0
        for s_, e_ in waits:
            while j < len(rng) and rng[j][0] <= s_:
                while stack and stack[-1][1] <= rng[j][0]:
                    stack.pop()
                stack.append(rng[j]); j += 1
            while stack and stack[-1][1] <= s_:
                stack.pop()
            by[stack[-1][2] if stack else '(outside ranges)'].append(e_-s_)
        for t in sorted(by, key=lambda k: -sum(by[k])):
            ds = sorted(by[t]); n = len(ds)
            print(f'  {t:26s} {n:8d} {sum(ds)/1e9:9.3f} {sum(ds)/n/1e3:9.1f} {pct(ds, .9)/1e3:9.1f}')
        for name, tid in threads:
            for label, key in (('waits', 'Waiting for GIL'), ('holds', 'Holding GIL')):
                ds = sorted(e_-s_ for s_, e_ in c.execute(
                    "select start, end from NVTX_EVENTS where globalTid = ? and textId = ? and end is not null and start >= ? and end <= ?",
                    (tid, gil_ids[key], lo, hi)))
                if ds:
                    n = len(ds)
                    print(f'  {name} GIL {label}: n {n}, p50 {ds[n//2]/1e3:.1f} us, p90 {pct(ds, .9)/1e3:.1f} us, '
                          f'p99 {pct(ds, .99)/1e3:.1f} us, max {ds[-1]/1e6:.2f} ms, total {sum(ds)/1e9:.2f} s')

    print('\n== range duration percentiles (ms): range, n, p50, p90, p99, max')
    for name, tid in threads:
        rows = c.execute("select text, end - start from NVTX_EVENTS where globalTid = ? and text is not null and end is not null "
                         "and domainId = 0 and start >= ? and end <= ?", (tid, lo, hi)).fetchall()
        by = defaultdict(list)
        for t, d in rows:
            by[t].append(d)
        for t in sorted(by, key=lambda k: -sum(by[k])):
            ds = sorted(by[t]); n = len(ds)
            print(f'  {name:8s} {t:26s} {n:8d} {ms(ds[n//2]):8.3f} {ms(pct(ds, .9)):8.3f} {ms(pct(ds, .99)):8.3f} {ms(ds[-1]):8.3f}')

    # ---- g.sync tail: the GPU wait split by duration (a step graph alone takes about 2 ms on the idle 4090)
    ds = sorted(e_-s_ for s_, e_ in c.execute("select start, end from NVTX_EVENTS where globalTid = ? and text = 'g.sync' "
                                               "and end is not null and start >= ? and end <= ?", (actor_tid, lo, hi)))
    if ds:
        print('\n== g.sync (GPU wait at the step graph sync) by duration bucket: bucket, syncs, total s, share')
        tot = sum(ds)
        for a, b in ((0, 2.5e6), (2.5e6, 5e6), (5e6, 10e6), (10e6, 50e6), (50e6, 1e12)):
            sel = [d for d in ds if a <= d < b]
            if sel:
                print(f'  {a/1e6:5.1f}-{min(b, 1e9)/1e6:6.1f} ms {len(sel):8d} {sum(sel)/1e9:9.3f} {100*sum(sel)/tot:6.1f}%')
        base = 2.0e6*len(ds)
        print(f'  if every sync took 2.0 ms (idle graph time): {base/1e9:.2f} s; excess over that: {(tot-base)/1e9:.2f} s')

    # ---- GPU: kernels / graphs per stream (window), utilisation (cuda trace only)
    if 'CUPTI_ACTIVITY_KIND_KERNEL' in T:
        print('\n== GPU kernels per stream (window): stream, launches, busy s (sum of durations), share of window')
        rows = c.execute("select streamId, count(*), sum(min(end, ?) - max(start, ?)) from CUPTI_ACTIVITY_KIND_KERNEL "
                         "where start < ? and end > ? group by streamId order by 3 desc", (hi, lo, hi, lo)).fetchall()
        for sid, n, tot in rows:
            print(f'  stream {sid:4d} {n:9d} {tot/1e9:9.3f} {100*tot/win:6.1f}%')
        ev = c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL where start < ? and end > ? order by start",
                       (hi, lo)).fetchall()
        busy, cur_s, cur_e = 0, None, None
        for s, e in ev:
            s, e = max(s, lo), min(e, hi)
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    busy += cur_e-cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            busy += cur_e-cur_s
        print(f'  GPU busy (union of kernels): {busy/1e9:.2f} s = {100*busy/win:.1f}% of the window')
    if 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE' in T:
        print('\n== CUDA graph launches (graph trace): count, GPU duration ms p50 / p90 / mean, by stream')
        rows = c.execute("select streamId, end - start from CUPTI_ACTIVITY_KIND_GRAPH_TRACE where start >= ? and end <= ?",
                         (lo, hi)).fetchall()
        by = defaultdict(list)
        for sid, d in rows:
            by[sid].append(d)
        for sid, ds in sorted(by.items(), key=lambda kv: -len(kv[1])):
            ds.sort()
            n = len(ds)
            print(f'  stream {sid:4d} {n:7d} {ms(ds[n//2]):7.3f} {ms(pct(ds, .9)):7.3f} {ms(sum(ds)/n):7.3f}')
    # ---- step graph: replay call to sync end (the actor's g.replay start to g.sync end)
    ev = c.execute("select start, end, text from NVTX_EVENTS where globalTid = ? and text in ('g.replay','g.sync') "
                   "and end is not null and start >= ? and end <= ? order by start", (actor_tid, lo, hi)).fetchall()
    spans = []
    t_rep = None
    for s, e, t in ev:
        if t == 'g.replay':
            t_rep = s
        elif t_rep is not None:
            spans.append(e-t_rep); t_rep = None
    if spans:
        spans.sort(); n = len(spans)
        print(f'\n== actor step: replay call to sync end (ms): n {n}, p50 {ms(spans[n//2]):.3f}, p90 {ms(pct(spans, .9)):.3f}, '
              f'mean {ms(sum(spans)/n):.3f}')


if __name__ == '__main__':
    main()
