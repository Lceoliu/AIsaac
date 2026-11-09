"""B50 phase 0: per-thread and per-phase breakdown of an Nsight Systems report of train_tok (block memory).

usage: python3 b50_nsys.py <report.nsys-rep | report.sqlite> [--nsys /usr/local/cuda/bin/nsys]

The report comes from `nsys profile -t cuda,nvtx,python-gil --cpuctxsw=process-tree --sample=process-tree
--python-sampling=true --capture-range=cudaProfilerApi` of train_tok.py run with ISAAC_BLOCK_NVTX=1 and
ISAAC_NSYS_RANGE=a:b (isaac_bridge/block_nvtx.py). Prints, for the actor (main) thread and the learner thread of
the train_tok process over the capture window: GIL held / GIL waited (python-gil ranges), CUDA synchronisation
waits (CUPTI runtime API cudaStreamSynchronize / cudaEventSynchronize / cudaDeviceSynchronize), running / blocked
time (sched events), and the NVTX ranges of block_nvtx (inclusive and exclusive sums, per-call ms). Only the
standard library is used (sqlite3)."""
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
    if not rows:
        print('no actor.poll NVTX ranges: was ISAAC_BLOCK_NVTX=1 set?'); return
    actor_tid = max(rows, key=lambda r: r[1])[0]
    pid = actor_tid >> 24   # globalTid = (pid << 24) | tid on Linux targets
    rows = c.execute("select globalTid, count(*) from NVTX_EVENTS where text like 'learner.%' group by globalTid").fetchall()
    learner_tid = max(rows, key=lambda r: r[1])[0] if rows else None
    # ---- the capture window: first to last event of the process
    lo, hi = c.execute("select min(start), max(coalesce(end, start)) from NVTX_EVENTS where (globalTid >> 24) = ?",
                       (pid,)).fetchone()
    if 'CUPTI_ACTIVITY_KIND_RUNTIME' in T:
        lo2, hi2 = c.execute("select min(start), max(end) from CUPTI_ACTIVITY_KIND_RUNTIME where (globalTid >> 24) = ?",
                             (pid,)).fetchone()
        if lo2:
            lo, hi = min(lo, lo2), max(hi, hi2)
    win = hi-lo
    print(f'process pid {pid}: capture window {win/1e9:.2f} s; actor tid {actor_tid & 0xffffff}, '
          f'learner tid {learner_tid & 0xffffff if learner_tid else None}')
    threads = [('actor', actor_tid)] + ([('learner', learner_tid)] if learner_tid else [])

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
            # SCHED_EVENTS: start, cpu, isSchedIn, globalTid: a thread runs from a sched-in to its next sched-out
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
        stack = []
        for s, e, t in ev:
            while stack and stack[-1][1] <= s:
                stack.pop()
            d = e-s
            incl[t] += d; cnt[t] += 1
            excl[t] += d
            if stack:
                excl[stack[-1][2]] -= d
            stack.append((s, e, t))
        for t in sorted(incl, key=lambda k: -incl[k]):
            print(f'  {name:8s} {t:26s} {cnt[t]:8d} {incl[t]/1e9:9.3f} {excl[t]/1e9:9.3f} {ms(incl[t])/max(cnt[t],1):8.3f}')

    # ---- GPU: kernels / graphs / memcpy per stream (window), utilisation
    if 'CUPTI_ACTIVITY_KIND_KERNEL' in T:
        print('\n== GPU kernels per stream (window): stream, launches, busy s (sum of durations), share of window')
        rows = c.execute("select streamId, count(*), sum(min(end, ?) - max(start, ?)) from CUPTI_ACTIVITY_KIND_KERNEL "
                         "where start < ? and end > ? group by streamId order by 3 desc", (hi, lo, hi, lo)).fetchall()
        for sid, n, tot in rows:
            print(f'  stream {sid:4d} {n:9d} {tot/1e9:9.3f} {100*tot/win:6.1f}%')
        # union of all kernels (GPU busy at all)
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
            print(f'  stream {sid:4d} {n:7d} {ms(ds[n//2]):7.3f} {ms(ds[int(n*.9)]):7.3f} {ms(sum(ds)/n):7.3f}')
    # ---- step graph: launch to sync (the actor's g.replay start to g.sync end) vs GPU duration
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
        print(f'\n== actor step: replay call to sync end (ms): n {n}, p50 {ms(spans[n//2]):.3f}, p90 {ms(spans[int(n*.9)]):.3f}, '
              f'mean {ms(sum(spans)/n):.3f}')
    # ---- Python sampling: top frames per thread (domain 'Python Backtrace' events carry a jsonText callchain)
    if 'SAMPLING_CALLCHAINS' in T and 'COMPOSITE_EVENTS' in T:
        print('\n== CPU samples per thread (window): count')
        rows = c.execute("select globalTid, count(*) from COMPOSITE_EVENTS where start >= ? and start <= ? and (globalTid >> 24) = ? "
                         "group by globalTid order by 2 desc limit 6", (lo, hi, pid)).fetchall()
        for tid, n in rows:
            tag = {actor_tid: 'actor', learner_tid: 'learner'}.get(tid, '')
            print(f'  tid {tid & 0xffffff:8d} {tag:8s} {n:8d}')


if __name__ == '__main__':
    main()
