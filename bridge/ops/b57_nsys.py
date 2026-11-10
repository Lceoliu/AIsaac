"""B57: the cycle budget of the C++ actor path from an Nsight Systems report (nvtx + python-gil) of train_tok, plus the
per-thread CPU samples of b57_threads.py.

usage: python3 b57_nsys.py <report.nsys-rep | .sqlite> [threads.csv] [--nsys /usr/local/cuda/bin/nsys]

Prints, over the B50window capture window (hand-offs a..b, ISAAC_NSYS_RANGE):
  1. per thread of the trainer process (every thread with python-gil events, by name): GIL held / waited, per cycle;
  2. the learner's segments (learner.after_*, forward_packed, host_decode) split by what the actor's main thread was
     doing meanwhile: collecting (actor.collect) or waiting for the learner (h.wait_learner): count, total s, mean ms;
     the difference of the means is the contention cost per segment;
  3. the main thread's hand-off segments (h.*) and consumer ranges (c.record, actor.tick, lane.*) per cycle;
  4. with threads.csv: CPU seconds per thread and of the workers, core busy shares, over the window's wall time.
Standard library only."""
import bisect
import csv
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict


def export(path, nsys):
    if path.endswith('.sqlite'):
        return path
    out = os.path.splitext(path)[0]+'.sqlite'
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(path):
        subprocess.run([nsys, 'export', '--type', 'sqlite', '--force-overwrite', 'true', '-o', out, path], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def main():
    args = sys.argv[1:]
    nsys = '/usr/local/cuda/bin/nsys'
    if '--nsys' in args:
        i = args.index('--nsys'); nsys = args[i+1]; del args[i:i+2]
    db = export(args[0], nsys)
    threads_csv = args[1] if len(args) > 1 else None
    c = sqlite3.connect(db)
    S = dict(c.execute('select id, value from StringIds'))
    txt = "coalesce(text, (select value from StringIds where id = textId))"
    row = c.execute("select start, end from NVTX_EVENTS where text = 'B50window' and end is not null order by start limit 1").fetchone()
    if row is None:
        print('no B50window range'); return
    lo, hi = row
    main_tid = c.execute("select globalTid from NVTX_EVENTS where text = 'actor.collect' group by globalTid order by count(*) desc limit 1").fetchone()
    learner_tid = c.execute("select globalTid from NVTX_EVENTS where text like 'learner.%' group by globalTid order by count(*) desc limit 1").fetchone()
    if not main_tid or not learner_tid:
        print('no actor.collect / learner.* ranges (ISAAC_BLOCK_NVTX=1?)'); return
    main_tid, learner_tid = main_tid[0], learner_tid[0]
    pid = main_tid >> 24

    def ranges(tid, name, like=False):
        op = 'like' if like else '='
        return c.execute(f"select start, end, {txt} from NVTX_EVENTS where globalTid = ? and {txt} {op} ? and end is not null "
                         "and end > ? and start < ? order by start", (tid, name, lo, hi)).fetchall()

    collects = ranges(main_tid, 'actor.collect')
    n_cyc = sum(1 for s, e, _ in collects if s >= lo and e <= hi) or max(len(collects), 1)
    win = (hi-lo)/1e9
    print(f'window {win:.1f} s, {n_cyc} complete collects in it (per-cycle numbers divide by {n_cyc}); pid {pid}')

    # ---- 1. GIL per thread
    names = {}
    if 'ThreadNames' in {r[0] for r in c.execute("select name from sqlite_master where type='table'")}:
        for name_id, gtid in c.execute('select nameId, globalTid from ThreadNames'):
            names[gtid] = S.get(name_id, str(name_id))
    gil = {v: k for k, v in S.items() if v in ('Holding GIL', 'Waiting for GIL')}
    print('\n== 1. GIL per thread of the trainer (s over the window; per cycle)')
    print(f'  {"thread":28s} {"held":>8s} {"wait":>8s} {"held/cyc":>9s} {"wait/cyc":>9s}')
    rows = c.execute("select globalTid, textId, sum(min(end, ?) - max(start, ?)) from NVTX_EVENTS where (globalTid >> 24) = ? "
                     "and textId in (?, ?) and end is not null and end > ? and start < ? group by globalTid, textId",
                     (hi, lo, pid, gil.get('Holding GIL', -1), gil.get('Waiting for GIL', -1), lo, hi)).fetchall()
    per = defaultdict(lambda: [0.0, 0.0])
    for gtid, tid_, s_ in rows:
        per[gtid][0 if tid_ == gil.get('Holding GIL') else 1] += s_/1e9
    tot_other = [0.0, 0.0]
    for gtid, (h, w) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        label = ('MAIN ' if gtid == main_tid else 'LEARNER ' if gtid == learner_tid else '')+names.get(gtid, str(gtid & 0xffffff))
        print(f'  {label[:28]:28s} {h:8.2f} {w:8.2f} {h/n_cyc:9.2f} {w/n_cyc:9.2f}')
        if gtid != learner_tid:
            tot_other[0] += h; tot_other[1] += w
    print(f'  {"all but the learner":28s} {tot_other[0]:8.2f} {tot_other[1]:8.2f} {tot_other[0]/n_cyc:9.2f} {tot_other[1]/n_cyc:9.2f}')
    # (B58e: the main thread's entries into Python = its GIL acquisitions: Holding GIL intervals, per cycle)
    n_hold, = c.execute("select count(*) from NVTX_EVENTS where globalTid = ? and textId = ? and end is not null and start >= ? "
                        "and end <= ?", (main_tid, gil.get('Holding GIL', -1), lo, hi)).fetchone()
    print(f'  main thread GIL acquisitions (Python entries): {n_hold} in the window, {n_hold/n_cyc:.0f} per cycle')

    # ---- 2. learner segments by the actor's window
    waits = ranges(main_tid, 'h.wait_learner')
    win_list = sorted([(s, e, 'collect') for s, e, _ in collects]+[(s, e, 'waiting') for s, e, _ in waits])
    starts = [w[0] for w in win_list]

    def overlap(s, e, kind):
        tot = 0
        i = max(0, bisect.bisect_right(starts, s)-1)
        while i < len(win_list) and win_list[i][0] < e:
            ws, we, k = win_list[i]
            if k == kind:
                tot += max(0, min(e, we)-max(s, ws))
            i += 1
        return tot

    seg = defaultdict(lambda: {'collect': [0, 0.0], 'waiting': [0, 0.0], 'other': [0, 0.0]})
    for s, e, name in c.execute(f"select start, end, {txt} from NVTX_EVENTS where globalTid = ? and end is not null and start >= ? "
                                f"and end <= ? and ({txt} like 'learner.%')", (learner_tid, lo, hi)).fetchall():
        d = e-s
        oc, ow = overlap(s, e, 'collect'), overlap(s, e, 'waiting')
        k = 'collect' if oc > 0.5*d else 'waiting' if ow > 0.5*d else 'other'
        seg[name][k][0] += 1; seg[name][k][1] += d/1e9
    print('\n== 2. learner segments by what the actor did meanwhile: count, s, mean ms (collect | waiting | mixed)')
    for name in sorted(seg, key=lambda n: -sum(v[1] for v in seg[n].values())):
        v = seg[name]
        f = lambda k: f'{v[k][0]:5d} {v[k][1]:8.2f} {1e3*v[k][1]/max(v[k][0], 1):9.1f}'
        print(f'  {name[:30]:30s} {f("collect")} | {f("waiting")} | {f("other")}')

    # ---- 3. main thread ranges per cycle
    print('\n== 3. main thread ranges (per cycle s, calls, mean ms, max ms)')
    agg = defaultdict(lambda: [0, 0.0, 0.0])
    for s, e, name in c.execute(f"select start, end, {txt} from NVTX_EVENTS where globalTid = ? and end is not null and start >= ? "
                                f"and end <= ? and coalesce(textId, -9) not in (?, ?)",
                                (main_tid, lo, hi, gil.get('Holding GIL', -1), gil.get('Waiting for GIL', -1))).fetchall():
        if name in ('Holding GIL', 'Waiting for GIL', 'B50window') or name is None:
            continue
        a = agg[name]; d = (e-s)/1e9
        a[0] += 1; a[1] += d; a[2] = max(a[2], d)
    for name, (n, t, mx) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:25]:
        print(f'  {name[:30]:30s} {t/n_cyc:8.3f} {n:7d} {1e3*t/max(n, 1):9.2f} {1e3*mx:9.1f}')

    # ---- 4. CPU per thread (b57_threads.py samples)
    if threads_csv and os.path.exists(threads_csv):
        # the window's wall clock: the B50window range is relative to nsys's session start; the samples carry unix time.
        # (map by the session's start: TARGET_INFO_SESSION_START_TIME when present, else the samples' whole span)
        t0 = t1 = None
        try:
            base = c.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
            t0, t1 = (base+lo)/1e9, (base+hi)/1e9
        except sqlite3.Error:
            pass
        data = defaultdict(list)
        with open(threads_csv) as f:
            for r in csv.DictReader(f):
                data[(r['kind'], r['name'])].append((float(r['t']), int(r['ticks'])))
        hz = os.sysconf('SC_CLK_TCK') if hasattr(os, 'sysconf') else 100

        def delta(series):
            pts = [p for p in series if (t0 is None or t0 <= p[0] <= t1)]
            if len(pts) < 2:
                return 0.0, 0.0
            return (pts[-1][1]-pts[0][1])/hz, pts[-1][0]-pts[0][0]
        print(f'\n== 4. CPU over the window ({"mapped" if t0 else "whole run"}): seconds, cores (seconds / wall)')
        th = sorted(((delta(v)[0], delta(v)[1], k[1]) for k, v in data.items() if k[0] == 'thread'), reverse=True)
        for sec, wall, name in th[:20]:
            print(f'  {name[:32]:32s} {sec:8.1f} {sec/max(wall, 1e-9):6.2f}')
        for k, v in data.items():
            if k[0] == 'workers':
                sec, wall = delta(v)
                print(f'  {"workers + games (all)":32s} {sec:8.1f} {sec/max(wall, 1e-9):6.2f}')
        cores = sorted((delta(v)[0]/max(delta(v)[1], 1e-9), k[1]) for k, v in data.items() if k[0] == 'core')
        if cores:
            busy = [b for b, _ in cores]
            print(f'  cores: {len(busy)}, busy share min {min(busy):.2f} median {busy[len(busy)//2]:.2f} max {max(busy):.2f}, '
                  f'sum {sum(busy):.1f} cores busy')


if __name__ == '__main__':
    main()
