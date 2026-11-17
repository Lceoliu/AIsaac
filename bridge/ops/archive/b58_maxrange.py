"""B58d: what happened inside the longest occurrence of a main-thread NVTX range (e.g. lane.submit's 204 ms in
prof-b58c-m): the ranges nested in it, the main thread's GIL holding / waiting inside it, the CUDA runtime calls the main
thread made inside it, the learner's ranges overlapping it. Reads the .sqlite b57_nsys.py exported.
usage: python3 b58_maxrange.py <report.sqlite> <range name> [top k = 3]"""
import sqlite3
import sys
from collections import defaultdict


def main():
    db, name = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    c = sqlite3.connect(db)
    txt = "coalesce(text, (select value from StringIds where id = textId))"
    main_tid = c.execute("select globalTid from NVTX_EVENTS where text = 'actor.collect' group by globalTid order by count(*) desc limit 1").fetchone()[0]
    learner = c.execute("select globalTid from NVTX_EVENTS where text like 'learner.%' group by globalTid order by count(*) desc limit 1").fetchone()
    top = c.execute(f"select start, end from NVTX_EVENTS where globalTid = ? and {txt} = ? and end is not null order by end-start desc limit ?",
                    (main_tid, name, k)).fetchall()
    for s, e in top:
        print(f'== {name} {1e3*(e-s)/1e9:.1f} ms at {s/1e9:.3f} s')
        agg = defaultdict(lambda: [0, 0.0])
        for s2, e2, n2 in c.execute(f"select start, end, {txt} from NVTX_EVENTS where globalTid = ? and start >= ? and end <= ? "
                                    "and end is not null", (main_tid, s, e)).fetchall():
            if n2 == name and s2 == s:
                continue
            a = agg[n2 or '?']; a[0] += 1; a[1] += (e2-s2)/1e9
        for n2, (cnt, tot) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:12]:
            print(f'  inside: {n2[:40]:40s} {cnt:5d} x {1e3*tot:8.1f} ms')
        try:
            rt = c.execute("select s.value, count(*), sum(r.end-r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId "
                           "where r.globalTid = ? and r.start >= ? and r.end <= ? group by s.value order by 3 desc limit 6",
                           (main_tid, s, e)).fetchall()
            for n2, cnt, tot in rt:
                print(f'  cuda runtime: {n2[:40]:40s} {cnt:5d} x {1e3*tot/1e9:8.1f} ms')
        except sqlite3.Error as ex:
            print(f'  (no CUDA runtime table: {ex})')
        if learner:
            ov = c.execute(f"select {txt}, max(start, ?), min(end, ?) from NVTX_EVENTS where globalTid = ? and end > ? and start < ? "
                           "and end is not null", (s, e, learner[0], s, e)).fetchall()
            la = defaultdict(float)
            for n2, a_, b_ in ov:
                la[n2 or '?'] += (b_-a_)/1e9
            print('  learner meanwhile: ' + ', '.join(f'{n2} {1e3*v:.0f} ms' for n2, v in sorted(la.items(), key=lambda kv: -kv[1])[:6]))


if __name__ == '__main__':
    main()
