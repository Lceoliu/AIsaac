"""bm1: the learner segments' table of a run (train_tok with ISAAC_BLOCK_TIMING=1: <run dir>/segments.jsonl).

  python3 seg_table.py <run dir> [first update=10]

Per segment, the medians over updates >= first of: host (s per update), span (s per update), lag (mean per instance, s),
count per update; sorted by span. host: the learner thread's time in the segment (no synchronisation); span: the
learner stream's time from the segment's start event to its end event (stream idle gaps included); lag: how long the
segment's last work was still queued when the host left it (negative: the stream had already finished)."""
import json
import statistics as st
import sys


def main():
    d = sys.argv[1]
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rows = [json.loads(x) for x in open(f'{d}/segments.jsonl') if x.strip()]
    rows = [r for r in rows if r['update'] >= first]
    names = sorted({k for r in rows for k in r['segments']})
    tab = []
    for k in names:
        v = [r['segments'][k] for r in rows if k in r['segments']]
        tab.append((k, st.median(x[0] for x in v), st.median(x[1] for x in v), st.median(x[2] for x in v),
                    st.median(x[3] for x in v), len(v)))
    tab.sort(key=lambda t: -t[2])
    print(f'{d}: updates {rows[0]["update"] if rows else "-"}..{rows[-1]["update"] if rows else "-"} ({len(rows)}), '
          f'medians per update')
    print(f'{"segment":24s} {"host s":>8s} {"span s":>8s} {"lag s":>8s} {"n":>5s}')
    for k, h, s, lag, n, m in tab:
        print(f'{k:24s} {h:8.3f} {s:8.3f} {lag:8.3f} {n:5.0f}')
    print(f'{"sum":24s} {sum(t[1] for t in tab):8.3f} {sum(t[2] for t in tab):8.3f}')


if __name__ == '__main__':
    main()
