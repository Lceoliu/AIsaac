"""B58 step 0 / later 24-update runs: per run, the medians of u2-u23 and u10-u23 (collect, update, x real time), the
first and last update, the driver's time by phase (median per collect over u2-u23), the CF counters at the end.
usage: b58_ab_summary.py <run dir> [<run dir> ...]   (each a prof2 output directory with progress.csv)"""
import csv
import statistics
import sys

PHASES = ('poll', 'prep', 'sync', 'reply', 'pages', 'cons', 'push', 'mlane', 'lane', 'hold', 'pause', 'loop')
CF = ('cf_queued', 'cf_done', 'cf_nobank', 'cf_dropped', 'cf_skipped', 'cf_replaced')


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def med(rows, key, lo, hi):
    xs = [num(r.get(key)) for r in rows if lo <= int(r['update']) <= hi]
    xs = [x for x in xs if x == x]
    return statistics.median(xs) if xs else float('nan')


def main():
    for d in sys.argv[1:]:
        with open(f'{d}/progress.csv') as f:
            rows = list(csv.DictReader(f))
        n = len(rows)
        print(f'== {d}: {n} updates, {num(rows[-1]["game_hours"]):.2f} game hours' if rows else f'== {d}: no updates')
        if not rows:
            continue
        hi = min(23, n-1)
        for lo in (2, 10):
            print(f'  u{lo}-u{hi} medians: collect {med(rows, "collect_s", lo, hi):.2f} s, update {med(rows, "update_s", lo, hi):.2f} s, '
                  f'x real time {med(rows, "x_real_time", lo, hi):.1f}, rows/call {med(rows, "rows_per_call", lo, hi):.2f}')
        for r in (rows[0], rows[-1]):
            print(f'  u{r["update"]}: collect {num(r["collect_s"]):.2f} s, update {num(r["update_s"]):.2f} s, x {num(r["x_real_time"]):.1f}')
        ph = {p: med(rows, f'drv_{p}_s', 2, hi) for p in PHASES}
        print('  driver s per collect (u2-u%d medians): ' % hi + ', '.join(f'{p} {v:.2f}' for p, v in ph.items()))
        print(f'  driver steps {med(rows, "drv_steps", 2, hi):.0f}, polls {med(rows, "drv_polls", 2, hi):.0f} '
              f'({med(rows, "drv_idle_polls", 2, hi):.0f} idle), lane jobs {med(rows, "drv_lane_jobs", 2, hi):.0f}, '
              f'lane-only calls {med(rows, "drv_lane_calls", 2, hi):.0f}, merged rows {med(rows, "drv_merged_rows", 2, hi):.0f} per collect')
        last = rows[-1]
        print('  CF at the end: ' + ', '.join(f'{k} {last.get(k, "")}' for k in CF))
        tot = lambda k: sum(num(r.get(k)) for r in rows if num(r.get(k)) == num(r.get(k)))
        print(f'  totals: runtime captures {tot("drv_captures"):.0f}, lane jobs in Python {tot("drv_lane_python"):.0f}, '
              f'deferred {tot("drv_lane_deferred_requests"):.0f}, orphan {tot("drv_lane_orphan_requests"):.0f}, '
              f'Python lane rows {tot("drv_py_lane_rows")+tot("drv_py_lane_flushed_rows"):.0f}, paused {tot("drv_pause_s"):.2f} s')


if __name__ == '__main__':
    main()
