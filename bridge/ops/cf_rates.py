"""The planner's CF yardstick for the B58d timing reports (per run): collect / update / wait_learner medians from u10,
the u15 CF counters, lane calls per second, CF points done per second and per game hour (u15 totals over the first 15
updates' wall seconds and game hours), and the lane / point timing p50 / p90 medians (LANE_TIMING_FIELDS).
usage: cf_rates.py <run dir> [<run dir> ...]"""
import csv
import statistics
import sys

TIMING = ('lt_window_s', 'lt_driver_s', 'lt_gpu_s', 'lt_rt_s', 'pt_first_s', 'pt_run_s', 'pt_life_s', 'pt_life_v')


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def med(rows, k, lo=10):
    xs = [num(r.get(k)) for r in rows if int(r['update']) >= lo]
    xs = [x for x in xs if x == x]
    return statistics.median(xs) if xs else float('nan')


def main():
    for d in sys.argv[1:]:
        with open(f'{d}/progress.csv') as f:
            rows = list(csv.DictReader(f))
        print(f'== {d}: {len(rows)} updates')
        if len(rows) < 15:
            print('  (fewer than 15 updates)')
            continue
        u = rows[14]
        walls = [num(r.get('wall_s')) for r in rows[:15]]
        wall = walls[-1] if all(walls[i] <= walls[i+1] for i in range(14)) else sum(walls)   # (cumulative or per update)
        gh = num(u['game_hours'])
        lane, done = num(u.get('cf_lane_calls')), num(u.get('cf_done'))
        print(f'  u10+ medians: collect {med(rows, "collect_s"):.2f} s, update {med(rows, "update_s"):.2f} s, '
              f'wait_learner {med(rows, "wait_learner_s"):.2f} s')
        print('  u15: ' + ', '.join(f'{k} {u.get(k, "")}' for k in ('cf_done', 'cf_nobank', 'cf_lane_calls', 'cf_queue')))
        print(f'  rates over u1-u15 ({wall:.0f} s wall, {gh:.2f} game hours): lane calls {lane/wall:.0f}/s, done {done/wall:.3f}/s, '
              f'{done/gh:.2f}/game hour')
        both = [num(r.get('pt_all_pinned_s')) for r in rows[1:]] if 'pt_all_pinned_s' in rows[0] and rows[1].get('pt_all_pinned_s', '') != ''             else [num(r.get('d1_gen_both_s')) for r in rows[1:]]   # (native points / the Python lane service's)
        both = [x for x in both if x == x]
        life = ('pt_life_s_p50', 'pt_life_s_p90') if rows[1].get('pt_life_s_p50', '') != '' or rows[2].get('pt_life_s_p50', '') != ''             else ('d1_life_p50_s', 'd1_life_p90_s')
        print(f'  points: life p50 {med(rows, life[0], 2):.1f} s, p90 {med(rows, life[1], 2):.1f} s (u2+ medians of {life[0][:7]}), '
              f'every generation pinned {statistics.median(both) if both else float("nan"):.1f} s per collect (median)')
        print('  timing medians (u2+): ' + ', '.join(f'{k} p50 {med(rows, k+"_p50", 2):.3f} p90 {med(rows, k+"_p90", 2):.3f}'
                                                   for k in TIMING))


if __name__ == '__main__':
    main()
