"""Truncated-memory runs: per update the learner sections, audit columns and TRUNC_PROFILE lines; steady medians.
usage: python -X utf8 trunc_table.py <run dir> [first steady row (default 3)]"""
import csv, json, statistics, sys

d = sys.argv[1]
first = int(sys.argv[2]) if len(sys.argv) > 2 else 3
rows = list(csv.DictReader(open(f'{d}/progress.csv')))
cols = ['update', 'x_real_time', 'game_hours', 'wall_s', 'collect_s', 'update_s', 'wait_learner_s', 'u_trunc_plan',
        'u_recompute', 'u_gae', 'u_batch', 'u_teach_new', 'u_ppo', 'u_teach_build', 'u_teach', 'u_opt', 'u_tail',
        'memory_constant_grad_count', 'memory_grad_window_mean', 'memory_carry_mismatch', 'gpu_reserved_gb',
        'source_horizon_dropped_grad', 'v2_step', 'teach_steps']
have = [c for c in cols if c in rows[0]]
print('| ' + ' | '.join(have) + ' |')
print('|' + '---|' * len(have))
for r in rows:
    print('| ' + ' | '.join(r.get(c, '') for c in have) + ' |')
steady = rows[first:]
if steady:
    med = {}
    for c in have[1:]:
        try:
            med[c] = statistics.median(float(r[c]) for r in steady if r.get(c) not in (None, ''))
        except (ValueError, statistics.StatisticsError):
            pass
    print(f'\nmedians over rows {first}..{len(rows) - 1} ({len(steady)}):', json.dumps(med, indent=1))
for line in open(f'{d}/train.log', errors='replace'):
    if line.startswith('TRUNC_PROFILE '):
        print(line.strip())
