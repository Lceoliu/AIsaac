"""B38: medians of the speed columns of runs' progress.csv (the first --skip updates left out: compile, warm-up),
the GPU monitor's means (<run>.gpu of hpc_b38.sh) and the learning numbers (kl, ev, clip_frac) per run.
usage: python b38_summary.py [--skip 8] <run dir> ...   (prints a markdown table)"""
import argparse
import csv
import re
import statistics
from pathlib import Path

COLS = ('x_real_time', 'collect_s', 'update_s', 'wait_learner_s', 'wall_s', 'infer_ms', 'infer_upd_ms', 'idle_share',
        'stale_share', 'gpu_reserved_gb', 'gpu_reserved_gb_actor', 'handoff_mb', 'handoff_s', 'take_weights_s',
        'c_trainer_cores', 'cpu_collect', 'cpu_update', 'kl', 'explained_variance', 'clip_frac', 'entropy')


def med(rows, c):
    v = [float(r[c]) for r in rows if r.get(c) not in (None, '')]
    return statistics.median(v) if v else None


def gpu_means(path):
    """mean memory (MiB) and utilisation (%) per GPU index, mean load, from the monitor lines."""
    if not path.is_file():
        return {}
    acc = {}
    loads = []
    for line in path.read_text().splitlines():
        for g, m, u in re.findall(r'(\d+), (\d+) MiB, (\d+) %', line):
            acc.setdefault(int(g), []).append((float(m), float(u)))
        got = re.search(r'load ([\d.]+)', line)
        if got:
            loads.append(float(got.group(1)))
    out = {f'gpu{g}': (max(m for m, _ in v), statistics.mean(u for _, u in v)) for g, v in acc.items()}
    if loads:
        out['load'] = statistics.mean(loads)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=8)
    p.add_argument('runs', nargs='+')
    a = p.parse_args()
    print('| run | updates | ' + ' | '.join(COLS) + ' | GPU1 max MiB / mean util | GPU2 max MiB / mean util | load |')
    print('|' + '---|' * (len(COLS) + 5))
    for r in a.runs:
        path = Path(r)
        with open(path / 'progress.csv', newline='') as f:
            rows = list(csv.DictReader(f))
        use = rows[a.skip:]
        vals = []
        for c in COLS:
            m = med(use, c)
            vals.append('' if m is None else (f'{m:.0f}' if c == 'x_real_time' else f'{m:.3g}'))
        g = gpu_means(Path(str(path) + '.gpu'))
        gs = [f'{g[k][0]:.0f} / {g[k][1]:.0f}%' if k in g else '' for k in ('gpu1', 'gpu2')]
        print(f'| {path.name} | {len(use)} of {len(rows)} | ' + ' | '.join(vals) + ' | ' + ' | '.join(gs) +
              f" | {g.get('load', 0):.0f} |")


if __name__ == '__main__':
    main()
