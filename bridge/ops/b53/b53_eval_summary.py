"""B53: the held-out evaluation's metrics with uncertainty, and the evaluation's resolution.

  python b53_eval_summary.py <eval_dir> [<eval_dir> ...]          (repeats of one checkpoint, or several checkpoints)

Per directory (eval_tok --mode run output: episodes.json, summary.json): episodes, floor clear rate (runs with >= 1
floor cleared), floors cleared per run, rooms cleared per run, hurts per run (half hearts lost), deaths, game hours,
x real time; per-episode bootstrap 95% CI of each mean. Across directories (repeats of the same checkpoint): the mean
and SD of the per-repeat means (evaluation-to-evaluation noise). Resolution: the smallest arm-vs-arm difference of
each per-run metric detectable with 512 / 1024 / 2048 episodes per arm (two-sided alpha 0.05, power 0.8:
2.80 * sqrt(2 * var / n), var the per-episode variance; for rates the binomial variance), and the episodes per arm
needed for differences of 0.1 / 0.25 / 0.5 per-episode SD."""
import json
import sys
from pathlib import Path

import numpy as np

METRICS = ('floor_clear', 'floors_cleared', 'rooms_cleared', 'hurt', 'death')


def per_episode(eps):
    return dict(floor_clear=np.array([float(e.get('exits', 0) >= 1) for e in eps]),
                floors_cleared=np.array([float(e.get('exits', 0)) for e in eps]),
                rooms_cleared=np.array([float(e.get('rooms', 0)) for e in eps]),
                hurt=np.array([float(e.get('hurt', 0)) for e in eps]),
                death=np.array([float(e.get('done') == 2) for e in eps]))


def boot_ci(x, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) == 0:
        return (float('nan'), float('nan'))
    m = rng.choice(x, (reps, len(x)), replace=True).mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def main():
    dirs = [Path(d) for d in sys.argv[1:]]
    out = dict(runs={})
    pooled = {k: [] for k in METRICS}
    for d in dirs:
        eps = json.loads((d/'episodes.json').read_text())
        summ = json.loads((d/'summary.json').read_text()) if (d/'summary.json').is_file() else {}
        pe = per_episode(eps)
        r = dict(episodes=len(eps), game_hours=summ.get('game_hours'), x_real_time=summ.get('x_real_time'),
                 seconds=summ.get('seconds'))
        for k, v in pe.items():
            r[k] = dict(mean=float(v.mean()) if len(v) else None, ci95=boot_ci(v))
            pooled[k].append(v)
        out['runs'][d.name] = r
    if len(dirs) > 1:
        out['repeat_spread'] = {k: dict(mean=float(np.mean([v.mean() for v in pooled[k]])),
                                        sd_between_repeats=float(np.std([v.mean() for v in pooled[k]], ddof=1)))
                                for k in METRICS}
    allv = {k: np.concatenate(v) for k, v in pooled.items()}
    res = {}
    for k, v in allv.items():
        var = float(v.var(ddof=1)) if len(v) > 1 else float('nan')
        sd = var**0.5
        res[k] = dict(per_episode_sd=sd,
                      mde={str(n): 2.80*np.sqrt(2*var/n) for n in (512, 1024, 2048)},
                      episodes_per_arm_for={f'{f}sd': int(np.ceil(2*(2.80/f)**2)) for f in (0.1, 0.25, 0.5)})
    out['resolution'] = res
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
