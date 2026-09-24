"""Paired simulator vs AB+ comparison of one checkpoint on the same arena seeds.

Inputs: the simulator audit's per-seed results (runs/ppo-audit/20260923-e1e2/e2/results/<ckpt>/<mode>/
batch-*.json) and abplus_eval.py's results.jsonl. Episodes pair by arena seed (same layout, entrance
and Monstro spawn); the Monstro RNG streams differ between the engines, so a pair shares the start,
not the trajectory.

usage: python abplus_transfer_report.py --sim DIR --abplus results.jsonl [--out summary.json]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

OUTCOMES = ('win', 'death', 'time_limit')
METRICS = ('frames', 'damage_frac', 'boss_hp_final', 'hurt_events', 'first_hurt_s', 'hits_per_min', 'mean_dist',
           'close_frac', 'far_frac', 'shoot_frac', 'idle_move_frac', 'aim_axis_ok', 'bombs_used', 'mean_projectiles')


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def mcnemar(a_only, b_only):
    """Exact two-sided binomial test on the discordant pairs."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def rates(rows):
    n = len(rows)
    out = {'n': n}
    for o in OUTCOMES:
        k = sum(r['outcome'] == o for r in rows)
        out[o] = {'k': k, 'rate': round(k / n, 4) if n else None, 'ci95': wilson(k, n)}
    return out


def mean(values):
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 4) if values else None


def animation_segments(samples):
    """[(frame, anim, aframe)] sampled every action -> [(anim, first_frame, frames)] of contiguous plays.

    A new segment starts when the animation name changes or its frame index goes backwards (replay).
    Durations are in logic frames between the first and last sample plus one sampling step.
    """
    out, step = [], 2
    for frame, anim, aframe in samples:
        if out and out[-1][0] == anim and aframe >= out[-1][3]:
            out[-1][2], out[-1][3] = frame, aframe
        else:
            out.append([anim, frame, frame, aframe])
    return [(a, first, last - first + step) for a, first, last, _ in out]


def monstro_samples(path):
    import gzip
    rows = [json.loads(x) for x in gzip.open(path, 'rt')]
    samples = []
    if rows[0].get('metadata', {}).get('format') == 'abplus-raw-obs-v1':
        origin = rows[1]['obs']['logic_frames']
        for r in rows[1:]:
            m = [e for e in r['obs']['entities'] if e['type'] == 20]
            if m:
                samples.append((r['obs']['logic_frames'] - origin, m[0]['anim'], m[0]['aframe']))
    else:
        for r in rows[1:]:
            b = [b for b in r['state']['bosses'] if not b.get('dead')]
            if b:
                samples.append((r['state']['frame'], b[0]['anim'], b[0]['frame']))
    return samples


def animation_table(paths):
    """Duration statistics per Monstro animation; the last (possibly cut) segment of an episode is dropped."""
    durations = {}
    for path in paths:
        segs = animation_segments(monstro_samples(path))[:-1]
        for i, (anim, first, frames) in enumerate(segs):
            durations.setdefault('Appear (episode start)' if anim == 'Appear' and i == 0 else anim, []).append(frames)
    return {a: {'n': len(v), 'mean': round(float(np.mean(v)), 1), 'p10': float(np.percentile(v, 10)),
                'median': float(np.median(v)), 'p90': float(np.percentile(v, 90))} for a, v in sorted(durations.items())}


def boss_hp_trace(path):
    """Boss HP after every action of a replay (absolute; AB+ from the visible boss bar x 250)."""
    import gzip
    rows = [json.loads(x) for x in gzip.open(path, 'rt')]
    trace, last = [], None
    if rows[0].get('metadata', {}).get('format') == 'abplus-raw-obs-v1':
        for r in rows[1:]:
            m = [e for e in r['obs']['entities'] if e['type'] == 20 and e.get('boss')]
            last = sum(e.get('boss_hp', 0.0) for e in m) * 250 if m else last
            trace.append(last)
    else:
        for r in rows[1:]:
            trace.append(sum(b['hp'] for b in r['state']['bosses'] if not b.get('dead')))
    return trace


def opening_bomb(paths, outcomes, steps=30, threshold=50.0):
    """Episodes whose boss loses >= threshold HP in one action within the first `steps` actions (a bomb:
    3.5 per tear, 60 AB+ / 100 Repentance per bomb), and the win rate with and without that hit."""
    hit = {}
    for seed, path in paths.items():
        trace = [v for v in boss_hp_trace(path)[:steps + 1] if v is not None]
        hit[seed] = any(a - b >= threshold for a, b in zip(trace, trace[1:]))
    yes = [s for s in hit if hit[s]]
    no = [s for s in hit if not hit[s]]
    win = lambda seeds: sum(outcomes[s] == 'win' for s in seeds)
    return {'episodes': len(hit), 'opening_bomb_hit': len(yes),
            'win_given_hit': [win(yes), len(yes)], 'win_given_no_hit': [win(no), len(no)]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sim', required=True, help='directory with the simulator batch-*.json files')
    p.add_argument('--abplus', required=True, help='abplus_eval.py results.jsonl')
    p.add_argument('--sim-replays', help='simulator replay directory (seed-<s>.jsonl.gz)')
    p.add_argument('--abplus-replays', nargs='+', help='abplus_eval.py replay directories (seed-<s>-<i>.jsonl.gz)')
    p.add_argument('--out')
    args = p.parse_args()
    sim = {}
    for path in sorted(Path(args.sim).glob('batch-*.json')):
        for r in json.loads(path.read_text())['results']:
            sim[r['seed']] = r
    abp = {}
    errors = 0
    for line in Path(args.abplus).read_text().splitlines():
        r = json.loads(line)
        if r['outcome'] == 'error':
            errors += 1
            continue
        abp.setdefault(r['seed'], r)  # first episode per seed (repeats are identical)
    seeds = sorted(set(sim) & set(abp))
    pairs = [(sim[s], abp[s]) for s in seeds]
    layouts = sorted({s['layout'] for s, _ in pairs})
    summary = {'paired_seeds': len(seeds), 'abplus_errors': errors,
               'sim': rates([s for s, _ in pairs]), 'abplus': rates([a for _, a in pairs])}
    sim_win = np.array([s['outcome'] == 'win' for s, _ in pairs])
    abp_win = np.array([a['outcome'] == 'win' for _, a in pairs])
    summary['paired_win'] = {'both': int((sim_win & abp_win).sum()), 'sim_only': int((sim_win & ~abp_win).sum()),
                             'abplus_only': int((~sim_win & abp_win).sum()), 'neither': int((~sim_win & ~abp_win).sum()),
                             'mcnemar_p': round(mcnemar(int((sim_win & ~abp_win).sum()), int((~sim_win & abp_win).sum())), 5)}
    summary['by_layout'] = {}
    for layout in layouts:
        sub = [(s, a) for s, a in pairs if s['layout'] == layout]
        summary['by_layout'][str(layout)] = {
            'n': len(sub),
            'sim_win': round(np.mean([s['outcome'] == 'win' for s, _ in sub]), 4),
            'abplus_win': round(np.mean([a['outcome'] == 'win' for _, a in sub]), 4),
            'sim_damage_frac': mean([s['damage_frac'] for s, _ in sub]),
            'abplus_damage_frac': mean([a['damage_frac'] for _, a in sub])}
    summary['metrics'] = {}
    for m in METRICS:
        sv, av = [s.get(m) for s, _ in pairs], [a.get(m) for _, a in pairs]
        diffs = [a - s for s, a in zip(sv, av) if s is not None and a is not None]
        summary['metrics'][m] = {'sim': mean(sv), 'abplus': mean(av), 'paired_diff': mean(diffs),
                                 'paired_n': len(diffs)}
    if args.sim_replays and args.abplus_replays:
        # One replay per seed (AB+ episodes of a seed are identical).
        by_seed = {}
        for d in args.abplus_replays:
            for q in sorted(Path(d).glob('seed-*.jsonl.gz')):
                by_seed.setdefault(int(q.name.split('-')[1].split('.')[0]), q)
        replay_seeds = sorted(by_seed)
        abp_paths = [by_seed[seed] for seed in replay_seeds]
        sim_paths = [Path(args.sim_replays) / f'seed-{seed}.jsonl.gz' for seed in replay_seeds]
        sim_by_seed = {seed: Path(args.sim_replays) / f'seed-{seed}.jsonl.gz' for seed in seeds}
        summary['opening_bomb'] = {
            'sim_all_seeds': opening_bomb({k: v for k, v in sim_by_seed.items() if v.exists()},
                                          {k: sim[k]['outcome'] for k in seeds}),
            'abplus_replayed_seeds': opening_bomb(dict(zip(replay_seeds, abp_paths)),
                                                  {k: abp[k]['outcome'] for k in replay_seeds if k in abp})}
        summary['monstro_animation_frames'] = {
            'seeds': len(replay_seeds),
            'sim': animation_table([q for q in sim_paths if q.exists()]),
            'abplus': animation_table(abp_paths)}
    text = json.dumps(summary, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    print()
    print('| metric | simulator | AB+ | paired diff (AB+ - sim) |')
    print('|---|---|---|---|')
    for o in OUTCOMES:
        s, a = summary['sim'][o], summary['abplus'][o]
        print(f"| {o} rate | {s['rate']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] | "
              f"{a['rate']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] | {a['rate'] - s['rate']:+.3f} |")
    for m, v in summary['metrics'].items():
        if v['sim'] is not None and v['abplus'] is not None:
            print(f"| {m} | {v['sim']:.3f} | {v['abplus']:.3f} | {v['paired_diff']:+.3f} |")
    bomb = summary.get('opening_bomb')
    if bomb:
        print()
        print('Opening bomb (>= 50 boss HP in one action within the first 30 actions)')
        for side, v in bomb.items():
            print(f"  {side}: hit {v['opening_bomb_hit']}/{v['episodes']}; wins given hit "
                  f"{v['win_given_hit'][0]}/{v['win_given_hit'][1]}, given no hit {v['win_given_no_hit'][0]}/{v['win_given_no_hit'][1]}")
    anims = summary.get('monstro_animation_frames')
    if anims:
        print()
        print(f"Monstro animation durations, logic frames ({anims['seeds']} replayed seeds; median [p10, p90], n)")
        print('| animation | simulator | AB+ |')
        print('|---|---|---|')
        for a in sorted(set(anims['sim']) | set(anims['abplus'])):
            cells = []
            for side in ('sim', 'abplus'):
                v = anims[side].get(a)
                cells.append(f"{v['median']:.0f} [{v['p10']:.0f}, {v['p90']:.0f}], n={v['n']}" if v else '-')
            print(f'| {a} | {cells[0]} | {cells[1]} |')


if __name__ == '__main__':
    main()
