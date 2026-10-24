"""C39 curves for the figures: the four runs stitched by game hours (t3 to update 100, t3r 100-200, t3r2 200-275, t3r3 from
275), binned fresh-seed and all-episode outcomes per group, every evaluation, and the per-rollout training and buffer
statistics. Writes one JSON (stdout)."""
import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path.home() / 'isaac-abplus/train'
SEGMENTS = [('abp-c39-buffer-t3', 0.0, 122.577), ('abp-c39-buffer-t3r', 122.577, 245.1532),
            ('abp-c39-buffer-t3r2', 245.1532, 337.389), ('abp-c39-buffer-t3r3', 337.389, 1001.0)]
GROUPS = ('normal', 'horf', 'horf_rocks', 'spawners')
FPD, FPS = 4, 30
EDGES = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100] + list(range(120, 1001, 20))
EDGES[-1] = 1001

episodes = []
for run, start, end in SEGMENTS:
    total = 0
    for line in open(ROOT / run / 'episodes.jsonl'):
        e = json.loads(line)
        total += e['l']
        h = start + total * FPD / FPS / 3600
        if h < end:
            episodes.append((h, e.get('group'), bool(e.get('replay')), e['outcome'], e['frames'] / FPS,
                             -e['reward_components']['hurt'] / 0.5 if 'reward_components' in e else None))

bins = []
for a, b in zip(EDGES, EDGES[1:]):
    row = {'lo': a, 'hi': min(b, 1000), 'groups': {}}
    for g in GROUPS:
        sub = [e for e in episodes if e[1] == g and a <= e[0] < b]
        fresh = [e for e in sub if not e[2]]
        def rates(xs):
            n = len(xs)
            if not n:
                return None
            wins = [e[4] for e in xs if e[3] == 'win']
            hurt = [e[5] for e in xs if e[5] is not None]
            return {'n': n, 'win': sum(e[3] == 'win' for e in xs) / n, 'death': sum(e[3] == 'death' for e in xs) / n,
                    'timeout': sum(e[3] == 'time_limit' for e in xs) / n,
                    'hurt': statistics.mean(hurt) if hurt else None,
                    'clear_s': statistics.median(wins) if wins else None}
        row['groups'][g] = {'fresh': rates(fresh), 'all': rates(sub),
                            'replay_share': (sum(e[2] for e in sub) / len(sub)) if sub else None}
    bins.append(row)

evals = []
seen = set()
for run, start, end in SEGMENTS:
    for ck in sorted({p.name for g in GROUPS for p in (ROOT / run / f'evaluations-{g}').glob('update-*')}):
        m = re.match(r'update-(\d+)-step-(\d+)', ck)
        hours = int(m.group(2)) * FPD / FPS / 3600
        if ck in seen or hours >= end + 0.01:
            continue
        seen.add(ck)
        item = {'update': int(m.group(1)), 'hours': hours, 'groups': {}}
        for g in GROUPS:
            item['groups'][g] = {}
            for mode, sub in (('greedy', ''), ('sampled', 'sampled')):
                p = ROOT / run / f'evaluations-{g}' / ck / sub / 'results.jsonl'
                if not p.exists():
                    continue
                res = [json.loads(l) for l in open(p)]
                res = [r for r in res if r.get('outcome') != 'error']
                if not res:
                    continue
                wins = [r for r in res if r['outcome'] == 'win']
                item['groups'][g][mode] = {
                    'n': len(res), 'win': len(wins), 'death': sum(r['outcome'] == 'death' for r in res),
                    'timeout': sum(r['outcome'] == 'time_limit' for r in res),
                    'clean': sum(1 for r in wins if r.get('health_lost', 0) == 0),
                    'hurt': statistics.mean(r.get('health_lost', 0) for r in res),
                    'clear_s': statistics.median(r['frames'] / FPS for r in wins) if wins else None}
        evals.append(item)

KEYS = ('train/kl_full', 'train/approx_kl', 'train/entropy_move', 'train/entropy_shoot', 'train/explained_variance',
        'train/learning_rate', 'train/clip_fraction', 'train/value_loss', 'game/speed_x_realtime',
        'abplus/exchange_ms', 'abplus/instance_recycles', 'abplus/memory_recycles', 'abplus/instance_rss_max_mib')
BUF = ('p_mean', 'p_frontier_share', 'p_solved_share', 'p_unsolved_share', 'size')
rollouts = []
for run, start, end in SEGMENTS:
    for r in csv.DictReader(open(ROOT / run / 'progress.csv')):
        h = float(r.get('game/hours_total') or 0)
        if not (start < h <= end + 0.6):
            continue
        item = {'run': run, 'hours': h}
        for k in KEYS:
            v = r.get(k)
            item[k] = float(v) if v not in (None, '') else None
        for g in GROUPS:
            for k in BUF:
                v = r.get(f'buffer/{g}/{k}')
                item[f'buffer/{g}/{k}'] = float(v) if v not in (None, '') else None
        rollouts.append(item)

print(json.dumps({'segments': SEGMENTS, 'groups': GROUPS, 'bins': bins, 'evals': evals, 'rollouts': rollouts,
                  'episodes': len(episodes)}))
