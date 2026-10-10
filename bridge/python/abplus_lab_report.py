"""Phase C (2026-10-08): tables of the build lab's results (a run's lab.jsonl, train_tok --lab-share; tok_lab.py).

- noise floor: twins (a replicate job with the same rep; per state with the same weights the trace must be equal: the
  count of equal traces), seed noise (replicates with another rep: the spread of a build's panel mean between reps);
- baseline: the empty build's panel means (rep 0);
- single items: gain = the build's panel mean minus the baseline's, per target, paired per state (common random
  numbers: rep 0 of both); the top / bottom --top by the scalar 'ret' (the trainer's undiscounted return);
- "cannot use" candidates: single items whose transplanted stats are better than the baseline's (damage up, fire delay
  down, range / shot speed up, or another weapon type) while their gain is below 0;
- pairs: gain(A + B) - gain(A) - gain(B) for the pair builds of the run (or of --pairs-from's top singles).

usage: python abplus_lab_report.py <run dir> [<run dir> ...] [--top 15] [--out report.json]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGETS = ('ret', 'clear', 'death', 'hurt', 'damage_share', 'seconds')
PS = ('clear', 'death', 'hurt', 'damage_share', 'frames', 'ret', 'end', 'trace', 'version0', 'version1', 'first')


def load(dirs, names_file=None):
    rows = []
    for d in dirs:
        for line in open(Path(d) / 'lab.jsonl'):
            r = json.loads(line)
            r['run'] = str(d)
            rows.append(r)
    names = {}
    if names_file and Path(names_file).is_file():
        raw = json.loads(Path(names_file).read_text(encoding='utf8'))
        names = {int(r['id']): r for r in raw['items'] + raw.get('excluded', [])}
    return rows, names


def per_state(r, k='ret'):
    j = PS.index('frames') if k == 'seconds' else PS.index(k)
    out = {int(s): float(v[j]) / (30.0 if k == 'seconds' else 1.0) for s, v in (r.get('per_state') or {}).items()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--top', type=int, default=15)
    p.add_argument('--items-file', default=str(Path(__file__).resolve().parent.parent / 'abplus' / 'catalog' /
                                               'lab_items.json'))
    p.add_argument('--out', default='')
    p.add_argument('--pairs-json', default='', help='write the pairs of the top-N singles (N = --pairs-n) here')
    p.add_argument('--pairs-n', type=int, default=20)
    p.add_argument('--model-check', action='store_true', help='replay the build model prequentially (tok_lab.LabModel)')
    p.add_argument('--model-every', type=int, default=4, help='builds between the replayed model\'s training rounds')
    p.add_argument('--model-steps', type=int, default=50)
    args = p.parse_args()
    rows, names = load(args.runs, args.items_file)
    rows = [r for r in rows if r.get('complete')]

    def nm(i):
        return names.get(i, {}).get('name', str(i))
    rep: dict = {}
    # ---- noise floor
    by_job = {(r['run'], r['job']): r for r in rows}
    twin_eq = twin_n = 0
    seed_d = []
    seed_d_state = []
    for r in rows:
        if r['kind'] != 'replicate':
            continue
        o = by_job.get((r['run'], r['flags']))
        if o is None:
            continue
        if o['rep'] == r['rep']:
            for s, v in r['per_state'].items():
                w = o['per_state'].get(s)
                if w is not None:
                    twin_n += 1
                    twin_eq += int(v[7] == w[7])
        else:
            seed_d.append(r['ret'] - o['ret'])
            a, b = per_state(r), per_state(o)
            seed_d_state += [a[s] - b[s] for s in a if s in b]
    sd_build = float(np.std(seed_d) / math.sqrt(2)) if len(seed_d) > 1 else None
    rep['noise'] = dict(twin_states=twin_n, twin_equal=twin_eq, seed_pairs=len(seed_d),
                        seed_sd_build_ret=sd_build,
                        seed_sd_state_ret=float(np.std(seed_d_state) / math.sqrt(2)) if seed_d_state else None,
                        seed_abs_diff_median=float(np.median(np.abs(seed_d))) if seed_d else None)
    # ---- baseline (rep 0, per run)
    base = defaultdict(list)
    for r in rows:
        if not r['items'] and r['rep'] == 0:
            base[r['run']].append(r)
    if not base:
        raise SystemExit('no baseline (empty build, rep 0) in these runs')
    base_ps = {}
    for run, lst in base.items():
        base_ps[run] = {k: {s: float(np.mean([per_state(b, k)[s] for b in lst if s in per_state(b, k)]))
                            for s in per_state(lst[0], k)} for k in ('ret', 'clear', 'death', 'hurt', 'damage_share',
                                                                      'seconds')}
    b0 = next(iter(base.values()))
    rep['baseline'] = {k: float(np.mean([b[k] for b in b0])) for k in TARGETS}
    rep['baseline']['n'] = sum(len(v) for v in base.values())
    # distinct per-state outcome sets among all rep-0 empty builds of all runs (1: all the same games)
    rep['baseline']['distinct'] = len({json.dumps({s: v[:8] + v[10:] for s, v in b['per_state'].items()},
                                                  sort_keys=True) for v_ in base.values() for b in v_})
    base_stats = b0[0]['stats']

    def gain(r):
        # a run without its own baseline uses another run's (same panel and frozen weights: the empty build's rep-0
        # games are the same games; base 'identical_runs' checks that within and across the runs that have one)
        bp = base_ps.get(r['run']) or next(iter(base_ps.values()))
        out = {}
        for k in ('ret', 'clear', 'death', 'hurt', 'damage_share', 'seconds'):
            a = per_state(r, k)
            d = [a[s] - bp[k][s] for s in a if s in bp[k]]
            out[k] = float(np.mean(d)) if d else float('nan')
            if k == 'ret':
                out['ret_se_states'] = float(np.std(d) / math.sqrt(len(d))) if len(d) > 1 else float('nan')
        return out
    # ---- single items (rep 0; repeated rep-0 jobs of an item are averaged)
    singles = defaultdict(list)
    for r in rows:
        if len(r['items']) == 1 and r['rep'] == 0:
            singles[r['items'][0]].append(r)
    table = []
    for item, lst in singles.items():
        g = [gain(r) for r in lst]
        st = lst[0]['stats']
        better = []
        if st['damage'] > base_stats['damage'] + 1e-6:
            better.append('damage')
        if st['fire_delay_max'] < base_stats['fire_delay_max'] - 1e-6:
            better.append('tears')
        if st['range'] > base_stats['range'] + 1e-6:
            better.append('range')
        if st['shot_speed'] > base_stats['shot_speed'] + 1e-6:
            better.append('shot_speed')
        if st['speed'] > base_stats['speed'] + 1e-6:
            better.append('speed')
        if int(st['weapons']) != int(base_stats['weapons']):
            better.append('weapon')
        if st['can_fly'] > base_stats['can_fly']:
            better.append('flight')
        if lst[0].get('familiars', 0) > 0:
            better.append('familiar')
        table.append(dict(item=item, name=nm(item), kind=names.get(item, {}).get('kind'), n=len(lst),
                          **{f'g_{k}': float(np.mean([x[k] for x in g])) for k in g[0]},
                          ret=float(np.mean([r['ret'] for r in lst])), clear=float(np.mean([r['clear'] for r in lst])),
                          death=float(np.mean([r['death'] for r in lst])), hurt=float(np.mean([r['hurt'] for r in lst])),
                          seconds=float(np.mean([r['seconds'] for r in lst])), stats=st, better=better))
    table.sort(key=lambda t: -t['g_ret'])
    rep['singles_n'] = len(table)
    rep['singles_gain_ret'] = dict(mean=float(np.mean([t['g_ret'] for t in table])) if table else None,
                                   above0=sum(t['g_ret'] > 0 for t in table),
                                   # a gain is the difference of two build means (each with the seed SD sd_build;
                                   # common random numbers can only make it smaller): 2 x sqrt(2) x sd_build
                                   above_noise=sum(t['g_ret'] > 2 * math.sqrt(2) * sd_build for t in table)
                                   if sd_build else None,
                                   below_noise=sum(t['g_ret'] < -2 * math.sqrt(2) * sd_build for t in table)
                                   if sd_build else None, threshold=2 * math.sqrt(2) * sd_build if sd_build else None)
    keep = ('item', 'name', 'kind', 'n', 'g_ret', 'g_ret_se_states', 'g_clear', 'g_death',
            'g_hurt', 'g_seconds', 'g_damage_share', 'better')

    def short(t):
        return {k: (round(t[k], 3) if isinstance(t.get(k), float) else t.get(k)) for k in keep}
    rep['top'] = [short(t) for t in table[:args.top]]
    rep['bottom'] = [short(t) for t in table[-args.top:][::-1]]
    noise2 = 2 * math.sqrt(2) * sd_build if sd_build else 0.0
    rep['cannot_use'] = [short(t) for t in sorted(table, key=lambda t: t['g_ret'])
                         if t['g_ret'] < -noise2 and any(b in t['better'] for b in ('damage', 'tears', 'range',
                                                                                     'shot_speed', 'weapon'))]
    # ---- pairs: synergy = gain(A+B) - gain(A) - gain(B)
    sg = {t['item']: t['g_ret'] for t in table}
    pairs = []
    for r in rows:
        if len(r['items']) == 2 and r['rep'] == 0 and all(i in sg for i in r['items']):
            a, b = r['items']
            gab = gain(r)['ret']
            pairs.append(dict(items=[a, b], names=[nm(a), nm(b)], g_ab=round(gab, 3), g_a=round(sg[a], 3),
                              g_b=round(sg[b], 3), synergy=round(gab - sg[a] - sg[b], 3)))
    pairs.sort(key=lambda x: -x['synergy'])
    rep['pairs_n'] = len(pairs)
    if pairs:
        syn = np.array([x['synergy'] for x in pairs])
        rep['pairs_synergy'] = dict(mean=float(syn.mean()), sd=float(syn.std()), above0=int((syn > 0).sum()),
                                    # synergy = m(AB) - m(A) - m(B) + m(base): four build means, SD 2 x sd_build
                                    above_noise=int((syn > 4 * (sd_build or 0)).sum()), threshold=4 * (sd_build or 0))
        rep['pairs_top'] = pairs[:args.top]
        rep['pairs_bottom'] = pairs[-5:]
    if args.pairs_json:
        top = [t['item'] for t in table[:args.pairs_n]]
        actives = {i for i, r in names.items() if r.get('kind') == 'active'}
        out = [[a, b] for k, a in enumerate(top) for b in top[k + 1:] if not (a in actives and b in actives)]
        Path(args.pairs_json).write_text(json.dumps(out))
        rep['pairs_written'] = len(out)
    if args.model_check:   # the build model replayed prequentially over these builds (each predicted before it is added)
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from isaac_bridge.tok_lab import LabModel
        m = LabModel(k=5, device='cpu', seed=1)
        preds, means, ys, sds = [], [], [], []
        seen = []
        for j, r in enumerate(sorted(rows, key=lambda r: (r['run'], r['job']))):
            pre = m.prequential(r)
            if pre is not None:
                preds.append(pre[0])
                sds.append(pre[1])
                means.append(float(np.mean(seen)))
                ys.append(r['ret'])
            m.add(r)
            seen.append(r['ret'])
            if (j + 1) % args.model_every == 0:
                m.train(args.model_steps)
        p_, y_, mu_ = np.array(preds), np.array(ys), np.array(means)
        half = len(y_) // 2
        rep['model_check'] = dict(
            predicted=len(y_), mae=float(np.abs(p_ - y_).mean()), mae_running_mean=float(np.abs(mu_ - y_).mean()),
            corr=float(np.corrcoef(p_, y_)[0, 1]) if len(y_) > 2 else None,
            mae_second_half=float(np.abs(p_[half:] - y_[half:]).mean()),
            mae_running_mean_second_half=float(np.abs(mu_[half:] - y_[half:]).mean()),
            corr_second_half=float(np.corrcoef(p_[half:], y_[half:])[0, 1]) if len(y_) - half > 2 else None,
            head_sd_mean=float(np.mean(sds)),
            sd_vs_error_corr=float(np.corrcoef(sds, np.abs(p_ - y_))[0, 1]) if len(y_) > 2 else None)
    text = json.dumps(rep, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == '__main__':
    main()
