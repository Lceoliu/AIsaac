"""VJP / collection overlap per update from TIMELINE lines (ISAAC_TIMELINE=1), markdown.

usage: python -X utf8 timeline_table.py <run dir> [--from 10]
Times in seconds from the update's learner start. Per update: the rollout's sources ready (the rc_sources mark: the learner's
resident SourceBatch built), the learner's minibatches end (last opt mark), the previous interval's VJP dispatch (first
source_backward start) and end per device (last end), the commit wait (commit_wait host event), the learner
end, and the actors' next collect (from the next update's line: its collect window against this update).
Summary: medians and quartiles of each, and what the commit wait would be if the VJP were dispatched
when the rollout's sources are ready (scheduling estimate: max(0, sources_ready + vjp_duration - minibatches_end)).
"""
import json, sys
import numpy as np

d = sys.argv[1]
lo = int(sys.argv[sys.argv.index('--from') + 1]) if '--from' in sys.argv else 10
lines = [json.JSONDecoder().raw_decode(l.split(' ', 1)[1])[0] for l in open(f'{d}/train.log', errors='replace') if l.startswith('TIMELINE ')]
rows = []
for i, t in enumerate(lines):
    s0 = t['learn_start']
    ev = {}
    for k, dev, a, b in t['events']: ev.setdefault((k, dev), []).append((a - s0, b - s0))
    enc_end = max((b for (k, _), v in ev.items() if k == 'encode' for a, b in v if a >= 0), default=np.nan)
    rcs = [x - s0 for n, x in t['marks'] if n == 'rc_sources']
    rcs_end = rcs[0] if rcs else np.nan   # (the rollout's sources ready: the learner's resident SourceBatch built)
    opts = [x - s0 for n, x in t['marks'] if n == 'opt']
    mb_end = max(opts) if opts else np.nan
    vjp = {dev: (min(a for a, b in v), max(b for a, b in v)) for (k, dev), v in ev.items() if k == 'source_backward'}
    vjp_start = min((a for a, b in vjp.values()), default=np.nan)
    vjp_end = max((b for a, b in vjp.values()), default=np.nan)
    vjp_dur = max((b - a for a, b in vjp.values()), default=np.nan)
    cw = sum(b - a for a, b in ev.get(('commit_wait', 'host'), []))
    rew = sum(b - a for a, b in ev.get(('ready_encode_wait', 'host'), []))
    nxt = lines[i + 1]['actors'] if i + 1 < len(lines) else None
    coll = [(a[0] - s0, a[1] - s0) for a in nxt] if nxt and all(nxt) else None
    rows.append(dict(update=i + 1, learn_end=t['learn_end'] - s0, enc_end=enc_end, rcs_end=rcs_end, mb_end=mb_end, vjp_start=vjp_start,
                     vjp_end=vjp_end, vjp_dur=vjp_dur, commit_wait=cw, ready_encode_wait=rew,
                     next_collect=(min(c[0] for c in coll), max(c[1] for c in coll)) if coll else None))
steady = [r for r in rows if r['update'] > lo]
q = lambda k: np.nanpercentile([r[k] for r in steady], [25, 50, 75])
print('| update | rollout sources ready | minibatches end | VJP dispatch | VJP end (last card) | VJP longest card | commit wait | learner end | actors\' next collect |')
print('|---|---|---|---|---|---|---|---|---|')
for r in rows:
    nc = '%.1f–%.1f' % r['next_collect'] if r['next_collect'] else ''
    print(f"| {r['update']} | {r['rcs_end']:.1f} | {r['mb_end']:.1f} | {r['vjp_start']:.1f} | {r['vjp_end']:.1f} | {r['vjp_dur']:.1f} | "
          f"{r['commit_wait']:.2f} | {r['learn_end']:.1f} | {nc} |")
print()
print(f'steady (updates > {lo}, n={len(steady)}): quartiles 25/50/75')
for k in ('rcs_end', 'mb_end', 'vjp_start', 'vjp_end', 'vjp_dur', 'commit_wait', 'ready_encode_wait', 'learn_end'):
    a = q(k); print(f'  {k}: {a[0]:.2f} / {a[1]:.2f} / {a[2]:.2f}')
est = [max(0.0, r['rcs_end'] + r['vjp_dur'] - r['mb_end']) for r in steady]
print('  dispatched after the sources were ready by (vjp_start - rcs_end): %.2f s median'
      % np.nanmedian([r['vjp_start'] - r['rcs_end'] for r in steady]))
print('  commit wait if dispatched when the sources are ready: %.2f / %.2f / %.2f' % tuple(np.nanpercentile(est, [25, 50, 75])))
print('  learner end then: %.2f s median' % np.nanmedian([r['learn_end'] - r['commit_wait'] + e for r, e in zip(steady, est)]))
