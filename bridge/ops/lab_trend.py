#!/usr/bin/env python3
"""Trend of a training run's online lab results (lab.jsonl): per third of the jobs, the mean return of the empty build,
of single-item builds, of builds containing a 'cannot use' item (the frozen survey's list) and of actives.
usage: lab_trend.py <lab.jsonl> <labA2-report300.json> <lab_items.json>"""
import json
import sys

import numpy as np

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
recs = [r for r in recs if r.get('complete') and r.get('ok', 0) >= 30]
rep = json.load(open(sys.argv[2]))
items = {int(r['id']): r for r in json.load(open(sys.argv[3]))['items']}
cannot = {int(r['item']) for r in rep['cannot_use']}
bottom = {int(r['item']) for r in rep['bottom']}
top = {int(r['item']) for r in rep['top']}
actives = {i for i, r in items.items() if r['kind'] == 'active'}
print('complete jobs', len(recs), '| cannot-use ids', sorted(cannot))
n = len(recs)
parts = 3
print('%-6s %6s | %-14s %-14s %-14s %-14s %-14s %-14s' % ('third', 'jobs', 'empty', 'singles', 'cannot_use', 'bottom15', 'top15', 'actives'))
for p in range(parts):
    part = recs[p * n // parts:(p + 1) * n // parts]

    def stat(sel):
        v = [r['ret'] for r in part if sel(r)]
        return '%5.2f n%4d' % (np.mean(v), len(v)) if v else '    -      '
    print('%-6d %6d | %s %s %s %s %s %s' % (
        p, len(part),
        stat(lambda r: not r['items']),
        stat(lambda r: len(r['items']) == 1),
        stat(lambda r: any(i in cannot for i in r['items'])),
        stat(lambda r: any(i in bottom for i in r['items'])),
        stat(lambda r: any(i in top for i in r['items'])),
        stat(lambda r: any(i in actives for i in r['items']))))
# per cannot-use item: all its jobs in order
print('per cannot-use item (ret of each job containing it, in order; empty-build mean for reference):')
for i in sorted(cannot):
    v = ['%.2f' % r['ret'] for r in recs if i in r['items']]
    print('  %4d %-18s %s' % (i, items.get(i, {}).get('name', '?')[:18], ' '.join(v)))
