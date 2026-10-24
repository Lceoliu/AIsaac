#!/usr/bin/env python3
"""Paired comparison of two frozen single-item surveys (same panel, seeds, rep 0): per item, gain = panel-mean return
of the single build minus the empty build's, in survey A and survey B; the shift B - A over all items, the
cannot-use list of A, actives, weapons. usage: lab_compare.py <reportA.json> <lab_items.json> A_dir[,A_dir] B_dir[,B_dir]"""
import json
import sys
from pathlib import Path

import numpy as np

rep = json.load(open(sys.argv[1]))
items = {int(r['id']): r for r in json.load(open(sys.argv[2]))['items']}
cannot = [int(r['item']) for r in rep['cannot_use']]
top = [int(r['item']) for r in rep['top']]
bottom = [int(r['item']) for r in rep['bottom']]


def survey(dirs):
    base, single = None, {}
    for d in dirs.split(','):
        for line in open(Path(d) / 'lab.jsonl'):
            r = json.loads(line)
            if not r.get('complete') or r.get('rep', 0) != 0 or r.get('kind') == 'replicate':
                continue
            if not r['items'] and base is None:
                base = r['ret']
            elif len(r['items']) == 1 and r['items'][0] not in single:
                single[r['items'][0]] = r['ret']
    return base, single


ba, sa = survey(sys.argv[3])
bb, sb = survey(sys.argv[4])
common = sorted(set(sa) & set(sb))
print('A: base %.3f singles %d | B: base %.3f singles %d | common %d' % (ba, len(sa), bb, len(sb), len(common)))
ga = np.array([sa[i] - ba for i in common])
gb = np.array([sb[i] - bb for i in common])
d = gb - ga
print('gain B - A over common items: mean %+.3f  median %+.3f  sd %.3f  (seed noise sd of a build ~0.10, so a pair '
      'differs beyond noise at |d| > 0.29)' % (d.mean(), np.median(d), d.std()))
print('items better in B by > 0.29: %d, worse: %d' % ((d > 0.29).sum(), (d < -0.29).sum()))
print('raw single-build returns: A mean %.3f  B mean %.3f  corr %.3f' % (
    np.mean([sa[i] for i in common]), np.mean([sb[i] for i in common]), np.corrcoef(ga, gb)[0, 1]))


def group(name, ids):
    ids = [i for i in ids if i in sa and i in sb]
    if not ids:
        return
    print('%-14s n %3d  gain A %+.3f -> B %+.3f  (mean d %+.3f)' % (
        name, len(ids), np.mean([sa[i] - ba for i in ids]), np.mean([sb[i] - bb for i in ids]),
        np.mean([(sb[i] - bb) - (sa[i] - ba) for i in ids])))


group('cannot_use(A)', cannot)
group('bottom15(A)', bottom)
group('top15(A)', top)
group('actives', [i for i in common if items[i]['kind'] == 'active'])
group('passives', [i for i in common if items[i]['kind'] == 'passive'])
group('familiars', [i for i in common if items[i]['kind'] == 'familiar'])
print('cannot-use items:')
for i in cannot:
    if i in sa and i in sb:
        print('  %4d %-22s gain A %+.2f  B %+.2f  d %+.2f' % (i, items[i]['name'][:22], sa[i] - ba, sb[i] - bb,
                                                             (sb[i] - bb) - (sa[i] - ba)))
print('largest rises in B:')
for i in sorted(common, key=lambda i: -((sb[i] - bb) - (sa[i] - ba)))[:10]:
    print('  %4d %-22s %-8s gain A %+.2f  B %+.2f' % (i, items[i]['name'][:22], items[i]['kind'], sa[i] - ba, sb[i] - bb))
print('largest drops in B:')
for i in sorted(common, key=lambda i: ((sb[i] - bb) - (sa[i] - ba)))[:10]:
    print('  %4d %-22s %-8s gain A %+.2f  B %+.2f' % (i, items[i]['name'][:22], items[i]['kind'], sa[i] - ba, sb[i] - bb))
