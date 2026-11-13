#!/usr/bin/env python3
"""Per test file: passed / failed / skipped (with the reasons) from a pytest log run with -rA (2026-11-12, the
coordinator's list of the tests reported passing since 5ee81f3, re-run on the final tree). Read-only.

  python3 pytest_counts.py <log>"""
import collections
import re
import sys

c = collections.defaultdict(lambda: collections.Counter())
why = collections.defaultdict(set)
ext = []
with open(sys.argv[1], errors='replace') as f:
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('EXT '):
            ext.append(line)
        m = re.match(r'(PASSED|FAILED|ERROR|XFAIL|XPASS) (tests/[^:\s]+)', line)
        if m:
            c[m.group(2)][m.group(1).lower()] += 1
            continue
        m = re.match(r'SKIPPED \[(\d+)\] (tests/[^:]+):\d+: (.*)', line)
        if m:
            c[m.group(2)]['skipped'] += int(m.group(1))
            why[m.group(2)].add(m.group(3)[:120])
print(' '.join(ext))
for k in sorted(c):
    n = c[k]
    print(f"{k:45s} passed {n['passed']:3d}  skipped {n['skipped']:3d}  failed {n['failed'] + n['error']:3d}"
          + (f"  ({'; '.join(sorted(why[k]))})" if why[k] else ''))
print(f"total: passed {sum(n['passed'] for n in c.values())}, skipped {sum(n['skipped'] for n in c.values())}, "
      f"failed {sum(n['failed'] + n['error'] for n in c.values())}")
