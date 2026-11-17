"""C111 (2026-11-16, the coordinator): one 50-update window of a run, stdlib only. c111_win50.py <run>/<name>-s0 <first u> <last u>
Prints: episodes (episodes.jsonl rows whose 'update' is in [first, last]), their rooms>=1 share ('rooms' = rooms cleared),
and the mean of progress.csv's ent_move over the updates [first, last]."""
import csv
import json
import sys

d, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
n = r1 = 0
with open(d + '/episodes.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if a <= int(e.get('update', -1)) <= b:
            n += 1
            r1 += float(e.get('rooms', 0)) >= 1
ent = [float(r['ent_move']) for r in csv.DictReader(open(d + '/progress.csv'))
       if a <= int(r['update']) <= b and r.get('ent_move') not in (None, '')]
print(f'u{a}-{b} episodes {n} rooms>=1 {r1 / max(n, 1):.3f} ent_move {sum(ent) / max(len(ent), 1):.3f} ({len(ent)} updates)')
