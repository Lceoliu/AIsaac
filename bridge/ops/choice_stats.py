#!/usr/bin/env python3
"""Summarise a choices.jsonl from a --branch-* run: counts by kind, take-vs-skip score gap, archive use, and the online
(pre-branch) choice-head prediction quality in thirds of the file. usage: choice_stats.py choices.jsonl"""
import json
import sys
from collections import defaultdict

import numpy as np

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print("records", len(recs))
by_kind = defaultdict(list)
for r in recs:
    by_kind[r.get("kind_name", r["kind"])].append(r)
for k, rs in by_kind.items():
    gaps, took_any, arch, boot, wall = [], 0, 0, 0, []
    for r in rs:
        d = r.get("branches_detail", [])
        opt = defaultdict(list)
        for b in d:
            if b.get("valid", 1):
                opt[int(b["option"])].append(b["score"])
            arch += int(b.get("archived", 0))
            boot += int(b.get("boot", 0))
        if any(b.get("took", 0) for b in d):
            took_any += 1
        if 0 in opt and 1 in opt:
            gaps.append(np.mean(opt[1]) - np.mean(opt[0]))
        wall.append(r.get("wall_s", 0))
    gaps = np.array(gaps)
    print("kind %-10s n %5d  pairs %5d  gap(opt1-opt0) mean %+.3f sd %.3f  |gap|>0.3: %.2f  took_any %d  archived %d  boot %d  wall %.1fs" % (
        k, len(rs), len(gaps), gaps.mean() if len(gaps) else 0, gaps.std() if len(gaps) else 0,
        (np.abs(gaps) > 0.3).mean() if len(gaps) else 0, took_any, arch, boot, np.mean(wall)))
# online prediction quality: pred (choice head, before branching) vs realised gap
rows = []
for r in recs:
    d = r.get("branches_detail", [])
    opt = defaultdict(list)
    for b in d:
        if b.get("valid", 1):
            opt[int(b["option"])].append(b["score"])
    if 0 in opt and 1 in opt and "pred" in r:
        rows.append((r["pred"], np.mean(opt[1]) - np.mean(opt[0]), r.get("uncert", 0)))
rows = np.array(rows)
n = len(rows)
print("pred vs gap, thirds:")
for i in range(3):
    part = rows[i * n // 3:(i + 1) * n // 3]
    if len(part) > 2:
        c = np.corrcoef(part[:, 0], part[:, 1])[0, 1]
        print("  third %d n %5d corr %.3f  mae(pred-gap) %.3f  mae(0-gap) %.3f  uncert %.3f" % (
            i, len(part), c, np.abs(part[:, 0] - part[:, 1]).mean(), np.abs(part[:, 1]).mean(), part[:, 2].mean()))
