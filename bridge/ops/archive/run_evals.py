#!/usr/bin/env python3
"""Tabulate run-mode evals.jsonl files: one line per checkpoint. usage: run_evals.py evals.jsonl [...]"""
import json
import sys

for path in sys.argv[1:]:
    print(path)
    print("  update   gh  death  tout  s1  s2  s3  s4+  floors  rooms  boss  hurt  secs  items  act  pills")
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        g = s["groups"]["run"]
        st = g.get("stage_reached", {})
        s4 = sum(v for k, v in st.items() if int(k) >= 4)
        print("  %6d %5.0f  %5.2f %5.2f %3d %3d %3d %4d  %6.2f %6.2f %5.2f %5.2f %5.0f  %5.2f %4.2f %5.2f" % (
            s["update"], s.get("game_hours", 0), g["death"], g["timeout"], st.get("1", 0), st.get("2", 0), st.get("3", 0), s4,
            g["floors_cleared"], g["rooms_cleared"], g["boss_clear"], g["hurt_per_run"], g["seconds"],
            g.get("items_taken", 0), g.get("active_uses", 0), g.get("pills_used", 0)))
