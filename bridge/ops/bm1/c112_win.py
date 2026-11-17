"""gpu57 C112: means over updates lo..hi of a ladder arm's progress.csv. c112_win.py <run>/<name>-s0 <lo> <hi>
door arms add diag_target; room arms (L4) add diag_per_end (mean over the updates), diag_entries / diag_ends (sums), and
the window's episodes.jsonl episodes (rows whose 'update' is in lo..hi): rooms>=1 share ('rooms' = rooms CLEARED) and
the mean episode length (decisions, and seconds = decisions * frames_per_decision / 30)."""
import csv
import json
import sys

d, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
R = [r for r in csv.DictReader(open(d + '/progress.csv')) if lo <= int(r['update']) <= hi]


def m(k):
    return sum(float(r[k]) for r in R) / max(len(R), 1)


def has(k):
    return bool(R) and R[0].get(k) not in (None, '')


out = f'n {len(R)} diag_hit {m("diag_hit"):.3f} '
if has('diag_right'):
    out += (f'diag_left {m("diag_left"):.3f} diag_right {m("diag_right"):.3f} '
            f'hit/lr {m("diag_hit") / max(m("diag_left") + m("diag_right"), 1e-9):.3f} ')
if has('diag_target'):
    out += f'diag_target {m("diag_target"):.3f} '
if has('diag_per_end'):
    fpd = json.load(open(d + '/config.json')).get('frames_per_decision', 4) or 4
    E = [e for e in map(json.loads, open(d + '/episodes.jsonl')) if lo <= int(e.get('update', -1)) <= hi]
    dec = sum(float(e.get('decisions', 0)) for e in E) / max(len(E), 1)
    out += (f'diag_per_end {m("diag_per_end"):.3f} diag_entries {sum(float(r["diag_entries"]) for r in R):.0f} '
            f'diag_ends {sum(float(r["diag_ends"]) for r in R):.0f} episodes {len(E)} '
            f'rooms>=1 {sum(float(e.get("rooms", 0)) >= 1 for e in E) / max(len(E), 1):.3f} '
            f'ep_dec {dec:.0f} ep_s {dec * fpd / 30:.1f} ')
out += (f'ent_move {m("ent_move"):.3f} diag_r {m("diag_r"):.4f} diag_move {m("diag_move"):.3f} '
        f'gh {float(R[-1]["game_hours"]) if R else 0:.1f}')
print(out)
