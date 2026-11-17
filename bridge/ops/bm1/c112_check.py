"""gpu57 C112 (2026-11-16, the coordinator): a reward-ladder arm's smoke checks. c112_check.py <run>/<name>-s0 <strict|coef>
Every update row: errors 0; the diag_* columns there and numeric; reward_mean within 5% of diag_r (diag_r > 0); diag_n
between 0.5 and 1.05 of the update's new decisions (the 'decisions' column's step). Losses -- strict: teach_loss,
danger_loss, v2_loss_pi, v2_loss_vf 0 or empty in every row; coef (these columns are the losses before their
coefficients): config.json teach_coef, danger_coef, teacher_v2_coef, teacher_v2_vf 0 and pickup_force '0:0', the four
columns finite or empty, pickup_forced_steps and forced_share 0 where present. Prints the rows' values; exit 1 on a failure.
Mode of the arm (the guard.pre next to <run>: ISAAC_DIAG_REWARD; the coordinator, 2026-11-16): right / side as above;
door: diag_target in every row and between 0.6 and 0.95, the mean diag_hit of updates 1..3 between 0.10 and 0.16;
room: diag_entries > 0 and diag_ends > 0 in every row (no diag_right / diag_left in either)."""
import csv
import json
import math
import os
import re
import sys

d, mode = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(d + '/progress.csv')))
bad = []
pre = open(os.path.join(os.path.dirname(os.path.abspath(d)), 'guard.pre')).read()
kind = (re.findall(r'ISAAC_DIAG_REWARD=(\w+)', pre) or ['?'])[0]
cols = {'door': ('diag_target',), 'room': ('diag_entries', 'diag_ends', 'diag_per_end')}.get(kind, ('diag_right', 'diag_left'))
print('mode', kind)
hits = []


def num(r, k):
    try:
        return float(r.get(k))
    except (TypeError, ValueError):
        return None


if not rows:
    bad.append('no update rows')
prev = 0.0
for r in rows:
    u = r.get('update')
    e = num(r, 'errors')
    if e is None or e != 0:
        bad.append(f'u{u}: errors {r.get("errors")}')
    vals = {k: num(r, k) for k in ('diag_hit', 'diag_move', 'diag_r', 'diag_n') + cols}
    dec = num(r, 'decisions')
    new = (dec - prev) if dec is not None else None
    prev = dec if dec is not None else prev
    if any(v is None for v in vals.values()):
        bad.append(f'u{u}: diag columns missing / not numeric {vals}')
        continue
    if int(u) <= 3:
        hits.append(vals['diag_hit'])
    if kind == 'door' and not (0.6 <= vals['diag_target'] <= 0.95):
        bad.append(f'u{u}: door: diag_target {vals["diag_target"]} outside 0.6 .. 0.95')
    if kind == 'room' and not (vals['diag_entries'] > 0 and vals['diag_ends'] > 0):
        bad.append(f'u{u}: room: diag_entries {vals["diag_entries"]} diag_ends {vals["diag_ends"]}')
    rm, dr = num(r, 'reward_mean'), vals['diag_r']
    if rm is None or dr <= 0 or abs(rm - dr) > 0.05 * dr:
        bad.append(f'u{u}: reward_mean {rm} vs diag_r {dr}')
    if not new or not (0.5 <= vals['diag_n'] / new <= 1.05):
        bad.append(f'u{u}: diag_n {vals["diag_n"]} vs new decisions {new}')
    loss = {k: r.get(k, '') for k in ('teach_loss', 'danger_loss', 'v2_loss_pi', 'v2_loss_vf')}
    extra = {k: r.get(k) for k in ('pickup_forced_steps', 'forced_share') if k in r}
    print(f'u{u} reward_mean {rm} diag {vals} new {new} ent_move {r.get("ent_move")} losses {loss} {extra}')
    if mode == 'strict':
        if any(v not in ('', None) and float(v) != 0 for v in loss.values()):
            bad.append(f'u{u}: strict: loss columns not 0 {loss}')
    else:
        if any(v not in ('', None) and not math.isfinite(float(v)) for v in loss.values()):
            bad.append(f'u{u}: loss columns not finite {loss}')
        if any(v not in ('', None) and float(v) != 0 for v in extra.values()):
            bad.append(f'u{u}: forced decisions {extra}')
if kind == 'door' and not (hits and 0.10 <= sum(hits) / len(hits) <= 0.16):
    bad.append(f'door: mean diag_hit of updates 1..3 {hits} outside 0.10 .. 0.16')
if mode != 'strict':
    c = json.load(open(d + '/config.json'))
    cs = {k: c.get(k) for k in ('teach_coef', 'danger_coef', 'teacher_v2_coef', 'teacher_v2_vf', 'pickup_force')}
    print('config', cs)
    if any(cs[k] != 0 for k in ('teach_coef', 'danger_coef', 'teacher_v2_coef', 'teacher_v2_vf')) or \
            str(cs['pickup_force']) != '0:0':
        bad.append(f'config {cs}')
print('CHECK ' + ('OK' if not bad else 'FAIL: ' + '; '.join(bad[:4])))
sys.exit(1 if bad else 0)
