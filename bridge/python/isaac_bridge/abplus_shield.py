"""Collision shield for the floor runner's COMBAT (floor-clear mandate, 2026-10-02): a test-time safety layer over the policy's
move, in the spirit of C30's block_moves mask (which drops the moves the terrain stops) and of shielding in safe RL.

At each COMBAT decision the shield predicts, for each of the 9 moves, the player's path over the next HORIZON logic frames
and every harmful entity's path, and marks the move dangerous when they come closer than the two radii plus MARGIN:
  player   velocity from its last two positions; under move m it relaxes towards FREE_SPEED x unit(m) by RELAX per frame
           (the engine's own acceleration is close to this; m = 0, and a move the terrain stops dead (blocked_moves),
           relaxes towards rest);
  entities the enemies with contact damage (cdmg > 0, a collision class) and every enemy projectile (type 9), moving in a
           straight line at the velocity of their last two positions (matched by id; a new one stands still);
           radius = size x the larger SizeMulti.
When the policy's move is dangerous and at least one move is not, the safe move the policy rates highest (its move logits)
replaces it; the shot, the bomb and the item stay the policy's. The shield uses only what the observation carries.
"""
import math
import os

import numpy as np

from .abplus_geometry import MOVES, blocked_moves

HORIZON = 8          # logic frames looked ahead (two 4-frame decisions)
MARGIN = 4.0         # px beyond the two radii
FREE_SPEED = 5.4     # px per frame at full speed (transformer_obs.FREE_SPEED)
RELAX = 0.75         # the player's velocity keeps this share of the gap to the target velocity per frame
PROJECTILE = 9
# v2 (A17 follow-up, 2026-10-03), off unless switched on: live bombs, as a blast of BLAST px from the frame they go off
# (their age reaches FUSE: 48 frames measured for the player's bombs, ~58 for troll bombs), and enemy creep (effects
# CREEP_VARIANTS, standing in it hurts) as static threats of their own size
BOMBS = os.environ.get('ABP_SHIELD_BOMBS') == '1'   # training workers (ABP_SHIELD_TRAIN): switched on by the environment
CREEP = os.environ.get('ABP_SHIELD_CREEP') == '1'   # the floor runner sets both from --shield-bombs / --shield-creep
BLAST = 80.0
BOMB_HORIZON = 24   # frames a bomb is looked ahead for: running 90 px out of a blast takes ~18
FUSE = {0: 48, 3: 58, 4: 58}
CREEP_VARIANTS = (22, 23)


def harmful(e):
    if e.get('type') == PROJECTILE or e.get('projectile'):
        return True
    return bool(e.get('enemy')) and float(e.get('cdmg', 0.0)) > 0 and int(e.get('coll', 0)) > 0


def danger(prev, cur):
    """[bool] * 9 in MOVES order: the moves that bring the player into a harmful entity within HORIZON frames."""
    p = cur['players'][0]
    px, py = p['pos']
    dt = max(1, int(cur.get('logic_frames', 0)) - int(prev.get('logic_frames', 0))) if prev else 1
    if prev and prev.get('room', {}).get('room_idx') == cur.get('room', {}).get('room_idx'):
        qx, qy = prev['players'][0]['pos']
        vx, vy = (px - qx) / dt, (py - qy) / dt
    else:
        vx = vy = 0.0
    rp = float(p.get('size', 10.0))
    before = {e['id']: e['pos'] for e in (prev or {}).get('entities', ()) if 'id' in e} \
        if prev and prev.get('room', {}).get('room_idx') == cur.get('room', {}).get('room_idx') else {}
    threats = []
    for e in cur.get('entities', ()):
        if not harmful(e):
            continue
        ex, ey = e['pos']
        if e.get('id') in before:
            bx, by = before[e['id']]
            evx, evy = (ex - bx) / dt, (ey - by) / dt
        else:
            evx = evy = 0.0
        r = float(e.get('size', 10.0)) * max(1.0, *[float(v) for v in e.get('size_multi', (1.0, 1.0))])
        if math.hypot(ex - px, ey - py) > rp + r + MARGIN + HORIZON * (FREE_SPEED + math.hypot(evx, evy)) + 1:
            continue   # cannot meet within the horizon
        threats.append((ex, ey, evx, evy, rp + r + MARGIN, 1, HORIZON))
    for e in cur.get('entities', ()):
        if BOMBS and e.get('type') == 4:
            left = FUSE.get(int(e.get('variant', 0)), 48) - int(e.get('age', 0))
            if 0 <= left <= BOMB_HORIZON:
                ex, ey = e['pos']
                threats.append((ex, ey, 0.0, 0.0, rp + BLAST, max(1, left), left + 2))
        elif CREEP and e.get('type') == 1000 and int(e.get('variant', -1)) in CREEP_VARIANTS and float(e.get('size', 0)) > 0:
            ex, ey = e['pos']
            threats.append((ex, ey, 0.0, 0.0, rp + float(e['size']), 1, HORIZON))
    out = [False] * len(MOVES)
    if not threats:
        return out
    stopped = blocked_moves(cur)
    t = np.arange(1, max(HORIZON, max(th[6] for th in threats)) + 1, dtype=np.float64)
    for m, (mx, my) in enumerate(MOVES):
        n = math.hypot(mx, my)
        tx, ty = (FREE_SPEED * mx / n, FREE_SPEED * my / n) if n and not stopped[m] else (0.0, 0.0)
        # v(t) = target + (v0 - target) RELAX^t; position = sum of the velocities so far
        decay = RELAX ** t
        vxs, vys = tx + (vx - tx) * decay, ty + (vy - ty) * decay
        fx, fy = px + np.cumsum(vxs), py + np.cumsum(vys)
        for ex, ey, evx, evy, reach, first, last in threats:
            d = np.hypot(fx - (ex + evx * t), fy - (ey + evy * t))
            if ((d < reach) & (t >= first) & (t <= last)).any():
                out[m] = True
                break
    return out


def shield(prev, cur, move_logits, move):
    """The move to play: `move` unless it is dangerous and a safe move exists (then the safe move of the highest logit).
    Returns (move, intervened)."""
    bad = danger(prev, cur)
    if not bad[move] or all(bad):
        return move, False
    logits = np.asarray(move_logits, dtype=np.float64).copy()
    logits[np.asarray(bad)] = -np.inf
    return int(np.argmax(logits)), True
