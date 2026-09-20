"""A visible-observation-only heuristic for smoke tests, NOT a trained agent."""
from math import hypot

from .env import MOVE, SHOOT
from .geometry import circle_rect


def scripted_action(obs):
    x, y, _, _, radius, *_ = map(float, obs['player'])
    enemies = obs['enemies']
    if len(enemies) == 0:
        return 0, 0
    target = min(enemies, key=lambda e: hypot(e[1] - x, e[2] - y))
    width, height = obs['room']
    best = (-float('inf'), 0)
    for move, (mx, my) in enumerate(MOVE):
        px, py = x + mx * 165 * 0.30, y + my * 165 * 0.30
        blocked = not (radius + 8 < px < width - radius - 8
                       and radius + 8 < py < height - radius - 8)
        blocked |= any(circle_rect(px, py, radius + 5, block) for block in obs['obstacles'])
        score = -1000.0 if blocked else 0.0
        tx, ty = target[1] - px, target[2] - py
        score -= abs(hypot(tx, ty) - 165) * 0.035
        score -= min(abs(tx), abs(ty)) * 0.11  # seek a cardinal firing lane
        for enemy in enemies:
            dist = hypot(enemy[1] + enemy[3] * 0.3 - px,
                         enemy[2] + enemy[4] * 0.3 - py)
            score -= max(0, 105 - dist) * 0.6
        for bullet in obs['bullets']:
            if bullet[0] != 1:
                continue
            ox, oy = float(bullet[1]) - x, float(bullet[2]) - y
            vx, vy = float(bullet[3]) - mx * 165, float(bullet[4]) - my * 165
            speed2 = vx * vx + vy * vy
            t = min(0.6, max(0, -(ox * vx + oy * vy) / speed2)) if speed2 else 0
            miss = hypot(ox + vx * t, oy + vy * t)
            score -= max(0, radius + float(bullet[5]) + 18 - miss) * 1.8
        score -= 0.03 if move else 0
        if score > best[0]:
            best = score, move
    tx, ty = float(target[1]) - x, float(target[2]) - y
    shoot = max(range(1, 5), key=lambda s: SHOOT[s][0] * tx + SHOOT[s][1] * ty)
    return best[1], shoot
