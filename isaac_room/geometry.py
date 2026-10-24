"""2D collision primitives. Coordinates are room pixels, time is seconds."""
from math import sqrt


def circle_rect(x, y, radius, rect):
    left, top, width, height = rect
    nx = min(max(x, left), left + width)
    ny = min(max(y, top), top + height)
    return (x - nx) ** 2 + (y - ny) ** 2 < radius ** 2 - 1e-9


def move_circle(x, y, dx, dy, radius, blocks, width, height):
    # Axis-separated sliding; callers use bounded 60 Hz movement increments.
    nx = min(max(x + dx, radius), width - radius)
    for rect in blocks:
        if circle_rect(nx, y, radius, rect):
            left, top, rw, rh = rect
            near_y = min(max(y, top), top + rh)
            gap = sqrt(max(0.0, radius ** 2 - (y - near_y) ** 2))
            nx = left - gap if dx > 0 else left + rw + gap
    ny = min(max(y + dy, radius), height - radius)
    for rect in blocks:
        if circle_rect(nx, ny, radius, rect):
            left, top, rw, rh = rect
            near_x = min(max(nx, left), left + rw)
            gap = sqrt(max(0.0, radius ** 2 - (nx - near_x) ** 2))
            ny = top - gap if dy > 0 else top + rh + gap
    return nx, ny


def segment_circle(ax, ay, bx, by, cx, cy, radius):
    """First impact t in [0, 1], or None, including starting inside."""
    dx, dy, ox, oy = bx - ax, by - ay, ax - cx, ay - cy
    c = ox * ox + oy * oy - radius * radius
    if c <= 0:
        return 0.0
    a = dx * dx + dy * dy
    if a == 0:
        return None
    b = 2 * (ox * dx + oy * dy)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t = (-b - sqrt(disc)) / (2 * a)
    return t if 0 <= t <= 1 else None


def segment_rect(ax, ay, bx, by, rect):
    left, top, width, height = rect
    lo, hi = 0.0, 1.0
    for p, d, low, high in ((ax, bx - ax, left, left + width),
                            (ay, by - ay, top, top + height)):
        if d == 0:
            if not low <= p <= high:
                return None
        else:
            t0, t1 = sorted(((low - p) / d, (high - p) / d))
            lo, hi = max(lo, t0), min(hi, t1)
            if lo > hi:
                return None
    return lo


def swept_circle_rect(ax, ay, bx, by, radius, rect):
    # Rounded Minkowski sum: two strips plus four corner circles.
    left, top, width, height = rect
    hits = [segment_rect(ax, ay, bx, by, (left - radius, top, width + 2 * radius, height)),
            segment_rect(ax, ay, bx, by, (left, top - radius, width, height + 2 * radius))]
    for x in (left, left + width):
        for y in (top, top + height):
            hits.append(segment_circle(ax, ay, bx, by, x, y, radius))
    return min((t for t in hits if t is not None), default=None)


def wall_impact(ax, ay, bx, by, radius, width, height):
    hits = []
    for a, b, low, high in ((ax, bx, radius, width - radius),
                            (ay, by, radius, height - radius)):
        if a < low or a > high:
            return 0.0
        if b < low:
            hits.append((low - a) / (b - a))
        if b > high:
            hits.append((high - a) / (b - a))
    return min(hits, default=None)
