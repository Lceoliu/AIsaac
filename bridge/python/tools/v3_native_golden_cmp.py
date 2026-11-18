"""Compare <name>.native.bin with <name>.v3.bin (the v3 native encoder's stage checks on the golden states).
usage: python3 v3_native_golden_cmp.py <golden dir> <section,...>

(2026-11-18, Codex round 3) A pass needs: manifest.json in the directory and, for every frame it lists, both files;
each file's header valid for its own length (every word a whole number, size word = the counts' size = the floats
given: no extra float); the two frames equal word for word -- header included -- but for the stall columns (P 40 /
53 of every body, which the worker writes; the fixtures carry 0.5 / 1, the native side 0); and the body order. The
per-section lines (the listed sections) are for reading only. Exit 0 and PASS, else 1 and FAIL with why."""
import json
import os
import struct
import sys

H, C, M, PF, LOC, EF, NF, SF = 8, 16 * 28 * 5, 169, 192, 405, 64, 8, 24
STALL = (40, 53)


def floats(b):
    return list(struct.unpack('<%df' % (len(b) // 4), b))


def header_ok(f, nbytes):
    """The frame's own header against its length: whole numbers, p >= 1, e / n / s >= 0, size = the counts' size =
    the floats there (and the bytes whole floats)."""
    if nbytes % 4 or len(f) < H or any(v != v or v != int(v) for v in f[:H]):
        return False
    p, e, n, s = (int(v) for v in f[2:6])
    want = H + C + M + p * (PF + LOC) + e * EF + n * NF + s * SF
    return p >= 1 and min(e, n, s) >= 0 and int(f[1]) == want == len(f)


def sections(f):
    p, e, n, s = (int(v) for v in f[2:6])
    out, at = {'header': f[:H]}, H
    for name, size in (('canvas', C), ('minimap', M), ('P', p * PF), ('local', p * LOC), ('E', e * EF), ('N', n * NF),
                       ('S', s * SF)):
        out[name] = f[at:at + size]
        at += size
    return out


def unstall(f):
    p = int(f[2])
    g = list(f)
    for i in range(p):
        for c in STALL:
            g[H + C + M + i * PF + c] = 0.0
    return g


def body_order(f):   # schema batch 4 (visible_producer.v3_body_order, without numpy): [] when it holds
    P = sections(f)['P']
    P = [P[i:i + PF] for i in range(0, len(P), PF)]
    bad = [] if P and int(P[0][57]) == 1 and int(P[0][54]) == 1 else ['row 0']
    for r, row in enumerate(P):
        if int(row[57]) == 3 and (r == 0 or int(row[39]) != r or int(P[r - 1][39]) != r + 1 or int(P[r - 1][57]) != 1):
            bad.append('sub row %d' % r)
        if int(row[57]) == 1 and int(row[39]) and int(row[39]) != r + 2:
            bad.append('main row %d' % r)
    return bad


def main(argv):
    d, names = argv[1], [k for k in argv[2].split(',') if k]
    why = []
    try:
        frames = [f['name'] for f in json.load(open(os.path.join(d, 'manifest.json')))['frames']]
    except (OSError, ValueError, KeyError) as e:
        frames = []
        why.append(f'no manifest.json frame list ({type(e).__name__})')
    if not frames and not why:
        why.append('the manifest lists no frame')
    tot = {k: [0, 0] for k in names}
    equal = 0
    for name in frames:
        try:
            nb, vb = (open(os.path.join(d, f'{name}.{x}.bin'), 'rb').read() for x in ('native', 'v3'))
        except OSError:
            why.append(f'{name}: a file is missing')
            continue
        a, b = floats(nb), floats(vb)
        if not header_ok(a, len(nb)) or not header_ok(b, len(vb)):
            why.append(f'{name}: a header does not fit its file (native {len(nb)} bytes, v3 {len(vb)} bytes)')
            continue
        bad = body_order(a)
        if bad:
            why.append(f'{name}: native body order {bad}')
        ua, ub = unstall(a), unstall(b)
        if ua == ub:
            equal += 1
        else:
            i = next((i for i, (x, y) in enumerate(zip(ua, ub)) if x != y), min(len(ua), len(ub)))
            why.append(f'{name}: differs (lengths {len(ua)} / {len(ub)}, first float {i})')
        sa, sb = sections(ua), sections(ub)
        for k in names:
            tot[k][0] += 1
            tot[k][1] += sa[k] == sb[k]
    print({k: f'{v[1]}/{v[0]} equal' for k, v in tot.items()}, f'frames {equal}/{len(frames)} equal')
    for w in why[:20]:
        print(' ', w)
    ok = not why and equal == len(frames) > 0
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
