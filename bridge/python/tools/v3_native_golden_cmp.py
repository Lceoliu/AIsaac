"""Compare <name>.native.bin with <name>.v3.bin section by section (the v3 native encoder's stage checks on the golden
states). usage: python3 native_golden_cmp.py <golden dir> <section,...>"""
import glob, os, sys, struct
H, C, M, PF, LOC, EF, NF, SF = 8, 16 * 28 * 5, 169, 192, 405, 64, 8, 24
def secs(b):
    f = struct.unpack('<%dI' % (len(b) // 4), b)
    p, e, n, s = (int(struct.unpack('<f', struct.pack('<I', v))[0]) for v in f[2:6])
    out, at = {'header': [f[0], f[6], f[7]]}, H
    for name, size in (('canvas', C), ('minimap', M), ('P', p * PF), ('local', p * LOC), ('E', e * EF), ('N', n * NF), ('S', s * SF)):
        out[name] = list(f[at:at + size]); at += size
    return out
def unstall(sec):   # P cols 40 / 53 (stall_share / stall_known): the worker writes them, the fixtures carry 0.5 / 1
    P = sec['P']
    for i in range(0, len(P), PF):
        P[i + 40] = P[i + 53] = 0
d, names = sys.argv[1], sys.argv[2].split(',')
tot = {k: [0, 0] for k in names}
for nb in sorted(glob.glob(os.path.join(d, '*.native.bin'))):
    name = os.path.basename(nb)[:-11]
    a, b = secs(open(nb, 'rb').read()), secs(open(os.path.join(d, name + '.v3.bin'), 'rb').read())
    unstall(a); unstall(b)
    for k in names:
        tot[k][0] += 1
        if a[k] == b[k]: tot[k][1] += 1
        else:
            i = next((i for i, (x, y) in enumerate(zip(a[k], b[k])) if x != y), None)
            print(name, k, 'differs', 'len', len(a[k]), len(b[k]), 'first index', i)
print({k: f'{v[1]}/{v[0]} equal' for k, v in tot.items()})
