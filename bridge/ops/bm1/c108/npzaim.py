"""read-only, stdlib: C112 L5 (aim) baselines from dumped rollouts (non-forced decisions t < length).
Target: the shoot direction (1 up, 2 right, 3 down, 4 left; env.Action) of the 90-degree sector holding the bearing player
-> nearest enemy. Enemy: an entity slot < n_ent with ROW 'ent' column 23 (flags & 1, FLAG_ENEMY) set; variant 'ev' also
needs column 24 (flags & 2, FLAG_VULNERABLE). Columns 0, 1 = (entity - player) / 200 (world pixels, y down), 2 = dist / 300.
Prints: share of decisions with no target, the shoot share, hit share among shooting decisions with a target, the best
fixed shoot direction's share (max over d of P(target == d)), per variant."""
import sys, zipfile, ast, array, struct, collections
def size(t, shape=()):
    n = int(t[2:])
    for s in shape: n *= s
    return n
def head(h):
    m = h.read(10); hl = int.from_bytes(m[8:10], 'little'); return ast.literal_eval(h.read(hl).decode('latin1'))
def sector(dx, dy):
    if abs(dx) >= abs(dy): return 2 if dx > 0 else 4
    return 3 if dy > 0 else 1
for f in sys.argv[1:]:
    z = zipfile.ZipFile(f)
    with z.open('length.npy') as h: head(h); L = array.array('q'); L.frombytes(h.read())
    with z.open('actions.npy') as h: d = head(h); A = array.array('q'); A.frombytes(h.read()); NH = d['shape'][-1]
    off = 0; fo = {}; fs = {}
    C = {v: collections.Counter() for v in ('e', 'ev')}; TG = {v: collections.Counter() for v in ('e', 'ev')}
    with z.open('rows.npy') as h:
        d = head(h); W, T = d['shape']
        for fld in d['descr']:
            name, t = fld[0], fld[1]; shp = fld[2] if len(fld) > 2 else ()
            fo[name] = off; fs[name] = shp; off += size(t, shp)
        rs = off; EC, EF = fs['ent']
        for w in range(W):
            for t in range(T):
                b = h.read(rs)
                if t >= L[w] or b[fo['pforce']] != 0: continue
                ne = min(struct.unpack_from('<i', b, fo['n_ent'])[0], EC)
                sh = A[(w * T + t) * NH + 1]
                best = {'e': None, 'ev': None}
                for i in range(ne):
                    o = fo['ent'] + 4 * EF * i
                    if struct.unpack_from('<f', b, o + 4 * 23)[0] < 0.5: continue
                    dx, dy, dist = struct.unpack_from('<3f', b, o)
                    vul = struct.unpack_from('<f', b, o + 4 * 24)[0] > 0.5
                    for v in ('e', 'ev'):
                        if v == 'ev' and not vul: continue
                        if best[v] is None or dist < best[v][0]: best[v] = (dist, dx, dy)
                for v in ('e', 'ev'):
                    c = C[v]; c['n'] += 1; c['shoot'] += sh != 0
                    if best[v] is None: c['none'] += 1; continue
                    want = sector(best[v][1], best[v][2])
                    if sh == 0: continue
                    c['sh_t'] += 1; c['hit'] += sh == want; TG[v][want] += 1
    print(f.split('/')[-3], f.split('/')[-1], 'ent', (EC, EF), 'recsize', rs)
    for v in ('e', 'ev'):
        c, tg = C[v], TG[v]; m = max(c['sh_t'], 1)
        print('  %-2s n %d  no-target %.3f  shoot share %.3f  hit/shooting-with-target %.3f  best fixed %s %.3f  targets %s' % (
            v, c['n'], c['none'] / max(c['n'], 1), c['shoot'] / max(c['n'], 1), c['hit'] / m,
            max(tg, key=tg.get) if tg else '-', max(tg.values(), default=0) / m, {k: round(x / m, 3) for k, x in sorted(tg.items())}))
