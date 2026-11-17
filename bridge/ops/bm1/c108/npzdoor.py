"""read-only, stdlib: C112 L3 / L4 baselines from dumped rollouts (non-forced decisions t < length).
L3 (door): target = the move (8 directions) closest to the bearing player -> nearest OPEN door (ROW 'doors' cols 0, 1 =
(door - player) / 200, world pixels, y down; col 2 = open; slots < n_doors); share of decisions with no open door; hit
share among moving decisions with a target; the best fixed-direction policy's share (max over moves of P(target == m)).
L4 (room): EV_NEW_ROOM | EV_EXIT bits of the records (the step that led there), per decision and per episode start."""
import sys, zipfile, ast, array, struct, collections, math
OCT = (3, 4, 5, 6, 7, 8, 1, 2)   # octant of atan2(dy, dx) (0 = right, +1 = 45 deg clockwise on screen) -> move action
def size(t, shape=()):
    n = int(t[2:])
    for s in shape: n *= s
    return n
def head(h):
    m = h.read(10); hl = int.from_bytes(m[8:10], 'little'); return ast.literal_eval(h.read(hl).decode('latin1'))
for f in sys.argv[1:]:
    z = zipfile.ZipFile(f)
    with z.open('length.npy') as h: head(h); L = array.array('q'); L.frombytes(h.read())
    with z.open('actions.npy') as h: d = head(h); A = array.array('q'); A.frombytes(h.read()); NH = d['shape'][-1]
    off = 0; fo = {}
    c = collections.Counter(); tgt = collections.Counter()
    with z.open('rows.npy') as h:
        d = head(h); W, T = d['shape']
        for fld in d['descr']:
            name, t = fld[0], fld[1]; shp = fld[2] if len(fld) > 2 else ()
            fo[name] = off; off += size(t, shp)
        rs = off
        for w in range(W):
            for t in range(T):
                b = h.read(rs)
                if t >= L[w]: continue
                ev = b[fo['events']]
                c['dec'] += 1; c['starts'] += b[fo['first']] != 0; c['entries'] += (ev & (2 | 8)) != 0
                if b[fo['pforce']] != 0: c['forced'] += 1; continue
                c['n'] += 1
                nd = struct.unpack_from('<i', b, fo['n_doors'])[0]
                best = None
                for i in range(min(nd, 8)):
                    dx, dy, op = struct.unpack_from('<3f', b, fo['doors'] + 28 * i)
                    if op > 0.5 and (best is None or dx * dx + dy * dy < best[0]): best = (dx * dx + dy * dy, dx, dy)
                mv = A[(w * T + t) * NH]
                if best is None: c['nodoor'] += 1; continue
                want = OCT[round(math.atan2(best[2], best[1]) / (math.pi / 4)) % 8]
                if mv == 0: continue
                c['mov_t'] += 1; c['hit'] += mv == want; tgt[want] += 1
    m = max(c['mov_t'], 1)
    print(f.split('/')[-3], f.split('/')[-1], dict(c), 'recsize', rs)
    print('  L3 no-open-door share %.3f  hit/moving-with-target %.3f  best fixed move %s %.3f  targets %s' % (
        c['nodoor'] / max(c['n'], 1), c['hit'] / m, max(tgt, key=tgt.get) if tgt else '-', max(tgt.values(), default=0) / m,
        {k: round(v / m, 3) for k, v in sorted(tgt.items())}))
    print('  L4 entries %d per 1000 decisions %.2f  per episode start %.3f' % (
        c['entries'], 1000 * c['entries'] / max(c['dec'], 1), c['entries'] / max(c['starts'], 1)))
