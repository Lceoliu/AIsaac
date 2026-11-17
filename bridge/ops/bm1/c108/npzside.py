"""read-only, stdlib: random-ish baselines for C112 from dumped rollouts: share of move==3 (right) among moving steps
(L1) and of the L2 target (x<0.5 -> 3, else 7) among moving steps, non-forced steps (pforce == 0) of t < length."""
import sys, zipfile, ast, array, struct, collections
def size(t, shape=()):
    n = int(t[2:]) if t[1] in 'fiu' else int(t[2:]) if t[1] == 'V' else int(t[2:])
    for s in shape: n *= s
    return n
def head(h):
    m = h.read(10); hl = int.from_bytes(m[8:10], 'little'); return ast.literal_eval(h.read(hl).decode('latin1'))
for f in sys.argv[1:]:
    z = zipfile.ZipFile(f)
    with z.open('length.npy') as h: d = head(h); L = array.array('q'); L.frombytes(h.read())
    with z.open('actions.npy') as h: d = head(h); A = array.array('q'); A.frombytes(h.read())
    off = 0; fo = {}
    with z.open('rows.npy') as h:
        d = head(h); W, T = d['shape']
        for fld in d['descr']:
            name, t = fld[0], fld[1]; shp = fld[2] if len(fld) > 2 else ()
            fo[name] = off; off += size(t, shp)
        rs = off; c = collections.Counter(); xs = []
        for w in range(W):
            for t in range(T):
                b = h.read(rs)
                if t >= L[w]: continue
                if b[fo['pforce']] != 0: c['forced'] += 1; continue
                x = struct.unpack_from('<f', b, fo['player'])[0]; mv = A[(w*T+t)*5]
                c['n'] += 1; xs.append(x)
                if mv == 0: continue
                c['mov'] += 1; c['right'] += mv == 3; c['left'] += mv == 7
                c['l2'] += mv == (3 if x < 0.5 else 7)
    xs.sort(); n = len(xs)
    print(f.split('/')[-3], f.split('/')[-1], dict(c), 'recsize', rs)
    print('  L1 right/moving %.3f  left/moving %.3f  L2 target/moving %.3f  x<0.5 share %.3f  x p1/p50/p99 %.3f %.3f %.3f' % (
        c['right']/c['mov'], c['left']/c['mov'], c['l2']/c['mov'], sum(x < .5 for x in xs)/n, xs[n//100], xs[n//2], xs[99*n//100]))
