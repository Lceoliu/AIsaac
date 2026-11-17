"""read-only, stdlib: per dumped rollout, |reward| stats and the move-action histogram over valid steps (t < length)"""
import sys, zipfile, ast, array, collections
def load(z, name):
    with z.open(name) as h:
        m = h.read(10); hl = int.from_bytes(m[8:10], 'little')
        d = ast.literal_eval(h.read(hl).decode('latin1')); b = h.read()
    t = {'<f4': 'f', '<i8': 'q', '|b1': 'B'}[d['descr']]
    a = array.array(t); a.frombytes(b); return a, d['shape']
for f in sys.argv[1:]:
    z = zipfile.ZipFile(f)
    r, (W, T) = load(z, 'reward.npy'); act, _ = load(z, 'actions.npy'); L, _ = load(z, 'length.npy')
    vals = []; mv = collections.Counter()
    for w in range(W):
        for t in range(L[w]):
            vals.append(r[w*T+t]); mv[act[(w*T+t)*5]] += 1
    n = len(vals); nz = [abs(v) for v in vals if v != 0]
    print(f.split('/')[-3], f.split('/')[-1], 'steps', n, 'mean r %.5f' % (sum(vals)/n), 'mean|r| %.5f' % (sum(map(abs, vals))/n),
          'nonzero %.3f' % (len(nz)/n), 'mean|r| nonzero %.4f' % (sum(nz)/max(1, len(nz))), 'max|r| %.3f' % max(map(abs, vals)))
    print('  move hist', sorted(mv.items()))
    cnt = collections.Counter(round(v, 4) for v in vals); print('  top values', cnt.most_common(8))
