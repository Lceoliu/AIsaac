"""C109 D5 v2: summary of d5_grad.jsonl + d5_gram.json (read-only). d5_sum.py [--n 16] <run dir> ...
Per run and parameter group (and shared = all groups but heads / other, all = every group), over the first N updates:
  pairwise cos of the whole-update pg gradients (N(N-1)/2 pairs): mean, jackknife SE over updates, positive share;
  cos(mean of the first N/2, mean of the last N/2); SNR = mean off-diagonal dot / mean |g|^2 (|mu|^2 / total energy);
  the same pairwise mean for total; the halves' cos per update (0-7 / 8-15 and even / odd workers), mean and SE
  (sd / sqrt N); the terms' gradient norms, median over the updates; the rows per half."""
import json
import math
import statistics as st
import sys

TERMS = ('pg_A', 'pg_B', 'pg_E', 'pg_O', 'full_A', 'full_B', 'vf', 'ent', 'imit', 'glob', 'priv', 'aux', 'teacher',
         'v2', 'total')
args = sys.argv[1:]
N = 16
if args[:1] == ['--n']:
    N, args = int(args[1]), args[2:]


def pair_cos(M, idx):
    c = [M[i][j] / math.sqrt(M[i][i] * M[j][j]) for a, i in enumerate(idx) for j in idx[a + 1:]
         if M[i][i] > 0 and M[j][j] > 0]
    return c


def stats(M, n):
    idx = list(range(n))
    c = pair_cos(M, idx)
    if len(c) < 2:
        return (float('nan'),) * 5
    mean = st.mean(c)
    jk = [st.mean(pair_cos(M, [i for i in idx if i != k])) for k in idx]
    jm = st.mean(jk)
    se = math.sqrt((n - 1) / n * sum((x - jm) ** 2 for x in jk))
    pos = sum(1 for x in c if x > 0) / len(c)
    h = n // 2
    aa = sum(M[i][j] for i in range(h) for j in range(h))
    bb = sum(M[i][j] for i in range(h, n) for j in range(h, n))
    ab = sum(M[i][j] for i in range(h) for j in range(h, n))
    c8 = ab / math.sqrt(aa * bb) if aa > 0 and bb > 0 else float('nan')
    off = st.mean(M[i][j] for i in idx for j in idx if i != j)
    diag = st.mean(M[i][i] for i in idx)
    return mean, se, pos, c8, off / diag


def ms(x):
    x = [v for v in x if v is not None]
    return (st.mean(x), st.stdev(x) / math.sqrt(len(x))) if len(x) > 1 else (float('nan'), float('nan'))


def f(v, w=7, p=3):
    return (f'%.{p}f' % v).rjust(w) if v is not None and v == v else 'n/a'.rjust(w)


for d in args:
    R = [json.loads(line) for line in open(d + '/d5_grad.jsonl')]
    R = [r for r in R if r['parts'] > 0]
    Gm = json.load(open(d + '/d5_gram.json'))
    ups = Gm['updates']
    n = min(N, len(ups))
    R = [r for r in R if r['update'] in ups[:n]]
    G = Gm['groups'] + ['shared', 'all']
    gram = {}
    for t in ('pg', 'total'):
        gram[t] = dict(Gm['gram'][t])
        for name, members in (('shared', Gm['shared']), ('all', Gm['groups'])):
            gram[t][name] = [[sum(Gm['gram'][t][g][i][j] for g in members) for j in range(len(ups))]
                             for i in range(len(ups))]
    print(f'== {d}: {n} updates {ups[:n]}; parts {sorted({r["parts"] for r in R})}; decisions '
          f'{min(r["decisions"] for r in R)}-{max(r["decisions"] for r in R)}')
    hv = {h: [(r['halves'][h]['rows'], r['halves'][h]['free']) for r in R] for h in 'ABEO'}
    print('  rows per half (all / PPO-trained, mean): ' + '  '.join(
        f'{h} {st.mean(x[0] for x in hv[h]):.0f}/{st.mean(x[1] for x in hv[h]):.0f}' for h in 'ABEO'))
    print('  group     n_params | pg pairwise cos mean  SE  pos | cos(first8,last8) | SNR |mu|^2/E | '
          'total pairwise | halves 0-7/8-15 mean SE | even/odd mean SE | full halves 0-7/8-15, even/odd')
    for g in G:
        pm, pse, pp, c8, snr = stats(gram['pg'][g], n)
        tm = stats(gram['total'][g], n)
        npar = sum(Gm['n_params'][x] for x in (Gm['shared'] if g == 'shared' else Gm['groups'] if g == 'all' else [g]))
        ab = ms([r['groups'][g]['cos_pg_halves'] for r in R])
        eo = ms([r['groups'][g]['cos_pg_eo'] for r in R])
        fab = ms([r['groups'][g]['cos_full_halves'] for r in R])
        feo = ms([r['groups'][g]['cos_full_eo'] for r in R])
        print(f'  {g:8s} {npar:9d} | {f(pm)} {f(pse)} {f(pp, 5, 2)} | {f(c8)} | {f(snr, 8, 4)} | {f(tm[0])} | '
              f'{f(ab[0])} {f(ab[1])} | {f(eo[0])} {f(eo[1])} | {f(fab[0])} {f(feo[0])}')
    print('  term gradient norms, median over the updates:')
    print('  group    ' + ' '.join(t.rjust(7) for t in TERMS))
    for g in G:
        print(f'  {g:8s} ' + ' '.join(f(st.median(r['groups'][g]['norm'][t] for r in R), 7, 3) for t in TERMS))
