"""C108 D1: hook d1diag into a COPY of a tree's train_tok.py (throwaway trees maa1e98a-d1 / m1ae82d3-d1).
  d1_patch.py <tree dir> new|old     (each anchor must occur exactly once; the original tree is never touched;
  a copy with CRLF line ends keeps them)"""
import shutil
import sys
from pathlib import Path

CR, LF = chr(13), chr(10)
tree, kind = Path(sys.argv[1]), sys.argv[2]
assert tree.name.endswith('-d1'), tree
py = tree / 'bridge' / 'python'
raw = (py / 'train_tok.py').read_bytes().decode()
crlf = (CR + LF) in raw
src = raw.replace(CR + LF, LF)
shutil.copy(Path(__file__).with_name('d1diag.py'), py / 'd1diag.py')
edits = [('import argparse\n', 'import argparse\nimport d1diag as _d1   # C108 D1 (throwaway)\n')]
if kind == 'new':
    edits += [
        ('                prox_all = torch.cat(lp_parts)\n                vgs = globals_b\n',
         '                prox_all = torch.cat(lp_parts)\n                vgs = globals_b\n'
         "                if _d1.ON:   # C108 D1\n"
         "                    _d1.prox(rl, rp_b, all_flat, prox_all, L['update'], out, learner_model)\n"),
        ("            value = value_[:k_].float()\n            a = R['a']\n",
         "            value = value_[:k_].float()\n            a = R['a']\n"
         "            if _d1.ON:   # C108 D1\n"
         "                _d1.part(logp, R['logp'], R.get('pf'), None)\n"),
    ]
else:
    edits += [
        ('                    value = value_[:k_].float()\n                    a = a_all[lo_:hi_]\n',
         '                    value = value_[:k_].float()\n                    a = a_all[lo_:hi_]\n'
         '                    if _d1.ON:   # C108 D1\n'
         "                        _d1.part(logp, b_logp[part], b_pf[part] if b_pf is not None else None, "
         "L['update'], out, learner_model)\n"),
    ]
for a, b in edits:
    n = src.count(a)
    assert n == 1, (a[:60], n)
    src = src.replace(a, b)
(py / 'train_tok.py').write_bytes((src.replace(LF, CR + LF) if crlf else src).encode())
print('patched', py / 'train_tok.py', len(edits), 'edits', 'crlf' if crlf else 'lf')
