"""C108 D4b: hook d4hook into a COPY of a tree's train_tok.py (throwaway trees maa1e98a-d4 / m1ae82d3-d4).
  d4_patch.py <tree dir> new|old     (each anchor exactly once; a CRLF copy keeps its line ends)"""
import shutil
import sys
from pathlib import Path

CR, LF = chr(13), chr(10)
tree, kind = Path(sys.argv[1]), sys.argv[2]
assert tree.name.endswith('-d4'), tree
py = tree / 'bridge' / 'python'
raw = (py / 'train_tok.py').read_bytes().decode()
crlf = (CR + LF) in raw
src = raw.replace(CR + LF, LF)
shutil.copy(Path(__file__).with_name('d4hook.py'), py / 'd4hook.py')
edits = [('import argparse\n', 'import argparse\nimport d4hook as _d4   # C108 D4b (throwaway)\n')]
if kind == 'new':
    edits += [
        ('                    values_b.append(vb_.float()); globals_b.append(gb_.float())\n',
         '                    values_b.append(vb_.float()); globals_b.append(gb_.float())\n'
         '                    if _d4.ON:   # C108 D4b\n'
         '                        _d4.new_group(ctx_b)\n'),
        ('                prox_all = torch.cat(lp_parts)\n                vgs = globals_b\n',
         '                prox_all = torch.cat(lp_parts)\n                vgs = globals_b\n'
         "                if _d4.ON:   # C108 D4b\n"
         "                    _d4.new_prox(rl, rp_b, all_flat, L['update'], out, learner_model)\n"),
    ]
else:
    edits += [
        ('                    value = value_[:k_].float()\n                    a = a_all[lo_:hi_]\n',
         '                    value = value_[:k_].float()\n                    a = a_all[lo_:hi_]\n'
         '                    if _d4.ON:   # C108 D4b\n'
         "                        _d4.old_part(learner_model, k_, part, flat, rl, L['update'], out)\n"),
    ]
for a, b in edits:
    n = src.count(a)
    assert n == 1, (a[:60], n)
    src = src.replace(a, b)
(py / 'train_tok.py').write_bytes((src.replace(LF, CR + LF) if crlf else src).encode())
print('patched', py / 'train_tok.py', len(edits), 'edits', 'crlf' if crlf else 'lf')
