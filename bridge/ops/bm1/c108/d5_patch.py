"""C109 D5: hook d5hook into a COPY of a tree's train_tok.py (throwaway trees maa1e98a-d5 / m1ae82d3-d5).
  d5_patch.py <tree dir> new|old     (each anchor exactly once; a CRLF copy keeps its line ends)
v2: train_tok.py is always read from the production tree next to it (<tree> without -d5), so a re-patch starts clean;
the block tree also gets the ISAAC_D5_CHECK call before the PPO part's forward (inside its autocast)."""
import shutil
import sys
from pathlib import Path

CR, LF = chr(13), chr(10)
tree, kind = Path(sys.argv[1]), sys.argv[2]
assert tree.name.endswith('-d5'), tree
prod = tree.with_name(tree.name[:-3])
py = tree / 'bridge' / 'python'
raw = (prod / 'bridge' / 'python' / 'train_tok.py').read_bytes().decode()
crlf = (CR + LF) in raw
src = raw.replace(CR + LF, LF)
shutil.copy(Path(__file__).with_name('d5hook.py'), py / 'd5hook.py')
edits = [('import argparse\n', 'import argparse\nimport d5hook as _d5   # C109 D5 (throwaway)\n')]
if kind == 'new':
    edits.append(("            m_ = mark('b38:part_backward', m_)\n",
                  "            if _d5.ON:   # C109 D5\n"
                  "                _d5.part(dict(locals()), flat, capacity, L['update'], out, args, 'new')\n"
                  "            m_ = mark('b38:part_backward', m_)\n"))
    a = ("                            with bpt.GpuSpan('learn_fwd_part'):\n"
         "                                outs_,ctx_b = forward_bucketed(learner_model,chunks_b,"
         "max_rows=args.micro or args.minibatch,\n")
    edits.append((a, "                            if _d5.CHECK:   # C109 D5\n"
                     "                                _d5.check(learner_model, forward_bucketed, chunks_b, "
                     "args.micro or args.minibatch, L['update'], args.block_history_mode, out)\n" + a))
else:
    a = ("                    loss.backward()\n                    with torch.no_grad():\n"
         "                        if pf_on:   # (B23: kl and clip share over the PPO-trained decisions' rows)\n")
    edits.append((a, "                    if _d5.ON:   # C109 D5\n"
                     "                        _d5.part(dict(locals()), flat, rl.cap, L['update'], out, args, 'old')\n" + a))
for a, b in edits:
    n = src.count(a)
    assert n == 1, (a[:60], n)
    src = src.replace(a, b)
(py / 'train_tok.py').write_bytes((src.replace(LF, CR + LF) if crlf else src).encode())
print('patched', py / 'train_tok.py', 'from', prod, len(edits), 'edits', 'crlf' if crlf else 'lf')
