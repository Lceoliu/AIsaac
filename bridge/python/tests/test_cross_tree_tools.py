"""tools/cross_tree_exact.py --analyze and tools/a8_digests.py compare (2026-11-18, Codex round 3): a verdict, not a
diagnostic -- analyze passes only with records on both sides, every part equal but the launch-dependent ones (players in
the id field only, logic_frames as increments, terrain / visible JSON as values with no field differing); the A8
comparison needs both probes' own summaries all ok, besides equal digests. Both tools clear ISAAC_RL_TRIGGER_ONCE too."""
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from isaac_bridge import abplus_lean as al

PY = Path(__file__).resolve().parents[1]


def payload(hearts=6, pid=0, visible=None):
    b = bytearray(struct.pack('<IIIBBBB', al.MAGIC, 1000, 990, 32 if visible is not None else 0, 1, 0, 0))
    b += bytes(48) + bytes(32)
    pl = np.zeros(1, al.PLAYER)
    pl['hearts'], pl['id'] = hearts, pid
    b += pl.tobytes() + struct.pack('<H', 0)
    if visible is not None:
        t = json.dumps(visible).encode()
        b += struct.pack('<I', len(t)) + t
    return bytes(b)


def recording(d, payloads):
    d.mkdir()
    with open(d / 'payloads.bin', 'wb') as fp, open(d / 'index.jsonl', 'w') as fi:
        fi.write(json.dumps(dict(env={}, row_itemsize=0, args={})) + '\n')
        for t, p in enumerate(payloads):
            fi.write(json.dumps(dict(seed=1, items=False, t=t, p_off=fp.tell(), p_len=len(p))) + '\n')
            fp.write(p)
    return d


def analyze(a, b):
    return subprocess.run([sys.executable, str(PY / 'tools' / 'cross_tree_exact.py'), '--analyze', str(a), str(b)],
                          cwd=PY, capture_output=True, text=True).returncode


def test_analyze_verdict(tmp_path):
    vis = lambda v: dict(schema='isaac-visible-prlmens-g-v2', bodies=[dict(a=v)])
    base = recording(tmp_path / 'a', [payload(visible=vis(1)), payload(visible=vis(2))])
    assert analyze(base, recording(tmp_path / 'same', [payload(visible=vis(1)), payload(visible=vis(2))])) == 0
    assert analyze(base, recording(tmp_path / 'id', [payload(pid=5, visible=vis(1)), payload(pid=5, visible=vis(2))])) == 0
    assert analyze(base, recording(tmp_path / 'hp', [payload(5, visible=vis(1)), payload(visible=vis(2))])) == 1
    assert analyze(base, recording(tmp_path / 'vis', [payload(visible=vis(1)), payload(visible=vis(3))])) == 1
    assert analyze(base, recording(tmp_path / 'one', [payload(visible=vis(1)), payload()])) == 1
    assert analyze(recording(tmp_path / 'e1', []), recording(tmp_path / 'e2', [])) == 1


def test_a8_compare_needs_both_summaries_ok(tmp_path):
    def run(name, ok, digests=('x', 'y')):
        d = tmp_path / name
        d.mkdir()
        (d / 'digests.jsonl').write_text(''.join(json.dumps(dict(call=i, d=v)) + '\n' for i, v in enumerate(digests)))
        (d / 'summary.json').write_text(json.dumps(dict(seeds=8, ok=ok)))
        return d
    cmp = lambda a, b: subprocess.run([sys.executable, str(PY / 'tools' / 'a8_digests.py'), 'compare', str(a), str(b)],
                                      capture_output=True, text=True).returncode
    assert cmp(run('a', 8), run('b', 8)) == 0
    assert cmp(run('c', 8), run('d', 7)) == 1                      # a probe seed not ok: no pass on equal digests
    assert cmp(run('e', 8), run('f', 8, ('x', 'z'))) == 1
    assert cmp(run('g', 8, ()), run('h', 8, ())) == 1


def test_the_old_arm_clears_the_trigger_switch():
    for tool in ('cross_tree_exact.py', 'a8_digests.py'):
        assert 'ISAAC_RL_TRIGGER_ONCE' in (PY / 'tools' / tool).read_text(encoding='utf-8')
