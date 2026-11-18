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


def test_close_stress_verdict_needs_coverage():
    """tools/v3_close_stress.verdict (Codex rounds 5 / 9): coverage and outcome apart. A kill counts only when it ended
    the traced operation in an error and its worker gave records afterwards -- ten 'interrupted' operations that all
    ended 'ok' (a kill that missed the I/O) are no coverage; nor kills without recovery; death-teaching coverage is the
    line's own events; in mode 0 a mode query sent fails; an exit 86 / a worker not ended fails."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('close_stress', PY / 'tools' / 'v3_close_stress.py')
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)

    def row(k, kind, records=0, hit=None, codes=(0, 0, 0, 0), death=0, sent=0):
        return dict(round=k, kind=kind, records=records, hit=hit, exit_codes=list(codes), fatal_during=None,
                    fatal_on_close=None, fatal_files=[], site_errors=[], death_events=death, queries_sent=sent)
    kinds = ('play', 'reset-kill', 'early')
    broke = dict(pid=1, index=0, t=1.0, killed=1, interrupted=[dict(tag='r-1-1', outcome='BridgeError')], failed=True,
                 recovered=True)
    good = [row(k, kinds[k % 3], records=50 if k % 3 == 0 else 0, hit=broke if k % 3 == 1 else None,
                death=1 if k % 3 == 0 else 0) for k in range(30)]
    v = cs.verdict(good)
    assert v['ok'], v
    assert v['coverage'] == dict(play_with_records=10, kills_that_broke_and_recovered=10, early=10,
                                 death_teaching_events=10)
    missed = dict(broke, interrupted=[dict(tag='r-1-1', outcome='ok')], failed=False)   # (Codex's case)
    assert not cs.verdict([dict(r, hit=missed) if r['hit'] else r for r in good])['covered']
    stuck = dict(broke, recovered=False)
    assert not cs.verdict([dict(r, hit=stuck) if r['hit'] else r for r in good])['covered']
    assert not cs.verdict([dict(r, death_events=0) for r in good])['covered']
    assert not cs.verdict(good[:-1] + [row(29, 'early', sent=1)])['ok']             # (mode 0: a query sent)
    assert cs.verdict([dict(r, kind='query-kill') if r['kind'] == 'reset-kill' else r for r in good] [:-1]
                      + [row(29, 'early', sent=1)], v3=1)['ok']                        # (v3: queries are sent)
    assert not cs.verdict(good[:-1] + [row(29, 'early', codes=(0, None, 0, 0))])['ok']
    assert not cs.verdict(good[:-1] + [row(29, 'early', codes=(0, 86, 0, 0))])['ok']


def test_golden_v3_diff_checks_hashes_and_empty_manifests(tmp_path):
    """tools/golden_v3_diff.py (2026-11-18, Codex round 3 notes): the batch-5 migration still passes (old = 9e941f3~1);
    a manifest sha256 not matching its file fails; two empty manifests are no pass."""
    import shutil
    tool = PY / 'tools' / 'golden_v3_diff.py'
    args = ['--old-version', '2026111801', '--old-word', '3', '--new-version', '2026111802', '--new-word', '111802']
    run = lambda *a: subprocess.run([sys.executable, str(tool), *a, *args], cwd=PY, capture_output=True, text=True)
    import pytest
    try:   # (the gate runs on a git-archive tree: no history to read the old fixtures from)
        git = subprocess.run(['git', 'cat-file', '-e', '9e941f3~1'], cwd=PY, capture_output=True)
    except OSError:
        git = None
    have_git = git is not None and git.returncode == 0
    if have_git:   # (the batch-5 migration against the old fixtures read from git)
        r = run('--old-rev', '9e941f3~1')
        assert r.returncode == 0, r.stdout[-500:]
    fix = PY / 'tests' / 'fixtures' / 'visible_v3'
    d = tmp_path / 'new'
    shutil.copytree(fix, d)
    m = json.loads((d / 'manifest.json').read_text())
    m['frames'][0]['sha256'] = '0' * 64
    (d / 'manifest.json').write_text(json.dumps(m))
    r = run('--old', str(fix), '--new', str(d))   # (the fixtures against themselves: only the sha256 can fail)
    assert r.returncode == 1 and 'does not match its sha256' in r.stdout
    e1, e2 = tmp_path / 'e1', tmp_path / 'e2'
    for e in (e1, e2):
        e.mkdir()
        (e / 'manifest.json').write_text(json.dumps(dict(frames=[], errors=[])))
    r = run('--old', str(e1), '--new', str(e2))
    assert r.returncode == 1 and 'lists no frame' in r.stdout
