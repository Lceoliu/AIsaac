"""(2026-11-22, the coordinator) train_tok --recompute-stage 1: the recompute pass's records staged once and its chunks
decoded from slices of the bytes give the same update as --recompute-stage 0, bit for bit -- the real learn() on the
CPU (tests/aux_learn_driver.py: --decoupled 1, so the pass runs; two updates): the weights, the optimiser's state and
progress.csv's loss and statistics columns equal. The v3 records' side (compact and ROW rollouts):
tests/test_rollout_compact.py::test_recompute_stage_chunks_equal_take. The refusals with --learner-fast 2 and
--block-memory 1: test_refuses_with_fast2.

(Codex round 30) the stage's growth: recompute_stage_buffer replaces a stage too small only when torch can release its
pinned cache (torch._C._host_emptyCache), else the update falls back to the unstaged pass and the stage is kept --
test_stage_buffer_grows_or_falls_back (the helper alone, both cases) and test_recompute_stage_forced_growth_and_fallback
(the real learn(): the second update's stage forced too small, through each path, the update bit for bit that of
--recompute-stage 0, the path's line in the log)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
TREE = HERE.parent.parent.parent
# (progress.csv's columns of the update's result: the losses and statistics; timing, CPU and memory columns vary)
COLS = ('update', 'decisions', 'lr', 'pg', 'vf', 'entropy', 'kl', 'clip_frac', 'explained_variance', 'reward_mean',
        'teach_loss', 'danger_loss', 'safe_mass', 'v_global_loss', 'v_global_ev', 'ent_move', 'ent_bomb', 'ent_item',
        'ent_pill', 'ent_tokens', 'teach_held')
LINUX = pytest.mark.skipif(not hasattr(os, 'sched_getaffinity'), reason='train_tok needs os.sched_getaffinity (Linux)')

# (the driver with the second update's stage forced too small: its need x 4 -- past the stage's 1.25 headroom -- from the helper's second call on;
#  'grow': torch._C._host_emptyCache present -- a counting stand-in where this build lacks it; 'fallback': removed)
FORCE = r'''
import os, runpy, sys
mode, driver, tree = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [driver] + sys.argv[4:]
sys.path.insert(0, os.path.join(tree, 'bridge', 'python'))
import torch
import train_tok as TT
if mode == 'fallback':
    if hasattr(torch._C, '_host_emptyCache'):
        delattr(torch._C, '_host_emptyCache')
elif not hasattr(torch._C, '_host_emptyCache'):
    torch._C._host_emptyCache = lambda: print('stand-in host cache release', flush=True)
orig, calls = TT.recompute_stage_buffer, [0]
def forced(L, need, log=print):
    calls[0] += 1
    return orig(L, need * (4 if calls[0] > 1 else 1), log)
TT.recompute_stage_buffer = forced
runpy.run_path(driver, run_name='__main__')
'''


def _driver(out, stage, force=None):
    args = ['--tree', str(TREE), '--out', str(out), '--updates', '2', '--fast', '0', f'--extra=--recompute-stage,{stage}']
    drv = str(HERE / 'aux_learn_driver.py')
    cmd = ([sys.executable, '-X', 'utf8', drv] + args if force is None else
           [sys.executable, '-X', 'utf8', '-c', FORCE, force, drv, str(TREE)] + args)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                       env=dict(os.environ, OMP_NUM_THREADS='2'), timeout=900)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-3000:]
    return torch.load(Path(out) / 'result.pt', weights_only=False), p.stdout


def _steady(rows):
    return [{k: r[k] for k in COLS} for r in rows]


def _same_update(a, b):
    assert list(a['weights']) == list(b['weights'])
    for k in a['weights']:
        assert torch.equal(a['weights'][k], b['weights'][k]), k
    for i, st in a['opt']['state'].items():
        for k, v in st.items():
            if torch.is_tensor(v):
                assert torch.equal(v, b['opt']['state'][i][k]), (i, k)
    assert len(a['progress']) == len(b['progress']) == 2
    assert _steady(a['progress']) == _steady(b['progress'])


@LINUX
def test_recompute_stage_same_update():
    tmp = Path(tempfile.mkdtemp(prefix='rstage-'))
    (a, _), (b, out) = _driver(tmp / 's0', 0), _driver(tmp / 's1', 1)
    _same_update(a, b)
    assert '--recompute-stage: the stage' in out and 'MiB pinned' in out


@LINUX
def test_recompute_stage_forced_growth_and_fallback():
    tmp = Path(tempfile.mkdtemp(prefix='rstage-f-'))
    a, _ = _driver(tmp / 's0', 0)
    g, out_g = _driver(tmp / 'grow', 1, 'grow')
    _same_update(a, g)
    assert 'the stage grows' in out_g and 'growth 1' in out_g, out_g[-2000:]
    f, out_f = _driver(tmp / 'fallback', 1, 'fallback')
    _same_update(a, f)
    assert 'recomputes unstaged' in out_f and 'fallback 1' in out_f, out_f[-2000:]
    assert 'pinned-cache release (torch._C._host_emptyCache) not available' in out_f


def test_stage_buffer_grows_or_falls_back(monkeypatch):
    """recompute_stage_buffer alone (CPU: pin_memory a no-op here): a stage large enough is reused; a larger need with
    the release present replaces it (one stage kept, the release called once, logged); without it, None (the update
    unstaged), the stage kept, logged."""
    sys.path.insert(0, str(HERE.parent))
    import train_tok as TT
    monkeypatch.setattr(torch.Tensor, 'pin_memory', lambda self, *a, **k: self)
    released = []
    monkeypatch.setattr(torch._C, '_host_emptyCache', lambda: released.append(1), raising=False)
    logs = []
    log = lambda m, **k: logs.append(m)   # noqa: E731  (the helper's log takes print's flush=)
    L = {}
    s1 = TT.recompute_stage_buffer(L, 1000, log)
    assert L['stage_rc'] is s1 and s1.numel() >= 1000 and 'MiB pinned' in logs[-1] and 'available' in logs[-1]
    assert TT.recompute_stage_buffer(L, s1.numel(), log) is s1 and len(logs) == 1   # (fits: reused, quiet)
    s2 = TT.recompute_stage_buffer(L, 10 * s1.numel(), log)
    assert s2 is not s1 and L['stage_rc'] is s2 and s2.numel() >= 10 * s1.numel()
    assert released == [1] and L['stage_rc_grows'] == 1 and 'grows' in logs[-1]
    monkeypatch.delattr(torch._C, '_host_emptyCache')
    L2 = {}
    t1 = TT.recompute_stage_buffer(L2, 1000, log)
    assert 'not available' in logs[-1]
    assert TT.recompute_stage_buffer(L2, 10 * t1.numel(), log) is None
    assert L2['stage_rc'] is t1 and L2['stage_rc_fallbacks'] == 1 and 'recomputes unstaged' in logs[-1]
    assert released == [1]


def test_refuses_with_fast2(capsys, tmp_path):
    sys.path.insert(0, str(HERE.parent))
    import train_tok
    with pytest.raises(SystemExit):
        train_tok.main(['--groups-file', str(tmp_path / 'missing.json'), '--out', str(tmp_path / 'o'),
                        '--recompute-stage', '1', '--learner-fast', '2'])
    assert '--recompute-stage 1 with --learner-fast 2' in capsys.readouterr().err
    with pytest.raises(SystemExit):   # (A1's review: the block memory's recompute is a pass of its own)
        train_tok.main(['--groups-file', str(tmp_path / 'missing.json'), '--out', str(tmp_path / 'b'),
                        '--recompute-stage', '1', '--block-memory', '1'])
    assert '--recompute-stage 1 does not act on --block-memory 1' in capsys.readouterr().err
