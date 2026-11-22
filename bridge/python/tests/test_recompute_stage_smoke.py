"""(2026-11-22, the coordinator) train_tok --recompute-stage 1: the recompute pass's records staged once and its chunks
decoded from slices of the bytes give the same update as --recompute-stage 0, bit for bit -- the real learn() on the
CPU (tests/aux_learn_driver.py: --decoupled 1, so the pass runs; two updates): the weights, the optimiser's state and
progress.csv's loss and statistics columns equal. The v3 records'
side (compact and ROW rollouts): tests/test_rollout_compact.py::test_recompute_stage_chunks_equal_take. The refusal
with --learner-fast 2: test_refuses_with_fast2."""
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


def _driver(out, stage):
    cmd = [sys.executable, '-X', 'utf8', str(HERE / 'aux_learn_driver.py'), '--tree', str(TREE), '--out', str(out),
           '--updates', '2', '--fast', '0', f'--extra=--recompute-stage,{stage}']
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                       env=dict(os.environ, OMP_NUM_THREADS='2'), timeout=900)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-3000:]
    return torch.load(Path(out) / 'result.pt', weights_only=False)


def _steady(rows):
    return [{k: r[k] for k in COLS} for r in rows]


@pytest.mark.skipif(not hasattr(os, 'sched_getaffinity'), reason='train_tok needs os.sched_getaffinity (Linux)')
def test_recompute_stage_same_update():
    tmp = Path(tempfile.mkdtemp(prefix='rstage-'))
    a, b = _driver(tmp / 's0', 0), _driver(tmp / 's1', 1)
    assert list(a['weights']) == list(b['weights'])
    for k in a['weights']:
        assert torch.equal(a['weights'][k], b['weights'][k]), k
    for i, st in a['opt']['state'].items():
        for k, v in st.items():
            if torch.is_tensor(v):
                assert torch.equal(v, b['opt']['state'][i][k]), (i, k)
    assert len(a['progress']) == len(b['progress']) == 2
    assert _steady(a['progress']) == _steady(b['progress'])


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
