"""seq-v1 (2026-11-18, the coordinator): tok_obs.write_visible's three cases after package C (05f1622) met step 0's
--obs-visible 0, and the one pair of settings that cannot both hold. CPU only.

- v3 on: a record with a v3 frame gets that frame (visible_schema = schema_v3.VERSION, the bytes as they came);
- v3 off and a v2 visible block: the v2 frame (visible_native.frame -> visible_producer.write_frame, the function
  structured_input re-exports: write_visible takes it from visible_producer, without torch -- tests/test_worker_lean.py);
- --obs-visible 0 (ISAAC_NO_VISIBLE=1): no frame at all, v3 or v2, even with both present (visible_schema 0);
- a worker configured for v3 without a v3 frame refuses the record (05f1622);
- --obs-visible 0 together with ISAAC_RL_VISIBLE_V3 = a mode is refused by train_tok / eval_tok at argument parsing
  (tok_obs.obs_visible_conflict), so a v3 configuration never reaches write_visible's --obs-visible 0 branch."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from isaac_bridge import schema_v3 as V3
from isaac_bridge import tok_obs, visible_native
from isaac_bridge import visible_producer as VP
from isaac_bridge.tok_obs import ROW

HERE = Path(__file__).resolve().parent
FIX = HERE / 'fixtures' / 'visible_v3'
GOLDEN = (FIX / f"{json.loads((FIX / 'manifest.json').read_text())['frames'][0]['name']}.v3.bin").read_bytes()
V2_SCHEMA = 4242   # (the stub's v2 schema number)


class Obs:
    def __init__(self, v3=None, v2=None):
        self.visible_v3_frame, self.visible = v3, v2


@pytest.fixture
def v2_stub(monkeypatch):
    calls = []
    monkeypatch.setattr(visible_native, 'frame', lambda o: b'v2-frame')

    def write_frame(row, payload):
        calls.append(payload)
        row['visible_schema'] = V2_SCHEMA
    monkeypatch.setattr(VP, 'write_frame', write_frame)   # (where write_visible takes it; structured_input's is it)
    monkeypatch.delenv('ISAAC_NO_VISIBLE', raising=False)
    monkeypatch.setattr(VP, 'VISIBLE_V3_MODE', 0)
    return calls


def row():
    r = np.zeros(1, ROW)
    r['visible_schema'] = 7
    return r


def test_v3_on_writes_the_v3_frame(v2_stub, monkeypatch):
    monkeypatch.setattr(VP, 'VISIBLE_V3_MODE', 2)   # (Codex round 9: a v3 record needs a v3 worker)
    r = row()
    assert tok_obs.write_visible(Obs(v3=GOLDEN, v2={'schema': 'v2'}), r) is True
    assert int(r['visible_schema'][0]) == V3.VERSION and int(r['visible_bytes'][0]) == len(GOLDEN)
    assert r['visible_frame'][0, :len(GOLDEN)].tobytes() == GOLDEN
    assert v2_stub == []


def test_v3_off_with_a_v2_block_writes_the_v2_frame(v2_stub):
    r = row()
    assert tok_obs.write_visible(Obs(v2={'schema': 'v2'}), r) is False
    assert v2_stub == [b'v2-frame'] and int(r['visible_schema'][0]) == V2_SCHEMA


def test_obs_visible_0_writes_no_frame(v2_stub, monkeypatch):
    monkeypatch.setenv('ISAAC_NO_VISIBLE', '1')
    with pytest.raises(ValueError, match='not configured for v3'):   # (mode 0: a v3 record is refused, Codex round 9)
        tok_obs.write_visible(Obs(v3=GOLDEN, v2={'schema': 'v2'}), row())
    for obs in (Obs(v2={'schema': 'v2'}), Obs()):
        r = row()
        assert tok_obs.write_visible(obs, r) is False
        assert int(r['visible_schema'][0]) == 0 and int(r['visible_bytes'][0]) == 0 and not r['visible_frame'].any()
    assert v2_stub == []
    # (and the v3 configuration never gets here: refused at argument parsing, below)
    assert tok_obs.obs_visible_conflict(0, '2')


def test_a_v3_worker_without_a_v3_frame_refuses(v2_stub, monkeypatch):
    monkeypatch.setattr(VP, 'VISIBLE_V3_MODE', 2)
    with pytest.raises(Exception, match='visible-v3'):
        tok_obs.write_visible(Obs(v2={'schema': 'v2'}), row())
    assert v2_stub == []


def test_obs_visible_conflict():
    for mode in ('1', '2', '3', ' 2 '):
        assert 'contradict' in tok_obs.obs_visible_conflict(0, mode)
    for ov, mode in ((1, '2'), (0, '0'), (0, ''), (1, '0'), (1, '')):
        assert tok_obs.obs_visible_conflict(ov, mode) is None


def test_train_tok_refuses_the_pair(monkeypatch, capsys, tmp_path):
    import train_tok
    monkeypatch.setenv('ISAAC_RL_VISIBLE_V3', '2')
    monkeypatch.delenv('ISAAC_NO_VISIBLE', raising=False)
    monkeypatch.setenv('ISAAC_SHARD_WORLD', 'stop-here')   # (were the check missing: int() fails right after it)
    with pytest.raises(SystemExit):
        train_tok.main(['--groups-file', 'x.json', '--out', str(tmp_path), '--obs-visible', '0'])
    assert 'contradict' in capsys.readouterr().err
    assert 'ISAAC_NO_VISIBLE' not in __import__('os').environ


def test_train_tok_frame_v3_refuses_item_tokens(monkeypatch, capsys, tmp_path):
    """(2026-11-18, the coordinator: the v3 arm takes no item tokens -- they read ROW inv / pitem; the frame's N / S
    tokens carry the items. The first smoke's T1 line had --item-tokens 1: refused at argument parsing, not by the
    model's constructor inside a retried attempt.)"""
    import train_tok
    from isaac_bridge import visible_producer
    monkeypatch.setattr(visible_producer, 'VISIBLE_V3_MODE', 2)
    monkeypatch.delenv('ISAAC_NO_VISIBLE', raising=False)
    monkeypatch.setenv('ISAAC_RL_TRIGGER_ONCE', '0')   # (recorded for the undo: absent again after the test)
    monkeypatch.delenv('ISAAC_RL_TRIGGER_ONCE')
    base = ['--groups-file', str(tmp_path / 'missing.json'), '--frame', 'v3', '--temporal', '1', '--temporal-rooms', '0',
            '--temporal-feedback', '0', '--global-token', '1', '--items']
    for i, extra in enumerate((['--item-tokens', '1'], ['--item-tokens', '1', '--item-shared', '1'])):
        with pytest.raises(SystemExit):
            train_tok.main(base + ['--out', str(tmp_path / f'o{i}')] + extra)
        assert '--frame v3 takes no item tokens' in capsys.readouterr().err
    with pytest.raises(FileNotFoundError):   # (without them the arguments pass: the missing groups file is next)
        train_tok.main(base + ['--out', str(tmp_path / 'ok')])
    assert __import__('os').environ.get('ISAAC_RL_TRIGGER_ONCE') == '1'   # (train_tok's v3 default, set unasked)
    assert '--frame v3 takes no item tokens' not in capsys.readouterr().err


def test_train_tok_frame_v3_refuses_learner_fast(monkeypatch, capsys, tmp_path):
    """(2026-11-22, the coordinator: a known defect) --frame v3 with --learner-fast 1 / 2 is refused at argument
    parsing: B18's raw minibatch merge cannot join v3-truncated records with the teacher ring's (torch.cat fails on the
    widths inside the first update); --learner-fast 0 passes."""
    import train_tok
    from isaac_bridge import visible_producer
    monkeypatch.setattr(visible_producer, 'VISIBLE_V3_MODE', 2)
    monkeypatch.delenv('ISAAC_NO_VISIBLE', raising=False)
    monkeypatch.setenv('ISAAC_RL_TRIGGER_ONCE', '0')
    monkeypatch.delenv('ISAAC_RL_TRIGGER_ONCE')
    base = ['--groups-file', str(tmp_path / 'missing.json'), '--frame', 'v3', '--temporal', '1', '--temporal-rooms', '0',
            '--temporal-feedback', '0', '--global-token', '1']
    for fast in ('1', '2'):
        with pytest.raises(SystemExit):
            train_tok.main(base + ['--out', str(tmp_path / f'f{fast}'), '--learner-fast', fast])
        assert f'--frame v3 does not support --learner-fast {fast}' in capsys.readouterr().err
    with pytest.raises(FileNotFoundError):   # (--learner-fast 0: the arguments pass, the missing groups file is next)
        train_tok.main(base + ['--out', str(tmp_path / 'f0'), '--learner-fast', '0'])
    assert 'does not support --learner-fast' not in capsys.readouterr().err


def test_eval_tok_refuses_the_pair(monkeypatch, capsys, tmp_path):
    sys.path.insert(0, str(HERE.parent))
    import eval_tok
    monkeypatch.setenv('ISAAC_RL_VISIBLE_V3', '2')
    monkeypatch.delenv('ISAAC_NO_VISIBLE', raising=False)
    # (--random with --ablate: were the check missing, eval_tok stops at its next check with another message)
    monkeypatch.setattr(sys, 'argv', ['eval_tok.py', '--groups-file', 'x.json', '--out', str(tmp_path), '--random',
                                      '--ablate', 'items', '--obs-visible', '0'])
    with pytest.raises(SystemExit):
        eval_tok.main()
    assert 'contradict' in capsys.readouterr().err


def test_eval_tok_frame_and_workers_agree(monkeypatch, capsys, tmp_path):
    """(seq-v1 step 4) eval_tok: a v3 checkpoint needs ISAAC_RL_VISIBLE_V3 (its workers send v3 frames); a ROW one is
    refused with it set; ROW-encoder options are refused for a v3 checkpoint. Each refusal at argument checking, before
    any worker starts."""
    import torch
    sys.path.insert(0, str(HERE.parent))
    import eval_tok
    from isaac_bridge import visible_producer
    monkeypatch.delenv('ISAAC_NO_VISIBLE', raising=False)
    for frame in ('v3', None):
        ck = tmp_path / f'{frame}.pt'
        torch.save(dict(config=dict(frame=frame, items=True), model={}), ck)
    cases = ((0, 'v3.pt', [], 'needs ISAAC_RL_VISIBLE_V3'), (2, 'v3.pt', ['--ablate', 'items'], 'ROW-encoder options'),
             (2, 'None.pt', [], 'reads ROW records'))
    for i, (mode, name, extra, msg) in enumerate(cases):
        monkeypatch.setattr(visible_producer, 'VISIBLE_V3_MODE', mode)
        monkeypatch.setenv('ISAAC_RL_VISIBLE_V3', str(mode))
        monkeypatch.setattr(sys, 'argv', ['eval_tok.py', '--groups-file', 'x.json', '--out', str(tmp_path / f'o{i}'),
                                          '--checkpoint', str(tmp_path / name)] + extra)
        with pytest.raises(SystemExit):
            eval_tok.main()
        assert msg in capsys.readouterr().err, (mode, name)


def test_v3_init_resume_and_eval_keep_revision_and_trigger_mode(monkeypatch, capsys, tmp_path):
    """(Codex round 12) train_tok --frame v3: --init only from a v3 model of this input revision (a revision-1 v3 model,
    a ROW model: refused before any weight is read); --resume refused in another trigger mode than the checkpoint's;
    eval_tok refused in another trigger mode than the checkpoint's."""
    import torch
    import train_tok
    sys.path.insert(0, str(HERE.parent))
    import eval_tok
    from isaac_bridge import visible_producer
    from isaac_bridge.v3_input import INPUT_REV
    monkeypatch.setattr(visible_producer, 'VISIBLE_V3_MODE', 2)
    monkeypatch.setenv('ISAAC_RL_VISIBLE_V3', '2')
    monkeypatch.delenv('ISAAC_NO_VISIBLE', raising=False)
    monkeypatch.setenv('ISAAC_RL_TRIGGER_ONCE', '1')
    old, new0 = tmp_path / 'rev1.pt', tmp_path / 'rev2-t0.pt'
    torch.save(dict(config=dict(frame='v3', items=True), model={}), old)
    torch.save(dict(config=dict(frame='v3', items=True, v3_inputs=INPUT_REV, trigger_once=0), model={}), new0)
    base = ['--groups-file', str(tmp_path / 'missing.json'), '--frame', 'v3', '--temporal', '1', '--temporal-rooms', '0',
            '--temporal-feedback', '0', '--global-token', '1', '--items']
    with pytest.raises(SystemExit):
        train_tok.main(base + ['--out', str(tmp_path / 'i'), '--init', str(old)])
    assert 'input revision' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        train_tok.main(base + ['--out', str(tmp_path / 'r'), '--resume', str(new0)])
    assert 'ISAAC_RL_TRIGGER_ONCE=0' in capsys.readouterr().err
    monkeypatch.setattr(sys, 'argv', ['eval_tok.py', '--groups-file', 'x.json', '--out', str(tmp_path / 'e'),
                                      '--checkpoint', str(new0)])
    with pytest.raises(SystemExit):
        eval_tok.main()
    assert 'trained with 0' in capsys.readouterr().err
