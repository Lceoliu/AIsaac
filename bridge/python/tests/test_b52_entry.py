"""B52 entry gate: train_tok's own construction path (not a test-made driver) up to the first collect and update.

The unit gates (test_b52_cpp_actor.py) make and start the driver themselves, so they did not catch train_tok building
the driver and never starting it (the 4090 smoke of 2026-11-10 hung: workers waited for replies). This gate runs the
production entry twice, one update each, with the short-profile configuration (16 workers, rollout 32768, micro 8192,
CF, teacher, block memory v1, lanes) and real game workers:

- driver on: "B52 cpp actor driver: on", then "B52 driver first collect: thread started, N replies" with N > 0 (the
  driver thread ran and answered workers), an update line u1, no traceback;
- ISAAC_BLOCK_CPP_ACTOR=0: "B52 cpp actor driver: off (ISAAC_BLOCK_CPP_ACTOR=0)", no driver line, u1 (the same path
  falls back to the Python loop).

Needs the game (ABP_HOME, ABP_BRIDGE_LUA, xvfb-run), a CUDA GPU and the GPU lock; skipped unless ISAAC_B52_ENTRY=1
(bridge/ops/run_b52g.sh sets it). Each run takes the Spatial warm-up (cached by inductor after the first run).
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PY = HERE.parent
pytestmark = pytest.mark.skipif(os.environ.get('ISAAC_B52_ENTRY') != '1' or not os.environ.get('ABP_HOME')
                                or shutil.which('xvfb-run') is None,
                                reason='B52 entry gate: needs ISAAC_B52_ENTRY=1, the game (ABP_HOME) and xvfb-run')

# the short-profile command (as the 4090's xnew2.sh), one update
ARGS = ('--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items '
        '--epochs 3 --teacher --teacher-queue 12 --archive-size 12 --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 '
        '--teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.001 '
        '--item-tokens 1 --global-token 1 --item-shared 1 --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal '
        '--pickup-force 1:0 --reward heal=0.5,resource=0.25 --item-slots 8 '
        '--teach-batch 512 --teach-reuse 8 --teach-weight 8 --teach-buffer 60000 --teach-coef 1 --danger-coef 1 --teach-merge 1 '
        '--teacher-v2-batch 256 --teacher-v2-reuse 4 --teacher-v2-weight 1 --teacher-v2-coef 1 --teacher-v2-vf 0.25 --teacher-v2-buffer 30000 '
        '--cf 1 --cf-points 6 --cf-branches 4 --cf-seconds 180 --cf-target-seconds 20 --cf-margin 0.3 --cf-cap 2 --cf-distill-steps 8 --cf-random 1 --cf-advantage 0 '
        '--aux-predict 1 --aux-horizons 1,8 --aux-coef 0.1 --priv-critic 1 '
        '--memory off --temporal 0 --block-memory 1 --block-size v1 --block-slots 1 --width 192 --layers 4 --heads 4 '
        '--actor-bf16 1 --graph-entities 128 '
        '--workers 16 --rollout 32768 --minibatch 8192 --micro 8192 --max-updates 1 --checkpoint-every 1000 '
        '--overlap --decoupled 1 --learner-fast 0 --actor-processes 1 --learner-process 0 --amp bf16 '
        '--port 39260 --name b52entry --seed 1000').split()
LIMIT = int(os.environ.get('ISAAC_B52_ENTRY_LIMIT', '1500'))


def run_entry(tmp_path, cpp, lanes=None):
    out = tmp_path / ('cpp' if cpp else 'py') / ('lanes%s' % lanes if lanes is not None else '')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               LIBGL_ALWAYS_SOFTWARE='1', SDL_AUDIODRIVER='dummy', ALSOFT_DRIVERS='null',
               ISAAC_BLOCK_CPP_ACTOR='1' if cpp else '0')
    if lanes is not None:   # (B58d: the native lane service is the default; =0 keeps the B54 Python lane service covered)
        env['ISAAC_BLOCK_CPP_LANES'] = str(lanes)
    cmd = ['xvfb-run', '-a', '-s', '-screen 0 640x480x24 -nolisten tcp -noreset', 'timeout', '--signal=INT', '--kill-after=60s',
           str(LIMIT), sys.executable, '-u', 'train_tok.py'] + ARGS + ['--out', str(out)]
    p = subprocess.run(cmd, cwd=str(PY), env=env, capture_output=True, text=True, encoding='utf-8', errors='replace',
                       timeout=LIMIT + 180)
    text = p.stdout + '\n' + p.stderr
    (tmp_path / f"{'cpp' if cpp else 'py'}.log").write_text(text, encoding='utf-8')
    return p.returncode, text


MAIN_TB = re.compile(r'Traceback \(most recent call last\):\n\s+File "[^"]*train_tok\.py"')


def updates(text):
    return re.findall(r'^u(\d+) ', text, re.M)


def test_train_tok_entry_driver_on_starts_and_answers(tmp_path):
    rc, text = run_entry(tmp_path, cpp=True)
    tail = text[-3000:]
    assert 'B52 cpp actor driver: on' in text, tail
    m = re.search(r'B52 driver first collect: thread started, (\d+) replies, (\d+) rows, (\d+) steps', text)
    assert m is not None, 'the driver never finished a collect\n' + tail
    assert int(m.group(1)) > 0 and int(m.group(2)) > 0 and int(m.group(3)) > 0, m.group(0)
    assert 'B58d native lanes: on' in text, tail   # (the default since the d1 gate)
    assert updates(text), 'no update line\n' + tail
    assert MAIN_TB.search(text) is None, 'train_tok raised\n' + tail   # (workers' shutdown tracebacks are not the gate's)
    assert rc == 0, f'rc {rc}\n' + tail


def test_train_tok_entry_driver_off_uses_python_path(tmp_path):
    rc, text = run_entry(tmp_path, cpp=False)
    tail = text[-3000:]
    assert 'B52 cpp actor driver: off (ISAAC_BLOCK_CPP_ACTOR=0)' in text, tail
    assert 'B52 driver first collect' not in text, tail
    assert updates(text), 'no update line\n' + tail
    assert MAIN_TB.search(text) is None, 'train_tok raised\n' + tail   # (workers' shutdown tracebacks are not the gate's)
    assert rc == 0, f'rc {rc}\n' + tail


def test_train_tok_entry_driver_on_python_lane_service(tmp_path):
    """(B58d) ISAAC_BLOCK_CPP_LANES=0: the driver with the B54 Python lane service (the fallback stays covered)."""
    rc, text = run_entry(tmp_path, cpp=True, lanes=0)
    tail = text[-3000:]
    assert 'B52 cpp actor driver: on' in text and 'B58d native lanes: off (ISAAC_BLOCK_CPP_LANES=0' in text, tail
    assert re.search(r'B52 driver first collect: thread started, [1-9]', text) is not None, tail
    assert updates(text), 'no update line' + chr(10) + tail
    assert MAIN_TB.search(text) is None, 'train_tok raised' + chr(10) + tail
    assert rc == 0, f'rc {rc}' + chr(10) + tail
