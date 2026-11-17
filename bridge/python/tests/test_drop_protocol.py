"""visible-v3 F2, the drop action on the wire (2026-11-17, package C; DESIGN 3.2): action_code with + 360 drop (720 codes,
equal to the frozen schema_v3.action_code), the step line's optional seventh number and the JSON step's drop field.
drop 0 sends exactly what was sent before (5 or 6 numbers, no JSON field): AB+ trajectories with drop 0 unchanged (G6).
The bridge (abp_bridge.lua: play_next, the step-line parse, begin_step_line) and abp_turbo.c (step_read: 7 numbers) are
checked here by their source text; their behaviour on the game is the drop probe's (abplus_probe_drop.py)."""
import itertools
import re
from pathlib import Path

import pytest

from isaac_bridge import schema_v3
from isaac_bridge.abplus import AbplusTrainingEnv, action_code
from isaac_bridge.env import BridgeError

BRIDGE = Path(__file__).resolve().parents[2]


def test_action_code_equals_the_frozen_one():
    codes = set()
    for a in itertools.product(*(range(n) for n in schema_v3.HEAD_SIZES)):
        c = action_code(*a)
        assert c == schema_v3.action_code(*a)
        assert tuple(schema_v3.action_of_code(c)) == a
        codes.add(c)
    assert codes == set(range(schema_v3.N_ACTION_CODES))
    assert action_code(3, 2, 1, 0, 1) == 3 + 18 + 45 + 180   # (no drop: the code as before)


class _Sock:
    def __init__(self):
        self.sent = []

    def sendall(self, b):
        self.sent.append(b)


def _bridge(hello):
    b = AbplusTrainingEnv.__new__(AbplusTrainingEnv)
    b.hello, b.action_repeat, b._sock = dict(hello), 4, _Sock()
    b.msgs = []
    b._send = b.msgs.append
    return b


@pytest.mark.parametrize('line', [True, False])
def test_step_line_drop(monkeypatch, line):
    import isaac_bridge.abplus as ab
    monkeypatch.setattr(ab, 'STEP_LINE', line)
    b = _bridge(dict(step_line=True, items=True, drop=True))
    b.step_line(1, 2, 0, 1, 4)
    b.step_line(1, 2, 0, 1, 4, pill=1)
    b.step_line(1, 2, 0, 1, 4, pill=0, drop=1)
    b.step_line(0, 0, 0, 0, 4, pill=1, drop=1)
    if line:
        assert b._sock.sent == [b'S 4 1 2 0 1\n', b'S 4 1 2 0 1 1\n', b'S 4 1 2 0 1 0 1\n', b'S 4 0 0 0 0 1 1\n']
        assert b.msgs == []
    else:
        assert b._sock.sent == []
        assert [m.get('drop') for m in b.msgs] == [None, None, 1, 1]
        assert [m.get('pill') for m in b.msgs] == [None, 1, None, 1]


def test_drop_needs_the_bridges_word():
    b = _bridge(dict(step_line=True, items=True))
    b.step_line(1, 0, 0, 0, 4)          # drop 0: as before
    assert b._sock.sent == [b'S 4 1 0 0 0\n']
    with pytest.raises(BridgeError):
        b.step_line(1, 0, 0, 0, 4, drop=1)


def test_bridge_and_turbo_sources_take_the_seventh_number():
    lua = (BRIDGE / 'abplus' / 'abp_bridge.lua').read_text(encoding='utf-8')
    assert 'drop = (code // 360) % 2' in lua
    assert lua.count('"^S (%d+) (%d+) (%d+) (%d+) (%d+) (%d+) (%d+)$"') == 1
    assert 'begin_step_line(a, b, cc, d, e, f, g)' in lua
    assert len(re.findall(r'lean_items = state\.lean_items, drop = true \}', lua)) == 2   # (both hellos)
    c = (BRIDGE / 'engine' / 'abp_turbo.c').read_text(encoding='utf-8')
    assert 'static int step_read(int fd, long long v[7], const char **why)' in c
    assert 'for (int i = 0; i < 7; i++) LPUSHINT(L, v[i]);' in c
