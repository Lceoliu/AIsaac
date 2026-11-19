"""LeanDecoder.decode on synthetic lean payloads (abp_bridge.lua pack_lean's layout): the real decode path the workers
and the frame probe take, with and without the visible block (flag 32). 2026-11-12: decode set o.visible_at, a field
LeanObs.__slots__ lacked -- every decode raised AttributeError (acceptg-c-78ad1e8: all probes and workers failed) while
no unit test decoded a payload."""
import ast
import inspect
import json
import struct
import textwrap
from pathlib import Path

import numpy as np
import pytest

from isaac_bridge import abplus_lean as al


def payload(visible=None, n_players=1, entities=2):
    flags = 32 if visible is not None else 0
    b = bytearray(al._HEADER.pack(al.MAGIC, 1234, 1200, flags | 1, n_players, 0, 0))
    b += al._ROOM.pack(*([0] * 8), 1.0, 2.0, 3.0, 4.0)
    b += al._TOTALS.pack(5.0, 6.0, 7.0, 8.0)
    pl = np.zeros(n_players, al.PLAYER)
    pl['hearts'] = 6
    b += pl.tobytes()
    b += struct.pack('<H', entities)
    en = np.zeros(entities, al.ENTITY)
    en['type'] = 10
    b += en.tobytes()
    if visible is not None:
        text = json.dumps(visible).encode()
        b += struct.pack('<I', len(text)) + text
    return bytes(b)


def test_decode_without_visible_block():
    o = al.LeanDecoder().decode(payload())
    assert o.logic_frames == 1234 and o.game_frame == 1200 and o.clear and not o.paused
    assert o.visible is None and o.visible_at is None
    assert len(o.players) == 1 and o.players['hearts'][0] == 6 and len(o.entities) == 2


def test_decode_with_visible_block():
    vis = {'schema': 'isaac-visible-prlmens-g-v2', 'bodies': [[1, 2]], 'inventory': [], 'capabilities': 1}
    p = payload(vis)
    o = al.LeanDecoder().decode(p)
    assert o.visible == vis
    assert struct.unpack_from('<I', p, o.visible_at - 4)[0] == len(p) - o.visible_at
    assert json.loads(p[o.visible_at:]) == vis


def test_every_field_decode_sets_is_a_slot():
    """(static) every `o.<name> = ...` in LeanDecoder.decode names one of LeanObs.__slots__."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(al.LeanDecoder.decode)))
    names = set()
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else []
        for t in targets:
            for e in (t.elts if isinstance(t, ast.Tuple) else [t]):
                if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == 'o':
                    names.add(e.attr)
    assert names and names <= set(al.LeanObs.__slots__), sorted(names - set(al.LeanObs.__slots__))


# ------------------------------------------------------------------ visible-v3 blocks (2026-11-18, Codex round 2)
# The header and the block prefixes written with literal formats (pack_lean's: '<IIIBBBB' magic, logic_frames,
# game_frame, flags, players, doors, lasers; a block = '<I' size + bytes, the raw block (flag 32) before the frame (64)),
# not with the decoder's own structs.
def v3_payload(raw=None, frame=None, tail=b''):
    flags = (32 if raw is not None else 0) | (64 if frame is not None else 0)
    b = bytearray(struct.pack('<IIIBBBB', 0x33504241, 77, 70, flags, 1, 0, 0))
    b += bytes(48) + bytes(32)                       # room (12 x 4), totals (4 x 8): zeros
    b += bytes(al.PLAYER.itemsize) + struct.pack('<H', 0)
    if raw is not None:
        text = json.dumps(raw).encode()
        b += struct.pack('<I', len(text)) + text
    if frame is not None:
        b += struct.pack('<I', len(frame)) + frame
    return bytes(b) + tail


def _frame():
    from isaac_bridge import visible_producer as VP
    raw = json.loads((Path(__file__).parent / 'fixtures' / 'visible_v3' / 'forgotten.raw.json').read_text())['raw']
    return raw, VP.observation_frame_v3(raw)


def test_decode_the_native_frame_alone_and_with_the_raw_block():
    raw, frame = _frame()
    o = al.LeanDecoder().decode(v3_payload(frame=frame))
    assert o.visible is None and o.visible_v3_frame == frame
    o = al.LeanDecoder().decode(v3_payload(raw=raw, frame=frame))   # the check mode (3): both blocks
    assert o.visible == raw and o.visible_v3_frame == frame
    o = al.LeanDecoder().decode(v3_payload(raw=raw))                  # a handed-back step: the raw block only
    assert o.visible == raw and o.visible_v3_frame is None


def test_malformed_blocks_are_errors():
    raw, frame = _frame()
    good = v3_payload(frame=frame)
    for bad, why in ((good[:-1], 'runs past'), (good + bytes(1), 'trailing'),
                     (v3_payload(raw=raw)[:-3], 'runs past'), (v3_payload(raw=raw, frame=frame)[:-10], 'runs past')):
        with pytest.raises(ValueError, match=why):
            al.LeanDecoder().decode(bad)


def test_a_malformed_frame_header_is_refused_when_written():
    """The decoder hands the frame bytes on as they came; the worker's write_v3 checks the header (magic, version word,
    size, exact length) before the ROW."""
    from types import SimpleNamespace
    from isaac_bridge import schema_v3 as V3, visible_producer as VP
    from isaac_bridge.tok_obs import ROW
    _, frame = _frame()
    f = np.frombuffer(frame, '<f4').copy()
    f[V3.H_VERSION] = 3   # (a batch 1-4 producer)
    o = al.LeanDecoder().decode(v3_payload(frame=f.tobytes()))
    with pytest.raises(VP.VisibleV3Fatal, match='frame version'):
        VP.write_v3(np.zeros(1, ROW), SimpleNamespace(visible=None, visible_v3_frame=o.visible_v3_frame))
