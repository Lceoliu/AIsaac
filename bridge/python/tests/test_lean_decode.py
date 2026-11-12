"""LeanDecoder.decode on synthetic lean payloads (abp_bridge.lua pack_lean's layout): the real decode path the workers
and the frame probe take, with and without the visible block (flag 32). 2026-11-12: decode set o.visible_at, a field
LeanObs.__slots__ lacked -- every decode raised AttributeError (acceptg-c-78ad1e8: all probes and workers failed) while
no unit test decoded a payload."""
import ast
import inspect
import json
import struct
import textwrap

import numpy as np

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
