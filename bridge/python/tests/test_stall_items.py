"""--stall-progress items (2026-11-12): besides pickups' resources, an acquired collectible, trinket, pill or card
counts as progress for the floor's stall rule (tok_obs._inventory's inv_gain; encode_row resets the progress time on
it). Inventory blocks always take encode_row (abp_row_encode hands a payload with one back, -2), so the native fast path
needs nothing new. Here: _inventory's inv_gain on a sequence of inventory blocks, and the plumbing."""
import inspect
import types

import numpy as np

from isaac_bridge import tok_floor, tok_obs as TO

INV = np.dtype([('id', np.int32), ('count', np.int32)])


def _o(ver, items, tr0=0, tr1=0, pill=0, card=0):
    inv = np.array(list(items.items()), INV) if items else np.zeros(0, INV)
    return types.SimpleNamespace(inv_version=ver, inv_head=(len(inv), 0, 6, tr0, tr1, pill, 0, card, 0), inv=inv)


def test_inv_gain():
    st = TO.EpisodeState(100, floor=True, stall=10, run=True, items=True, stall_pickups=True, stall_items=True)
    seq = [   # (obs, inv_gain expected, events expected)
        (_o(1, {1: 1}), False, 0),                                   # (the first block: nothing gained)
        (_o(1, {1: 1}), False, 0),                                   # (the same version again)
        (_o(2, {1: 1}, tr0=5), True, 0),                             # a trinket picked up
        (_o(3, {1: 1}, tr0=5, tr1=0), False, 0),                     # (nothing new)
        (_o(4, {1: 1, 7: 1}, tr0=5), True, TO.EV_ITEM),              # a collectible
        (_o(5, {1: 1, 7: 1}, tr0=5, pill=3), True, 0),               # a pill
        (_o(6, {1: 1, 7: 1}, tr0=5, pill=0), False, TO.EV_PILL),     # (the pill used: no gain)
        (_o(7, {1: 1, 7: 1}, tr0=5, card=9), True, 0),               # a card
        (_o(8, {1: 1, 7: 1}, tr0=5, card=12), True, 0),              # another card in its place
        (_o(9, {1: 1, 7: 1}, tr0=6), True, TO.EV_PILL),              # a trinket swapped (the card used)
        (_o(10, {1: 1}, tr0=6), False, 0),                           # (a collectible lost)
    ]
    for k, (o, gain, ev) in enumerate(seq):
        got = TO._inventory(o, st)
        assert st.inv_gain == gain and got == ev, (k, st.inv_gain, got)


def test_encode_row_and_workers_use_it():
    src = inspect.getsource(TO.encode_row)
    assert 'if st.stall_items and st.inv_gain:' in src
    fl = inspect.getsource(tok_floor)
    assert fl.count('stall_items=stall_it') == 3 and "== 'items'" in fl
