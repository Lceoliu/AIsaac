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


def test_first_held_only():
    """(2026-11-13) an acquisition is progress only the first time the episode holds that id: active items swapped on a
    pedestal (A -> B -> A -> B), trinkets / cards swapped back, a pill of a color held before; the first block's ids
    count as held. EV_ITEM / EV_PILL unchanged (a swap still raises EV_ITEM)."""
    st = TO.EpisodeState(100, floor=True, stall=10, run=True, items=True, stall_pickups=True, stall_items=True)
    A, B, P = 105, 486, 1   # two active items, a passive one held from the start
    seq = [   # (obs, inv_gain expected, events expected)
        (_o(1, {P: 1, A: 1}, tr0=5, card=9), False, 0),              # (the first block: all of it held)
        (_o(2, {P: 1, B: 1}, tr0=5, card=9), True, TO.EV_ITEM),      # B for the first time
        (_o(3, {P: 1, A: 1}, tr0=5, card=9), False, TO.EV_ITEM),     # A back: held before
        (_o(4, {P: 1, B: 1}, tr0=5, card=9), False, TO.EV_ITEM),     # B back
        (_o(5, {P: 1, A: 1}, tr0=5, card=9), False, TO.EV_ITEM),
        (_o(6, {P: 1, A: 1, 7: 1}, tr0=5, card=9), True, TO.EV_ITEM),   # a new collectible still counts
        (_o(7, {P: 1, A: 1, 7: 1}, tr0=6, card=9), True, 0),         # trinket 6 first time
        (_o(8, {P: 1, A: 1, 7: 1}, tr0=5, card=9), False, 0),        # trinket 5 back (held in the first block)
        (_o(9, {P: 1, A: 1, 7: 1}, tr0=6, card=9), False, 0),        # 6 back
        (_o(10, {P: 1, A: 1, 7: 1}, tr0=6, card=12), True, 0),       # card 12 first time
        (_o(11, {P: 1, A: 1, 7: 1}, tr0=6, card=9), False, 0),       # card 9 back
        (_o(12, {P: 1, A: 1, 7: 1}, tr0=6, card=12), False, 0),
        (_o(13, {P: 1, A: 1, 7: 1}, tr0=6, card=12, pill=3), True, 0),             # pill color 3 first time
        (_o(14, {P: 1, A: 1, 7: 1}, tr0=6, card=12, pill=0), False, TO.EV_PILL),   # (used)
        (_o(15, {P: 1, A: 1, 7: 1}, tr0=6, card=12, pill=3), False, 0),            # color 3 again
        (_o(16, {P: 1, A: 1, 7: 1}, tr0=6, card=12, pill=4), True, 0),             # color 4 first time
        (_o(17, {P: 1, A: 1, 7: 1, 8: 1}, tr0=6, tr1=11, card=12, pill=4), True, TO.EV_ITEM),   # both at once
    ]
    for k, (o, gain, ev) in enumerate(seq):
        got = TO._inventory(o, st)
        assert st.inv_gain == gain and got == ev, (k, st.inv_gain, got)


def test_swaps_do_not_hold_off_the_stall():
    """encode_row end to end (test_obs_v2_smoke's world, no events, no monster hurt, resources constant): inventory
    blocks A, B, A, B, ... every 2 records; the progress time moves at the first B only and the episode stalls STALL
    records later; with stall_items off it stalls at STALL."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_stall_progress_smoke as S
    from isaac_bridge import abplus_lean as AL
    STALL, N = 5, 16
    seq0 = S.SEQ
    S.SEQ = [seq0[0]] * N   # (resources constant: no pickup gains; restored below)

    def run(stall_items):
        dec = AL.LeanDecoder()
        st = TO.EpisodeState(10 ** 6, floor=True, stall=STALL, items=True, stall_pickups=True, stall_items=stall_items)
        rows = np.zeros(N, TO.ROW)
        prog, done = [], []
        for t in range(N):
            o = dec.decode(S.payload(t, S.flags_of(t)))
            if t % 2 == 0:   # an inventory block: 105 at t = 0, 4, 8, ...; 486 at t = 2, 6, 10, ...
                o.inv, o.inv_version = np.array([(105 if t % 4 == 0 else 486, 1)], AL.INV_ITEM), 1 + t
                o.inv_head = (1, 0, 6, 0, 0, 0, 0, 0, 0)
            done.append(int(TO.encode_row(o, st, rows[t:t + 1], t)))
            prog.append(st.progress_t)
        return prog, done

    try:
        prog, done = run(True)
        prog_off, done_off = run(False)
    finally:
        S.SEQ = seq0
    assert prog == [0, 0] + [2] * (N - 2), prog
    assert done.index(3) == 2 + STALL, done
    prog, done = prog_off, done_off
    assert set(prog) == {0} and done.index(3) == STALL, (prog, done)


def test_encode_row_and_workers_use_it():
    src = inspect.getsource(TO.encode_row)
    assert 'if st.stall_items and st.inv_gain:' in src
    fl = inspect.getsource(tok_floor)
    assert fl.count('stall_items=stall_it') == 3 and "== 'items'" in fl
