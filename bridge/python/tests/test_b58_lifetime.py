"""B58c: the native store's and arenas' memory outlives its owner while a view of it is alive (abp_core.h, Lifetimes).

Close order: the page store (and the driver holding it) go first while a learner snapshot still holds page tensors /
metadata numpy views; then new stores of the same size are made (the freed memory would be handed out again, zeroed)
and the old views are read: they must still show what was written. Same for an arena buffer view across the arena's
growth, its reset and a slot's re-set. With plain from_blob views (no owner in the deleter) the old views read the new
stores' zeroed memory (or freed memory), so this fails.
Skipped without the extension (no CUDA torch, no g++, ISAAC_BLOCK_CPP_ACTOR=0)."""
import gc

import numpy as np
import pytest
import torch

from test_b52_cpp_actor import EXT

pytestmark = pytest.mark.skipif(EXT is None, reason='cpp actor extension unavailable')

P, KV, Z, B, BS, NH = 64, 96, 8, 16, 4, 3


def make_store():
    s = EXT.PageStore(P, KV, False, Z, B, BS, NH, 64, 16)
    s.grow_ahead(4*P, P)
    return s


def test_page_views_outlive_the_store():
    s = make_store()
    kv, z, b = s.kv_page(1), s.z_page(0), s.b_page(0)
    meta = s.d_meta.numpy()   # (as PagedStore keeps them: numpy views of the tensors)
    kv.fill_(7.0); z.fill_(3.0); b.fill_(5.0)
    meta[:10] = np.arange(20).reshape(10, 2)
    del s
    gc.collect()
    others = [make_store() for _ in range(6)]   # (the freed memory, if any, handed out again and zeroed)
    for o in others:
        o.kv_page(1).fill_(-1.0); o.z_page(0).fill_(-1.0); o.b_page(0).fill_(-1.0)
    assert bool((kv == 7.0).all()) and bool((z == 3.0).all()) and bool((b == 5.0).all())
    np.testing.assert_array_equal(meta[:10], np.arange(20).reshape(10, 2))
    del kv, z, b, meta, others
    gc.collect()


def test_arena_views_outlive_growth_and_reset():
    from test_b58_native_consumer import make_pair, W
    from test_block_memory import make_policy
    from test_b52_cpp_actor import fake_sampler, make_driver_for
    from isaac_bridge.block_paged import SlotBlockActor
    from isaac_bridge.tok_priv import PRIV as PRIV_DTYPE
    from isaac_bridge import block_driver as bdrv
    from test_b58_native_consumer import make_book
    m = make_policy('cuda').eval()
    nh = len(m.heads)
    actor = SlotBlockActor(m, W, graph_entities=128, bf16=True, lane_slots=0)
    d = make_driver_for(actor, fake_sampler(W, nh))
    rolls = make_pair(nh, PRIV_DTYPE)
    bdrv.attach_consumer(d, make_book(np.zeros((W, 2))), rolls, 0.99)
    slot = rolls[0].native_slot
    blob = np.arange(4096, dtype=np.uint8)
    d.arena_append_from(slot, torch.from_numpy(blob), torch.tensor([0]), torch.tensor([4096]))
    old = d.arena_buf(slot)                     # the buffer before growth
    first = old[:4096].clone()
    cap0 = int(old.numel())
    n = 0
    while d.arena_grown(slot) == 0:             # grow it (appends until the arena reallocates)
        d.arena_append_from(slot, torch.from_numpy(blob), torch.tensor([0]), torch.tensor([4096])); n += 1
        assert n < 100000
    assert torch.equal(old[:4096], first) and int(old.numel()) == cap0
    d.arena_reset(slot)
    del rolls, d, actor
    gc.collect()
    junk = [torch.zeros(cap0, dtype=torch.uint8) for _ in range(8)]   # (freed memory handed out again)
    for j in junk:
        j.fill_(255)
    assert torch.equal(old[:4096], first)
