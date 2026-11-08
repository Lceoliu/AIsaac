"""A1 equivalence gates: production slot/paged path against the object reference.

Fixed weights and fixed random numbers. The object path (BlockActor / Snapshot /
forward_chunks over EpisodeMemory) is the reference; SlotBlockActor / PagedSnapshot /
forward_chunks_paged is the production path.
"""
import copy
import gc

import numpy as np
import pytest
import torch

from test_block_memory import make_policy
from test_visible_input import synth_rows, synth_pending
from isaac_bridge.block_training import (BlockActor, Chunk, forward_chunks, forward_bucketed, ppo_loss, cf_loss,
                                         injury_loss, pack_snapshots, unpack_snapshots)
from isaac_bridge.block_paged import SlotBlockActor, PagedSnapshot, PagedPayload
from train_tok import Rollout, store, gae

DEV = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
TOL = dict(rtol=2e-5, atol=2e-6)


def pair(device='cpu', slots=4, seed=7, graph=0, bf16=False):
    torch.manual_seed(seed)
    m = make_policy(device).eval()
    ref = copy.deepcopy(m)
    return m, ref, SlotBlockActor(m, slots, graph_entities=graph, bf16=bf16), BlockActor(ref, slots, graph_entities=graph, bf16=bf16)


def frame(rng, n, t, episode, first):
    rows = synth_rows(rng, n)
    rows['first'] = first
    rows['episode'] = episode
    rows['t'] = t
    rows['done'] = 0
    return rows


def compare_state(new, old, slot, nslot=None):
    """Arena row vs object EpisodeMemory: step, short K/V, bank, open tail."""
    nslot = slot if nslot is None else nslot
    mem = old.memory[slot]
    st = new.state
    g = st.gather(st._ids([nslot])[1])
    assert int(st.step_host[nslot]) == mem.step
    n = len(mem.kv_steps)
    torch.testing.assert_close(g['kv_steps'][0, :n].cpu(), mem.kv_steps.cpu(), rtol=0, atol=0)
    assert (g['kv_steps'][0, n:] == -1).all()
    for layer in range(len(mem.kv)):
        for j in (0, 1):
            torch.testing.assert_close(g['kv'][layer, j, 0, :, :n].float().cpu(), mem.kv[layer][j].float().cpu(), **TOL)
    for k, block in enumerate(mem.blocks):
        assert int(g['ends'][0, k]) == block.end_step
        torch.testing.assert_close(g['bank'][0, k].cpu(), block.value.float().cpu(), **TOL)
    assert (g['ends'][0, len(mem.blocks):] == -1).all()
    p = len(mem.partial)
    torch.testing.assert_close(g['partial'][0, :p].cpu(), mem.partial.float().cpu(), **TOL)


def compare_result(a, b):
    for x, y in zip(a, b):
        torch.testing.assert_close(x.float(), y.float(), **TOL)


@pytest.mark.parametrize('device', DEV)
def test_stepwise_actor_outputs_states_reset_room_change_and_fifo(device):
    m, ref, new, old = pair(device)
    rng = np.random.default_rng(3)
    episodes = np.array([1, 1, 1, 1])
    blocks_seen = 0
    for t in range(23):
        first = np.zeros(4, np.int64)
        if t == 0:
            first[:] = 1
        if t == 9:            # one episode reset mid-stream
            first[2] = 1; episodes[2] += 1
        rows = frame(rng, 4, t, episodes, first)
        if t == 13:           # room change inside an episode: memory continues
            rows['room_type'] = rows['room_type'] + 1
        pending = synth_pending(rng, 4, m.heads)
        keys = (np.full(4, t), np.arange(4))
        compare_result(new.step(rows, pending, range(4), ukeys=keys), old.step(rows, pending, range(4), ukeys=keys))
        np.testing.assert_allclose(new.host_output, old.host_output, rtol=2e-5, atol=2e-6)
        for s in range(4):
            compare_state(new, old, s)
            np.testing.assert_array_equal(new.pending[s], old.pending[s])
        blocks_seen = max(blocks_seen, len(old.memory[0].blocks))
    cfg = m.block_memory.config
    # FIFO wrapped: more blocks completed than the bank holds.
    assert old.memory[0].step // cfg.block_steps > cfg.memory_blocks and blocks_seen == cfg.memory_blocks
    # Snapshots are paged descriptors; contexts are pre-observation states.
    ctx = new.last_contexts[0]
    assert isinstance(ctx, PagedSnapshot) and ctx.step == old.last_contexts[0].memory.step


@pytest.mark.parametrize('device', DEV)
def test_fork_restore_siblings_and_main_isolation(device):
    m, ref, new, old = pair(device, slots=2, seed=11)
    lane_new = new.lane_actor(m, 3)
    lane_old = BlockActor(ref, 3)
    rng = np.random.default_rng(5)
    for t in range(7):
        rows = frame(rng, 2, t, np.array([1, 1]), int(t == 0))
        pending = synth_pending(rng, 2, m.heads)
        new.step(rows, pending, [0, 1]); old.step(rows, pending, [0, 1])
    snap_new, snap_old = new.snapshot(0).fork(), old.snapshot(0).fork()
    for lane in range(3):
        lane_new.restore(lane, snap_new); lane_old.restore(lane, snap_old)
        compare_state(lane_new, lane_old, lane)
    for t in range(9):
        # siblings get different observations; main keeps going
        rows = frame(rng, 3, 100+t, np.array([1, 1, 1]), 0)
        pending = synth_pending(rng, 3, m.heads)
        keys = (np.full(3, t), np.arange(3))
        compare_result(lane_new.step(rows, pending, [0, 1, 2], ukeys=keys),
                       lane_old.step(rows, pending, [0, 1, 2], ukeys=keys))
        rows = frame(rng, 2, 7+t, np.array([1, 1]), 0)
        pending = synth_pending(rng, 2, m.heads)
        keys = (np.full(2, 50+t), np.arange(2))
        compare_result(new.step(rows, pending, [0, 1], ukeys=keys), old.step(rows, pending, [0, 1], ukeys=keys))
        for lane in range(3):
            compare_state(lane_new, lane_old, lane)
        compare_state(new, old, 0)
    # A sibling replayed alone from the same snapshot gives the same answer: no cross-sibling leakage.
    solo = new.lane_actor(m, 1)
    solo.restore(0, snap_new)
    rows = frame(rng, 1, 0, np.array([1]), 0)
    pending = synth_pending(rng, 1, m.heads)
    lane_new.restore(1, snap_new)
    a = solo.step(rows, pending, [0], ukeys=(np.array([9]), np.array([0])))
    b = lane_new.step(rows, pending, [1], ukeys=(np.array([9]), np.array([0])))
    compare_result(a, b)


@pytest.mark.parametrize('device', DEV)
def test_merged_main_and_lane_batch_matches_separate_calls(device):
    m, ref, new, old = pair(device, slots=2, seed=13)
    lane_new = new.lane_actor(m, 2)
    lane_old = BlockActor(ref, 2)
    rng = np.random.default_rng(13)
    for t in range(5):
        rows = frame(rng, 2, t, np.array([1, 1]), int(t == 0))
        pending = synth_pending(rng, 2, m.heads)
        new.step(rows, pending, [0, 1]); old.step(rows, pending, [0, 1])
    lane_new.restore(0, new.snapshot(1)); lane_new.restore(1, new.snapshot(0))
    lane_old.restore(0, old.snapshot(1)); lane_old.restore(1, old.snapshot(0))
    for t in range(6):
        rows = frame(rng, 4, 10+t, np.array([1, 1, 1, 1]), 0)
        pending = synth_pending(rng, 4, m.heads)
        u = torch.rand((4, sum(m.heads)), device=device)
        # interleaved owner order exercises the permutation path
        order = [0, 2, 1, 3]
        slots = [0, 0, 1, 1]
        owners_n = [new, lane_new, new, lane_new]
        owners_o = [old, lane_old, old, lane_old]
        r, p = rows[order], pending[order]
        compare_result(new.step(r, p, slots, owners=owners_n, uniform=u[order]),
                       old.step(r, p, slots, owners=owners_o, uniform=u[order]))
        np.testing.assert_allclose(new.host_output, old.host_output, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(lane_new.host_output, lane_old.host_output, rtol=2e-5, atol=2e-6)
        for s in range(2):
            compare_state(new, old, s); compare_state(lane_new, lane_old, s)


def _collect_rollout(device, n=29, seed=21):
    m, ref, new, old = pair(device, slots=2, seed=seed)
    rng = np.random.default_rng(seed)
    rolls = [Rollout(2, 40, 5, value_g=True, block_steps=4) for _ in range(2)]
    episodes = np.array([1, 1])
    for t in range(n):
        first = np.array([int(t == 0), int(t == 0)])
        if t == 11:
            first[1] = 1; episodes[1] += 1
        rows = frame(rng, 2, t, episodes, first)
        pending = synth_pending(rng, 2, m.heads)
        keys = (np.full(2, t), np.arange(2))
        for actor, roll in ((new, rolls[0]), (old, rolls[1])):
            result = actor.step(rows, pending, [0, 1], ukeys=keys)
            action, lp, v, vg = [x.float().cpu().numpy() for x in result]
            store(roll, np.array([0, 1]), rows, pending, action.astype(np.int64), lp, v, 0, vg=vg,
                  block_context=actor.last_contexts)
            roll.reward[:, t] = .1
    return m, ref, new, old, rolls


def _losses(model, out, batch, rng_seed):
    g = torch.Generator().manual_seed(rng_seed)
    n = len(out[0])
    actions = torch.stack([torch.randint(0, k, (n,), generator=g) for k in model.heads], 1).to(out[0].device)
    adv = torch.randn(n, generator=g).to(out[0].device)
    ret = torch.randn(n, generator=g).to(out[0].device)
    blp = -torch.rand(n, generator=g).to(out[0].device)
    loss, _ = ppo_loss(model, out, batch, actions, blp, adv, ret, global_returns=ret)
    return loss


@pytest.mark.parametrize('device', DEV)
def test_learner_rollout_chunks_ppo_loss_and_all_gradients(device):
    m, ref, new, old, rolls = _collect_rollout(device)
    _, _, mask = gae(rolls[0], .99, .95)
    plans = [r.block.plan(r, mask) for r in rolls]
    assert [len(c.rows) for c in plans[0]['chunks']] == [len(c.rows) for c in plans[1]['chunks']]
    assert all(isinstance(c.snapshot, PagedSnapshot) for c in plans[0]['chunks'])
    m.train(); ref.train()
    outs = []
    for model, plan in ((m, plans[0]), (ref, plans[1])):
        out, ctx = forward_bucketed(model, plan['chunks'], generator=torch.Generator().manual_seed(5))
        rows = np.concatenate([c.rows for c in plan['chunks']])
        pend = np.concatenate([c.pending for c in plan['chunks']])
        from isaac_bridge.structured_input import from_rows
        batch = from_rows(rows, device, pend)
        loss = _losses(model, out, batch, 3)
        loss.backward()
        outs.append((loss.detach(), out, ctx['recompute_stats']))
    torch.testing.assert_close(outs[0][0], outs[1][0], rtol=1e-5, atol=1e-6)
    for x, y in zip(outs[0][1], outs[1][1]):
        torch.testing.assert_close(x.detach(), y.detach(), rtol=1e-5, atol=2e-6)
    assert outs[0][2]['selected_blocks'] == outs[1][2]['selected_blocks'] > 0
    # values the learner recomputes equal the actor's stored values
    torch.testing.assert_close(outs[0][1][1].detach().cpu(), torch.from_numpy(rolls[0].value.reshape(-1)[plans[0]['pos']]),
                               rtol=3e-5, atol=3e-6)
    pa, pb = dict(m.named_parameters()), dict(ref.named_parameters())
    touched = 0
    for name in pa:
        ga, gb = pa[name].grad, pb[name].grad
        assert (ga is None) == (gb is None), name
        if ga is not None:
            torch.testing.assert_close(ga, gb, rtol=1e-4, atol=5e-6, msg=name)
            touched += int(ga.abs().sum() > 0)
    assert pa['block_memory.writer.queries'].grad.abs().sum() > 0
    assert pa['structured_spatial.body.0.weight'].grad.abs().sum() > 0
    assert touched > 20


@pytest.mark.parametrize('device', DEV)
@pytest.mark.parametrize('kind', ['cf', 'injury'])
def test_learner_teacher_records_cf_injury_loss_and_gradients(device, kind):
    m, ref, new, old = pair(device, slots=3, seed=31)
    rng = np.random.default_rng(31)
    chunks = ([], [])
    for t in range(19):
        rows = frame(rng, 3, t, np.array([1, 1, 1]), int(t == 0))
        pending = synth_pending(rng, 3, m.heads)
        if t >= 6:   # one-row teacher records at assorted phases (incl. ones that complete a block)
            for i in range(3):
                chunks[0].append(Chunk(rows[i:i+1].copy(), pending[i:i+1].copy(), new.snapshot(i)))
                chunks[1].append(Chunk(rows[i:i+1].copy(), pending[i:i+1].copy(), old.snapshot(i)))
        new.step(rows, pending, range(3)); old.step(rows, pending, range(3))
    # IPC hand-off: in-process descriptors, never serialised
    packed = pack_snapshots([c.snapshot for c in chunks[0]])
    assert isinstance(packed, PagedPayload) and unpack_snapshots(packed, 'cpu')[0] is chunks[0][0].snapshot
    m.train(); ref.train()
    losses = []
    for model, cs in ((m, chunks[0]), (ref, chunks[1])):
        out, _ = forward_bucketed(model, cs, max_rows=8, generator=torch.Generator().manual_seed(17))
        n = len(cs)
        g = torch.Generator().manual_seed(2)
        if kind == 'cf':
            actions = torch.stack([torch.randint(0, k, (n,), generator=g) for k in model.heads], 1).to(device)
            rows = np.concatenate([c.rows for c in cs]); pend = np.concatenate([c.pending for c in cs])
            from isaac_bridge.structured_input import from_rows
            batch = from_rows(rows, device, pend)
            loss = cf_loss(model, out, batch, actions, torch.rand(n, generator=g).to(device),
                           torch.randn(n, generator=g).to(device), .4, 1., .25)
            loss = loss[0] if isinstance(loss, tuple) else loss
        else:
            meta = torch.randint(0, 2, (n, 16), generator=g).to(device)
            loss = injury_loss(out[0], out[2], meta, .4, 1., 1.)
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(losses[0], losses[1], rtol=1e-5, atol=1e-6)
    pa, pb = dict(m.named_parameters()), dict(ref.named_parameters())
    for name in pa:
        if pa[name].grad is not None or pb[name].grad is not None:
            torch.testing.assert_close(pa[name].grad, pb[name].grad, rtol=1e-4, atol=5e-6, msg=name)
    assert pa['block_memory.writer.queries'].grad.abs().sum() > 0


def test_cross_rollout_carry_keeps_pre_observation_snapshot():
    m, ref, new, old, rolls = _collect_rollout('cpu', n=13)
    carries = []
    for roll in rolls:
        carry = Rollout(2, 40, 5, value_g=True, block_steps=4)
        carry.carry(roll)
        carries.append(carry)
    for w in range(2):
        a, b = carries[0].block.starts[w][0], carries[1].block.starts[w][0]
        assert a is rolls[0].block.open[w][1] and a.step == b.memory.step
    mask = np.zeros((2, 40), bool); mask[:, 0] = True
    outs = []
    for model, carry in ((m, carries[0]), (ref, carries[1])):
        plan = carry.block.plan(carry, mask)
        with torch.no_grad():
            outs.append(forward_bucketed(model, plan['chunks'], train_history=False)[0])
    compare_result(outs[0], outs[1])
    torch.testing.assert_close(outs[0][1], torch.from_numpy(carries[0].value[:, 0]), rtol=3e-5, atol=3e-6)


def test_leases_keep_referenced_history_and_free_the_rest():
    m, ref, new, old = pair('cpu', slots=2, seed=41)
    rng = np.random.default_rng(41)
    held_new = held_old = None
    for t in range(40):
        rows = frame(rng, 2, t, np.array([1, 1]), int(t == 0))
        pending = synth_pending(rng, 2, m.heads)
        if t == 12:
            held_new, held_old = new.snapshot(0), old.snapshot(0)
            held_rows, held_pending = rows[:1].copy(), pending[:1].copy()
        new.step(rows, pending, [0, 1]); old.step(rows, pending, [0, 1])
        if t % 5 == 4:
            gc.collect(); new.store.collect()
    gc.collect()
    stats = new.store.collect()
    cfg = m.block_memory.config
    # Far more decisions were made than are kept: only live windows/banks/sources survive.
    assert new.store.next_decision == 80 and stats['kv'] < 80 and stats['freed_kv'] + new.store.freed['kv'] > 0
    # The held snapshot (FIFO long since wrapped on the live slot) is still exact.
    chunk_n = [Chunk(held_rows, held_pending, held_new)]
    chunk_o = [Chunk(held_rows, held_pending, held_old)]
    with torch.no_grad():
        a = forward_chunks(m, chunk_n, generator=torch.Generator().manual_seed(1))[0]
        b = forward_chunks(ref, chunk_o, generator=torch.Generator().manual_seed(1))[0]
    compare_result(a, b)
    lane = new.lane_actor(m, 1); lane.restore(0, held_new)
    lane_o = BlockActor(ref, 1); lane_o.restore(0, held_old)
    compare_state(lane, lane_o, 0)
    before = new.store.stats()['kv_rows']
    del chunk_n, held_new, lane
    gc.collect()
    after = new.store.collect()
    assert after['kv'] < before and after['freed_raw'] > 0


def test_object_and_paged_reject_mixed_stores():
    m, ref, new, old = pair('cpu', slots=1)
    other = SlotBlockActor(m, 1)
    rows = frame(np.random.default_rng(1), 1, 0, np.array([1]), 1)
    new.step(rows, synth_pending(np.random.default_rng(1), 1, m.heads), [0])
    with pytest.raises(ValueError):
        other.restore(0, new.snapshot(0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph slot path')
@pytest.mark.parametrize('bf16', [False, True])
def test_graph_slot_actor_matches_object_graph_actor(bf16):
    m, ref, new, old = pair('cuda', slots=4, seed=51, graph=128, bf16=bf16)
    rng = np.random.default_rng(51)
    tol = dict(rtol=.02, atol=.02) if bf16 else TOL
    for t in range(12):
        rows = frame(rng, 4, t, np.array([1, 1, 1, 1]), int(t == 0))
        pending = synth_pending(rng, 4, m.heads)
        keys = (np.full(4, t), np.arange(4))
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            a = new.step(rows, pending, range(4), ukeys=keys)
            b = old.step(rows, pending, range(4), ukeys=keys)
        for x, y in zip(a[1:], b[1:]):
            torch.testing.assert_close(x.float(), y.float(), **tol)
        if not bf16:
            torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    if new.step_graph is not None:   # B42 glue 5: the whole step is one graph (Reader inside)
        assert new.step_graph.replays == 12
    else:
        assert new.reader.replays == 12


def test_microbatched_checkpointed_spatial_matches_reference_gradients():
    m, ref, new, old, rolls = _collect_rollout('cpu', n=21, seed=61)
    _, _, mask = gae(rolls[0], .99, .95)
    plans = [r.block.plan(r, mask) for r in rolls]
    m.train(); ref.train()
    res = []
    for model, plan, micro in ((m, plans[0], 3), (ref, plans[1], 512)):
        out, _, _, _ = forward_chunks(model, [c for c in plan['chunks'] if len(c.rows) == 4],
                                      generator=torch.Generator().manual_seed(8), spatial_micro=micro)
        loss = out[1].square().mean()+out[0].float().logsumexp(-1).mean()
        loss.backward()
        res.append(loss.detach())
    torch.testing.assert_close(res[0], res[1], rtol=1e-5, atol=1e-6)
    pa, pb = dict(m.named_parameters()), dict(ref.named_parameters())
    for name in pa:
        if pa[name].grad is not None or pb[name].grad is not None:
            torch.testing.assert_close(pa[name].grad, pb[name].grad, rtol=1e-4, atol=5e-6, msg=name)


def _visible_rows(spec, seed=0):
    from isaac_bridge.tok_obs import ROW
    from isaac_bridge.structured_input import pack_frame, write_frame
    from test_visible_input import components
    rows = np.zeros(len(spec), ROW)
    for i, args in enumerate(spec):
        write_frame(rows[i:i+1], pack_frame(*components(*args, seed=seed+i)))
    return rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason='compiled Spatial (CUDA)')
def test_compiled_spatial_matches_eager_fp32_and_bf16():
    """B42 glue 3 gate: compiled fixed-bucket Spatial == eager (FP32 <= 1e-5, gradients too;
    BF16 within the graph-test tolerance) at every body bucket and several entity buckets."""
    from isaac_bridge.tok_policy import TokPolicy
    from isaac_bridge.structured_input import from_rows, SPATIAL_KEYS
    from isaac_bridge import block_compile
    torch.manual_seed(3)
    m = TokPolicy(192, 4, 4, items=True, item_tokens=True, global_token=True, block_memory=True,
                  block_config=dict(width=256, layers=4, heads=4)).cuda().train()
    specs = [[(1, 3, 20, 5), (1, 0, 0, 0), (1, 9, 2, 1)],
             [(2, 40, 7, 3), (1, 17, 31, 16), (2, 1, 0, 0)],
             [(4, 128, 12, 9), (3, 70, 1, 2)],
             [(1, 5, 3, 3)]]
    for spec in specs:
        rows = _visible_rows(spec, seed=len(spec))
        pending = synth_pending(np.random.default_rng(1), len(rows), m.heads)
        b = from_rows(rows, 'cuda', pending)
        x = {k: b[k] for k in SPATIAL_KEYS}
        outs = []
        for compiled in (False, True):
            m.zero_grad(set_to_none=True)
            y, pad = m.structured_spatial(x) if compiled else m.structured_spatial.forward_eager(x)
            (y.masked_fill(pad[..., None], 0).square().mean()).backward()
            outs.append((y.detach(), pad, {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}))
        (ye, pe, ge), (yc, pc, gc_) = outs
        assert torch.equal(pe, pc)
        assert (ye-yc).abs().max().item() <= 1e-5
        assert ge.keys() == gc_.keys()
        for k in ge:
            torch.testing.assert_close(gc_[k], ge[k], rtol=1e-4, atol=1e-5, msg=k)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            ye16, _ = m.structured_spatial.forward_eager(x)
            yc16, _ = m.structured_spatial(x)
        torch.testing.assert_close(yc16.float(), ye16.float(), rtol=.02, atol=.02)
    assert block_compile.STATS['compiles'] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actor step graph (CUDA)')
@pytest.mark.parametrize('bf16', [False, True])
def test_actor_step_graph_matches_eager_slot_actor(bf16):
    """B42 glue 5 gate: whole-step CUDA graph == eager slot step (FP32 <= 1e-5, BF16 graph-test
    tolerance): outputs, arena rows, pages; resets, block boundaries (Writer), mixed replay
    masks, fork/restore."""
    from isaac_bridge import block_paged
    if not block_paged.STEP_GRAPH:
        pytest.skip('ISAAC_BLOCK_STEP_GRAPH=0')
    torch.manual_seed(77)
    m = make_policy('cuda').eval()
    g = SlotBlockActor(m, 4, graph_entities=128, bf16=bf16)
    e = SlotBlockActor(m, 4, graph_entities=128, bf16=bf16)
    e.step_graph = None
    assert g.step_graph is not None
    rng = np.random.default_rng(77)
    tol = dict(rtol=.02, atol=.02) if bf16 else dict(rtol=1e-5, atol=1e-5)
    cfg = m.block_memory.config
    T = 70
    snaps = None
    for t in range(T):
        n = 3 if t % 5 == 4 else 4
        slots = list(range(n))
        first = np.array([int(t == 0 or (t == 17 and i == 1)) for i in slots])
        rows = frame(rng, n, t, np.arange(n)+1, first)
        pending = synth_pending(rng, n, m.heads)
        keys = (np.full(n, t), np.arange(n))
        replay = np.array([t % 7 == 3 and i == 2 for i in slots])
        if t == 30:
            snaps = [g.snapshot(0), e.snapshot(0)]
        if t == 45:
            g.restore(3, snaps[0]); e.restore(3, snaps[1])
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            a = g.step(rows, pending, slots, ukeys=keys, evaluate=replay)
            b = e.step(rows, pending, slots, ukeys=keys, evaluate=replay)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0) if not bf16 else None
        for x, y in zip(a[1:], b[1:]):
            torch.testing.assert_close(x.float(), y.float(), **tol)
        np.testing.assert_allclose(g.host_output, e.host_output, **tol)
    assert g.step_graph.replays > 0 and g.step_graph.captures > 0
    for slot in range(4):
        assert g.state.step_host[slot] == e.state.step_host[slot]
        ga, ea = g.state.gather(g.state._ids([slot])[1]), e.state.gather(e.state._ids([slot])[1])
        for k in ('kv', 'bank', 'partial', 'kv_steps', 'ends', 'bank_versions', 'partial_versions'):
            torch.testing.assert_close(ga[k].float(), ea[k].float(), **tol, msg=k)
        gl, el = g.logs[slot], e.logs[slot]
        assert gl.nb == el.nb and gl.n == el.n
        hi = int(g.state.step_host[slot])
        _, kg = gl.decisions(max(gl.offset, hi-8), hi)
        _, ke = el.decisions(max(el.offset, hi-8), hi)
        torch.testing.assert_close(g.store.kv(kg).float(), e.store.kv(ke).float(), **tol)
        torch.testing.assert_close(g.store.z(kg).float(), e.store.z(ke).float(), **tol)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_packed_learner_spatial_matches_padded(device):
    """B42 glue 4 gate: encode_sorted with packed Transformer layers (real tokens only) ==
    padded layers: outputs and every gradient (FP32 <= 1e-5; CUDA runs the compiled path)."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from isaac_bridge import block_paged
    from isaac_bridge.tok_policy import TokPolicy
    from isaac_bridge.structured_input import decode_frame_host, from_host, host_counts
    torch.manual_seed(5)
    m = TokPolicy(64, 2, 4, items=True, item_tokens=True, global_token=True, block_memory=True,
                  block_config=dict(width=64, layers=2, heads=4, latent_width=64)).to(device).train()
    rows = _visible_rows([(1, 3, 20, 5), (1, 0, 0, 0), (2, 40, 7, 3), (1, 17, 31, 16), (4, 128, 12, 9),
                          (3, 70, 1, 2), (1, 9, 2, 1)], seed=9)
    d = decode_frame_host(rows, packed=device == 'cuda')
    d['pending'] = synth_pending(np.random.default_rng(2), len(rows), m.heads)
    d['first'] = np.asarray(rows['first'])
    res = []
    old = block_paged.SPATIAL_PACKED
    try:
        for packed in (False, True):
            block_paged.SPATIAL_PACKED = packed
            m.zero_grad(set_to_none=True)
            z = block_paged.encode_sorted(m, from_host(d, device), 3, host_counts(d))
            z.square().mean().backward()
            res.append((z.detach(), {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}))
    finally:
        block_paged.SPATIAL_PACKED = old
    (za, ga), (zb, gb) = res
    assert (za-zb).abs().max().item() <= 1e-5
    assert ga.keys() == gb.keys()
    for k in ga:
        torch.testing.assert_close(gb[k], ga[k], rtol=1e-4, atol=1e-5, msg=k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actor step graph (CUDA)')
def test_step_graph_capture_refused_falls_back_to_eager(monkeypatch):
    """B42 fix 1: a step whose graph cannot be captured (admission refused) runs the eager slot
    path with the same outputs; once admitted, the graph (captured without gc/empty_cache)
    replays identically."""
    from isaac_bridge import block_paged, block_training
    if not block_paged.STEP_GRAPH:
        pytest.skip('ISAAC_BLOCK_STEP_GRAPH=0')
    torch.manual_seed(5)
    m = make_policy('cuda').eval()
    g = SlotBlockActor(m, 3, graph_entities=128)
    e = SlotBlockActor(m, 3, graph_entities=128)
    e.step_graph = None
    rng = np.random.default_rng(5)
    refuse = [True]
    real = block_training.capture_admission
    monkeypatch.setattr(block_training, 'capture_admission', lambda key: False if refuse[0] else real(key))
    for t in range(9):
        refuse[0] = t < 4
        rows = frame(rng, 3, t, np.arange(3)+1, int(t == 0))
        pending = synth_pending(rng, 3, m.heads)
        keys = (np.full(3, t), np.arange(3))
        a = g.step(rows, pending, range(3), ukeys=keys)
        b = e.step(rows, pending, range(3), ukeys=keys)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        for x, y in zip(a[1:], b[1:]):
            torch.testing.assert_close(x.float(), y.float(), rtol=1e-5, atol=1e-5)
        assert g.step_graph.captures == (0 if t < 4 else 1)
    assert g.step_graph.busy_misses >= 4 and g.step_graph.replays == 5


def test_rollout_chunks_decoded_once_per_update():
    """B42 fix 3: the start-of-update pass and a PPO epoch plan different Chunk objects over the
    same rollout rows; the second reuses the first's host decode (same dict), and outputs, loss
    and every gradient are identical to decoding twice."""
    from isaac_bridge import block_paged
    m, ref, new, old, rolls = _collect_rollout('cpu', n=21, seed=61)
    _, _, mask = gae(rolls[0], .99, .95)
    block_paged.clear_decode_cache()
    res = []
    for cached in (False, True, True):
        if not cached:
            block_paged.clear_decode_cache()
        plan = rolls[0].block.plan(rolls[0], mask)
        chunks = [c for c in plan['chunks'] if len(c.rows) == 4]
        m.zero_grad(set_to_none=True)
        out, _, _, _ = forward_chunks(m, chunks, generator=torch.Generator().manual_seed(8))
        loss = out[1].square().mean()+out[0].float().logsumexp(-1).mean()
        loss.backward()
        res.append((loss.detach(), {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None},
                    [c.__dict__['_host'][1] for c in chunks]))
        if not cached:
            n_cached = len(block_paged.DECODE_CACHE)
            assert n_cached == len(chunks)
    assert len(block_paged.DECODE_CACHE) == n_cached
    assert all(a is b for a, b in zip(res[1][2], res[2][2]))
    for loss, grads, _ in res[1:]:
        torch.testing.assert_close(loss, res[0][0], rtol=0, atol=0)
        for k in grads:
            torch.testing.assert_close(grads[k], res[0][1][k], rtol=0, atol=0, msg=k)
    block_paged.clear_decode_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actor graph pre-capture (CUDA)')
def test_precaptured_actor_graphs_cover_steps_and_keep_values():
    """B42 fix 2: after precapture_actor the common keys need no new capture, and the replayed
    pre-captured graphs give the eager slot step's outputs."""
    from isaac_bridge import block_paged
    if not block_paged.STEP_GRAPH:
        pytest.skip('ISAAC_BLOCK_STEP_GRAPH=0')
    torch.manual_seed(9)
    m = make_policy('cuda').eval()
    g = SlotBlockActor(m, 4, graph_entities=128)
    e = SlotBlockActor(m, 4, graph_entities=128)
    e.step_graph = None
    made, _ = block_paged.precapture_actor(g)
    assert made == g.step_graph.captures + g.encoder.captures + g.reader.captures > 0
    before = (g.step_graph.captures, g.encoder.captures, g.reader.captures)
    rng = np.random.default_rng(9)
    for t in range(6):
        n = (4, 3, 2, 1, 4, 4)[t]
        rows = frame(rng, n, t, np.arange(n)+1, int(t == 0))
        rows['visible_frame'][:] = block_paged.synthetic_rows(n, 5)['visible_frame']
        rows['visible_bytes'][:] = block_paged.synthetic_rows(n, 5)['visible_bytes']
        pending = synth_pending(rng, n, m.heads)
        keys = (np.full(n, t), np.arange(n))
        a = g.step(rows, pending, range(n), ukeys=keys)
        b = e.step(rows, pending, range(n), ukeys=keys)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        for x, y in zip(a[1:], b[1:]):
            torch.testing.assert_close(x.float(), y.float(), rtol=1e-5, atol=1e-5)
    assert (g.step_graph.captures, g.encoder.captures, g.reader.captures) == before


def test_history_mode_detach_equals_no_history_and_random_unchanged():
    """B43 A arm: history_mode='detach' == no historical recompute (history_recompute_blocks 0),
    loss and every gradient bit-identical; 'random' (B/D arms) is the default path."""
    import dataclasses
    m, ref, new, old, rolls = _collect_rollout('cpu', n=21, seed=61)
    _, _, mask = gae(rolls[0], .99, .95)
    plan = rolls[0].block.plan(rolls[0], mask)
    chunks = [c for c in plan['chunks'] if len(c.rows) == 4]
    cfg = m.block_memory.config
    res = []
    for mode, blocks in (('detach', cfg.history_recompute_blocks), ('random', 0),
                         ('random', cfg.history_recompute_blocks), (None, cfg.history_recompute_blocks)):
        m.block_memory.config = dataclasses.replace(cfg, history_recompute_blocks=blocks)
        m.zero_grad(set_to_none=True)
        kw = {} if mode is None else dict(history_mode=mode)
        out, _, ctx, _ = forward_chunks(m, chunks, generator=torch.Generator().manual_seed(8), **kw)
        loss = out[1].square().mean()+out[0].float().logsumexp(-1).mean()
        loss.backward()
        res.append((loss.detach(), {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None},
                    ctx['recompute_stats']['selected_blocks']))
    m.block_memory.config = cfg
    (la, ga, sa), (lb, gb, sb), (lc, gc_, sc), (ld, gd, sd) = res
    assert sa == sb == 0 and sc == sd > 0
    torch.testing.assert_close(la, lb, rtol=0, atol=0)
    for k in ga:
        torch.testing.assert_close(ga[k], gb[k], rtol=0, atol=0, msg=k)
    torch.testing.assert_close(lc, ld, rtol=0, atol=0)
    for k in gc_:
        torch.testing.assert_close(gc_[k], gd[k], rtol=0, atol=0, msg=k)
