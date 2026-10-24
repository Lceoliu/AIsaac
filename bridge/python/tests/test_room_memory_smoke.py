"""Step-1 raw room memory: real policy path, causal replay, cross-chunk gradients and learning.

CPU, synthetic observations; no Isaac process. Does not test distributed/GPU throughput
or repair main-policy H/KV staleness. Existing test_memory_smoke supplies import setup.
"""
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge import tok_policy as TP
from isaac_bridge.room_memory import RawRoomReplay, RoomSequenceWriter

torch.set_num_threads(2)
torch.manual_seed(20261020)
rng = np.random.default_rng(20261020)
CFG = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED,
           temporal=True, temporal_width=32, temporal_layers=2, temporal_heads=4,
           temporal_window=4, temporal_rooms=4, room_memory=True)


def check(name, condition, detail=''):
    print(('ok ' if condition else 'FAIL ') + name + (' ' + detail if detail else ''), flush=True)
    assert condition, name


def delayed_batch(model, n, seed, hide=False):
    """Balanced pairs: only the EARLY frame identifies the answer; end frame is identical.

    Main-policy H and decision context are masked in this diagnostic to isolate the
    room path (not a production change). Current query is identical across labels.
    """
    r = np.random.default_rng(seed)
    raw = np.zeros(1, dtype=M.ROW)
    raw['stage'] = 1
    rows = np.repeat(raw, n * 2)
    labels = np.tile(np.array([0, 1], np.int64), n // 2)
    nuisance = np.repeat(r.normal(0, 0.1, n // 2), 2)
    rows['player'][0::2, 1] = nuisance
    rows['player'][0::2, 0] = 0 if hide else labels * 4 - 2
    rows['player'][1::2, 0] = 0
    pend = np.zeros((2 * n, len(model.heads)), np.int64)
    encoded = model.write_room_sequences(TP.to_batch(rows, pend, 'cpu'), [2] * n)
    ta = model.tatt
    sources = torch.cat((encoded, encoded.new_zeros((1, ta.D))))
    zero = torch.zeros(n + 1, dtype=torch.long)
    td = dict(r_src=torch.arange(n + 1), r_k=zero, r_stage=zero,
              r_grp=torch.arange(n + 1)[:, None], r_gpos=torch.arange(n + 1))
    table = ta.room_table(sources, td, track_grad=True)
    plan = dict(raw_room_memory=True, imap=torch.arange(n), omap=torch.arange(n),
                qep=torch.zeros(n, 1, dtype=torch.long), pre_valid=torch.ones(n, 1, dtype=torch.bool),
                pre_kv=[(k[:n, None], v[:n, None]) for k, v in table],
                in_valid=torch.zeros(n, 1, dtype=torch.bool), in_o=sources.new_zeros(n, 1, ta.D),
                in_k=torch.zeros(n, 1, dtype=torch.long), in_stage=torch.zeros(n, 1, dtype=torch.long),
                in_at=torch.zeros(n, 1, dtype=torch.long), in_ep=torch.full((n, 1), -3, dtype=torch.long),
                ctx_kv=sources.new_zeros(n, ta.W - 1, ta.L, 2, ta.D),
                ctx_ep=torch.full((n, ta.W - 1), -3, dtype=torch.long))
    future = np.repeat(raw, n)
    # Future frame has no cue, even if raw's random player[0] was nonzero.
    future['player'][:, 0] = 0
    out = model(TP.to_batch(future, np.zeros((n, len(model.heads)), np.int64), 'cpu'),
                temporal=dict(mode='chunk', plan=plan, o_prev=sources.new_zeros(n, ta.D),
                              has=torch.zeros(n, dtype=torch.bool)))
    return out[0][:, :2], torch.from_numpy(labels)


def main():
    model = TP.TokPolicy(**CFG).eval()
    replay = RawRoomReplay(model)
    seq = TS.make_seq(rng, nw=1, T=18, changes={(0, 3), (0, 10), (0, 13)}, resets={(0, 8)}, exits={(0, 6)})
    pend = [M.synth_pending(rng, 1, model.heads) for _ in seq]
    actor = [replay.append(row, p) for row, p in zip(seq[:7], pend[:7])]
    actor += [replay.append(row, p) for row, p in zip(seq[7:], pend[7:])]
    for start, stop in ((0, 3), (3, 6), (4, 8), (7, 12), (8, 14), (12, 18)):
        with torch.no_grad():
            out = replay.chunk(start, stop)
        error = max(float((out[k] - torch.cat([o[k] for o in actor[start:stop]])).abs().max())
                    for k in (0, 1, 2, 3))
        check(f'actor/chunk parity [{start},{stop})', error < 2e-5, f'max_abs={error:.3g}')

    observed = []
    def retain_source(module, args):
        args[0].retain_grad()
        observed.append(args[0])
    hook = model.room_writer.register_forward_pre_hook(retain_source)
    model.zero_grad(set_to_none=True)
    out = replay.chunk(4, 6)  # room from steps 0..2 is strictly before this training chunk
    logp = out[0][:, :model.heads[0]].log_softmax(-1)[:, 0]
    old_logp = logp.detach().clone()
    ratio = (logp - old_logp).exp()
    advantage = torch.tensor([1.0, -0.3])
    loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
    loss = loss + 0.5 * out[1].square().mean()
    loss.backward()
    hook.remove()
    early = float(observed[0].grad[0].abs().sum())
    writer = float(model.room_writer.query.grad.abs().sum())
    spatial = sum(float(p.grad.abs().sum()) for p in model.blocks.parameters() if p.grad is not None)
    check('future PPO/value loss reaches first source frame / write query / Spatial', min(early, writer, spatial) > 0,
          f'early={early:.6g} query={writer:.6g} spatial={spatial:.6g}')

    prefix = RawRoomReplay(model)
    for row, p in zip(seq[:6], pend[:6]):
        prefix.append(row, p)
    with torch.no_grad():
        before, after = prefix.chunk(4, 6), replay.chunk(4, 6)
    check('future rooms do not affect earlier outputs',
          all(torch.equal(a, b) for a, b in zip(before, after)))
    child = prefix.fork()
    child.append(seq[6], pend[6])
    check('CF fork preserves parent history/state', len(prefix.rows) == 6 and len(child.rows) == 7
          and int(prefix.state.steps[0]) == 6 and int(child.state.steps[0]) == 7)
    check('old raw snapshots are immutable', not prefix.rows[0].flags.writeable)
    check('visit excludes transition frame', prefix.rooms[0][:3] == (0, 3, 3))
    check('archive persists across two collection batches', len(replay.rows) == 18 and replay.rooms[0][:2] == (0, 3))

    # Recomputing the room path does not accidentally cache representations across optimizer steps.
    with torch.no_grad():
        old = replay.chunk(4, 6)[0].clone()
        model.room_writer.query.add_(torch.linspace(-0.1, 0.1, model.tatt.D)[None, None])
        new = replay.chunk(4, 6)[0]
    check('room sources are recomputed after parameter changes', not torch.equal(old, new))

    ck = io.BytesIO()
    torch.save(dict(config=model.config, model=model.state_dict()), ck)
    ck.seek(0)
    state = torch.load(ck, weights_only=True)
    twin = TP.TokPolicy(**state['config']).eval()
    twin.load_state_dict(state['model'], strict=True)
    check('checkpoint reconstructs writer and all weights',
          all(torch.equal(v, twin.state_dict()[k]) for k, v in model.state_dict().items()))
    legacy = TP.TokPolicy(**dict(CFG, room_memory=False))
    migrated = TP.TokPolicy(**CFG)
    TP.load_compatible(migrated, legacy.state_dict())
    check('legacy warm-start preserves every shared weight',
          all(torch.equal(v, migrated.state_dict()[k]) for k, v in legacy.state_dict().items()))
    try:
        TP.load_compatible(legacy, model.state_dict())
    except ValueError as exc:
        check('raw checkpoint cannot silently drop its writer', 'room_memory' in str(exc))
    else:
        raise AssertionError('load_compatible silently dropped the trained room writer')
    try:
        twin(TP.to_batch(seq[0], pend[0], 'cpu'), temporal=dict(mode='step', state=TP.TemporalState(twin, 1),
                                                             slot=torch.tensor([0])))
    except ValueError as exc:
        check('missing raw source fails instead of using o_exit', 'room_source' in str(exc))
    else:
        raise AssertionError('raw writer silently accepted legacy step')

    # Pooling sees sequence order, excludes padding, and can be differentiated through an early frame.
    f = torch.randn(5, 32, requires_grad=True)
    pooled = model.room_writer(f, [2, 3])
    one = model.room_writer(f[:2], [2])
    check('ragged padding does not affect source', torch.allclose(pooled[:1], one, atol=1e-6))
    check('time encoding distinguishes order',
          not torch.allclose(one, model.room_writer(f[:2].flip(0), [2]), atol=1e-6))

    torch.manual_seed(19)
    learner = TP.TokPolicy(**CFG).train()
    opt = torch.optim.Adam(learner.parameters(), lr=0.002)
    for step in range(100):
        opt.zero_grad(set_to_none=True)
        logits, labels = delayed_batch(learner, 32, 1000 + step)
        task_loss = torch.nn.functional.cross_entropy(logits, labels)
        task_loss.backward()
        opt.step()
    learner.eval()
    with torch.no_grad():
        logits, labels = delayed_batch(learner, 128, 9001)
        hidden, _ = delayed_batch(learner, 128, 9001, hide=True)
    accuracy = float((logits.argmax(-1) == labels).float().mean())
    ablated = float((hidden.argmax(-1) == labels).float().mean())
    check('held-out delayed cue learned through actual room/Temporal/policy path', accuracy >= 0.9,
          f'accuracy={accuracy:.4f}, cue_removed={ablated:.4f}, final_loss={float(task_loss.detach()):.6f}')
    check('removing early cue removes the advantage', ablated == 0.5)
    print('RESULT all raw-room memory checks passed; synthetic CPU only; H/KV freshness unchanged')
    return 0


if __name__ == '__main__':
    sys.exit(main())
