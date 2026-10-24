"""Rule 5: raw bytes of encoded, consumed frames are released (CPU, small model).

1. An episode grows over 8 rollouts; after each, the holders (learner's received
   histories, actor slot + rollout prefixes) release what was encoded / sent:
   their live raw bytes stay bounded by one rollout + one block, not the length.
   Reads after release equal reads of a pipeline that never released (bitwise).
2. A branch forked before its parent ends: with the parent reported as in-flight
   lineage its source values stay; the record (pickled, released prefix) encodes
   only its own frames. Without the lineage the request raises and is counted.
3. Eager ActorMemory.refresh after a weight change: Reader projections only, no
   raw replay, works on released histories.
"""
import copy, json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch, SourceView, ReleasedRows, ActorMemory
from isaac_bridge.record_retention import release_raw, lineage
from isaac_bridge.source_pipeline import SourcePipeline

B, ROLL, N = 4, 16, 8
CFG = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
           temporal_heads=4, temporal_window=4, rebuild_memory=True, memory_block=B, memory_tokens=2, source_cache=True)


def raw_bytes(*holders):
    seen, total = set(), 0
    def walk(x):
        nonlocal total
        if isinstance(x, RawHistory):
            for r, p in x.blocks + x.tail:
                for a in (r, p):
                    if isinstance(a, ReleasedRows): continue
                    while isinstance(a, np.ndarray) and a.base is not None: a = a.base
                    if id(a) not in seen: seen.add(id(a)); total += a.nbytes if isinstance(a, np.ndarray) else len(a)
        elif isinstance(x, dict): [walk(v) for v in x.values()]
        elif isinstance(x, (list, tuple)): [walk(v) for v in x]
    walk(holders)
    return total


def main():
    torch.set_num_threads(2); torch.manual_seed(5)
    base = TokPolicy(**CFG).eval()
    rng = np.random.default_rng(5)
    seq = TS.make_seq(rng, nw=1, T=ROLL * N + 1, changes={(0, t) for t in range(7, ROLL * N, 13)}, resets=set(), exits=set())
    pend = [M.synth_pending(rng, 1, base.heads)[0] for _ in seq]
    row_bytes = seq[0][0].dtype.itemsize
    models = [copy.deepcopy(base), copy.deepcopy(base)]
    pipes = [SourcePipeline(m, ['cpu', 'cpu'], period=0, micro=8, update_rollouts=2, horizon=8) for m in models]
    h = RawHistory.empty()
    slot = [h]; learner_hold = {}; prefixes = []
    full_slot = [h]   # the same episode, never released: pipeline 1's input
    sent, encoded = {}, {}
    sizes, outs = [], [[], []]
    fork_at, lane = 3 * ROLL + 5, None
    for r in range(N):
        req, req_full = [], []
        for t in range(r * ROLL, (r + 1) * ROLL):
            req.append((slot[0], seq[t][0], pend[t])); prefixes.append(slot[0])
            req_full.append((full_slot[0], seq[t][0], pend[t]))
            if t == fork_at: lane = slot[0].fork()
            slot[0] = slot[0].append(seq[t][0], pend[t], B)
            full_slot[0] = full_slot[0].append(seq[t][0], pend[t], B)
        for k, (model, pipe) in enumerate(zip(models, pipes)):
            req_k = req if k == 0 else req_full
            pipe.retain_histories({slot[0].uid} | ({lane.uid} if lane is not None else set()),
                                  frontiers={slot[0].uid: req[0][0].length})
            with torch.no_grad(): bank = SourceBatch(model, req_k, cache=False, pipeline=pipe)
            outs[k].append(SourceView(bank, np.arange(len(req))).forward(np.arange(len(req)), need_vg=True)[0].detach())
        # the actor sent everything up to its latest frame; the learner encoded the rollout
        sent[slot[0].uid] = slot[0].length; encoded[slot[0].uid] = req[-1][0].length + 1
        learner_hold[slot[0].uid] = req[-1][0]
        release_raw(lambda u: sent.get(u, 0), B, slot_lists=[slot], context_lists=[[prefixes]])
        release_raw(lambda u: encoded.get(u, 0), B, context_dicts=[learner_hold])
        prefixes = [p for p in prefixes[-ROLL:]]   # the actor keeps two rollouts of prefixes
        sizes.append(raw_bytes(slot, prefixes, learner_hold) / row_bytes)
    for a, b in zip(outs[0], outs[1]): assert torch.equal(a, b), 'released bytes changed a read'
    assert raw_bytes(full_slot) / row_bytes >= ROLL * N   # the control really kept every frame
    bound = 3 * (ROLL + B)   # three holders, each at most one rollout + one open block
    assert max(sizes[2:]) <= bound and sizes[-1] <= sizes[3] + B, (sizes, bound)
    last = [(learner_hold[slot[0].uid], seq[ROLL * N - 1][0], pend[ROLL * N - 1])]
    assert all(isinstance(b[0], ReleasedRows) for b in learner_hold[slot[0].uid].blocks[:-1])
    full = full_slot[0]
    # 2. a branch whose parent's bytes are released: values via lineage
    branch = lane
    for t in range(fork_at, fork_at + 6): branch = branch.append(seq[t][0], pend[t], B)
    record = pickle.loads(pickle.dumps(released(branch, lambda u: sent.get(u, 0))))
    assert any(isinstance(b[0], ReleasedRows) for b in record.blocks)
    assert lineage(slot_lists=[[branch]]) == {slot[0].uid}
    before = pipes[0].encoded_rows
    with torch.no_grad():
        SourceBatch(models[0], [(record, seq[record.length][0], pend[record.length])], cache=False, pipeline=pipes[0])
    own = record.length + 1 - record.own_start()
    assert pipes[0].encoded_rows - before == own, (pipes[0].encoded_rows - before, own)
    # without lineage the parent retires: a later branch request has neither values nor bytes
    pipes[1].retain_histories(set(), frontiers={})
    try:
        with torch.no_grad():
            SourceBatch(models[1], [(record, seq[record.length][0], pend[record.length])], cache=False, pipeline=pipes[1])
        raise AssertionError('missing ancestor values were not detected')
    except RuntimeError as error:
        assert 'neither values nor bytes' in str(error) and pipes[1].released_misses == 1, error
    # 3. eager refresh: projections only, on released raw
    mem = ActorMemory(1)
    model = models[0]
    temporal = None
    mem.raw[0] = released(full, lambda u: 10 ** 9)
    mem.recent[0] = torch.randn(4, model.reader.D); mem.mem[0] = torch.randn(full.length // B * model.reader.tokens, model.reader.D)
    mem.generation[0] = -5
    mem.refresh(model, [0])
    rd = model.reader
    rt = torch.arange(full.length - rd.W, full.length)
    expect = torch.stack([layer.kv(mem.recent[0] + rd.position(rt, mem.recent[0].dtype)) for layer in rd.layers], 1)
    assert torch.equal(mem.recent_kv[0], expect) and mem.generation[0] == model.source_generation
    for p in pipes: p.close()
    print(json.dumps(dict(result='PASS', live_raw_frames_per_rollout=[round(s, 1) for s in sizes], bound=bound,
                          record_own_frames=own, released_misses=pipes[1].released_misses)))


def released(h, limit):
    from isaac_bridge.rebuild_memory import released as rel
    return rel(h, limit, B)


if __name__ == '__main__':
    main()
