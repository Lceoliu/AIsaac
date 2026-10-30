"""--memory-train truncated, tests 1-3 and the N7 audit (CPU, FP32).

An eager rebuild actor steps 3 workers through 40 decisions (episode starts at step 0 and at RESETS) into train_tok
rollouts handed over at HANDOVERS (as tests/test_truncated_carry.py). For every rollout, with the actor's weights:
  1. actor = learner: the learner's batched forward over each worker's whole rollout (TruncatedBatch, one part and
     parts of single chunks) equals the actor's step outputs (logits, value, value_global) of the same decisions,
     max abs <= 1e-5 (FP32 CPU; different batch shapes only);
  2. no future leak: changing a worker's decisions after k (their observations) leaves the outputs of decisions <= k
     bit for bit unchanged;
  3. gradient window: a loss on each worker's last decision reaches (a) the source value of the decision before it
     (direct read), (b) the source value of the rollout's first frame through a Writer block when that frame is older
     than the direct window (a block begun before the rollout included), (c) the Writer's parameters; the constants
     (carried values / tokens) never require a gradient and get none (audit memory_constant_grad_count = 0);
  4. the gradient window of each decision (steps from the rollout's or its episode's first decision) is in [0, n);
  5. no context mismatch (TruncatedBatch.mismatch, LearnerCarry.mismatch = 0);
  6. rows after a part's decisions: context-free teacher records equal TokPolicy.forward without context, imitation
     records with re-encoded constant contexts (record_constants, M1) equal the source path's SourceBatch.
  python tests/test_truncated_batch.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
import train_tok as TT
from isaac_bridge import tok_policy as TP
from isaac_bridge.truncated_memory import LearnerCarry, TruncatedBatch, record_constants
from isaac_bridge.rebuild_memory import SourceBatch

W, BLOCK, TOKENS = 4, 4, 2
RESETS = {(1, 12), (2, 30)}
HANDOVERS = (10, 22, 31)


def worker_rows(rl, workers):
    rows = np.concatenate([rl.rows[w, :rl.length[w]] for w in workers])
    pend = np.concatenate([rl.pending[w, :rl.length[w]] for w in workers])
    return rows, pend


def compact_batch(model, rows, pend):
    """rows as the learner's compact packed batch (learner_plan + plan_tokens, as SourcePipeline._encode)."""
    ni, g = model.extra_tokens()
    plan = TP.learner_plan(rows['n_ent'], rows['n_doors'], 64,
                           n_item=TP.item_count(rows, model.itok.slots) if ni else None, item_tok=ni, glob=g)
    b = TP.to_batch(rows, pend, 'cpu', entities=64)
    b.update(TP.plan_tokens(plan, torch.from_numpy(plan['rec']), torch.from_numpy(plan['base'])))
    b['buckets'] = [(torch.as_tensor(c), w) for c, w in plan['buckets']]
    b['_source_packed'] = b['_source_compact'] = True
    b['_rec'] = plan['rec']
    return b


def evaluate(model, rl, carry, start_step, log, stats):
    tb = TruncatedBatch(model, rl, carry, 'cpu')
    carry.advance(rl)
    assert tb.mismatch == 0, tb.mismatch
    ws = list(tb.workers)
    rows, pend = worker_rows(rl, ws)
    # 1. one part, then single-chunk parts: the actor's outputs
    with torch.no_grad():
        out = tb.part(ws).forward(model, TP.to_batch(rows, pend, 'cpu'), need_vg=True)
        singles = []
        for w in ws:
            r_, p_ = worker_rows(rl, [w])
            singles.append(tb.part([w]).forward(model, TP.to_batch(r_, p_, 'cpu'), need_vg=True))
        # (the learner's compact plan: real tokens only, padded for the fusion by pad_compact)
        cb = compact_batch(model, rows, pend)
        comp = tb.part(ws).forward(model, cb, need_vg=True)
        for j in (0, 1, -1):
            stats['compact_err'] = max(stats['compact_err'], float((comp[j] - out[j]).abs().max()))
        # (N4, the value recompute pass: parts of whole chunks by bounds(micro), rows of all_flat in order)
        vals = []
        for lo, hi, pw in tb.bounds(8):
            assert hi - lo == sum(int(rl.length[w]) for w in pw)
            vals.append(tb.part(pw).forward(model, TP.to_batch(rows[lo:hi], pend[lo:hi], 'cpu'), need_vg=True)[1])
        stats['max_err'] = max(stats['max_err'], float((torch.cat(vals) - out[1]).abs().max()))
    at = 0
    for i, w in enumerate(ws):
        n = int(rl.length[w])
        steps = start_step + np.arange(n)
        for j, (got, ref) in enumerate([(out[0], log['logits']), (out[1], log['value']), (out[-1], log['vg'])]):
            exp = torch.stack([ref[s][w] for s in steps])
            err = float((got[at:at + n] - exp).abs().max())
            err1 = float(([singles[i][0], singles[i][1], singles[i][-1]][j] - exp).abs().max())
            stats['max_err'] = max(stats['max_err'], err, err1)
        at += n
    # 6. teacher records (context-free) and imitation records (constant contexts) after the decisions
    steps_ = sorted({min(s_, len(log['ctx']) - 1) for s_ in (start_step + 3, start_step + 7)})
    req = [(log['ctx'][s_][w_], log['rows'][s_][w_], log['pend'][s_][w_]) for s_ in steps_ for w_ in range(len(ws))]
    teach_rows = np.stack([log['rows'][start_step][0], log['rows'][start_step + 1][1]])
    teach_pend = np.stack([log['pend'][start_step][0], log['pend'][start_step + 1][1]])
    with torch.no_grad():
        rec = record_constants(model, req, 5)
        allr = np.concatenate([rows, teach_rows, np.stack([r_ for _, r_, _ in req])])
        allp = np.concatenate([pend, teach_pend, np.stack([p_ for _, _, p_ in req])])
        got = tb.part(ws, n_teach=2, records=rec).forward(model, TP.to_batch(allr, allp, 'cpu'), need_vg=True)
        ref_t = model(TP.to_batch(teach_rows, teach_pend, 'cpu'), need_vg=True)
        ref_r = SourceBatch(model, req, micro=5, cache=False).forward(list(range(len(req))), need_vg=True)
    nm = len(rows)
    for j in (0, 1, -1):
        stats['max_err'] = max(stats['max_err'], float((got[j][:nm] - out[j]).abs().max()),
                               float((got[j][nm:nm + 2] - ref_t[j]).abs().max()),
                               float((got[j][nm + 2:] - ref_r[j]).abs().max()))
    stats['records'] += len(req) + 2
    # 2. no future leak (per worker: decisions after k changed)
    for w in ws:
        r_, p_ = worker_rows(rl, [w])
        n = len(r_)
        k = n // 2
        changed = r_.copy()
        changed['player'][k + 1:] += 0.3
        changed['grid'][k + 1:] = 255 - changed['grid'][k + 1:]
        with torch.no_grad():
            a = tb.part([w]).forward(model, TP.to_batch(r_, p_, 'cpu'), need_vg=True)
            b = tb.part([w]).forward(model, TP.to_batch(changed, p_, 'cpu'), need_vg=True)
        for x, y in zip(a, b):
            assert torch.equal(x[:k + 1], y[:k + 1]), ('future leak', w, k)
        assert not torch.equal(a[1][k + 1:], b[1][k + 1:]), 'the change changed nothing'
        stats['leak_checks'] += 1
    # 3. gradient window
    model.zero_grad(set_to_none=True)
    tb.debug = {}
    out = tb.part(ws).forward(model, TP.to_batch(rows, pend, 'cpu'), need_vg=True)
    u = tb.debug['u']
    u.retain_grad()
    last = np.cumsum([int(rl.length[w]) for w in ws]) - 1
    out[1][torch.as_tensor(last)].sum().backward()
    assert all(not c.requires_grad and c.grad is None for c in tb.constants()), 'a constant has a gradient'
    stats['constant_grad_count'] += sum(int(c.requires_grad or c.grad is not None) for c in tb.constants())
    g = u.grad.abs().sum(1)
    at = 0
    for i, w in enumerate(ws):
        n = int(rl.length[w])
        meta = tb.meta[i]
        if len(meta['segs']) == 1 and n > 1:
            assert g[at + n - 2] > 0, ('direct read got no gradient', w)
            stats['direct'] += 1
        if len(meta['segs']) == 1 and n - 1 > W:
            # frame 0 is older than the direct window of the last decision: its gradient comes through the Writer
            # block holding it, closed before the last decision (a block begun before the rollout when t0 % B != 0)
            t0, tl = int(meta['t'][0]), int(meta['t'][-1])
            closes = (t0 // BLOCK + 1) * BLOCK - 1
            if closes < tl // BLOCK * BLOCK:
                assert g[at] > 0, ('Writer path gave no gradient', w, t0)
                stats['writer'] += 1
                stats['boundary'] += int(t0 % BLOCK != 0)
        at += n
    assert model.reader.queries.grad is not None and model.reader.queries.grad.abs().sum() > 0, 'Writer untrained'
    tb.debug = None
    # 4. windows
    win = tb.window()
    at = 0
    for w in ws:
        n = int(rl.length[w])
        assert (win[at:at + n] >= 0).all() and (win[at:at + n] < n).all()
        at += n
    stats['window_mean'].append(float(win.mean()))
    stats['rollouts'] += 1


def main():
    torch.set_num_threads(2); torch.manual_seed(11)
    cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
               temporal_heads=4, temporal_window=W, rebuild_memory=True, memory_block=BLOCK, memory_tokens=TOKENS,
               source_cache=True)
    model = TP.TokPolicy(**cfg).eval()
    rng = np.random.default_rng(11)
    nw, T = 3, 40
    seq = TS.make_seq(rng, nw=nw, T=T, changes={(0, 5), (1, 9), (2, 20)}, resets=RESETS, exits=set())
    pend = [M.synth_pending(rng, nw, model.heads) for _ in range(T)]
    ta = model.tatt
    tcfg = dict(P=W, L=ta.L, D=ta.D, R=ta.R, kv='fp32', device='cpu', truncated=True, tokens=TOKENS)
    rolls = [TT.Rollout(nw, 64, len(model.heads), 0, True, dict(tcfg, rows=None)),
             TT.Rollout(nw, 64, len(model.heads), 0, True, dict(tcfg, rows=7))]
    st = TP.TemporalState(model, nw, 'cpu', torch.float32)
    carry = LearnerCarry(nw, BLOCK, TOKENS, ta.D, 'cpu')
    log = dict(logits=[], value=[], vg=[], ctx=[], rows=[], pend=[])
    stats = dict(compact_err=0.0, records=0, max_err=0.0, leak_checks=0, constant_grad_count=0, direct=0, writer=0, boundary=0, window_mean=[],
                 rollouts=0)
    roll, start = rolls[0], 0
    for t in range(T + 1):
        if t in HANDOVERS or t == T:
            other = rolls[1] if roll is rolls[0] else rolls[0]
            other.carry(roll)
            evaluate(model, roll, carry, start, log, stats)
            roll, start = other, t - 1
        if t == T:
            break
        rows = seq[t]
        temporal = st.step_inputs(model, rows, pend[t], np.arange(nw))
        with torch.no_grad():
            out = model(TP.to_batch(rows, pend[t], 'cpu'), temporal=temporal, need_vg=True)
        log['logits'].append(out[0]); log['value'].append(out[1]); log['vg'].append(out[-1])
        log['ctx'].append(temporal['out']['raw_contexts']); log['rows'].append(rows); log['pend'].append(pend[t])
        TT.store(roll, np.arange(nw), rows, pend[t], np.zeros((nw, len(model.heads)), np.int64),
                 np.zeros(nw, np.float32), out[1].numpy(), 0, None, out[-1].numpy(),
                 TP.temporal_out(temporal['out']))
    assert stats['max_err'] <= 1e-5 and stats['compact_err'] <= 1e-5, stats
    assert carry.mismatch == 0 and stats['constant_grad_count'] == 0
    assert stats['writer'] > 0 and stats['boundary'] > 0 and stats['direct'] > 0, stats
    print(f"PASS truncated batch over {stats['rollouts']} rollouts: actor = learner max_abs {stats['max_err']:.2e} "
          f"(compact plan {stats['compact_err']:.2e}); "
          f"future leak checks {stats['leak_checks']} (bit for bit); gradient: direct {stats['direct']}, Writer "
          f"{stats['writer']} (block begun before the rollout {stats['boundary']}); memory_constant_grad_count "
          f"{stats['constant_grad_count']}; window means {[round(x, 2) for x in stats['window_mean']]}; "
          f"mismatches {carry.mismatch}; records {stats['records']} (teacher context-free, imitation re-encoded)")


if __name__ == '__main__':
    main()
