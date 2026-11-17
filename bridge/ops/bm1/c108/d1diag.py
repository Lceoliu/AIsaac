"""C108 D1 (2026-11-15, throwaway: only in the tree copies maa1e98a-d1 / m1ae82d3-d1, never committed).

The sampled actions' log-probabilities as the actor stored them (rollout logp: the sum over the heads, float32)
against the learner's recomputation of the same actions with the same weights. ISAAC_D1_DIAG=1 switches it on.
  prox(): the block tree's start-of-update pass (--decoupled: every decision of the rollout once, learner weights at
          the update's start). One JSON line per update in <out>/d1_diag.jsonl (kind prox) and the raw arrays in
          <out>/d1_raw_u<update>.npz (la = actor logp, lb = learner logp, forced, step = the decision's index in its
          episode (-1: unknown), version = the weights' version that sampled it, worker).
  part(): every PPO part (both trees; the old tree has no start-of-update pass): d = stored - learner logp of the
          part's non-forced rows, gathered over the update's parts (all epochs: with lr 0 the same values each
          epoch) and written at the next update / at exit (kind part).
Each line also carries the SHA-256 of the learner's parameters (their bytes in state_dict order) when it was taken.
Forced pickup decisions are left out of every statistic: their stored log-probability is the sampled action's, not
the forced one's (tok_pickup.ppo_part_terms)."""
import atexit
import hashlib
import json
import os

import numpy as np

ON = os.environ.get('ISAAC_D1_DIAG', '0') == '1'
BLOCK = int(os.environ.get('ISAAC_D1_BLOCK', '32'))   # block-memory v1's block_steps
_state = dict(out=None, update=None, parts=[], forced=0, model=None)


def _wh(model):
    if model is None:
        return ''
    import torch
    h = hashlib.sha256()
    with torch.no_grad():
        for k, v in model.state_dict().items():
            h.update(k.encode())
            h.update(v.detach().contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _q(x):
    x = np.asarray(x, np.float64)
    if not len(x):
        return dict(n=0)
    a = np.abs(x)
    return dict(n=int(len(x)), mean=float(x.mean()), mean_abs=float(a.mean()), p50=float(np.quantile(a, .5)),
                p90=float(np.quantile(a, .9)), p99=float(np.quantile(a, .99)), p999=float(np.quantile(a, .999)),
                max=float(a.max()), gt_0_01=float((a > 0.01).mean()), gt_0_1=float((a > 0.1).mean()),
                gt_clip_0_2=float((a > np.log(1.2)).mean()), w_clamped_2=float((np.exp(-x) > 2.0).mean()))


def _write(rec):
    out = _state['out']
    if out is None:
        return
    with open(os.path.join(str(out), 'd1_diag.jsonl'), 'a') as f:
        f.write(json.dumps(rec) + '\n')


def prox(rl, rp, flat, lp, update, out, model):
    """The block tree's start-of-update pass: rl the rollout, rp rl.block.plan(...) (its chunks in flat's order)."""
    _state['out'] = out
    _flush()
    _state['update'], _state['model'] = int(update), model
    lb = lp.detach().float().cpu().numpy().astype(np.float64)
    la = rl.logp.reshape(-1)[flat].astype(np.float64)
    rows = rl.rows.reshape(-1)[flat]
    forced = rows['pforce'] > 0 if 'pforce' in rows.dtype.names else np.zeros(len(flat), bool)
    ver = rl.version.reshape(-1)[flat].astype(np.int64)
    step = np.full(len(flat), -1, np.int64)
    for (off, n, w, b), ch in zip(rp['tch'], rp['chunks']):
        s0 = getattr(ch.snapshot, 'step', None)
        if s0 is not None:
            step[off:off + n] = int(s0) + np.arange(n)
    worker = (np.asarray(flat) // rl.cap).astype(np.int32) if hasattr(rl, 'cap') else np.zeros(len(flat), np.int32)
    np.savez_compressed(os.path.join(str(out), f'd1_raw_u{int(update):06d}.npz'), la=la, lb=lb, forced=forced,
                        step=step, version=ver, worker=worker)
    keep = ~forced
    d, st, v = (la - lb)[keep], step[keep], ver[keep]
    k = np.where(st >= 0, st % BLOCK, -1)
    rec = dict(kind='prox', update=int(update), n=int(len(flat)), n_forced=int(forced.sum()), weights=_wh(model),
               all=_q(d), first_block=_q(d[(st >= 0) & (st < BLOCK)]), later_blocks=_q(d[st >= BLOCK]),
               step_unknown=int((st < 0).sum()), stale=_q(d[v < update]), fresh=_q(d[v >= update]),
               by_step_in_block={int(i): dict(n=int((k == i).sum()), mean_abs=float(np.abs(d[k == i]).mean()),
                                              p99=float(np.quantile(np.abs(d[k == i]), .99)))
                                 for i in range(BLOCK) if (k == i).any()})
    _write(rec)


def part(logp, b_logp, pf, update, out=None, model=None):
    """One PPO part: logp the learner's (with the graph), b_logp the stored, pf the forced flags (or None)."""
    if out is not None:
        _state['out'] = out
    if model is not None:
        _state['model'] = model
    u = int(update) if update is not None else _state['update']
    if _state['update'] is not None and u != _state['update']:
        _flush()
    _state['update'] = u
    d = (b_logp.detach().float() - logp.detach().float()).cpu().numpy().astype(np.float64)
    if pf is not None:
        f = pf.detach().cpu().numpy().astype(bool)
        _state['forced'] += int(f.sum())
        d = d[~f]
    _state['parts'].append(d)


def _flush():
    if _state['parts']:
        d = np.concatenate(_state['parts'])
        _write(dict(kind='part', update=_state['update'], n_forced=_state['forced'], weights=_wh(_state['model']),
                    all=_q(d)))
    _state['parts'], _state['forced'] = [], 0


atexit.register(_flush)
