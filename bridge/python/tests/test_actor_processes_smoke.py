"""Smoke test of B38 step 5 (2026-10-17, train_tok --actor-processes 2): two actor processes' halves of a rollout.

Plain script, CPU (no game, no GPU):
  cd rl/bridge/python && python tests/test_actor_processes_smoke.py
Checks (test_temporal_smoke's temporal model with memory, item and global tokens; two actors of 3 workers each, each
with its own sequence of records and its own pair of rollout buffers; the second buffer carries the first's context and
its temporal pool grows):
  (a) full_from_local (the first actor) / slice_into (the second): every host array of all 6 workers, after both actors
      stored their sequences through their own rollout objects, == the arrays of the same sequences stored into
      ordinary rollouts (rows 0-2 the first actor's, 3-5 the second's), bit for bit; the shared arrays are what the
      actors' rollouts write (no copy);
  (b) merge_rollouts: the learner's chunk forward over the merged rollout (6 workers, the two temporal pools in one
      mirror, the second actor's row indices moved) == over each actor's own rollout, per decision (logits, value,
      value_global, GRU state), for both buffers; the bytes = the used rows of both pools;
  (c) merge_jobs: episodes and records concatenated in actor order, counters summed, the workers' stats stacked,
      collect / cycle / wait seconds the slowest actor's, the per-actor collect seconds kept.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_temporal_smoke as TS   # noqa: E402
import test_memory_smoke as M   # noqa: E402

import numpy as np   # noqa: E402
import torch   # noqa: E402

from isaac_bridge import tok_policy as TP   # noqa: E402
import train_tok as TT   # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def np_alloc(shape, dtype):
    return np.zeros(shape, dtype)


def arrays_of(r):
    return {k: v for k, v in vars(r).items() if isinstance(v, np.ndarray)}


def by_flat(outs, flat, cap, w_off=0):
    """{(worker + w_off, t): [output rows]} of TS.learner_outputs' outputs."""
    d = {}
    for i, f in enumerate(flat):
        d[(int(f // cap) + w_off, int(f % cap))] = [o[i] for o in outs]
    return d


def main():
    cfg = dict(TS.SMALL, memory=TS.DM, **TS.BASE, **TS.FUSED, **TS.TK)
    torch.manual_seed(5)
    m = TP.TokPolicy(**cfg).eval()
    TS.perturb_all(m, 6)
    seqs, pends = [], []
    for a in range(2):   # (each actor its own records)
        rng = np.random.default_rng(50 + a)
        seqs.append(TS.make_seq(rng))
        pends.append([M.synth_pending(rng, TS.NW, m.heads) for _ in seqs[-1]])
    nw = TS.NW
    # the actors' rollouts on the shared arrays of all workers, and reference rollouts of their own
    a_rolls = [TS.new_rolls(m) for _ in range(2)]
    fulls = [TT.full_from_local(r, 2 * nw, 0, np_alloc) for r in a_rolls[0]]
    for bid, r in enumerate(a_rolls[1]):
        TT.slice_into(r, fulls[bid], nw)
    refs = [TS.new_rolls(m) for _ in range(2)]
    for a in range(2):
        TS.run_actor(m, seqs[a], pends[a], TS.DM, TP.TemporalState(m, nw, 'cpu', torch.float32), a_rolls[a])
        TS.run_actor(m, seqs[a], pends[a], TS.DM, TP.TemporalState(m, nw, 'cpu', torch.float32), refs[a])
    same, views = True, True
    for bid in range(2):
        for k, f in fulls[bid].items():
            for a in range(2):
                same &= np.array_equal(f[a * nw:(a + 1) * nw], arrays_of(refs[a][bid])[k])
                views &= np.shares_memory(getattr(a_rolls[a][bid], k), f)
    grown = [r[1].t_grown for r in a_rolls]
    check('(a) the shared arrays of all workers == the two actors\' own rollouts, bit for bit (both buffers, every '
          'array); the actors\' rollouts write into them', same and views and min(grown) > 0, f'pools grew {grown}')
    worst, nbytes_ok = 0.0, True
    for bid in range(2):
        parts = [(a * nw, TT.rollout_scalars(a_rolls[a][bid]), (a_rolls[a][bid].t_kv, a_rolls[a][bid].t_o))
                 for a in range(2)]
        view, nb = TT.merge_rollouts(fulls[bid], parts, {}, 'cpu')
        exp_nb = sum((s['n'] * s['tP'] + s['t_used']) * kv[0].numel() * kv.element_size() +
                     (s['n'] * s['tR'] + s['n'] * s['tP'] + s['t_used']) * o[0].numel() * o.element_size()
                     for _, s, (kv, o) in parts)
        nbytes_ok &= nb == exp_nb
        o_m, flat_m, w_m = TS.learner_outputs(m, view, 'fast', TS.TT_CH)
        got = by_flat(o_m, flat_m, view.cap)
        for a in range(2):
            o_r, flat_r, _ = TS.learner_outputs(m, refs[a][bid], 'fast', TS.TT_CH)
            ref = by_flat(o_r, flat_r, refs[a][bid].cap, a * nw)
            for key, vals in ref.items():
                if key not in got:
                    worst = float('inf')
                    continue
                worst = max([worst] + [float((x - y).detach().abs().max()) for x, y in zip(vals, got[key])])
    check('(b) merge_rollouts: learner chunks over the merged rollout == over each actor\'s own (logits, value, '
          'value_global, GRU state; both buffers); bytes = the used rows of both pools', worst < 1e-5 and nbytes_ok,
          f'max |diff| {worst:.1e}')
    h0 = dict(roll=None, bid=1, aid=0, seq=4, finished=[{'a': 0}], branch_finished=[], collect_s=1.5, wait_learner_s=0.2,
              cycle_s=2.0, cycle_wait_s=0.0, take_s=0.01, t_put=10.0, t_start=1.0,
              prof=dict(calls=10, latency=0.5, rows=100), worker=dict(step_s=3.0, wait_s=1.0),
              lane_stat=dict(calls=2, s=0.1), stats=np.ones((3, 4)), teach=(np.zeros(2), np.ones((2, 3))), v2=None,
              choices=None, cpu_collect=0.2, proc_cores=1.0, dec_min=5.0, dec_max=9.0, dec_w='5 9 7', actor_mem=100)
    h1 = dict(h0, aid=1, finished=[{'a': 1}, {'a': 2}], collect_s=1.8, wait_learner_s=0.1, cycle_s=2.1, t_put=11.0,
              t_start=0.5, prof=dict(calls=12, latency=0.7, rows=120), worker=dict(step_s=4.0, wait_s=0.5),
              lane_stat=dict(calls=3, s=0.2), stats=2 * np.ones((3, 4)), teach=(np.zeros(1), np.ones((1, 3))),
              cpu_collect=0.4, proc_cores=1.5, dec_min=4.0, dec_max=8.0, dec_w='4 8 6', actor_mem=200)
    j = TT.merge_jobs([h0, h1])
    ok = (j['finished'] == [{'a': 0}, {'a': 1}, {'a': 2}] and j['collect_s'] == 1.8 and j['cycle_s'] == 2.1 and
          j['wait_learner_s'] == 0.2 and j['t_put'] == 11.0 and j['t_start'] == 0.5 and
          j['prof'] == dict(calls=22, latency=1.2, rows=220) and j['worker'] == dict(step_s=7.0, wait_s=1.5) and
          j['lane_stat'] == dict(calls=5, s=0.30000000000000004) and j['stats'].shape == (6, 4) and
          float(j['stats'][3:].sum()) == 24.0 and j['teach'][0].shape == (3,) and j['teach'][1].shape == (3, 3) and
          j['v2'] is None and abs(j['cpu_collect'] - 0.3) < 1e-12 and j['proc_cores'] == 2.5 and j['dec_min'] == 4.0
          and j['dec_max'] == 9.0 and j['dec_w'] == '5 9 7 4 8 6' and j['collect_actors'] == '1.500 1.800' and
          j['actor_mem_all'] == [100, 200])
    check('(c) merge_jobs: lists and records concatenated, counters summed, the slowest actor\'s seconds', ok)
    print('FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
