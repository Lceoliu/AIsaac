"""B18 (2026-10-10): the PPO learner of train_tok.py timed and checked on a fixed rollout, without the engine.

train_tok.py runs its real learn() here, inside its own main() (the same model, optimiser, teacher ring, arguments):
the first run with the engine learns its first jobs as usual and benches the ISAAC_RL_LEARNER_AT-th (default 8: the
teacher ring has filled; ISAAC_RL_LEARNER_MIN_ENT=k: the first one from there with a record of >= k entities), saving
it to --job (the rollout's records up to the longest worker, the teacher records, the sampler's counters, and the
learner's state before it: weights, optimiser, teacher ring); every later run with the same --job loads it and starts no
engine at all (a stand-in sampler; the collect loop is skipped). On that job:
  - timing: learn() --repeat times per path (the old learner, --learner-fast 0, and the new one, 1, alternating), each
    from the same weights, optimiser state, teacher ring and random generators; printed: ms per update, ms per
    optimiser step, rows per second (epochs x rollout rows / update seconds), the u_* parts of the last run;
  - --check: both paths on the same start state with a trace: the loss sums after every optimiser step (pg, vf,
    entropy, kl, clipped share, teacher, danger, safe mass), the first step's gradient (before clipping and the
    optimiser: the same minibatch, the same weights) and the weights after the update; max abs / relative differences,
    once with this run's TF32 setting and once in strict float32 (TF32 off: the summation-order floor);
  - --profile: torch.profiler over one update per path (tables by GPU time and by CPU time, a chrome trace in --out);
    ISAAC_RL_LEARNER_CPROFILE=1: cProfile of one update per path (the host's Python time);
  - ISAAC_RL_LEARNER_PARTS (train_tok, a measurement aid): a subset of the fast path's parts (default all).
The run ends after the bench (train_tok's finally block: last.pt of the weights as they were before, the sampler
closed).

usage: python abplus_bench_learner.py --job <path.pkl> [--repeat 5] [--check] [--profile] -- <train_tok.py arguments>
(the train_tok arguments as for the training it stands for: --workers, --rollout, --minibatch, --micro, --epochs,
--teacher, --resume ...; with a saved job, --workers must be the one it was saved with)
"""
import argparse
import copy
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np

JOB_KEYS = ('rows', 'pending', 'actions', 'logp', 'value', 'reward', 'terminal', 'version', 'boot')


class JobSampler:
    """Stand-in for TokSampler when the job is loaded from a file: the counters learn() logs, no workers."""

    def __init__(self, stats):
        self.stats = stats
        self.aux = None
        self.control = np.zeros(4, np.int64)
        self.live = True

    def close(self):
        pass


L_KEYS = ('teach_rows', 'teach_meta', 'teach_n', 'teach_at', 'teach_rate', 'update', 'decisions', 'seconds')


def save_job(path, job, stats, model, opt, L):
    """The job as a pickle: the rollout's arrays up to the longest worker, everything else as it is, and the learner's
    state before it (weights, optimiser, the teacher ring: its GPU bytes up to the records held)."""
    import torch
    rl = job['roll']
    m = int(rl.length.max())
    roll = {k: np.ascontiguousarray(getattr(rl, k)[:, :m]) for k in JOB_KEYS}
    roll.update(length=rl.length.copy(), open=rl.open.copy(), n=rl.n, cap=rl.cap, nh=rl.nh)
    out = {k: v for k, v in job.items() if k != 'roll'}
    out['roll'], out['stats'] = roll, np.array(stats)
    lst = {k: L[k] for k in L_KEYS if k in L}
    if L.get('teach_raw') is not None:
        lst['teach_raw'] = L['teach_raw'][:max(L['teach_n'], 1)].cpu()
        lst['teach_raw_len'] = L['teach_raw'].shape[0]
    out['learner'] = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                          opt=torch.utils._pytree.tree_map(lambda v: v.cpu() if torch.is_tensor(v) else v,
                                                           opt.state_dict()), L=lst)
    tmp = str(path) + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(out, f, protocol=5)
    os.replace(tmp, path)
    print(f'bench: job saved to {path} ({Path(path).stat().st_size / 2 ** 20:.0f} MiB, '
          f'{int(rl.length.sum())} records)', flush=True)


def load_job(path, rollout_cls, capacity):
    """(job, sampler counters) from save_job's file; the rollout rebuilt with this run's capacity."""
    with open(path, 'rb') as f:
        job = pickle.load(f)
    r = job['roll']
    rl = rollout_cls(r['n'], capacity, r['nh'])
    m = r['rows'].shape[1]
    for k in JOB_KEYS:
        getattr(rl, k)[:, :m] = r[k]
    rl.length[:], rl.open[:] = r['length'], r['open']
    job['roll'] = rl
    return job, job.pop('stats')


def restore_learner(saved, model, opt, L):
    """The learner's state saved with a job (save_job) into this run's model, optimiser and L."""
    import torch
    model.load_state_dict(saved['model'])
    osd = saved['opt']
    fused = bool(opt.param_groups[0].get('fused'))
    for g in osd['param_groups']:   # (this run's implementation choice wins, as train_tok's --resume)
        g['fused'], g['foreach'] = (True, None) if fused else (None, None)
    opt.load_state_dict(osd)
    if not fused:
        for st in opt.state.values():
            if torch.is_tensor(st.get('step')) and st['step'].is_cuda:
                st['step'] = st['step'].cpu()
    lst = dict(saved['L'])
    raw = lst.pop('teach_raw', None)
    size = lst.pop('teach_raw_len', None)
    L.update(lst)
    if raw is not None:
        dev = next(model.parameters()).device
        L['teach_raw'] = torch.zeros((size, raw.shape[1]), dtype=torch.uint8, device=dev)
        L['teach_raw'][:raw.shape[0]] = raw.to(dev)


def _snapshot(model, opt, L):
    import torch
    return dict(model={k: v.detach().clone() for k, v in model.state_dict().items()},
                opt=copy.deepcopy(opt.state_dict()),
                L={k: (v.copy() if isinstance(v, np.ndarray) else v.clone() if torch.is_tensor(v) else v)
                   for k, v in L.items() if k not in ('writer', 'stage')},   # (stage: the pinned buffer, kept)
                np_rng=np.random.get_state(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())


def _restore(s, model, opt, L):
    import torch
    model.load_state_dict(s['model'])
    opt.load_state_dict(copy.deepcopy(s['opt']))
    for k, v in s['L'].items():
        L[k] = v.copy() if isinstance(v, np.ndarray) else v.clone() if torch.is_tensor(v) else v
    np.random.set_state(s['np_rng'])
    torch.set_rng_state(s['cpu_rng'])
    torch.cuda.set_rng_state(s['cuda_rng'])
    torch.cuda.synchronize()


def _diff(a, b):
    """(max abs difference, max abs difference / max abs value of a)."""
    d = float((a - b).abs().max()) if a.numel() else 0.0
    m = float(a.abs().max()) if a.numel() else 0.0
    return d, d / max(m, 1e-30)


def run_bench(ctx):
    """In train_tok's learner thread, for every job: False (learn() as usual) before the ISAAC_RL_LEARNER_AT-th job of
    a run with the engine (default 8: the teacher ring has filled and the teacher's records join the steps); on that
    job, or on a loaded one, the timing / check / profile above, then True (the run ends). ctx: learn, job, model, opt,
    L, args, stats (the sampler's counters)."""
    import torch
    learn, job, model, opt, L, args = (ctx[k] for k in ('learn', 'job', 'model', 'opt', 'L', 'args'))
    path = os.environ.get('ISAAC_RL_LEARNER_JOB', '')
    if 'learner' in job:   # a loaded job: the learner's state as it was before it
        restore_learner(job.pop('learner'), model, opt, L)
    else:
        L['bench_jobs'] = L.get('bench_jobs', 0) + 1
        if L['bench_jobs'] < int(os.environ.get('ISAAC_RL_LEARNER_AT', '8')):
            return False
        # ISAAC_RL_LEARNER_MIN_ENT=k: only a rollout with a record of at least k entities (its minibatches then pad to
        # that many entity tokens: the slow case of a long run); the jobs before are learned as usual
        rl = job['roll']
        top = max((int(rl.rows['n_ent'][i, :rl.length[i]].max()) for i in range(rl.n) if rl.length[i]), default=0)
        if top < int(os.environ.get('ISAAC_RL_LEARNER_MIN_ENT', '0')):
            print(f'bench: job {L["bench_jobs"]} has at most {top} entities in a record; the next one', flush=True)
            return False
        if path and not Path(path).is_file():
            save_job(path, job, ctx['stats'], model, opt, L)
    repeat = int(os.environ.get('ISAAC_RL_LEARNER_BENCH', '1'))
    check = os.environ.get('ISAAC_RL_LEARNER_CHECK', '0') == '1'
    profile = os.environ.get('ISAAC_RL_LEARNER_PROFILE', '0') == '1'
    paths = [int(v) for v in os.environ.get('ISAAC_RL_LEARNER_PATHS', '0,1').split(',')]
    L['bench'] = True
    params = list(opt.param_groups[0]['params'])
    by_id = {id(p): nm_ for nm_, p in model.named_parameters()}
    names = [by_id.get(id(p), '?') for p in params]
    start = _snapshot(model, opt, L)
    print(f'bench: {int(job["roll"].length.sum())} records in the job, --rollout {args.rollout} --minibatch '
          f'{args.minibatch} --micro '
          f'{args.micro} --epochs {args.epochs} --packed {args.packed} --amp {args.amp} --tf32 {args.tf32} '
          f'--fused-adam {args.fused_adam} (Adam fused: {bool(opt.param_groups[0].get("fused"))}) paths {paths}',
          flush=True)

    def one(fast, trace=False):
        _restore(start, model, opt, L)
        args.learner_fast = fast
        L['trace'] = [] if trace else None
        torch.cuda.synchronize()
        t = time.perf_counter()
        learn(job)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        tr = L['trace']
        L['trace'] = None
        return dict(s=dt, update_s=L['last_update_s'], steps=L['last_steps'], count=L['last_count'],
                    U=dict(L['last_U']), trace=tr,
                    weights=[p.detach().clone() for p in params] if trace else None)

    for fast in paths:   # warm-up (first-call costs: cuDNN plans, allocator, compilation)
        for _ in range(2):
            one(fast)
    if check:
        tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        for label, setting in (('this run (tf32 %s)' % bool(tf32[0]), tf32), ('strict float32', (False, False))):
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = setting
            ref, new = one(paths[0], trace=True), one(paths[-1], trace=True)
            ref2 = one(paths[0], trace=True)   # the old path against itself (run-to-run floor of the GPU kernels)
            new2 = one(paths[-1], trace=True)  # and the new one (its gathers' gradients are atomic sums)
            for name, other, base in (('new vs old', new, ref), ('old vs old', ref2, ref),
                                      ('new vs new', new2, new)):
                n_ = min(len(base['trace']), len(other['trace']))
                acc_r = torch.stack([torch.cat([a, t]) for a, t, _ in base['trace'][:n_]])
                acc_o = torch.stack([torch.cat([a, t]) for a, t, _ in other['trace'][:n_]])
                step_r = torch.diff(acc_r, dim=0, prepend=torch.zeros_like(acc_r[:1]))
                step_o = torch.diff(acc_o, dim=0, prepend=torch.zeros_like(acc_o[:1]))
                g_r, g_o = base['trace'][0][2], other['trace'][0][2]
                gd = [(_diff(a, b)) for a, b in zip(g_r, g_o) if a is not None and b is not None]
                flat_r = torch.cat([a.reshape(-1) for a in g_r if a is not None])
                flat_o = torch.cat([b.reshape(-1) for b in g_o if b is not None])
                cos = float(torch.nn.functional.cosine_similarity(flat_r.double(), flat_o.double(), 0))
                wd = [_diff(a, b) for a, b in zip(base['weights'], other['weights'])]
                first = _diff(step_r[0], step_o[0])
                print(f'check [{label}] {name}: steps {len(base["trace"])}/{len(other["trace"])}; first step losses '
                      f'max abs {first[0]:.3e}; all steps losses max abs {float((step_r - step_o).abs().max()):.3e} '
                      f'(columns pg vf ent kl clip teach danger safe: '
                      f'{" ".join("%.1e" % v for v in (step_r - step_o).abs().max(0).values.tolist())}); '
                      f'first gradient max abs {max(d for d, _ in gd):.3e}, max rel per tensor '
                      f'{max(r for _, r in gd):.3e}, norm rel {float((flat_r - flat_o).norm() / flat_r.norm()):.3e}, '
                      f'cosine {cos:.9f}; weights after the update max abs {max(d for d, _ in wd):.3e}',
                      flush=True)
                worst = sorted(((float((a - b).norm() / a.norm().clamp(min=1e-30)), nm_, float(a.norm()))
                                for nm_, a, b in zip(names, g_r, g_o) if a is not None and b is not None),
                               reverse=True)[:4]
                print(f'check [{label}] {name}: first gradient, tensors with the largest relative difference (rel '
                      f'norm, name, gradient norm): {" | ".join("%.2e %s %.2e" % w_ for w_ in worst)}', flush=True)
                print(f'check [{label}] {name}: first step losses old {step_r[0].tolist()}', flush=True)
                print(f'check [{label}] {name}: first step losses new {step_o[0].tolist()}', flush=True)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    res = {fast: [] for fast in paths}
    for _ in range(repeat):
        for fast in paths:
            res[fast].append(one(fast))
    for fast in paths:
        r = res[fast]
        s = statistics.median(x['s'] for x in r)
        us = statistics.median(x['update_s'] for x in r)
        steps, count = r[0]['steps'], r[0]['count']
        print(f'bench --learner-fast {fast}: update {1000 * s:.0f} ms (inside {1000 * us:.0f} ms; runs '
              f'{" ".join("%.0f" % (1000 * x["s"]) for x in r)}), {steps} steps, {1000 * s / max(steps, 1):.1f} ms per '
              f'step, {args.epochs * count / s:.0f} rows/s ({count} rows x {args.epochs} epochs) | u_* '
              f'{" ".join("%s %.3f" % kv for kv in r[-1]["U"].items())}', flush=True)
    if os.environ.get('ISAAC_RL_LEARNER_CPROFILE', '0') == '1':   # the host's Python time per function
        import cProfile
        import io
        import pstats
        for fast in paths:
            _restore(start, model, opt, L)
            args.learner_fast = fast
            pr = cProfile.Profile()
            pr.enable()
            learn(job)
            torch.cuda.synchronize()
            pr.disable()
            s_ = io.StringIO()
            pstats.Stats(pr, stream=s_).sort_stats('tottime').print_stats(35)
            print(f'cProfile --learner-fast {fast} (by own time):\n' + s_.getvalue(), flush=True)
            s_ = io.StringIO()
            pstats.Stats(pr, stream=s_).sort_stats('cumulative').print_stats(45)
            print(f'cProfile --learner-fast {fast} (cumulative):\n' + s_.getvalue(), flush=True)
    if profile:
        from torch.profiler import ProfilerActivity, profile as tprof
        for fast in paths:
            _restore(start, model, opt, L)
            args.learner_fast = fast
            with tprof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                learn(job)
                torch.cuda.synchronize()
            ka = prof.key_averages()
            print(f'profile --learner-fast {fast}: by GPU time', flush=True)
            print(ka.table(sort_by='self_cuda_time_total', row_limit=45, max_name_column_width=70), flush=True)
            print(f'profile --learner-fast {fast}: by CPU time', flush=True)
            print(ka.table(sort_by='self_cpu_time_total', row_limit=30, max_name_column_width=70), flush=True)
            try:
                prof.export_chrome_trace(str(Path(args.out) / f'learner-fast{fast}.trace.json'))
            except Exception as exc:   # (a measurement aid: the tables above are the result)
                print('profile: no chrome trace', exc)
    _restore(start, model, opt, L)
    L['bench'] = False
    L['stop'] = True
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--job', required=True, help='the job file (saved by the first run, loaded by later ones)')
    p.add_argument('--repeat', type=int, default=5, help='timed updates per path')
    p.add_argument('--check', action='store_true', help='compare the losses / gradient / weights of the two paths')
    p.add_argument('--profile', action='store_true', help='torch.profiler over one update per path')
    p.add_argument('--paths', default='0,1', help='--learner-fast values to run (first = the reference)')
    p.add_argument('rest', nargs=argparse.REMAINDER, help='-- and the train_tok.py arguments')
    a = p.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ['--'] else a.rest
    os.environ['ISAAC_RL_LEARNER_BENCH'] = str(max(1, a.repeat))
    os.environ['ISAAC_RL_LEARNER_JOB'] = a.job
    os.environ['ISAAC_RL_LEARNER_CHECK'] = '1' if a.check else '0'
    os.environ['ISAAC_RL_LEARNER_PROFILE'] = '1' if a.profile else '0'
    os.environ['ISAAC_RL_LEARNER_PATHS'] = a.paths
    sys.argv = ['train_tok.py'] + rest
    import train_tok
    train_tok.main()


if __name__ == '__main__':
    main()
