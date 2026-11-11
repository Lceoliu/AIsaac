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
OPTIONAL_KEYS = ('value_g', 'close_hurt', 'close_done', 'close_ev', 'priv')


class JobSampler:
    """Stand-in for TokSampler when the job is loaded from a file: the counters learn() logs, no workers."""

    def __init__(self, stats):
        self.stats = stats
        self.aux = None
        self.control = np.zeros(4, np.int64)
        self.live = True
        self.lane_rows = True

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
    roll.update({k: np.ascontiguousarray(getattr(rl,k)[:,:m]) for k in OPTIONAL_KEYS if getattr(rl,k) is not None})
    if rl.block is not None:
        roll['block_payload'] = rl.block.payload()
    roll.update(length=rl.length.copy(), open=rl.open.copy(), n=rl.n, cap=rl.cap, nh=rl.nh)
    out = {k: v for k, v in job.items() if k != 'roll'}
    out['roll'], out['stats'] = roll, np.array(stats)
    out['input_schema'] = model.config.get('input_schema')
    lst = {k: L[k] for k in L_KEYS if k in L}
    if L.get('teach_raw') is not None:
        lst['teach_raw'] = L['teach_raw'][:max(L['teach_n'], 1)].cpu()
        lst['teach_raw_len'] = L['teach_raw'].shape[0]
    out['learner'] = dict(input_schema=model.config.get('input_schema'),model={k: v.detach().cpu() for k, v in model.state_dict().items()},
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
    from isaac_bridge.structured_input import SCHEMA
    if job.get('input_schema') != SCHEMA:
        raise ValueError('offline job is not visible-v1; recollect from the new random model')
    r = job['roll']
    from isaac_bridge.block_training import BlockRollout
    rl = rollout_cls(r['n'], capacity, r['nh'], value_g='value_g' in r,
                     close='close_hurt' in r, priv=r['priv'].dtype if 'priv' in r else None,
                     block_steps=128 if 'block_payload' in r else 0)
    m = r['rows'].shape[1]
    for k in JOB_KEYS + tuple(k for k in OPTIONAL_KEYS if k in r):
        getattr(rl, k)[:, :m] = r[k]
    rl.length[:], rl.open[:] = r['length'], r['open']
    job['roll'] = rl
    if 'block_payload' in r:
        rl.block = BlockRollout.from_payloads([(0,r['block_payload'])])
    return job, job.pop('stats')


def restore_learner(saved, model, opt, L):
    """The learner's state saved with a job (save_job) into this run's model, optimiser and L."""
    import torch
    if saved.get('input_schema') != model.config.get('input_schema'):
        raise ValueError('offline learner input schema mismatch')
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


def _snapshot(model, opt, L, extra=()):
    """extra: (name, object) of the other learner state an update changes (BM1: the privileged critic, the aux head,
    their optimisers: state_dict(); the aux targets' normalisation: its attributes)."""
    import torch

    def grab(o):
        if hasattr(o, 'state_dict'):
            return copy.deepcopy(o.state_dict())
        return copy.deepcopy(vars(o))
    return dict(model={k: v.detach().clone() for k, v in model.state_dict().items()},
                extra=[(nm, o, grab(o)) for nm, o in extra if o is not None],
                opt=copy.deepcopy(opt.state_dict()),
                L={k: (v.copy() if isinstance(v, np.ndarray) else v.clone() if torch.is_tensor(v) else v)
                   for k, v in L.items() if k not in ('writer', 'stage')},   # (stage: the pinned buffer, kept)
                np_rng=np.random.get_state(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())


def _restore(s, model, opt, L):
    import torch
    model.load_state_dict(s['model'])
    opt.load_state_dict(copy.deepcopy(s['opt']))
    for _, o, st_ in s.get('extra', ()):
        if hasattr(o, 'load_state_dict'):
            o.load_state_dict(copy.deepcopy(st_))
        else:
            vars(o).update(copy.deepcopy(st_))
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
    if os.environ.get('ISAAC_RL_LEARNER_SEQUENTIAL') == '1':
        # Exercise the actual production update without resetting Adam, teacher rings or weights.
        paths = [path] + list(job.pop('next_jobs', []))
        for i in range(repeat):
            if i:
                job, stats = load_job(paths[i % len(paths)], type(job['roll']), job['roll'].cap)
                job.pop('learner', None)
                ctx['stats'][:] = stats
            job['t_start'] = time.perf_counter()
            learn(job)
            torch.cuda.synchronize()
            print(f'native learner update {L["update"]}: {L["last_count"]} rows, '
                  f'{L["last_steps"]} optimizer steps, {L["last_update_s"]:.3f}s', flush=True)
        L['stop'] = True
        return True
    check = os.environ.get('ISAAC_RL_LEARNER_CHECK', '0') == '1'
    profile = os.environ.get('ISAAC_RL_LEARNER_PROFILE', '0') == '1'
    # a path: --learner-fast 0 / 1, optionally with a BM1 P2 suffix: 'o' P2 off (whole parts on the learner card),
    # 's' the parts' two halves one after the other on the learner card, 'p' the halves on two cards (--learner-devices)
    # (letters after the digit, any order: o / s / p the BM1 P2 mode as above; k: the learner's Spatial without its
    #  gradient checkpoints, block_learner.SPATIAL_CKPT = 0, recomputation traded for memory; f: float32, the learner's
    #  forward passes without the run's bf16 autocast)
    paths = [int(v) if v.isdigit() else v for v in os.environ.get('ISAAC_RL_LEARNER_PATHS', '0,1').split(',')]
    from isaac_bridge import block_learner as bl_
    ckpt0 = bl_.SPATIAL_CKPT
    amp_ = ctx.get('amp')   # (train_tok's AMP: [True] in a --amp bf16 run)
    amp0 = amp_[0] if amp_ is not None else None

    def set_path(fast):
        s_ = str(fast)
        args.learner_fast = int(''.join(ch for ch in s_ if ch.isdigit()))
        L['p2_mode'] = next((m_ for ch, m_ in (('o', 'off'), ('s', 'serial'), ('p', 'two')) if ch in s_), None)
        bl_.SPATIAL_CKPT = 0 if 'k' in s_ else ckpt0
        if amp_ is not None:   # f: learn() without bf16 autocast (float32 forward passes) within a bf16 run
            amp_[0] = False if 'f' in s_ else amp0
    L['bench'] = True
    params = list(opt.param_groups[0]['params'])
    by_id = {id(p): nm_ for nm_, p in model.named_parameters()}
    names = [by_id.get(id(p), '?') for p in params]
    start = _snapshot(model, opt, L, ctx.get('extra', ()))   # (BM1: priv / aux state restored too)
    print(f'bench: {int(job["roll"].length.sum())} records in the job, --rollout {args.rollout} --minibatch '
          f'{args.minibatch} --micro '
          f'{args.micro} --epochs {args.epochs} --packed {args.packed} --amp {args.amp} --tf32 {args.tf32} '
          f'--fused-adam {args.fused_adam} (Adam fused: {bool(opt.param_groups[0].get("fused"))}) paths {paths}',
          flush=True)

    def one(fast, trace=False):
        _restore(start, model, opt, L)
        set_path(fast)
        L['trace'] = [] if trace else None
        torch.cuda.synchronize()
        dev_ = next(model.parameters()).device
        for i_ in range(torch.cuda.device_count()):   # (every visible card: P2's second learner card too)
            torch.cuda.reset_peak_memory_stats(i_)
        t = time.perf_counter()
        learn(job)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        peak_ = (torch.cuda.max_memory_allocated(dev_) / 2 ** 30, torch.cuda.max_memory_reserved(dev_) / 2 ** 30,
                 {i_: torch.cuda.max_memory_allocated(i_) / 2 ** 30 for i_ in range(torch.cuda.device_count())})
        tr = L['trace']
        L['trace'] = None
        return dict(s=dt, update_s=L['last_update_s'], steps=L['last_steps'], count=L['last_count'],
                    U=dict(L['last_U']), trace=tr, peak=peak_,
                    weights=[p.detach().clone() for p in params] if trace else None)

    for fast in list(paths):   # warm-up (first-call costs: cuDNN plans, allocator, compilation)
        try:
            for _ in range(2):
                one(fast)
        except torch.OutOfMemoryError as e:   # (a path that does not fit, e.g. k without checkpoints: reported, dropped)
            L['trace'] = None
            torch.cuda.empty_cache()
            print(f'bench: path {fast} ran out of GPU memory in its warm-up, dropped: {str(e).splitlines()[0][:300]}',
                  flush=True)
            paths.remove(fast)
    if check:
        tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        settings = (('this run (tf32 %s)' % bool(tf32[0]), tf32), ('strict float32', (False, False)))
        for label, setting in settings[:1 if os.environ.get('ISAAC_RL_LEARNER_CHECK_STRICT', '1') == '0' else 2]:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = setting
            # (each later path against the first; BM1 P2: o / s / p; ISAAC_RL_LEARNER_CHECK_PAIRS='a:b,c:d': these pairs,
            #  a pair with a path dropped above skipped)
            pv_ = lambda v: int(v) if v.isdigit() else v
            pairs_ = os.environ.get('ISAAC_RL_LEARNER_CHECK_PAIRS', '')
            pairs_ = ([tuple(pv_(x) for x in q.split(':')) for q in pairs_.split(',')] if pairs_ else
                      [(paths[0], q) for q in paths[1:]])
            for path_b, path_o in pairs_:
                if path_b not in paths or path_o not in paths:
                    print(f'check: pair {path_b} / {path_o} skipped (a path dropped)', flush=True)
                    continue
                ref, new = one(path_b, trace=True), one(path_o, trace=True)
                ref2 = one(path_b, trace=True)   # the old path against itself (run-to-run floor of the GPU kernels)
                new2 = one(path_o, trace=True)  # and the new one (its gathers' gradients are atomic sums)
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
                    print(f'check [{label}] {path_b} / {path_o} {name}: steps {len(base["trace"])}/{len(other["trace"])}; first step losses '
                          f'max abs {first[0]:.3e}; all steps losses max abs {float((step_r - step_o).abs().max()):.3e} '
                          f'(columns pg vf ent kl clip teach danger safe: '
                          f'{" ".join("%.1e" % v for v in (step_r - step_o).abs().max(0).values.tolist())}); '
                          f'first gradient max abs {max(d for d, _ in gd):.3e}, max rel per tensor '
                          f'{max(r for _, r in gd):.3e}, norm rel {float((flat_r - flat_o).norm() / flat_r.norm()):.3e}, '
                          f'cosine {cos:.9f}; weights after the update max abs {max(d for d, _ in wd):.3e}, max rel per tensor '
                          f'{max(r for _, r in wd):.3e}',
                          flush=True)
                    worst = sorted(((float((a - b).norm() / a.norm().clamp(min=1e-30)), nm_, float(a.norm()))
                                    for nm_, a, b in zip(names, g_r, g_o) if a is not None and b is not None),
                                   reverse=True)[:4]
                    print(f'check [{label}] {path_b} / {path_o} {name}: first gradient, tensors with the largest relative difference (rel '
                          f'norm, name, gradient norm): {" | ".join("%.2e %s %.2e" % w_ for w_ in worst)}', flush=True)
                    # per module: ||g_new - g_old|| / ||g_old|| over the module's parameters concatenated (block_memory /
                    # structured_spatial by their sub-module, layer numbers merged), the module's share of ||g_old||
                    grp = {}
                    for nm_, a, b in zip(names, g_r, g_o):
                        if a is None or b is None:
                            continue
                        parts_ = nm_.split('.')
                        key_ = parts_[0] if parts_[0] not in ('block_memory', 'structured_spatial') or len(parts_) < 3 \
                            else parts_[0]+'.'+parts_[1]
                        s_ = grp.setdefault(key_, [0.0, 0.0])
                        s_[0] += float((a.double() - b.double()).pow(2).sum())
                        s_[1] += float(a.double().pow(2).sum())
                    tot_ = sum(v[1] for v in grp.values())
                    print(f'check [{label}] {path_b} / {path_o} {name}: first gradient per module (||dg||/||g||, '
                          f'share of ||g||^2): ' + ' | '.join(
                              '%s %.2e %.3f' % (k_, (v[0] / max(v[1], 1e-300)) ** 0.5, v[1] / max(tot_, 1e-300))
                              for k_, v in sorted(grp.items(), key=lambda kv: -kv[1][1])), flush=True)
                    print(f'check [{label}] {path_b} / {path_o} {name}: first step losses old {step_r[0].tolist()}', flush=True)
                    print(f'check [{label}] {path_b} / {path_o} {name}: first step losses new {step_o[0].tolist()}', flush=True)
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
              f'step, {args.epochs * count / s:.0f} rows/s ({count} rows x {args.epochs} epochs), learner card peak '
              f'{max(x["peak"][0] for x in r):.2f} GB allocated / {max(x["peak"][1] for x in r):.2f} GB reserved (every '
              f'card allocated: {" ".join("cuda:%d %.2f" % (i_, max(x["peak"][2][i_] for x in r)) for i_ in r[0]["peak"][2])}) | u_* '
              f'{" ".join("%s %.3f" % kv for kv in r[-1]["U"].items())}', flush=True)
    if os.environ.get('ISAAC_RL_LEARNER_CPROFILE', '0') == '1':   # the host's Python time per function
        import cProfile
        import io
        import pstats
        for fast in paths:
            _restore(start, model, opt, L)
            set_path(fast)
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
            set_path(fast)
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
    bl_.SPATIAL_CKPT = ckpt0
    if amp_ is not None:
        amp_[0] = amp0
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
    p.add_argument('--sequential', action='store_true', help='advance the production learner instead of resetting each trial')
    p.add_argument('rest', nargs=argparse.REMAINDER, help='-- and the train_tok.py arguments')
    a = p.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ['--'] else a.rest
    os.environ['ISAAC_RL_LEARNER_BENCH'] = str(max(1, a.repeat))
    os.environ['ISAAC_RL_LEARNER_JOB'] = a.job
    os.environ['ISAAC_RL_LEARNER_CHECK'] = '1' if a.check else '0'
    os.environ['ISAAC_RL_LEARNER_PROFILE'] = '1' if a.profile else '0'
    os.environ['ISAAC_RL_LEARNER_PATHS'] = a.paths
    os.environ['ISAAC_RL_LEARNER_SEQUENTIAL'] = str(int(a.sequential))
    sys.argv = ['train_tok.py'] + rest
    import train_tok
    train_tok.main()


if __name__ == '__main__':
    main()
