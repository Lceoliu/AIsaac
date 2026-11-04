"""Runs train_tok's real learn() on the CPU, without the engine and without CUDA (the helper of test_aux_predict_smoke.py).

train_tok's main() builds everything as for a run and hands learn() to the learner bench (abplus_bench_learner.run_bench,
ISAAC_RL_LEARNER_BENCH); here torch.cuda and torch.device('cuda') are shims that mean the CPU, load_job / run_bench are
replaced by a synthetic rollout (TokPolicy's eager actor over test_temporal_smoke's records with deaths, EV_EXIT and room
changes, closed by the real Episodes.arrive, stored by the real store) and repeated learn() calls. The tree under test
comes first on sys.path (--tree: this one, or an export of the base commit), so the same driver runs the code before the
change and after it. Results (weights, optimiser state, progress.csv's rows, the rollout's arrays) go to --out/result.pt.

usage: python aux_learn_driver.py --tree <dir with bridge/python> --out <dir> [--updates 2] [--fast 1] [--aux 0]
       [--horizons 1,4,16] [--coef 0.1] [--resume <ck>] [--decoupled 1]
"""
import argparse
import copy
import csv
import hashlib
import os
import sys
import time
from pathlib import Path


def install_shims(torch):
    """torch.cuda on the CPU: streams and events do nothing, memory counters read 0, 'cuda' devices are the CPU."""
    real_device = torch.device

    class _DevMeta(type):
        def __instancecheck__(cls, x):
            return isinstance(x, real_device)

    class device(metaclass=_DevMeta):
        def __new__(cls, *a, **k):
            if a and (str(a[0]).startswith('cuda')):
                return real_device('cpu')
            return real_device(*a, **k)

    class Stream:
        def __init__(self, *a, **k):
            pass

        def synchronize(self):
            pass

        def wait_stream(self, s):
            pass

        def wait_event(self, e):
            pass

    class Event(Stream):
        def record(self, *a):
            pass

    S = Stream()

    class StreamCtx:
        def __init__(self, s):
            pass

        def __enter__(self):
            return S

        def __exit__(self, *a):
            return False
    torch.device = device
    c = torch.cuda
    c.Stream, c.Event, c.stream = Stream, Event, StreamCtx
    c.current_stream = lambda *a, **k: S
    c.set_stream = lambda s: None
    c.set_device = lambda d: None
    c.current_device = lambda: 0
    c.device_count = lambda: 1
    c.get_device_name = lambda *a: 'cpu shim'
    c.synchronize = lambda *a: None
    c.max_memory_reserved = lambda *a, **k: 0
    c.is_bf16_supported = lambda *a, **k: True
    torch.Tensor.pin_memory = lambda self, *a, **k: self


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--tree', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--updates', type=int, default=2)
    a.add_argument('--fast', type=int, default=1)
    a.add_argument('--aux', type=int, default=0)
    a.add_argument('--horizons', default='1,4,16')
    a.add_argument('--coef', type=float, default=0.1)
    a.add_argument('--resume', default='')
    a.add_argument('--decoupled', type=int, default=1)
    a.add_argument('--steps', type=int, default=64)
    a.add_argument('--priv', type=int, default=0)
    a.add_argument('--priv-seed', type=int, default=21)
    a.add_argument('--extra', default='', help='more train_tok arguments, comma-separated')
    a.add_argument('--dump-z', type=int, default=0, help='also: the policy\'s z, gn of the first update\'s start vs its stream log')
    o = a.parse_args()
    py = Path(o.tree) / 'bridge' / 'python'
    sys.path.insert(0, str(py / 'tests'))
    sys.path.insert(0, str(py))
    out = Path(o.out)
    run_dir = out / f'run{int(time.time() * 1000)}'   # (train_tok: --out must be new)
    out.mkdir(parents=True, exist_ok=True)
    job_flag = out / 'job.flag'
    job_flag.write_text('stand-in')
    os.environ['ISAAC_RL_LEARNER_BENCH'] = '1'
    os.environ['ISAAC_RL_LEARNER_JOB'] = str(job_flag)
    import test_temporal_smoke as TS   # (its imports: the path set-up, the gymnasium stand-in)
    import numpy as np
    import torch
    install_shims(torch)
    import abplus_bench_learner as BL
    import train_tok as TT
    from isaac_bridge import tok_policy as TP
    from isaac_bridge.tok_sampler import ST

    W, T = 4, o.steps
    argv = ['--groups-file', str(py.parent / 'abplus' / 'catalog' / 'scaling2_groups.json'), '--mode', 'run', '--items',
            '--workers', str(W), '--rollout', str(W * T // 2), '--minibatch', '32', '--micro', '32', '--epochs', '2',
            '--width', '64', '--layers', '2', '--heads', '4',
            '--item-tokens', '1', '--global-token', '1', '--item-shared', '1', '--item-slots', '8',
            '--temporal', '1', '--temporal-width', '32', '--temporal-layers', '2', '--temporal-heads', '4',
            '--temporal-window', '8', '--temporal-rooms', '16', '--temporal-bptt', '8', '--obs-v2', '1',
            '--graph-entities', '0', '--learner-fast', str(o.fast), '--decoupled', str(o.decoupled),
            '--lr', '1e-3', '--lr-final', '1e-3', '--game-hours', '1000', '--checkpoint-every', '100000',
            '--name', 'auxcpu', '--port', '49990', '--out', str(run_dir)]
    if o.aux:
        argv += ['--aux-predict', '1', '--aux-horizons', o.horizons, '--aux-coef', str(o.coef)]
    if o.priv:
        argv += ['--priv-critic', '1']
    if o.extra:
        argv += o.extra.split(',')
    if o.resume:
        argv += ['--resume', o.resume]
    res = {}

    def fake_load_job(path, cls, capacity):
        return dict(stand_in=True, capacity=capacity), np.zeros((W, len(ST)))

    def collect(model, args, capacity):
        rng = np.random.default_rng(7)
        torch.manual_seed(11)
        changes = {(0, 3), (0, 9), (0, 20), (0, 33), (1, 5), (1, 22), (2, 12), (2, 40), (3, 2), (3, 50)}
        resets = {(1, 15), (2, 30), (3, 44)}    # (a death or a time-out just before each)
        exits = {(1, 27), (0, 41)}
        seq = TS.make_seq(rng, W, T, changes, resets, exits)
        for w, t in resets:
            seq[t - 1]['done'][w] = 2 if w != 3 else 3
            seq[t - 1]['hurt'][w] = 1.0
        for r_ in seq:
            r_['done'] = np.where(r_['first'] != 0, 0, r_['done'])
            r_['group'] = 0
            r_['hurt'] = np.where(r_['done'] != 0, r_['hurt'], np.where(rng.random(W) < 0.05, 1.0, 0.0))
            r_['damage'] = rng.random(W) * 0.05
            r_['pforce'] = 0
        rw = dict(room=1.0, explore=0.5, boss=2.0, exit=3.0, damage=1.0, hurt=0.75, death=2.0, timeout=1.0, time=0.002,
                  heal=0.5, resource=0.25, clear=1.0)
        tcfg = dict(P=args.temporal_window - 1, L=args.temporal_layers, D=args.temporal_width, R=args.temporal_rooms,
                    kv=args.temporal_kv, device='cpu', rows=args.rollout + 4 * W + 64)
        kw = dict(close=True) if (o.aux or '--log-stream' in o.extra) else {}
        p_rng = np.random.default_rng(o.priv_seed)
        p_rows = None
        if o.priv:   # (random privileged records: a stand-in for the bridge's, per decision)
            from isaac_bridge.tok_priv import PRIV
            kw['priv'] = PRIV
            p_rows = np.zeros(W, PRIV)
        actor_hash = hashlib.sha1()
        roll = TT.Rollout(W, capacity, len(model.heads), 0, True, tcfg, **kw)
        book = TT.Episodes(W, rw, True, run=True, items=True)
        pend = np.zeros((W, len(model.heads)), np.int64)
        kdt = torch.bfloat16 if args.temporal_kv == 'bf16' else torch.float32
        st = TP.TemporalState(model, W, 'cpu', kdt)
        model.eval()
        for rows in seq:
            idx = np.arange(W, dtype=np.int64)
            go = book.arrive(roll, rows, idx, pend)
            a_, sub = idx[go], rows[go]
            if not len(a_):
                continue
            b = TP.to_batch(sub, pend[a_], 'cpu')
            for k_ in sorted(b):   # every tensor the actor's forward takes, byte by byte
                actor_hash.update(k_.encode())
                actor_hash.update(b[k_].numpy().tobytes())
            td = dict(mode='step', state=st, slot=torch.from_numpy(a_))
            with torch.no_grad():
                acts, logp, v, vg = model.act(b, need_vg=True, temporal=td)
            acts_n = acts.numpy()
            actor_hash.update(logp.numpy().tobytes())
            actor_hash.update(acts_n.tobytes())
            s_kw = {}
            if o.priv:
                p_rows['map'] = (p_rng.random((W,) + p_rows['map'].shape[1:]) < 0.02) * 1
                p_rows['map_ver'] = np.arange(W) * 0 + rows['t'][0] // 8
                p_rows['hp'] = p_rng.random(p_rows['hp'].shape)
                p_rows['scal'] = p_rng.random(p_rows['scal'].shape)
                p_rows['items'] = p_rng.integers(0, 1024, p_rows['items'].shape)
                s_kw['priv'] = p_rows[a_]
            TT.store(roll, a_, sub, pend[a_], acts_n, logp.numpy(), v.numpy(), 0, None, vg.numpy(),
                     TP.temporal_out(td['out']), **s_kw)
            pend[a_] = acts_n
        res['actor_hash'] = actor_hash.hexdigest()
        prof = dict(poll=0.0, book=0.0, act=0.0, reply=0.0, store=0.0, polls=1, calls=1, rows=1, eager=0, act_busy=0.0,
                    calls_busy=1, latency=0.0, snap_calls=0, snap_miss=0, cf_snaps=0)
        worker = dict(wait_s=0.0, step_s=0.0, encode_s=0.0, fork_s=0.0, teach_s=0.0)
        return dict(roll=roll, prof=prof, worker=worker, finished=list(book.finished), t_start=time.perf_counter(),
                    collect_s=0.0, cycle_s=1.0, wait_learner_s=0.0, teach=(np.zeros(0, TT.ROW), np.zeros((0, 16))))

    def z_ref(m, roll, args):
        """The start-of-update pass's z, gn per (worker, decision), from the update's start weights."""
        out_ = {}
        cap = roll.cap
        for w in range(roll.n):
            L_ = int(roll.length[w])
            flat = w * cap + np.arange(L_, dtype=np.int64)
            t_r = TT.temporal_rows(roll, flat, cap, 'cpu')
            b = TP.to_batch(roll.rows[w, :L_], roll.pending[w, :L_], 'cpu')
            d_ = dict(mode='stored', o=t_r['o'], o_prev=t_r['o_prev'], has=t_r['has'], keep_zg=True)
            with torch.no_grad():
                m(b, need_vg=True, temporal=d_)
            z_, g_ = d_['zg_in']
            for t in range(L_):
                out_[(w, int(roll.rows[w, t]['t']), int(roll.rows[w, t]['episode']))] = (z_[t].numpy().copy(), g_[t].numpy().copy())
        return out_

    def fake_run_bench(d):
        model, args = d['model'], d['args']
        job = collect(model, args, args.rollout // args.workers * 4 + 64)
        res['before'] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        res['rollout'] = {k: getattr(job['roll'], k).copy() for k in ('rows', 'reward', 'terminal', 'boot', 'length',
                                                                      'open', 'actions', 'logp', 'value')}
        for name in ('close_hurt', 'close_done', 'close_ev'):
            if getattr(job['roll'], name, None) is not None:
                res['rollout'][name] = getattr(job['roll'], name).copy()
        # (the rollout's arrays stay as they are: learn() reads them only)
        model0 = copy.deepcopy(model).eval() if o.dump_z else None
        torch.manual_seed(13)
        np.random.seed(13)
        t0 = time.perf_counter()
        for _ in range(o.updates):
            d['learn'](job)
        res['learn_s'] = (time.perf_counter() - t0) / o.updates
        res['weights'] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if o.dump_z:
            res['z_ref'] = z_ref(model0, job['roll'], args)
        res['opt'] = d['opt'].state_dict()
        d['L']['stop'] = True
        return True
    BL.load_job = fake_load_job
    BL.run_bench = fake_run_bench
    BL.JobSampler.lane_rows = None
    TT.main(argv)
    with open(run_dir / 'progress.csv', newline='') as f:
        res['progress'] = list(csv.DictReader(f))
    res['last'] = str(run_dir / 'checkpoints' / 'last.pt')
    torch.save(res, out / 'result.pt')
    print('driver done', len(res['progress']), 'updates', round(res['learn_s'], 3), 's per update', flush=True)


if __name__ == '__main__':
    main()
