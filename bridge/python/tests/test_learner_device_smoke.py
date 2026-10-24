"""Smoke test of B38 (2026-10-17, train_tok --learner-device): the learner on a GPU of its own.

Plain script (no game; the CUDA parts run where there are GPUs):
  cd rl/bridge/python && python tests/test_learner_device_smoke.py
Checks:
  (a) CPU: train_tok.mirror_temporal (the hand-off's copy of a rollout's temporal tensors): the learner's chunk
      forward over the mirror's view == over the rollout itself, bit for bit (rollouts A and B of test_temporal_smoke's
      sequence: a carried context, rooms carried, B's pool grown; padded / fast paths, parts), with the mirror's rows
      past the copied ones (and its scratch row) poisoned with NaN: no unassigned row is read; one mirror dict reused
      across rollouts of other pool sizes (reallocated), the view's t_s None, the rollout's own tensors untouched, the
      bytes = (n * P + t_used) rows of K / V + (t_obase + n * P + t_used) rows of o;
  (b) two GPUs: train_tok.hand_off from cuda:0 to cuda:1 on the stream pair, a learner stream on cuda:1 waiting on
      the event: the copied rows equal the rollout's; temporal_inputs + room_table on the cuda:1 view == on the
      cuda:0 rollout (same kernels, same values);
  (c) two GPUs: train_tok.weight_mover / move_weights (take_weights' flat path): the actor's parameters and buffers
      (cuda:0) equal the learner's (cuda:1) bit for bit after the move, in place (the same tensors), twice in a row
      (the buffers reused); the per-tensor fallback gives the same;
  (d) Linux, B38 step 2 (--learner-process 1): rollouts made with train_tok.SharedArrays (memfd shared memory) hold
      the same values as numpy's (the alloc hook changes nothing); a spawned process maps them (attach_arrays,
      rollout_view with the CPU tensors) and its learner chunk forward over them == the parent's, bit for bit; its
      writes are seen by the parent;
  (e) Linux, two GPUs, B38 step 2: a spawned process (on cuda:1) receives the rollouts' cuda:0 tensors and the
      weights' cuda:0 buffer through CUDA IPC: its hand_off to cuda:1 copies the rows exactly and its room tables ==
      the parent's on cuda:0; it publishes another set of weights into the buffer and sets an interprocess event, after
      which the parent's foreach copy (weight_mover without a learner side, as take_weights) gives exactly those
      weights. (The failure paths of the process mode, the learner's error message, the actor's end, SIGINT, are
      tested on the real trainer: rl/bridge/ops/hpc_b38.sh fail / kill / smoke.)
"""
import copy
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_temporal_smoke as TS   # noqa: E402  (its path set-up, the sequence, the actor and learner helpers)
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


def poisoned(roll):
    """A mirror dict with the rollout's shapes, every value NaN (rows the copy does not write stay NaN)."""
    return {name: torch.full(getattr(roll, name).shape, float('nan'), dtype=getattr(roll, name).dtype)
            for name in ('t_kv', 't_o')}


def expect_bytes(roll):
    k = roll.n * roll.tP + roll.t_used
    return k * roll.t_kv[0].numel() * roll.t_kv.element_size() + \
        (roll.t_obase + k) * roll.t_o[0].numel() * roll.t_o.element_size()


CFG_ON = dict(TS.SMALL, memory=TS.DM, **TS.BASE, **TS.FUSED, **TS.TK)


def base_model():
    torch.manual_seed(5)
    m = TP.TokPolicy(**CFG_ON).eval()
    TS.perturb_all(m, 6)
    return m


def shared_rolls(model, sa):
    """test_temporal_smoke.new_rolls with the shared allocator."""
    ta = model.tatt
    tcfg = dict(P=ta.W - 1, L=ta.L, D=ta.D, R=ta.R, kv='fp32', device='cpu')
    return (TT.Rollout(TS.NW, 40, len(model.heads), model.memory, True, dict(tcfg, rows=None), alloc=sa),
            TT.Rollout(TS.NW, 40, len(model.heads), model.memory, True, dict(tcfg, rows=5), alloc=sa))


def _child_cpu(q, pid, descs, scals, tdevs):
    """(d) in a spawned process: the shared rollouts' learner outputs; a write the parent must see."""
    keep = []
    m = base_model()
    res = {}
    for k in sorted(descs):
        v = TT.rollout_view(TT.attach_arrays(pid, descs[k], keep), scals[k], tdevs[k])
        res[k] = [x.detach().numpy().copy() for x in TS.learner_outputs(m, v, 'fast', TS.TT_CH)[0]]
        v.reward[0, 0] = 42.0 + k
    q.put(res)


def _child_cuda(q, ev, pid, descs, scals, tdevs, fa):
    """(e) in a spawned process on cuda:1: hand_off of the IPC rollouts, room tables, weights published into fa."""
    d1 = torch.device('cuda', 1)
    torch.cuda.set_device(d1)
    keep, mirror = [], {}
    streams = (torch.cuda.Stream(device=0), torch.cuda.Stream(device=d1))
    ls = torch.cuda.Stream(device=d1)
    m1 = base_model().to(d1)
    out = {}
    for k in sorted(descs):
        v = TT.rollout_view(TT.attach_arrays(pid, descs[k], keep), scals[k], tdevs[k])
        view, ev_c, nb = TT.hand_off(v, mirror, d1, streams)
        kk = v.n * v.tP + v.t_used
        plan = TP.temporal_chunks(v.length, np.arange(v.cap)[None] < v.length[:, None], TS.TT_CH, TS.TT_MEM)
        with torch.no_grad(), torch.cuda.stream(ls):
            ls.wait_event(ev_c)
            tab = m1.tatt.room_table(view.t_o, TP.temporal_inputs(view, plan['tch']))
            out[k] = dict(kv=view.t_kv[:kk].cpu().numpy(), o=view.t_o[:v.t_obase + kk].cpu().numpy(), nb=nb,
                          tab=[(a.cpu().numpy(), b.cpu().numpy()) for a, b in tab])
    mw = base_model()   # (perturbed on the CPU, as the parent's reference: the CPU generator)
    TS.perturb_all(mw, 77)
    m1 = mw.to(d1)
    params = list(m1.parameters()) + list(m1.buffers())
    with torch.no_grad():
        fb = torch.cat([b.detach().reshape(-1).view(torch.uint8) for b in params])
        fa.copy_(fb, non_blocking=True)
    torch.cuda.synchronize(d1)
    q.put(out)
    ev.set()


def main():
    if sys.platform.startswith('linux') and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        TT.ipc_safe_allocator()   # (as the trainer with --learner-process 1: shared CUDA tensors in ordinary segments)
    rng = np.random.default_rng(38)
    m = base_model()
    seq = TS.make_seq(rng)
    pend = [M.synth_pending(rng, TS.NW, m.heads) for _ in seq]
    st = TP.TemporalState(m, TS.NW, 'cpu', torch.float32)
    ra, rb = TS.new_rolls(m)
    TS.run_actor(m, seq, pend, TS.DM, st, (ra, rb))

    # ------------------------------------------------------------ (a) the mirror's view == the rollout (CPU)
    worst, finite, nb_ok, same_src = 0.0, True, True, True
    for roll in (ra, rb):
        kv0, o0, s0 = roll.t_kv, roll.t_o, roll.t_s
        kv_c, o_c = kv0.clone(), o0.clone()
        view, nb = TT.mirror_temporal(roll, poisoned(roll), 'cpu')
        nb_ok &= nb == expect_bytes(roll) and view.t_s is None and view.t_kv is not kv0
        same_src &= roll.t_kv is kv0 and roll.t_o is o0 and roll.t_s is s0 and torch.equal(kv0, kv_c) and \
            torch.equal(o0, o_c)
        for path in ('padded', 'fast'):
            for micro in (None, TS.TT_CH):
                o_r, flat_r, _ = TS.learner_outputs(m, roll, path, micro)
                o_v, flat_v, _ = TS.learner_outputs(m, view, path, micro)
                finite &= all(bool(torch.isfinite(x).all()) for x in o_v)
                worst = max(worst, 0.0 if all(torch.equal(x, y) for x, y in zip(o_r, o_v)) else
                            TS.maxd(o_r, o_v)) if np.array_equal(flat_r, flat_v) else float('inf')
    check('(a) learner chunks over the mirror == over the rollout, bit for bit (A and B, padded / fast, parts; the '
          'rows not copied NaN)', worst == 0.0 and finite, f'max |diff| {worst:.1e}, finite {finite}')
    check('(a) bytes = the used rows of K / V and o; t_s not in the view; the rollout untouched', nb_ok and same_src)
    mirror = {}
    shapes = []
    for roll in (ra, rb, ra):   # (A and B have other pool sizes: B grew)
        view, _ = TT.mirror_temporal(roll, mirror, 'cpu')
        shapes.append(tuple(view.t_kv.shape))
        o_r = TS.learner_outputs(m, roll, 'fast', TS.TT_CH)[0]
        o_v = TS.learner_outputs(m, view, 'fast', TS.TT_CH)[0]
        worst = max(worst, 0.0 if all(torch.equal(x, y) for x, y in zip(o_r, o_v)) else TS.maxd(o_r, o_v))
    check('(a) one mirror dict across rollouts of other pool sizes (reallocated each time), still exact',
          worst == 0.0 and shapes[0] != shapes[1] and shapes[0] == shapes[2], f'shapes {shapes}')

    # ------------------------------------------------------------ (d) shared rollouts in a spawned process (Linux)
    linux = sys.platform.startswith('linux') and Path('/proc/self/fd').is_dir()
    srolls = None
    if linux:
        sa = TT.SharedArrays()
        srolls = shared_rolls(m, sa)
        rng_d = np.random.default_rng(38)
        seq_d = TS.make_seq(rng_d)
        pend_d = [M.synth_pending(rng_d, TS.NW, m.heads) for _ in seq_d]
        TS.run_actor(m, seq_d, pend_d, TS.DM, TP.TemporalState(m, TS.NW, 'cpu', torch.float32), srolls)
        same_arr = all(np.array_equal(getattr(r1, k_), getattr(r2, k_))
                       for r1, r2 in zip((ra, rb), srolls) for k_, v_ in vars(r1).items() if isinstance(v_, np.ndarray)) \
            and all(torch.equal(getattr(r1, k_), getattr(r2, k_)) for r1, r2 in zip((ra, rb), srolls)
                    for k_ in ('t_kv', 't_s', 't_o'))
        check('(d) rollouts on SharedArrays == on numpy (every array and tensor after the same sequence)', same_arr)
        mpc = torch.multiprocessing.get_context('spawn')
        q = mpc.Queue()
        descs = {k: sa.describe(r) for k, r in enumerate(srolls)}
        scals = {k: TT.rollout_scalars(r) for k, r in enumerate(srolls)}
        tdevs = {k: (r.t_kv, r.t_o) for k, r in enumerate(srolls)}
        pr = mpc.Process(target=_child_cpu, args=(q, os.getpid(), descs, scals, tdevs))
        pr.start()
        res = q.get(timeout=600)
        pr.join(60)
        d_out = 0.0
        for k, r in enumerate(srolls):
            ref = [x.detach().numpy() for x in TS.learner_outputs(m, r, 'fast', TS.TT_CH)[0]]
            d_out = max([d_out] + [float(np.abs(a - b).max()) if a.shape == b.shape else float('inf')
                                   for a, b in zip(ref, res[k])])
        seen = [float(r.reward[0, 0]) for r in srolls]
        check('(d) spawned process: learner chunks over the mapped rollouts == the parent\'s, bit for bit; its writes '
              'seen here', d_out == 0.0 and seen == [42.0, 43.0] and pr.exitcode == 0,
              f'max |diff| {d_out:.1e}, writes {seen}, exit {pr.exitcode}')
        for r in srolls:
            r.reward[0, 0] = 0.0
    else:
        print('(d) / (e) skipped: not Linux (memfd, /proc)')

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print('(b) / (c) / (e) skipped: fewer than two GPUs')
        print('FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed')
        return 1 if FAIL else 0
    d0, d1 = torch.device('cuda', 0), torch.device('cuda', 1)
    torch.cuda.set_device(d0)

    # ------------------------------------------------------------ (b) hand_off cuda:0 -> cuda:1
    hand = (torch.cuda.Stream(device=d0), torch.cuda.Stream(device=d1))
    lstream = torch.cuda.Stream(device=d1)
    mirror = {}
    m0 = copy.deepcopy(m).to(d0)
    m1 = copy.deepcopy(m).to(d1)
    rows_ok, tab_d = True, 0.0
    for roll in (ra, rb, ra):
        r0 = copy.copy(roll)
        r0.t_kv, r0.t_s, r0.t_o = roll.t_kv.to(d0), roll.t_s.to(d0), roll.t_o.to(d0)
        torch.cuda.synchronize(d0)
        view, ev, nb = TT.hand_off(r0, mirror, d1, hand)
        k = roll.n * roll.tP + roll.t_used
        with torch.cuda.stream(lstream):
            lstream.wait_event(ev)
            kv_l = view.t_kv[:k].cpu()
            o_l = view.t_o[:roll.t_obase + k].cpu()
        rows_ok &= torch.equal(kv_l, roll.t_kv[:k]) and torch.equal(o_l, roll.t_o[:roll.t_obase + k]) and \
            view.t_kv.device == d1 and nb == expect_bytes(roll)
        plan = TP.temporal_chunks(roll.length, np.arange(roll.cap)[None] < roll.length[:, None], TS.TT_CH,
                                  TS.TT_MEM)
        with torch.no_grad():
            with torch.cuda.stream(lstream):
                t1 = m1.tatt.room_table(view.t_o, TP.temporal_inputs(view, plan['tch']))
                t1 = [(a.cpu(), b.cpu()) for a, b in t1]
            t0 = m0.tatt.room_table(r0.t_o, TP.temporal_inputs(r0, plan['tch']))
            t0 = [(a.cpu(), b.cpu()) for a, b in t0]
        tab_d = max([tab_d] + [float((a - c).abs().max()) for (a, b), (c, e) in zip(t0, t1)] +
                    [float((b - e).abs().max()) for (a, b), (c, e) in zip(t0, t1)])
    torch.cuda.synchronize(d1)
    check('(b) hand_off cuda:0 -> cuda:1: the copied rows == the rollout\'s (A, B, A again: reallocated)', rows_ok)
    check('(b) the room table from the cuda:1 view == from the cuda:0 rollout', tab_d < 1e-6, f'max |diff| {tab_d:.1e}')

    # ------------------------------------------------------------ (c) take_weights' flat path
    learner = copy.deepcopy(m).to(d1)
    actor = copy.deepcopy(m).to(d0)
    TS.perturb_all(learner, 77)
    a_params = list(actor.parameters()) + list(actor.buffers())
    l_params = list(learner.parameters()) + list(learner.buffers())
    ptrs = [a.data_ptr() for a in a_params]
    wm = TT.weight_mover(a_params, d1)
    astream = torch.cuda.Stream(device=d0)
    ok_ = wm is not None
    for rep in range(2):
        if rep == 1:
            TS.perturb_all(learner, 78)
        with torch.no_grad(), torch.cuda.stream(astream):
            TT.move_weights(wm, l_params, lstream)
        astream.synchronize()
        ok_ &= all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(a_params, l_params))
    ok_ &= [a.data_ptr() for a in a_params] == ptrs
    check('(c) move_weights: the actor\'s tensors == the learner\'s, bit for bit, in place, twice', ok_,
          f'{len(a_params)} tensors, {sum(wm["sizes"]) / 2 ** 20:.2f} MiB' if wm else 'no mover')
    actor2 = copy.deepcopy(m).to(d0)
    with torch.no_grad(), torch.cuda.stream(astream):
        for a, b in zip(list(actor2.parameters()) + list(actor2.buffers()), l_params):
            a.copy_(b)
    astream.synchronize()
    check('(c) the per-tensor fallback gives the same', all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(
        list(actor2.parameters()) + list(actor2.buffers()), a_params)))

    # ------------------------------------------------------------ (e) process mode over CUDA IPC (Linux, two GPUs)
    if srolls is not None:
        mpc = torch.multiprocessing.get_context('spawn')
        q, ev = mpc.Queue(), mpc.Event()
        g_rolls = []
        for r in srolls:   # (the actor's rollouts on cuda:0, host arrays shared)
            r.t_kv, r.t_s, r.t_o = r.t_kv.to(d0), r.t_s.to(d0), r.t_o.to(d0)
            g_rolls.append(r)
        torch.cuda.synchronize(d0)
        actor3 = copy.deepcopy(m).to(d0)
        a3 = list(actor3.parameters()) + list(actor3.buffers())
        wm3 = TT.weight_mover(a3, None)
        descs = {k: sa.describe(r) for k, r in enumerate(g_rolls)}
        scals = {k: TT.rollout_scalars(r) for k, r in enumerate(g_rolls)}
        tdevs = {k: (r.t_kv, r.t_o) for k, r in enumerate(g_rolls)}
        pr = mpc.Process(target=_child_cuda, args=(q, ev, os.getpid(), descs, scals, tdevs, wm3['fa']))
        pr.start()
        got_ev = False
        for _ in range(600):
            got_ev = ev.wait(1.0)
            if got_ev or not pr.is_alive():
                break
        out = q.get(timeout=60) if got_ev else {k: dict(kv=None, o=None, nb=-1, tab=[]) for k in range(len(g_rolls))}
        with torch.no_grad():
            torch._foreach_copy_(wm3['av'], list(wm3['fa'].split(wm3['sizes'])))
        torch.cuda.synchronize(d0)
        pr.join(60)
        rows_ok, tab_d = got_ev, 0.0
        for k, r in enumerate(g_rolls):
            kk = r.n * r.tP + r.t_used
            rows_ok &= out[k]['kv'] is not None and np.array_equal(out[k]['kv'], r.t_kv[:kk].cpu().numpy()) and \
                np.array_equal(out[k]['o'], r.t_o[:r.t_obase + kk].cpu().numpy()) and out[k]['nb'] == expect_bytes(r)
            plan = TP.temporal_chunks(r.length, np.arange(r.cap)[None] < r.length[:, None], TS.TT_CH, TS.TT_MEM)
            with torch.no_grad():
                t0 = m0.tatt.room_table(r.t_o, TP.temporal_inputs(r, plan['tch']))
            for (a, b), (c, e) in zip(t0, out[k]['tab']):
                tab_d = max(tab_d, float(np.abs(a.cpu().numpy() - c).max()), float(np.abs(b.cpu().numpy() - e).max()))
        check('(e) learner process: hand_off of the IPC rollouts (cuda:0 -> cuda:1) copies the rows exactly; its room '
              'tables == the parent\'s', rows_ok and tab_d < 1e-6 and pr.exitcode == 0,
              f'max |diff| {tab_d:.1e}, exit {pr.exitcode}')
        ref = base_model()
        TS.perturb_all(ref, 77)
        w_ok = got_ev and all(torch.equal(a.cpu(), b) for a, b in zip(a3, list(ref.parameters()) + list(ref.buffers())))
        check('(e) weights published into the IPC buffer, then the event: the actor\'s foreach copy gives them exactly',
              w_ok, f'{len(a3)} tensors')
    print('FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
