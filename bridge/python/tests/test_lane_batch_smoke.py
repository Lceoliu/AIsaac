"""CPU smoke test of B36 (2026-10-16): the lanes' service batched (isaac_bridge/tok_lanes.py LaneService; train_tok
--cf-lane-wait / --cf-lane-quiet / --cf-lane-async / --cf-lane-ukeys / --cf-lane-buckets).

No game, no GPU needed (the service's eager path; the GraphActor path needs CUDA). Plain script:
  cd rl/bridge/python && python tests/test_lane_batch_smoke.py
Checks:
  (a) N lanes requesting at once, served in ONE call, over several rounds with lanes joining / leaving and branch
      begins (a snapshot from the bank, a reset), eager path, float32: the call == the policy itself on that batch
      with the lanes' temporal slots and keyed uniforms, bit for bit (actions, log-probability, value, value_global,
      the weights' version, the echo, lane_t (o_prev, has, o), every per-slot tensor of the lanes' state after);
      vs each lane served alone (N one-row calls): the same actions, version and echo, and the floats (logits too)
      equal up to float32 summation order (bit equality across batch sizes is not possible on the CPU: a one-row
      matmul takes another kernel, a plain nn.Linear differs by ~5e-7); a memory model (GRU table, the begin from
      lane_h, h_used) likewise;
  (b) lane_uniforms: a function of (sequence number, lane) only, inside (0, 1), mean ~ 1/2;
  (c) the wait window: window_due's cases; LaneService.due with the "every active lane waiting" rule; tick() does not
      serve inside the window and serves all waiting lanes in one call at its end; --cf-lane-wait 0 serves at once;
      serve_wait (the actor waiting for the learner);
  (d) stale requests: the echo is the number read when the request was taken (a new child's request written while
      the call was in flight gets an answer that its LanePolicy skips, then its own); a pending GRU begin is not
      overwritten by a stale answer; a lane listed twice (a stale and a new message) is one row;
  (e) train_tok wires the service (tick in the loop, quiesce before take_weights, serve_wait while waiting for the
      learner, the flags) and has no per-lane path left;
  (f) CUDA only: the lane GraphActor through the service, asynchronous (in flight after submit, answered at the
      event), fold off / on with row buckets == the service's eager path (the same actions with the keyed uniforms,
      values, lane_t and lane states within 1e-3); tick() leaves a call in flight and answers it at a later pass;
  info: (--gpu-bench, CUDA) the same through the lane GraphActor;
  info: CPU per-decision cost of one eager lane call with N = 1 / 16 / 32 lanes (C99's sizes: width 192 x 4 layers,
      temporal 256 x 4, window 256; indicative only: the GPU's fixed cost per call is what batching saves).
"""
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_memory_smoke as M   # noqa: E402  (its path set-up: the package, the gymnasium stand-in)
import test_fused_tokens_smoke as FT   # noqa: E402
import test_temporal_smoke as TS   # noqa: E402

import numpy as np   # noqa: E402
import torch   # noqa: E402

from isaac_bridge import tok_lanes as TL   # noqa: E402
from isaac_bridge import tok_policy as TP   # noqa: E402
from isaac_bridge import tok_teacher2 as T2   # noqa: E402
from isaac_bridge.tok_cf import t_split   # noqa: E402
from isaac_bridge.tok_obs import ROW   # noqa: E402
from isaac_bridge.tok_sampler import TokSampler   # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


class FakeSampler:
    """The sampler's lane side (TokSampler's lane blocks and lane_ready; its take_lanes) without workers or pipes:
    request() is what a child's LanePolicy.request + the sampler's _lane_message do."""
    take_lanes = TokSampler.take_lanes

    def __init__(self, n, D=0, mem=0):
        self.lane_rows = np.zeros(n, ROW)
        self.lane_io = np.zeros((n, T2.LANE_IO), np.int32)
        self.lane_out = np.zeros((n, T2.LANE_OUT), np.float64)
        self.lane_h = np.zeros((n, mem + 1), np.float32) if mem else None
        self.lane_t = np.zeros((n, 2 * D + 2), np.float32) if D else None
        self.lane_ready = []
        self.lane_first_t = self.lane_last_t = 0.0
        self.replies = []
        self.on_reply = None

    def arrive(self, i, now=None):
        now = time.perf_counter() if now is None else now
        if not self.lane_ready:
            self.lane_first_t = now
        self.lane_last_t = now
        self.lane_ready.append(int(i))

    def request(self, i, row, pend, seq, tbegin=0, now=None):
        self.lane_rows[i] = row
        self.lane_io[i, :5] = pend
        self.lane_io[i, 10] = seq
        if tbegin:
            self.lane_io[i, 11] = tbegin
        self.arrive(i, now)

    def poll_lanes(self):
        return len(self.lane_ready)

    def wait_lanes(self, timeout):
        return bool(self.lane_ready)

    def reply_lanes(self, idx):
        self.replies.append([int(i) for i in idx])
        if self.on_reply is not None:
            self.on_reply(idx)


def temporal_model(seed=3, **over):
    cfg = dict(TS.SMALL, memory=0, **TS.BASE, **TS.FUSED, **TS.TK)
    cfg.update(over)
    torch.manual_seed(seed)
    m = TP.TokPolicy(**cfg).eval()
    TS.perturb_all(m, seed + 1)
    return m


def answers(F, idx):
    """Everything a lane's child reads after its answer, per lane in idx."""
    return dict(io=F.lane_io[idx, 5:10].copy(), out=F.lane_out[idx, :5].copy(),
                t=F.lane_t[idx].copy() if F.lane_t is not None else None,
                h=F.lane_h[idx].copy() if F.lane_h is not None else None)


def same_state(a, b):
    return all(torch.equal(x, y) for (x, _), (y, _) in zip(a._tensors(), b._tensors()))


def state_diff(a, b, slots=None):
    """The largest difference over every per-slot tensor of two TemporalStates (slots: those slots only, e.g. without
    the scratch slot a GraphActor's padded rows write to)."""
    sel = (lambda x, ax: x) if slots is None else (lambda x, ax: x.narrow(ax, 0, slots))
    return max(float((sel(x, ax).float() - sel(y, ax).float()).abs().max())
               for (x, ax), (y, _) in zip(a._tensors(), b._tensors()))


def test_batched_equals_single():
    from isaac_bridge.tok_cf import t_pack
    m = temporal_model()
    cols = int(sum(m.heads))
    rng = np.random.default_rng(21)
    n_l, D = 8, TS.TK['temporal_width']
    bank = TP.TemporalState(m, 6, 'cpu', torch.float32)   # a bank with some history in its rows
    with torch.no_grad():
        for t in range(5):
            m(TP.to_batch(FT.realistic_rows(rng, 6), M.synth_pending(rng, 6, m.heads), 'cpu'), need_vg=True,
              temporal=dict(mode='step', state=bank, slot=torch.arange(6)))
    st_b, st_s, st_r = (TP.TemporalState(m, n_l, 'cpu', torch.float32) for _ in range(3))
    F_b, F_s = FakeSampler(n_l, D), FakeSampler(n_l, D)
    svc_b = TL.LaneService(F_b, m, 'cpu', len(m.heads), tstate=st_b, bank=bank, gtok=True, wait=0.0,
                           version=lambda: 7)
    svc_s = TL.LaneService(F_s, m, 'cpu', len(m.heads), tstate=st_s, bank=bank, gtok=True, wait=0.0,
                           version=lambda: 7)
    # rounds: (lanes requesting, {lane: tbegin}); a begin is the bank row + 1, -1 a reset state
    rounds = [([0, 1, 2, 3, 4, 5], {0: 1, 1: 3, 2: -1, 3: 5, 4: 2, 5: 6}), ([0, 1, 2, 3, 4, 5], {}),
              ([1, 3, 5], {}), ([0, 2, 3, 4, 6, 7], {6: 4, 7: -1}), ([0, 1, 2, 3, 4, 5, 6, 7], {2: 2}),
              ([7, 0], {}), ([0, 1, 2, 3, 4, 5, 6, 7], {})]
    w = dict(exact=True, calls=True, act=True, lg=0.0, out=0.0, t=0.0, st=0.0)
    for k, (lanes, begins) in enumerate(rounds):
        rows = FT.realistic_rows(rng, len(lanes))
        rows['first'] = 0
        pend = M.synth_pending(rng, len(lanes), m.heads)
        seqs = rng.integers(1, 2 ** 31 - 1, len(lanes))
        order = np.argsort(lanes)   # (the service answers in lane order)
        L = np.asarray(lanes, np.int64)[order]
        R, P, S = rows[order], pend[order], seqs[order]
        # the reference: the policy itself, one batch, the lanes' slots and keyed uniforms (what one call must be)
        for i, b_ in begins.items():
            if b_ > 0:
                st_r.copy_slot_([b_ - 1], [i], other=bank)
            else:
                st_r.reset_slot_([i])
        sl = torch.from_numpy(L)
        tw_b, tw_s = (TP.TemporalState(m, n_l, 'cpu', torch.float32) for _ in range(2))
        for tw in (tw_b, tw_s):
            tw.copy_slot_(range(n_l), range(n_l), other=st_r)
        op_r = st_r.prev_o.index_select(0, sl).numpy().copy()
        has_r = (st_r.steps.index_select(0, sl) > 0).numpy() & (R['first'] == 0)
        td = dict(mode='step', state=st_r, slot=sl)
        with torch.no_grad():
            a_r, lp_r, v_r, vg_r = m.act(TP.to_batch(R, P, 'cpu'), u=torch.from_numpy(TL.lane_uniforms(S, L, cols)),
                                         need_vg=True, temporal=td)
            # the logits, one call for the N lanes vs N one-lane steps (twin states)
            lg_b = m(TP.to_batch(R, P, 'cpu'), need_vg=True, temporal=dict(mode='step', state=tw_b, slot=sl))[0]
            lg_s = torch.cat([m(TP.to_batch(R[j:j + 1], P[j:j + 1], 'cpu'), need_vg=True,
                                temporal=dict(mode='step', state=tw_s, slot=sl[j:j + 1]))[0] for j in range(len(L))])
        w['lg'] = max(w['lg'], float((lg_b - lg_s).abs().max()))
        t_r = t_pack(op_r, has_r, td['out']['o'].numpy())
        # the service: all at once vs one at a time
        c0 = svc_b.stat['calls']
        for j, i in enumerate(lanes):
            F_b.request(i, rows[j], pend[j], seqs[j], begins.get(i, 0))
        svc_b.submit()
        svc_b.finish()
        w['calls'] &= svc_b.stat['calls'] == c0 + 1 and F_b.replies[-1] == sorted(lanes)
        for j, i in enumerate(lanes):
            F_s.request(i, rows[j], pend[j], seqs[j], begins.get(i, 0))
            svc_s.submit()
            svc_s.finish()
        A, B = answers(F_b, L), answers(F_s, L)
        w['exact'] &= (np.array_equal(A['io'], a_r.numpy()) and np.array_equal(A['out'][:, 0], v_r.double().numpy())
                       and np.array_equal(A['out'][:, 1], lp_r.double().numpy())
                       and np.array_equal(A['out'][:, 3], vg_r.double().numpy())
                       and np.array_equal(A['out'][:, 4], S.astype(np.float64)) and np.all(A['out'][:, 2] == 7)
                       and np.array_equal(A['t'], t_r) and same_state(st_b, st_r))
        w['act'] &= np.array_equal(A['io'], B['io']) and np.array_equal(A['out'][:, [2, 4]], B['out'][:, [2, 4]])
        w['out'] = max(w['out'], float(np.abs(A['out'][:, [0, 1, 3]] - B['out'][:, [0, 1, 3]]).max()))
        w['t'] = max(w['t'], float(np.abs(A['t'] - B['t']).max()))
        w['st'] = max(w['st'], state_diff(st_b, st_s))
        if k == 0:   # the begins are taken and cleared; o_prev / has of a reset lane are zero / False
            op, has, o = t_split(A['t'], D)
            j2 = list(L).index(2)
            check('(a) begins cleared, a reset lane steps without history (has False, o_prev 0)',
                  not F_b.lane_io[:, 11].any() and not has[j2] and not op[j2].any() and np.abs(o).sum() > 0)
    check('(a) one call per round, every waiting lane answered', w['calls'])
    check("(a) the batched call == the policy on that batch with the lanes' slots and keyed uniforms, bit for bit "
          "(actions, log-probability, value, value_global, version, echo, lane_t, the lanes' state after)", w['exact'])
    check('(a) batched (one call) vs single (N one-lane calls): the same actions, version and echo', w['act'])
    tol = 2e-5
    check('(a) batched vs single: logits, log-probability / value / value_global, lane_t, the state after: '
          'float32 summation order only', max(w['lg'], w['out'], w['t'], w['st']) < tol,
          f"max |diff| logits {w['lg']:.1e}, outputs {w['out']:.1e}, lane_t {w['t']:.1e}, state {w['st']:.1e}; "
          'CPU matmul is not batch-invariant (one row: gemv) - a plain Linear differs by ~5e-7 too')

    # a memory model: the GRU table, the begin from lane_h, h_used
    cfg = dict(TS.SMALL, memory=TS.DM, **TS.BASE)
    torch.manual_seed(9)
    mm = TP.TokPolicy(**cfg).eval()
    TS.perturb_all(mm, 10)
    cols = int(sum(mm.heads))
    tab_b, tab_s, tab_r = (torch.zeros(n_l, TS.DM) for _ in range(3))
    G_b, G_s = FakeSampler(n_l, mem=TS.DM), FakeSampler(n_l, mem=TS.DM)
    sv_b = TL.LaneService(G_b, mm, 'cpu', len(mm.heads), table=tab_b, wait=0.0)
    sv_s = TL.LaneService(G_s, mm, 'cpu', len(mm.heads), table=tab_s, wait=0.0)
    ok, act, worst = True, True, 0.0
    for k in range(4):
        lanes = [0, 2, 3, 5, 6] if k % 2 == 0 else [1, 2, 3, 4, 5, 6, 7]
        rows = FT.realistic_rows(rng, len(lanes))
        pend = M.synth_pending(rng, len(lanes), mm.heads)
        seqs = rng.integers(1, 2 ** 31 - 1, len(lanes))
        h0 = rng.normal(size=(len(lanes), TS.DM)).astype(np.float32)
        beg = [i for i in lanes if k == 0 or i == 4]   # branch begins (LanePolicy.begin)
        for G in (G_b, G_s):
            for j, i in enumerate(lanes):
                if i in beg:
                    G.lane_h[i, :TS.DM], G.lane_h[i, TS.DM] = h0[j], 1.0
        for j, i in enumerate(lanes):
            if i in beg:
                tab_r[i] = torch.from_numpy(h0[j])
        L = np.asarray(sorted(lanes), np.int64)
        o_ = np.argsort(lanes)
        h_in = tab_r[torch.from_numpy(L)].clone()
        with torch.no_grad():
            a_r, lp_r, v_r, hn = mm.act(TP.to_batch(rows[o_], pend[o_], 'cpu'),
                                        u=torch.from_numpy(TL.lane_uniforms(seqs[o_], L, cols)), mem=h_in)[:4]
        tab_r[torch.from_numpy(L)] = hn
        for j, i in enumerate(lanes):
            G_b.request(i, rows[j], pend[j], seqs[j])
        sv_b.submit()
        sv_b.finish()
        for j, i in enumerate(lanes):
            G_s.request(i, rows[j], pend[j], seqs[j])
            sv_s.submit()
            sv_s.finish()
        A, B = answers(G_b, L), answers(G_s, L)
        ok &= (np.array_equal(A['io'], a_r.numpy()) and np.array_equal(A['out'][:, 0], v_r.double().numpy())
               and np.array_equal(A['out'][:, 1], lp_r.double().numpy())
               and np.array_equal(A['h'][:, :TS.DM], h_in.numpy()) and not A['h'][:, TS.DM].any()
               and torch.equal(tab_b, tab_r))
        act &= np.array_equal(A['io'], B['io'])
        worst = max(worst, float(np.abs(A['out'][:, :2] - B['out'][:, :2]).max()), float(np.abs(A['h'] - B['h']).max()),
                    float((tab_b - tab_s).abs().max()))
    check('(a) memory model: the batched call == the policy on the batch (begin states from lane_h, h_used, the GRU '
          'table), bit for bit; vs single: the same actions, floats within 2e-5', ok and act and worst < tol,
          f'max |diff| vs single {worst:.1e}')


def test_uniforms():
    u = TL.lane_uniforms([5, 5, 6], [1, 2, 1], 16)
    u2 = TL.lane_uniforms([6], [1], 16)
    big = TL.lane_uniforms(np.arange(1, 2001), np.arange(2000) % 64, 16)
    check('(b) lane_uniforms: keyed by (sequence number, lane) only, in (0, 1), mean ~ 1/2',
          np.array_equal(u[2:3], u2) and not np.array_equal(u[0], u[1]) and float(big.min()) > 0
          and float(big.max()) < 1 and abs(float(big.mean()) - 0.5) < 0.01, f'mean {big.mean():.4f}')


def test_window():
    wd = TL.window_due
    cases = [(wd(0, 0.0, 0.0, 1.0, 0.002, 0.0005), False, 'nothing waiting'),
             (wd(1, 0.0, 0.0, 0.0, 0.0, 0.0), True, 'wait 0: at once'),
             (wd(3, 0.0, 0.0018, 0.0021, 0.002, 0.0005), True, 'the window ended'),
             (wd(3, 0.0, 0.0010, 0.0016, 0.002, 0.0005), True, 'quiet for 0.6 ms'),
             (wd(3, 0.0, 0.0010, 0.0013, 0.002, 0.0005), False, 'inside, arrivals'),
             (wd(3, 0.0, 0.0010, 0.0013, 0.002, 0.0), False, 'quiet off'),
             (wd(3, 0.0, 0.0010, 0.0013, 0.002, 0.0005, active=3), True, 'every active lane waiting'),
             (wd(3, 0.0, 0.0010, 0.0013, 0.002, 0.0005, active=5), False, 'two active lanes still out')]
    bad = [d for got, want, d in cases if got != want]
    check('(c) window_due cases', not bad, ', '.join(bad))
    m = temporal_model(4)
    n_l = 8
    F = FakeSampler(n_l, TS.TK['temporal_width'])
    st = TP.TemporalState(m, n_l, 'cpu', torch.float32)
    svc = TL.LaneService(F, m, 'cpu', len(m.heads), tstate=st, gtok=True, wait=0.002, quiet=0.0005)
    now = 100.0
    svc.seen[:] = -1e9
    svc.seen[[0, 1, 2]] = now - 0.01   # three lanes active lately
    F.lane_ready, F.lane_first_t, F.lane_last_t = [0, 1], now - 0.0012, now - 0.0002
    d1 = svc.due(now)
    F.lane_ready = [0, 1, 2]
    d2 = svc.due(now)
    svc.seen[2] = now - 1.0   # lane 2 gone quiet long ago: only 0 and 1 are active
    F.lane_ready = [0, 1]
    d3 = svc.due(now)
    check('(c) due: waits for an active lane, not when every active lane waits, not for a long-quiet one',
          (d1, d2, d3) == (False, True, True), str((d1, d2, d3)))
    # tick: a real-time window (a long one, so that the timing is not flaky)
    rng = np.random.default_rng(5)
    svc = TL.LaneService(F, m, 'cpu', len(m.heads), tstate=st, gtok=True, wait=0.25, quiet=0.08)
    svc.seen[:] = time.perf_counter() + 100.0   # every lane active (for the test's whole span)
    F.lane_ready, F.replies = [], []
    rows = FT.realistic_rows(rng, 4)
    pend = M.synth_pending(rng, 4, m.heads)
    F.request(3, rows[0], pend[0], 11)
    F.request(1, rows[1], pend[1], 12)
    early = svc.tick()
    time.sleep(0.03)
    F.request(5, rows[2], pend[2], 13)
    mid = svc.tick()
    time.sleep(0.12)   # quiet for > 80 ms: the window ends
    late = svc.tick()
    check('(c) tick: nothing inside the window, then ONE call for all waiting lanes', (early, mid, late) ==
          (False, False, True) and F.replies == [[1, 3, 5]] and svc.stat['calls'] == 1, f'{(early, mid, late)} '
          f'{F.replies}')
    svc.seen[:] = time.perf_counter() + 100.0
    F.request(2, rows[3], pend[3], 14)
    t0 = time.perf_counter()
    while not svc.tick():
        time.sleep(0.005)
    took = time.perf_counter() - t0
    check('(c) the window\'s end at most --cf-lane-wait after the first request', 0.07 <= took < 0.4 and
          F.replies[-1] == [2], f'{took * 1000:.0f} ms')
    svc0 = TL.LaneService(F, m, 'cpu', len(m.heads), tstate=st, gtok=True, wait=0.0)
    F.request(4, rows[0], pend[0], 15)
    check('(c) --cf-lane-wait 0: served at the first tick', svc0.tick() and F.replies[-1] == [4])
    F.request(6, rows[1], pend[1], 16)
    F.request(7, rows[2], pend[2], 17)
    svc.wait = 0.01
    t0 = time.perf_counter()
    r = svc.serve_wait(0.005)
    check('(c) serve_wait: the window waited out, one call, answered', r and F.replies[-1] == [6, 7] and
          time.perf_counter() - t0 < 0.5 and svc.fly is None)
    F.lane_ready = []
    check('(c) serve_wait without requests: nothing', not svc.serve_wait(0.0))


def test_stale():
    from multiprocessing import Pipe
    m = temporal_model(6)
    n_l = 4
    F = FakeSampler(n_l, TS.TK['temporal_width'])
    st = TP.TemporalState(m, n_l, 'cpu', torch.float32)
    svc = TL.LaneService(F, m, 'cpu', len(m.heads), tstate=st, gtok=True, wait=0.0)
    rng = np.random.default_rng(8)
    rows = FT.realistic_rows(rng, 3)
    pend = M.synth_pending(rng, 3, m.heads)
    ours, theirs = Pipe()
    F.on_reply = lambda idx: [ours.send_bytes(b'a') for i in idx if int(i) == 2]
    # child A's request is taken (the call "in flight"); A is killed; child B starts on the same lane and asks
    F.request(2, rows[0], pend[0], 1111)
    svc.submit()
    stop = [False]
    pol = T2.LanePolicy(theirs, F.lane_io, F.lane_out, 2, lambda: stop[0], lane_t=F.lane_t)
    F.lane_rows[2] = rows[1]
    pol.request(F.lane_rows[2:3], tuple(int(v) for v in pend[1]))
    ours.recv_bytes()   # (the sampler's read of B's message)
    F.arrive(2)
    got = []
    th = threading.Thread(target=lambda: got.append(pol.answer()))
    th.start()
    svc.finish()   # A's answer: echo 1111, B's number is another
    time.sleep(0.2)
    waiting = th.is_alive() and F.lane_out[2, 4] == 1111
    svc.submit()   # B's request: its own row and number
    svc.finish()
    th.join(5)
    # B's answer == lane 2 served alone from A's state after A's step
    check('(d) a stale answer (taken before the next child asked) carries the old number and is skipped',
          waiting and bool(got) and F.lane_out[2, 4] == pol.seq and got[0][0] == tuple(int(v) for v in
                                                                                      F.lane_io[2, 5:10]),
          f'waiting {waiting}, echo {F.lane_out[2, 4]:.0f} seq {pol.seq}')
    # a lane listed twice (a stale message and the new one): one row
    F.request(1, rows[2], pend[2], 2222)
    F.arrive(1)
    F.arrive(3)
    F.request(3, rows[0], pend[0], 3333)
    c0, r0 = svc.stat['calls'], svc.stat['rows']
    svc.submit()
    svc.finish()
    check('(d) a lane listed twice is one row', svc.stat['rows'] - r0 == 2 and svc.stat['calls'] - c0 == 1 and
          F.replies[-1] == [1, 3])
    # memory: a stale answer does not overwrite a pending begin
    cfg = dict(TS.SMALL, memory=TS.DM, **TS.BASE)
    mm = TP.TokPolicy(**cfg).eval()
    G = FakeSampler(n_l, mem=TS.DM)
    tab = torch.zeros(n_l, TS.DM)
    sv = TL.LaneService(G, mm, 'cpu', len(mm.heads), table=tab, wait=0.0)
    G.request(1, rows[0], pend[0], 5)
    sv.submit()
    h0 = rng.normal(size=TS.DM).astype(np.float32)
    G.lane_h[1, :TS.DM], G.lane_h[1, TS.DM] = h0, 1.0   # child B's begin, while A's call is in flight
    sv.finish()
    kept = np.array_equal(G.lane_h[1, :TS.DM], h0) and G.lane_h[1, TS.DM] == 1.0
    G.request(1, rows[1], pend[1], 6)
    sv.submit()
    sv.finish()
    check('(d) memory: a pending begin survives a stale answer, then starts B\'s branch',
          kept and torch.allclose(torch.from_numpy(G.lane_h[1, :TS.DM]), torch.from_numpy(h0)) and
          G.lane_h[1, TS.DM] == 0.0)


def test_source():
    src = (HERE.parent / 'train_tok.py').read_text(encoding='utf-8')
    # (B38 step 5: the loop's condition grew a parenthesis)
    loop = src[re.search(r'\n        while \(?handed < target_decisions', src).start():]
    wl = src[src.index('    def wait_learner():'):src.index('    aux = sampler.aux')]
    check('(e) train_tok: LaneService built, tick in the loop, quiesce before take_weights, serve_wait in wait_learner',
          'LaneService(sampler, model, device, nh' in src and 'lanes.tick()' in loop and
          bool(re.search(r'lanes\.quiesce\(\)\s*\n\s*take_weights\(\)', loop)) and 'lanes.serve_wait(' in wl and
          bool(re.search(r'lanes\.quiesce\(\)\s*\n\s*take_weights\(\)', wl)) and 'lanes.idle(' in loop)
    check('(e) train_tok: the flags; no per-lane path left',
          all(f in src for f in ('--cf-lane-wait', '--cf-lane-quiet', '--cf-lane-async', '--cf-lane-ukeys',
                                 '--cf-lane-buckets')) and 'def serve_lanes' not in src and
          'sampler.take_lanes' not in src)
    pol = (HERE.parent / 'isaac_bridge' / 'tok_policy.py').read_text(encoding='utf-8')
    check('(e) GraphActor: ukeys in submit / _pack_fold / act',
          'def submit(self, rows, pending, widx=None, reset=None, tdest=None, ukeys=None)' in pol and
          'def _pack_fold(self, rows, pending, widx, reset, ukeys=None)' in pol and
          'self.c_cu[:n] = crn_uniforms(ukeys[0], ukeys[1]' in pol)


def test_cuda():
    """(f) CUDA only: the lane GraphActor through the service (asynchronous: submit, then the answer at the event) ==
    the service's eager path on the same requests (keyed uniforms: the same actions), fold off / on with row buckets;
    tick() leaves a call in flight and answers it at a later pass."""
    cuda = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = temporal_model(12).to(cuda)
    rng = np.random.default_rng(31)
    n_l, D, nh = 8, TS.TK['temporal_width'], len(m.heads)
    ta = m.tatt
    bank = TP.TemporalState(m, 6, cuda, torch.float32)
    with torch.no_grad():
        for t in range(4):
            m(TP.to_batch(FT.realistic_rows(rng, 6), M.synth_pending(rng, 6, m.heads), cuda), need_vg=True,
              temporal=dict(mode='step', state=bank, slot=torch.arange(6, device=cuda)))
    rounds = [([0, 1, 2, 3, 4, 5], {0: 1, 1: 3, 2: -1, 3: 5, 4: 2, 5: 6}), ([1, 3, 5], {}),
              ([0, 2, 3, 4, 6, 7], {6: 4, 7: -1}), ([0, 1, 2, 3, 4, 5, 6, 7], {2: 2}), ([7], {})]
    for fold in (False, True):
        st_g = TP.TemporalState(m, n_l, cuda, torch.float32)
        st_e = TP.TemporalState(m, n_l, cuda, torch.float32)
        tdest = dict(kv=torch.zeros((n_l, ta.L, 2, ta.D), device=cuda), s=torch.zeros((n_l, ta.D), device=cuda),
                     o=torch.zeros((n_l, ta.D), device=cuda), obase=0)
        actor = TP.GraphActor(m, n_l, 32, stream=torch.cuda.Stream(), crn=False, temporal=st_g, fold=fold,
                              row_sizes=[2, 4] if fold else None)
        F_g, F_e = FakeSampler(n_l, D), FakeSampler(n_l, D)
        svc_g = TL.LaneService(F_g, m, cuda, nh, actor=actor, tstate=st_g, bank=bank, tdest=tdest, gtok=True,
                               wait=0.0, version=lambda: 3)
        svc_e = TL.LaneService(F_e, m, cuda, nh, tstate=st_e, bank=bank, gtok=True, wait=0.0, version=lambda: 3)
        ok_a, flew = True, True
        worst = dict(out=0.0, t=0.0, st=0.0)
        for lanes, begins in rounds:
            rows = FT.realistic_rows(rng, len(lanes))
            rows['first'] = 0
            rows['n_ent'] = np.minimum(rows['n_ent'], 32)
            pend = M.synth_pending(rng, len(lanes), m.heads)
            seqs = rng.integers(1, 2 ** 31 - 1, len(lanes))
            for F in (F_g, F_e):
                for j, i in enumerate(lanes):
                    F.request(i, rows[j], pend[j], seqs[j], begins.get(i, 0))
            n_rep = len(F_g.replies)
            svc_g.submit()
            flew &= svc_g.fly is not None and not svc_g.fly['ready'] and len(F_g.replies) == n_rep
            svc_g.finish()
            svc_e.submit()
            svc_e.finish()
            L = np.asarray(sorted(lanes), np.int64)
            A, B = answers(F_g, L), answers(F_e, L)
            ok_a &= np.array_equal(A['io'], B['io']) and np.array_equal(A['out'][:, [2, 4]], B['out'][:, [2, 4]])
            worst['out'] = max(worst['out'], float(np.abs(A['out'][:, [0, 1, 3]] - B['out'][:, [0, 1, 3]]).max()))
            worst['t'] = max(worst['t'], float(np.abs(A['t'] - B['t']).max()))
            worst['st'] = max(worst['st'], state_diff(st_g, st_e, n_l))   # (the scratch slot: the graphs' padding)
        check(f'(f) CUDA lane GraphActor via the service (async) == the eager service: actions, version, echo; '
              f'values / lane_t / lane states close [fold {int(fold)}]',
              ok_a and max(worst.values()) < 1e-3 and flew, f"actions {ok_a}, max |diff| outputs {worst['out']:.1e}, "
              f"lane_t {worst['t']:.1e}, states {worst['st']:.1e}, in flight after submit {flew}")
        # tick: the call left in flight, answered at a later pass
        F_g.replies = []
        F_g.request(5, rows[0], pend[0], 99)
        first = svc_g.tick()
        in_flight = svc_g.fly is not None
        t0 = time.perf_counter()
        while not F_g.replies and time.perf_counter() - t0 < 5:
            svc_g.tick()
        check(f'(f) CUDA tick: submitted and left in flight, answered by a later pass [fold {int(fold)}]',
              first and F_g.replies == [[5]] and F_g.lane_out[5, 4] == 99, f'in flight after the first tick '
              f'{in_flight}')


def bench_gpu(reps=40):
    """--gpu-bench (CUDA): the lane GraphActor through the service (fold, row buckets 16 / 32 of 64, C99's sizes, no
    compile): ms per call (submit to answer) and per decision for N = 1 / 16 / 32 lanes; indicative (a shared GPU)."""
    cuda = torch.device('cuda')
    cfg = dict(width=192, layers=4, heads=4, memory=0, **TS.BASE, **TS.FUSED)
    cfg.update(temporal=True, temporal_width=256, temporal_layers=4, temporal_heads=4, temporal_window=256,
               temporal_rooms=128)
    torch.manual_seed(1)
    m = TP.TokPolicy(**cfg).to(cuda).eval()
    n_l, ta = 64, m.tatt
    st = TP.TemporalState(m, n_l, cuda, torch.bfloat16)
    tdest = dict(kv=torch.zeros((n_l, ta.L, 2, ta.D), dtype=torch.bfloat16, device=cuda),
                 s=torch.zeros((n_l, ta.D), dtype=torch.bfloat16, device=cuda),
                 o=torch.zeros((n_l, ta.D), device=cuda), obase=0)
    actor = TP.GraphActor(m, n_l, [16, 32, 40], stream=torch.cuda.Stream(priority=-5), crn=False, temporal=st,
                          fold=True, row_sizes=[16, 32])
    F = FakeSampler(n_l, ta.D)
    svc = TL.LaneService(F, m, cuda, len(m.heads), actor=actor, tstate=st, tdest=tdest, gtok=True, wait=0.0)
    rng = np.random.default_rng(3)
    rows = FT.realistic_rows(rng, n_l)
    rows['first'] = 0
    pend = M.synth_pending(rng, n_l, m.heads)
    out = []
    for N in (1, 16, 32, 64):
        ts = []
        for r in range(reps + 3):
            for i in range(N):
                F.request(i, rows[i], pend[i], 1 + r * 64 + i)
            t0 = time.perf_counter()
            svc.submit()
            svc.finish()
            ts.append(time.perf_counter() - t0)
        per_call = float(np.median(ts[3:]))
        out.append(f'N={N}: {per_call * 1e3:.2f} ms/call = {per_call / N * 1e3:.3f} ms/decision')
    print('info  GPU lane call through the service (C99 sizes, fold, buckets 16/32/64, no compile): ' + '; '.join(out),
          flush=True)


def bench(reps=6):
    cfg = dict(width=192, layers=4, heads=4, memory=0, **TS.BASE, **TS.FUSED)
    cfg.update(temporal=True, temporal_width=256, temporal_layers=4, temporal_heads=4, temporal_window=256,
               temporal_rooms=128)
    torch.manual_seed(1)
    m = TP.TokPolicy(**cfg).eval()
    n_l = 32
    rng = np.random.default_rng(3)
    F = FakeSampler(n_l, 256)
    st = TP.TemporalState(m, n_l, 'cpu', torch.float32)
    svc = TL.LaneService(F, m, 'cpu', len(m.heads), tstate=st, gtok=True, wait=0.0)
    rows = FT.realistic_rows(rng, n_l)
    rows['first'] = 0
    pend = M.synth_pending(rng, n_l, m.heads)
    out = []
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    for N in (1, 16, 32):
        ts = []
        for r in range(reps + 1):
            for i in range(N):
                F.request(i, rows[i], pend[i], 1 + r * 64 + i)
            t0 = time.perf_counter()
            svc.submit()
            svc.finish()
            ts.append(time.perf_counter() - t0)
        per_call = float(np.median(ts[1:]))
        out.append(f'N={N}: {per_call * 1e3:.1f} ms/call = {per_call / N * 1e3:.2f} ms/decision')
    print('info  eager CPU lane call (C99 sizes, ' + f'{torch.get_num_threads()} threads): ' + '; '.join(out),
          flush=True)


if __name__ == '__main__':
    t0 = time.perf_counter()
    for fn in (test_batched_equals_single, test_uniforms, test_window, test_stale, test_source):
        fn()
    if torch.cuda.is_available():
        test_cuda()
    else:
        print('skip (f) CUDA: no GPU here', flush=True)
    if '--no-bench' not in sys.argv:
        bench()
    if '--gpu-bench' in sys.argv and torch.cuda.is_available():
        bench_gpu()
    print(f'{len(FAIL)} failed' + (': ' + ', '.join(FAIL) if FAIL else '') + f'  ({time.perf_counter() - t0:.1f} s)')
    sys.exit(1 if FAIL else 0)
