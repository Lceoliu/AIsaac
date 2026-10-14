"""CPU smoke test of B30 (2026-10-14): the temporal transformer (TokPolicy(temporal=True), train_tok --temporal 1,
rl/docs/TEMPORAL_ATTENTION.md): TemporalAttention / TemporalState / the history token / the learner's chunks.

No game, no GPU needed (the GraphActor part runs only where CUDA is available). Plain script:
  cd rl/bridge/python && python tests/test_temporal_smoke.py
Checks:
  (a) temporal off: the policy module == git HEAD's, bit for bit (plain, items + memory, fused C95-like; forward padded /
      packed / fast path, act with uniforms, chunked BPTT), no temporal config key;
  (b) temporal on with zero gates (the other weights random) == off, bit for bit: the actor's steps (forward, act) over
      a sequence with room changes and episode resets, the learner's chunk forward (padded, packed, fast path with math
      attention on / off, several parts), the stored-output mode, temporal=None;
  (c) the actor's step-by-step forward (TemporalState, float32 K / V) == the learner's one-shot chunk forward over the
      same decisions (perturbed weights and gates g1 / g2 / g3, a window of 8 that wraps, room changes incl. one at a
      chunk's first decision and one in a new episode inside a chunk, EV_EXIT, the room cap, an episode reset, a
      rollout hand-over (train_tok Rollout.carry: the context and the rooms of the going episode carried)): logits,
      values, value_global, GRU states; parts of whole chunks; bfloat16 K / V close to float32;
  (d) the chunked learner forward / backward (memory + temporal, fast path) reaches every temporal parameter (relative
      bias, room / stage embeddings, W_r, Q, the history kind, P, P_g, g1 / g2 / g3); at zero gates only the gates get
      a gradient from the temporal branch;
  (e) a synthetic C95-style checkpoint (fused, memory 128, 8 item slots, global token, shared item table, gates of
      C95's size) resumed into a temporal model (default sizes 256 x 4, window 256, rooms 128): only tatt.* missing,
      Adam remapped by name and stepping, and the outputs (actor steps, act, learner chunks) == the checkpoint's model,
      bit for bit;
  (f) eval_tok: a temporal checkpoint builds (build_eval_model), policy_actions with the per-worker CPU state ==
      act step by step; summary key 'temporal';
  (g) CUDA only: GraphActor with a TemporalState == the eager step (fold off / on).
  info: CPU times (indicative only): the learner's chunk forward + backward (256 decisions, fast path) and the actor's
      eager step (16 records), temporal off vs on.
"""
import io
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_memory_smoke as M   # noqa: E402  (its path set-up, synthetic records, HEAD loader)
import test_fused_tokens_smoke as FT   # noqa: E402

import numpy as np   # noqa: E402
import torch   # noqa: E402

from isaac_bridge import tok_policy as TP   # noqa: E402
from isaac_bridge.tok_obs import EV_EXIT   # noqa: E402

FAIL = []
BASE = dict(items=True, charge=True, ent_ext=True, pchar=True)
SMALL = dict(width=64, layers=2, heads=4)
DM = 16
FUSED = dict(item_tokens=True, global_token=True, item_shared=True, item_tokens_fused=True, item_slots=8)
TK = dict(temporal=True, temporal_width=32, temporal_layers=2, temporal_heads=4, temporal_window=8, temporal_rooms=3)
NW = 3   # workers of the sequences
T1, T2 = 22, 40   # a rollout hand-over after step T1 - 1 (that decision open, carried), the sequence's length
TT_CH, TT_MEM = 8, 4   # temporal chunk, GRU chunk
CHANGES = {(0, 3), (0, 6), (0, 8), (0, 14), (0, 18), (0, 25), (0, 29), (0, 33), (1, 5), (1, 14), (1, 16), (1, 30),
           (2, 22), (2, 37)}
RESETS = {(1, 12), (2, 30)}
EXITS = {(1, 20), (2, 9)}


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def make_seq(rng, nw=NW, T=T2, changes=CHANGES, resets=RESETS, exits=EXITS):
    """Records [T][nw]: realistic inventories; the current room as one cell of map channel 3 (it moves at a room
    change), episode starts (first) at step 0 and at `resets`, EV_EXIT (same map cell, next stage) at `exits`."""
    out, room, stage = [], np.zeros(nw, np.int64), np.ones(nw, np.int64)
    for t in range(T):
        r = FT.realistic_rows(rng, nw)
        r['map'] = 0
        r['events'] = 0
        r['first'] = 0
        r['t'] = t
        for w in range(nw):
            if t == 0 or (w, t) in resets:
                r['first'][w] = 1
                room[w], stage[w] = 0, 1
            elif (w, t) in changes:
                room[w] += 1
            elif (w, t) in exits:
                r['events'][w] = EV_EXIT
                stage[w] += 1
            cell = int(room[w] * 7 + w) % 169
            r['map'][w, 3, cell // 13, cell % 13] = 1
            r['stage'][w] = stage[w]
        out.append(r)
    return out


def perturb_all(m, seed, gates=(0.7, 0.6, 0.5)):
    """Random weights everywhere (FT.perturb), the temporal gates set (None: zero), the relative bias random."""
    FT.perturb(m, seed, 0.05)
    with torch.no_grad():
        if m.tatt is not None:
            g = torch.Generator().manual_seed(seed + 1)
            m.tatt.rel.copy_(torch.randn(m.tatt.rel.shape, generator=g) * 0.5)
            for p_, v_ in zip((m.tatt.g_read, m.tatt.g_feed, m.tatt.g_vg), gates or (0.0, 0.0, 0.0)):
                if p_ is not None:
                    p_.fill_(v_)
        if m.memory:
            m.mem_out.weight.normal_(0, 0.1)


def off_twin(m_on, cfg_off):
    """The model without the temporal transformer, with m_on's other weights."""
    m = TP.TokPolicy(**cfg_off).eval()
    state = {k: v for k, v in m_on.state_dict().items() if not k.startswith('tatt.')}
    m.load_state_dict(state, strict=True)
    return m


def run_actor(model, seq, pend, mem_dim, st=None, kv_rolls=None, u=None):
    """The actor's steps over the sequence (all workers each step): per step forward(mem, need_vg, temporal step) and,
    with u, act with those uniforms. kv_rolls: (rollout A, rollout B) of train_tok: decisions stored into A until step
    T1 - 1, then B.carry(A) and B. Returns dict of [T, nw, ...] tensors."""
    import train_tok as TT
    nw = len(seq[0])
    table = torch.zeros(nw, mem_dim)
    log = dict(logits=[], value=[], vg=[], h=[], act=[], logp=[], o=[])
    roll = kv_rolls[0] if kv_rolls else None
    for t, rows in enumerate(seq):
        if kv_rolls and t == T1:
            kv_rolls[1].carry(kv_rolls[0])
            roll = kv_rolls[1]
        b = TP.to_batch(rows, pend[t], 'cpu')
        h_in = table * torch.from_numpy(rows['first'] == 0).float()[:, None]
        td = dict(mode='step', state=st, slot=torch.arange(nw)) if st is not None else None
        kw = dict(temporal=td) if td is not None else {}
        with torch.no_grad():
            lg, v, _, hn, vg = model(b, mem=h_in, need_vg=True, **kw)
            log['logits'].append(lg)
            log['value'].append(v)
            log['vg'].append(vg)
            log['h'].append(hn)
            log['o'].append(td['out']['o'] if td is not None else torch.zeros(nw, 1))
            if roll is not None:
                TT.store(roll, np.arange(nw), rows, pend[t], np.zeros((nw, len(model.heads)), np.int64),
                         np.zeros(nw, np.float32), v.numpy(), 0, h_in.numpy(), vg.numpy(), TP.temporal_numpy(td['out']))
        table = hn.clone()
    if u is not None:   # act with uniforms: a second pass on a fresh state copy
        st2 = TP.TemporalState(model, nw, 'cpu', st.dtype) if st is not None else None
        table = torch.zeros(nw, mem_dim)
        for t, rows in enumerate(seq):
            b = TP.to_batch(rows, pend[t], 'cpu')
            h_in = table * torch.from_numpy(rows['first'] == 0).float()[:, None]
            kw = dict(temporal=dict(mode='step', state=st2, slot=torch.arange(nw))) if st2 is not None else {}
            with torch.no_grad():
                a_, lp_, _, hn, _ = model.act(b, u=u[t], mem=h_in, need_vg=True, **kw)
            log['act'].append(a_)
            log['logp'].append(lp_)
            table = hn.clone()
    return {k: torch.stack(v) for k, v in log.items() if v}


def new_rolls(model, cap=40):
    import train_tok as TT
    ta = model.tatt
    tcfg = dict(P=ta.W - 1, L=ta.L, D=ta.D, R=ta.R, kv='fp32')
    return TT.Rollout(NW, cap, len(model.heads), model.memory, True, tcfg), \
        TT.Rollout(NW, cap, len(model.heads), model.memory, True, tcfg)


def learner_outputs(model, roll, path='padded', micro=None, mode='chunk'):
    """The learner's forward over a rollout's chunks (train_tok's layout: temporal_chunks, temporal_inputs, parts of
    whole chunks): (outputs [rows, ...] list: logits, value, value_global, h_new; flat positions; weights). path:
    padded / packed / fast / fast-nomath; micro: rows per part (None: one part)."""
    import train_tok as TT
    cap = roll.cap
    mask = np.arange(cap)[None] < roll.length[:, None]
    plan = TP.temporal_chunks(roll.length, mask, TT_CH, TT_MEM)
    flat, tch = plan['pos'], plan['tch']
    td = TP.temporal_device(TP.temporal_inputs(roll, tch, roll.tP), 'cpu')
    trow = TT.temporal_rows(roll, flat, cap, 'cpu')
    rows_all = roll.rows.reshape(-1)[flat]
    pend_all = roll.pending.reshape(-1, len(model.heads))[flat]
    h0_all = torch.from_numpy(roll.h.reshape(-1, model.memory)[plan['start']])
    outs = []
    for lo, hi, cids in TT.temporal_bounds(tch, np.arange(len(tch)), micro or 10 ** 9):
        rows = rows_all[lo:hi]
        b = TP.to_batch(rows, pend_all[lo:hi], 'cpu')
        memd = dict(h0=h0_all[lo // TT_MEM:hi // TT_MEM], T=TT_MEM,
                    reset=torch.from_numpy(rows['first'] != 0).view(-1, TT_MEM))
        if mode == 'chunk':
            tk = dict(mode='chunk', plan=TP.temporal_part(td, np.asarray(cids), tch[cids, 1], trow['ep'][lo:hi]),
                      o_prev=trow['o_prev'][lo:hi], has=trow['has'][lo:hi])
        else:
            tk = dict(mode='stored', o=trow['o'][lo:hi], o_prev=trow['o_prev'][lo:hi], has=trow['has'][lo:hi])
        if model.tatt is None:
            tk = None
        kw = dict(temporal=tk) if tk is not None else {}
        model.set_math_attention(path == 'fast')
        if path == 'padded':
            o = model(b, mem=memd, need_vg=True, **kw)
        elif path == 'packed':
            o = model(FT.host_batch(model, b, rows), packed=True, mem=memd, need_vg=True, **kw)
        else:
            o = model(FT.fast_batch(model, b, rows), packed=True, mem=memd, need_vg=True, **kw)
        model.set_math_attention(False)
        outs.append([o[0], o[1], o[-1], o[-2]])
    return [torch.cat(x) for x in zip(*outs)], flat, plan['weight']


def actor_rows(log, flat, cap, offset):
    """The actor's per-step outputs at the rollout's flat positions (rollout decision k = global step k + offset)."""
    w, k = flat // cap, flat % cap
    t = torch.from_numpy(k + offset)
    wt = torch.from_numpy(w)
    return [log['logits'][t, wt], log['value'][t, wt], log['vg'][t, wt], log['h'][t, wt]]


def maxd(xs, ys, sel=None):
    if sel is not None:
        xs, ys = [x[sel] for x in xs], [y[sel] for y in ys]
    return max(float((x.detach().float() - y.detach().float()).abs().max()) for x, y in zip(xs, ys))


def same(xs, ys):
    return all(torch.equal(x, y) for x, y in zip(xs, ys))


def main():
    rng = np.random.default_rng(30)
    dev = torch.device('cpu')

    # ------------------------------------------------------------ (a) off == HEAD
    old = M.head_module()
    cases = (('plain', dict(SMALL)), ('items+memory', dict(SMALL, memory=DM, **BASE)),
             ('fused C95-like', dict(SMALL, memory=DM, **BASE, **FUSED)))
    for tag, cfg in cases:
        torch.manual_seed(1)
        new = TP.TokPolicy(**cfg).eval()
        FT.perturb(new, 2)
        n = 12
        rows = FT.realistic_rows(rng, n)
        b = TP.to_batch(rows, M.synth_pending(rng, n, new.heads), dev)
        u = torch.rand((n, sum(new.heads)))
        check(f'(a) no temporal key in config [{tag}]', not any(k.startswith('temporal') for k in new.config))
        if old is None:
            print('skip (a) HEAD comparison: git HEAD not readable')
            continue
        ref = old.TokPolicy(**cfg).eval()
        ref.load_state_dict(new.state_dict())
        mk = dict(mem=torch.randn(n, DM)) if cfg.get('memory') else {}
        vk = dict(need_vg=True) if cfg.get('global_token') else {}
        with torch.no_grad():
            o_new = [new(b, **mk, **vk), new(FT.host_batch(new, b, rows), packed=True, **mk, **vk),
                     new.act(b, u=u, **mk)]
            o_ref = [ref(b, **mk, **vk), ref(FT.host_batch(ref, b, rows), packed=True, **mk, **vk),
                     ref.act(b, u=u, **mk)]
            new.set_math_attention(True)
            ref.set_math_attention(True)
            o_new.append(new(FT.fast_batch(new, b, rows), packed=True, **mk, **vk))
            o_ref.append(ref(FT.fast_batch(ref, b, rows), packed=True, **mk, **vk))
            if cfg.get('memory'):
                cd = FT.chunk_dict(n, DM)
                o_new.append(new(b, mem=cd, **vk))
                o_ref.append(ref(b, mem=cd, **vk))
        check(f'(a) temporal off == HEAD, every path [{tag}]', all(same(x, y) for x, y in zip(o_new, o_ref)))

    # ------------------------------------------------------------ (b) on, zero gates == off
    cfg_off = dict(SMALL, memory=DM, **BASE, **FUSED)
    cfg_on = dict(cfg_off, **TK)
    for tag, co, cn in (('fused C95-like', cfg_off, cfg_on),
                        ('plain + memory', dict(SMALL, memory=DM), dict(SMALL, memory=DM, **TK))):
        torch.manual_seed(3)
        m_on = TP.TokPolicy(**cn).eval()
        perturb_all(m_on, 4, gates=None)
        m_off = off_twin(m_on, co)
        seq = make_seq(rng)
        pend = [M.synth_pending(rng, NW, m_on.heads) for _ in seq]
        u = [torch.rand((NW, sum(m_on.heads))) for _ in seq]
        if not m_on.global_token:   # (need_vg needs a global token: compare without it)
            check(f'(b) on (zero gates) == off: forward without a global token [{tag}]',
                  _plain_equal(m_on, m_off, seq, pend))
            continue
        st = TP.TemporalState(m_on, NW, 'cpu', torch.float32)
        ra, rb = new_rolls(m_on)
        log_on = run_actor(m_on, seq, pend, DM, st, (ra, rb), u)
        log_off = run_actor(m_off, seq, pend, DM, None, None, u)
        check(f'(b) actor steps: on (zero gates) == off, bit for bit (forward, act) [{tag}]',
              same([log_on[k] for k in ('logits', 'value', 'vg', 'h', 'act', 'logp')],
                   [log_off[k] for k in ('logits', 'value', 'vg', 'h', 'act', 'logp')]))
        check(f'(b) the temporal state moved (rooms made, o non-zero) [{tag}]',
              int(st.nroom.max()) > 0 and float(log_on['o'].abs().max()) > 0, f'rooms {st.nroom.tolist()}')
        ok_l, det = True, []
        for path in ('padded', 'packed', 'fast', 'fast-nomath'):
            for micro in (None, 2 * TT_CH):
                for roll in (ra, rb):
                    o1, _, _ = learner_outputs(m_on, roll, path, micro)
                    o2, _, _ = learner_outputs(m_off, roll, path, micro)
                    if not same(o1, o2):
                        ok_l = False
                        det.append(f'{path}/{micro}: {maxd(o1, o2):.1e}')
        o1, _, _ = learner_outputs(m_on, rb, 'packed', None, mode='stored')
        o2, _, _ = learner_outputs(m_off, rb, 'packed', None)
        check(f'(b) learner chunks: on (zero gates) == off, bit for bit (padded, packed, fast math on / off, parts; '
              f'stored mode) [{tag}]', ok_l and same(o1, o2), ', '.join(det))
        b0 = TP.to_batch(seq[3], pend[3], dev)
        with torch.no_grad():
            check(f'(b) temporal=None == off [{tag}]',
                  same(m_on(b0, mem=torch.ones(NW, DM), need_vg=True), m_off(b0, mem=torch.ones(NW, DM), need_vg=True)))

    # ------------------------------------------------------------ (c) step by step == one-shot chunks
    torch.manual_seed(5)
    m = TP.TokPolicy(**cfg_on).eval()
    perturb_all(m, 6)
    seq = make_seq(rng)
    pend = [M.synth_pending(rng, NW, m.heads) for _ in seq]
    st = TP.TemporalState(m, NW, 'cpu', torch.float32)
    ra, rb = new_rolls(m)
    log = run_actor(m, seq, pend, DM, st, (ra, rb))
    check('(c) the sequence: rooms made, the cap reached, rooms carried, an in-chunk room at a chunk start',
          int(ra.rooms_n.sum()) > 0 and int(st.nroom.max()) == TK['temporal_rooms'] and int(rb.rooms_n.max()) > 0
          and bool((rb.rooms_t[:, :max(int(rb.rooms_n.max()), 1)] == -1).any()),
          f'rooms A {ra.rooms_n.tolist()}, B {rb.rooms_n.tolist()}, state {st.nroom.tolist()}')
    P_ = ra.tP
    made = [int(ra.t_rc[w, P_:P_ + ra.length[w]].sum() + rb.t_rc[w, P_ + 1:P_ + rb.length[w]].sum()) for w in range(NW)]
    check('(c) room tokens made per worker = room changes + EV_EXIT per episode, capped at temporal_rooms',
          made == [3, 4, 3], f'{made}')
    st1 = TP.TemporalState(m, 1, 'cpu', torch.float32)
    rr = make_seq(rng, 1, 6, set(), set(), set())
    cells = [5, None, 5, 9, 9, 9]   # (room A, the map without a current room, A, B, B + EV_EXIT, B)
    for t_, c_ in enumerate(cells):
        rr[t_]['map'][0, 3] = 0
        if c_ is not None:
            rr[t_]['map'][0, 3, c_ // 13, c_ % 13] = 1
    rr[4]['events'][0] = EV_EXIT
    flags = []
    for t_ in range(6):
        td = dict(mode='step', state=st1, slot=torch.zeros(1, dtype=torch.int64))
        with torch.no_grad():
            m(TP.to_batch(rr[t_], M.synth_pending(rng, 1, m.heads), dev), mem=torch.zeros(1, DM), temporal=td)
        flags.append(int(td['out']['create'][0]))
    check('(c) room-change detection: none at the first record or for a map without a current room, one per room '
          'change and per EV_EXIT', flags == [0, 0, 0, 1, 1, 0], f'{flags}')
    errs = {}
    for name, roll, offs in (('A', ra, 0), ('B', rb, T1 - 1)):
        for path in ('padded', 'packed', 'fast'):
            for micro in (None, TT_CH):
                o, flat, wgt = learner_outputs(m, roll, path, micro)
                sel = torch.from_numpy(wgt > 0)
                errs[(name, path, micro)] = maxd(o, actor_rows(log, flat, roll.cap, offs), sel)
    worst = max(errs.values())
    check('(c) actor step by step == learner chunks (logits, value, value_global, GRU state; both rollouts, every path, '
          'parts)', worst < 2e-4, f'max |diff| {worst:.1e}')
    with torch.no_grad():   # the gates matter: the same records with the gates closed differ
        m0 = TP.TokPolicy(**cfg_on).eval()
        m0.load_state_dict(m.state_dict())
        m0.tatt.g_read.zero_()
        m0.tatt.g_feed.zero_()
        m0.tatt.g_vg.zero_()
    log0 = run_actor(m0, seq, pend, DM, TP.TemporalState(m0, NW, 'cpu', torch.float32))
    dg = float((log0['logits'] - log['logits']).abs().max())
    check('(c) the temporal branch changes the outputs once the gates are open', dg > 1e-3, f'max logit change {dg:.3f}')
    st16 = TP.TemporalState(m, NW, 'cpu', torch.bfloat16)
    log16 = run_actor(m, seq, pend, DM, st16)
    d16 = float((log16['logits'] - log['logits']).abs().max())
    check('(c) bfloat16 K / V close to float32 (actor)', d16 < 0.1, f'max logit diff {d16:.3f}')
    m.eval()

    # ------------------------------------------------------------ (d) gradients of the chunked learner
    learner = TP.TokPolicy(**cfg_on)
    learner.load_state_dict(m.state_dict())
    learner.train()

    def grads(model):
        model.zero_grad()
        o, _, wgt = learner_outputs(model, ra, 'fast', TT_CH)
        w_ = torch.from_numpy(wgt)
        ((o[0].pow(2).sum(-1) + o[1].pow(2) + o[2].pow(2)) * w_).sum().backward()
        return {n: (0.0 if p.grad is None else float(p.grad.abs().sum())) for n, p in model.named_parameters()
                if n.startswith('tatt.')}
    g = grads(learner)
    zero = [n for n, v in g.items() if v == 0.0]
    check('(d) the gradient reaches every temporal parameter (fast path, memory + temporal, parts)', not zero,
          f'{len(g)} tensors' + (f', zero: {zero}' if zero else '') + f'; |g rel| {g["tatt.rel"]:.2e}, '
          f'|g room_emb| {g["tatt.room_emb.weight"]:.2e}, |g g_feed| {g["tatt.g_feed"]:.2e}')
    with torch.no_grad():
        learner.tatt.g_read.zero_()
        learner.tatt.g_feed.zero_()
        learner.tatt.g_vg.zero_()
    g0 = grads(learner)
    gates_ = ('tatt.g_read', 'tatt.g_feed', 'tatt.g_vg')
    check('(d) at zero gates: the gates get a gradient, the branch behind them none',
          all(g0[k] > 0 for k in gates_) and all(v == 0.0 for k, v in g0.items() if k not in gates_),
          ', '.join(f'{k} {g0[k]:.2e}' for k in gates_))

    # ------------------------------------------------------------ (e) a C95-style checkpoint resumed with temporal
    _c95_resume(rng)

    # ------------------------------------------------------------ (f) eval_tok
    _eval_path(rng, m, cfg_on)

    # ------------------------------------------------------------ (g) CUDA: GraphActor
    if torch.cuda.is_available():
        _graph_actor(rng)
    else:
        print('skip (g) GraphActor: no CUDA here')

    bench()
    print('RESULT', 'FAIL ' + ', '.join(FAIL) if FAIL else 'all checks passed', flush=True)
    return 1 if FAIL else 0


def _plain_equal(m_on, m_off, seq, pend):
    st = TP.TemporalState(m_on, NW, 'cpu', torch.float32)
    t_on = torch.zeros(NW, DM)
    t_off = torch.zeros(NW, DM)
    ok = True
    for t, rows in enumerate(seq):
        b = TP.to_batch(rows, pend[t], 'cpu')
        keep = torch.from_numpy(rows['first'] == 0).float()[:, None]
        with torch.no_grad():
            o1 = m_on(b, mem=t_on * keep, temporal=dict(mode='step', state=st, slot=torch.arange(NW)))
            o2 = m_off(b, mem=t_off * keep)
        ok &= same(o1, o2)
        t_on, t_off = o1[-1], o2[-1]
    return ok


def _c95_resume(rng):
    """(e) a synthetic C95 checkpoint (default widths, memory 128, fused, 8 slots, global, shared table) -> temporal."""
    from isaac_bridge.tok_policy import main_parameter_names, remap_optimizer_state
    Dm = 128
    cfg = dict(BASE, memory=Dm, **FUSED)
    torch.manual_seed(11)
    src = TP.TokPolicy(**cfg)
    FT.perturb(src, 11)
    with torch.no_grad():   # (C95's gates: fused multipliers around 0.2 / 0.07, trained memory projections)
        src.itok.gate.fill_(0.2)
        src.gtok.gate.fill_(0.07)
        src.mem_out.weight.normal_(0, 0.05)
        src.gtok.mem_in.weight.normal_(0, 0.05)
    opt = torch.optim.Adam(src.main_parameters(), lr=1e-4, eps=1e-5)
    nb = TP.to_batch(FT.realistic_rows(rng, 4), M.synth_pending(rng, 4, src.heads), 'cpu')
    src(nb, mem=torch.zeros(4, Dm), need_vg=True)[-1].sum().backward()
    opt.step()
    buf = io.BytesIO()
    torch.save(dict(model=src.state_dict(), optimizer=opt.state_dict(), config=src.config), buf)
    buf.seek(0)
    ck = torch.load(buf, map_location='cpu')
    src.eval()
    dst = TP.TokPolicy(**dict(ck['config'], temporal=True))
    missing, unexpected = dst.load_state_dict(ck['model'], strict=False)
    osd = ck['optimizer']
    added = remap_optimizer_state(osd, main_parameter_names(list(ck['model'])),
                                  main_parameter_names([k for k, _ in dst.named_parameters()]))
    opt2 = torch.optim.Adam(dst.main_parameters(), lr=1e-4, eps=1e-5)
    opt2.load_state_dict(osd)
    n_t = sum(1 for k, _ in dst.named_parameters() if k.startswith('tatt.'))
    dst.train()
    dst.zero_grad()
    dst(nb, mem=torch.zeros(4, Dm), need_vg=True)[-1].sum().backward()
    opt2.step()
    p0 = dst.main_parameters()[0]
    check('(e) C95 checkpoint -> temporal: only tatt.* missing, Adam remapped by name (new tensors without state) '
          'and stepping', not unexpected and missing and all(k.startswith('tatt.') for k in missing) and
          added == n_t and int(opt2.state[p0]['step']) == 2,
          f'{len(missing)} new tensors, {sum(p.numel() for k, p in dst.named_parameters() if k.startswith("tatt.")) / 1e6:.2f}M '
          f'parameters; config {dict((k, v) for k, v in dst.config.items() if k.startswith("temporal"))}')
    dst.load_state_dict(dict(TP.TokPolicy(**dst.config).state_dict(), **ck['model']), strict=True)   # (fresh tatt)
    dst.eval()
    seq = make_seq(rng, 2, 12, {(0, 4), (1, 7), (0, 9)}, {(1, 9)}, {(0, 6)})
    pend = [M.synth_pending(rng, 2, src.heads) for _ in seq]
    u = [torch.rand((2, sum(src.heads))) for _ in seq]
    import train_tok as TT
    st = TP.TemporalState(dst, 2, 'cpu', torch.float32)
    tcfg = dict(P=dst.tatt.W - 1, L=dst.tatt.L, D=dst.tatt.D, R=dst.tatt.R, kv='fp32')
    ra = TT.Rollout(2, 16, len(dst.heads), Dm, True, tcfg)
    lo = _run2(dst, seq, pend, Dm, st, ra, u)
    lr_ = _run2(src, seq, pend, Dm, None, None, u)
    check('(e) resumed model == the checkpoint\'s, bit for bit: actor steps (forward, act)',
          all(torch.equal(lo[k], lr_[k]) for k in lo))
    mask = np.arange(ra.cap)[None] < ra.length[:, None]
    plan = TP.temporal_chunks(ra.length, mask, 8, 4)
    flat, tch = plan['pos'], plan['tch']
    td = TP.temporal_device(TP.temporal_inputs(ra, tch, ra.tP), 'cpu')
    trow = TT.temporal_rows(ra, flat, ra.cap, 'cpu')
    rows = ra.rows.reshape(-1)[flat]
    b = TP.to_batch(rows, ra.pending.reshape(-1, len(dst.heads))[flat], 'cpu')
    memd = dict(h0=torch.from_numpy(ra.h.reshape(-1, Dm)[plan['start']]), T=4,
                reset=torch.from_numpy(rows['first'] != 0).view(-1, 4))
    tk = dict(mode='chunk', plan=TP.temporal_part(td, np.arange(len(tch)), tch[:, 1], trow['ep']),
              o_prev=trow['o_prev'], has=trow['has'])
    ok = True
    for fast in (False, True):
        for mdl in (dst, src):
            mdl.set_math_attention(fast)
        bb = FT.fast_batch(dst, b, rows) if fast else FT.host_batch(dst, b, rows)
        with torch.no_grad():
            ok &= same(dst(bb, packed=True, mem=memd, need_vg=True, temporal=tk),
                       src(bb, packed=True, mem=memd, need_vg=True))
    for mdl in (dst, src):
        mdl.set_math_attention(False)
    check('(e) resumed model == the checkpoint\'s, bit for bit: learner chunks (packed, fast path)', ok)


def _run2(model, seq, pend, Dm, st, roll, u):
    import train_tok as TT
    nw = len(seq[0])
    table = torch.zeros(nw, Dm)
    out = dict(lg=[], v=[], vg=[], h=[], a=[], lp=[])
    for t, rows in enumerate(seq):
        b = TP.to_batch(rows, pend[t], 'cpu')
        h_in = table * torch.from_numpy(rows['first'] == 0).float()[:, None]
        td = dict(mode='step', state=st, slot=torch.arange(nw)) if st is not None else None
        kw = dict(temporal=td) if td is not None else {}
        with torch.no_grad():
            lg, v, _, hn, vg = model(b, mem=h_in, need_vg=True, **kw)
            if roll is not None:
                TT.store(roll, np.arange(nw), rows, pend[t], np.zeros((nw, len(model.heads)), np.int64),
                         np.zeros(nw, np.float32), v.numpy(), 0, h_in.numpy(), vg.numpy(), TP.temporal_numpy(td['out']))
        for k, x in zip(('lg', 'v', 'vg', 'h'), (lg, v, vg, hn)):
            out[k].append(x)
        table = hn.clone()
    st2 = TP.TemporalState(model, nw, 'cpu', torch.float32) if st is not None else None
    table = torch.zeros(nw, Dm)
    for t, rows in enumerate(seq):
        b = TP.to_batch(rows, pend[t], 'cpu')
        h_in = table * torch.from_numpy(rows['first'] == 0).float()[:, None]
        kw = dict(temporal=dict(mode='step', state=st2, slot=torch.arange(nw))) if st2 is not None else {}
        with torch.no_grad():
            a_, lp_, _, hn = model.act(b, u=u[t], mem=h_in, **kw)
        out['a'].append(a_)
        out['lp'].append(lp_)
        table = hn.clone()
    return {k: torch.stack(v) for k, v in out.items()}


def _eval_path(rng, m, cfg_on):
    """(f) eval_tok with a temporal checkpoint: build_eval_model, policy_actions over steps with the CPU state."""
    import eval_tok as E
    buf = io.BytesIO()
    torch.save(dict(model=m.state_dict(), config=m.config, update=1, decisions=0), buf)
    buf.seek(0)
    ck = torch.load(buf, map_location='cpu')
    em, changed, note = E.build_eval_model(ck, True, torch.device('cpu'))
    check('(f) build_eval_model: a temporal checkpoint builds with every tensor exact',
          em.tatt is not None and not changed and note is None and em.config.get('temporal'))
    seq = make_seq(rng)
    pend = [M.synth_pending(rng, NW, em.heads) for _ in seq]
    t_state = TP.TemporalState(em, NW, 'cpu', torch.float32)
    mem_h, mem_reset = torch.zeros(NW, DM), np.zeros(NW, bool)
    ref_st = TP.TemporalState(em, NW, 'cpu', torch.float32)
    table = torch.zeros(NW, DM)
    ok = True
    for t, rows in enumerate(seq):
        order = [2, 0, 1] if t % 2 else [0, 1, 2]   # (the workers in another order every other step)
        mem_reset[np.flatnonzero(rows['first'])] = True
        a_e = E.policy_actions(em, rows[order], pend[t][order], order, torch.device('cpu'), True, mem_h, mem_reset,
                               t_state)
        b = TP.to_batch(rows, pend[t], 'cpu')
        h_in = table * torch.from_numpy(rows['first'] == 0).float()[:, None]
        with torch.no_grad():
            a_r, _, _, hn = em.act(b, greedy=True, mem=h_in,
                                   temporal=dict(mode='step', state=ref_st, slot=torch.arange(NW)))
        table = hn.clone()
        ok &= np.array_equal(a_e, a_r.numpy()[order])
    d_o = float((t_state.prev_o - ref_st.prev_o).abs().max())   # (rows in another order: rounding only)
    ok &= d_o < 1e-4 and torch.equal(t_state.nroom, ref_st.nroom) and float((mem_h - table).abs().max()) < 1e-4
    check('(f) policy_actions with the per-worker CPU state == act step by step (greedy actions; states up to '
          'rounding: workers reordered)', ok, f'max |o diff| {d_o:.1e}')
    src = (HERE.parent / 'eval_tok.py').read_text(encoding='utf-8')
    check("(f) eval_tok's summary records 'temporal'",
          bool(re.search(r"'item_tokens_fused', 'temporal'\)", src)) and 'TemporalState(model, n, device' in src)


def _graph_actor(rng):
    """(g) CUDA: GraphActor with a TemporalState == the eager step on a copy of the state."""
    cuda = torch.device('cuda')
    cfg = dict(SMALL, memory=DM, **BASE, **FUSED, **TK)
    gm = TP.TokPolicy(**cfg).to(cuda).eval()
    perturb_all(gm, 21)
    seq = make_seq(rng, 4, 10, {(0, 3), (1, 5)}, {(2, 6)}, {(3, 4)})
    for fold in (False, True):
        tab = torch.zeros((6, DM), device=cuda)
        st = TP.TemporalState(gm, 6, cuda, torch.float32)
        actor = TP.GraphActor(gm, 6, 32, stream=torch.cuda.Stream(), crn=False, memory=tab, temporal=st, fold=fold)
        st_e = TP.TemporalState(gm, 6, cuda, torch.float32)
        tab_e = torch.zeros((6, DM), device=cuda)
        widx = np.array([5, 0, 3, 1])
        ok = True
        for t, rows in enumerate(seq):
            rows['n_ent'] = np.minimum(rows['n_ent'], 32)
            pend = M.synth_pending(rng, 4, gm.heads)
            _, lp, val = actor.act(rows, pend, widx, rows['first'] != 0)
            torch.cuda.synchronize()
            wi = torch.from_numpy(widx).to(cuda)
            h_in = tab_e[wi] * torch.from_numpy(rows['first'] == 0).float().to(cuda)[:, None]
            td = dict(mode='step', state=st_e, slot=wi)
            with torch.no_grad():
                _, v_ref, _, hn, _ = gm(TP.to_batch(rows, pend, cuda), mem=h_in, need_vg=True, temporal=td)
            tab_e[wi] = hn
            ok &= np.allclose(val, v_ref.cpu().numpy(), atol=1e-3) and \
                np.allclose(actor.t_out['o'], td['out']['o'].cpu().numpy(), atol=1e-3) and \
                np.array_equal(actor.t_out['nroom'], td['out']['nroom'].cpu().numpy())
        check(f'(g) GraphActor + TemporalState == the eager step (values, o, room counts) [fold {int(fold)}]', ok)


def bench(reps=3):
    """CPU times (indicative only): C95-sized model (width 192, memory 128, fused, 8 slots), temporal off vs on
    (256 x 4, window 256, rooms 128): the learner's chunk of 256 decisions (fast path, 8 GRU chunks, 255 context
    decisions, 12 rooms) forward + backward; the actor's eager step of 16 records."""
    rng = np.random.default_rng(9)
    torch.manual_seed(9)
    cfg = dict(BASE, memory=128, **FUSED)
    n = 256
    rows = FT.realistic_rows(rng, n)
    rows['first'] = 0
    rows['first'][0] = 0
    m_on = TP.TokPolicy(**dict(cfg, temporal=True))
    perturb_all(m_on, 10)
    m_off = off_twin(m_on, cfg)
    ta = m_on.tatt
    P = ta.W - 1
    td = dict(ctx_kv=torch.randn(1, P, ta.L, 2, ta.D), ctx_ep=torch.zeros(1, P, dtype=torch.int64),
              pre_o=torch.randn(1, 10, ta.D), pre_k=torch.arange(10)[None], pre_stage=torch.ones(1, 10, dtype=torch.int64),
              pre_valid=torch.ones(1, 10, dtype=torch.bool), in_o=torch.randn(1, 2, ta.D),
              in_k=torch.tensor([[10, 11]]), in_stage=torch.ones(1, 2, dtype=torch.int64),
              in_at=torch.tensor([[40, 200]]), in_ep=torch.zeros(1, 2, dtype=torch.int64),
              in_valid=torch.ones(1, 2, dtype=torch.bool), pre_n=np.array([10]), in_n=np.array([2]))
    tp = TP.temporal_part(td, np.array([0]), np.array([n]), torch.zeros(n, dtype=torch.int64))
    tk = dict(mode='chunk', plan=tp, o_prev=torch.randn(n, ta.D), has=torch.ones(n, dtype=torch.bool))
    b = TP.to_batch(rows, M.synth_pending(rng, n, m_on.heads), 'cpu')
    memd = dict(h0=torch.zeros(n // 32, 128), T=32, reset=torch.zeros(n // 32, 32, dtype=torch.bool))
    res = {}
    for tag, mdl, kw in (('off', m_off, {}), ('on', m_on, dict(temporal=tk))):
        mdl.train()
        mdl.set_math_attention(True)
        fb = FT.fast_batch(mdl, b, rows)
        ts = []
        for k in range(reps + 1):
            t0 = time.perf_counter()
            mdl.zero_grad()
            o = mdl(fb, packed=True, mem=memd, need_vg=True, **kw)
            (o[0].pow(2).mean() + o[1].pow(2).mean() + o[-1].pow(2).mean()).backward()
            if k:
                ts.append(time.perf_counter() - t0)
        mdl.set_math_attention(False)
        mdl.eval()
        res[tag] = [np.median(ts)]
    k_ = 16
    rows16 = FT.realistic_rows(rng, k_)
    b16 = TP.to_batch(rows16, M.synth_pending(rng, k_, m_on.heads), 'cpu')
    for tag, mdl in (('off', m_off), ('on', m_on)):
        st = TP.TemporalState(m_on, 64, 'cpu', torch.float32) if tag == 'on' else None
        ts = []
        for k in range(reps + 2):
            kw = dict(temporal=dict(mode='step', state=st, slot=torch.arange(k_))) if st is not None else {}
            t0 = time.perf_counter()
            with torch.no_grad():
                mdl.act(b16, mem=torch.zeros(k_, 128), need_vg=True, **kw)
            if k > 1:
                ts.append(time.perf_counter() - t0)
        res[tag].append(np.median(ts))
    print(f'info: CPU, {torch.get_num_threads()} threads, median of {reps} (indicative only): learner chunk of 256 '
          f'decisions fwd+bwd (fast path) off {res["off"][0] * 1000:.0f} ms / on {res["on"][0] * 1000:.0f} ms '
          f'(x{res["on"][0] / res["off"][0]:.2f}); actor eager step of 16 records (64 slots) off '
          f'{res["off"][1] * 1000:.1f} ms / on {res["on"][1] * 1000:.1f} ms (x{res["on"][1] / res["off"][1]:.2f})')


if __name__ == '__main__':
    sys.exit(main())
