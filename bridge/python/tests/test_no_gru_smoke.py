"""CPU smoke test of B32 (2026-10-15): the temporal model (B30 / B31, TokPolicy(temporal=True), train_tok --temporal 1)
WITHOUT the B20 GRU memory (TokPolicy(memory=0); train_tok --temporal 1 without --memory gru, or --memory off on a
memory checkpoint: tok_policy.drop_memory), for the ablation run C98 (C96's checkpoint with the GRU removed).

No game, no GPU needed (the GraphActor part runs only where CUDA is available). Plain script:
  cd rl/bridge/python && python tests/test_no_gru_smoke.py
Checks:
  (a) a temporal model without memory (C96-like fused + global token + shared item table, and a plain one): forward,
      act and the learner's chunks run on the padded, packed and fast paths; temporal_chunks(T_mem=None): the
      temporal chunks alone (no padding, no nested GRU chunks; untrained decisions weight 0); the learner's gradient
      reaches the temporal parameters; train_tok accepts --temporal 1 without --memory gru (source checks);
  (b) the actor step by step (TemporalState, eager, float32 K / V) == the learner's one-shot chunks over the same
      decisions (episode resets, room changes, EV_EXIT, a rollout hand-over with the carried context, minibatches of
      whole chunks in a random order, parts of whole chunks): logits, value, value_global to ~1e-5;
  (c) resume conversion (train_tok --resume with --memory off): a synthetic memory + temporal checkpoint (C96-like,
      gates open, trained-looking mem_out / mem_in) into the no-memory temporal model: only the GRU tensors dropped
      (mem_gru.*, mem_out.*, gtok.mem_in.*), every other tensor exact, Adam remapped by name (the dropped entries
      gone, the others' state carried); outputs == the source model with mem_out (and mem_in) zeroed (actor steps,
      act, stored-mode forward); the outputs differ from the unchanged source (the GRU's contribution);
  (d) models WITH memory: run the existing smoke tests (see the bottom of this docstring); here only the temporal
      chunks with T_mem are checked to be unchanged by B32;
  (e) eval_tok: build_eval_model / policy_actions with the per-worker temporal state and no GRU state (== act step by
      step), --ablate global on a model without mem_in;
  (f) CUDA only: GraphActor with a TemporalState and no memory table == the eager step (fold off / on).
Models with memory (d): tests/test_temporal_smoke.py, test_memory_smoke.py, test_fused_tokens_smoke.py,
test_item_global_smoke.py, test_pickup_smoke.py, test_eval_ablate_smoke.py.
"""
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_memory_smoke as M   # noqa: E402  (its path set-up, synthetic records)
import test_fused_tokens_smoke as FT   # noqa: E402
import test_temporal_smoke as TS   # noqa: E402

import numpy as np   # noqa: E402
import torch   # noqa: E402

from isaac_bridge import tok_policy as TP   # noqa: E402

FAIL = []
SMALL, BASE, FUSED, TK = TS.SMALL, TS.BASE, TS.FUSED, TS.TK
NW, T1 = TS.NW, TS.T1
TT_CH = TS.TT_CH
DM = 16
CFG = dict(SMALL, **BASE, **FUSED, **TK)          # C96-like without the GRU
CFG_PLAIN = dict(SMALL, **TK)                      # no item / global tokens, no GRU
PATHS = ('padded', 'packed', 'fast', 'fast-nomath')


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def outs_of(model, o, vg):
    """(logits, value, value_global or None) of a forward's outputs."""
    return o[0], o[1], (o[-1] if vg else None)


def run_actor(model, seq, pend, st, rolls=None, u=None, mem_dim=0):
    """The actor's steps over the sequence (all workers each step), eager: forward(temporal step) and, with u, act with
    those uniforms on a second state. mem_dim > 0: a memory model (its GRU state carried per worker). rolls: (A, B)
    train_tok Rollouts: decisions stored into A until step T1 - 1, then B.carry(A). Returns dict of [T, nw, ...]."""
    import train_tok as TT
    vg_on = model.gtok is not None
    nw = len(seq[0])
    log = dict(logits=[], value=[], vg=[], act=[], logp=[])
    roll = rolls[0] if rolls else None
    table = torch.zeros(nw, mem_dim)
    for t, rows in enumerate(seq):
        if rolls and t == T1:
            rolls[1].carry(rolls[0])
            roll = rolls[1]
        b = TP.to_batch(rows, pend[t], 'cpu')
        td = dict(mode='step', state=st, slot=torch.arange(nw))
        kw = dict(need_vg=True) if vg_on else {}
        h_in = table * torch.from_numpy(rows['first'] == 0).float()[:, None]
        if mem_dim:
            kw['mem'] = h_in
        with torch.no_grad():
            o = model(b, temporal=td, **kw)
        lg, v, vg = outs_of(model, o, vg_on)
        if mem_dim:
            table = (o[-2] if vg_on else o[-1]).clone()
        log['logits'].append(lg)
        log['value'].append(v)
        log['vg'].append(vg if vg is not None else torch.zeros(nw))
        if roll is not None:
            TT.store(roll, np.arange(nw), rows, pend[t], np.zeros((nw, len(model.heads)), np.int64),
                     np.zeros(nw, np.float32), v.numpy(), 0, h_in.numpy() if mem_dim else None,
                     vg.numpy() if vg is not None else None, TP.temporal_out(td['out']))
    if u is not None:
        st2 = TP.TemporalState(model, nw, 'cpu', st.dtype)
        table = torch.zeros(nw, mem_dim)
        for t, rows in enumerate(seq):
            b = TP.to_batch(rows, pend[t], 'cpu')
            kw = dict(temporal=dict(mode='step', state=st2, slot=torch.arange(nw)))
            if mem_dim:
                kw['mem'] = table * torch.from_numpy(rows['first'] == 0).float()[:, None]
            with torch.no_grad():
                out = model.act(b, u=u[t], **kw)
            if mem_dim:
                table = out[3].clone()
            log['act'].append(out[0])
            log['logp'].append(out[1])
    return {k: torch.stack(v) for k, v in log.items() if v}


def new_rolls(model, cap=40, kv='fp32'):
    """Rollouts A and B (device tensors on the CPU; B's small pool has to grow), without the GRU's Rollout.h."""
    import train_tok as TT
    ta = model.tatt
    tcfg = dict(P=ta.W - 1, L=ta.L, D=ta.D, R=ta.R, kv=kv, device='cpu')
    vg = model.gtok is not None
    return TT.Rollout(NW, cap, len(model.heads), model.memory, vg, dict(tcfg, rows=None)), \
        TT.Rollout(NW, cap, len(model.heads), model.memory, vg, dict(tcfg, rows=5))


def train_mask(roll):
    """gae's mask: every stored decision except a worker's open last one (it only bootstraps)."""
    m = np.arange(roll.cap)[None] < roll.length[:, None]
    for w in range(roll.n):
        if roll.open[w] >= 0:
            m[w, roll.open[w]] = False
    return m


def learner_outputs(model, roll, path='padded', micro=None, order_seed=None, grad=False):
    """train_tok's learner layout without the GRU: temporal_chunks(T_mem=None), temporal_inputs, the room table once,
    a minibatch = whole temporal chunks (in a random order with order_seed), parts of whole chunks (temporal_bounds,
    micro rows at most). Returns (outputs [rows, ...] in the plan's row order: logits, value, value_global; flat;
    weight; plan)."""
    import train_tok as TT
    vg_on = model.gtok is not None
    cap = roll.cap
    plan = TP.temporal_chunks(roll.length, train_mask(roll), TT_CH, None)
    flat, tch = plan['pos'], plan['tch']
    td = TP.temporal_inputs(roll, tch)
    trow = TT.temporal_rows(roll, flat, cap, 'cpu')
    has_all = trow['has'].numpy()
    model.set_math_attention(path == 'fast')
    with torch.no_grad():
        table = model.tatt.room_table(roll.t_o, td)
    model.set_math_attention(False)
    rows_all = roll.rows.reshape(-1)[flat]
    pend_all = roll.pending.reshape(-1, len(model.heads))[flat]
    cids = np.arange(len(tch))
    if order_seed is not None:
        cids = np.random.default_rng(order_seed).permutation(len(tch))
    sel_h = np.concatenate([np.arange(o_, o_ + c_) for o_, c_ in tch[cids, :2]])   # (train_tok's sel_h)
    res = [None, None, None]
    pieces = []
    for lo, hi, cc in TT.temporal_bounds(tch, cids, micro or 10 ** 9):
        part = sel_h[lo:hi]
        rows = rows_all[part]
        b = TP.to_batch(rows, pend_all[part], 'cpu')
        pt = torch.from_numpy(part)
        tk = dict(mode='chunk', plan=TP.temporal_part(td, np.asarray(cc), tch[cc, 1], trow['ep'][pt], table),
                  o_prev=trow['o_prev'][pt], has=trow['has'][pt])
        hs = has_all[part]
        kw = dict(need_vg=True) if vg_on else {}
        model.set_math_attention(path == 'fast')
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            if path == 'padded':
                o = model(b, temporal=tk, **kw)
            elif path == 'packed':
                o = model(FT.host_batch(model, b, rows, hs), packed=True, temporal=tk, **kw)
            else:
                o = model(FT.fast_batch(model, b, rows, hist=hs), packed=True, temporal=tk, **kw)
        model.set_math_attention(False)
        pieces.append((part, outs_of(model, o, vg_on)))
    n = len(flat)
    for j in range(3):
        if pieces[0][1][j] is None:
            continue
        x0 = pieces[0][1][j]
        full = x0.new_zeros((n,) + tuple(x0.shape[1:]))
        idx = torch.from_numpy(np.concatenate([p_ for p_, _ in pieces]))
        full = full.index_copy(0, idx, torch.cat([o_[j] for _, o_ in pieces]))
        res[j] = full
    return res, flat, plan['weight'], plan


def actor_rows(log, flat, cap, offset, vg_on):
    w, k = flat // cap, flat % cap
    t = torch.from_numpy(k + offset)
    wt = torch.from_numpy(w)
    return [log['logits'][t, wt], log['value'][t, wt], log['vg'][t, wt] if vg_on else None]


def maxd(xs, ys):
    return max(float((x.detach().float() - y.detach().float()).abs().max()) for x, y in zip(xs, ys)
               if x is not None and y is not None)


def main():
    rng = np.random.default_rng(32)
    import train_tok as TT

    # ------------------------------------------------------------ (a) runs; the chunk layout; gradients; train_tok
    for tag, cfg in (('C96-like, no GRU', CFG), ('plain, no GRU', CFG_PLAIN)):
        torch.manual_seed(1)
        m = TP.TokPolicy(**cfg).eval()
        TS.perturb_all(m, 2)
        check(f'(a) no memory modules, temporal on [{tag}]',
              m.memory == 0 and not hasattr(m, 'mem_gru') and (m.gtok is None or m.gtok.mem_in is None)
              and 'memory' not in m.config and m.config.get('temporal') is True)
        seq = TS.make_seq(rng)
        pend = [M.synth_pending(rng, NW, m.heads) for _ in seq]
        u = [torch.rand((NW, sum(m.heads))) for _ in seq]
        st = TP.TemporalState(m, NW, 'cpu', torch.float32)
        ra, rb = new_rolls(m)
        log = run_actor(m, seq, pend, st, (ra, rb), u)
        vg_on = m.gtok is not None
        ok_run = all(torch.isfinite(log[k]).all() for k in ('logits', 'value', 'vg', 'logp')) and \
            log['act'].shape[:2] == (len(seq), NW)
        check(f'(a) actor: forward + act (uniforms) step by step run, finite; rooms made; rollouts without h '
              f'[{tag}]', ok_run and int(st.nroom.max()) > 0 and ra.h is None and rb.h is None,
              f'rooms {st.nroom.tolist()}')
        worst, ran = 0.0, []
        for path in PATHS:
            for micro in (None, TT_CH, 2 * TT_CH):
                for name_, roll, offs in (('A', ra, 0), ('B', rb, T1 - 1)):
                    o, flat, wgt, plan = learner_outputs(m, roll, path, micro, order_seed=7)
                    worst = max(worst, maxd(o, actor_rows(log, flat, roll.cap, offs, vg_on)))
                    ran.append(path)
        # (b) is this comparison: report it under (b) for the C96-like model, (a) for both
        check(f'(a)/(b) learner chunks (padded, packed, fast math on / off; one part, parts of 1 and 2 chunks; random '
              f'chunk order) == actor step by step (logits, value' + (', value_global' if vg_on else '') +
              f'; both rollouts, every decision incl. untrained) [{tag}]', worst < 1e-5, f'max |diff| {worst:.1e}')
        # the chunk layout without the GRU
        mask = train_mask(rb)
        plan = TP.temporal_chunks(rb.length, mask, TT_CH, None)
        tch = plan['tch']
        lay_ok = (np.all(tch[:, 1] == tch[:, 4])                              # rows = real decisions: no padding
                  and np.all(tch[:, 0] == np.cumsum(tch[:, 1]) - tch[:, 1])   # contiguous
                  and np.all(tch[:, 3] % TT_CH == 0)                          # cut at 0, T, 2T, ...
                  and len(plan['pos']) == int(tch[:, 1].sum()) == plan['weight'].size
                  and plan['chunks'] == len(tch) == len(plan['start'])
                  and np.array_equal(plan['start'], tch[:, 2] * rb.cap + tch[:, 3])
                  and float(plan['weight'].sum()) == float(mask.sum())
                  and all(plan['weight'][np.flatnonzero(plan['pos'] == w * rb.cap + rb.open[w])].sum() == 0
                          for w in range(rb.n) if rb.open[w] >= 0))
        full = TP.temporal_chunks(rb.length, mask, TT_CH, 4)
        check(f'(a) temporal_chunks(T_mem=None): the temporal chunks alone (rows = decisions, contiguous, cut at '
              f'multiples of --temporal-bptt, no padding, open decisions weight 0); with T_mem 4: padded as before '
              f'[{tag}]', lay_ok and np.array_equal(full['tch'][:, [2, 3, 4]], tch[:, [2, 3, 4]])
              and np.all(full['tch'][:, 1] % 4 == 0),
              f'{len(tch)} chunks, {len(plan["pos"])} rows (GRU layout {len(full["pos"])} rows)')
        # gradients (fast path, parts)
        learner = TP.TokPolicy(**cfg)
        learner.load_state_dict(m.state_dict())
        learner.train()
        learner.zero_grad()
        o, flat, wgt, _ = learner_outputs(learner, ra, 'fast', TT_CH, order_seed=3, grad=True)
        w_ = torch.from_numpy(wgt)
        loss = (o[0].pow(2).sum(-1) + o[1].pow(2) + (o[2].pow(2) if o[2] is not None else 0)) * w_
        loss.sum().backward()
        zero = [n for n, p in learner.named_parameters() if n.startswith('tatt.') and
                (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
        check(f'(a) the learner\'s gradient (fast path, parts, no GRU) reaches every temporal parameter [{tag}]',
              not zero, f'zero: {zero}' if zero else '')
        learner.eval()
    src = (HERE.parent / 'train_tok.py').read_text(encoding='utf-8')
    check('(a) train_tok: --temporal 1 no longer needs --memory gru; chunks without the GRU '
          '(temporal_chunks(..., None)); --memory defaults to the checkpoint\'s; --memory off drops a GRU',
          "needs --memory gru (the learner" not in src and 'T_mem if mem_on else None' in src
          and "p.add_argument('--memory', default=None" in src and "args.memory == 'off'" in src
          and 'drop_memory(learner_model, ck_model)' in src and 'chunked = mem_on or t_on' in src)

    # ------------------------------------------------------------ (c) resume conversion
    _resume_conversion(rng)

    # ------------------------------------------------------------ (d) memory models: the chunk layout unchanged
    L = np.array([19, 0, 7])
    msk = np.arange(24)[None] < L[:, None]
    msk[0, 18] = False
    p4 = TP.temporal_chunks(L, msk, 8, 4)
    exp_tch = [[0, 8, 0, 0, 8], [8, 8, 0, 8, 8], [16, 4, 0, 16, 3], [20, 8, 2, 0, 7]]
    check('(d) temporal_chunks with a GRU (T_mem 4): the B30 layout (padded to whole GRU chunks, nested starts)',
          p4['tch'].tolist() == exp_tch and p4['chunks'] == 7 and len(p4['pos']) == 28
          and p4['start'].tolist() == [0, 4, 8, 12, 16, 48, 52])

    # ------------------------------------------------------------ (e) eval_tok
    _eval_path(rng)

    # ------------------------------------------------------------ (f) CUDA: GraphActor without a memory table
    if torch.cuda.is_available():
        _graph_actor(rng)
    else:
        print('skip (f) GraphActor: no CUDA here')
    print('(d) models with memory: run tests/test_temporal_smoke.py, test_memory_smoke.py, test_fused_tokens_smoke.py, '
          'test_item_global_smoke.py, test_pickup_smoke.py, test_eval_ablate_smoke.py')
    print('RESULT', 'FAIL ' + ', '.join(FAIL) if FAIL else 'all checks passed', flush=True)
    return 1 if FAIL else 0


def _resume_conversion(rng):
    """(c) a synthetic memory + temporal checkpoint (C96-like) -> the no-memory temporal model, as train_tok's resume
    with --memory off: drop_memory, load_state_dict(strict=False), remap_optimizer_state by name."""
    cfg_src = dict(CFG, memory=DM)
    torch.manual_seed(21)
    src = TP.TokPolicy(**cfg_src)
    TS.perturb_all(src, 22)   # (random weights, gates g1 / g2 / g3 open)
    with torch.no_grad():   # (trained-looking GRU projections; mem_out's bias too: it is part of what the GRU adds)
        g = torch.Generator().manual_seed(23)
        src.mem_out.weight.copy_(torch.randn(src.mem_out.weight.shape, generator=g) * 0.1)
        src.mem_out.bias.copy_(torch.randn(src.mem_out.bias.shape, generator=g) * 0.1)
        src.gtok.mem_in.weight.copy_(torch.randn(src.gtok.mem_in.weight.shape, generator=g) * 0.1)
    opt = torch.optim.Adam(src.main_parameters(), lr=1e-4, eps=1e-5)
    nb = TP.to_batch(FT.realistic_rows(rng, 4), M.synth_pending(rng, 4, src.heads), 'cpu')
    for _ in range(2):
        opt.zero_grad()
        o = src(nb, mem=torch.randn(4, DM), need_vg=True)
        (o[0].pow(2).mean() + o[1].pow(2).mean() + o[-1].pow(2).mean()).backward()
        opt.step()
    buf = io.BytesIO()
    torch.save(dict(model=src.state_dict(), optimizer=opt.state_dict(), config=src.config, update=5, decisions=1,
                    seconds=1.0), buf)
    buf.seek(0)
    ck = torch.load(buf, map_location='cpu')
    src.eval()
    # (train_tok: the checkpoint's config with --memory off: TokPolicy(memory=0, the rest as saved))
    cfg_dst = {k: v for k, v in ck['config'].items() if k != 'memory'}
    dst = TP.TokPolicy(**cfg_dst)
    state, names, note = TP.drop_memory(dst, ck['model'])
    print('    log line:', note)
    missing, unexpected = dst.load_state_dict(state, strict=False)
    expect = sorted(k for k in ck['model'] if k.startswith(('mem_gru.', 'mem_out.', 'gtok.mem_in.')))
    exact = all(torch.equal(v, ck['model'][k]) for k, v in dst.state_dict().items())
    check('(c) only the GRU tensors dropped (mem_gru.*, mem_out.*, gtok.mem_in.*), nothing missing / unexpected, every '
          'other tensor exact; the input state dict unchanged',
          sorted(names) == expect and len(expect) == 8 and not missing and not unexpected and exact
          and set(dst.state_dict()) == set(ck['model']) - set(expect) and all(k in ck['model'] for k in expect)
          and note is not None and 'dropped 8 tensors' in note,
          f'dropped {names}')
    osd = ck['optimizer']
    old_names = TP.main_parameter_names(list(ck['model']))
    new_names = TP.main_parameter_names([k for k, _ in dst.named_parameters()])
    st_old = {nm: osd['state'][i] for nm, i in zip(old_names, osd['param_groups'][0]['params']) if i in osd['state']}
    added = TP.remap_optimizer_state(osd, old_names, new_names)
    opt2 = torch.optim.Adam(dst.main_parameters(), lr=1e-4, eps=1e-5)
    opt2.load_state_dict(osd)
    by_name = dict(zip(new_names, dst.main_parameters()))
    with_state = {nm for nm in new_names if by_name[nm] in opt2.state}
    carried = all(torch.equal(opt2.state[by_name[nm]]['exp_avg'], st_old[nm]['exp_avg']) and
                  torch.equal(opt2.state[by_name[nm]]['exp_avg_sq'], st_old[nm]['exp_avg_sq']) for nm in new_names
                  if nm in st_old)
    dst.train()
    dst.zero_grad()
    o = dst(nb, need_vg=True)
    (o[0].pow(2).mean() + o[-1].pow(2).mean()).backward()
    opt2.step()
    p0 = dst.main_parameters()[0]
    check('(c) Adam remapped by name: the GRU\'s entries gone, every other parameter\'s state carried, stepping',
          added == 0 and len(new_names) == len(old_names) - len(expect) and with_state == set(st_old) - set(expect)
          and not (set(expect) - set(st_old)) and carried and int(opt2.state[p0]['step']) == 3,
          f'{len(old_names)} -> {len(new_names)} parameters with state')
    dst.load_state_dict(state, strict=True)   # (the converted weights again: the step above moved them)
    dst.eval()
    # the reference: the source model with mem_out zeroed (weight and bias) and mem_in zeroed
    ref = TP.TokPolicy(**cfg_src).eval()
    ref.load_state_dict(ck['model'])
    ref2 = TP.TokPolicy(**cfg_src).eval()   # (mem_out zeroed only: mem_in does not matter then)
    ref2.load_state_dict(ck['model'])
    with torch.no_grad():
        for mdl in (ref, ref2):
            mdl.mem_out.weight.zero_()
            mdl.mem_out.bias.zero_()
        ref.gtok.mem_in.weight.zero_()
        ref.gtok.mem_in.bias.zero_()
    seq = TS.make_seq(rng)
    pend = [M.synth_pending(rng, NW, dst.heads) for _ in seq]
    u = [torch.rand((NW, sum(dst.heads))) for _ in seq]
    l_dst = run_actor(dst, seq, pend, TP.TemporalState(dst, NW, 'cpu', torch.float32), u=u)
    l_ref = run_actor(ref, seq, pend, TP.TemporalState(ref, NW, 'cpu', torch.float32), u=u, mem_dim=DM)
    l_ref2 = run_actor(ref2, seq, pend, TP.TemporalState(ref2, NW, 'cpu', torch.float32), u=u, mem_dim=DM)
    l_src = run_actor(src, seq, pend, TP.TemporalState(src, NW, 'cpu', torch.float32), u=u, mem_dim=DM)
    keys = ('logits', 'value', 'vg', 'logp')
    d_ref = maxd([l_dst[k] for k in keys], [l_ref[k] for k in keys])
    d_ref2 = maxd([l_dst[k] for k in keys], [l_ref2[k] for k in keys])
    same_act = bool(torch.equal(l_dst['act'], l_ref['act']))
    d_src = maxd([l_dst[k] for k in ('logits', 'value', 'vg')], [l_src[k] for k in ('logits', 'value', 'vg')])
    check('(c) converted model == the source with mem_out and mem_in zeroed (actor steps over resets / room changes: '
          'logits, value, value_global, act log-probabilities; the same actions); == with mem_out zeroed alone',
          d_ref < 1e-6 and d_ref2 < 1e-6 and same_act, f'max |diff| {d_ref:.1e} / {d_ref2:.1e}')
    check('(c) the change is the GRU\'s contribution: the converted outputs differ from the unchanged source',
          d_src > 1e-3, f'max |diff| vs the source {d_src:.3f}')
    # stored mode (the learner's recompute pass) on a rollout of the converted model, against the reference
    import train_tok as TT
    ra, _ = new_rolls(dst)
    run_actor(dst, seq[:T1], pend[:T1], TP.TemporalState(dst, NW, 'cpu', torch.float32), (ra, ra))
    flat = np.flatnonzero((np.arange(ra.cap)[None] < ra.length[:, None]).reshape(-1))
    trow = TT.temporal_rows(ra, flat, ra.cap, 'cpu')
    rows = ra.rows.reshape(-1)[flat]
    b = TP.to_batch(rows, ra.pending.reshape(-1, len(dst.heads))[flat], 'cpu')
    tk = dict(mode='stored', o=trow['o'], o_prev=trow['o_prev'], has=trow['has'])
    hs = trow['has'].numpy()
    with torch.no_grad():
        d_st = maxd(dst(FT.host_batch(dst, b, rows, hs), packed=True, need_vg=True, temporal=tk),
                    [x for j, x in enumerate(ref(FT.host_batch(ref, b, rows, hs), packed=True, need_vg=True,
                                                 mem=torch.randn(len(flat), DM), temporal=tk)) if j != 3])
    check('(c) stored mode (packed): converted == source with mem_out / mem_in zeroed, whatever the GRU state',
          d_st < 1e-6, f'max |diff| {d_st:.1e}')
    # drop_memory refuses a model with memory; a memory-free state passes through
    try:
        TP.drop_memory(src, ck['model'])
        refused = False
    except ValueError:
        refused = True
    s2, n2, note2 = TP.drop_memory(dst, state)
    check('(c) drop_memory: refuses a memory model; nothing to drop -> unchanged, no log line',
          refused and s2 is state and n2 == [] and note2 is None)


def _eval_path(rng):
    """(e) eval_tok with a no-GRU temporal checkpoint."""
    import eval_tok as E
    torch.manual_seed(31)
    m = TP.TokPolicy(**CFG).eval()
    TS.perturb_all(m, 32)
    buf = io.BytesIO()
    torch.save(dict(model=m.state_dict(), config=m.config, update=1, decisions=0), buf)
    buf.seek(0)
    ck = torch.load(buf, map_location='cpu')
    em, changed, note = E.build_eval_model(ck, True, torch.device('cpu'))
    check('(e) build_eval_model: a no-GRU temporal checkpoint builds with every tensor exact, no memory',
          em.tatt is not None and not changed and note is None and em.memory == 0 and em.config.get('temporal'))
    seq = TS.make_seq(rng)
    pend = [M.synth_pending(rng, NW, em.heads) for _ in seq]
    t_state = TP.TemporalState(em, NW, 'cpu', torch.float32)
    ref_st = TP.TemporalState(em, NW, 'cpu', torch.float32)
    mem_reset = np.zeros(NW, bool)
    ok = True
    for t, rows in enumerate(seq):
        order = [2, 0, 1] if t % 2 else [0, 1, 2]
        mem_reset[np.flatnonzero(rows['first'])] = True
        a_e = E.policy_actions(em, rows[order], pend[t][order], order, torch.device('cpu'), True, None, mem_reset,
                               t_state)
        with torch.no_grad():
            a_r = em.act(TP.to_batch(rows, pend[t], 'cpu'), greedy=True,
                         temporal=dict(mode='step', state=ref_st, slot=torch.arange(NW)))[0]
        ok &= np.array_equal(a_e, a_r.numpy()[order])
    d_o = float((t_state.prev_o - ref_st.prev_o).abs().max())
    check('(e) policy_actions with the per-worker temporal state and no GRU state == act step by step (greedy '
          'actions; states up to rounding: workers reordered)', ok and d_o < 1e-4 and
          torch.equal(t_state.nroom, ref_st.nroom), f'max |o diff| {d_o:.1e}')
    em2, _, _ = E.build_eval_model(ck, True, torch.device('cpu'))
    try:
        E.apply_ablation(em2, 'global')
        st2 = TP.TemporalState(em2, NW, 'cpu', torch.float32)
        a2 = E.policy_actions(em2, seq[0], pend[0], [0, 1, 2], torch.device('cpu'), True, None, np.zeros(NW, bool),
                              st2)
        ok2 = a2.shape == (NW, len(em2.heads))
    except Exception as exc:   # noqa: BLE001
        ok2 = False
        print('   ', exc)
    check('(e) --ablate global on a no-GRU temporal model (no mem_in) runs', ok2)


def _graph_actor(rng):
    """(f) CUDA: GraphActor with a TemporalState, no memory table, a device rollout == the eager step."""
    import train_tok as TT
    cuda = torch.device('cuda')
    gm = TP.TokPolicy(**CFG).to(cuda).eval()
    TS.perturb_all(gm, 41)
    ta = gm.tatt
    seq = TS.make_seq(rng, 4, 10, {(0, 3), (1, 5)}, {(2, 6)}, {(3, 4)})
    for fold in (False, True):
        st = TP.TemporalState(gm, 6, cuda, torch.float32)
        actor = TP.GraphActor(gm, 6, 32, stream=torch.cuda.Stream(), crn=False, memory=None, temporal=st, fold=fold)
        st_e = TP.TemporalState(gm, 6, cuda, torch.float32)
        roll = TT.Rollout(6, 16, len(gm.heads), 0, True, dict(P=ta.W - 1, L=ta.L, D=ta.D, R=ta.R, kv='fp32',
                                                             device=cuda))
        widx = np.array([5, 0, 3, 1])
        ok = True
        for t, rows in enumerate(seq):
            rows['n_ent'] = np.minimum(rows['n_ent'], 32)
            pend = M.synth_pending(rng, 4, gm.heads)
            tdest = roll.t_dest(widx)
            _, lp, val = actor.act(rows, pend, widx, rows['first'] != 0, tdest)
            torch.cuda.synchronize()
            td = dict(mode='step', state=st_e, slot=torch.from_numpy(widx).to(cuda))
            with torch.no_grad():
                _, v_ref, _, vg = gm(TP.to_batch(rows, pend, cuda), need_vg=True, temporal=td)
            o_rows = roll.t_o[torch.from_numpy(tdest['rows'] + roll.t_obase).to(cuda)]
            ok &= np.allclose(val, v_ref.cpu().numpy(), atol=1e-3) and actor.h_used is None and \
                bool(torch.allclose(o_rows, td['out']['o'], atol=1e-3)) and \
                np.array_equal(actor.t_out['nroom'], td['out']['nroom'].cpu().numpy())
            TT.store(roll, widx, rows, pend, np.zeros((4, len(gm.heads)), np.int64), lp, val, 0, None, None,
                     actor.t_out)
        check(f'(f) GraphActor + TemporalState without a memory table == the eager step [fold {int(fold)}]', ok)


if __name__ == '__main__':
    sys.exit(main())
