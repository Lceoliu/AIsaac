"""CPU smoke test of B24 (2026-10-14): the fused item / global tokens (TokPolicy(item_tokens_fused=True), train_tok
--item-tokens-fused 1), the held-item slots (TokPolicy(item_slots=N), --item-slots N) and value_global stored by the
actor (GraphActor / act(need_vg=True) -> Rollout.value_g -> the learner's global returns).

No game, no GPU needed (the GraphActor part runs only where CUDA is available). Plain script:
  cd rl/bridge/python && python tests/test_fused_tokens_smoke.py
Checks:
  (a) off-mode (no item / global tokens; and the separate-term B21 / B22 / B23 model with open gates) == git HEAD's
      policy module, bit for bit, on every path (forward padded / packed / fast path, act, evaluate, chunked BPTT);
  (b) fused: every path (padded, packed with the device's or the host's positions, fast path with math attention off /
      on, buckets without xperm) gives the same outputs; backward reaches the item tokens, the gates, the global token
      and mem_in from the policy loss alone; the chunked BPTT forward reproduces the actor's per-step logits, states and
      value_global; B23's shared table and forced-pickup masking (tok_pickup.ppo_part_terms) work on fused outputs;
  (c) a synthetic B21 / B22 / B23 checkpoint (memory 128, item tokens, global token, shared table, gates the size C92
      logged: mean |gate| 0.10 / 0.03) converted with convert_to_fused: strict load, Adam state loads and steps; the
      output differences to the checkpoint's model are REPORTED (approximate, not exact) for the conversion (gates
      kept) and for gates reset to 1, next to the checkpoint's own gates-closed difference as a scale;
  (d) item_slots 8: token / mask shapes, config key, sequence lengths of the padded and planned layouts, the older
      held items (slots 8 .. 15) do not change the outputs, item_count;
  (e) value_global stored by the actor (act(need_vg=True); GraphActor on CUDA) == the learner's forward on the same
      records; train_tok's Rollout / store / gae use the stored values.
  info: CPU forward times of 256 records (the existing test's shapes): separate terms (HEAD) vs fused vs fused + 8 slots,
      and the learner's fast path forward + backward. CPU ratios are indicative only (the H20 is what counts).
"""
import io
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_memory_smoke as M   # noqa: E402  (its path set-up, synthetic records, HEAD loader)
import test_item_global_smoke as G   # noqa: E402

import numpy as np   # noqa: E402
import torch   # noqa: E402

from isaac_bridge import tok_policy as TP   # noqa: E402
from isaac_bridge import tok_pickup as PK   # noqa: E402
from isaac_bridge.tok_obs import INV_CAP, ROW   # noqa: E402

FAIL = []
D, W = 32, 4
BASE = G.BASE
FUSED = dict(item_tokens=True, global_token=True, item_tokens_fused=True)


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def realistic_rows(rng, n):
    """Synthetic records with inventories like a run's: 0-3 held collectibles, an active item half the time, a trinket
    30%, a pill 40%, 2-19 entities."""
    r = M.synth_rows(rng, n)
    k = rng.integers(0, 4, n)
    r['inv'][:] = 0
    for i in range(n):
        r['inv'][i, :k[i], 0] = rng.integers(1, 700, k[i])
        r['inv'][i, :k[i], 1] = 1
    r['pitem'][:, 0] *= rng.random(n) < 0.5
    r['pitem'][:, 1] *= rng.random(n) < 0.3
    r['pitem'][:, 2] = 0
    r['pitem'][:, 3] *= rng.random(n) < 0.4
    r['pitem'][:, 5] = 0
    r['n_ent'] = rng.integers(2, 20, n)
    return r


def plan_kw(model, rows):
    """(B24) the host plans' extra arguments of a model for ROW records (none when not fused)."""
    NI, Gn = model.extra_tokens()
    if not (NI or Gn):
        return {}
    return dict(n_item=TP.item_count(rows, model.itok.slots) if NI else None, item_tok=NI, glob=Gn)


def fast_batch(model, b, rows, nb=3, xperm=True):
    """The learner's fast path batch (train_tok --learner-fast 1, --packed 1), fused-aware."""
    plan = TP.learner_plan(b['n_ent'].numpy(), b['n_doors'].numpy(), b['ent'].shape[1], nb, **plan_kw(model, rows))
    tok = TP.plan_tokens(plan, torch.from_numpy(plan['rec']), torch.from_numpy(plan['base']))
    mb = dict(b)
    names = ['keep', 'starts', 'gpos'] + (['ekeep', 'dkeep', 'xperm'] if xperm else [])
    if xperm and 'ikeep' in tok:
        names += ['ikeep', 'tgroup', 'glast']
    mb.update((k, tok[k]) for k in names)
    mb['buckets'] = [(torch.from_numpy(c_), w_) for c_, w_ in plan['buckets']]
    return mb


def host_batch(model, b, rows):
    """forward(packed=True) with the host's positions (packed_index), fused-aware."""
    keep, starts = TP.packed_index(b['n_ent'].numpy(), b['n_doors'].numpy(), b['ent'].shape[1], **plan_kw(model, rows))
    mb = dict(b)
    mb['keep'], mb['starts'] = torch.from_numpy(keep), torch.from_numpy(starts)
    return mb


def perturb(m, seed, scale=0.02):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_((torch.randn(p.shape, generator=g) * scale).to(p.device))


def c92_gates(m, seed):
    """Gates of the size C92 logged at its end (mean |gate| 0.104 items, 0.026 global) and trained-looking memory
    projections."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        L_ = len(m.blocks)
        m.itok.gate.copy_(torch.randn(L_, generator=g) * 0.13)
        m.gtok.gate.copy_(torch.randn(L_, generator=g) * 0.033)
        if m.memory:
            m.mem_out.weight.normal_(0, 0.05, generator=g)
            m.gtok.mem_in.weight.normal_(0, 0.05, generator=g)


def chunk_dict(n, d=D):
    T = 3
    c = n // T
    reset = torch.zeros(c, T, dtype=torch.bool)
    reset[0, 1] = True
    return dict(h0=torch.randn(c, d), T=T, reset=reset, tail=torch.randn(n - c * T, d) if n > c * T else None)


def main():
    rng = np.random.default_rng(24)
    dev = torch.device('cpu')

    # ------------------------------------------------------------ (a) off == HEAD
    old = M.head_module()
    cases = (('plain', dict(), False), ('items', BASE, False), ('items+memory', dict(BASE, memory=D), False),
             ('B21 + B22 + B23 + memory, separate terms, gates open',
              dict(BASE, memory=D, item_tokens=True, global_token=True, item_shared=True), True))
    for tag, cfg, gates in cases:
        torch.manual_seed(1)
        new = TP.TokPolicy(**cfg).eval()
        if gates:
            G.open_gates(new)
        n = 11
        b = TP.to_batch(G.rows_with_empty(rng, n), M.synth_pending(rng, n, new.heads), dev)
        acts = torch.from_numpy(M.synth_pending(rng, n, new.heads))
        u = torch.rand((n, sum(new.heads)))
        mem = torch.randn(n, D) if cfg.get('memory') else None
        memd = G.chunk_dict(n) if cfg.get('memory') else None
        check(f'(a) no B24 keys in the config [{tag}]',
              'item_tokens_fused' not in new.config and 'item_slots' not in new.config and not new.fused)
        if old is None:
            print('skip (a) HEAD comparison: git HEAD not readable')
            continue
        ref = old.TokPolicy(**cfg).eval()
        ref.load_state_dict(new.state_dict())
        o_new, o_ref = G.outputs(new, b, acts, u, mem, memd), G.outputs(ref, b, acts, u, mem, memd)
        check(f'(a) == HEAD, all paths [{tag}]', M.same(o_new, o_ref) and len(o_new) == len(o_ref),
              f'{len(o_new)} tensors, max diff {M.maxdiff(o_new, o_ref):.1e}')

    # ------------------------------------------------------------ (b) fused: paths, backward, BPTT, B23
    for tag, extra in (('items + global', {}), ('items only', dict(global_token=False)),
                       ('global only', dict(item_tokens=False)), ('shared table, 8 slots', dict(item_shared=True,
                                                                                                item_slots=8))):
        cfg = dict(BASE, memory=D, **dict(FUSED, **extra))
        torch.manual_seed(2)
        m = TP.TokPolicy(**cfg).eval()
        perturb(m, 3)
        n = 13
        rows = G.rows_with_empty(rng, n)
        b = TP.to_batch(rows, M.synth_pending(rng, n, m.heads), dev)
        vg_on = m.gtok is not None
        kw = dict(mem=torch.randn(n, D), need_vg=vg_on)
        with torch.no_grad():
            ref = m(b, **kw)
            outs = {'packed (device)': m(b, packed=True, **kw), 'packed (host)': m(host_batch(m, b, rows), packed=True,
                                                                                      **kw)}
            for math in (False, True):
                m.set_math_attention(math)
                outs[f'fast math={int(math)}'] = m(fast_batch(m, b, rows), packed=True, **kw)
                outs[f'buckets, no xperm, math={int(math)}'] = m(fast_batch(m, b, rows, xperm=False), packed=True,
                                                                  **kw)
            m.set_math_attention(False)
        worst = max(M.maxdiff(o, ref) for o in outs.values())
        check(f'(b) fused, every path agrees [{tag}]', worst < 1e-4 and all(len(o) == len(ref) for o in outs.values()),
              f'{len(outs)} paths x {len(ref)} outputs, max diff {worst:.1e}')

    torch.manual_seed(4)
    learner = TP.TokPolicy(**BASE, memory=D, **FUSED)
    perturb(learner, 5)
    learner.train()
    n = 12
    rows = G.rows_with_empty(rng, n)
    b = TP.to_batch(rows, M.synth_pending(rng, n, learner.heads), dev)
    acts = torch.from_numpy(M.synth_pending(rng, n, learner.heads))
    adv, ret = torch.randn(n), torch.randn(n)
    memd = chunk_dict(n)
    learner.set_math_attention(True)
    learner.zero_grad()
    logits, value, _, _, vg = learner(fast_batch(learner, b, rows), packed=True, mem=memd, need_vg=True)
    logp, ent = learner.log_prob_fast(logits, learner.gate(b), acts)
    (-(logp * adv).mean() + 0.5 * ((value - ret) ** 2).mean() - 0.01 * ent.sum(-1).mean()).backward()
    learner.set_math_attention(False)
    gr = {k: (0.0 if p.grad is None else float(p.grad.abs().sum())) for k, p in learner.named_parameters()}
    need = ['itok.coll.weight', 'itok.mlp.0.weight', 'itok.slot.weight', 'itok.gate', 'gtok.tok', 'gtok.gate',
            'gtok.mem_in.weight']
    zero = [k for k in need if not gr[k] > 0]
    check('(b) fused backward (fast path, BPTT chunks): the policy loss alone reaches the item tokens, gates, the '
          'global token and mem_in', not zero and all(np.isfinite(v_) for v_ in gr.values()),
          f'zero: {zero}' if zero else ', '.join(f'{k} {gr[k]:.1e}' for k in need[:4]))

    # chunked BPTT reproduces the actor (padded, packed, fast)
    torch.manual_seed(7)
    model = TP.TokPolicy(**BASE, memory=D, **FUSED).eval()
    perturb(model, 8)
    heads = model.heads
    steps = 9
    seq_rows = [G.rows_with_empty(rng, W) for _ in range(steps)]
    firsts = np.zeros((steps, W), bool)
    firsts[0, :] = True
    firsts[4, 1] = True
    firsts[7, 3] = True
    for t in range(steps):
        seq_rows[t]['first'] = firsts[t]
    seq_pend = [M.synth_pending(rng, W, heads) for _ in range(steps)]
    table = torch.zeros((W, D))
    h_in_log, h_new_log, logit_log, vg_log = [], [], [], []
    with torch.no_grad():
        for t in range(steps):
            bt = TP.to_batch(seq_rows[t], seq_pend[t], dev)
            h_in = table * torch.from_numpy(~firsts[t]).float()[:, None]
            logits, value, _, h_new, vg = model(bt, mem=h_in, need_vg=True)
            table.copy_(h_new)
            h_in_log.append(h_in.clone())
            h_new_log.append(h_new.clone())
            logit_log.append(logits.clone())
            vg_log.append(vg.clone())
    T, cap = 3, steps + 2
    mask = np.zeros((W, cap), bool)
    mask[:, :steps] = True
    plan = TP.memory_chunks(np.full(W, steps), mask, T)
    roll_rows = np.zeros((W, cap), ROW)
    roll_pend = np.zeros((W, cap, len(heads)), np.int64)
    roll_h = np.zeros((W, cap, D), np.float32)
    for t in range(steps):
        roll_rows[:, t], roll_pend[:, t], roll_h[:, t] = seq_rows[t], seq_pend[t], h_in_log[t].numpy()
    pos = plan['pos']
    rows_c = roll_rows.reshape(-1)[pos]
    bc = TP.to_batch(rows_c, roll_pend.reshape(-1, len(heads))[pos], dev)
    memd = dict(h0=torch.from_numpy(roll_h.reshape(-1, D)[plan['start']]), T=T,
                reset=torch.from_numpy(rows_c['first'] != 0).view(-1, T))
    wsel = torch.from_numpy((pos // cap) * steps + pos % cap)
    ref_l = torch.stack(logit_log, 1).reshape(W * steps, -1)[wsel]
    ref_h = torch.stack(h_new_log, 1).reshape(W * steps, D)[wsel]
    ref_v = torch.stack(vg_log, 1).reshape(W * steps)[wsel]
    for ptag, fn in (('padded', lambda: model(bc, mem=memd, need_vg=True)),
                     ('packed (host)', lambda: model(host_batch(model, bc, rows_c), packed=True, mem=memd,
                                                     need_vg=True)),
                     ('fast path', lambda: model(fast_batch(model, bc, rows_c), packed=True, mem=memd, need_vg=True))):
        with torch.no_grad():
            o = fn()
        el, eh, ev = (float((o[0] - ref_l).abs().max()), float((o[3] - ref_h).abs().max()),
                      float((o[4] - ref_v).abs().max()))
        check(f'(b) fused BPTT chunks reproduce the actor [{ptag}]', el < 1e-4 and eh < 1e-4 and ev < 1e-4,
              f'logits {el:.1e}, states {eh:.1e}, value_global {ev:.1e}')

    # B23 in fused mode: the shared table gets gradient from the held items; forced-pickup masking on fused logits
    torch.manual_seed(9)
    sh = TP.TokPolicy(**BASE, memory=D, item_shared=True, **FUSED)
    perturb(sh, 10)
    n = 10
    rows = M.synth_rows(rng, n)
    rows['ent_item'][..., 0] = 0
    rows['inv'][..., 0] = 0
    rows['pitem'][:, 0] = 0
    rows['inv'][:, 0, 0], rows['inv'][:, 1, 0] = 801, 802   # (held only, never on a pedestal)
    rows['inv'][:, :2, 1] = 1
    rows['pforce'][[2, 5]] = (3, 7)
    forced = torch.from_numpy(rows['pforce'] > 0)
    b = TP.to_batch(rows, M.synth_pending(rng, n, sh.heads), dev)
    acts = torch.from_numpy(M.synth_pending(rng, n, sh.heads))
    acts[forced, 0] = torch.from_numpy(rows['pforce'][rows['pforce'] > 0].astype(np.int64))
    acts[forced, 1] = 0
    sh.train()
    sh.zero_grad()
    logits, value = sh(fast_batch(sh, b, rows), packed=True, mem=chunk_dict(n))[:2]
    logits.retain_grad()
    logp, entropy = sh.log_prob(logits, sh.gate(b), acts)
    a, _ = PK.normalise_advantages(torch.randn(n), forced)
    pg, vf, ent, imit, *_ = PK.ppo_part_terms(logits, logp, entropy, logp.detach() + 0.05, a, value, torch.randn(n),
                                              acts, forced, 0.2)
    pg.backward(retain_graph=True)
    g_pg = logits.grad.abs().sum(1).clone()
    logits.grad = None
    imit.backward()
    g_im = logits.grad.abs().sum(1)
    held_g = [float(sh.item_emb.weight.grad[i].abs().sum()) for i in (801, 802)]
    check('(b) fused + shared table: item_emb gets gradient from the held items (item tokens only)',
          sh.itok.coll is sh.item_emb and all(v_ > 0 for v_ in held_g), f'{held_g[0]:.1e}, {held_g[1]:.1e}')
    check('(b) fused + forced pickups: policy term zero on forced rows, imitation only on them',
          bool((g_pg[forced] == 0).all()) and bool((g_pg[~forced] > 0).all()) and bool((g_im[forced] > 0).all())
          and bool((g_im[~forced] == 0).all()))

    # ------------------------------------------------------------ (c) conversion of a B21 / B22 / B23 checkpoint
    Dm = 128
    ck_cfg = dict(BASE, memory=Dm, item_tokens=True, global_token=True, item_shared=True)
    rows_eval = realistic_rows(np.random.default_rng(99), 256)
    lines = []
    for seed in (11, 12, 13):
        torch.manual_seed(seed)
        src = TP.TokPolicy(**ck_cfg)
        perturb(src, seed)
        c92_gates(src, seed)
        opt = torch.optim.Adam(src.main_parameters(), lr=1e-4, eps=1e-5)
        nb_ = TP.to_batch(realistic_rows(rng, 8), M.synth_pending(rng, 8, src.heads), dev)
        src(nb_, need_vg=True, mem=torch.zeros(8, Dm))[-1].sum().backward()
        opt.step()
        buf = io.BytesIO()
        torch.save(dict(model=src.state_dict(), optimizer=opt.state_dict(), config=src.config), buf)
        buf.seek(0)
        ck = torch.load(buf, map_location='cpu')
        src.eval()
        for slots in (INV_CAP, 8):
            fu = TP.TokPolicy(**dict(ck['config'], item_tokens_fused=True, item_slots=slots)).eval()
            state, note = TP.convert_to_fused(fu, ck['model'])
            fu.load_state_dict(state, strict=True)
            if seed == 11 and slots == INV_CAP:
                check('(c) convert_to_fused: strict load, log line', 'gates' in note, note[:110] + ' ...')
                opt_f = torch.optim.Adam(fu.main_parameters(), lr=1e-4, eps=1e-5)
                opt_f.load_state_dict(ck['optimizer'])
                fu.train()
                fu.zero_grad()
                fu(nb_, need_vg=True, mem=torch.zeros(8, Dm))[-1].sum().backward()
                opt_f.step()
                p0 = fu.main_parameters()[0]
                check('(c) Adam state of the checkpoint loads as it is (same names / order) and steps',
                      int(opt_f.state[p0]['step']) == 2)
                fu.load_state_dict(state, strict=True)
                fu.eval()
            n_e = len(rows_eval)
            b_e = TP.to_batch(rows_eval, M.synth_pending(np.random.default_rng(seed), n_e, src.heads), dev)
            h_e = torch.randn(n_e, Dm, generator=torch.Generator().manual_seed(seed)) * 0.3
            with torch.no_grad():
                ref = src(b_e, mem=h_e, need_vg=True)
                kept = fu(b_e, mem=h_e, need_vg=True)
                fu.itok.gate.fill_(1.0)
                fu.gtok.gate.fill_(1.0)
                ones = fu(b_e, mem=h_e, need_vg=True)
                closed = TP.TokPolicy(**ck_cfg).eval()
                closed.load_state_dict(ck['model'])
                closed.itok.gate.zero_()
                closed.gtok.gate.zero_()
                shut = closed(b_e, mem=h_e, need_vg=True)

            def md(o):
                return (float((o[0] - ref[0]).abs().max()), float((o[0] - ref[0]).abs().mean()),
                        float((o[1] - ref[1]).abs().max()), float((o[-1] - ref[-1]).abs().max()))
            r_k, r_1, r_c = md(kept), md(ones), md(shut)
            ok_ = all(np.isfinite(r_k)) and r_k[0] > 0   # (not exact: one softmax is another function)
            lines.append((seed, slots, r_k, r_1, r_c))
            check(f'(c) converted checkpoint runs, outputs finite, not exact [seed {seed}, {slots} slots]', ok_,
                  f'max |logit diff| {r_k[0]:.3f} (mean {r_k[1]:.4f}), value {r_k[2]:.3f}, value_global {r_k[3]:.3f}')
    print('info (c): max |diff| to the checkpoint on 256 realistic synthetic records (logits max / mean, value, '
          'value_global)')
    for seed, slots, r_k, r_1, r_c in lines:
        print(f'  seed {seed} slots {slots:2d}: gates kept {r_k[0]:.3f} / {r_k[1]:.4f} / {r_k[2]:.3f} / {r_k[3]:.3f}'
              f' | gates 1 {r_1[0]:.3f} / {r_1[1]:.4f} / {r_1[2]:.3f} / {r_1[3]:.3f}'
              f' | (scale: checkpoint with its gates closed {r_c[0]:.3f} / {r_c[1]:.4f})')
    kept_m = np.mean([x[2][1] for x in lines])
    ones_m = np.mean([x[3][1] for x in lines])
    check('(c) gates kept is the closer conversion (mean |logit diff| over the seeds)', kept_m <= ones_m,
          f'kept {kept_m:.4f}, gates 1 {ones_m:.4f}')

    # ------------------------------------------------------------ (d) item_slots 8
    torch.manual_seed(14)
    m8 = TP.TokPolicy(**BASE, memory=D, item_slots=8, **FUSED).eval()
    perturb(m8, 15)
    n = 9
    rows = M.synth_rows(rng, n)
    rows['inv'][:, :, 1] = 1   # (every slot held)
    b = TP.to_batch(rows, M.synth_pending(rng, n, m8.heads), dev)
    tok, pres = m8.itok(b)
    NI, Gn = m8.extra_tokens()
    t_pad = 3 + 8 + b['ent'].shape[1] + NI + Gn   # (DOOR_CAP = 8)
    plan = TP.learner_plan(b['n_ent'].numpy(), b['n_doors'].numpy(), b['ent'].shape[1], 3, **plan_kw(m8, rows))
    ic = TP.item_count(rows, 8)
    check('(d) 8 slots: 12 item tokens + 1 global, config key, item_count <= 12',
          tok.shape == (n, 12, 192) and pres.shape == (n, 12) and (NI, Gn) == (12, 1) and
          m8.config.get('item_slots') == 8 and int(ic.max()) <= 12 and plan['t'] == t_pad and
          bool((torch.from_numpy(ic) == pres.sum(1)).all()), f'padded sequence {t_pad} tokens, held 8 of {INV_CAP}')
    rows2 = rows.copy()
    rows2['inv'][:, 8:, 0] = rng.integers(1, 700, (n, INV_CAP - 8))   # (older items: no tokens with 8 slots)
    b2 = TP.to_batch(rows2, M.synth_pending(rng, n, m8.heads), dev)
    b2['pending'] = b['pending']
    with torch.no_grad():
        o1 = m8.itok(b)[0]
        o2 = m8.itok(b2)[0]
    check('(d) 8 slots: held items past slot 8 do not reach the item tokens', torch.equal(o1, o2))
    with torch.no_grad():
        f1 = m8(fast_batch(m8, b, rows), packed=True, mem=torch.zeros(n, D), need_vg=True)
        f0 = m8(b, mem=torch.zeros(n, D), need_vg=True)
    check('(d) 8 slots: fast path == padded', M.maxdiff(f1, f0) < 1e-4, f'{M.maxdiff(f1, f0):.1e}')
    check('(d) item_slots without item tokens / out of range rejected',
          _raises(lambda: TP.TokPolicy(**BASE, global_token=True, item_slots=8)) and
          _raises(lambda: TP.TokPolicy(**BASE, item_tokens=True, item_slots=17)) and
          _raises(lambda: TP.TokPolicy(**BASE, item_tokens_fused=True)))

    # ------------------------------------------------------------ (e) stored value_global == forward value_global
    import train_tok as TT
    torch.manual_seed(16)
    for tag, cfg in (('fused', dict(BASE, memory=D, **FUSED)),
                     ('separate terms', dict(BASE, memory=D, item_tokens=True, global_token=True))):
        mv = TP.TokPolicy(**cfg).eval()
        perturb(mv, 17)
        if not mv.fused:
            G.open_gates(mv)
        n = 12
        rows = G.rows_with_empty(rng, n)
        pend = M.synth_pending(rng, n, mv.heads)
        b = TP.to_batch(rows, pend, dev)
        h = torch.randn(n, D)
        u = torch.rand((n, sum(mv.heads)))
        with torch.no_grad():
            a_, lp_, v_, hn_, vg_act = mv.act(b, u=u, mem=h, need_vg=True)
            a0, lp0, v0, hn0 = mv.act(b, u=u, mem=h)
            vg_l = mv(b, packed=True, mem=h, need_vg=True)[-1]
            vg_f = mv(fast_batch(mv, b, rows), packed=True, mem=h, need_vg=True)[-1]
        check(f'(e) act(need_vg) = act + value_global; == the learner\'s forward [{tag}]',
              M.same([a_, lp_, v_, hn_], [a0, lp0, v0, hn0]) and float((vg_act - vg_l).abs().max()) < 1e-5 and
              float((vg_act - vg_f).abs().max()) < 1e-5,
              f'packed {float((vg_act - vg_l).abs().max()):.1e}, fast {float((vg_act - vg_f).abs().max()):.1e}')
    roll = TT.Rollout(2, 6, len(mv.heads), D, True)
    rr = G.rows_with_empty(rng, 2)
    for k_ in range(4):
        vgk = np.array([0.5 + k_, -1.0 - k_], np.float32)
        TT.store(roll, np.array([0, 1]), rr, np.zeros((2, len(mv.heads)), np.int64), np.zeros((2, len(mv.heads)),
                 np.int64), np.zeros(2, np.float32), np.zeros(2, np.float32), 0, np.zeros((2, D), np.float32), vgk)
        roll.reward[:, k_] = 1.0
        roll.open[:] = k_
    roll.reward[:, 3] = 0.0
    _, ret_s, _ = TT.gae(roll, 0.9, 1.0, roll.value_g)
    exp_ = 1.0 + 0.9 * (1.0 + 0.9 * (1.0 + 0.9 * roll.value_g[0, 3]))   # (the open decision 3 bootstraps)
    spare = TT.Rollout(2, 6, len(mv.heads), D, True)
    spare.carry(roll)
    check('(e) Rollout stores value_global, gae bootstraps from it, carry keeps the open one',
          np.allclose(roll.value_g[:, :4], [[0.5, 1.5, 2.5, 3.5], [-1, -2, -3, -4]]) and
          abs(float(ret_s[0, 0]) - exp_) < 1e-5 and float(spare.value_g[0, 0]) == 3.5,
          f'return[0] {float(ret_s[0, 0]):.4f} = {exp_:.4f}')

    # ------------------------------------------------------------ (f) CUDA: GraphActor value_global
    if torch.cuda.is_available():
        cuda = torch.device('cuda')
        for tag, cfg in (('fused', dict(BASE, memory=D, **FUSED)), ('fused, 8 slots', dict(BASE, memory=D, item_slots=8,
                                                                                          **FUSED))):
            gm = TP.TokPolicy(**cfg).to(cuda).eval()
            perturb(gm, 18)
            tab = torch.randn((W + 2, D), device=cuda)
            start = tab.clone()
            actor = TP.GraphActor(gm, W + 2, 32, stream=torch.cuda.Stream(), crn=False, memory=tab)
            rows = G.rows_with_empty(rng, W)
            rows['n_ent'] = np.minimum(rows['n_ent'], 32)
            pend = M.synth_pending(rng, W, gm.heads)
            widx = np.array([5, 0, 3, 1])
            reset = np.array([False, True, False, False])
            _, lp, val = actor.act(rows, pend, widx, reset)
            torch.cuda.synchronize()
            h_ref = start[torch.from_numpy(widx).to(cuda)] * torch.from_numpy(~reset).float().to(cuda)[:, None]
            with torch.no_grad():
                _, v_ref, _, hn_ref, vg_ref = gm(TP.to_batch(rows, pend, cuda), mem=h_ref, need_vg=True)
            check(f'(f) GraphActor [{tag}] = eager (values, value_global, table rows, h_used)',
                  np.allclose(val, v_ref.cpu().numpy(), atol=1e-4) and
                  np.allclose(actor.vg, vg_ref.cpu().numpy(), atol=1e-4) and
                  np.allclose(actor.h_used, h_ref.cpu().numpy(), atol=1e-6) and
                  torch.allclose(tab[torch.from_numpy(widx).to(cuda)], hn_ref, atol=1e-4))
    else:
        print('skip (f) GraphActor: no CUDA here')

    # ------------------------------------------------------------ info: CPU timing (indicative only)
    bench()
    print('RESULT', 'FAIL ' + ', '.join(FAIL) if FAIL else 'all checks passed', flush=True)
    return 1 if FAIL else 0


def bench(reps=5):
    """CPU forward times of 256 records (the existing test's info line: synth_rows, <= 32 entities, memory 32): the
    actor's padded forward, and the learner's fast path forward + backward on realistic inventories."""
    rng = np.random.default_rng(8)
    torch.manual_seed(8)
    n = 256
    rows = M.synth_rows(rng, n)
    rows['n_ent'] = np.minimum(rows['n_ent'], 32)
    rows_r = realistic_rows(rng, n)
    cases = (('off (no item / global tokens)', {}),
             ('separate terms (HEAD B21 + B22)', dict(item_tokens=True, global_token=True)),
             ('fused, 16 slots', dict(FUSED)),
             ('fused, 8 slots', dict(FUSED, item_slots=8)))
    print(f'info: CPU, {torch.get_num_threads()} threads, {reps} repetitions, median ms (indicative only: the GPU '
          'is what counts)')
    for tag, extra in cases:
        m = TP.TokPolicy(**BASE, memory=D, **extra).eval()
        perturb(m, 1)
        b = TP.to_batch(rows, M.synth_pending(rng, n, TP.ITEM_HEADS), 'cpu')
        h = torch.zeros(n, D)
        kw = dict(need_vg=True) if m.gtok is not None else {}
        ts = []
        with torch.no_grad():
            m(b, mem=h, **kw)
            for _ in range(reps):
                t0 = time.perf_counter()
                m(b, mem=h, **kw)
                ts.append(time.perf_counter() - t0)
        br = TP.to_batch(rows_r, M.synth_pending(rng, n, TP.ITEM_HEADS), 'cpu')
        m.train()
        m.set_math_attention(True)
        mb = fast_batch(m, br, rows_r)
        tl = []
        for k_ in range(reps + 1):
            t0 = time.perf_counter()
            m.zero_grad()
            o = m(mb, packed=True, mem=chunk_dict(n), **kw)
            loss = o[0].float().pow(2).mean() + o[1].pow(2).mean() + (o[-1].pow(2).mean() if kw else 0)
            loss.backward()
            if k_:
                tl.append(time.perf_counter() - t0)
        m.set_math_attention(False)
        print(f'info: {tag:34s} actor forward (padded, {b["ent"].shape[1]} entity tokens) '
              f'{np.median(ts) * 1000:6.0f} ms | learner fast path fwd+bwd (realistic items) '
              f'{np.median(tl) * 1000:6.0f} ms')


def _raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


if __name__ == '__main__':
    sys.exit(main())
