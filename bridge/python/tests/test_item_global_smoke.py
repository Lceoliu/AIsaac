"""CPU smoke test of the token policy's item tokens (B21, TokPolicy(item_tokens=True), train_tok --item-tokens 1) and
global token + value_global head (B22, TokPolicy(global_token=True), train_tok --global-token 1); 2026-10-12.

No game, no GPU needed (the GraphActor part runs only where CUDA is available). Plain script:
  cd rl/bridge/python && python tests/test_item_global_smoke.py
Checks:
  (a) options off: the outputs (forward padded / packed / the learner's fast path with buckets, math attention on and
      off; act with uniforms; evaluate; with memory also the chunked BPTT forward) are bit-identical to the policy
      module of git HEAD (skipped if git or HEAD's file is not available);
  (b) options on with fresh new modules (gates 0, mem_in 0) loaded from an options-off state: the same outputs as the
      options-off model on every path (bit-identical, else the max abs difference is reported and the check fails);
  (c) gradients: at the start they reach both gates and mem_in, and the item embeddings through value_global's loss
      (the global token attends to the item tokens); the policy loss alone gives the item embeddings zero gradient at
      gate 0 (by construction) and a non-zero one once the gates are open; the opened tokens change the outputs;
  (d) checkpoint round trip (config keys, weights, outputs); resume of a C90-like checkpoint (items + memory, no
      B21 / B22): missing keys = itok.* / gtok.* only, Adam state remapped by name (old states kept, new parameters
      fresh, steps on); resume of a B21 checkpoint without memory into memory + B21 (new parameters in the middle);
  (e) chunked BPTT with the options on (gates open): the chunked forward (padded, packed, fast path) reproduces the
      actor's per-step logits, states and value_global;
  (f) CUDA only: GraphActor with the options on = the eager act.
"""
import io
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_memory_smoke as M   # noqa: E402  (its path set-up, synthetic records, HEAD loader)

import numpy as np   # noqa: E402
import torch   # noqa: E402

from isaac_bridge import tok_policy as TP   # noqa: E402
from isaac_bridge.tok_obs import ROW   # noqa: E402

FAIL = []
D, W = 32, 4
BASE = dict(items=True, charge=True, ent_ext=True, pchar=True)
OPTS = (('items', dict(item_tokens=True)), ('global', dict(global_token=True)),
        ('both', dict(item_tokens=True, global_token=True)))


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def rows_with_empty(rng, n):
    """Synthetic records; the last two without any held item, active item, trinket or pocket (has_item = 0)."""
    r = M.synth_rows(rng, n)
    r['inv'][-2:] = 0
    r['pitem'][-2:] = 0
    return r


def fast_batch(b):
    """The learner's fast path batch (train_tok --learner-fast 1 with --packed 1): learner_plan / plan_tokens'
    positions, the buckets, the packed tokens built from the real ones (xperm)."""
    plan = TP.learner_plan(b['n_ent'].numpy(), b['n_doors'].numpy(), b['ent'].shape[1], 3)
    tok = TP.plan_tokens(plan, torch.from_numpy(plan['rec']), torch.from_numpy(plan['base']))
    mb = dict(b)
    mb.update((k, tok[k]) for k in ('keep', 'starts', 'ekeep', 'gpos', 'dkeep', 'xperm'))
    mb['buckets'] = [(torch.from_numpy(c_), w_) for c_, w_ in plan['buckets']]
    return mb


def outputs(model, b, acts, u, mem=None, memd=None):
    """Every path's outputs: forward padded / packed / fast (math attention off and on), act with uniforms, evaluate,
    and with memd the chunked forward (padded, packed)."""
    kw = {} if mem is None else dict(mem=mem)
    out = []
    with torch.no_grad():
        out += list(model(b, **kw))
        out += list(model(b, packed=True, **kw))
        for math in (False, True):
            model.set_math_attention(math)
            out += list(model(fast_batch(b), packed=True, **kw))
        model.set_math_attention(False)
        out += list(model.act(b, u=u, **kw))
        out += list(model.evaluate(b, acts, **kw))
        if memd is not None:
            out += list(model(b, mem=memd)) + list(model(b, packed=True, mem=memd))
    return out


def chunk_dict(n):
    """A chunk layout for n records: chunks of 3 with a reset in the middle, the rest a tail (memory models)."""
    T = 3
    c = n // T
    reset = torch.zeros(c, T, dtype=torch.bool)
    reset[0, 1] = True
    return dict(h0=torch.randn(c, D), T=T, reset=reset, tail=torch.randn(n - c * T, D) if n > c * T else None)


def open_gates(m, scale=0.3):
    with torch.no_grad():
        if m.itok is not None:
            m.itok.gate.normal_(0, scale)
        if m.gtok is not None:
            m.gtok.gate.normal_(0, scale)
            if m.gtok.mem_in is not None:
                m.gtok.mem_in.weight.normal_(0, 0.05)
        if m.memory:
            m.mem_out.weight.normal_(0, 0.1)


def main():
    rng = np.random.default_rng(11)
    dev = torch.device('cpu')

    # ------------------------------------------------------------ (a) options off = HEAD's code, bit for bit
    old = M.head_module()
    for tag, cfg in (('plain', dict()), ('items', BASE), ('items+memory', dict(BASE, memory=D))):
        torch.manual_seed(1)
        new = TP.TokPolicy(**cfg).eval()
        heads = new.heads
        n = 11
        b = TP.to_batch(rows_with_empty(rng, n), M.synth_pending(rng, n, heads), dev)
        acts = torch.from_numpy(M.synth_pending(rng, n, heads))
        u = torch.rand((n, sum(heads)))
        mem = torch.randn(n, D) if cfg.get('memory') else None
        memd = chunk_dict(n) if cfg.get('memory') else None
        check(f'(a) no item_tokens / global_token key in the config [{tag}]',
              'item_tokens' not in new.config and 'global_token' not in new.config)
        if old is None:
            print('skip (a) HEAD comparison: git HEAD not readable')
            continue
        ref = old.TokPolicy(**cfg).eval()
        ref.load_state_dict(new.state_dict())
        o_new, o_ref = outputs(new, b, acts, u, mem, memd), outputs(ref, b, acts, u, mem, memd)
        check(f'(a) options off == HEAD, all paths [{tag}]', M.same(o_new, o_ref) and len(o_new) == len(o_ref),
              f'{len(o_new)} tensors, max diff {M.maxdiff(o_new, o_ref):.1e}')

    # ------------------------------------------------------------ (b) options on, fresh gates = options off
    for mtag, mcfg in (('no memory', {}), ('memory', dict(memory=D))):
        torch.manual_seed(2)
        off = TP.TokPolicy(**BASE, **mcfg).eval()
        heads = off.heads
        n = 13
        b = TP.to_batch(rows_with_empty(rng, n), M.synth_pending(rng, n, heads), dev)
        acts = torch.from_numpy(M.synth_pending(rng, n, heads))
        u = torch.rand((n, sum(heads)))
        mem = torch.randn(n, D) if mcfg else None
        memd = chunk_dict(n) if mcfg else None
        o_off = outputs(off, b, acts, u, mem, memd)
        for otag, ocfg in OPTS:
            on = TP.TokPolicy(**BASE, **mcfg, **ocfg).eval()
            missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
            ok_keys = not unexpected and missing and all(k.startswith(('itok.', 'gtok.')) for k in missing)
            o_on = outputs(on, b, acts, u, mem, memd)
            check(f'(b) fresh {otag} == off, all paths [{mtag}]', ok_keys and M.same(o_on, o_off),
                  f'{len(o_on)} tensors, max diff {M.maxdiff(o_on, o_off):.1e}, {len(missing)} new tensors')
            if on.gtok is not None:
                with torch.no_grad():
                    vg = on(b, need_vg=True, **({} if mem is None else dict(mem=mem)))[-1]
                    vg_f = on(fast_batch(b), packed=True, need_vg=True,
                              **({} if mem is None else dict(mem=mem)))[-1]
                check(f'(b) value_global appended last, finite, paths agree [{otag}, {mtag}]',
                      vg.shape == (n,) and bool(torch.isfinite(vg).all()) and float((vg - vg_f).abs().max()) < 1e-4,
                      f'padded vs fast {float((vg - vg_f).abs().max()):.1e}')
            check(f'(b) config keys [{otag}]', all(on.config.get(k) for k in ocfg))

    # ------------------------------------------------------------ (c) gradients
    torch.manual_seed(3)
    learner = TP.TokPolicy(**BASE, memory=D, item_tokens=True, global_token=True)
    learner.train()
    heads = learner.heads
    n = 12
    b = TP.to_batch(rows_with_empty(rng, n), M.synth_pending(rng, n, heads), dev)
    acts = torch.from_numpy(M.synth_pending(rng, n, heads))
    adv, ret, ret_g = torch.randn(n), torch.randn(n), torch.randn(n)
    memd = chunk_dict(n)
    with torch.no_grad():   # (as in a resumed memory checkpoint such as C90: mem_out trained, not zero; with a fresh
        #                   mem_out the GRU and mem_in get no gradient at the very start, as B20's GRU)
        learner.mem_out.weight.normal_(0, 0.1)

    def losses(m):
        logits, value, _, _, vg = m(fast_batch(b), packed=True, mem=memd, need_vg=True)
        logp, ent = m.log_prob(logits, m.gate(b), acts)
        ppo = -(logp * adv).mean() + 0.5 * ((value - ret) ** 2).mean() - 0.01 * ent.sum(-1).mean()
        return ppo, 0.5 * 0.5 * ((vg - ret_g) ** 2).mean()

    def grads(loss):
        learner.zero_grad()
        loss.backward()
        return {k: (0.0 if p.grad is None else float(p.grad.abs().sum())) for k, p in learner.named_parameters()}
    ppo, vgl = losses(learner)
    g_all = grads(ppo + vgl)
    check('(c) at the start: gradient reaches both gates and mem_in',
          g_all['itok.gate'] > 0 and g_all['gtok.gate'] > 0 and g_all['gtok.mem_in.weight'] > 0,
          f"item gate {g_all['itok.gate']:.2e}, global gate {g_all['gtok.gate']:.2e}, "
          f"mem_in {g_all['gtok.mem_in.weight']:.2e}")
    check('(c) at the start: value_global loss reaches the item tokens (through the global token)',
          g_all['itok.coll.weight'] > 0 and g_all['itok.mlp.0.weight'] > 0 and g_all['gtok.tok'] > 0,
          f"coll {g_all['itok.coll.weight']:.2e}, mlp {g_all['itok.mlp.0.weight']:.2e}, tok {g_all['gtok.tok']:.2e}")
    ppo, _ = losses(learner)
    g_pi = grads(ppo)
    check('(c) at the start: the policy loss alone gives the item embeddings zero gradient (gate 0)',
          g_pi['itok.coll.weight'] == 0.0 and g_pi['itok.gate'] > 0, f"coll {g_pi['itok.coll.weight']:.1e}")
    with torch.no_grad():
        before = learner(b, mem=memd)[0].clone()
    open_gates(learner)
    with torch.no_grad():
        after = learner(b, mem=memd)[0]
    check('(c) opened gates change the outputs', float((after - before).abs().max()) > 1e-5,
          f'max logit change {float((after - before).abs().max()):.2e}')
    ppo, _ = losses(learner)
    g_open = grads(ppo)
    new_names = [k for k in g_open if k.startswith(('itok.', 'gtok.')) and not k.startswith('gtok.value')
                 and not k.startswith('gtok.norm')]
    zero = [k for k in new_names if g_open[k] == 0.0]
    check('(c) gates open: the policy loss reaches every item / global token parameter (value head aside)', not zero,
          f'zero: {zero}' if zero else f'{len(new_names)} tensors')
    # (padding_idx 0 rows of the tables stay at zero gradient: absent items)

    # ------------------------------------------------------------ (d) checkpoints
    torch.manual_seed(4)
    model = TP.TokPolicy(**BASE, memory=D, item_tokens=True, global_token=True).eval()
    open_gates(model)
    buf = io.BytesIO()
    torch.save(dict(model=model.state_dict(), config=model.config), buf)
    buf.seek(0)
    ck = torch.load(buf, map_location='cpu')
    back = TP.TokPolicy(**ck['config']).eval()
    back.load_state_dict(ck['model'])
    hb = torch.randn(n, D)
    with torch.no_grad():
        o1, o2 = model(b, mem=hb, need_vg=True), back(b, mem=hb, need_vg=True)
    check('(d) checkpoint round trip (config keys, weights, outputs incl. value_global)',
          ck['config'].get('item_tokens') is True and ck['config'].get('global_token') is True and M.same(o1, o2))

    def adam_after_step(m, mem_):
        opt = torch.optim.Adam(m.main_parameters(), lr=1e-3, eps=1e-5)
        m.zero_grad()
        m(b, mem=mem_)[1].pow(2).mean().backward()
        opt.step()
        return opt
    # C90-like: items + memory, no B21 / B22
    torch.manual_seed(5)
    c90 = TP.TokPolicy(**BASE, memory=D)
    opt90 = adam_after_step(c90, torch.randn(n, D))
    ck90 = dict(model=c90.state_dict(), optimizer=opt90.state_dict(), config=c90.config)
    ck90 = torch.load(io.BytesIO(_dump(ck90)), map_location='cpu')
    run = TP.TokPolicy(**ck90['config'], item_tokens=True, global_token=True)
    missing, unexpected = run.load_state_dict(ck90['model'], strict=False)
    old_names = TP.main_parameter_names(list(ck90['model']))
    new_names = TP.main_parameter_names([k for k, _ in run.named_parameters()])
    osd = ck90['optimizer']
    saved = {nm: osd['state'][i]['exp_avg'].clone() for nm, i in zip(old_names, osd['param_groups'][0]['params'])
             if i in osd['state']}   # (parameters without a gradient in that step have no state)
    check('(d) the C90-like optimiser state has states to carry', len(saved) > 50, f'{len(saved)} states')
    added = TP.remap_optimizer_state(osd, old_names, new_names)
    opt_run = torch.optim.Adam(run.main_parameters(), lr=1e-3, eps=1e-5)
    opt_run.load_state_dict(osd)
    kept = all(torch.equal(opt_run.state[p]['exp_avg'], saved[nm])
               for nm, p in zip(new_names, run.main_parameters()) if nm in saved)
    fresh_none = all(p not in opt_run.state for nm, p in zip(new_names, run.main_parameters()) if nm not in old_names)
    run.zero_grad()
    run(b, mem=torch.randn(n, D), need_vg=True)[-1].pow(2).mean().backward()
    run(b, mem=torch.randn(n, D))[1].pow(2).mean().backward()
    opt_run.step()
    p0 = run.main_parameters()[0]
    check('(d) resume of a C90-like checkpoint: only itok./gtok. missing, Adam remapped by name, stepping',
          not unexpected and missing and all(k.startswith(('itok.', 'gtok.')) for k in missing) and kept and
          fresh_none and added == len(missing) and int(opt_run.state[p0]['step']) == 2,
          f'{len(missing)} new tensors, added {added}, step {int(opt_run.state[p0]["step"])}')
    with torch.no_grad():   # and it starts with the checkpoint's outputs
        c90.eval()
        run2 = TP.TokPolicy(**ck90['config'], item_tokens=True, global_token=True).eval()
        run2.load_state_dict(ck90['model'], strict=False)
        hh = torch.randn(n, D)
        same90 = M.same(c90(b, mem=hh), run2(b, mem=hh)[:4])
    check('(d) resumed C90-like model starts with the checkpoint\'s outputs (logits, value, danger, h_new)', same90)
    # memory added to a B21 checkpoint: the memory modules sit before itok.* (remap by name, not padding)
    torch.manual_seed(6)
    b21 = TP.TokPolicy(**BASE, item_tokens=True)
    opt21 = adam_after_step(b21, None)
    run3 = TP.TokPolicy(**BASE, item_tokens=True, memory=D)
    missing3, _ = run3.load_state_dict(b21.state_dict(), strict=False)
    osd3 = opt21.state_dict()
    names_b21 = TP.main_parameter_names(list(b21.state_dict()))
    names_3 = TP.main_parameter_names([k for k, _ in run3.named_parameters()])
    TP.remap_optimizer_state(osd3, names_b21, names_3)
    opt3 = torch.optim.Adam(run3.main_parameters(), lr=1e-3, eps=1e-5)
    opt3.load_state_dict(osd3)
    st_coll = opt3.state[dict(run3.named_parameters())['itok.coll.weight']]
    check('(d) memory added to a B21 checkpoint: itok state found by name behind the new mem_* parameters',
          all(k.startswith('mem_') for k in missing3) and torch.equal(
              st_coll['exp_avg'], opt21.state[dict(b21.named_parameters())['itok.coll.weight']]['exp_avg']))

    # ------------------------------------------------------------ (e) chunked BPTT with the options on (gates open)
    torch.manual_seed(7)
    model = TP.TokPolicy(**BASE, memory=D, item_tokens=True, global_token=True).eval()
    open_gates(model)
    heads = model.heads
    steps = 9
    seq_rows = [rows_with_empty(rng, W) for _ in range(steps)]
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
                     ('packed', lambda: model(bc, packed=True, mem=memd, need_vg=True)),
                     ('fast path', lambda: model(fast_batch(bc), packed=True, mem=memd, need_vg=True))):
        with torch.no_grad():
            o = fn()
        el, eh, ev = (float((o[0] - ref_l).abs().max()), float((o[3] - ref_h).abs().max()),
                      float((o[4] - ref_v).abs().max()))
        check(f'(e) BPTT chunks with item / global tokens reproduce the actor [{ptag}]',
              el < 1e-4 and eh < 1e-4 and ev < 1e-4, f'logits {el:.1e}, states {eh:.1e}, value_global {ev:.1e}')

    # ------------------------------------------------------------ (f) CUDA: GraphActor with the options on
    if torch.cuda.is_available():
        cuda = torch.device('cuda')
        gm = TP.TokPolicy(**BASE, memory=D, item_tokens=True, global_token=True).to(cuda).eval()
        open_gates(gm)
        tab = torch.randn((W + 2, D), device=cuda)
        start = tab.clone()
        actor = TP.GraphActor(gm, W + 2, 32, stream=torch.cuda.Stream(), crn=False, memory=tab)
        rows = rows_with_empty(rng, W)
        rows['n_ent'] = np.minimum(rows['n_ent'], 32)
        pend = M.synth_pending(rng, W, gm.heads)
        widx = np.array([5, 0, 3, 1])
        reset = np.array([False, True, False, False])
        _, lp, val = actor.act(rows, pend, widx, reset)
        torch.cuda.synchronize()
        h_ref = start[torch.from_numpy(widx).to(cuda)] * torch.from_numpy(~reset).float().to(cuda)[:, None]
        with torch.no_grad():
            _, v_ref, _, hn_ref = gm(TP.to_batch(rows, pend, cuda), mem=h_ref)
        check('(f) GraphActor with item / global tokens = eager (values, table rows)',
              np.allclose(val, v_ref.cpu().numpy(), atol=1e-4) and
              torch.allclose(tab[torch.from_numpy(widx).to(cuda)], hn_ref, atol=1e-4))
    else:
        print('skip (f) GraphActor: no CUDA here')

    # ------------------------------------------------------------ cost indication (CPU, not a check)
    torch.manual_seed(8)
    n = 256
    rows = M.synth_rows(rng, n)
    rows['n_ent'] = np.minimum(rows['n_ent'], 32)
    bt = TP.to_batch(rows, M.synth_pending(rng, n, TP.ITEM_HEADS), dev)
    for tag, extra in (('off', {}), ('items + global', dict(item_tokens=True, global_token=True))):
        m = TP.TokPolicy(**BASE, memory=D, **extra).eval()
        with torch.no_grad():
            m(bt, mem=torch.zeros(n, D))
            t0 = time.perf_counter()
            for _ in range(3):
                m(bt, mem=torch.zeros(n, D))
        print(f'info: CPU forward of {n} records, {bt["ent"].shape[1]} entity tokens, {tag}: '
              f'{(time.perf_counter() - t0) / 3 * 1000:.0f} ms')

    print('RESULT', 'FAIL ' + ', '.join(FAIL) if FAIL else 'all checks passed', flush=True)
    return 1 if FAIL else 0


def _dump(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


if __name__ == '__main__':
    sys.exit(main())
