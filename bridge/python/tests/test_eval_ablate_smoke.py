"""CPU smoke test of eval_tok.py's evaluation-only switches (A28, 2026-10-14): --fused-override, --item-slots,
--ablate items|global|both (eval_tok.build_eval_model / apply_ablation) and --no-pedestals (tok_pickup.NO_PEDESTAL_LUA,
no_pedestal_due, tok_floor's call site).

No game, no GPU needed. Plain script:
  cd rl/bridge/python && python tests/test_eval_ablate_smoke.py
A small separate-term model (B21 / B22 / B23 layout: items, memory 16, item tokens, global token, shared item table)
with random weights and opened gates / mem_in, synthetic records with held items. Checks:
  (a) --ablate items == a reference built directly (ItemTokens subclass with an all-False present mask, gate zeroed),
      bit for bit; ~= a model without item tokens at all (same weights): no other token or the memory sees them; the
      output differs from the unablated one; the player token's own item inputs still matter; the fused layout too;
  (b) --ablate global: gate and mem_in zero; == a reference built directly (gate 0, mem_in 0 and the global token's
      key masked in every key set: a TokPolicy subclass's extra-token mask / a Block subclass's key mask), bit for bit;
      ~= a model without the global token (same weights); differs from gate / mem_in zeroing alone and from the
      unablated one; --ablate both == both references, ~= a model with neither token; the fused layout too;
  (c) --fused-override: a checkpoint file round trip, the model is fused (config item_tokens_fused) and runs act with a
      memory state; == a fused model loaded with the same state; --item-slots override; refusals;
  (d) defaults (no flag): build_eval_model + apply_ablation('') == HEAD's recipe (TokPolicy(**config, items) +
      load_compatible of git HEAD's tok_policy), bit for bit; nothing in the state dict changes;
  (e) --no-pedestals: the Lua snippet's text, no_pedestal_due on synthetic records, the config / stats defaults and
      tok_floor's call site (evaluation only, before the step). The Lua itself runs only in the game.
"""
import copy
import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_memory_smoke as M   # noqa: E402  (its path set-up, gymnasium stub, synthetic records, HEAD loader)

import numpy as np   # noqa: E402
import torch   # noqa: E402

import eval_tok as E   # noqa: E402
from isaac_bridge import tok_policy as TP   # noqa: E402
from isaac_bridge.tok_obs import ENT_CAP   # noqa: E402

FAIL = []
D = 16
CFG = dict(width=64, layers=2, heads=4, items=True, charge=True, ent_ext=True, pchar=True, memory=D,
           item_tokens=True, global_token=True, item_shared=True)


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def rows_with_items(rng, n):
    """Synthetic records, every one holding 1-3 collectibles (the item tokens present)."""
    r = M.synth_rows(rng, n)
    r['inv'][:] = 0
    for i in range(n):
        k = 1 + i % 3
        r['inv'][i, :k, 0] = rng.integers(1, 700, k)
        r['inv'][i, :k, 1] = 1
    return r


def open_up(m, seed=5):
    """Trained-looking gates, mem_in and mem_out (zero at a fresh start)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.itok.gate.copy_(torch.randn(m.itok.gate.shape, generator=g) * 0.3)
        m.gtok.gate.copy_(torch.randn(m.gtok.gate.shape, generator=g) * 0.3)
        m.gtok.mem_in.weight.copy_(torch.randn(m.gtok.mem_in.weight.shape, generator=g) * 0.05)
        m.gtok.mem_in.bias.copy_(torch.randn(m.gtok.mem_in.bias.shape, generator=g) * 0.05)
        m.mem_out.weight.copy_(torch.randn(m.mem_out.weight.shape, generator=g) * 0.1)


class MaskedItemTokens(TP.ItemTokens):
    """The reference of --ablate items: every item token absent."""

    def forward(self, b, compact=False):
        tok, present = super().forward(b, compact)
        return tok, present & False


class NoGlobalKey(TP.TokPolicy):
    """Reference (separate layout): the global token masked in the extra tokens' key mask."""

    def _ext_tokens(self, b):
        ext = super()._ext_tokens(b)
        yb = ext['ybias'].clone()
        yb[..., ext['glob']] = -1e9
        ext['ybias'] = yb
        return ext


class LastKeyMasked(TP.Block):
    """Reference (fused layout): the last token (the global one) masked as a key."""

    def forward(self, x, bias, ext=None, vscale=None):
        bias = bias.clone()
        bias[..., -1] = -1e9
        return super().forward(x, bias, ext, vscale)


def global_reference(m0):
    """m0's copy with gate 0, mem_in 0 and the global token absent from every key set (built directly)."""
    ref = copy.deepcopy(m0)
    with torch.no_grad():
        ref.gtok.gate.zero_()
        ref.gtok.mem_in.weight.zero_()
        ref.gtok.mem_in.bias.zero_()
    if ref.fused:
        for blk in ref.blocks:
            blk.__class__ = LastKeyMasked
    else:
        ref.__class__ = NoGlobalKey
    return ref


def outs(model, b, h):
    """forward (padded, with memory: logits, value, danger, h_new) and the eval's act (greedy, with memory)."""
    with torch.no_grad():
        f = model(b, mem=h)
        a = model.act(b, greedy=True, mem=h)
    return list(f) + [a[0], a[1], a[2], a[3]]


def without(m_cfg, drop):
    return {k: v for k, v in m_cfg.items() if k not in drop}


def load_subset(model, state):
    """A model without some of the options, loaded with the matching tensors of a full state (strict)."""
    own = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items() if k in own}, strict=True)
    return model


def main():
    rng = np.random.default_rng(3)
    dev = torch.device('cpu')
    torch.manual_seed(1)
    base = TP.TokPolicy(**CFG).eval()
    open_up(base)
    n = 9
    rows = rows_with_items(rng, n)
    b = TP.to_batch(rows, M.synth_pending(rng, n, base.heads), dev)
    h = torch.randn(n, D)
    o_base = outs(base, b, h)
    state = base.state_dict()

    fused = TP.TokPolicy(**dict(CFG, item_tokens_fused=True)).eval()
    fused.load_state_dict(TP.convert_to_fused(fused, state)[0])
    o_fused = outs(fused, b, h)

    # ------------------------------------------------------------ (a) --ablate items
    for tag, m0, o0 in (('separate', base, o_base), ('fused', fused, o_fused)):
        m = copy.deepcopy(m0)
        E.apply_ablation(m, 'items')
        o_abl = outs(m, b, h)
        ref = copy.deepcopy(m0)
        ref.itok.__class__ = MaskedItemTokens
        if not m0.fused:
            with torch.no_grad():
                ref.itok.gate.zero_()
            check(f'(a) items [{tag}]: ItemTokens.gate zeroed', bool((m.itok.gate == 0).all()))
        else:
            check(f'(a) items [{tag}]: gate kept (masking is enough)', torch.equal(m.itok.gate, m0.itok.gate))
        o_ref = outs(ref, b, h)
        check(f'(a) items [{tag}] == direct reference (masked, gated off), bit for bit', M.same(o_abl, o_ref),
              f'max diff {M.maxdiff(o_abl, o_ref):.1e}')
        none = TP.TokPolicy(**without(dict(CFG, item_tokens_fused=m0.fused), ('item_tokens', 'item_shared'))).eval()
        load_subset(none, m0.state_dict())
        o_none = outs(none, b, h)
        d_none = M.maxdiff(o_abl[:4], o_none[:4])
        check(f'(a) items [{tag}] ~= a model without item tokens (same weights)', d_none < 1e-5, f'max diff {d_none:.1e}')
        d0 = M.maxdiff(o_abl[:4], o0[:4])
        check(f'(a) items [{tag}] differs from the unablated output (held items present)', d0 > 1e-4,
              f'max diff {d0:.1e}')
        rows2 = rows.copy()
        rows2['inv'][:, :, 0] = np.where(rows2['inv'][:, :, 0] > 0, (rows2['inv'][:, :, 0] + 37) % 700 + 1, 0)
        b2 = TP.to_batch(rows2, b['pending'].numpy(), dev)
        d_own = M.maxdiff(outs(m, b2, h)[:4], o_abl[:4])
        check(f'(a) items [{tag}]: the player token\'s own item inputs still matter', d_own > 1e-5,
              f'max diff {d_own:.1e} with other held ids')

    # ------------------------------------------------------------ (b) --ablate global / both
    for tag, m0, o0 in (('separate', base, o_base), ('fused', fused, o_fused)):
        m = copy.deepcopy(m0)
        E.apply_ablation(m, 'global')
        check(f'(b) global [{tag}]: gate and mem_in zero',
              bool((m.gtok.gate == 0).all()) and bool((m.gtok.mem_in.weight == 0).all())
              and bool((m.gtok.mem_in.bias == 0).all()))
        check(f'(b) global [{tag}]: item gate untouched', torch.equal(m.itok.gate, m0.itok.gate))
        ref = global_reference(m0)
        o_abl, o_ref = outs(m, b, h), outs(ref, b, h)
        check(f'(b) global [{tag}] == gate 0, mem_in 0, key masked (built directly), bit for bit', M.same(o_abl, o_ref),
              f'max diff {M.maxdiff(o_abl, o_ref):.1e}')
        nog = TP.TokPolicy(**without(dict(CFG, item_tokens_fused=m0.fused), ('global_token',))).eval()
        load_subset(nog, m0.state_dict())
        d_nog = M.maxdiff(o_abl[:4], outs(nog, b, h)[:4])
        check(f'(b) global [{tag}] ~= a model without the global token (same weights)', d_nog < 1e-5,
              f'max diff {d_nog:.1e}')
        zo = copy.deepcopy(m0)
        with torch.no_grad():
            zo.gtok.gate.zero_()
            zo.gtok.mem_in.weight.zero_()
            zo.gtok.mem_in.bias.zero_()
        d_zo = M.maxdiff(o_abl[:4], outs(zo, b, h)[:4])
        check(f'(b) global [{tag}]: the key mask matters (differs from gate / mem_in zeroing alone)', d_zo > 1e-5,
              f'max diff {d_zo:.1e}')
        d0 = M.maxdiff(o_abl[:4], o0[:4])
        check(f'(b) global [{tag}] differs from the unablated output', d0 > 1e-4, f'max diff {d0:.1e}')
        both = copy.deepcopy(m0)
        E.apply_ablation(both, 'both')
        rb = global_reference(m0)
        rb.itok.__class__ = MaskedItemTokens
        if not m0.fused:
            with torch.no_grad():
                rb.itok.gate.zero_()
        o_both, o_rb = outs(both, b, h), outs(rb, b, h)
        check(f'(b) both [{tag}] == both references, bit for bit', M.same(o_both, o_rb),
              f'max diff {M.maxdiff(o_both, o_rb):.1e}')
        plain = load_subset(TP.TokPolicy(**without(CFG, ('item_tokens', 'item_shared', 'global_token'))).eval(),
                            m0.state_dict())
        d_pl = M.maxdiff(o_both[:4], outs(plain, b, h)[:4])
        check(f'(b) both [{tag}] ~= a model with neither token (same weights)', d_pl < 1e-5, f'max diff {d_pl:.1e}')
    try:
        g_ = copy.deepcopy(fused)
        E.apply_ablation(g_, 'global')
        with torch.no_grad():
            g_(TP.to_batch(rows, b['pending'].numpy(), dev), packed=True, mem=h)
        check('(b) global [fused]: packed layers refuse', False)
    except NotImplementedError:
        check('(b) global [fused]: packed layers refuse', True)

    # ------------------------------------------------------------ (c) --fused-override (checkpoint round trip)
    buf = io.BytesIO()
    torch.save(dict(config=dict(base.config), model=state, update=7), buf)
    buf.seek(0)
    ck = torch.load(buf, map_location='cpu')
    m, changed, note = E.build_eval_model(ck, True, dev, fused_override=True)
    check('(c) fused-override: model fused, config item_tokens_fused',
          m.fused and m.config.get('item_tokens_fused') is True and not changed and bool(note), f'changed {changed}')
    o_m = outs(m, b, h)
    check('(c) fused-override == a fused model loaded with the same state, bit for bit', M.same(o_m, o_fused),
          f'max diff {M.maxdiff(o_m, o_fused):.1e}')
    check('(c) fused-override: act with a memory state runs (actions, new state)',
          o_m[4].shape == (n, len(m.heads)) and o_m[7].shape == (n, D))
    m3, _, _ = E.build_eval_model(ck, True, dev, fused_override=True, item_slots=3)
    o3 = outs(m3, b, h)
    check('(c) --item-slots 3: 3 held slots, config item_slots, outputs finite',
          m3.itok.slots == 3 and m3.itok.n_tok == 7 and m3.config.get('item_slots') == 3
          and all(bool(torch.isfinite(x.float()).all()) for x in o3))
    ck_f = dict(ck, config=dict(m.config), model=m.state_dict())
    for what, fn in (('fused checkpoint', lambda: E.build_eval_model(ck_f, True, dev, fused_override=True)),
                     ('no extra tokens', lambda: E.build_eval_model(
                         dict(ck, config=without(ck['config'], ('item_tokens', 'global_token', 'item_shared'))),
                         True, dev, fused_override=True)),
                     ('ablate items without item tokens', lambda: E.apply_ablation(
                         TP.TokPolicy(**without(CFG, ('item_tokens', 'item_shared'))), 'items')),
                     ('ablate unknown', lambda: E.apply_ablation(copy.deepcopy(base), 'pedestals'))):
        try:
            fn()
            ok = False
        except ValueError:
            ok = True
        check(f'(c) refused: {what}', ok)

    # ------------------------------------------------------------ (d) defaults = HEAD's recipe
    m_d, changed_d, note_d = E.build_eval_model(ck, True, dev)
    sd0 = {k: v.clone() for k, v in m_d.state_dict().items()}
    E.apply_ablation(m_d, '')
    check('(d) default: not fused, no note, nothing changed by apply_ablation(\'\')',
          not m_d.fused and note_d is None and all(torch.equal(sd0[k], v) for k, v in m_d.state_dict().items())
          and 'forward' not in m_d.itok.__dict__)
    o_d = outs(m_d, b, h)
    check('(d) default == the checkpoint\'s model, bit for bit', M.same(o_d, o_base),
          f'max diff {M.maxdiff(o_d, o_base):.1e}')
    old = M.head_module()
    if old is None:
        print('skip (d) HEAD comparison: git HEAD not readable')
    else:
        ref = old.TokPolicy(**dict(ck['config'], items=True)).to(dev).eval()   # HEAD's eval_tok recipe
        ch_ref = old.load_compatible(ref, ck['model'])
        o_h = outs(ref, b, h)
        check('(d) default == HEAD (TokPolicy(**config, items) + load_compatible), bit for bit',
              M.same(o_d, o_h) and changed_d == ch_ref, f'max diff {M.maxdiff(o_d, o_h):.1e}')

    # ------------------------------------------------------------ (e) --no-pedestals
    from isaac_bridge import tok_pickup as PK
    from isaac_bridge.tok_obs import ROW
    from isaac_bridge.tok_sampler import STATS, TokSamplerConfig
    lua = PK.NO_PEDESTAL_LUA
    check('(e) Lua: every type 5 / variant 100 entity removed, the count returned',
          'Isaac.GetRoomEntities()' in lua and 'e.Type == 5 and e.Variant == 100' in lua and 'e:Remove()' in lua
          and 'return tostring(n)' in lua
          and len(re.findall(r'\b(?:do|then|function)\b', lua)) == len(re.findall(r'\bend\b', lua)) == 2)
    r = np.zeros(1, ROW)
    r['n_ent'] = 3
    r['ent_id'][0, :3] = [[10, 0, 0], [5, 10, 1], [5, 100, 0]]   # (an empty pedestal counts too)
    due_peds = PK.no_pedestal_due(r)
    r2 = r.copy()
    r2['ent_id'][0, 2] = [5, 20, 0]
    r2['ent_id'][0, 5] = [5, 100, 33]   # beyond n_ent: not seen
    due_none = PK.no_pedestal_due(r2)
    r3 = r2.copy()
    r3['n_ent'] = ENT_CAP
    check('(e) no_pedestal_due: pedestal seen -> True, none -> False, ENT_CAP entities -> True',
          due_peds and not due_none and PK.no_pedestal_due(r3) and PK.no_pedestal_due(r[0]))
    check('(e) TokSamplerConfig.no_pedestals off by default; stats np_runs / np_removed appended last',
          TokSamplerConfig().no_pedestals is False and STATS[-2:] == ('np_runs', 'np_removed'))
    src = (HERE.parent / 'isaac_bridge' / 'tok_floor.py').read_text(encoding='utf-8')
    i_call = src.find('episode.lua(NO_PEDESTAL_LUA)')
    i_step = src.find('step_action(episode, action', i_call)
    i_send = src.find("conn.send_bytes(b'1' if t_ep & 1 else b'0')", i_call)
    check('(e) tok_floor: evaluation only, once per record that shows a pedestal, before the record goes out and the '
          'step', "np_on = bool(getattr(cfg, 'no_pedestals', False)) and evaluation" in src
          and 'if np_on and not done and no_pedestal_due(row):' in src and 0 < i_call < i_send < i_step)
    ev = (HERE.parent / 'eval_tok.py').read_text(encoding='utf-8')
    check('(e) eval_tok passes no_pedestals to the sampler and records it',
          'no_pedestals=bool(args.no_pedestals)' in ev and "info['no_pedestals'] = True" in ev)

    print('FAILED: ' + ', '.join(FAIL) if FAIL else 'all ok', flush=True)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
