"""CPU smoke test of B23 (2026-10-13): the shared collectible table (TokPolicy(item_shared=True), train_tok --item-shared
1), the learner's handling of forced pickup decisions (tok_pickup.ppo_part_terms / normalise_advantages, train_tok
--pickup-force) and the worker's forced move (tok_pickup.forced_move / PickupForcer).

No game, no GPU. Plain script:
  cd rl/bridge/python && python tests/test_pickup_smoke.py
Checks:
  (a) options off == git HEAD's policy module, bit for bit, on every path (plain, items, items + memory, and B21 item
      tokens / B22 global token with open gates, not shared); ROW's old fields keep HEAD's offsets;
  (b) item_shared: itok.coll is item_emb (one parameter, no itok.coll key), forward / backward work, and item_emb gets
      gradient from the pedestal path (ids only on pedestals) and from the held-item path through the item tokens
      (the gradient with the gates open minus the gradient with the gates closed is non-zero on the held ids);
  (c) a synthetic B21 checkpoint (own 32-column itok.coll) into an item_shared model through share_item_table /
      load_compatible: every other tensor kept, the first Linear's 44 other columns kept, its 16 collectible columns
      the fresh model's, the log line; Adam state remapped with the converted Linear dropped (a step runs); a
      checkpoint without item tokens / without memory loads as before;
  (d) the learner's masking on a tiny rollout: the policy term and the entropy are zero (value and gradient) on forced
      decisions, the imitation term is non-zero only on them, the value loss covers all; advantages normalised over
      the non-forced decisions;
  (e) forced_move: the 8-direction index toward a pedestal at several offsets, the nearest pedestal, empty pedestals
      and other entities ignored; PickupForcer: clear rooms only, priced rooms skipped, counts, the cap.
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_memory_smoke as M   # noqa: E402  (its path set-up, synthetic records, HEAD loader)
import test_item_global_smoke as G   # noqa: E402

import numpy as np   # noqa: E402
import torch   # noqa: E402

from isaac_bridge import tok_policy as TP   # noqa: E402
from isaac_bridge import tok_pickup as PK   # noqa: E402
from isaac_bridge.tok_obs import ROW   # noqa: E402

FAIL = []
D = 32
BASE = G.BASE


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def randomise(m, seed):
    """Every parameter non-default (a trained-looking checkpoint)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def head_rows():
    """ROW of git HEAD's tok_obs (None if not readable)."""
    import subprocess
    try:
        src = subprocess.run(['git', '-C', str(M.PY), 'show', 'HEAD:bridge/python/isaac_bridge/tok_obs.py'],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    ns = {'__name__': 'isaac_bridge._tok_obs_head', '__package__': 'isaac_bridge'}
    exec(compile(src, 'HEAD:tok_obs.py', 'exec'), ns)
    return ns['ROW']


def main():
    rng = np.random.default_rng(23)
    dev = torch.device('cpu')

    # ------------------------------------------------------------ (a) off == HEAD
    old = M.head_module()
    cases = (('plain', dict(), False), ('items', BASE, False), ('items+memory', dict(BASE, memory=D), False),
             ('B21 item tokens, gates open', dict(BASE, item_tokens=True), True),
             ('B21 + B22 + memory, gates open', dict(BASE, memory=D, item_tokens=True, global_token=True), True))
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
        check(f'(a) no item_shared key [{tag}]', 'item_shared' not in new.config)
        if old is None:
            print('skip (a) HEAD comparison: git HEAD not readable')
            continue
        ref = old.TokPolicy(**cfg).eval()
        ref.load_state_dict(new.state_dict())
        o_new, o_ref = G.outputs(new, b, acts, u, mem, memd), G.outputs(ref, b, acts, u, mem, memd)
        check(f'(a) == HEAD, all paths [{tag}]', M.same(o_new, o_ref) and len(o_new) == len(o_ref),
              f'{len(o_new)} tensors, max diff {M.maxdiff(o_new, o_ref):.1e}')
    row_head = head_rows()
    if row_head is not None:
        moved = [k for k in row_head.names if ROW.fields[k][1] != row_head.fields[k][1]]
        check('(a) ROW: the old fields keep their offsets, B23 fields appended',
              not moved and ROW.names[:len(row_head.names)] == row_head.names and
              ROW.names[-3:] == ('pforce', 'pmode', 'bped'),   # (HEAD has them since 7003c76: then nothing appended)
              f'HEAD {row_head.itemsize} B, now {ROW.itemsize} B')

    # ------------------------------------------------------------ (b) shared table: forward / backward, gradients
    torch.manual_seed(2)
    sh = TP.TokPolicy(**BASE, memory=D, item_tokens=True, global_token=True, item_shared=True)
    names = [k for k, _ in sh.named_parameters()]
    check('(b) itok.coll is item_emb, one parameter, no itok.coll key',
          sh.itok.coll is sh.item_emb and 'itok.coll.weight' not in sh.state_dict() and
          'itok.coll.weight' not in names and names.count('item_emb.weight') == 1 and sh.config.get('item_shared'),
          f'item-token input {sh.itok.mlp[0].in_features} columns (B21: {TP.IT_F})')
    check('(b) item_shared needs item tokens', _raises(lambda: TP.TokPolicy(**BASE, item_shared=True)))
    n = 10
    rows = M.synth_rows(rng, n)
    ped_ids = np.array([701, 702, 703])          # only on pedestals
    held_ids = np.array([801, 802])              # only held
    rows['ent_item'][..., 0] = 0
    rows['inv'][..., 0] = 0
    rows['pitem'][:, 0] = 0
    rows['n_ent'] = np.maximum(rows['n_ent'], 3)
    rows['ent_item'][:, 0, 0], rows['ent_item'][:, 1, 0], rows['ent_item'][:, 2, 0] = ped_ids
    rows['inv'][:, 0, 0], rows['inv'][:, 1, 0] = held_ids
    rows['inv'][:, :2, 1] = 1
    b = TP.to_batch(rows, M.synth_pending(rng, n, sh.heads), dev)
    acts = torch.from_numpy(M.synth_pending(rng, n, sh.heads))
    adv, ret = torch.randn(n), torch.randn(n)
    memd = G.chunk_dict(n)
    with torch.no_grad():
        sh.mem_out.weight.normal_(0, 0.1)

    def emb_grad(m):
        m.zero_grad()
        logits, value, _, _ = m(G.fast_batch(b), packed=True, mem=memd)
        logp, ent = m.log_prob(logits, m.gate(b), acts)
        (-(logp * adv).mean() + 0.5 * ((value - ret) ** 2).mean() - 0.01 * ent.sum(-1).mean()).backward()
        return m.item_emb.weight.grad.clone()
    sh.train()
    g_closed = emb_grad(sh)
    check('(b) backward runs, pedestal path: gradient on the pedestal-only ids',
          bool(torch.isfinite(g_closed).all()) and all(float(g_closed[i].abs().sum()) > 0 for i in ped_ids),
          ', '.join(f'{i}: {float(g_closed[i].abs().sum()):.2e}' for i in ped_ids))
    G.open_gates(sh)
    g_open = emb_grad(sh)
    dh = [float((g_open[i] - g_closed[i]).abs().sum()) for i in held_ids]
    check('(b) held path: the item tokens add gradient on the held ids (gates open - gates closed != 0)',
          all(v > 0 for v in dh), ', '.join(f'{i}: {v:.2e}' for i, v in zip(held_ids, dh)))
    with torch.no_grad():   # (the item tokens alone: their table is item_emb)
        tok, present = sh.itok(b)
    check('(b) item tokens read item_emb', bool(present[:, :2].all()) and tok.shape[1] == TP.N_ITEM_TOK)
    sh.zero_grad()

    # ------------------------------------------------------------ (c) B21 checkpoint into item_shared
    torch.manual_seed(4)
    b21 = TP.TokPolicy(**BASE, memory=D, item_tokens=True, global_token=True)
    randomise(b21, 5)
    sd = {k: v.clone() for k, v in b21.state_dict().items()}
    torch.manual_seed(6)
    new = TP.TokPolicy(**BASE, memory=D, item_tokens=True, global_token=True, item_shared=True)
    fresh_cols = new.state_dict()[TP.ITEM_SHARED_KEY][:, :new.itok.dc].clone()
    conv, note = TP.share_item_table(new, sd)
    missing, unexpected = new.load_state_dict(conv, strict=True), None
    got = new.state_dict()
    others = [k for k in sd if k not in ('itok.coll.weight', TP.ITEM_SHARED_KEY)]
    kept = all(torch.equal(got[k], sd[k]) for k in others)
    w_new, w_old = got[TP.ITEM_SHARED_KEY], sd[TP.ITEM_SHARED_KEY]
    check('(c) every other tensor kept (gates, slot / kind, mlp rest, item_emb, ...)', kept and len(others) == len(got) - 1,
          f'{len(others)} tensors')
    check('(c) first Linear: 44 other columns kept, 16 collectible columns = the fresh model\'s',
          w_new.shape == (w_old.shape[0], w_old.shape[1] - 16) and torch.equal(w_new[:, 16:], w_old[:, 32:]) and
          torch.equal(w_new[:, :16], fresh_cols), f'{tuple(w_old.shape)} -> {tuple(w_new.shape)}')
    check('(c) the log line', note is not None and 'itok.coll' in note and 're-initialised' in note, note)
    print('     ' + str(note))
    check('(c) the input state dict is not changed', 'itok.coll.weight' in sd and sd[TP.ITEM_SHARED_KEY].shape[1] == 76)
    new2 = TP.TokPolicy(**BASE, memory=D, item_tokens=True, global_token=True, item_shared=True)
    changed = TP.load_compatible(new2, sd)
    check('(c) load_compatible: converted, nothing else changed', len(changed) == 1 and 'shared' in changed[0] and
          torch.equal(new2.state_dict()['itok.gate'], sd['itok.gate']), str(changed)[:120])
    # the optimiser: Adam over the B21 model's main parameters, one step, remapped with the converted Linear dropped
    opt_old = torch.optim.Adam(b21.main_parameters(), lr=1e-3)
    nb = TP.to_batch(M.synth_rows(rng, 6), M.synth_pending(rng, 6, b21.heads), dev)
    G.open_gates(b21)
    b21(nb, need_vg=True)[-1].sum().backward()
    opt_old.step()
    osd = opt_old.state_dict()
    old_names = TP.main_parameter_names(list(sd))
    new_names = TP.main_parameter_names([k for k, _ in new.named_parameters()])
    added = TP.remap_optimizer_state(osd, old_names, new_names, drop=(TP.ITEM_SHARED_KEY,))
    j_lin = new_names.index(TP.ITEM_SHARED_KEY)
    shapes_ok = all(osd['state'][j]['exp_avg'].shape == p.shape for j, p in enumerate(new.main_parameters())
                    if j in osd['state'])
    opt_new = torch.optim.Adam(new.main_parameters(), lr=1e-3)
    opt_new.load_state_dict(osd)
    new(nb, need_vg=True, mem=torch.zeros(6, D))[-1].sum().backward()
    opt_new.step()
    check('(c) Adam state: remapped by name, the converted Linear without state, a step runs',
          j_lin not in osd['state'] and shapes_ok and added == 1 and len(new_names) == len(old_names) - 1,
          f'{added} tensor(s) without state, {len(osd["state"])} with')
    # checkpoints without item tokens / without memory: as before
    for tag, cfg_old, cfg_new in (('items-free', dict(), dict()), ('memory-free items', BASE, BASE),
                                  ('items, no item tokens -> shared', BASE, dict(BASE, item_tokens=True,
                                                                                   item_shared=True))):
        src = TP.TokPolicy(**cfg_old)
        st = src.state_dict()
        dst = TP.TokPolicy(**cfg_new)
        st2, note2 = TP.share_item_table(dst, st)
        ok_ = note2 is None and st2 is st
        if cfg_new.get('item_tokens'):   # (fresh item tokens: missing itok.* keys only, as B21's resume)
            mi, un = dst.load_state_dict(st2, strict=False)
            ok_ = ok_ and not un and all(k.startswith('itok.') for k in mi) and 'itok.coll.weight' not in mi
        else:
            dst.load_state_dict(st2)
        check(f'(c) {tag} checkpoint: loads as before', ok_)

    # ------------------------------------------------------------ (d) the learner's masking (tiny rollout)
    roll = np.zeros(8, ROW)
    roll['pforce'][[2, 5]] = (3, 7)   # two forced decisions (moves right / left)
    forced = torch.from_numpy(roll['pforce'] > 0)
    nh = 5
    torch.manual_seed(9)
    logits = torch.randn(8, sum(TP.ITEM_HEADS), requires_grad=True)
    value = torch.randn(8, requires_grad=True)
    ret = torch.randn(8)
    acts = torch.from_numpy(M.synth_pending(rng, 8, TP.ITEM_HEADS))
    acts[forced, 0] = torch.from_numpy(roll['pforce'][roll['pforce'] > 0].astype(np.int64))
    acts[forced, 1] = 0
    gate = torch.ones(8, 3, dtype=torch.bool)
    pol = TP.TokPolicy(**BASE)
    logp, entropy = pol.log_prob(logits, gate, acts)
    b_logp = (logp.detach() + 0.05 * torch.randn(8))   # (inside the clip range: every kept row has a gradient)
    b_logp[forced] = -60.0   # (the sampled action's log-probability: far from the forced action's)
    adv_raw = torch.randn(8)
    a, std = PK.normalise_advantages(adv_raw, forced)
    keep = ~forced
    check('(d) advantages: 0 on forced, mean 0 / std 1 over the others',
          bool((a[forced] == 0).all()) and abs(float(a[keep].mean())) < 1e-6 and abs(float(a[keep].std()) - 1) < 1e-4)
    pg, vf, ent, imit, ratio, kl, clipped = PK.ppo_part_terms(logits, logp, entropy, b_logp, a, value, ret, acts,
                                                              forced, 0.2)
    check('(d) forced ratio exactly 1 (no overflow from the sampled action\'s log-probability)',
          bool((ratio[forced] == 1).all()) and bool(torch.isfinite(pg)))

    def row_grads(loss):
        if logits.grad is not None:
            logits.grad = None
        if value.grad is not None:
            value.grad = None
        loss.backward(retain_graph=True)
        gl = logits.grad.abs().sum(1) if logits.grad is not None else torch.zeros(8)
        gv = value.grad.abs() if value.grad is not None else torch.zeros(8)
        return gl, gv
    gl_pg, _ = row_grads(pg)
    gl_ent, _ = row_grads(-ent)
    gl_im, _ = row_grads(imit)
    _, gv_vf = row_grads(vf)
    check('(d) policy term: zero gradient on forced, non-zero on the others',
          bool((gl_pg[forced] == 0).all()) and bool((gl_pg[keep] > 0).all()), f'{gl_pg.tolist()}')
    check('(d) entropy: zero gradient on forced', bool((gl_ent[forced] == 0).all()) and bool((gl_ent[keep] > 0).all()))
    check('(d) imitation: non-zero only on forced', bool((gl_im[forced] > 0).all()) and bool((gl_im[keep] == 0).all()),
          f'imit {float(imit.detach()):.3f}')
    check('(d) value loss covers all decisions', bool((gv_vf > 0).all()))
    lm = torch.log_softmax(logits[:, :9], -1)[torch.arange(8), acts[:, 0]]
    ls = torch.log_softmax(logits[:, 9:14], -1)[torch.arange(8), acts[:, 1]]
    check('(d) imitation = mean over the part of -forced x (log pi_move + log pi_shoot)',
          abs(float(imit.detach()) - float(-((lm + ls) * forced.float()).mean().detach())) < 1e-6)
    ratio_k = torch.exp(logp - b_logp)
    pg_ref = -(torch.min(ratio_k * a, torch.clamp(ratio_k, 0.8, 1.2) * a) * keep.float()).sum() / 8
    check('(d) policy term = the plain clipped term over the non-forced rows', abs(float(pg.detach()) - float(pg_ref.detach())) < 1e-6)
    pm = torch.tensor([1, 1, 1, 0, 1, 1, 0, 1], dtype=torch.float32)   # (memory: two rows only stepped through)
    a_m, _ = PK.normalise_advantages(adv_raw, forced, pm)
    pg_m, vf_m, *_ = PK.ppo_part_terms(logits, logp, entropy, b_logp, a_m, value, ret, acts, forced, 0.2, pm,
                                       float(pm.sum()))
    check('(d) memory weights: padding rows and forced rows out of the advantages',
          bool((a_m[(pm == 0) | forced] == 0).all()) and bool(torch.isfinite(pg_m)) and bool(torch.isfinite(vf_m)))
    prox = logp.detach() + 0.1
    pg_d, *_ = PK.ppo_part_terms(logits, logp, entropy, b_logp, a, value, ret, acts, forced, 0.2, prox=prox,
                                 weight_max=2.0)
    check('(d) decoupled: finite, forced rows contribute nothing', bool(torch.isfinite(pg_d)) and
          bool((row_grads(pg_d)[0][forced] == 0).all()))
    _ = nh

    # ------------------------------------------------------------ (e) forced_move and PickupForcer
    def record(ents, clear=1.0):
        r = np.zeros(1, ROW)
        r['player'][0, 23] = clear
        for j, (dx, dy, typ, var, item) in enumerate(ents):
            r['ent'][0, j, 0], r['ent'][0, j, 1] = dx / 200, dy / 200
            r['ent'][0, j, 2] = math.hypot(dx, dy) / 300
            r['ent_id'][0, j] = (typ, var, item % 256)
            r['ent_item'][0, j, 0] = item if (typ, var) == (5, 100) else 0
        r['n_ent'] = len(ents)
        return r
    expect = {(100, 0): 3, (0, -100): 1, (0, 100): 5, (-100, 0): 7, (70, -70): 2, (70, 70): 4, (-70, 70): 6,
              (-70, -70): 8, (100, -30): 3, (30, -100): 1, (100, -60): 2, (-90, 50): 6}
    got_m = {k: PK.forced_move(record([(k[0], k[1], 5, 100, 12)]))[0] for k in expect}
    check('(e) forced_move: the 8-direction index toward the pedestal', got_m == expect,
          ', '.join(f'{k}->{v}' for k, v in got_m.items()))
    mixed = record([(-30, 0, 10, 0, 0), (50, 50, 5, 100, 0), (0, -120, 5, 100, 7), (200, 0, 5, 100, 9),
                    (0, 40, 5, 10, 0)])
    check('(e) nearest pedestal with an item; empty pedestals, other entities ignored',
          PK.forced_move(mixed) == (1, 7) and PK.forced_move(mixed, {9}) == (3, 9) and
          PK.forced_move(record([(10, 10, 20, 0, 0)])) == (0, 0))
    f = PK.PickupForcer(cap=5)
    c = dict(steps=0, forced=0, free=0)
    f.begin(True)
    r = record([(100, 0, 5, 100, 12)])
    check('(e) forcer: not while the room is not clear', f.step(record([(100, 0, 5, 100, 12)], clear=0.0), (1, 84),
                                                              False, 0, c) == 0)
    seq = [f.step(r, (1, 84), False, 0, c) for _ in range(7)]
    check('(e) forcer: forced moves, then gives up after the cap', seq == [3] * 5 + [0, 0] and c['steps'] == 5)
    f.reset_room(None)
    f.begin(True)
    f.step(r, (1, 85), False, 0, c)
    f.step(record([]), (1, 85), True, 0, c)   # (the item gained: a forced pickup)
    f.begin(False)
    f.step(r, (1, 86), True, 0, c)            # (gained in an unforced episode: a free pickup)
    check('(e) forcer: forced / free pickups counted', c['forced'] == 1 and c['free'] == 1, str(c))
    shop = record([(100, 0, 5, 100, 12)])
    shop['map'][0, 3, 5, 5], shop['map'][0, 6, 5, 5] = 1, 1
    f.begin(True)
    check('(e) forcer: a shop (priced pedestals) is skipped', f.step(shop, (1, 87), False, 0, c) == 0)
    check('(e) parse_pickup_force / pickup_force_p', PK.parse_pickup_force('1:0') == (1.0, 0.0) and
          PK.parse_pickup_force('0:0') == (0.0, 0.0) and PK.pickup_force_p((1.0, 0.0), 0.25) == 0.75 and
          _raises(lambda: PK.parse_pickup_force('2:0')))
    lua = PK.spawn_lua([12, 34])
    check('(e) spawn_lua: ids in the chunk, braces balanced, parse_spawned',
          '{12, 34}' in lua and lua.count('{') == lua.count('}') and 'FindFreePickupSpawnPosition' in lua and
          PK.parse_spawned('12:200:280,34:440:280') == [(12, 200.0, 280.0), (34, 440.0, 280.0)] and
          PK.parse_spawned(None) == [])

    print('RESULT', 'FAIL ' + ', '.join(FAIL) if FAIL else 'all checks passed', flush=True)
    return 1 if FAIL else 0


def _raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


if __name__ == '__main__':
    sys.exit(main())
