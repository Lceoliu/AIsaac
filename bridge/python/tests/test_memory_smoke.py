"""CPU smoke test of the token policy's optional GRU memory (2026-10-11; TokPolicy(memory=D), train_tok --memory gru).

No game, no GPU needed (the GraphActor part runs only where CUDA is available). Plain script:
  cd rl/bridge/python && python tests/test_memory_smoke.py
Checks:
  (a) memory off: the outputs (forward, packed forward, act with uniforms, evaluate) are bit-identical to the policy
      module of git HEAD (the code before this change; skipped if git or HEAD's file is not available), and a memory
      model loaded from a memory-free state (fresh GRU, zero output projection) computes the same logits and values;
  (b) actor side: the state changes from step to step and is zero at an episode's first record (the actor's reset);
  (c) learner side: a chunked forward (truncated BPTT over T records per worker, from the stored state at each chunk's
      start, resets inside the chunk) reproduces the actor's per-step outputs and states; memory_chunks' layout; a
      backward pass reaches the GRU's parameters (once the zero-initialised output projection has moved);
  (d) checkpoint round trip (memory config and weights), resume of a memory-free checkpoint into a memory model
      (missing keys = the memory modules only, Adam's state padded with pad_optimizer_state);
  (e) CUDA only: GraphActor with a memory table = the eager act (values, input states, the table's new rows).
"""
import importlib.util
import io
import os
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE.parent
sys.path.insert(0, str(PY))
try:   # tok_obs imports transformer_obs, which imports gymnasium at module level (not used here)
    import gymnasium  # noqa: F401
except ImportError:
    g, s = types.ModuleType('gymnasium'), types.ModuleType('gymnasium.spaces')

    class _Any:
        def __init__(self, *a, **k):
            pass
    for nm in ('Box', 'Dict', 'Discrete', 'MultiDiscrete', 'Tuple', 'MultiBinary'):
        setattr(s, nm, _Any)
    g.spaces, g.Env, g.Wrapper = s, object, object
    sys.modules['gymnasium'], sys.modules['gymnasium.spaces'] = g, s

import numpy as np
import torch

from isaac_bridge import tok_policy as TP
from isaac_bridge.tok_obs import DOOR_CAP, N_CHAR, ROW

torch.set_num_threads(4)
FAIL = []


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def synth_rows(rng, n):
    """Random ROW records within the policy's input ranges."""
    r = np.zeros(n, ROW)
    r['player'] = rng.normal(size=r['player'].shape)
    r['player'][:, 22] = rng.integers(0, 2, n)
    r['n_ent'] = rng.integers(0, 24, n)
    r['n_doors'] = rng.integers(0, DOOR_CAP + 1, n)
    r['ent'] = rng.normal(size=r['ent'].shape)
    r['ent_id'][..., 0] = rng.integers(0, 1024, r['ent_id'].shape[:2])
    r['ent_id'][..., 1] = rng.integers(0, 1024, r['ent_id'].shape[:2])
    r['ent_id'][..., 2] = rng.integers(0, 256, r['ent_id'].shape[:2])
    r['ent_item'][..., 0] = rng.integers(0, 1024, r['ent_item'].shape[:2])
    r['ent_item'][..., 1] = rng.integers(0, 256, r['ent_item'].shape[:2])
    r['doors'] = rng.normal(size=r['doors'].shape)
    for k in ('patch', 'grid', 'map'):
        r[k] = rng.integers(0, 2, r[k].shape)
    r['bombs'] = rng.integers(0, 3, n)
    r['inv'][..., 0] = rng.integers(0, 1024, r['inv'].shape[:2])
    r['inv'][..., 1] = rng.integers(0, 3, r['inv'].shape[:2])
    hi = np.array([1024, 256, 256, 16, 256, 256])
    r['pitem'] = rng.integers(0, hi[None], (n, 6))
    r['pinv'] = rng.random(r['pinv'].shape)
    r['pcharge'] = rng.random(r['pcharge'].shape)
    r['pchar'] = rng.integers(0, N_CHAR, n)
    return r


def synth_pending(rng, n, heads):
    return np.stack([rng.integers(0, k, n) for k in heads], 1).astype(np.int64)


def head_module():
    """The policy module as it is in git HEAD (the code before the memory change), or None."""
    try:
        src = subprocess.run(['git', '-C', str(PY), 'show', 'HEAD:bridge/python/isaac_bridge/tok_policy.py'],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    name = 'isaac_bridge._tok_policy_head'
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'isaac_bridge'
    exec(compile(src, 'HEAD:tok_policy.py', 'exec'), mod.__dict__)
    return mod


def outputs(model, b, acts, u, **kw):
    """forward (padded and packed), act with Gumbel uniforms, evaluate."""
    with torch.no_grad():
        f = model(b, **kw)[:3]
        fp = model(b, packed=True, **kw)[:3]
        a = model.act(b, u=u, **kw)[:3]
        e = model.evaluate(b, acts, **kw)
    return list(f) + list(fp) + list(a) + list(e)


def same(xs, ys):
    return all(torch.equal(x, y) for x, y in zip(xs, ys))


def maxdiff(xs, ys):
    return max(float((x.float() - y.float()).abs().max()) for x, y in zip(xs, ys))


def main():
    rng = np.random.default_rng(7)
    dev = torch.device('cpu')
    D, W = 32, 4
    kw_items = dict(items=True, charge=True, ent_ext=True, pchar=True)

    # ---------------------------------------------------------------- (a) memory off = HEAD's code, bit for bit
    old = head_module()
    for cfg in (dict(), kw_items):
        torch.manual_seed(1)
        new = TP.TokPolicy(**cfg).eval()
        heads = new.heads
        rows = synth_rows(rng, 12)
        b = TP.to_batch(rows, synth_pending(rng, 12, heads), dev)
        acts = torch.from_numpy(synth_pending(rng, 12, heads))
        u = torch.rand((12, sum(heads)))
        tag = 'items' if cfg else 'plain'
        if old is None:
            print('skip (a) HEAD comparison: git HEAD not readable')
        else:
            ref = old.TokPolicy(**cfg).eval()
            ref.load_state_dict(new.state_dict())
            check(f'(a) memory off == HEAD [{tag}]', same(outputs(new, b, acts, u), outputs(ref, b, acts, u)))
        check(f'(a) no memory key in config [{tag}]', 'memory' not in new.config)
        # a memory model from the memory-free weights: the same logits / values / danger (mem_out starts at zero)
        mem = TP.TokPolicy(memory=D, **cfg).eval()
        missing, unexpected = mem.load_state_dict(new.state_dict(), strict=False)
        check(f'(a) memory model loads memory-free weights [{tag}]',
              not unexpected and all(k.startswith('mem_') for k in missing) and len(missing) == 6, str(missing))
        h_rand = torch.randn(12, D)
        o_ref = outputs(new, b, acts, u)
        check(f'(a) fresh memory (h = 0) == memory-free outputs [{tag}]', same(outputs(mem, b, acts, u), o_ref),
              f'max diff {maxdiff(outputs(mem, b, acts, u), o_ref):.2e}')
        check(f'(a) fresh memory (random h) == memory-free outputs [{tag}]',
              same(outputs(mem, b, acts, u, mem=h_rand), o_ref))

    # ---------------------------------------------------------------- (b) actor: state per worker, reset at a first
    torch.manual_seed(2)
    model = TP.TokPolicy(memory=D, **kw_items).eval()
    with torch.no_grad():   # (a trained-looking output projection, so that the state matters for the outputs)
        model.mem_out.weight.normal_(0, 0.1)
    heads = model.heads
    steps = 9
    seq_rows = [synth_rows(rng, W) for _ in range(steps)]
    firsts = np.zeros((steps, W), bool)
    firsts[0, :] = True       # every worker's episode starts at step 0
    firsts[4, 1] = True       # worker 1 starts a new episode at step 4 (not a room or floor change: a new game)
    firsts[7, 3] = True
    for t in range(steps):
        seq_rows[t]['first'] = firsts[t]
        seq_rows[t]['t'] = 0
    seq_pend = [synth_pending(rng, W, heads) for _ in range(steps)]
    table = torch.zeros((W, D))
    h_in_log, h_new_log, logit_log, value_log = [], [], [], []
    widx = np.arange(W, dtype=np.int64)
    with torch.no_grad():
        for t in range(steps):
            b = TP.to_batch(seq_rows[t], seq_pend[t], dev)
            h_in = table.index_select(0, torch.from_numpy(widx)) * torch.from_numpy(~firsts[t]).float()[:, None]
            logits, value, _, h_new = model(b, mem=h_in)
            table.index_copy_(0, torch.from_numpy(widx), h_new)
            h_in_log.append(h_in.clone())
            h_new_log.append(h_new.clone())
            logit_log.append(logits.clone())
            value_log.append(value.clone())
    check('(b) state zero at the first record of every episode',
          all(float(h_in_log[t][w].abs().max()) == 0.0 for t in range(steps) for w in range(W) if firsts[t, w]))
    check('(b) state carried (non-zero) inside an episode',
          all(float(h_in_log[t][w].abs().max()) > 0 for t in range(steps) for w in range(W) if not firsts[t, w]))
    check('(b) state changes from step to step',
          all(not torch.equal(h_new_log[t], h_new_log[t + 1]) for t in range(steps - 1)))
    with torch.no_grad():   # the state matters: the same records with h = 0 give other logits mid-episode
        b4 = TP.to_batch(seq_rows[5], seq_pend[5], dev)
        diff = float((model(b4, mem=torch.zeros(W, D))[0] - logit_log[5]).abs().max())
    check('(b) outputs depend on the carried state', diff > 1e-6, f'max logit change {diff:.3e}')
    with torch.no_grad():   # act(mem=) returns the same new state as forward(mem=)
        a_out = model.act(TP.to_batch(seq_rows[2], seq_pend[2], dev), mem=h_in_log[2],
                          u=torch.rand(W, sum(heads)))
    check('(b) act(mem=) returns the new state', len(a_out) == 4 and torch.equal(a_out[3], h_new_log[2]))

    # ---------------------------------------------------------------- (c) learner: chunks, BPTT, gradients
    T = 3
    cap = steps + 2
    length = np.full(W, steps)
    mask = np.zeros((W, cap), bool)
    mask[:, :steps] = True
    mask[2, 3:6] = False     # a chunk without a trained decision is dropped
    mask[0, steps - 1] = False   # (the open last decision of a worker)
    plan = TP.memory_chunks(length, mask, T)
    check('(c) memory_chunks: chunk count and layout', plan['chunks'] == W * (steps // T) - 1 and
          len(plan['pos']) == plan['chunks'] * T and np.all(plan['start'] == plan['pos'][::T]),
          f"chunks {plan['chunks']}")
    plan2 = TP.memory_chunks(np.array([7, 0]), np.ones((2, 9), bool), 4)
    check('(c) memory_chunks: padding repeats the last decision with weight 0',
          plan2['pos'].tolist() == [0, 1, 2, 3, 4, 5, 6, 6] and plan2['weight'].tolist() == [1] * 7 + [0])
    # the rollout as the actor stored it: rows / pending / h_in per [worker, step]
    roll_rows = np.zeros((W, cap), ROW)
    roll_pend = np.zeros((W, cap, len(heads)), np.int64)
    roll_h = np.zeros((W, cap, D), np.float32)
    for t in range(steps):
        roll_rows[:, t] = seq_rows[t]
        roll_pend[:, t] = seq_pend[t]
        roll_h[:, t] = h_in_log[t].numpy()
    pos = plan['pos']
    rows_c = roll_rows.reshape(-1)[pos]
    b = TP.to_batch(rows_c, roll_pend.reshape(-1, len(heads))[pos], dev)
    h0 = torch.from_numpy(roll_h.reshape(-1, D)[plan['start']])
    reset = torch.from_numpy(rows_c['first'] != 0).view(-1, T)
    memd = dict(h0=h0, T=T, reset=reset)
    with torch.no_grad():
        out_c = model(b, mem=memd)
        out_p = model(b, packed=True, mem=memd)
    ref_logits = torch.stack(logit_log, 1).reshape(W * steps, -1)   # [worker, step] order, flat
    ref_h = torch.stack(h_new_log, 1).reshape(W * steps, D)
    wsel = torch.from_numpy((pos // cap) * steps + pos % cap)
    err_l = float((out_c[0] - ref_logits[wsel]).abs().max())
    err_h = float((out_c[-1] - ref_h[wsel]).abs().max())
    err_p = float((out_p[0] - ref_logits[wsel]).abs().max())
    check('(c) BPTT chunks reproduce the actor (logits, states)', err_l < 1e-5 and err_h < 1e-5,
          f'logits {err_l:.1e}, states {err_h:.1e}')
    check('(c) packed forward with chunks reproduces the actor', err_p < 1e-4, f'logits {err_p:.1e}')
    # tail records after the chunks (the teacher's records): one step each from their own state
    extra = synth_rows(rng, 5)
    b_tail = TP.cat_batch(b, TP.to_batch(extra, synth_pending(rng, 5, heads), dev))
    h_tail = torch.randn(5, D)
    with torch.no_grad():
        out_t = model(b_tail, mem=dict(memd, tail=h_tail))
        solo = model(TP.to_batch(extra, b_tail['pending'][-5:].numpy(), dev), mem=h_tail)
    check('(c) tail records = their own one-step forward',
          float((out_t[0][len(pos):] - solo[0]).abs().max()) < 1e-5 and
          float((out_t[0][:len(pos)] - out_c[0]).abs().max()) < 1e-5)
    # backward: a PPO-shaped loss on the trained records
    torch.manual_seed(3)
    learner = TP.TokPolicy(memory=D, **kw_items)
    learner.train()
    acts = torch.from_numpy(synth_pending(rng, len(pos), heads))
    wgt = torch.from_numpy(plan['weight'])

    def loss_of(m):
        logits, value, _, _ = m(b, packed=True, mem=memd)
        logp, ent = m.log_prob(logits, m.gate(b), acts)
        return -((logp * torch.randn(len(pos)) + 0.5 * value ** 2 - 0.01 * ent.sum(-1)) * wgt).sum() / wgt.sum()
    learner.zero_grad()
    loss_of(learner).backward()
    g_out = float(learner.mem_out.weight.grad.abs().sum())
    g_gru0 = sum(float(p.grad.abs().sum()) for p in learner.mem_gru.parameters())
    check('(c) zero-initialised projection: gradient reaches mem_out, GRU gradient 0 at the very start',
          g_out > 0 and g_gru0 == 0.0, f'|g mem_out| {g_out:.3e}, |g gru| {g_gru0:.1e}')
    with torch.no_grad():
        learner.mem_out.weight.normal_(0, 0.05)
    learner.zero_grad()
    loss_of(learner).backward()
    g_gru = {n: float(p.grad.abs().sum()) for n, p in learner.mem_gru.named_parameters()}
    check('(c) gradient reaches every GRU parameter (BPTT)', all(v > 0 for v in g_gru.values()),
          ', '.join(f'{k} {v:.2e}' for k, v in g_gru.items()))
    # BPTT really goes back in time: the loss on a chunk's last record has a gradient on its first record's input
    zprobe = torch.randn(2, T, learner.config['width'], requires_grad=True)
    hT = learner.recur(zprobe.reshape(2 * T, -1), dict(h0=torch.zeros(2, D), T=T, reset=torch.zeros(2, T, dtype=bool)))
    hT.view(2, T, D)[:, -1].sum().backward()
    check('(c) gradient flows through time inside a chunk', float(zprobe.grad[:, 0].abs().sum()) > 0)

    # ---------------------------------------------------------------- (d) checkpoints
    buf = io.BytesIO()
    torch.save(dict(model=model.state_dict(), config=model.config), buf)
    buf.seek(0)
    ck = torch.load(buf, map_location='cpu')
    back = TP.TokPolicy(**ck['config']).eval()
    back.load_state_dict(ck['model'])
    bb = TP.to_batch(seq_rows[3], seq_pend[3], dev)
    with torch.no_grad():
        o1, o2 = model(bb, mem=h_in_log[3]), back(bb, mem=h_in_log[3])
    check('(d) checkpoint round trip (config memory, weights, outputs)',
          ck['config'].get('memory') == D and same(o1, o2))
    # resume of a memory-free checkpoint into a memory run: weights + Adam state
    torch.manual_seed(4)
    free = TP.TokPolicy(**kw_items)
    opt_free = torch.optim.Adam(free.main_parameters(), lr=1e-3, eps=1e-5)
    loss = free(bb)[1].pow(2).mean()
    loss.backward()
    opt_free.step()
    ck_free = dict(model=free.state_dict(), optimizer=opt_free.state_dict(), config=free.config)
    run = TP.TokPolicy(memory=D, **kw_items)
    missing, unexpected = run.load_state_dict(ck_free['model'], strict=False)
    opt_run = torch.optim.Adam(run.main_parameters(), lr=1e-3, eps=1e-5)
    added = TP.pad_optimizer_state(ck_free['optimizer'], len(run.main_parameters()))
    opt_run.load_state_dict(ck_free['optimizer'])
    run.zero_grad()
    run(bb, mem=torch.randn(W, D))[1].pow(2).mean().backward()
    opt_run.step()
    st_old = opt_run.state[run.main_parameters()[0]]
    check('(d) resume of a memory-free checkpoint: only mem_* missing, Adam padded and stepping',
          not unexpected and all(k.startswith('mem_') for k in missing) and added == 6 and
          int(st_old['step']) == 2, f'added {added}, step of an old parameter {int(st_old["step"])}')

    # ---------------------------------------------------------------- (e) CUDA: GraphActor with the state table
    if torch.cuda.is_available():
        cuda = torch.device('cuda')
        gm = TP.TokPolicy(memory=D, **kw_items).to(cuda).eval()
        with torch.no_grad():
            gm.mem_out.weight.normal_(0, 0.1)
        tab = torch.randn((W + 2, D), device=cuda)
        start = tab.clone()
        actor = TP.GraphActor(gm, W + 2, 32, stream=torch.cuda.Stream(), crn=False, memory=tab)
        rows = synth_rows(rng, W)
        rows['n_ent'] = np.minimum(rows['n_ent'], 32)
        pend = synth_pending(rng, W, gm.heads)
        widx = np.array([5, 0, 3, 1])
        reset = np.array([False, True, False, False])
        _, lp, val = actor.act(rows, pend, widx, reset)
        torch.cuda.synchronize()
        h_ref = start[torch.from_numpy(widx).to(cuda)] * torch.from_numpy(~reset).float().to(cuda)[:, None]
        with torch.no_grad():
            _, v_ref, _, hn_ref = gm(TP.to_batch(rows, pend, cuda), mem=h_ref)
        ok_h = np.allclose(actor.h_used, h_ref.cpu().numpy(), atol=1e-6)
        ok_v = np.allclose(val, v_ref.cpu().numpy(), atol=1e-4)
        ok_t = torch.allclose(tab[torch.from_numpy(widx).to(cuda)], hn_ref, atol=1e-4) and \
            torch.equal(tab[2], start[2]) and torch.equal(tab[4], start[4])
        check('(e) GraphActor + memory = eager (input states, values, table rows; others untouched)',
              ok_h and ok_v and ok_t)
    else:
        print('skip (e) GraphActor: no CUDA here')

    print('RESULT', 'FAIL ' + ', '.join(FAIL) if FAIL else 'all checks passed', flush=True)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
