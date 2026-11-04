"""CPU tests of the privileged critic (train_tok --priv-critic, isaac_bridge/tok_priv.py, rl/docs/
PREDICTIVE_MEMORY_DESIGN.md section 4). No game, no GPU (the bridge's side: abplus_probe_priv.py on an engine host). Plain
script:  cd rl/bridge/python && python -X utf8 tests/test_priv_critic_smoke.py [--quick]
Checks:
  (a) the frame decoder by hand: the true layout's channels (hidden rooms, multi-cell shapes), HP sorted and capped, the
      scalars, Blind's ids only under the curse, the layout kept between frames and map_ver moving only with it, missing,
      a broken frame refused, the bridge's and the decoder's formats (the Lua source's pack strings);
  (b) the critic: value heads from the policy's give the policy's values whatever p is; every part reaches a gradient
      after a step; the missing row equals the encoder on zeros with the flag; dedup == no dedup;
  (c) visibility audit: the policy's modules know nothing of priv (source, state_dict); the critic's gradient reaches the
      shared layers and no policy head; the actor's input tensors, its actions and log-probabilities in the real
      collection loop are the same bytes with --priv-critic 0 and 1, and with different privileged data;
  (d) the real learn() under the CPU shim (tests/aux_learn_driver.py), --learner-fast 1 and 2: runs, the policy's start
      weights as off, priv_* columns, values really from the privileged heads (other privileged data: other returns' use,
      same start weights), the policy's value heads keep learning; A3 (with --aux-predict);
  (e) --resume: with the critic (restored), without it (a C99 checkpoint: heads from the policy's, policy exact);
  (f) refused flags. Off == the base commit: test_aux_predict_smoke.py (g), the same driver.
--quick skips (c)'s collection runs, (d), (e), (f).
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_temporal_smoke as TS   # noqa: E402  (its path set-up, gymnasium stand-in, records)

import numpy as np   # noqa: E402
import torch   # noqa: E402

from isaac_bridge import tok_policy as TP   # noqa: E402
from isaac_bridge import tok_priv as P   # noqa: E402
from isaac_bridge.tok_obs import ROW   # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def frame(npcs=(), peds=(), counts=(0,) * 9, blind=0, total=None, layout=None):
    """A frame as the bridge's priv_payload builds it. layout: (rooms [(gi, shape, type, flags)], left 9, unvisited 9)."""
    total = sum(n[0] for n in npcs) if total is None else total
    body = struct.pack('<IBBfHH9H', P.MAGIC, 1 if layout else 0, blind, total, len(npcs), len(peds), *counts)
    for hp, mx, boss in npcs:
        body += struct.pack('<ffB', hp, mx, boss)
    for i in peds:
        body += struct.pack('<H', i)
    if layout:
        rooms, left, unv = layout
        body += struct.pack('<B', len(rooms))
        for r in rooms:
            body += struct.pack('<BBBB', *r)
        body += struct.pack('<18H', *left, *unv)
    return body


def case_a():
    row = np.zeros(1, P.PRIV)
    d = P.PrivDecoder()
    s2 = np.zeros(2, P.PRIV)
    v2 = (s2[:1], s2[1:2])
    P.priv_record(P.PrivDecoder(), None, v2, 0)
    check('(a) priv_record: a connection\'s first record without a frame is missing', s2['missing'][0] == 1)
    d0 = P.PrivDecoder()
    P.priv_record(d0, frame(npcs=[(7.0, 9.0, 0)], layout=([(84, 1, 1, 7)], [0] * 9, [0] * 9)), v2, 0)
    P.priv_record(d0, None, v2, 1)
    check('(a) priv_record: no frame = the record before, every field', s2[0].tobytes() == s2[1].tobytes() and s2['missing'][1] == 0 and
          abs(float(s2['hp'][1][0]) - 7.0 / P.HP_SCALE) < 1e-6)
    P.priv_record(d0, frame(npcs=[(5.0, 9.0, 0)]), v2, 2)
    check('(a) priv_record: the next frame in the other slot, the layout kept', abs(float(s2['hp'][0][0]) - 5.0 / P.HP_SCALE) < 1e-6 and
          s2['map_ver'][0] == 1 and s2['map'][0].sum() > 0)
    # the start room (visited, clear, here), a boss, a secret, a super secret, a treasure, a shop, a 1x2 normal room (shape 4
    # covers its GridIndex and the cell below), a curse room (type 10: another special room)
    rooms = [(84, 1, 1, 1 + 2 + 4), (95, 1, 5, 0), (98, 1, 7, 0), (45, 1, 8, 0), (123, 1, 4, 0), (86, 1, 2, 0), (72, 4, 1, 0),
             (110, 1, 10, 0)]
    left, unv = [1, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 3, 0, 1]
    f = frame(npcs=[(3.0, 10.0, 0), (20.0, 30.0, 0), (250.0, 500.0, 1)], counts=[2, 1, 0, 0, 0, 0, 0, 0, 0], peds=[33, 44],
              blind=0, layout=(rooms, left, unv))
    d.decode(f, row)
    m = row['map'][0]

    def cell(g):
        return divmod(g, 13)
    ok = m[0].sum() == 9 and all(m[0][cell(g)] == 1 for g in (84, 95, 98, 45, 123, 86, 72, 85, 110))
    check('(a) every room is on the map, found or not, a 1x2 room on both its cells', bool(ok), f'{int(m[0].sum())} cells')
    ok = m[1].sum() == 1 and m[1][cell(84)] == 1 and m[2].sum() == 1 and m[2][cell(84)] == 1 and m[3].sum() == 1 and m[3][cell(84)] == 1
    check('(a) visited, clear, the player\'s room', bool(ok))
    for g, ch, name in ((95, 4, 'boss'), (123, 5, 'treasure'), (86, 6, 'shop'), (110, 7, 'another special room'), (98, 8, 'secret'),
                        (45, 9, 'super secret')):
        check(f'(a) channel of the {name} room', m[ch].sum() == 1 and m[ch][cell(g)] == 1)
    hp, s = row['hp'][0], row['scal'][0]
    check('(a) HP sorted, scaled, zeros after', np.allclose(hp[:4], [5.0, 0.4, 0.06, 0.0]) and hp[3:].sum() == 0, str(hp[:4]))
    check('(a) totals and the boss', np.allclose(s[:6], [273.0 / P.HP_SCALE, 3 / P.NPC_K, 250 / P.BOSS_SCALE, 500 / P.BOSS_SCALE, 0.5, 0.0]))
    check('(a) pickups per class: the room\'s, the visited rooms\', the unvisited rooms\'',
          np.allclose(s[6:15], np.array([2, 1, 0, 0, 0, 0, 0, 0, 0]) / 5) and np.allclose(s[15:24], np.array(left) / 10) and
          np.allclose(s[24:33], np.array(unv) / 10))
    check('(a) no ids without the curse (the policy sees them then)', row['items'][0].sum() == 0)
    check('(a) not missing, a map version', row['missing'][0] == 0 and row['map_ver'][0] == 1)
    r2 = np.zeros(1, P.PRIV)
    d.decode(frame(npcs=[(1.0, 2.0, 0)], peds=[33, 44], blind=1), r2)
    check('(a) the layout stays between frames, its version too', np.array_equal(r2['map'], row['map']) and r2['map_ver'][0] == 1)
    check('(a) true ids under Curse of the Blind', r2['items'][0].tolist()[:3] == [33, 44, 0] and r2['scal'][0][5] == 1.0)
    d.decode(frame(layout=([(84, 1, 1, 1)], [0] * 9, [0] * 9)), r2)
    check('(a) a new layout moves the version', r2['map_ver'][0] == 2 and r2['map'][0][3].sum() == 0)
    d2 = P.PrivDecoder(d.versions)
    d2.decode(frame(layout=([(84, 1, 1, 1)], [0] * 9, [0] * 9)), r2)
    check('(a) the worker\'s counter runs over its episodes', r2['map_ver'][0] == 3)
    d.decode(None, r2)
    check('(a) no frame: zeros and missing', r2['missing'][0] == 1 and r2['scal'].sum() == 0 and r2['map'].sum() == 0)
    for bad, name in ((b'\0' * 40, 'magic'), (frame() + b'\0', 'trailing bytes')):
        try:
            d.decode(bad, r2)
            check(f'(a) a broken frame is refused ({name})', False)
        except (ValueError, struct.error):
            check(f'(a) a broken frame is refused ({name})', True)
    d3 = P.PrivDecoder()
    try:
        d3.decode(frame(), r2)
        check('(a) a first frame without a layout is refused', False)
    except ValueError:
        check('(a) a first frame without a layout is refused', True)
    # the bridge's formats, from its source: the Lua pack strings say what the decoder reads
    lua = (HERE.parent.parent / 'abplus' / 'abp_bridge.lua').read_text(encoding='utf-8', errors='replace')
    fmt_head = re.search(r'spack\("(<IBB[^"]*)", PRIV_MAGIC', lua).group(1)
    check('(a) Lua\'s header format == the decoder\'s', fmt_head == '<IBBfHH' + 'H' * 9, fmt_head)
    check('(a) Lua\'s monster record == the decoder\'s', '"<ffB"' in lua and P._NPC.itemsize == struct.calcsize('<ffB'))
    check('(a) Lua: off by default, a clone starts off', 'priv = false,' in lua and 'state.priv = false   -- abp-0.2.19' in lua)


def small_model():
    cfg = dict(TS.SMALL, **TS.BASE, **TS.FUSED, **TS.TK)
    torch.manual_seed(5)
    m = TP.TokPolicy(**cfg)
    TS.perturb_all(m, 6)
    return m


def fake_batch(n=12, seed=1):
    g = np.random.default_rng(seed)
    roll = type('R', (), {})()
    roll.cap = 8
    roll.priv = np.zeros((4, 8), P.PRIV)
    pv = roll.priv
    pv['map'] = g.random(pv['map'].shape) < 0.05
    pv['map_ver'] = np.arange(8)[None] // 3
    # the same version of a worker holds the same map
    for w in range(4):
        for v in range(3):
            sel = pv['map_ver'][w] == v
            pv['map'][w, sel] = pv['map'][w, np.flatnonzero(sel)[0]]
    pv['hp'] = g.random(pv['hp'].shape)
    pv['scal'] = g.random(pv['scal'].shape)
    pv['items'] = g.integers(0, 1024, pv['items'].shape)
    flat = np.arange(32, dtype=np.int64)
    return roll, flat


def case_b():
    m = small_model()
    c = P.make_critic(TS.SMALL['width'])
    c.init_from(m)
    zr, zg = torch.randn(9, TS.SMALL['width']), torch.randn(9, TS.SMALL['width'])
    v0, vg0 = m.value(zr)[:, 0], m.gtok.value(zg)[:, 0]
    v1, vg1 = c(zr, zg, torch.randn(9, c.p_dim))
    v2, vg2 = c(zr, zg, torch.randn(9, c.p_dim))
    e_ = max(float((a_ - b_).abs().max()) for a_, b_ in ((v1, v0), (vg1, vg0), (v2, v0), (vg2, vg0)))
    check('(b) value heads from the policy\'s: its values, whatever p is (zero columns: rounding only)', e_ < 1e-5, f'max |diff| {e_:.1e}')
    roll, flat = fake_batch()
    b = P.PrivBatch(roll, flat, 'cpu')
    check('(b) PrivBatch: one copy of each distinct map', b.maps.shape[0] == 4 * 3 and len(b) == 32, f'{b.maps.shape[0]} maps for 32 records')
    idx = torch.arange(32)
    w, t = flat // roll.cap, flat % roll.cap
    check('(b) each record\'s id gives its own map', np.array_equal(b.maps[b.ids].numpy(), roll.priv['map'][w, t]))
    p_all = c.encode(b, idx)
    p_part = torch.cat([c.encode(b, idx[:7]), c.encode(b, idx[7:])])
    check('(b) encoding in parts == at once', float((p_all - p_part).abs().max()) < 1e-5)
    mf = c.missing(3, p_all)
    roll0 = type('R', (), dict(cap=1, priv=np.zeros((1, 1), P.PRIV)))()
    roll0.priv['missing'] = 1
    p0 = c.encode(P.PrivBatch(roll0, np.zeros(1, np.int64), 'cpu'), torch.zeros(1, dtype=torch.long))
    check('(b) missing == the encoder on zeros with the flag', mf.shape == (3, c.p_dim) and float((mf[0] - p0[0]).abs().max()) < 1e-6)
    check('(b) no records: an empty p', c.missing(0, p_all).shape == (0, c.p_dim))
    opt = torch.optim.Adam(c.parameters(), lr=1e-3)
    for _ in range(2):   # (the first step reaches only the zero columns: the encoder from the second on)
        opt.zero_grad(set_to_none=True)
        v, vg = c(zr[:9], zg[:9], c.encode(b, idx[:9]))
        (v.pow(2).mean() + vg.pow(2).mean()).backward()
        opt.step()
    check('(b) a step reaches every part (the new columns, the CNN, the MLP, the items\' table)', all(
        q.grad is not None and float(q.grad.abs().sum()) > 0 for n_, q in c.named_parameters() if n_ != 'item.weight') and
        float(c.item.weight.grad.abs().sum()) > 0)
    v3, _ = c(zr, zg, torch.randn(9, c.p_dim))
    v4, _ = c(zr, zg, torch.randn(9, c.p_dim))
    check('(b) after a step p matters', float((v3 - v4).abs().max()) > 0)


def case_c_static():
    src = (HERE.parent / 'isaac_bridge' / 'tok_policy.py').read_text(encoding='utf-8')
    check('(c) tok_policy never names the privileged data', not re.search(r'\bpriv\b|PRIV|tok_priv|\bp_t\b', src.replace("--priv-critic", "")),
          str(re.findall(r'.{0,30}(?:\bpriv\b|PRIV|tok_priv|\bp_t\b).{0,30}', src.replace("--priv-critic", ""))[:3]))
    obs = (HERE.parent / 'isaac_bridge' / 'tok_obs.py').read_text(encoding='utf-8')
    check('(c) ROW and its encoder never name it', not re.search(r'\bpriv\b|PRIV|tok_priv', obs))
    check('(c) ROW has no privileged field', not any('priv' in k for k in ROW.names))
    m = small_model()
    check('(c) nothing privileged in the policy\'s state', not any('priv' in k or k.startswith('p_') for k in m.state_dict()))
    # the critic's gradient: the shared layers yes, no head of the policy
    c = P.make_critic(TS.SMALL['width'])
    c.init_from(m)
    roll, flat = fake_batch()
    b = P.PrivBatch(roll, flat, 'cpu')
    rng = np.random.default_rng(3)
    rows = TS.make_seq(rng, TS.NW, 4, set(), set(), set())[3]
    pend = np.zeros((TS.NW, len(m.heads)), np.int64)
    bt = TP.to_batch(rows, pend, 'cpu')
    st = TP.TemporalState(m, TS.NW, 'cpu', torch.float32)
    d = dict(mode='stored', o=torch.randn(TS.NW, TS.TK['temporal_width']), o_prev=torch.randn(TS.NW, TS.TK['temporal_width']),
             has=torch.ones(TS.NW, dtype=torch.bool), keep_z=True)
    m.zero_grad(set_to_none=True)
    m(bt, need_vg=True, temporal=d)
    zr, zg = d['z_out']
    v, vg = c(zr, zg, c.encode(b, torch.arange(TS.NW)))
    (v.sum() + vg.sum()).backward()
    named = dict(m.named_parameters())

    def has_g(pre):
        return any(q.grad is not None and float(q.grad.abs().sum()) > 0 for n_, q in named.items() if n_.startswith(pre))
    for pre in ('blocks', 'player', 'gtok.tok', 'tatt.read'):
        check(f'(c) the critic\'s gradient reaches {pre}', has_g(pre))
    for pre in ('pi.', 'danger.', 'value.', 'gtok.value'):
        check(f'(c) the critic\'s gradient does not reach {pre}', not has_g(pre))
    del st


def driver(tree, out, *extra):
    cmd = [sys.executable, '-X', 'utf8', str(HERE / 'aux_learn_driver.py'), '--tree', str(tree), '--out', str(out)] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                       env=dict(os.environ, OMP_NUM_THREADS='2'), timeout=900)
    if p.returncode != 0:
        print(p.stdout[-1500:], p.stderr[-3000:])
        raise RuntimeError(f'driver failed: {extra}')
    return torch.load(Path(out) / 'result.pt', weights_only=False), p.stdout


def same_tensors(a, b):
    return list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)


def case_runs():
    tmp = Path(tempfile.mkdtemp(prefix='priv-test-', dir=os.environ.get('AUX_TEST_TMP') or None))
    tree = HERE.parent.parent.parent
    off, _ = driver(tree, tmp / 'off', '--updates', '2')
    on, out_on = driver(tree, tmp / 'on', '--updates', '2', '--priv', '1')
    other, _ = driver(tree, tmp / 'other', '--updates', '2', '--priv', '1', '--priv-seed', '99')
    # ---- (c) the audit in the real loop
    check('(c) the actor\'s input tensors, actions and log-probabilities: the same bytes off / on', off['actor_hash'] == on['actor_hash'], off['actor_hash'][:12])
    check('(c) ... and with other privileged data', other['actor_hash'] == on['actor_hash'])
    check('(c) the rollout\'s records: the same bytes off / on', all(
        np.array_equal(off['rollout'][k], on['rollout'][k]) for k in ('rows', 'actions', 'logp', 'value', 'reward', 'terminal')))
    # ---- (d)
    check('(d) the policy\'s start weights as off', same_tensors(off['before'], on['before']))
    check('(d) the update moves the weights, other than off', not same_tensors(on['before'], on['weights']) and not same_tensors(off['weights'], on['weights']))
    check('(d) other privileged data trains another policy (the values really read it)', not same_tensors(on['weights'], other['weights']))
    row = on['progress'][-1]
    cols = ['priv_missing', 'priv_plain_vf', 'priv_plain_vg', 'u_priv']
    check('(d) the columns', all(row.get(c, '') != '' for c in cols), str({c: row.get(c) for c in cols}))
    check('(d) no missing rows in the rollout', float(row['priv_missing']) == 0.0)
    check('(d) off: no such columns', not any(k.startswith('priv_') or k == 'u_priv' for k in off['progress'][-1]))
    plain_v = ('value.0.weight', 'gtok.value.0.weight')
    check('(d) the policy\'s value heads keep training', all(not torch.equal(on['before'][k], on['weights'][k]) for k in plain_v))
    check('(d) the policy head keeps training', not torch.equal(on['before']['pi.weight'], on['weights']['pi.weight']))
    print(f"info: CPU shim, one update of the tiny test model: off {off['learn_s']:.3f} s, on {on['learn_s']:.3f} s "
          f"(x{on['learn_s'] / off['learn_s']:.2f}; not the real model's ratio: the time gate is the CUDA run), u_priv {float(row['u_priv']) * 1000:.1f} ms")
    for fast in (2,):
        r2, _ = driver(tree, tmp / f'on{fast}', '--updates', '2', '--priv', '1', '--fast', str(fast))
        check(f'(d) --learner-fast {fast}: runs, the same actor bytes', r2['actor_hash'] == on['actor_hash'] and len(r2['progress']) == 2)
    a3, _ = driver(tree, tmp / 'a3', '--updates', '2', '--priv', '1', '--aux', '1')
    check('(d) A3 (with --aux-predict): both sets of columns, the same actor bytes', a3['actor_hash'] == on['actor_hash'] and
          a3['progress'][-1].get('aux_loss', '') != '' and a3['progress'][-1].get('priv_plain_vf', '') != '')
    # ---- (e)
    long_, _ = driver(tree, tmp / 'long', '--updates', '6', '--priv', '1')
    res, out_res = driver(tree, tmp / 'res', '--updates', '1', '--priv', '1', '--resume', long_['last'])
    ck = torch.load(long_['last'], map_location='cpu', weights_only=False)
    check('(e) a checkpoint keeps the critic apart from the model', set(ck['priv']) == {'model', 'opt'} and not any('priv' in k for k in ck['model']))
    check('(e) --resume with the critic: restored', 'resumed' in out_res)
    fresh, out_fresh = driver(tree, tmp / 'c99', '--updates', '1', '--priv', '1', '--resume', off['last'])
    c0 = torch.load(off['last'], map_location='cpu', weights_only=False)
    check('(e) --resume without it (C99): heads from the policy\'s, the policy exact', 'priv' not in c0 and 'value heads from the policy' in out_fresh and
          same_tensors(c0['model'], fresh['before']))
    # ---- (f)
    refused = 0
    py = HERE.parent
    for i_, (extra, needle) in enumerate(((['--priv-critic', '1'], 'needs --temporal 1'),
                                          (['--priv-critic', '1', '--temporal', '1', '--global-token', '1', '--item-tokens', '1',
                                            '--learner-compile', 'default'], 'learner-compile'),
                                          (['--priv-critic', '1', '--temporal', '1', '--global-token', '1', '--item-tokens', '1',
                                            '--teach-merge', '0'], 'teach-merge'))):
        code = ('import sys; sys.path.insert(0, "tests"); import test_memory_smoke; sys.path.insert(0, "."); import train_tok; '
                f'train_tok.main(["--groups-file", "x", "--out", {str(tmp / f"ref{i_}")!r}, "--mode", "run", "--items"] + {extra!r})')
        p = subprocess.run([sys.executable, '-X', 'utf8', '-c', code], capture_output=True, text=True, encoding='utf-8',
                           errors='replace', cwd=str(py))
        refused += p.returncode != 0 and needle in p.stderr
    check('(f) the bad flag combinations are refused', refused == 3, f'{refused} of 3')
    print('info: test directory', tmp)


def main():
    case_a()
    case_b()
    case_c_static()
    if '--quick' not in sys.argv:
        case_runs()
    print('RESULT', 'all checks passed' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
