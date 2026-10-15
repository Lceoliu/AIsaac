"""CPU smoke test of B35 (2026-10-16): observation v2 (bridge abp-0.2.18; pickup prices, heart split, room type;
train_tok --obs-v2 1, TokPolicy obs_prices / obs_hearts / obs_room_type).

No game, no GPU. Plain script:
  cd rl/bridge/python && python tests/test_obs_v2_smoke.py
Checks:
  (a) the observation path on synthetic lean payloads (abplus_lean's layout, 148-byte entity records with prices,
      a terrain block with a trapdoor): LeanDecoder + encode_row write ROW 'ent_price' (the prices, 0 on the exit
      token), 'phearts' (black half hearts from the GetBlackHearts bit mask, eternal, bone / 6) and 'room_type'
      (clipped); price_columns' values (numpy and torch alike); every field of git HEAD's ROW is byte-identical to
      what HEAD's decoder + encoder write from the same payloads with 144-byte records (the old fields unchanged);
  (b) TokPolicy without the keys == git HEAD's policy module, bit for bit (forward padded / packed, act, evaluate) on
      records whose new fields are not zero, for the plain, the items and the fused (item / global tokens, 8 slots)
      configurations;
  (c) TokPolicy with the keys: config keys, input widths, forward / act / evaluate, packed and the learner's fast path
      against the padded forward, the new inputs change the outputs, backward reaches the new weights; decode_rows
      (GraphActor's and the learner's raw path) carries the new fields;
  (d) an old-config checkpoint loads into the v2 model with load_compatible (only the two first Linears padded,
      room_type_emb zero) and computes the same outputs; a v2 checkpoint round-trips through TokPolicy(**config) and
      eval_tok.build_eval_model;
  (e) optional, Linux only: with ISAAC_RL_TEST_TURBO_SO=<path of libabp_turbo.so> (built from this abp_turbo.c),
      FastRow (abp_row_encode version 7) writes the same ROW bytes as encode_row on the synthetic payloads.
"""
import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_memory_smoke as M   # noqa: E402  (its path set-up, gymnasium stub, synthetic records, HEAD loader)
import test_fused_tokens_smoke as FT   # noqa: E402  (the learner's fast-path batch)

import numpy as np   # noqa: E402
import torch   # noqa: E402

from isaac_bridge import abplus_lean as AL   # noqa: E402
from isaac_bridge import tok_obs as TO   # noqa: E402
from isaac_bridge import tok_policy as TP   # noqa: E402

PY = HERE.parent
FAIL = []
V2 = dict(obs_prices=True, obs_hearts=True, obs_room_type=True)
SMALL = dict(width=64, layers=2, heads=4)
CONFIGS = (('plain', dict()), ('items', dict(items=True, charge=True, ent_ext=True, pchar=True)),
           ('fused', dict(items=True, charge=True, ent_ext=True, pchar=True, item_tokens=True, global_token=True,
                          item_shared=True, item_slots=8)))
PRICES = [15, 3, -1, -2, -3, -1000, 0, 7]
EXPECT_COLS = {15: (1, 0, 0, 1), 3: (3 / 15, 0, 0, 1), -1: (0, 1, 0, 1), -2: (0, 2, 0, 1), -3: (0, 0, 1, 1),
               -1000: (0, 0, 0, 1), 0: (0, 0, 0, 0), 7: (7 / 15, 0, 0, 1)}


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def head_source(rel):
    try:
        return subprocess.run(['git', '-C', str(PY), 'show', 'HEAD:bridge/python/' + rel], capture_output=True,
                              text=True, check=True, encoding='utf-8').stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def load_head(rel, name):
    """A module of the bridge's python dir as it is in git HEAD (package isaac_bridge: its relative imports are the
    working tree's), or None."""
    src = head_source(rel)
    if src is None:
        return None
    spec = importlib.util.spec_from_loader('isaac_bridge.' + name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'isaac_bridge'
    exec(compile(src, 'HEAD:' + rel, 'exec'), mod.__dict__)
    return mod


# ---------------------------------------------------------------- synthetic lean payloads
W_GRID, H_GRID, X0, Y0 = 15, 9, 80.0, 160.0


def terrain_json():
    cells = []
    for i in range(W_GRID * H_GRID):
        r, c = divmod(i, W_GRID)
        wall = r in (0, H_GRID - 1) or c in (0, W_GRID - 1)
        rock = (r, c) == (3, 5)
        cells.append([i, X0 + 40 * c, Y0 + 40 * r, 1 if wall or rock else 0, int(not wall), int(not (wall or rock)),
                      0, 0, int(wall or rock), int(rock)])
    terrain = dict(width=W_GRID, height=H_GRID, cells=cells)
    grid = [[3 * W_GRID + 5, 2, 0, 0, 0, X0 + 200, Y0 + 120], [4 * W_GRID + 10, 17, 0, 0, 0, X0 + 400, Y0 + 160]]
    return json.dumps(terrain).encode(), json.dumps(grid).encode()


def world(k):
    """Record k's state: player, two doors, entities (pickups with PRICES, an NPC, a tear), room fields."""
    p = np.zeros(1, AL.PLAYER)
    p['x'], p['y'] = 300.0 + 3.25 * k, 280.0 - 1.5 * k
    p['vx'], p['vy'], p['size'] = 1.5, -0.5, 10.0
    p['hearts'], p['max_hearts'] = 4 - (k == 2), 6
    p['soul'], p['black'] = (5, 7, 3)[k], (0b101, 0b111, -1)[k]   # -1: every bit (the int Lua would push for 0xFFFFFFFF)
    p['bone'], p['eternal'] = (1, 2, 0)[k], (1, 0, 1)[k]
    p['bombs'], p['keys'], p['coins'] = 1, 2, 17
    p['damage'], p['fire_delay_max'], p['shot_speed'], p['range'], p['speed'] = 3.5, 10, 1.0, 260.0, 1.0
    p['fire_delay'], p['damage_cooldown'], p['controls'] = 3, 0, 1
    d = np.zeros(2, AL.DOOR)
    d['slot'], d['open'], d['x'], d['y'], d['target_type'] = (0, 2), (1, 0), (X0, X0 + 280), (Y0 + 160, Y0), (1, 2)
    n = len(PRICES) + 2
    e = np.zeros(n, AL.ENTITY)
    e['id'] = np.arange(n) + 11
    e['type'] = [5] * len(PRICES) + [10, 2]
    e['variant'] = [100, 10, 100, 100, 100, 30, 20, 100, 0, 0]
    e['subtype'] = [1, 1, 2, 3, 4, 1, 1, 5, 0, 0]
    e['x'] = X0 + 40 + 37.5 * np.arange(n)
    e['y'] = Y0 + 60 + 11.0 * np.arange(n) + k
    e['vx'][-2:], e['vy'][-2:] = (1.0, 8.0), (-0.5, 0.0)
    e['size'], e['size_mx'], e['size_my'] = 13.0, 1.0, 1.0
    e['coll'], e['gcoll'] = 4, 5
    e['kind'] = [5] * len(PRICES) + [6, 1]
    e['flags'][-2] = 3
    e['cdmg'][-2] = 1.0
    e['aframe'], e['age'] = 2, 30 + k
    e['champion'] = -1
    e['height'][-1], e['fall'][-1], e['scale'][-1] = -23.0, 0.5, 1.0
    e['price'][:len(PRICES)] = PRICES
    if k == 1:   # a price changes (a Steam Sale picked up, say)
        e['price'][0] = 7
    room = ((1, 5, 2)[k] if k < 2 else 40, 1, W_GRID, H_GRID, 45, 1, 1, 100 + k,
            X0 - 20, Y0 - 20, X0 + 40 * (W_GRID - 1) + 20, Y0 + 40 * (H_GRID - 1) + 20)
    return p, d, e, room


def payload(k, flags, entity_bytes=148):
    p, d, e, room = world(k)
    out = struct.pack('<IIIBBBB', AL.MAGIC, 100 + 4 * k, 200 + 4 * k, flags, len(p), len(d), 0)
    out += struct.pack('<iiiiiiiiffff', *room)
    out += struct.pack('<dddd', 0.0, 30.0, 30.0, 1.0)
    out += p.tobytes() + d.tobytes() + struct.pack('<H', len(e))
    rec = np.ascontiguousarray(e).view(np.uint8).reshape(len(e), AL.ENTITY.itemsize)
    out += rec[:, :entity_bytes].tobytes()
    if flags & 2:
        tj, gj = terrain_json()
        out += struct.pack('<II', 7, len(tj)) + tj + struct.pack('<I', len(gj)) + gj
    return out


PAYLOAD_FLAGS = (2, 0, 2)   # a terrain block, none, the same terrain block again


def black_reference(soul, mask):
    """Entity_Player::IsBlackHeart(i) = bit (i - 1) >> 1 of the mask, for the half hearts i = 1 .. soul."""
    mask &= 0xFFFFFFFF
    return sum((mask >> ((i - 1) >> 1)) & 1 for i in range(1, int(soul) + 1))


def part_a():
    tj, _ = terrain_json()
    dec, st = AL.LeanDecoder(), TO.EpisodeState(1000)
    rows = np.zeros(len(PAYLOAD_FLAGS), TO.ROW)
    for k, fl in enumerate(PAYLOAD_FLAGS):
        o = dec.decode(payload(k, fl))
        TO.encode_row(o, st, rows[k:k + 1], k)
        p, _, e, room = world(k)
        n = len(e)
        r = rows[k]
        check(f'a{k} decoder: price field', list(o.entities['price']) == list(e['price']))
        check(f'a{k} ent_price = the prices, exit token 0', int(r['n_ent']) == n + 1 and
              list(r['ent_price'][:n, 0]) == list(e['price']) and int(r['ent_price'][n, 0]) == 0,
              f"n_ent {int(r['n_ent'])}, {r['ent_price'][:n + 1, 0].tolist()}")
        black = black_reference(p['soul'][0], int(p['black'][0]))
        want = np.array([black / 12, p['eternal'][0], p['bone'][0] / 6], np.float32)
        check(f'a{k} phearts', r['phearts'].tobytes() == want.tobytes(), f"{r['phearts'].tolist()} black {black}")
        check(f'a{k} room_type', int(r['room_type']) == min(max(room[0], 0), 31), f"{int(r['room_type'])}")
    # black half hearts: the closed form against IsBlackHeart's rule
    rng = np.random.default_rng(3)
    ok = all(TO.black_half_hearts(s, m) == black_reference(s, m)
             for s, m in zip(rng.integers(0, 25, 500), rng.integers(-2 ** 31, 2 ** 31, 500)))
    check('a black_half_hearts == IsBlackHeart rule (500 random)', ok)
    # price columns
    pr = np.array(PRICES, np.int16)
    cols_np = TO.price_columns(pr)
    want = np.array([EXPECT_COLS[v] for v in PRICES], np.float32)
    cols_t = TO.price_columns(torch.from_numpy(pr)).numpy()
    check('a price_columns values', np.array_equal(cols_np, want), str(cols_np.tolist()))
    check('a price_columns numpy == torch', cols_np.tobytes() == cols_t.tobytes())
    # the old fields: HEAD's decoder + encoder on the same payloads with 144-byte records
    lean_h, obs_h = load_head('isaac_bridge/abplus_lean.py', '_abplus_lean_head'), \
        load_head('isaac_bridge/tok_obs.py', '_tok_obs_head')
    if lean_h is None or obs_h is None:
        print('skip (a) HEAD comparison: git HEAD not readable')
        return rows
    check('a HEAD entity record is 144 bytes', lean_h.ENTITY.itemsize == 144)
    dec_h, st_h = lean_h.LeanDecoder(), obs_h.EpisodeState(1000)
    rows_h = np.zeros(len(PAYLOAD_FLAGS), obs_h.ROW)
    for k, fl in enumerate(PAYLOAD_FLAGS):
        obs_h.encode_row(dec_h.decode(payload(k, fl, 144)), st_h, rows_h[k:k + 1], k)
    names = obs_h.ROW.names
    diff = [nm for nm in names if rows_h[nm].tobytes() != rows[nm].tobytes()]
    off = [nm for nm in names if obs_h.ROW.fields[nm][1] != TO.ROW.fields[nm][1]]
    check('a HEAD ROW fields byte-identical (3 records)', not diff, str(diff))
    check('a HEAD ROW offsets kept', not off, str(off))
    check('a ROW grows by the appended fields only', set(TO.ROW.names) - set(names) == {'ent_price', 'phearts',
                                                                                         'room_type'},
          f'{obs_h.ROW.itemsize} -> {TO.ROW.itemsize} bytes')
    return rows


# ---------------------------------------------------------------- the policy
def v2_rows(rng, n):
    r = FT.realistic_rows(rng, n)
    r['ent_price'][..., 0] = rng.choice(np.array(PRICES, np.int16), r['ent_price'].shape[:2])
    r['phearts'] = rng.random(r['phearts'].shape)
    r['room_type'] = rng.integers(0, 32, n)
    return r


def outputs(model, b, acts, u):
    with torch.no_grad():
        return M.outputs(model, b, acts, u)


def part_b(rng):
    old = M.head_module()
    if old is None:
        print('skip (b) HEAD comparison: git HEAD not readable')
        return
    for tag, cfg in CONFIGS:
        torch.manual_seed(1)
        new = TP.TokPolicy(**SMALL, **cfg).eval()
        FT.perturb(new, 5)
        ref = old.TokPolicy(**SMALL, **cfg).eval()
        ref.load_state_dict(new.state_dict())
        rows = v2_rows(rng, 12)
        pend = M.synth_pending(rng, 12, new.heads)
        b = TP.to_batch(rows, pend, 'cpu')
        acts = torch.from_numpy(M.synth_pending(rng, 12, new.heads))
        u = torch.rand((12, sum(new.heads)))
        a, r_ = outputs(new, b, acts, u), outputs(ref, old.to_batch(rows, pend, 'cpu'), acts, u)
        check(f'b {tag}: keys off == HEAD (non-zero v2 fields)', M.same(a, r_), f'max {M.maxdiff(a, r_):.2e}')
        check(f'b {tag}: config without v2 keys', not any(k in new.config for k in V2))


def grads_of(model, b):
    model.zero_grad()
    logits, value, _ = model(b)[:3]
    (logits.float().pow(2).mean() + value.float().pow(2).mean()).backward()


def part_c(rng):
    for tag, cfg in CONFIGS:
        torch.manual_seed(2)
        m = TP.TokPolicy(**SMALL, **cfg, **V2).eval()
        base = TP.TokPolicy(**SMALL, **cfg).eval()
        check(f'c {tag}: config keys', all(m.config.get(k) is True for k in V2))
        we, wb = m.entity[0].weight.shape[1], base.entity[0].weight.shape[1]
        wp, wq = m.player[0].weight.shape[1], base.player[0].weight.shape[1]
        check(f'c {tag}: input widths', we == wb + TO.PRICE_F and wp == wq + TO.PHEART_F and
              tuple(m.room_type_emb.weight.shape) == (TO.N_ROOM_TYPE, SMALL['width']) and
              float(m.room_type_emb.weight.detach().abs().sum()) == 0.0, f'entity {wb}->{we}, player {wq}->{wp}')
        FT.perturb(m, 9)
        rows = v2_rows(rng, 16)
        pend = M.synth_pending(rng, 16, m.heads)
        b = TP.to_batch(rows, pend, 'cpu')
        acts = torch.from_numpy(M.synth_pending(rng, 16, m.heads))
        u = torch.rand((16, sum(m.heads)))
        with torch.no_grad():
            f = m(b)[:3]
            fp = m(TP.to_batch(rows, pend, 'cpu'), packed=True)[:3]
            fb = m(FT.fast_batch(m, TP.to_batch(rows, pend, 'cpu'), rows), packed=True)[:3]
            a = m.act(b, u=u)[:3]
            ev = m.evaluate(b, acts)
        fin = all(torch.isfinite(x).all() for x in list(f) + list(a) + list(ev))
        check(f'c {tag}: forward / act / evaluate finite', fin)
        check(f'c {tag}: packed == padded', M.maxdiff(f, fp) < 1e-4, f'{M.maxdiff(f, fp):.2e}')
        check(f'c {tag}: learner fast path == padded', M.maxdiff(f, fb) < 1e-4, f'{M.maxdiff(f, fb):.2e}')
        # every new input moves the outputs
        for field, change in (('ent_price', lambda r: r['ent_price'].__setitem__(Ellipsis, -1)),
                              ('phearts', lambda r: r['phearts'].__setitem__(Ellipsis, 0.9)),
                              ('room_type', lambda r: r['room_type'].__setitem__(Ellipsis, 5))):
            r2 = rows.copy()
            change(r2)
            with torch.no_grad():
                g = m(TP.to_batch(r2, pend, 'cpu'))[:3]
            check(f'c {tag}: {field} changes the outputs', M.maxdiff(f, g) > 1e-6, f'{M.maxdiff(f, g):.2e}')
        # backward reaches the new weights
        grads_of(m, b)
        ge = m.entity[0].weight.grad[:, -TO.PRICE_F:].abs().sum()
        gp = m.player[0].weight.grad[:, -TO.PHEART_F:].abs().sum()
        gr = m.room_type_emb.weight.grad.abs().sum()
        check(f'c {tag}: gradients on the new weights', float(ge) > 0 and float(gp) > 0 and float(gr) > 0,
              f'price {float(ge):.2e} hearts {float(gp):.2e} room {float(gr):.2e}')
    # decode_rows (GraphActor's / the learner's raw path) carries the new fields as to_batch does
    rows = v2_rows(rng, 8)
    pend = M.synth_pending(rng, 8, TP.HEADS)
    raw = torch.from_numpy(np.ascontiguousarray(rows).view(np.uint8).reshape(8, TP.ROW_BYTES))
    d, t = TP.decode_rows(raw, torch.from_numpy(pend)), TP.to_batch(rows, pend, 'cpu')
    same = all(torch.equal(d[k], t[k]) for k in ('ent_price', 'phearts', 'room_type'))
    check('c decode_rows == to_batch for the v2 fields', same and d['ent_price'].shape == t['ent_price'].shape,
          str(tuple(d['ent_price'].shape)))
    check('c ROW_BYTES = ROW.itemsize', TP.ROW_BYTES == TO.ROW.itemsize == 20800, str(TP.ROW_BYTES))
    # cat_batch / take pad and cut ent_price like the other entity fields
    rows2 = v2_rows(rng, 4)
    rows2['n_ent'] = 30
    b1, b2 = TP.to_batch(rows, pend, 'cpu'), TP.to_batch(rows2, pend[:4], 'cpu')
    cb = TP.cat_batch(b1, b2)
    check('c cat_batch pads ent_price', cb['ent_price'].shape[:2] == cb['ent'].shape[:2], str(tuple(cb['ent_price'].shape)))


def part_d(rng):
    for tag, cfg in CONFIGS:
        torch.manual_seed(3)
        old = TP.TokPolicy(**SMALL, **cfg).eval()
        FT.perturb(old, 11)
        m = TP.TokPolicy(**SMALL, **cfg, **V2).eval()
        changed = TP.load_compatible(m, old.state_dict())
        check(f'd {tag}: load_compatible pads the two first Linears only',
              sorted(changed) == ['entity.0.weight (padded)', 'player.0.weight (padded)'], str(changed))
        check(f'd {tag}: room_type_emb stays zero', float(m.room_type_emb.weight.abs().sum()) == 0.0)
        rows = v2_rows(rng, 12)
        pend = M.synth_pending(rng, 12, m.heads)
        b = TP.to_batch(rows, pend, 'cpu')
        acts = torch.from_numpy(M.synth_pending(rng, 12, m.heads))
        u = torch.rand((12, sum(m.heads)))
        a, r_ = outputs(m, b, acts, u), outputs(old, b, acts, u)
        diff = M.maxdiff(a, r_)
        check(f'd {tag}: old checkpoint in the v2 model: same outputs', diff < 1e-5,
              f"{'bit-identical' if M.same(a, r_) else f'max {diff:.2e}'}")
        # round trip of a v2 checkpoint: config -> TokPolicy(**config) -> strict load -> the same outputs
        FT.perturb(m, 12)
        ck = dict(model=m.state_dict(), config=dict(m.config))
        m2 = TP.TokPolicy(**ck['config']).eval()
        m2.load_state_dict(ck['model'])
        check(f'd {tag}: v2 checkpoint round trip', M.same(outputs(m, b, acts, u), outputs(m2, b, acts, u)))
        try:
            import eval_tok as E
        except Exception as exc:   # noqa: BLE001
            print(f'skip eval_tok.build_eval_model: {exc!r}')
            continue
        me, ch = E.build_eval_model(ck, ck['config'].get('items', False), 'cpu')
        check(f'd {tag}: eval_tok.build_eval_model keeps the v2 keys',
              all(me.config.get(k) for k in V2) and not ch and M.same(outputs(m, b, acts, u), outputs(me, b, acts, u)),
              str(ch))


def part_e(rows_ref):
    path = os.environ.get('ISAAC_RL_TEST_TURBO_SO', '')
    if not path:
        print('skip (e) FastRow: ISAAC_RL_TEST_TURBO_SO not set (Linux, a libabp_turbo.so of this abp_turbo.c)')
        return
    fn = TO.fast_row_function(path)
    check('e FastRow library of version 7', fn is not None, path)
    if fn is None:
        return
    dec, st = AL.LeanDecoder(), TO.EpisodeState(1000)
    rows = np.zeros(len(PAYLOAD_FLAGS), TO.ROW)
    fr = TO.FastRow(fn, st, dec, 0, 0, 0)
    for k, fl in enumerate(PAYLOAD_FLAGS):
        pl = payload(k, fl)
        item = dec.decode(pl) if k == 0 else pl
        fr.encode(item, rows[k:k + 1], rows[k:k + 1].ctypes.data, k)
    ref = rows_ref.copy()
    ref['episode'], ref['seed'], ref['group'] = 0, 0, 0
    ref['first'] = [1, 0, 0]
    diff = [nm for nm in TO.ROW.names if ref[nm].tobytes() != rows[nm].tobytes()]
    check('e FastRow == encode_row (3 records)', not diff and fr.fast == 2, f'fast {fr.fast} slow {fr.slow} '
          f'reasons {fr.reasons} diff {diff}')


def main():
    torch.set_num_threads(4)
    rng = np.random.default_rng(35)
    rows = part_a()
    part_b(rng)
    part_c(rng)
    part_d(rng)
    part_e(rows)
    print(f'{len(FAIL)} failed' + (': ' + ', '.join(FAIL) if FAIL else ''))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
