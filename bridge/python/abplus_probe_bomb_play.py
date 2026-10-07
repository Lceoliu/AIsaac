"""Play == step with bombs (2026-10-07): is a batched `play` of N actions the same game as the same N actions sent as
step lines (and as JSON steps, and as one-action plays) when the actions include bomb presses?

Per seed: reset (group room, `--bombs` bombs), a warm-up of sticky random actions without bombs, a template clone parked
there (as tok_sampler's worker), then one list of N sticky random actions with bomb presses (probability --bomb-prob,
the first one forced at action --first-bomb). Each path is a lean clone of the template reseeded the same way:
  line    step lines (the sampler's episode path: native step, terrain cache, post-update gate as the instance has them)
  json    JSON step commands
  play1   one-action plays (N plays of one action)
  play    prefix plays: for every k the actions 1..k in one play of a fresh clone (--every k: only every k-th prefix and
          the last); its observation and digest are compared with those of the line path after k decisions
The line path's payload after every decision is kept; every path's observation (fixed part + the decoder's terrain /
grid) and hidden-state digest (abplus_goexplore.DIGEST_LUA) must be equal to the line path's at every compared decision.
--variant NAME=LUA runs that Lua on a path's clone before its first action (e.g. pu_off="ABP_PU.enabled=false; return
'ok'"); variants are extra paths compared like the others (step lines unless the name starts with play).
On a difference: the first decision k whose play prefix differs, then the frame inside it (play with repeats ending in
j frames against step lines whose last step has j frames), and both digest texts' differing lines (--diag).

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_bomb_play.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:8 --out <dir>
"""
import argparse
import json
import os
import struct
import time
from pathlib import Path

import numpy as np

from abplus_probe_fork import split_code
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV, action_code
from isaac_bridge.abplus_goexplore import DIGEST_LUA, GxConfig, Instance
from isaac_bridge.abplus_lean import (_HEADER, _LASER, _ROOM, _TOTALS, DOOR, ENTITY, PLAYER, LeanDecoder,
                                      read_lean_raw)
from isaac_bridge.tok_sampler import apply_instance_defaults, default_stub_list
import hashlib


def fixed_part(payload):
    """The payload up to the end of the laser records (no terrain / map / inventory block), and the flags byte masked
    to the bits that are not about blocks (1 clear, 4 paused)."""
    magic, lf, gf, flags, n_players, n_doors, n_lasers = _HEADER.unpack_from(payload, 0)
    off = _HEADER.size + _ROOM.size + _TOTALS.size + PLAYER.itemsize * n_players + DOOR.itemsize * n_doors
    n = struct.unpack_from('<H', payload, off)[0]
    off += 2 + ENTITY.itemsize * n
    for _ in range(n_lasers):
        k = _LASER.unpack_from(payload, off)[-1]
        off += _LASER.size + 16 * k
    head = bytearray(payload[:_HEADER.size])
    head[12] = flags & 5
    return bytes(head) + payload[_HEADER.size:off]


def obs_key(payload, decoder):
    decoder.decode(payload)
    return fixed_part(payload), decoder.terrain_raw, json.dumps(decoder.grid)


def dec_dead(dec, key):
    """Player 0 dead in an observation key (the play command stops there; step lines would go on)."""
    off = _HEADER.size + _ROOM.size + _TOTALS.size
    return bool(np.frombuffer(key[0], PLAYER, 1, off)['dead'][0])


COUNTERS = {'obs_checks': 0, 'obs_mismatches': 0, 'obs_first': None, 'tc_checks': 0, 'tc_mismatches': 0,
            'native_steps': 0, 'native_blocks': 0}


def collect(c):
    """Add a clone's native obs check counters (mode 2) and terrain cache check counters to COUNTERS."""
    c._send({"cmd": "native_obs"})
    m = c._recv()
    COUNTERS['obs_checks'] += int(m.get('checks') or 0)
    COUNTERS['obs_mismatches'] += int(m.get('mismatches') or 0)
    if m.get('first') and COUNTERS['obs_first'] is None:
        COUNTERS['obs_first'] = m.get('first')
    c._send({"cmd": "native_step"})
    m = c._recv()
    COUNTERS['tc_checks'] += int(m.get('tc_checks') or 0)
    COUNTERS['tc_mismatches'] += int(m.get('tc_mismatches') or 0)
    COUNTERS['native_steps'] += int(m.get('steps') or 0)
    COUNTERS['native_blocks'] += int(m.get('blocks') or 0)


def digest_text(c):
    return c.lua('return ABPGX_DIGEST(false)')


def dg(text):
    return hashlib.blake2b(text.encode('utf8'), digest_size=16).hexdigest()


def sticky(rng, n, bomb_prob, first_bomb, repeat_prob=0.9):
    out, move, shoot = [], 0, 0
    for i in range(n):
        if rng.random() >= repeat_prob:
            move, shoot = int(rng.integers(9)), int(rng.integers(5))
        bomb = int(rng.random() < bomb_prob) or int(i in first_bomb)
        out.append((move, shoot, bomb))
    return out


def clone_of(template, reseed, lua=None):
    c = template.fork(reseed=reseed, lean=True, alarm=600)
    if lua:
        r = c.lua(lua)
        if r != 'ok':
            raise RuntimeError(f'variant lua answered {r!r}')
    return c


def run_steps(template, reseed, acts, fpd, how, lua=None, last_rep=None, digests=None, park_at=None, parked=None):
    """Step lines / JSON steps / one-action plays; (per-decision observation keys, digest text at the end).
    digests: a list that gets the digest text after every decision (a lua command between the steps)."""
    c = clone_of(template, reseed, lua)
    dec = LeanDecoder()
    keys = []
    try:
        for i, (m, s, b) in enumerate(acts):
            if park_at is not None and i == park_at:   # a snapshot of the episode here (as tok_floor's room entries)
                parked.append(c.fork(tag='snap', alarm=0))
            rep = last_rep if (last_rep is not None and i == len(acts) - 1) else fpd
            if how == 'line':
                c._sock.sendall(b"S %d %d %d %d %d\n" % (rep, m, s, b, 0))
            elif how == 'json':
                c._send({"cmd": "step", "repeat": rep, "move": m, "shoot": s, "bomb": b, "item": 0})
            else:
                c._send({"cmd": "play", "actions": [action_code(m, s, b)], "repeat": rep, "stop_clear": False})
            keys.append(obs_key(read_lean_raw(c), dec))
            if digests is not None:
                digests.append(digest_text(c))
            if dec_dead(dec, keys[-1]):
                break
        text = digest_text(c)
        collect(c)
        return keys, text
    finally:
        c.close()


LAST_ACK = {}


def read_raw_ack(c, ack):
    """read_lean_raw that keeps the play's ok message in `ack`."""
    while True:
        line = c._read_line()
        if line.startswith(b'L '):
            return c._read_exact(int(line.split(b' ')[1]))
        msg = json.loads(line.decode('utf-8'))
        if msg.get('type') == 'error':
            raise RuntimeError(msg.get('msg', 'error'))
        ack.clear()
        ack.update(msg)


def run_play(template, reseed, acts, fpd, lua=None, last_rep=None):
    """One play of all the actions; (observation key, digest text)."""
    c = clone_of(template, reseed, lua) if reseed != 'snap' else template.fork(lean=True, alarm=600)
    try:
        msg = {"cmd": "play", "actions": [action_code(*a) for a in acts], "repeat": fpd, "stop_clear": False}
        if last_rep is not None:
            msg["repeats"] = [fpd] * (len(acts) - 1) + [last_rep]
        c._send(msg)
        key = obs_key(read_raw_ack(c, LAST_ACK), LeanDecoder())
        text = digest_text(c)
        collect(c)
        return key, text
    finally:
        c.close()


def key_diff(a, b):
    """Where two observation keys differ: header / room / totals fields, player fields, entity fields."""
    out = []
    if a[1] != b[1] or a[2] != b[2]:
        out.append('terrain/grid')
        try:   # which cells: grid records (index, type, variant, state, collision, x, y), terrain cells
            ga = {tuple(g[:1])[0]: g for g in json.loads(a[2])}
            gb = {tuple(g[:1])[0]: g for g in json.loads(b[2])}
            for i in sorted(set(ga) | set(gb)):
                if ga.get(i) != gb.get(i):
                    out.append(f'grid cell {i}: {ga.get(i)} vs {gb.get(i)}')
            ta, tb = a[1], b[1]
            if ta is not None and tb is not None:
                na = struct.unpack_from('<I', ta, 0)[0]
                nb = struct.unpack_from('<I', tb, 0)[0]
                ca = json.loads(ta[4:4 + na])['cells']
                cb = json.loads(tb[4:4 + nb])['cells']
                for x, y in zip(ca, cb):
                    if x != y:
                        out.append(f'terrain cell {x} vs {y}')
        except Exception as e:   # noqa: BLE001
            out.append(f'terrain decode: {e}')
    fa, fb = a[0], b[0]
    if len(fa) != len(fb):
        out.append(f'fixed length {len(fa)} vs {len(fb)}')
    h = _HEADER.size
    for name, x, y in (('header', fa[:h], fb[:h]), ('room', fa[h:h + _ROOM.size], fb[h:h + _ROOM.size]),
                       ('totals', fa[h + _ROOM.size:h + _ROOM.size + _TOTALS.size],
                        fb[h + _ROOM.size:h + _ROOM.size + _TOTALS.size])):
        if x != y:
            out.append(name)
    off = h + _ROOM.size + _TOTALS.size
    pa = np.frombuffer(fa, PLAYER, 1, off)
    pb = np.frombuffer(fb, PLAYER, 1, off) if len(fb) >= off + PLAYER.itemsize else None
    if pb is not None:
        for f in PLAYER.names:
            if pa[f][0] != pb[f][0]:
                out.append(f'player.{f} {pa[f][0]} vs {pb[f][0]}')
    n_p, n_d = fa[13], fa[14]
    off_e = h + _ROOM.size + _TOTALS.size + PLAYER.itemsize * n_p + DOOR.itemsize * n_d
    if fa[off - PLAYER.itemsize + PLAYER.itemsize * 0:off_e] != fb[off:off_e] and False:
        pass
    if fa[off + PLAYER.itemsize * n_p:off_e] != fb[off + PLAYER.itemsize * n_p:off_e]:
        out.append('doors')
    try:
        na = struct.unpack_from('<H', fa, off_e)[0]
        nb = struct.unpack_from('<H', fb, off_e)[0]
        ea = np.frombuffer(fa, ENTITY, na, off_e + 2)
        eb = np.frombuffer(fb, ENTITY, nb, off_e + 2)
        if na != nb:
            out.append(f'entities {na} vs {nb}')
        for i in range(min(na, nb)):
            for f in ENTITY.names:
                if ea[f][i] != eb[f][i]:
                    out.append(f'entity[{i}] type {ea["type"][i]}.{ea["variant"][i]} {f} {ea[f][i]} vs {eb[f][i]}')
    except Exception as e:   # noqa: BLE001
        out.append(f'entity decode: {e}')
    return out[:20]


def diff_lines(a, b, limit=12):
    la, lb = a.split('\n'), b.split('\n')
    out = []
    for i in range(max(len(la), len(lb))):
        x = la[i] if i < len(la) else None
        y = lb[i] if i < len(lb) else None
        if x != y:
            out.append(dict(i=i, a=x, b=y))
            if len(out) >= limit:
                break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:4')
    p.add_argument('--warm', type=int, default=10)
    p.add_argument('--n', type=int, default=60)
    p.add_argument('--bomb-prob', type=float, default=0.05)
    p.add_argument('--first-bomb', default='2', help='decisions (0-based, comma list) with a forced bomb press')
    p.add_argument('--bombs', type=int, default=5)
    p.add_argument('--fpd', type=int, default=4)
    p.add_argument('--park-at', type=int, default=20, help='snap path: the decision the episode is parked at')
    p.add_argument('--every', type=int, default=1, help='compare the play prefix of every k-th decision')
    p.add_argument('--paths', default='line,json,play1,play')
    p.add_argument('--variant', action='append', default=[], help='NAME=LUA')
    p.add_argument('--env', action='append', default=[], help='KEY=VALUE for the instance environment')
    p.add_argument('--no-defaults', action='store_true', help='do not apply the workers\' instance defaults')
    p.add_argument('--stub-list', default='')
    p.add_argument('--locate', action='store_true', help='on a difference find the first decision and frame')
    p.add_argument('--diag', action='store_true')
    p.add_argument('--port', type=int, default=42000)
    p.add_argument('--name', default='bmb')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    os.environ.update(dict(kv.split('=', 1) for kv in args.env))
    defaults = {} if args.no_defaults else apply_instance_defaults()
    stub = args.stub_list or default_stub_list(default_preload())
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  stub_list=stub, bombs=args.bombs)
    inst = Instance(args.name, args.port, gx, spec)
    paths = [x for x in args.paths.split(',') if x]
    variants = dict(v.split('=', 1) for v in args.variant)
    fpd = args.fpd
    rows = []
    t0 = time.perf_counter()
    try:
        for seed in parse_seeds(args.seeds):
            rng = np.random.default_rng(seed)
            inst.reset(seed)
            warm = sticky(rng, args.warm, 0.0, set())
            _, _, stop = inst.env.bridge.play([action_code(*a) for a in warm], repeat=fpd, stop_clear=True)
            if stop != 'done':
                row = dict(seed=seed, skipped=stop)
                rows.append(row)
                print(json.dumps(row), flush=True)
                continue
            template = inst.env.bridge.fork(tag='template', alarm=0)
            try:
                reseed = int(rng.integers(1, 2 ** 31 - 1))
                acts = sticky(rng, args.n, args.bomb_prob, {int(x) for x in args.first_bomb.split(',') if x})
                ref_keys, ref_text = run_steps(template, reseed, acts, fpd, 'line')
                if len(ref_keys) < len(acts):   # died: the actions up to the death
                    acts = acts[:len(ref_keys)]
                row = dict(seed=seed, n=len(acts), bombs=sum(a[2] for a in acts), ref_digest=dg(ref_text))
                res = {}
                for path in paths:
                    if path == 'line':
                        continue
                    if path in ('json', 'play1'):
                        keys, text = run_steps(template, reseed, acts, fpd, path)
                        first = next((i for i in range(len(acts)) if keys[i] != ref_keys[i]), None)
                        res[path] = dict(obs_first_diff=first, digest_equal=text == ref_text)
                    elif path == 'play':
                        step_digests = []
                        keys_d, _ = run_steps(template, reseed, acts, fpd, 'line', digests=step_digests)
                        res['lined_fixed_equal'] = sum(a[0] == b[0] for a, b in zip(keys_d, ref_keys))
                        ks = sorted(set(list(range(args.every, len(acts) + 1, args.every)) + [len(acts)]))
                        first, n_cmp, n_eq, dig_first = None, 0, 0, None
                        n_fixed = n_dig = 0
                        for k in ks:
                            key, text = run_play(template, reseed, acts[:k], fpd)
                            n_cmp += 1
                            same = key == ref_keys[k - 1]
                            n_eq += same
                            n_fixed += key[0] == ref_keys[k - 1][0]
                            n_dig += text == step_digests[k - 1]
                            if text != step_digests[k - 1] and dig_first is None:
                                dig_first = k
                                if args.diag:
                                    res['play_digest_diff'] = diff_lines(step_digests[k - 1], text)
                            if not same and first is None:
                                first = k
                                res['play_ack'] = dict(LAST_ACK)
                                res['play_obs_diff'] = key_diff(ref_keys[k - 1], key)
                            if k == len(acts):
                                res['play_final_digest_equal'] = text == ref_text
                        res['play'] = dict(compared=n_cmp, equal=n_eq, first_diff=first, fixed_equal=n_fixed,
                                           digest_equal=n_dig, digest_first_diff=dig_first)
                if 'snap' in paths and len(acts) > args.park_at:
                    # a snapshot parked by the stepping episode after park_at decisions, restored with plays
                    parked, sd = [], []
                    keys_s, _ = run_steps(template, reseed, acts, fpd, 'line', digests=sd, park_at=args.park_at,
                                          parked=parked)
                    snap = parked[0]
                    try:
                        n_cmp = n_dig = n_fixed = 0
                        first = None
                        for k in range(args.park_at + 1, len(keys_s) + 1):
                            key, text = run_play(snap, 'snap', acts[args.park_at:k], fpd)
                            n_cmp += 1
                            n_dig += text == sd[k - 1]
                            n_fixed += key[0] == keys_s[k - 1][0]
                            if (text != sd[k - 1] or key[0] != keys_s[k - 1][0]) and first is None:
                                first = k
                                res['snap_obs_diff'] = key_diff(keys_s[k - 1], key)
                                res['snap_ack'] = dict(LAST_ACK)
                                if args.diag:
                                    res['snap_digest_diff'] = diff_lines(sd[k - 1], text)
                        res['snap'] = dict(compared=n_cmp, digest_equal=n_dig, fixed_equal=n_fixed, first_diff=first,
                                           park_held=acts[args.park_at - 1])
                    finally:
                        snap.close()
                for name, lua in variants.items():
                    if name.startswith('play'):
                        key, text = run_play(template, reseed, acts, fpd, lua)
                        res[name] = dict(obs_equal=key == ref_keys[-1], digest_equal=text == ref_text)
                    else:
                        keys, text = run_steps(template, reseed, acts, fpd, 'line', lua)
                        first = next((i for i in range(len(acts)) if keys[i] != ref_keys[i]), None)
                        res[name] = dict(obs_first_diff=first, digest_equal=text == ref_text)
                row.update(res)
                if args.locate and res.get('play', {}).get('first_diff'):
                    k = res['play']['first_diff']
                    # the first k whose play digest differs (observations may lag the hidden state): search below k
                    lo = 1
                    for kk in range(max(1, k - args.every), k + 1):
                        _, tp = run_play(template, reseed, acts[:kk], fpd)
                        _, ts = run_steps(template, reseed, acts[:kk], fpd, 'line')
                        if tp != ts:
                            lo = kk
                            break
                    frames = []
                    for j in range(1, fpd + 1):
                        _, tp = run_play(template, reseed, acts[:lo], fpd, last_rep=j)
                        _, ts = run_steps(template, reseed, acts[:lo], fpd, 'line', last_rep=j)
                        frames.append(tp == ts)
                        if tp != ts:
                            row['first_frame'] = dict(decision=lo, frame=j, action=acts[lo - 1],
                                                      prev=acts[max(0, lo - 4):lo - 1])
                            if args.diag:
                                row['digest_diff'] = diff_lines(ts, tp)
                            break
                    row['frames_equal'] = frames
                rows.append(row)
                print(json.dumps(row, default=str), flush=True)
            finally:
                template.close()
    finally:
        inst.close()
        (out / 'rows.json').write_text(json.dumps(rows, indent=1, default=str))
        done = [r for r in rows if 'skipped' not in r]
        summary = dict(group=args.group, seeds=len(rows), tested=len(done), defaults=defaults, stub=stub, env=args.env,
                       fpd=args.fpd, counters=COUNTERS,
                       seconds=round(time.perf_counter() - t0, 1))
        for path in paths + list(variants):
            if path == 'line':
                continue
            if path == 'snap':
                summary['snap_compared'] = sum(r.get('snap', {}).get('compared', 0) for r in done)
                summary['snap_digest_equal'] = sum(r.get('snap', {}).get('digest_equal', 0) for r in done)
                summary['snap_fixed_equal'] = sum(r.get('snap', {}).get('fixed_equal', 0) for r in done)
                continue
            if path == 'play':
                summary['play_equal_seeds'] = sum(r.get('play', {}).get('first_diff') is None for r in done)
                summary['play_compared'] = sum(r.get('play', {}).get('compared', 0) for r in done)
                summary['play_equal'] = sum(r.get('play', {}).get('equal', 0) for r in done)
                summary['play_fixed_equal'] = sum(r.get('play', {}).get('fixed_equal', 0) for r in done)
                summary['play_digest_equal'] = sum(r.get('play', {}).get('digest_equal', 0) for r in done)
                summary['play_digest_equal_seeds'] = sum(r.get('play', {}).get('digest_first_diff') is None
                                                         for r in done)
            else:
                summary[path + '_equal_seeds'] = sum(bool(r.get(path, {}).get('digest_equal')) and
                                                     r.get(path, {}).get('obs_first_diff') is None for r in done)
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
