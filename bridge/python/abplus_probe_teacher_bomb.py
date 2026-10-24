"""Teacher restores with bombs (2026-10-07): does tok_sampler.hindsight's restore (a reseeded clone of the template and
one `play` of the episode's actions) reach the episode's own state when the episode pressed bombs, and are the labels
the same with and without fork_many?

Per seed: reset (the group's room, --bombs bombs), a template parked there (as tok_sampler's worker), then --episodes
episodes of sticky random actions with bomb presses (probability --bomb-prob, one forced at decision --first-bomb)
stepped with step lines on lean clones of the template reseeded as the worker reseeds; after every decision the
episode's observation (fixed part, terrain and grid content) and hidden-state digest (abplus_goexplore.DIGEST_LUA) are
kept. Every decision whose observation shows damage is a hurt; up to --hurts per episode are searched twice:
hindsight with fork_many (the default) and without (one fork + play per held move). Per depth: the restored
observation must equal the episode's at that decision, the two searches' lost[] must be equal; per hurt the first
depth's restore is repeated once more on its own and its digest compared with the episode's. Hurts after a bomb press
(a press at a decision before the hurt) are counted apart.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_teacher_bomb.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147550000:16 --out <dir>
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from abplus_probe_bomb_play import fixed_part, sticky
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean_raw
from isaac_bridge.tok_sampler import (TokSamplerConfig, apply_instance_defaults, default_stub_list, hindsight,
                                      play_actions)


def dg(c):
    return hashlib.blake2b(c.lua('return ABPGX_DIGEST(false)').encode('utf8'), digest_size=16).hexdigest()


def content(o):
    """What a decoded lean observation says (fixed part without the block flags, terrain and grid content)."""
    return (o.logic_frames, o.game_frame, o.clear, o.room, o.damage_taken, o.monsters_hp, o.blocking_hp,
            o.blocking_count, o.players.tobytes(), o.doors.tobytes(), o.entities.tobytes(),
            json.dumps([(l['id'], l['circle'], l['radius'], l['angle'], l['length'], l['width'], l['end'],
                         l['samples'].tobytes().hex()) for l in o.lasers]),
            json.dumps(o.terrain and {k: v for k, v in o.terrain.items() if k != 'version'}), json.dumps(o.grid))


def episode(template, reseed, acts, fpd):
    """Step lines; (applied, hurts, observation contents per decision 0.., digests per decision 0..)."""
    ep = template.fork(alarm=600, reseed=reseed, lean=True)
    try:
        dec = LeanDecoder()
        ep._send({"cmd": "obs"})
        o = dec.decode(read_lean_raw(ep))
        contents, digests, applied, hurts = [content(o)], [dg(ep)], [], []
        prev = o.damage_taken
        if o.dead or o.clear:
            return applied, hurts, contents, digests
        for t, (m, s, b) in enumerate(acts):
            applied.append((m, s, b))
            ep._sock.sendall(b"S %d %d %d %d 0\n" % (fpd, m, s, b))
            o = dec.decode(read_lean_raw(ep))
            contents.append(content(o))
            digests.append(dg(ep))
            if o.damage_taken - prev > 0:
                hurts.append(t + 1)
            prev = o.damage_taken
            if o.dead or o.clear:
                break
        return applied, hurts, contents, digests
    finally:
        ep.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147550000:8')
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--hurts', type=int, default=4)
    p.add_argument('--bomb-prob', type=float, default=0.05)
    p.add_argument('--first-bomb', type=int, default=2)
    p.add_argument('--bombs', type=int, default=5)
    p.add_argument('--port', type=int, default=42600)
    p.add_argument('--name', default='bmbtch')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    apply_instance_defaults()
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  stub_list=default_stub_list(default_preload()), bombs=args.bombs)
    cfg = TokSamplerConfig()
    fpd = cfg.frames_per_decision
    inst = Instance(args.name, args.port, gx, spec)
    rows, t0 = [], time.perf_counter()
    tot = dict(searches=0, after_bomb=0, depths=0, obs_equal=0, labels_equal=0, digests=0, digests_equal=0,
               searches_equal=0, after_bomb_equal=0)
    try:
        for seed in parse_seeds(args.seeds):
            rng = np.random.default_rng(seed)
            inst.reset(seed)
            template = inst.env.bridge.fork(tag='template', alarm=0)
            try:
                for e in range(args.episodes):
                    reseed = int(rng.integers(1, 2 ** 31 - 1))
                    acts = sticky(rng, args.steps, args.bomb_prob, {args.first_bomb})
                    applied, hurts, contents, digests = episode(template, reseed, acts, fpd)
                    for hurt_at in hurts[:args.hurts]:
                        bombs_before = sum(a[2] for a in applied[:hurt_at])
                        res = {}
                        for many in (True, False):
                            res[many] = hindsight(template, reseed, applied, hurt_at, cfg, 10 ** 6, fork_many=many)
                        depths = [r[0] for r in res[True]]
                        obs_eq = [content(r[1]) == contents[hurt_at - 1 - r[0]] for r in res[True]]
                        lab_eq = [a[3] == b[3] and a[0] == b[0] for a, b in zip(res[True], res[False])] + \
                            [len(res[True]) == len(res[False])]
                        d = hurt_at - 1 - cfg.teacher_depths[0]
                        dig_eq = None
                        if d >= 1:
                            st = template.fork(reseed=reseed, lean=True, alarm=300)
                            try:
                                play_actions(st, LeanDecoder(), applied[:d], fpd)
                                dig_eq = dg(st) == digests[d]
                            finally:
                                st.close()
                        same = all(obs_eq) and all(lab_eq) and dig_eq is not False
                        tot['searches'] += 1
                        tot['after_bomb'] += bombs_before > 0
                        tot['depths'] += len(depths)
                        tot['obs_equal'] += sum(obs_eq)
                        tot['labels_equal'] += all(lab_eq)
                        tot['digests'] += dig_eq is not None
                        tot['digests_equal'] += bool(dig_eq)
                        tot['searches_equal'] += same
                        tot['after_bomb_equal'] += same and bombs_before > 0
                        row = dict(seed=seed, episode=e, hurt_at=hurt_at, decisions=len(applied),
                                   bombs_before=bombs_before, depths=depths, obs_equal=obs_eq,
                                   labels_equal=all(lab_eq), digest_equal=dig_eq,
                                   lost=[r[3] for r in res[True]])
                        rows.append(row)
                        print(json.dumps(row), flush=True)
            finally:
                template.close()
    finally:
        inst.close()
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
        summary = dict(group=args.group, seeds=args.seeds, seconds=round(time.perf_counter() - t0, 1), **tot)
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
