"""B9 checks: do the lean bridge mode and the Hallucinations stub leave the game untouched, and does the lean
observation carry the same values as format 2?

Per seed: reset, `warm` decisions, then three clones of that state play the same `steps` decisions one by one:
  full   the bridge as it was (format 2 observations);
  lean   the bridge's lean mode (fork(lean=True)): no per-frame bookkeeping, lean observations.
The hidden-state digest (abplus_goexplore.DIGEST_LUA) of the two must be equal every `check` decisions and at the end:
lean mode changes what the bridge computes, not the game. At every decision the lean observation is compared with the
format-2 one of the same state: the player's and every entity's fields the two share, the room flags, the doors.
With --stub-list a second instance runs with that ABP_STUB_LIST (stage G: Hallucinations::RecordPlayer returns at once)
and plays the same seeds and actions from its own reset; its digests must equal the first instance's.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_lean.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:32 --stub-list ../tools/stub_render_g.txt --out <dir>
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from abplus_probe_fork import digest, split_code, sticky_actions
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.fork_sampler import lean_step, raw_step

ENTITY_PAIRS = (('type', 'type'), ('variant', 'variant'), ('subtype', 'subtype'), ('size', 'size'), ('coll', 'coll'),
                ('gcoll', 'gcoll'), ('cdmg', 'cdmg'), ('aframe', 'aframe'), ('age', 'age'), ('id', 'id'))
PLAYER_PAIRS = ('hearts', 'max_hearts', 'soul', 'black', 'bone', 'eternal', 'bombs', 'keys', 'coins', 'damage',
                'fire_delay_max', 'shot_speed', 'range', 'speed', 'luck', 'active', 'head_dir', 'fire_dir',
                'move_dir', 'aframe', 'size')


def compare(full, lean):
    """First difference between a format-2 observation dict and a LeanObs of the same state, or None."""
    p, q = full['players'][0], lean.players[0]
    if (p['pos'][0], p['pos'][1]) != (q['x'], q['y']):
        return 'player.pos', p['pos'], (q['x'], q['y'])
    for k in PLAYER_PAIRS:
        if float(p[k]) != float(q[k]):
            return 'player.' + k, p[k], float(q[k])
    if bool(p['dead']) != bool(q['dead']) or bool(p['invulnerable']) != bool(q['invulnerable']):
        return 'player.flags', (p['dead'], p['invulnerable']), (q['dead'], q['invulnerable'])
    if full['room']['clear'] != lean.clear or full['room']['alive'] != lean.room[6]:
        return 'room', (full['room']['clear'], full['room']['alive']), (lean.clear, lean.room[6])
    if len(full['entities']) != len(lean.entities):
        return 'entities#', len(full['entities']), len(lean.entities)
    for i, (e, f) in enumerate(zip(full['entities'], lean.entities)):
        if (e['pos'][0], e['pos'][1]) != (f['x'], f['y']):
            return f'entity[{i}].pos', e['pos'], (f['x'], f['y'])
        for a, b in ENTITY_PAIRS:   # the lean record holds sizes and contact damage as float32
            want = np.float32(e[a]) if lean.entities.dtype[b] == np.float32 else e[a]
            if want != f[b]:
                return f'entity[{i}].{a}', e[a], f[b].item()
        kind = 2 if e.get('projectile') else 6 if 'enemy' in e else None
        if kind is not None and f['kind'] != kind:
            return f'entity[{i}].kind', kind, int(f['kind'])
        if 'enemy' in e and (bool(f['flags'] & 1), bool(f['flags'] & 4), bool(f['flags'] & 8)) != \
                (e['enemy'], e['boss'], e['blocking']):
            return f'entity[{i}].flags', (e['enemy'], e['boss'], e['blocking']), int(f['flags'])
        if 'height' in e and (np.float32(e['height']), np.float32(e['fall'])) != (f['height'], f['fall']):
            return f'entity[{i}].height', (e['height'], e['fall']), (f['height'].item(), f['fall'].item())
    doors = [(d['slot'], d['open']) for d in full['doors']]
    if doors != [(int(d['slot']), bool(d['open'])) for d in lean.doors]:
        return 'doors', doors, lean.doors.tolist()
    return None


def run_seed(inst, seed, args, rep):
    rng = np.random.default_rng(seed)
    warm, test = sticky_actions(rng, args.warm), sticky_actions(rng, args.steps)
    inst.reset(seed)
    bridge = inst.env.bridge
    _, n, stop = bridge.play(warm, repeat=rep, stop_clear=True)
    if stop != 'done':
        return None
    return bridge, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:16')
    p.add_argument('--warm', type=int, default=20)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--check', type=int, default=10)
    p.add_argument('--stub-list', default='')
    p.add_argument('--port', type=int, default=27988)
    p.add_argument('--name', default='fklean')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0)
    rep = cfg.frames_per_decision
    inst = Instance(args.name, args.port, cfg, spec)
    stubbed = None
    if args.stub_list:
        stub_cfg = GxConfig(bridge_lua=cfg.bridge_lua, preload=cfg.preload, al_stopped=True, nice=0,
                            stub_list=str(Path(args.stub_list).resolve()))
        stubbed = Instance(args.name + 's', args.port + 1, stub_cfg, spec)
    rows = []
    try:
        for seed in parse_seeds(args.seeds):
            row = dict(seed=seed)
            started = run_seed(inst, seed, args, rep)
            if started is None:
                row['skipped'] = 'episode ended in the warm-up'
                rows.append(row)
                continue
            bridge, test = started
            full, lean = bridge.fork(tag='full'), bridge.fork(tag='lean', lean=True)
            decoder = LeanDecoder()
            lean._send({"cmd": "obs"})
            lean_obs = read_lean(lean, decoder)
            row['start_obs'] = compare(full.query_obs(), lean_obs)
            digests_full, digests_lean, mismatch, steps = [], [], None, 0
            for i, code in enumerate(test):
                action = split_code(code)
                obs_full = raw_step(full, action, rep)
                obs_lean = lean_step(lean, decoder, action, rep)
                steps += 1
                d = compare(obs_full, obs_lean)
                if d is not None and mismatch is None:
                    mismatch = dict(step=i, diff=[str(x)[:160] for x in d])
                if (i + 1) % args.check == 0:
                    digests_full.append(digest(full)[0])
                    digests_lean.append(digest(lean)[0])
                if obs_full['players'][0]['dead'] or obs_full['room']['clear']:
                    break
            digests_full.append(digest(full)[0])
            digests_lean.append(digest(lean)[0])
            full.close()
            lean.close()
            row.update(steps=steps, lean_game_equal=digests_full == digests_lean, obs_mismatch=mismatch)
            if stubbed is not None:
                started = run_seed(stubbed, seed, args, rep)
                if started is None:
                    row['stub_equal'] = False
                else:
                    sb, _ = started
                    digests_stub = []
                    for i, code in enumerate(test[:steps]):
                        sb.play([code], repeat=rep, stop_clear=False)
                        if (i + 1) % args.check == 0:
                            digests_stub.append(digest(sb)[0])
                    digests_stub.append(digest(sb)[0])
                    row['stub_equal'] = digests_stub == digests_full
                    if not row['stub_equal']:
                        row['stub_first'] = next((k for k, (a, b) in enumerate(zip(digests_stub, digests_full))
                                                  if a != b), None)
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        inst.close()
        if stubbed is not None:
            stubbed.close()
        done = [r for r in rows if 'skipped' not in r]
        summary = dict(seeds=len(rows), tested=len(done), decisions=sum(r['steps'] for r in done),
                       lean_game_equal=sum(r['lean_game_equal'] for r in done),
                       lean_obs_equal=sum(r['obs_mismatch'] is None and r['start_obs'] is None for r in done),
                       stub_equal=sum(r.get('stub_equal') is True for r in done) if args.stub_list else None)
        (out / 'rows.json').write_text(json.dumps(rows, indent=1, default=str))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
