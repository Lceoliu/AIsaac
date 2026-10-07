"""A19 (gate 0): is a fork()ed clone of the game an exact copy of it, and does forking leave the parent untouched?

Per seed (a room of the group, as Go-Explore's tasks): reset, `warm` decisions of a sticky random policy, then
  1. the parent's hidden-state digest D0 (abplus_goexplore.DIGEST_LUA: the player, every entity incl. NPC AI state,
     seeds, grid and the global MT) and a clone (bridge.fork): the clone's digest must be D0;
  2. `steps` further decisions drawn in advance, in chunks of `chunk`: the parent plays them (digest after each chunk),
     then the clone plays the same and must stop at the same decision for the same reason with the same digests and the
     same final observation;
  3. a clone of the clone, made before the clone moved, plays them too (clones can be cloned);
  4. reference: the same seed reset again in the same instance and every action replayed without a fork; the parent's
     final digest must equal it (forking did not change the parent; replays themselves are exact in about 99% of rooms,
     EXPERIMENTS.md A7/A8, so a rare mismatch here can be the known residual and is reported separately).
Also the fork latency (command sent to the clone's hello received) and the logic frames per second of parent and clone.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_fork.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:32 --out <dir>
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from abplus_probe_play import first_diff, obs_view
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV, action_code
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.env import BridgeError


def sticky_actions(rng, n, repeat_prob=0.9, bomb_prob=0.0):
    """n action codes of the bridge's play command (abplus.action_code): the move and the shot are kept with
    repeat_prob, else drawn anew."""
    out, move, shoot = [], 0, 0
    for _ in range(n):
        if rng.random() >= repeat_prob:
            move, shoot = int(rng.integers(9)), int(rng.integers(5))
        out.append(action_code(move, shoot, int(rng.random() < bomb_prob)))
    return out


def split_code(code):
    """(move, shoot, bomb, item) of a play code, as the bridge's play_next reads it: what a step command needs."""
    code = int(code)
    return code % 9, (code // 9) % 5, (code // 45) % 2, (code // 90) % 2


def digest(bridge):
    text = bridge.lua('return ABPGX_DIGEST(false)')
    return hashlib.blake2b(text.encode('utf8'), digest_size=16).hexdigest(), text


def play_chunks(bridge, actions, chunk, repeat):
    """The actions in chunks; (digests after each chunk, decisions played, stop reason, last observation, seconds)."""
    digests, played, stop, obs, t = [], 0, 'done', None, time.perf_counter()
    for at in range(0, len(actions), chunk):
        obs, n, stop = bridge.play(actions[at:at + chunk], repeat=repeat, stop_clear=True)
        played += n
        digests.append(digest(bridge)[0])
        if stop != 'done':
            break
    return digests, played, stop, obs, time.perf_counter() - t


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:16')
    p.add_argument('--warm', type=int, default=20)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--chunk', type=int, default=10)
    p.add_argument('--port', type=int, default=27910)
    p.add_argument('--name', default='fkprobe')
    p.add_argument('--bridge-lua', default='')
    p.add_argument('--preload', default='')
    p.add_argument('--driver-gl', action='store_true', help='keep the display driver\'s OpenGL (negative control)')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    env = dict(FORK_ENV)
    if args.driver_gl:
        for k in ('__GLX_VENDOR_LIBRARY_NAME', 'LIBGL_ALWAYS_SOFTWARE'):
            env.pop(k)
    os.environ.update(env)   # launch_abplus passes this process's environment on to the game
    cfg = GxConfig(bridge_lua=args.bridge_lua or default_bridge_lua(), preload=args.preload or default_preload(),
                   al_stopped=True)
    inst = Instance(args.name, args.port, cfg, spec)
    rows = []
    try:
        for seed in parse_seeds(args.seeds):
            rng = np.random.default_rng(seed)
            warm, test = sticky_actions(rng, args.warm), sticky_actions(rng, args.steps)
            row = dict(seed=seed)
            inst.reset(seed)
            bridge = inst.env.bridge
            raw = inst.env.raw_obs
            row['room'] = [raw['room'].get('type'), raw['room'].get('variant')]
            _, n, stop = bridge.play(warm, repeat=cfg.frames_per_decision, stop_clear=True)
            if stop != 'done':
                row['skipped'] = f'episode ended in the warm-up ({stop} after {n})'
                rows.append(row)
                continue
            d0, text0 = digest(bridge)
            t = time.perf_counter()
            try:
                clone = bridge.fork(tag=f'{seed}-a')
            except (BridgeError, OSError) as exc:
                row['fork_error'] = repr(exc)
                rows.append(row)
                print(json.dumps(row), flush=True)
                continue
            row['fork_ms'] = round(1000 * (time.perf_counter() - t), 3)
            row['turbo'] = clone.hello.get('turbo')
            dc0, textc0 = digest(clone)
            row['clone_start_equal'] = dc0 == d0
            if dc0 != d0:
                a, b = text0.split('\n'), textc0.split('\n')
                row['clone_start_diff'] = [(x[:200], y[:200]) for x, y in zip(a, b) if x != y][:3]
            t = time.perf_counter()
            clone2 = clone.fork(tag=f'{seed}-b')
            row['fork2_ms'] = round(1000 * (time.perf_counter() - t), 3)
            dp, played, stop, obs_p, sec_p = play_chunks(bridge, test, args.chunk, cfg.frames_per_decision)
            row.update(played=played, stop=stop, parent_fps=round(played * cfg.frames_per_decision / sec_p, 1))
            for label, c in (('clone', clone), ('clone2', clone2)):
                try:
                    dc, played_c, stop_c, obs_c, sec_c = play_chunks(c, test, args.chunk, cfg.frames_per_decision)
                    same = dc == dp and played_c == played and stop_c == stop
                    row[f'{label}_equal'] = same
                    row[f'{label}_fps'] = round(played_c * cfg.frames_per_decision / sec_c, 1)
                    diff = first_diff(obs_view(obs_p), obs_view(obs_c))
                    row[f'{label}_obs_equal'] = diff is None
                    if not same or diff is not None:
                        first = next((i for i, (x, y) in enumerate(zip(dp, dc)) if x != y), None)
                        row[f'{label}_detail'] = dict(played=played_c, stop=stop_c, first_chunk=first, obs=diff)
                except (BridgeError, OSError) as exc:
                    row[f'{label}_error'] = repr(exc)
                finally:
                    try:
                        c.close()
                    except OSError:
                        pass
            # the reference: no fork
            inst.reset(seed)
            bridge = inst.env.bridge
            bridge.play(warm, repeat=cfg.frames_per_decision, stop_clear=True)
            row['reference_start_equal'] = digest(bridge)[0] == d0
            dr, played_r, stop_r, _, _ = play_chunks(bridge, test, args.chunk, cfg.frames_per_decision)
            row['parent_equals_reference'] = dr == dp and played_r == played and stop_r == stop
            row['fork_counters'] = bridge.lua("return tostring(os.getenv('ABP_FORK_COUNTERS'))")
            row['rss_mib'] = round(inst.rss_mib(), 1)
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        inst.close()
        done = [r for r in rows if 'skipped' not in r and 'fork_error' not in r]

        def count(key):
            return sum(1 for r in done if r.get(key) is True)

        summary = dict(seeds=len(rows), tested=len(done), skipped=sum('skipped' in r for r in rows),
                       fork_errors=sum('fork_error' in r for r in rows),
                       clone_start_equal=count('clone_start_equal'), clone_equal=count('clone_equal'),
                       clone_obs_equal=count('clone_obs_equal'), clone2_equal=count('clone2_equal'),
                       clone_errors=sum('clone_error' in r or 'clone2_error' in r for r in done),
                       reference_start_equal=count('reference_start_equal'),
                       parent_equals_reference=count('parent_equals_reference'),
                       fork_ms_median=float(np.median([r['fork_ms'] for r in done])) if done else None,
                       parent_fps_median=float(np.median([r['parent_fps'] for r in done])) if done else None,
                       clone_fps_median=float(np.median([r['clone_fps'] for r in done if 'clone_fps' in r] or [0])))
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
