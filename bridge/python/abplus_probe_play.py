"""A8 check of the bridge's batched replay (abp-0.2.12 play) against step, on a live instance.

Per seed: an episode of the Go-Explore exploration policy played by step (A), the same actions replayed by play after a
reset (B), and half of them by play, the rest by step (C). The hidden-state digest (abplus_goexplore.DIGEST_LUA) and the
observation (without the bridge's frame counter, entity ids and tear credits, as in A7) at the end must be identical,
and B must stop where A ended (death, win, time limit). Also: malformed play commands are refused with the connection
still usable, and the logic frames per second of step and play.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_play.py --groups-file ../catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:16 --steps 300 --out <dir>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, load_spec, parse_seeds
from isaac_bridge.abplus_goexplore import Explorer, GxConfig, Instance
from isaac_bridge.env import BridgeError

IGN = {'logic_frames', 'game_frame', 'id', 'frame', 'credits'}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in IGN}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def first_diff(a, b, path=''):
    if type(a) != type(b):
        return path, str(a)[:200], str(b)[:200]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return path + '.' + k, str(a.get(k))[:200], str(b.get(k))[:200]
            d = first_diff(a[k], b[k], path + '.' + k)
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return path + '#len', len(a), len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f'{path}[{i}]')
            if d:
                return d
        return None
    return None if a == b else (path, str(a)[:200], str(b)[:200])


def obs_view(raw):
    return strip({k: raw.get(k) for k in ('players', 'entities', 'room', 'combat', 'events', 'doors')})


def digest_lines_diff(a, b):
    la, lb = a.split('\n'), b.split('\n')
    sa, sb = set(la), set(lb)
    return dict(only_a=[l[:300] for l in la if l not in sb][:3], only_b=[l[:300] for l in lb if l not in sa][:3])


def protocol_checks(inst):
    """Malformed play commands are errors; the game stays paused and the connection usable."""
    bridge, out = inst.env.bridge, {}
    for name, kwargs in (('empty', dict(codes=[])), ('repeats', dict(codes=[0, 0], repeats=[4]))):
        try:
            bridge.play(**kwargs)
            out[name] = 'accepted (wrong)'
        except BridgeError as exc:
            out[name] = f'refused: {exc}'
    before = inst.env.raw_obs['logic_frames']
    obs = bridge.query_obs()
    out['usable_after'] = obs['logic_frames'] == before
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--tasks')
    src.add_argument('--groups-file')
    p.add_argument('--group', default='normal')
    p.add_argument('--seconds', type=float, default=None)
    p.add_argument('--seeds', required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--frames-per-decision', type=int, default=4)
    p.add_argument('--bomb-prob', type=float, default=0.02)
    p.add_argument('--mode', default='exact')
    p.add_argument('--name', default='gxplay')
    p.add_argument('--port', type=int, default=27590)
    p.add_argument('--bridge-lua', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    spec = load_spec(args)
    cfg = GxConfig(frames_per_decision=args.frames_per_decision, bomb_prob=args.bomb_prob, mode=args.mode,
                   bridge_lua=args.bridge_lua or default_bridge_lua())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    inst = Instance(args.name, args.port, cfg, spec)
    rows = []
    try:
        env = inst.env
        inst.reset(seeds[0])
        checks = protocol_checks(inst)
        t = time.perf_counter()
        for _ in range(20):
            inst.digest()
        checks['digest_ms'] = round(1000 * (time.perf_counter() - t) / 20, 2)
        print(json.dumps(dict(protocol=checks)), flush=True)
        for seed in seeds:
            # A: step by step
            inst.reset(seed)
            explorer = Explorer(cfg, np.random.default_rng(seed))
            actions, outcome_a = [], 'running'
            t = time.perf_counter()
            for _ in range(args.steps):
                a = explorer.act(env.raw_obs)
                _, _, terminated, truncated, info = env.step(np.asarray(a))
                actions.append(a)
                outcome_a = info['outcome']
                if terminated or truncated:
                    break
            step_s = time.perf_counter() - t
            frames_a, dig_a, obs_a = env.elapsed_frames, inst.digest_text(), obs_view(env.raw_obs)
            # B: one play
            inst.reset(seed)
            t = time.perf_counter()
            _, played, outcome_b = env.play(actions)
            play_s = time.perf_counter() - t
            dig_b, obs_b = inst.digest_text(), obs_view(env.raw_obs)
            row = dict(seed=seed, steps=len(actions), frames=frames_a, outcome=outcome_a,
                       b_played=played, b_outcome=outcome_b, b_frames=env.elapsed_frames,
                       b_digest=dig_b == dig_a, b_obs=first_diff(obs_a, obs_b),
                       step_fps=round(frames_a / step_s), play_fps=round(frames_a / play_s))
            if dig_b != dig_a:
                row['b_digest_diff'] = digest_lines_diff(dig_a, dig_b)
            # C: half play, half step
            k = len(actions) // 2
            inst.reset(seed)
            if k:
                env.play(actions[:k])
            for a in actions[k:]:
                _, _, terminated, truncated, _ = env.step(np.asarray(a))
                if terminated or truncated:
                    break
            dig_c = inst.digest_text()
            row.update(c_digest=dig_c == dig_a, c_frames=env.elapsed_frames)
            if dig_c != dig_a:
                row['c_digest_diff'] = digest_lines_diff(dig_a, dig_c)
            row['ok'] = bool(played == len(actions) and outcome_b == outcome_a and row['b_frames'] == frames_a
                             and row['b_digest'] and row['b_obs'] is None and row['c_digest']
                             and row['c_frames'] == frames_a)
            rows.append(row)
            with open(out / 'play_check.jsonl', 'a') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps({k: v for k, v in row.items() if not k.endswith('_diff')}), flush=True)
    finally:
        inst.close()
    frames = sum(r['frames'] for r in rows)
    summary = dict(protocol=checks, seeds=len(rows), ok=sum(r['ok'] for r in rows),
                   outcomes={o: sum(r['outcome'] == o for r in rows) for o in {r['outcome'] for r in rows}},
                   game_hours=round(3 * frames / 30 / 3600, 4),
                   step_fps=round(np.median([r['step_fps'] for r in rows])) if rows else None,
                   play_fps=round(np.median([r['play_fps'] for r in rows])) if rows else None)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(summary=summary)), flush=True)


if __name__ == '__main__':
    main()
