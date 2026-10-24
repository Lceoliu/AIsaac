"""B9: where one decision's time goes on a single AB+ instance (the worker side of the training loop).

Per seed (a room of the group) the same sticky random actions are played four ways after the same reset, so the game
goes through the same states each time:
  play    one batched `play` (bridge abp-0.2.12): the engine and the bridge's per-frame Lua callbacks, one observation;
  raw     one `step` per decision, the binary observation received and thrown away: + Lua pack_obs and the socket;
  decode  the same with ObsDecoder.decode: + the Python dict;
  env     AbplusTransformerEnv.step: + VisibleHistory.encode, masks and the episode bookkeeping (the worker's path
          without reward, geometry labels and shared-memory copies).
Reported per way: logic frames per second and milliseconds per decision, and the mean entity count.
With --prof DIR the instance runs under abp_turbo's SIGPROF sampler; the sample index at each phase boundary is written
so prof_report.py can split the game process's own time (FIRST_SAMPLE argument).

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_bench_step.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:8 --steps 150 --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from abplus_probe_fork import split_code, sticky_actions
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance


def raw_step(bridge, code, repeat, decode):
    move, shoot, bomb, item = split_code(code)
    bridge._send({"cmd": "step", "repeat": repeat, "move": move, "shoot": shoot, "bomb": bomb, "item": item})
    line = bridge._read_line()
    if not line.startswith(b"B "):
        raise RuntimeError(f"unexpected reply {line[:80]!r}")
    payload = bridge._read_exact(int(line.split(b" ")[1]))
    return (bridge.decoder.decode(payload) if decode else None), len(payload)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:8')
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--port', type=int, default=27950)
    p.add_argument('--name', default='fkbench')
    p.add_argument('--bridge-lua', default='')
    p.add_argument('--preload', default='')
    p.add_argument('--driver-gl', action='store_true')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    env = dict(FORK_ENV)
    if args.driver_gl:
        for k in ('__GLX_VENDOR_LIBRARY_NAME', 'LIBGL_ALWAYS_SOFTWARE'):
            env.pop(k)
    os.environ.update(env)
    cfg = GxConfig(bridge_lua=args.bridge_lua or default_bridge_lua(), preload=args.preload or default_preload(),
                   al_stopped=True, nice=0)
    inst = Instance(args.name, args.port, cfg, spec)
    rep = cfg.frames_per_decision
    rows = []
    try:
        for seed in parse_seeds(args.seeds):
            actions = sticky_actions(np.random.default_rng(seed), args.steps)
            row = dict(seed=seed)
            # play: how many decisions the episode lasts with these actions
            inst.reset(seed)
            bridge = inst.env.bridge
            raw = inst.env.raw_obs
            row['room'] = [raw['room'].get('type'), raw['room'].get('variant')]
            t = time.perf_counter()
            _, n, stop = bridge.play(actions, repeat=rep, stop_clear=True)
            row['play_s'] = time.perf_counter() - t
            row.update(decisions=n, stop=stop)
            if n < 20:
                rows.append(row)
                continue
            acts = actions[:n - 1]   # the last decision ends the episode; the stepped ways stop before it
            for way in ('raw', 'decode'):
                inst.reset(seed)
                bridge = inst.env.bridge
                ents, size = [], 0
                t = time.perf_counter()
                for code in acts:
                    obs, nbytes = raw_step(bridge, code, rep, way == 'decode')
                    size += nbytes
                    if obs is not None:
                        ents.append(len(obs['entities']))
                row[f'{way}_s'] = time.perf_counter() - t
                if ents:
                    row['entities_mean'] = float(np.mean(ents))
                    row['entities_max'] = int(np.max(ents))
                row['payload_bytes_mean'] = size / len(acts)
            inst.reset(seed)
            t = time.perf_counter()
            env_steps = 0
            for code in acts:
                move, shoot, bomb, item = split_code(code)
                _, _, terminated, truncated, _ = inst.env.step(np.array([5 * move + shoot, bomb, item]))
                env_steps += 1
                if terminated or truncated:   # the environment's own episode end (its time limit) can come first
                    break
            row['env_s'] = time.perf_counter() - t
            k = len(acts)
            row['ms_per_decision'] = dict(play=1000 * row['play_s'] / n, raw=1000 * row['raw_s'] / k,
                                          decode=1000 * row['decode_s'] / k, env=1000 * row['env_s'] / env_steps)
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        inst.close()
        done = [r for r in rows if 'ms_per_decision' in r]
        if done:
            total = {w: sum(r['ms_per_decision'][w] * (r['decisions'] - 1) for r in done) /
                     sum(r['decisions'] - 1 for r in done) for w in ('play', 'raw', 'decode', 'env')}
            summary = dict(seeds=len(done), frames_per_decision=rep, ms_per_decision=total,
                           frames_per_s={w: 1000 * rep / v for w, v in total.items()},
                           entities_mean=float(np.mean([r['entities_mean'] for r in done])),
                           payload_bytes_mean=float(np.mean([r['payload_bytes_mean'] for r in done])))
            (out / 'summary.json').write_text(json.dumps(summary, indent=1))
            print('SUMMARY', json.dumps(summary))
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
