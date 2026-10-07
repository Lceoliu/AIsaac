"""Single-instance cost of the lean observation, Lua (native_obs mode 0) against native (mode 1).

abplus_bench_step.py times the format-2 path; this times the lean path the fork sampler uses. Per seed: reset, `warm`
decisions, then for each round and each mode (order alternating between rounds) a lean clone of that same state plays
the same sticky random actions one `step` at a time (until the player dies or the room is clear, at most --steps); each
observation is read and decoded with LeanDecoder, as a sampler worker does. The game goes through the same states in
both modes (abplus_probe_native_obs.py), so the difference is the observation's cost. Reported: ms per decision, logic
frames per second and x real time per mode and frames per decision (--fpd, e.g. 4,2), the decode share and the game process's CPU time per decision (schedstat).

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_bench_native_obs.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:12 --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from abplus_probe_fork import split_code, sticky_actions
from abplus_probe_native_obs import native_obs, read_raw
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder


def cpu_ns(pid):
    """CPU time the process has run (ns, /proc/<pid>/schedstat): less sensitive to a loaded host than wall time."""
    try:
        return int(open(f'/proc/{pid}/schedstat').read().split()[0])
    except (OSError, ValueError, IndexError):
        return 0


def run_clone(parent, mode, actions, fpd):
    c = parent.fork(tag=f'bench{mode}', lean=True, alarm=600)
    native_obs(c, mode)
    dec = LeanDecoder()
    c._send({"cmd": "obs"})
    dec.decode(read_raw(c))
    n, ents, t_dec = 0, 0, 0.0
    cpu0 = cpu_ns(c.pid)
    t0 = time.perf_counter()
    for code in actions:
        move, shoot, bomb, item = split_code(code)
        c._send({"cmd": "step", "repeat": fpd, "move": move, "shoot": shoot, "bomb": bomb, "item": item})
        raw = read_raw(c)
        t1 = time.perf_counter()
        o = dec.decode(raw)
        t_dec += time.perf_counter() - t1
        n += 1
        ents += len(o.entities)
        if o.dead or o.clear:
            break
    dt = time.perf_counter() - t0
    cpu = (cpu_ns(c.pid) - cpu0) / 1e9
    c.close()
    return n, dt, t_dec, ents, cpu


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:12')
    p.add_argument('--warm', type=int, default=10)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--fpd', default='4,2')
    p.add_argument('--port', type=int, default=31500)
    p.add_argument('--name', default='nobbench')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                   stub_list=stub if Path(stub).is_file() else '')
    inst = Instance(args.name, args.port, cfg, spec)
    fpds = [int(v) for v in args.fpd.split(',')]
    acc = {(f, m): [0, 0.0, 0.0, 0, 0.0] for f in fpds for m in (0, 1)}
    rows = []
    try:
        for seed in parse_seeds(args.seeds):
            rng = np.random.default_rng(seed)
            warm, actions = sticky_actions(rng, args.warm), sticky_actions(rng, args.steps)
            inst.reset(seed)
            _, _, stop = inst.env.bridge.play(warm, repeat=cfg.frames_per_decision, stop_clear=True)
            if stop != 'done':
                continue
            row = dict(seed=seed)
            for r in range(args.rounds):
                for f in fpds:
                    for m in ((0, 1) if r % 2 == 0 else (1, 0)):
                        n, dt, t_dec, ents, cpu = run_clone(inst.env.bridge, m, actions, f)
                        a = acc[(f, m)]
                        a[0] += n; a[1] += dt; a[2] += t_dec; a[3] += ents; a[4] += cpu
                        row[f'fpd{f}_mode{m}_ms'] = round(1000 * dt / max(n, 1), 4)
                        row[f'fpd{f}_decisions'] = n
            row['entities_mean'] = ents / max(n, 1)
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        inst.close()
        summary = dict(group=args.group, seeds=len(rows), load=os.getloadavg())
        for (f, m), (n, dt, t_dec, ents, cpu) in acc.items():
            if n:
                summary[f'fpd{f}_{"native" if m else "lua"}'] = dict(
                    decisions=n, ms_per_decision=round(1000 * dt / n, 4), decode_ms=round(1000 * t_dec / n, 4),
                    game_cpu_ms_per_decision=round(1000 * cpu / n, 4),
                    frames_per_s=round(f * n / dt, 1), x_real_time=round(f * n / dt / 30, 1),
                    decisions_per_s=round(n / dt, 1), entities_mean=round(ents / n, 2))
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
