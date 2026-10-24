"""Stack-sampling profile of lean clones driven as a sampler worker drives them (2026-10-04, frame-cost work).

The root instance runs with abp_turbo's ABP_PROF + ABP_PROF_STACK (each 1 kHz main-thread sample keeps the RIP and the
return addresses above it) and ABP_PROF_CLONES=<out>/clones (every clone samples into its own files). Per seed: reset,
`warm` decisions, then a lean clone plays sticky random actions with step lines (the native step path of the training
workers) until the player dies, the room is clear or --steps; the clone flushes its profile and is closed. Seeds cycle
until --seconds of clone play have passed. Optional --invincible keeps rooms in combat longer (as B9's profile).
Reports ms per decision of the clones; analyse with analysis/scripts/abplus/prof_stack_report.py.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_prof_frame.py --group normal --seconds 60 --out <dir> [--fpd 4] [--name frmprof --port 37100]
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
from isaac_bridge.abplus_lean import LeanDecoder, read_lean


def cpu_ns(pid):
    try:
        return int(open(f'/proc/{pid}/schedstat').read().split()[0])
    except (OSError, ValueError, IndexError):
        return 0


def drive_clone(c, actions, fpd):
    """Step lines until done; (decisions, seconds, last observation)."""
    dec = LeanDecoder()
    c._send({"cmd": "obs"})
    obs = read_lean(c, dec)
    sock = c._sock
    n, t0 = 0, time.perf_counter()
    for code in actions:
        if obs.dead or obs.clear:
            break
        move, shoot, bomb, _ = split_code(code)
        sock.sendall(b"S %d %d %d %d %d\n" % (fpd, move, shoot, bomb, 0))
        hdr = c._read_line()
        obs = dec.decode(c._read_exact(int(hdr.split(b' ')[1])))
        n += 1
    return n, time.perf_counter() - t0, obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='../abplus/catalog/scaling2_groups.json')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seeds', default='range:2147500000:400')
    p.add_argument('--seconds', type=float, default=60.0)
    p.add_argument('--warm', type=int, default=10)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--fpd', type=int, default=4)
    p.add_argument('--depth', type=int, default=40)
    p.add_argument('--invincible', action='store_true')
    p.add_argument('--no-prof', action='store_true', help='no sampling: timing only')
    p.add_argument('--stub-list', default='', help="ABP_STUB_LIST ('' = the copy's stub_render_g.txt)")
    p.add_argument('--env', default='', help='extra instance environment, k=v,k=v')
    p.add_argument('--port', type=int, default=37100)
    p.add_argument('--name', default='frmprof')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    (out / 'clones').mkdir(parents=True, exist_ok=False)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group=args.group, tasks=args.tasks, seconds=0.0))
    os.environ.update(FORK_ENV)
    if not args.no_prof:
        os.environ.update({'ABP_PROF': str(out / 'root.prof'), 'ABP_STATS': str(out / 'root.prof.stats'),
                           'ABP_PROF_STACK': str(args.depth), 'ABP_PROF_CLONES': str(out / 'clones')})
    for kv in filter(None, args.env.split(',')):
        k, v = kv.split('=', 1)
        os.environ[k] = v
    stub = args.stub_list or str(Path(default_preload()).parent / 'stub_render_g.txt')
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0, stub_list=stub)
    inst = Instance(args.name, args.port, cfg, spec)
    total_n, total_t, episodes, total_cpu = 0, 0.0, 0, 0
    load0 = os.getloadavg()
    try:
        for seed in parse_seeds(args.seeds):
            if total_t >= args.seconds:
                break
            rng = np.random.default_rng(seed)
            warm, actions = sticky_actions(rng, args.warm), sticky_actions(rng, args.steps)
            inst.reset(seed)
            b = inst.env.bridge
            if args.invincible:
                b.lua("AbpSetInvincible(true); return 1")
            _, _, stop = b.play(warm, repeat=args.fpd, stop_clear=True)
            if stop != 'done':
                continue
            c = b.fork(tag='prof', lean=True, alarm=600)
            cpu0 = cpu_ns(c.pid)
            n, t, _ = drive_clone(c, actions, args.fpd)
            total_cpu += cpu_ns(c.pid) - cpu0
            if not args.no_prof:
                c.lua("return tostring(os.getenv('ABP_PROF_FLUSH'))")
            c.close()
            if args.invincible:
                b.lua("AbpSetInvincible(false); return 1")
            total_n, total_t, episodes = total_n + n, total_t + t, episodes + 1
    finally:
        inst.close()
    summary = dict(group=args.group, fpd=args.fpd, episodes=episodes, decisions=total_n, seconds=round(total_t, 3),
                   ms_per_decision=round(1000 * total_t / max(total_n, 1), 4),
                   game_cpu_ms_per_decision=round(total_cpu / 1e6 / max(total_n, 1), 4),
                   frames_per_s=round(args.fpd * total_n / max(total_t, 1e-9), 1), load_start=load0,
                   load_end=os.getloadavg(), stub_list=stub, env=args.env)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
