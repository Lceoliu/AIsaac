"""SIGPROF profile (abp_turbo ABP_PROF) of the game process in the bridge's lean mode while it only plays frames
(`play` batches of 50 decisions, one observation per batch): what a logic frame costs and how much of it is Lua (the
bridge's per-frame callbacks), the render path and the rest. The player is invincible so the room stays in combat.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_prof_lean_play.py 2147500004 25 /path/out.prof
  python3 ../../analysis/.../prof_report2.py /path/out.prof <isaac.x64> <first sample from out.prof.marks>
"""
import argparse
import os
import time

import numpy as np

from abplus_probe_fork import sticky_actions
from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean


def main():
    p = argparse.ArgumentParser()
    p.add_argument('seed', type=int)
    p.add_argument('seconds', type=float)
    p.add_argument('out')
    p.add_argument('--group', default='normal')
    p.add_argument('--groups-file', default='../abplus/catalog/scaling2_groups.json')
    p.add_argument('--port', type=int, default=34700)
    args = p.parse_args()
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group=args.group, tasks='', seconds=0.0))
    os.environ.update(FORK_ENV)
    os.environ['ABP_PROF'] = args.out
    os.environ['ABP_STATS'] = args.out + '.stats'
    stub = os.path.join(os.path.dirname(default_preload()), 'stub_render_g.txt')
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                   stub_list=stub if os.path.isfile(stub) else '')
    inst = Instance('stpprof', args.port, cfg, spec)
    try:
        rng = np.random.default_rng(args.seed)
        dec = LeanDecoder()

        def start():
            inst.reset(args.seed)
            b = inst.env.bridge
            b.lua("AbpSetInvincible(true); return 1")
            b._send({"cmd": "lean", "enabled": True})
            b._recv()
            return b

        b = start()
        open(args.out + '.marks', 'w').write('start ' + open(args.out + '.stats').read().strip().split('\n')[-1] + '\n')
        t0, n, resets = time.perf_counter(), 0, 0
        while time.perf_counter() - t0 < args.seconds:
            acts = sticky_actions(rng, 50)
            b._send({"cmd": "play", "actions": acts, "repeat": 4, "stop_clear": False})
            o = read_lean(b, dec)
            n += len(acts)
            if o.clear or o.dead:
                b._send({"cmd": "lean", "enabled": False})
                b._recv()
                b = start()
                resets += 1
        dt = time.perf_counter() - t0
        print('decisions', n, 'resets', resets, 'ms/decision %.3f' % (1000 * dt / n), 'frames/s %.0f' % (4 * n / dt))
    finally:
        inst.close()


if __name__ == '__main__':
    main()
