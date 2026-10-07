"""B9: SIGPROF profile of the game process (abp_turbo ABP_PROF) while one instance steps in a chosen way.

  play   batched play (bridge abp-0.2.12): per-frame work only, one observation per 50 decisions;
  step   one format-2 observation per decision (the bridge as the training loop used it);
  lean   the bridge's lean mode (abp-0.2.14): no per-frame bookkeeping, native input, one lean observation per decision.
The player is made invincible so the room stays in combat for the whole run; when the room is cleared anyway the seed is
reset. Prints the decisions per second and writes <out>.marks with the profile sample index at the start of the timed
part (the FIRST_SAMPLE argument of analysis/scripts/abplus/prof_report2.py).

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_prof_step.py lean 2147500004 25 /path/out.prof [--group normal] [--stub-list FILE]
"""
import argparse
import os
import time

import numpy as np

from abplus_bench_step import raw_step
from abplus_probe_fork import split_code, sticky_actions
from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.fork_sampler import lean_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('play', 'step', 'lean'))
    p.add_argument('seed', type=int)
    p.add_argument('seconds', type=float)
    p.add_argument('out')
    p.add_argument('--group', default='normal')
    p.add_argument('--groups-file', default='../abplus/catalog/scaling2_groups.json')
    p.add_argument('--stub-list', default='')
    p.add_argument('--port', type=int, default=27995)
    args = p.parse_args()
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group=args.group, tasks='', seconds=0.0))
    os.environ.update(FORK_ENV)
    os.environ['ABP_PROF'] = args.out
    os.environ['ABP_STATS'] = args.out + '.stats'
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                   stub_list=os.path.abspath(args.stub_list) if args.stub_list else '')
    inst = Instance('fkprof', args.port, cfg, spec)
    try:
        rng = np.random.default_rng(args.seed)
        decoder = LeanDecoder()

        def start():
            inst.reset(args.seed)
            b = inst.env.bridge
            b.lua("AbpSetInvincible(true); return 1")
            if args.mode == 'lean':
                b._send({"cmd": "lean", "enabled": True})
                b._recv()
            return b

        b = start()
        open(args.out + '.marks', 'w').write('start ' + open(args.out + '.stats').read().strip().split('\n')[-1] + '\n')
        t0, n, resets = time.perf_counter(), 0, 0
        while time.perf_counter() - t0 < args.seconds:
            acts = sticky_actions(rng, 50)
            clear = False
            if args.mode == 'play':
                obs, k, stop = b.play(acts, repeat=4, stop_clear=False)
                n += k
                clear = obs['room']['clear']
            else:
                for code in acts:
                    if args.mode == 'lean':
                        obs = lean_step(b, decoder, split_code(code), 4)
                        clear = obs.clear
                    else:
                        raw_step(b, code, 4, False)
                    n += 1
                if args.mode == 'step':
                    clear = b.query_obs()['room']['clear']
            if clear:
                if args.mode == 'lean':
                    b._send({"cmd": "lean", "enabled": False})
                    b._recv()
                b = start()
                resets += 1
        dt = time.perf_counter() - t0
        counters = b.lua("return tostring(os.getenv('ABP_FORK_COUNTERS'))")
        print(args.mode, 'decisions', n, 'resets', resets, 'ms/decision %.3f' % (1000 * dt / n),
              'frames/s %.0f' % (4 * n / dt), counters[counters.find('native_input'):])
    finally:
        inst.close()


if __name__ == '__main__':
    main()
