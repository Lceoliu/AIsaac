"""Diagnostic (2026-10-04): how often a lean step observation carries a terrain block (flag 2) or a map block (flag 8),
and at which logic frames. Room mode, sticky random actions, step lines.

usage: python abplus_probe_lean_blocks.py --groups-file ../abplus/catalog/scaling2_groups.json --seeds range:2147500000:4
"""
import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np

from abplus_probe_fork import split_code, sticky_actions
from abplus_probe_native_obs import read_raw
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:4')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--port', type=int, default=34300)
    args = p.parse_args()
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                   stub_list=stub if Path(stub).is_file() else '')
    inst = Instance('stpblk', args.port, cfg, spec)
    total = collections.Counter()
    try:
        for seed in parse_seeds(args.seeds):
            rng = np.random.default_rng(seed)
            inst.reset(seed)
            c = inst.env.bridge.fork(tag='blk', lean=True, alarm=300)
            dec = LeanDecoder()
            c._send({"cmd": "obs"})
            dec.decode(read_raw(c))
            marks = []
            for t, code in enumerate(sticky_actions(rng, args.steps)):
                m, s, b, i = split_code(code)
                c._sock.sendall(b"S 4 %d %d %d 0\n" % (m, s, b))
                raw = read_raw(c)
                o = dec.decode(raw)
                f = raw[12]
                total['records'] += 1
                if f & 2:
                    total['terrain'] += 1
                    marks.append(('T', t, o.logic_frames))
                if f & 8:
                    total['map'] += 1
                    marks.append(('M', t, o.logic_frames))
                if o.dead or o.clear:
                    break
            c.close()
            print(seed, marks[:40], flush=True)
    finally:
        inst.close()
    print('TOTAL', json.dumps(total))


if __name__ == '__main__':
    main()
