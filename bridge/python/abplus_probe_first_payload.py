"""Diagnostic (2026-10-04): the first lean payload of a clone after reset + warm-up, header and totals decoded, plus a
few steps' headers; for comparing two bridge copies by eye.

usage: python abplus_probe_first_payload.py --seed 2147500000 --out <file.json>
"""
import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np

from abplus_probe_fork import split_code, sticky_actions
from abplus_probe_native_obs import read_raw
from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='../abplus/catalog/scaling2_groups.json')
    p.add_argument('--group', default='normal')
    p.add_argument('--seed', type=int, default=2147500000)
    p.add_argument('--port', type=int, default=34650)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group=args.group, tasks='', seconds=0.0))
    os.environ.update(FORK_ENV)
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                   stub_list=stub if Path(stub).is_file() else '')
    inst = Instance('stpfirst', args.port, cfg, spec)
    out = []
    try:
        rng = np.random.default_rng(args.seed)
        warm, test = sticky_actions(rng, 10), sticky_actions(rng, 5)
        inst.reset(args.seed)
        inst.env.bridge.play(warm, repeat=4, stop_clear=True)
        c = inst.env.bridge.fork(tag='first', lean=True, alarm=300)
        c._send({"cmd": "obs"})
        raws = [read_raw(c)]
        for code in test:
            m, s, b, i = split_code(code)
            c._send({"cmd": "step", "repeat": 4, "move": m, "shoot": s, "bomb": b, "item": i})
            raws.append(read_raw(c))
        c.close()
        for raw in raws:
            hdr = struct.unpack_from('<IIIBBBB', raw, 0)
            room = struct.unpack_from('<iiiiiiiiffff', raw, 16)
            totals = struct.unpack_from('<dddd', raw, 64)
            out.append(dict(len=len(raw), header=hdr, room=room, totals=totals, hex=raw.hex()))
    finally:
        inst.close()
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
