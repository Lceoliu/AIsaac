"""visible-v3 on the lean payload (package C, 2026-11-17): with ISAAC_RL_VISIBLE_V3=1 the bridge's lean payload carries
visible_v3_raw()'s block (flag 32, schema 'isaac-visible-raw-v3'); the worker's encode_row writes the v3 frame and,
after its progress update, the stall columns. Here, on AB+: a lean clone stepped with step lines (the sampler's path)
for --steps decisions, every payload decoded and encoded by encode_row (whole floors, stall rule --stall); per record:
the payload's block is v3, the ROW's frame equals observation_frame_v3(block) with the stall values encode_row's own
EpisodeState gives (recomputed here from its progress time), visible_schema = schema_v3.VERSION; and the Lua block's
game-side time per decision (the clone's CPU before / after, against the same steps with the switch off).

usage (bridge python dir, PYTHONPATH=.): python abplus_probe_v3_lean.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ['ISAAC_RL_VISIBLE_V3'] = '1'   # (before the worker modules read it; the game inherits it)

from abplus_probe_native_obs import read_raw   # noqa: E402
from goexplore_abplus import default_bridge_lua, default_preload, load_spec   # noqa: E402
from isaac_bridge import schema_v3 as V3, visible_producer as VP   # noqa: E402
from isaac_bridge.abplus import FORK_ENV   # noqa: E402
from isaac_bridge.abplus_goexplore import GxConfig, Instance   # noqa: E402
from isaac_bridge.abplus_lean import LeanDecoder   # noqa: E402
from isaac_bridge.tok_obs import ROW, EpisodeState, encode_row   # noqa: E402

MOVES = [(3, 0), (3, 2), (5, 0), (5, 3), (7, 1), (1, 4), (0, 0), (2, 2)]


def game_cpu(c):
    return float(c.lua("return tostring(os.clock())"))


def run(inst, seed, steps, stall, v3, out):
    b = inst.env.bridge
    b.reset_mode, b.character = 'floor', 0
    inst.reset(seed)
    c = b.fork(tag='v3lean', lean=True, alarm=900)
    stats = dict(seed=seed, v3=v3, records=0, v3_blocks=0, equal=0, schema_ok=0, mismatch=[])
    try:
        c._send({"cmd": "visible_v3", "enabled": v3})
        ack = c._recv()
        assert ack.get('type') == 'ok' and ack.get('enabled') == v3, ack
        dec = LeanDecoder()
        st = EpisodeState(limit=100000, floor=True, stall=stall)
        row = np.zeros(1, ROW)
        c._send({"cmd": "obs"})
        payload = read_raw(c)
        cpu0 = game_cpu(c)
        t0 = time.perf_counter()
        for t in range(steps):
            o = dec.decode(payload)
            done = encode_row(o, st, row, t)
            stats['records'] += 1
            if v3:
                blk = o.visible
                if VP.is_raw_v3(blk):
                    stats['v3_blocks'] += 1
                    known = st.floor and st.stall > 0
                    share = min((t - st.progress_t) / st.stall, 1.0) if known else 0.0
                    want = VP.observation_frame_v3(blk, stall_share=share, stall_known=1.0 if known else 0.0)
                    got = bytes(row['visible_frame'][0, :int(row['visible_bytes'][0])])
                    stats['equal'] += got == want
                    stats['schema_ok'] += int(row['visible_schema'][0]) == V3.VERSION
                    if got != want and len(stats['mismatch']) < 3:
                        stats['mismatch'].append(t)
            if done:
                break
            m, s = MOVES[t % len(MOVES)]
            c._sock.sendall(b'S 4 %d %d 0 0\n' % (m, s))
            payload = read_raw(c)
        stats['game_cpu_ms_per_step'] = round(1000 * (game_cpu(c) - cpu0) / max(1, stats['records']), 4)
        stats['wall_ms_per_step'] = round(1000 * (time.perf_counter() - t0) / max(1, stats['records']), 3)
    finally:
        c.close()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--seeds', default='2147900000:3')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--stall', type=int, default=120)
    ap.add_argument('--port', type=int, default=44530)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    inst = Instance('v3lean', args.port, gx, spec)
    first, count = (int(v) for v in args.seeds.split(':'))
    res = []
    try:
        for seed in range(first, first + count):
            for v3 in (True, False):
                r = run(inst, seed, args.steps, args.stall, v3, out)
                res.append(r)
                print(json.dumps(r), flush=True)
    finally:
        inst.kill()
    on = [r for r in res if r['v3']]
    ok = all(r['records'] == r['v3_blocks'] == r['equal'] == r['schema_ok'] for r in on)
    summary = dict(ok=ok, records=sum(r['records'] for r in on), equal=sum(r['equal'] for r in on),
                   game_cpu_ms_v3=round(np.mean([r['game_cpu_ms_per_step'] for r in on]), 4),
                   game_cpu_ms_v2=round(np.mean([r['game_cpu_ms_per_step'] for r in res if not r['v3']]), 4),
                   runs=res)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True))
    print('SUMMARY', json.dumps({k: v for k, v in summary.items() if k != 'runs'}), flush=True)


if __name__ == '__main__':
    main()
