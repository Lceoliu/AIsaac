"""Cross-version check (2026-10-04): the same seeds and actions through two copies of the bridge and sampler code (e.g.
the training's sandbox and this one), lean clones stepped one decision at a time. Per seed it writes to <out>/trace.json
the SHA-256 of every ROW record a sampler worker of that copy writes for the observations (tok_obs; --fast: through
FastRow), the hidden-state digest (abplus_goexplore.DIGEST_LUA) every --check decisions and at the end, and a
normalised payload hash (logic frame count relative to the first observation, terrain JSON with sorted keys). The
payloads themselves are not comparable across instance launches: the logic frame count, entity indices (the player's
id included) and the terrain JSON's key order (Lua's string hash seed) depend on the instance's start, not on the
code. Run once per copy (each from its own python dir: its own abp_bridge.lua and libabp_turbo.so), then
`--compare A B`.

  --line: step with the step line when the bridge offers it (the new path), else the JSON step command.

usage (from a bridge copy's python dir, PYTHONPATH=.):
  python abplus_probe_cross_version.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:16 --out <dir> [--line --fast]
  python abplus_probe_cross_version.py --compare <dirA> <dirB>
"""
import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np


def key(raw, lf0):
    """SHA-256 of a payload made comparable across instance launches: its logic frame count relative to the clone's
    first observation (the bridge counts MC_POST_UPDATE calls since the instance started, which depends on how long it
    ran before its first connection) and a terrain block's two JSON strings with sorted keys (json.encode writes a Lua
    table's keys in its hash order, and Lua's string hash seed differs per process launch). Within one instance (fork
    clones) both are the same anyway."""
    lf = struct.unpack_from('<I', raw, 4)[0]
    flags, n_players, n_doors, n_lasers = raw[12], raw[13], raw[14], raw[15]
    head = raw[:4] + struct.pack('<I', lf - lf0) + raw[8:]
    if not flags & 2:
        return hashlib.sha256(head).hexdigest()
    from isaac_bridge.abplus_lean import DOOR, ENTITY, PLAYER   # (2026-10-08: the record sizes of the decoder)
    off = 96 + PLAYER.itemsize * n_players + DOOR.itemsize * n_doors
    off += 2 + ENTITY.itemsize * struct.unpack_from('<H', raw, off)[0]
    for _ in range(n_lasers):
        off += 65 + 16 * struct.unpack_from('<q', raw, off + 57)[0]
    version, n = struct.unpack_from('<II', raw, off)
    terrain = json.loads(raw[off + 8:off + 8 + n])
    n2 = struct.unpack_from('<I', raw, off + 8 + n)[0]
    grid = json.loads(raw[off + 12 + n:off + 12 + n + n2])
    canon = json.dumps([version, terrain, grid], sort_keys=True).encode()
    return hashlib.sha256(head[:off] + canon + raw[off + 12 + n + n2:]).hexdigest()


class RowStream:
    """The ROW records a sampler worker of this copy writes for the payloads (two slots used alternately, as the
    worker); --fast: through tok_obs.FastRow when this copy has it. add(raw) -> SHA-256 of the slot after the record."""

    def __init__(self, fast):
        import numpy as np
        from goexplore_abplus import default_preload
        from isaac_bridge import tok_obs
        from isaac_bridge.abplus_lean import LeanDecoder
        self.tok_obs, self.dec = tok_obs, LeanDecoder()
        self.st = tok_obs.EpisodeState(10 ** 6)
        self.rows = np.zeros(2, tok_obs.ROW)
        self.t = 0
        self.enc = None
        if fast and hasattr(tok_obs, 'FastRow'):
            fn = tok_obs.fast_row_function(default_preload())
            if fn is not None:
                self.enc = tok_obs.FastRow(fn, self.st, self.dec, 1, 2, 0)

    def add(self, raw):
        t, row = self.t, self.rows[self.t & 1:(self.t & 1) + 1]
        if self.enc is not None:
            self.enc.encode(raw if t else self.dec.decode(raw), row, row.ctypes.data, t)
        else:
            self.tok_obs.encode_row(self.dec.decode(raw), self.st, row, t)
            row['episode'], row['seed'], row['group'], row['first'] = 1, 2, 0, t == 0
        self.t += 1
        return hashlib.sha256(row.tobytes()).hexdigest()

    def fast_count(self):
        return self.enc.fast if self.enc is not None else 0


def compare(a, b):
    ta = json.loads((Path(a) / 'trace.json').read_text())
    tb = json.loads((Path(b) / 'trace.json').read_text())
    seeds = sorted(set(ta) & set(tb))
    out = dict(seeds=len(seeds), records=sum(len(ta[s]['rows']) for s in seeds),
               rows_identical=sum(ta[s]['rows'] == tb[s]['rows'] for s in seeds),
               digests_identical=sum(ta[s]['digests'] == tb[s]['digests'] for s in seeds),
               payloads_identical=sum(ta[s]['payloads'] == tb[s]['payloads'] for s in seeds),
               fast_rows=(sum(ta[s].get('fast_rows', 0) for s in seeds), sum(tb[s].get('fast_rows', 0) for s in seeds)))
    for s in seeds:
        ra, rb = ta[s]['rows'], tb[s]['rows']
        if ra != rb:
            out['first_row_difference'] = dict(seed=s, index=next((i for i in range(min(len(ra), len(rb)))
                                                                   if ra[i] != rb[i]), None), lengths=(len(ra), len(rb)))
            break
    print('COMPARE', json.dumps(out))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        compare(sys.argv[2], sys.argv[3])
        return
    from abplus_probe_fork import digest, split_code, sticky_actions
    from abplus_probe_native_obs import read_raw
    from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
    from isaac_bridge.abplus import FORK_ENV
    from isaac_bridge.abplus_goexplore import GxConfig, Instance
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:16')
    p.add_argument('--warm', type=int, default=10)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--check', type=int, default=25)
    p.add_argument('--fpd', type=int, default=4)
    p.add_argument('--line', action='store_true')
    p.add_argument('--fast', action='store_true', help='the ROW records through tok_obs.FastRow when available')
    p.add_argument('--port', type=int, default=34600)
    p.add_argument('--name', default='stpcross')
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
    trace = {}
    try:
        for seed in parse_seeds(args.seeds):
            rng = np.random.default_rng(seed)
            warm, test = sticky_actions(rng, args.warm), sticky_actions(rng, args.steps)
            inst.reset(seed)
            _, _, stop = inst.env.bridge.play(warm, repeat=args.fpd, stop_clear=True)
            if stop != 'done':
                continue
            c = inst.env.bridge.fork(tag='cross', lean=True, alarm=900)
            line = args.line and c.hello.get('step_line')
            c._send({"cmd": "obs"})
            raw = read_raw(c)
            lf0 = struct.unpack_from('<I', raw, 4)[0]
            rows = RowStream(args.fast)
            payloads, digests, row_keys = [key(raw, lf0)], [], [rows.add(raw)]
            for t, code in enumerate(test):
                m, s, b, i = split_code(code)
                if line:
                    c._sock.sendall(b"S %d %d %d %d %d\n" % (args.fpd, m, s, b, i))
                else:
                    c._send({"cmd": "step", "repeat": args.fpd, "move": m, "shoot": s, "bomb": b, "item": i})
                raw = read_raw(c)
                payloads.append(key(raw, lf0))
                row_keys.append(rows.add(raw))
                if (t + 1) % args.check == 0:
                    digests.append(digest(c)[0])
                players_dead = raw[96 + 33 * 8:96 + 34 * 8] != b'\0' * 8
                if players_dead:
                    break
            digests.append(digest(c)[0])
            c.close()
            trace[str(seed)] = dict(payloads=payloads, rows=row_keys, digests=digests, line=bool(line),
                                    fast_rows=rows.fast_count())
            print(seed, len(payloads), flush=True)
    finally:
        inst.close()
        (out / 'trace.json').write_text(json.dumps(trace))
        print('DONE', len(trace))


if __name__ == '__main__':
    main()
