"""B45 visible-frame probe: real lean payload streams for the C frame encoder.

Plays seeds like abplus_probe_fast_row.py (room mode: sticky random actions; --floor: legs through doors, room changes,
map and terrain blocks; --items / --prices / --run as there) and, per seed, keeps every payload of the clone's stream in
order. Saved to <out>/payloads_<mode>_<seed>.npz (blob, lengths: a decoder replays them in order). Each payload also
goes through the reference (LeanDecoder + structured_input.observation_frame) and, when a worker library is given
(--lib, isaac_bridge.visible_native), through the C encoder: the two frames must be equal byte for byte. Reported per
seed: records, equal, the reference's and the C encoder's seconds per record, the largest frame and entity count.

usage (from the bridge's python dir, under xvfb-run like the trainer's games):
  python abplus_probe_visible_frame.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:8 --out <dir> [--floor --items --prices] [--lib <libabp_worker.so>]
"""
import json
import os
import time
from pathlib import Path

import numpy as np

import abplus_probe_fast_row as fr
from isaac_bridge.abplus_lean import LeanDecoder
from isaac_bridge.structured_input import observation_frame


class Recorder:
    """Pair's interface for abplus_probe_fast_row's room_seed / floor_seed: record(payload) -> (LeanObs, done 0)."""
    lib = None
    out = None
    mode = 'room'

    def __init__(self, fn, limit, floor, stall, run=False, items=False):
        self.dec = LeanDecoder()
        self.native = Recorder.lib.encoder() if Recorder.lib is not None else None
        self.payloads = []
        self.t = 0
        self.mismatch = None
        self.equal = 0
        self.ref_s = self.c_s = 0.0
        self.max_bytes = 0
        self.max_ents = 0
        self.counts, self.stages, self.kept = {}, set(), []
        self.last_row = np.zeros(1, [('inv', 'i2', (8,)), ('pitem', 'i2', (6,)), ('pinv', 'f4', (4,))])
        self.enc = type('E', (), dict(fast=0, slow=0, reasons={}))()
        Recorder.last = self

    def record(self, payload):
        self.payloads.append(bytes(payload))
        t0 = time.perf_counter()
        o = self.dec.decode(payload)
        ref = observation_frame(o) if o.visible is not None else None
        t1 = time.perf_counter()
        self.ref_s += t1-t0
        self.stages.add(int(o.room[5]))
        self.max_ents = max(self.max_ents, len(o.entities))
        if ref is not None:
            self.max_bytes = max(self.max_bytes, len(ref))
        if self.native is not None:
            t2 = time.perf_counter()
            got = self.native.encode(payload)
            self.c_s += time.perf_counter()-t2
            if got == ref:
                self.equal += 1
            elif self.mismatch is None:
                self.mismatch = dict(t=self.t, ref=None if ref is None else len(ref), got=None if got is None else len(got),
                                     first_diff=_first_diff(ref, got))
        self.t += 1
        return o, 0


def _first_diff(a, b):
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    x, y = np.frombuffer(a[:n], np.uint8), np.frombuffer(b[:n], np.uint8)
    d = np.flatnonzero(x != y)
    return int(d[0]) if len(d) else n


def save(rec, seed):
    if Recorder.out is None or not rec.payloads:
        return
    lens = np.array([len(p) for p in rec.payloads], np.int64)
    np.savez_compressed(Path(Recorder.out) / f'payloads_{Recorder.mode}_{seed}.npz',
                        blob=np.frombuffer(b''.join(rec.payloads), np.uint8), lengths=lens)


def main():
    import argparse
    p = argparse.ArgumentParser()
    for a, kw in (('--groups-file', dict(default='')), ('--group', dict(default='normal')), ('--tasks', dict(default='')),
                  ('--seconds', dict(type=float, default=0.0)), ('--seeds', dict(default='range:2147500000:8')),
                  ('--floors', dict(type=int, default=1)), ('--item-prob', dict(type=float, default=0.05)),
                  ('--legs', dict(type=int, default=6)), ('--warm', dict(type=int, default=10)),
                  ('--steps', dict(type=int, default=300)), ('--after-clear', dict(type=int, default=30)),
                  ('--fpd', dict(type=int, default=4)), ('--port', dict(type=int, default=34600)),
                  ('--name', dict(default='stpvis')), ('--lib', dict(default='')), ('--out', dict(required=True))):
        p.add_argument(a, **kw)
    for a in ('--floor', '--items', '--prices', '--run'):
        p.add_argument(a, action='store_true')
    args = p.parse_args()
    args.dump = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Recorder.out, Recorder.mode = str(out), 'floor' if args.floor else 'room'
    if args.lib:
        from isaac_bridge import visible_native
        path = visible_native.build() if args.lib == 'auto' else args.lib
        Recorder.lib = visible_native.load(path) if path else None
        if Recorder.lib is None:
            raise SystemExit(f'no visible encoder library ({args.lib})')
        print('library', path, flush=True)
    fr.Pair = Recorder
    _save = fr.dump_rows
    fr.dump_rows = lambda a, seed, pair: save(pair, seed)
    spec = fr.load_spec(args)
    os.environ.update(fr.FORK_ENV)
    if args.items:
        os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
    stub = str(Path(fr.default_preload()).parent / 'stub_render_g.txt') if fr.default_preload() else ''
    cfg = fr.GxConfig(bridge_lua=fr.default_bridge_lua(), preload=fr.default_preload(), al_stopped=True, nice=0,
                      stub_list=stub if stub and Path(stub).is_file() else '')
    inst = fr.Instance(args.name, args.port, cfg, spec)
    rows = []
    t0 = time.perf_counter()
    try:
        for seed in fr.parse_seeds(args.seeds):
            try:
                r = (fr.floor_seed if args.floor else fr.room_seed)(inst, seed, args, args.fpd, None)
            except Exception as exc:   # a seed that fails to play is reported, the probe goes on
                r = dict(seed=seed, skipped=f'{type(exc).__name__}: {exc}')
            rec = getattr(Recorder, 'last', None)
            if rec is not None and 'skipped' not in r:
                r.update(equal=rec.equal, ref_ms=round(1e3*rec.ref_s/max(rec.t, 1), 3),
                         c_ms=round(1e3*rec.c_s/max(rec.t, 1), 3), max_frame_bytes=rec.max_bytes,
                         max_entities=rec.max_ents)
            Recorder.last = None
            rows.append(r)
            print(json.dumps(r, default=str), flush=True)
    finally:
        inst.close()
        fr.dump_rows = _save
    (out / f'rows_{Recorder.mode}.json').write_text(json.dumps(rows, indent=1, default=str))
    print('SECONDS', round(time.perf_counter()-t0, 1))


if __name__ == '__main__':
    main()
