"""(ISAAC_RL_WORKER_LEAN) A fresh process doing a sampler worker's own work on given inputs -- the shared-memory
handshake (tok_sampler.attach), v2 visible frames (tok_obs.write_visible), v3 records through encode_row and, given a
libabp_turbo.so, through FastRow -- then reporting whether it holds torch, its private memory and the digests of the
records it wrote. Run by tests/test_worker_lean.py with ISAAC_RL_WORKER_LEAN 0 / 1 (the inputs pickled by the test:
this process imports no test helper, so nothing here loads torch but the code under test).

   usage: python tests/lean_probe_child.py <inputs.pkl> <libabp_turbo.so or ''> [old] [mp_main]
   old: structured_input imported first (what a worker did before: the comparison's memory and records);
   mp_main: afterwards train_tok imported as spawn imports it in a child (run_name __mp_main__), torch checked again."""
import hashlib
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np   # noqa: E402

from isaac_bridge import abplus_lean as AL   # noqa: E402
from isaac_bridge import tok_obs as TO   # noqa: E402
from isaac_bridge import tok_sampler as TS   # noqa: E402
from isaac_bridge import visible_producer as VP   # noqa: E402


def private_mb():
    try:
        tot = 0
        for line in open('/proc/self/smaps_rollup'):
            if line.startswith(('Private_Clean', 'Private_Dirty')):
                tot += int(line.split()[1])
        return round(tot / 1024.0, 1)
    except OSError:
        return None


def digest(rows):
    """The records' named fields (not the dtype's padding)."""
    h = hashlib.sha256()
    for name in TO.ROW.names:
        h.update(np.ascontiguousarray(rows[name]).tobytes())
    return h.hexdigest()


def episode_state():
    pk, it = TO.stall_flags('events')
    return TO.EpisodeState(10 ** 6, floor=True, run=False, stall=6, items=False, stall_pickups=pk, stall_items=it,
                           stall_first=TO.stall_first('events'), rewind=False)


def main():
    inputs = pickle.load(open(sys.argv[1], 'rb'))
    so = sys.argv[2] if len(sys.argv) > 2 else ''
    flags = set(sys.argv[3:])
    if 'old' in flags:
        import isaac_bridge.structured_input   # noqa: F401  (the worker before: its torch, its memory)
    out = dict(lean=TO.WORKER_LEAN, old='old' in flags)
    from multiprocessing import shared_memory   # 1. the worker's handshake (where torch came in)
    n = 1
    sizes = (TO.ROW.itemsize * 2 * n, 4 * TS.ACTION_F * n, 8 * len(TS.STATS) * n, 8 * 4)
    blocks = [shared_memory.SharedMemory(create=True, size=s) for s in sizes]
    try:
        got = TS.attach([b.name for b in blocks], n, initialize=True)
        for b in got[0]:
            b.close()
        del got
    finally:
        for b in blocks:
            b.close()
            b.unlink()
    out['torch_after_attach'] = 'torch' in sys.modules
    obs = inputs['v2']   # 2. v2 visible frames (write_visible's write_frame; the native encoder when it loads)
    rows = np.zeros(len(obs), TO.ROW)
    dec = AL.LeanDecoder()
    for j, pl in enumerate(obs):
        TO.write_visible(dec.decode(pl), rows[j:j + 1])
    out['v2'] = digest(rows)
    from isaac_bridge import visible_native as VN   # (the native v2 encoder took the frames, or the reference did)
    out['v2_native'] = bool(VN._DEFAULT and VN._DEFAULT[0] is not None)
    out['v2_bytes'] = [int(v) for v in rows['visible_bytes']]
    VP.VISIBLE_V3_MODE = 2   # 3. v3 records: encode_row, then FastRow (the native encoder) when given its library
    pls = inputs['v3']
    dec, st = AL.LeanDecoder(), episode_state()
    ref = np.zeros(len(pls), TO.ROW)
    for k, pl in enumerate(pls):
        TO.encode_row(dec.decode(pl), st, ref[k:k + 1], k)
    out['encode_row'] = digest(ref)
    if so:
        TO.FAST_ROW = True
        TO.FAST_ROW_CHECKING = False
        TO._LIBS = {}
        fn = TO.fast_row_function(so)
        dec, st = AL.LeanDecoder(), episode_state()
        fr = TO.FastRow(fn, st, dec, 0, 0, 0)
        got = np.zeros(len(pls), TO.ROW)
        for k, pl in enumerate(pls):
            s_ = got[k:k + 1]
            fr.encode(dec.decode(pl) if k == 0 else pl, s_, s_.ctypes.data, k)
        out['fast_row'] = digest(got)
        out['fast_row_fast'] = int(fr.fast)
    out['torch'] = 'torch' in sys.modules
    out['private_mb'] = private_mb()
    if 'mp_main' in flags:   # (spawn's import of the parent's main script in a worker: train_tok's top level)
        import runpy
        runpy.run_path(os.path.join(os.path.dirname(HERE), 'train_tok.py'), run_name='__mp_main__')
        out['torch_after_mp_main'] = 'torch' in sys.modules
    print('LEAN_PROBE ' + json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
