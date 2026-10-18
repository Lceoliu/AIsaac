"""CPU smoke test of B40's stall rule option (2026-10-18; train_tok / eval_tok --stall-progress pickups, TokSamplerConfig
stall_progress, tok_obs EpisodeState stall_pickups, abp_turbo.c abp_row_encode RC_STALL_MODE).

No game, no GPU. Plain script:
  cd rl/bridge/python && python tests/test_stall_progress_smoke.py
Checks, on a synthetic whole-floor sequence of lean payloads (test_obs_v2_smoke's world: no events, no monster hurt;
player 0's coins / keys / bombs / hearts / soul / bone changed along the way, gains and losses):
  (a) encode_row, stall_pickups off: the stall share (player column 30) and the done code as before (stalled at
      t = stall); on: every gain (coins, keys, bombs, health units) resets the progress time, losses do not; the
      share and the done code follow; nothing else in the records differs between the two;
  (b) optional, Linux only: with ISAAC_RL_TEST_TURBO_SO=<path of libabp_turbo.so built from this abp_turbo.c>,
      FastRow (abp_row_encode, fast path for every record after the first) writes the same ROW bytes and keeps the
      same progress time as encode_row, in both modes; an old-layout context (RC_COUNT) is refused by the version check.
"""
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_obs_v2_smoke as V   # noqa: E402  (its world / payload builders, path set-up)

import numpy as np   # noqa: E402

from isaac_bridge import abplus_lean as AL   # noqa: E402
from isaac_bridge import tok_obs as TO   # noqa: E402

FAIL = []
STALL = 5
# per record: coins, keys, bombs, hearts, soul, bone (gains at 4 coins, 7 keys, 10 bombs, 13 soul, 16 bone, 19 hearts;
# losses at 6 bombs, 9 hearts, 12 coins)
SEQ = [(5, 1, 1, 6, 0, 0)] * 4 + [(6, 1, 1, 6, 0, 0)] * 2 + [(6, 1, 0, 6, 0, 0)] + [(6, 2, 0, 6, 0, 0)] * 2 + \
      [(6, 2, 0, 5, 0, 0)] + [(6, 2, 1, 5, 0, 0)] * 2 + [(2, 2, 1, 5, 0, 0)] + [(2, 2, 1, 5, 1, 0)] * 3 + \
      [(2, 2, 1, 5, 1, 1)] * 3 + [(2, 2, 1, 6, 1, 1)] * 6
N = len(SEQ)
GAINS = {4, 7, 10, 13, 16, 19}


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def payload(k, flags):
    p, d, e, room = V.world(0)
    p['coins'], p['keys'], p['bombs'], p['hearts'], p['soul'], p['bone'] = SEQ[k]
    p['black'], p['eternal'] = 0, 0
    p['x'], p['y'] = 300.0 + 2.0 * k, 280.0 - 1.0 * k
    room = room[:7] + (100 + k,) + room[8:]
    out = struct.pack('<IIIBBBB', AL.MAGIC, 100 + 4 * k, 200 + 4 * k, flags, len(p), len(d), 0)
    out += struct.pack('<iiiiiiiiffff', *room)
    out += struct.pack('<dddd', 0.0, 30.0, 30.0, 1.0)
    out += p.tobytes() + d.tobytes() + struct.pack('<H', len(e))
    rec = np.ascontiguousarray(e).view(np.uint8).reshape(len(e), AL.ENTITY.itemsize)
    out += rec[:, :148].tobytes()
    if flags & 2:
        tj, gj = V.terrain_json()
        out += struct.pack('<II', 7, len(tj)) + tj + struct.pack('<I', len(gj)) + gj
    return out


def flags_of(k):
    return 2 if k % 6 == 0 else 0   # (a terrain block now and then, the same one: the fast path takes it)


def state(pickups):
    return TO.EpisodeState(10 ** 6, floor=True, stall=STALL, stall_pickups=pickups)


def reference(pickups):
    dec, st = AL.LeanDecoder(), state(pickups)
    rows = np.zeros(N, TO.ROW)
    prog = []
    for k in range(N):
        TO.encode_row(dec.decode(payload(k, flags_of(k))), st, rows[k:k + 1], k)
        prog.append(st.progress_t)
    rows['first'] = [1] + [0] * (N - 1)
    return rows, prog


def expected(pickups):
    """(stall share per record, done per record, progress time after each record)."""
    prog, share, done, out = 0, [], [], []
    for t in range(N):
        share.append(min((t - prog) / STALL, 1.0) if t > 0 else 0.0)
        if pickups and t in GAINS:
            prog = t
        out.append(prog)
        done.append(3 if t - prog >= STALL else 0)
    return share, done, out


def part_a():
    res = {}
    for pickups in (False, True):
        rows, prog = reference(pickups)
        share, done, prog_x = expected(pickups)
        tag = 'on' if pickups else 'off'
        check(f'(a) {tag}: progress time', prog == prog_x, f'{prog} vs {prog_x}')
        check(f'(a) {tag}: stall share (player column 30)',
              np.allclose(rows['player'][:, 30], np.float32(share)), str(np.round(rows['player'][:, 30], 2)))
        check(f'(a) {tag}: done codes', list(rows['done']) == done, f"{list(rows['done'])}")
        res[pickups] = rows
    same = [nm for nm in TO.ROW.names if nm not in ('player', 'done') and res[0][nm].tobytes() != res[1][nm].tobytes()]
    pl_diff = np.flatnonzero((res[0]['player'] != res[1]['player']).any(0))
    check('(a) off vs on: only player column 30 and done differ', not same and list(pl_diff) == [30],
          f'{same} {list(pl_diff)}')
    return res


def part_b(res):
    path = os.environ.get('ISAAC_RL_TEST_TURBO_SO', '')
    if not path:
        print('skip (b) FastRow: ISAAC_RL_TEST_TURBO_SO not set (Linux, a libabp_turbo.so of this abp_turbo.c)')
        return
    fn = TO.fast_row_function(path)
    check('(b) FastRow library of this layout (version 7, RC_COUNT %d)' % TO.RC_COUNT, fn is not None, path)
    if fn is None:
        return
    for pickups in (False, True):
        dec, st = AL.LeanDecoder(), state(pickups)
        rows = np.zeros(N, TO.ROW)
        fr = TO.FastRow(fn, st, dec, 0, 0, 0)
        prog = []
        for k in range(N):
            pl = payload(k, flags_of(k))
            item = dec.decode(pl) if k == 0 else pl
            fr.encode(item, rows[k:k + 1], rows[k:k + 1].ctypes.data, k)
            prog.append(int(fr.ctx[TO.RC_PROGRESS]))
        ref = res[pickups]
        diff = [nm for nm in TO.ROW.names if ref[nm].tobytes() != rows[nm].tobytes()]
        tag = 'on' if pickups else 'off'
        check(f'(b) {tag}: FastRow == encode_row, every record', not diff and fr.fast == N - 1,
              f'fast {fr.fast} slow {fr.slow} reasons {fr.reasons} diff {diff}')
        check(f'(b) {tag}: the same progress times', prog == expected(pickups)[2], str(prog))
        if pickups:   # a hand-back in the middle (encode_row) keeps the context in step: records 8 .. through the slow path
            dec2, st2 = AL.LeanDecoder(), state(True)
            rows2 = np.zeros(N, TO.ROW)
            fr2 = TO.FastRow(fn, st2, dec2, 0, 0, 0)
            for k in range(N):
                pl = payload(k, flags_of(k))
                item = dec2.decode(pl) if (k == 0 or 8 <= k <= 11) else pl
                fr2.encode(item, rows2[k:k + 1], rows2[k:k + 1].ctypes.data, k)
            diff2 = [nm for nm in TO.ROW.names if ref[nm].tobytes() != rows2[nm].tobytes()]
            check('(b) on: mixed fast / slow records == encode_row', not diff2 and fr2.slow == 5, f'{diff2} {fr2.slow}')


def main():
    res = part_a()
    part_b(res)
    print(f'{len(FAIL)} failed' + (': ' + ', '.join(FAIL) if FAIL else ''))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
