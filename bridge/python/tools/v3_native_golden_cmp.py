"""Compare <name>.native.bin with <name>.v3.bin (the v3 native encoder's stage checks on the golden states).
usage (bridge/python, numpy): python tools/v3_native_golden_cmp.py <golden dir> [<section,...>] [--expected <manifest>]

(2026-11-18, Codex rounds 3 / 4) A pass needs:
  - the run's manifest.json with no collection error, listing exactly the expected frames -- the frozen set of
    tests/fixtures/visible_v3/manifest.json unless --expected names another manifest;
  - for every frame both files, each passing visible_producer.check_frame_header_v3(exact=True) against this tree's
    schema (magic, version word, count ranges, size word = counts = the bytes given);
  - the two frames byte for byte equal once the stall words (P cols 40 / 53 of every body: the worker writes them;
    the fixtures carry 0.5 / 1, the native side 0) are zeroed in both -- bytes, not decoded values (+0.0 / -0.0
    differ); and the body order.
The per-section lines (the sections listed; default all) are for reading only. Exit 0 and PASS, else 1 and FAIL."""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaac_bridge import schema_v3 as V3   # noqa: E402
from isaac_bridge import visible_producer as VP   # noqa: E402

STALL = (V3.col(V3.P_COLS, 'stall_share'), V3.col(V3.P_COLS, 'stall_known'))
FROZEN = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'visible_v3' / 'manifest.json'


def unstalled(b):
    """The frame's bytes with the stall words of every body zeroed (after its header was checked)."""
    f = np.frombuffer(b, '<u4').copy()
    for i in range(int(np.frombuffer(b[8:12], '<f4')[0])):
        for c in STALL:
            f[V3.FIXED + i * V3.PF + c] = 0
    return f.tobytes()


def main(argv):
    args = [a for a in argv[1:]]
    expected_path = FROZEN
    if '--expected' in args:
        i = args.index('--expected')
        expected_path = Path(args[i + 1])
        del args[i:i + 2]
    d = args[0]
    names = [k for k in (args[1] if len(args) > 1 else 'header,canvas,minimap,P,local,E,N,S').split(',') if k]
    why = []
    try:
        expected = [f['name'] for f in json.loads(Path(expected_path).read_text())['frames']]
    except (OSError, ValueError, KeyError) as e:
        expected = []
        why.append(f'no expected frame set ({expected_path}: {type(e).__name__})')
    try:
        run = json.load(open(os.path.join(d, 'manifest.json')))
        frames = [f['name'] for f in run['frames']]
        if run.get('errors'):
            why.append(f"the run's manifest has {len(run['errors'])} collection error(s): {str(run['errors'])[:200]}")
    except (OSError, ValueError, KeyError) as e:
        frames = []
        why.append(f'no run manifest ({type(e).__name__})')
    if sorted(frames) != sorted(expected) or not expected:
        why.append(f'frames {sorted(set(frames) ^ set(expected))[:8]} differ from the expected set '
                   f'({len(frames)} run, {len(expected)} expected)')
    tot = {k: [0, 0] for k in names}
    equal = 0
    for name in frames:
        try:
            nb, vb = (open(os.path.join(d, f'{name}.{x}.bin'), 'rb').read() for x in ('native', 'v3'))
        except OSError:
            why.append(f'{name}: a file is missing')
            continue
        try:
            for b in (nb, vb):
                if len(b) % 4:
                    raise VP.VisibleV3Fatal(f'{len(b)} bytes, not whole floats')
                VP.check_frame_header_v3(np.frombuffer(b, '<f4'), exact=True)
        except VP.VisibleV3Fatal as e:
            why.append(f'{name}: header: {e}')
            continue
        bad = VP.v3_body_order(VP.unpack_frame_v3(nb))
        if bad:
            why.append(f'{name}: native body order {bad}')
        ua, ub = unstalled(nb), unstalled(vb)
        if ua == ub:
            equal += 1
        else:
            i = next((i for i in range(0, min(len(ua), len(ub)), 4) if ua[i:i + 4] != ub[i:i + 4]), min(len(ua), len(ub)))
            why.append(f'{name}: differs (bytes {len(ua)} / {len(ub)}, first at float {i // 4})')
        sa, sb = (VP.unpack_frame_v3(x) for x in (ua, ub))
        for k in names:
            tot[k][0] += 1
            tot[k][1] += sa[k].tobytes() == sb[k].tobytes()
    print({k: f'{v[1]}/{v[0]} equal' for k, v in tot.items()}, f'frames {equal}/{len(frames)} equal')
    for w in why[:20]:
        print(' ', w)
    ok = not why and equal == len(frames) == len(expected) > 0
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
