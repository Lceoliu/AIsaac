"""Check that two sets of visible-v3 golden frames (tests/fixtures/visible_v3, abplus_probe_v3_golden.py's output)
differ only by the schema version (batch 5, the coordinator 2026-11-18: a header-semantics change is proven to touch
nothing else). Per frame named in the old manifest:
  <name>.engine.json  byte for byte the same
  <name>.raw.json     the same bytes but for one '"version": <old>' -> '"version": <new>' (and nothing else)
  <name>.v3.bin       the same length; bytes differ only in header[6] (offset 24..27): the old word -> the new word
  manifest.json       the same but for each frame's sha256 (and 'seconds', a timing, when --allow-seconds)
The old set comes from a directory or from a git revision (read with git show, nothing written).

usage (bridge/python): python tools/golden_v3_diff.py --old-rev <rev> [--new tests/fixtures/visible_v3]
                       --old-version 2026111801 --old-word 3 --new-version 2026111802 --new-word 111802
       or --old <dir> in place of --old-rev. Exit 0 and 'GOLDEN DIFF OK' when every check holds.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

FIX = 'bridge/python/tests/fixtures/visible_v3'


def reader(args):
    if args.old:
        return lambda name: (Path(args.old) / name).read_bytes()
    root = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True, check=True).stdout.strip()

    def read(name):
        return subprocess.run(['git', 'show', f'{args.old_rev}:{FIX}/{name}'], capture_output=True, check=True,
                              cwd=root).stdout
    return read


def lf(b):
    return b.replace(b'\r\n', b'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old')
    ap.add_argument('--old-rev')
    ap.add_argument('--new', default=str(Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'visible_v3'))
    ap.add_argument('--old-version', type=int, required=True)
    ap.add_argument('--old-word', type=float, required=True)
    ap.add_argument('--new-version', type=int, required=True)
    ap.add_argument('--new-word', type=float, required=True)
    ap.add_argument('--allow-seconds', action='store_true')
    args = ap.parse_args()
    if bool(args.old) == bool(args.old_rev):
        ap.error('one of --old / --old-rev')
    old, new = reader(args), (lambda name: (Path(args.new) / name).read_bytes())
    bad, counts = [], dict(engine=0, raw=0, bin=0)
    mo, mn = json.loads(old('manifest.json')), json.loads(new('manifest.json'))
    for fo in mo['frames']:
        name = fo['name']
        if lf(old(f'{name}.engine.json')) != lf(new(f'{name}.engine.json')):
            bad.append(f'{name}.engine.json differs')
        else:
            counts['engine'] += 1
        ro, rn = lf(old(f'{name}.raw.json')), lf(new(f'{name}.raw.json'))
        a, b = b'"version": %d' % args.old_version, b'"version": %d' % args.new_version
        if ro.count(a) != 1 or rn.count(b) != 1 or ro.replace(a, b) != rn:
            bad.append(f'{name}.raw.json: not exactly the one version field changed')
        else:
            counts['raw'] += 1
        bo, bn = old(f'{name}.v3.bin'), new(f'{name}.v3.bin')
        if len(bo) != len(bn):
            bad.append(f'{name}.v3.bin: {len(bo)} -> {len(bn)} bytes')
            continue
        at = np.flatnonzero(np.frombuffer(bo, np.uint8) != np.frombuffer(bn, np.uint8))
        wo, wn = float(np.frombuffer(bo, '<f4')[6]), float(np.frombuffer(bn, '<f4')[6])
        if not set(at.tolist()) <= set(range(24, 28)) or (wo, wn) != (args.old_word, args.new_word):
            bad.append(f'{name}.v3.bin: bytes differ at {at.tolist()[:12]}, header[6] {wo} -> {wn}')
        else:
            counts['bin'] += 1
    names_o, names_n = [f['name'] for f in mo['frames']], [f['name'] for f in mn['frames']]
    if names_o != names_n:
        bad.append(f'manifest frames {names_o} -> {names_n}')
    for m in (mo, mn):
        for f in m['frames']:
            f.pop('sha256')
        if args.allow_seconds:
            m.pop('seconds', None)
    if mo != mn:
        bad.append('manifest.json differs beyond the sha256 values: ' + ', '.join(
            k for k in sorted(set(mo) | set(mn)) if mo.get(k) != mn.get(k)))
    print(json.dumps(dict(frames=len(names_o), same=counts, problems=bad), indent=1))
    print('GOLDEN DIFF OK' if not bad else 'GOLDEN DIFF FAILED')
    sys.exit(0 if not bad else 1)


if __name__ == '__main__':
    main()
