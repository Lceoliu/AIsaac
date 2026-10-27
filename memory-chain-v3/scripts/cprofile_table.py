"""SOURCE_CPROFILE lines of runs: named-function cumulative seconds, per-frame host us, top functions.
usage: python -X utf8 cprofile_table.py label=<run dir> ..."""
import json, sys

for spec in [a for a in sys.argv[1:] if '=' in a]:
    label, d = spec.split('=', 1)
    for line in open(f'{d}/train.log', errors='replace'):
        if not line.startswith('SOURCE_CPROFILE '): continue
        p = json.JSONDecoder().raw_decode(line.split(' ', 1)[1])[0]
        f = max(1, p['frames'])
        print(f"### {label} {p['device']} {p['label']}: {p['total_s']:.3f} s, {p['frames']} frames, {1e6 * p['total_s'] / f:.1f} us/frame")
        for k, v in sorted(p['named'].items(), key=lambda kv: -kv[1]['cum_s']):
            print(f"- {k}: {v['cum_s']:.3f} s ({1e6 * v['cum_s'] / f:.1f} us/frame, {v['calls']} calls)")
        print('top cumulative: ' + '; '.join(f'{n} {c} {t:.3f}' for n, c, t in p['top_cumulative'][:25]))
        print('top own: ' + '; '.join(f'{n} {c} {t:.3f}' for n, c, t in p['top_own'][:25]))
        print()
