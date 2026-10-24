"""Worker frames == VisibleHistory windows, on recorded AB+ raw observations.

For each replay the worker path (VisibleHistory.encode -> abplus_worker.write_frame into a
FRAME_DTYPE record) must reproduce, field by field and bit for bit, the newest row of the window
that VisibleHistory.append returns (the observation the evaluation and J460 paths feed the policy).
Replays of bridge abp-0.2.2 (blocking_count present) also check the room state input of PROFILE
(default combat-v3; combat-v4 has no remaining_time and two room state fields).
usage: python test_abplus_frames.py <replay dir> [max files] [profile]
"""
import glob, gzip, json, sys
import numpy as np
from isaac_bridge.transformer_obs import VisibleHistory
from isaac_bridge import abplus_worker as W

paths = sorted(glob.glob(sys.argv[1] + '/seed-*.jsonl.gz'))[:int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None]
profile = sys.argv[3] if len(sys.argv) > 3 else 'combat-v3'
checked = 0
for path in paths:
    rows = [json.loads(x) for x in gzip.open(path, 'rt')][1:]
    combat = 'blocking_count' in rows[0]['obs']['combat']
    used = profile if combat else 'combat-v2'
    dtype, keys = W.frame_layout(used)
    record = np.zeros((), dtype)
    a, b = (VisibleHistory(64, 256, **W.observation_options(used)) for _ in range(2))
    for row in rows:
        if row['action'] is not None:
            for h in (a, b):
                h.previous_action[:] = (*row['action'], 1)
        window = a.append(row['obs'])
        frame = b.encode(row['obs'])
        W.write_frame(record, frame, 0.0, False, False, 0, 0, 0, b.last_rows)
        last = int(window['history_mask'].sum()) - 1
        for key in keys:
            expected = window[key][last]
            got = record[key]
            if not np.array_equal(np.asarray(got, dtype=expected.dtype), expected):
                raise SystemExit(f'{path}: field {key} differs at step {checked}')
        remaining = max(0.0, 1 - float(record['time']) / 120)
        if 'remaining_time' in window and not np.isclose(window['remaining_time'][last], remaining):
            raise SystemExit(f'{path}: remaining_time differs')
        checked += 1
print(json.dumps({'replays': len(paths), 'frames_checked': checked, 'result': 'identical'}))
