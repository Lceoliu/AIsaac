"""Throughput and time breakdown per run (markdown + JSON).

usage: python -X utf8 throughput_table.py label=<run dir or flat prefix> ... [--steady 10:50] [--json out.json]

A run is results/<case>/ (progress.csv, train.log, telemetry.jsonl, memory.jsonl)
or a v2-style flat prefix (<prefix>-progress.csv, <prefix>-train.log, ...).
Steady rows: progress rows [10:50) (the first 10 discarded, 40 measured).
x_real_time over the steady rows is weighted by wall_s (as run_v3_case.py);
median and quartiles are of the per-update values. Updates alternate (one
source VJP / Adam every 2 rollouts), so means are given next to medians.
Learner source parts come from the SOURCE_PIPELINE / SOURCE_TIMELINE lines
(per device: max over the source devices = the slowest worker).
"""
import csv, json, sys
from pathlib import Path
import numpy as np

lo, hi = map(int, (sys.argv[sys.argv.index('--steady') + 1] if '--steady' in sys.argv else '10:50').split(':'))
out_json = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
specs = [a for a in sys.argv[1:] if '=' in a and not a.startswith('--')]


def files(path):
    p = Path(path)
    if p.is_dir(): return {k: p / f'{k}.{e}' for k, e in [('progress', 'csv'), ('train', 'log'), ('telemetry', 'jsonl'), ('memory', 'jsonl'), ('result', 'json')]}
    return {k: Path(f'{path}-{k}.{e}') for k, e in [('progress', 'csv'), ('train', 'log'), ('telemetry', 'jsonl'), ('memory', 'jsonl'), ('result', 'json')]}


def num(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan


def stats(v):
    v = np.asarray([x for x in v if x == x], float)
    if not len(v): return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return dict(median=float(med), q1=float(q1), q3=float(q3), mean=float(v.mean()), n=int(len(v)))


def source_lines(train):
    """update -> dict of source parts (seconds)."""
    out, pending, commit = {}, None, 0.
    if not train.exists(): return out
    for line in open(train, errors='replace'):
        if line.startswith('SOURCE_COMMIT_POLL '):
            commit += json.JSONDecoder().raw_decode(line.split(' ', 1)[1])[0].get('wait_seconds', 0.)
        elif line.startswith('SOURCE_PIPELINE '):
            pending = json.JSONDecoder().raw_decode(line.split(' ', 1)[1])[0]
        elif line.startswith('SOURCE_TIMELINE '):
            t = json.JSONDecoder().raw_decode(line.split(' ', 1)[1])[0]; ws = (pending or {}).get('worker_seconds', {})
            per = {}
            for key, sec in ws.items():
                dev, kind = key.rsplit(':', 1)
                if dev == 'host': continue
                per.setdefault(kind, {})[dev] = sec
            row = {f'src_{k}_max': max(v.values()) for k, v in per.items()}
            row.update({f'host_{k}': v for k, v in t.get('host_seconds', {}).items()})
            row['reader_gpu'] = t.get('reader_stream_seconds', np.nan)
            row['commit_poll_wait'] = commit
            out[int(t['update'])] = row
            pending, commit = None, 0.
    return out


def peak_gpu(tele, indices=None):
    peak = {}
    if not tele.exists(): return peak
    for line in open(tele):
        for r in json.loads(line)['gpus'].strip().splitlines():
            i, used = [x.strip() for x in r.split(',')[:2]]
            if indices is None or int(i) in indices: peak[int(i)] = max(peak.get(int(i), 0), float(used))
    return peak


def container(memory, at):
    """memory.current when progress row `at` was written (or the last row's, if the run stopped earlier); peak."""
    first, peak = {}, 0.
    if not memory.exists(): return None, None
    for line in open(memory):
        m = json.loads(line)
        if 'memory_gib' not in m: continue
        peak = max(peak, m['memory_gib'])
        first.setdefault(m['progress_rows'], m['memory_gib'])   # (one snapshot per new row)
    rows = [r for r in first if r <= at]
    return (first[max(rows)] if rows else None), peak


COLS = ['collect_s', 'update_s', 'wait_learner_s', 'wall_s', 'infer_ms', 'calls_upd', 'handoff_s', 'take_weights_s', 'publish_s',
        'p_poll', 'p_book', 'p_act', 'p_reply', 'p_store', 'idle_share', 'c_learner_cores', 'cpu_collect', 'cpu_update']
report = {}
for spec in specs:
    label, path = spec.split('=', 1)
    f = files(path)
    rows = list(csv.DictReader(open(f['progress'])))
    steady = rows[lo:hi]
    o = dict(path=path, updates=len(rows), steady=len(steady))
    x = [num(r['x_real_time']) for r in steady]; w = [num(r['wall_s']) for r in steady]
    o['x_weighted'] = float(np.nansum(np.multiply(x, w)) / np.nansum(w)) if steady else None
    o['x_real_time'] = stats(x)
    o['game_hours_steady'] = float(num(steady[-1]['game_hours']) - num(rows[lo - 1]['game_hours'])) if len(rows) > lo else None
    for c in COLS + sorted(k for k in rows[0] if k.startswith('u_')):
        if c in rows[0]: o[c] = stats(num(r[c]) for r in steady)
    src = source_lines(f['train'])
    if src:
        keys = sorted({k for r in steady for k in src.get(int(r['update']), {})})
        for k in keys: o[k] = stats(src.get(int(r['update']), {}).get(k, np.nan) for r in steady)
    result = json.loads(f['result'].read_text()) if f['result'].exists() else {}
    idx = (result.get('spec') or {}).get('gpu_indices') or ((result.get('args') or {}).get('gpu_indices'))
    o['exit_status'] = result.get('exit_status')
    o['peak_gpu_mib'] = peak_gpu(f['telemetry'], set(idx) if idx else None)
    o['container_gib_at_40'], o['container_gib_peak'] = container(f['memory'], lo + 40 if hi - lo == 40 else hi)
    report[label] = o

if out_json: Path(out_json).write_text(json.dumps(report, indent=1))
labels = list(report)
def cell(o, k):
    v = o.get(k)
    if isinstance(v, dict) and 'median' in v: return f"{v['median']:.2f} ({v['q1']:.2f}–{v['q3']:.2f}; μ {v['mean']:.2f})"
    if isinstance(v, float): return f'{v:.1f}'
    return '' if v is None else str(v)
keys = []
for o in report.values():
    for k in o:
        if k not in keys and k not in ('path',): keys.append(k)
print('| | ' + ' | '.join(labels) + ' |'); print('|---|' + '---|' * len(labels))
for k in keys: print(f'| {k} | ' + ' | '.join(cell(report[l], k) for l in labels) + ' |')
