"""B42 glue-layer short-profile summary (runs launched by prof.sh: ISAAC_BLOCK_TIMING=1).

  python b42_prof.py <reports/b42-speed/TAG>

Skips update 1 (graph capture / compile warm-up) and reports medians over updates 2..N of:
collect_s, update_s, wall_s, infer_ms (progress.csv); per-update learner Spatial seconds
(main + historical incl. Writer); actor per-call ms of decode+Spatial / slot+Reader /
heads+sampling / pages (GPU-synced phase timers); compile counters; and the mean GPU
utilisation of the nvidia-smi samples (1 Hz) in the last 60% of the run.
"""
import csv
import json
import statistics as st
import sys

base = sys.argv[1]
rows = list(csv.DictReader(open(base+'/progress.csv')))


def f(r, k):
    try:
        return float(r.get(k, ''))
    except ValueError:
        return float('nan')


timing = []
for line in open(base+'.stdout', encoding='utf-8', errors='replace'):
    if line.startswith('B42 timing '):
        timing.append(json.loads(line[len('B42 timing '):]))
use = rows[1:]
out = dict(updates=len(rows), decisions=int(f(rows[-1], 'decisions')) if rows else 0)
for k in ('collect_s', 'update_s', 'wall_s', 'infer_ms', 'block_compiles', 'block_steady_compiles',
          'block_compiled_calls'):
    v = [f(r, k) for r in use if r.get(k, '') != '']
    if v:
        out['median_'+k if not k.startswith('block_') else 'last_'+k] = (st.median(v) if not k.startswith('block_')
                                                                         else v[-1])
if use:
    dec = f(rows[-1], 'decisions')-f(rows[0], 'decisions')
    wall = sum(f(r, 'wall_s') for r in use)
    out['x_real_time_u2_end'] = dec/(wall*7.5)
t = timing[1:] or timing
if t:
    def med(key, per=None):
        v = [(x.get(key, 0)/(x[per] if per else 1)) for x in t if (not per or x.get(per))]
        return st.median(v) if v else None
    out['learn_main_spatial_s'] = med('learn.main_spatial')
    out['learn_hist_spatial_writer_s'] = med('learn.hist_spatial_writer')
    out['learn_main_decode_s'] = med('learn.main_decode')
    out['learn_reader_heads_s'] = med('learn.reader_heads')
    out['learn_pages_prefix_bank_s'] = med('learn.pages_prefix_bank')
    for name in ('decode_spatial', 'reader_slot', 'heads_sample', 'host_sync', 'pages', 'prepare',
                 'decode_host', 'graph', 'writer'):
        v = med('actor.'+name, 'actor.calls')
        out['actor_ms_'+name] = None if v is None else v*1e3
    out['actor_rows_per_call'] = med('actor.rows', 'actor.calls')
    out['learn_tokens_padded_over_real'] = med('learn.tokens_padded')/max(med('learn.tokens_real'), 1)
try:
    util = [float(l.split(',')[1].strip().rstrip(' %')) for l in open(base+'.gpu') if l.count(',') >= 2]
    tail = util[int(len(util)*.4):]
    out['gpu_util_mean_last60'] = sum(tail)/len(tail) if tail else None
except (OSError, ValueError, IndexError):
    pass
print(json.dumps(out, indent=1))
