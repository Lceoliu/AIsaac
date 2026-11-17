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


def per_update(base):
    """B42 spike localisation: one line per update (progress.csv b42_* columns, ISAAC_BLOCK_TIMING=1),
    with the mean GPU utilisation of the 1 Hz nvidia-smi samples inside the update window."""
    import time as time_
    samples = []
    try:
        for line in open(base+'.gpu'):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 3:
                continue
            t = time_.mktime(time_.strptime(parts[0].split('.')[0], '%Y/%m/%d %H:%M:%S'))
            samples.append((t, float(parts[1].rstrip(' %'))))
    except OSError:
        pass
    prev = {}
    out = []
    for r in rows:
        def g(k):
            return f(r, 'b42_'+k)
        t0, t1 = g('update_t0'), g('update_t1')
        win = [u for t, u in samples if t0 <= t <= t1]
        cum = {k: f(r, k) for k in ('searches', 'snaps', 'archive', 'block_teacher_forward_calls',
                                     'block_merged_lane_rows', 'block_graph_captures')}
        d = {k: cum[k]-prev.get(k, 0.0) for k in cum}
        prev = cum
        fwd = sum(f(r, 'u_'+k) for k in ('recompute',)) + g('learn_fwd_part_wall_s')
        out.append(dict(update=int(f(r, 'update')), update_s=round(f(r, 'update_s'), 1),
                        collect_s=round(f(r, 'collect_s'), 1), infer_ms=round(f(r, 'infer_ms'), 1),
                        u_recompute=round(f(r, 'u_recompute'), 1), u_ppo=round(f(r, 'u_ppo'), 1),
                        fwd_parts_wall=round(g('learn_fwd_part_wall_s'), 1), fwd_parts_gpu=round(g('learn_fwd_part_gpu_s'), 1),
                        bwd_wall=round(g('learn_bwd_wall_s'), 1), bwd_gpu=round(g('learn_bwd_gpu_s'), 1),
                        bwd_calls=int(g('learn_bwd_calls')),
                        opt=round(g('learn_opt_wall_s')+g('learn_clip_wall_s')+g('learn_zero_wall_s'), 2),
                        opt_gpu=round(g('learn_opt_gpu_s')+g('learn_clip_gpu_s'), 2),
                        learn_adm_wait=round(g('learn_admission_wait_s'), 2), learn_yield_wait=round(g('learn_yield_wait_s'), 2),
                        yields=int(g('learn_yields_given')), actor_adm_wait=round(g('actor_admission_wait_s'), 2),
                        captures=int(g('actor_captures_step')+g('actor_captures_other')),
                        capture_s=round(g('actor_capture_step_s'), 2),
                        eager_calls=int(g('actor_eager_calls')), eager_merged=int(g('actor_eager_merged_calls')),
                        eager_s=round(g('actor_eager_s'), 1), eager_overlap_s=round(g('actor_eager_overlap_s'), 1),
                        graph_calls=int(g('actor_graph_calls')), graph_s=round(g('actor_graph_s'), 1),
                        graph_overlap_s=round(g('actor_graph_overlap_s'), 1),
                        main_decode_fresh=int(g('learn_main_decode_rows_fresh')),
                        main_decode_cached=int(g('learn_main_decode_rows_cached')),
                        hist_decode_fresh=int(g('learn_hist_decode_rows_fresh')),
                        hist_decode_cached=int(g('learn_hist_decode_rows_cached')),
                        nivcsw=int(g('learn_nivcsw')), nvcsw=int(g('learn_nvcsw')), learn_cpu_s=round(g('learn_cpu_s'), 1),
                        gpu_util_update=round(sum(win)/len(win), 1) if win else None,
                        cf_searches=d['searches'], lane_forward_calls=d['block_teacher_forward_calls'],
                        merged_lane_rows=d['block_merged_lane_rows'], snaps=d['snaps'], archive=d['archive']))
    return out


if len(sys.argv) > 2 and sys.argv[2] == '--per-update':
    for line in per_update(base):
        print(json.dumps(line))
