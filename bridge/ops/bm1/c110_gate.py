"""C110 (2026-11-15): a smoke run's checks, read-only. c110_gate.py <run dir> <name> [key=value ...] [--driver on|any]
<run dir>/<name>-s0/{progress.csv,config.json}, <run dir>/<name>-s0.stdout* / .stderr*, <run dir>/guard.log.
Prints: the guard's end; updates and game hours; the error counters' sums (errors, stalls, cf_errors, v2_errors,
broken_episodes, w_overrun, ctr_oom_kills); the first rows' kl / clip_frac / stale_share / grad_norm / entropy; x_real
(median of the rows after the first two), collect / update seconds; the peaks of ctr_mem_gb, gpu_reserved_gb (learner)
and gpu_reserved_gb_actor; the pickup-force p and the forced share; the startup lines (B42 / B45 / B52 / actor cache);
Tracebacks in stderr; config.json keys given as key=value (compared as strings of the JSON value).
Exit 0 when: the guard reached its hours, the counters are 0, no Traceback, every key=value holds, the learner's
reserved peak < 23 GB, and (--driver on) the B52 line says on."""
import csv
import glob
import json
import statistics as st
import sys

d, name = sys.argv[1], sys.argv[2]
want = [a for a in sys.argv[3:] if '=' in a and not a.startswith('--')]
drv = 'on' if '--driver' in sys.argv and sys.argv[sys.argv.index('--driver') + 1] == 'on' else 'any'
s0 = f'{d}/{name}-s0'
ok = True


def bad(msg):
    global ok
    ok = False
    print('  FAIL', msg)


gl = open(f'{d}/guard.log').read().splitlines()
print('guard:', gl[-2] if len(gl) > 1 else gl)
if not any('game hours reached' in x for x in gl):
    bad('the guard did not reach its hours')
R = list(csv.DictReader(open(f'{s0}/progress.csv')))
print(f'rows {len(R)}, update {R[-1]["update"]}, game hours {float(R[-1]["game_hours"]):.2f}')


def col(k):
    return [float(r[k]) for r in R if r.get(k, '') not in ('', None)]


for k in ('errors', 'stalls', 'cf_errors', 'v2_errors', 'broken_episodes', 'w_overrun'):
    v = col(k)
    print(f'  {k}: sum {sum(v):g}' if v else f'  {k}: (no column / empty)')
    if v and sum(v) != 0:
        bad(f'{k} {sum(v):g}')
v = col('ctr_oom_kills')
print(f'  ctr_oom_kills: max {max(v):g}' if v else '  ctr_oom_kills: (none)')
if v and max(v) != 0:
    bad('ctr_oom_kills')
for k in ('kl', 'clip_frac', 'stale_share', 'grad_norm', 'grad_clip_frac', 'entropy', 'ev', 'pickup_force_p'):
    v = col(k)
    print(f'  {k}: ' + (' '.join(f'{x:.4g}' for x in v[:10]) if v else '(none)'))
xr = col('x_real_time')
print(f'  x_real: median after u2 {st.median(xr[2:]) if len(xr) > 2 else float("nan"):.1f} (all: '
      + ' '.join(f'{x:.0f}' for x in xr) + ')')
for k in ('collect_s', 'update_s'):
    v = col(k)
    print(f'  {k}: median {st.median(v[2:]) if len(v) > 2 else float("nan"):.2f}')
peaks = {}
for k in ('ctr_mem_gb', 'gpu_reserved_gb', 'gpu_reserved_gb_actor', 'b42_gpu_peak_allocated_gb'):
    v = col(k)
    peaks[k] = max(v) if v else None
    print(f'  peak {k}: {max(v):.2f}' if v else f'  peak {k}: (none)')
if peaks['gpu_reserved_gb'] is not None and peaks['gpu_reserved_gb'] >= 23:
    bad(f'learner reserved peak {peaks["gpu_reserved_gb"]:.1f} GB')
fe, fs = col('pickup_forced_eps'), col('episodes') or col('eps')
if fe:
    print(f'  forced episodes {sum(fe):g}' + (f' of {sum(fs):g}' if fs else ''))
out = ''.join(open(p, errors='replace').read() for p in sorted(glob.glob(f'{d}/{name}-s0.stdout*')))
err = ''.join(open(p, errors='replace').read() for p in sorted(glob.glob(f'{d}/{name}-s0.stderr*')))
for key in ('B42 compiled Spatial warm-up', 'B45 visible frame encoder', 'B52 cpp actor driver', 'actor cache',
            'parameters'):
    ls_ = [x for x in out.splitlines() if key in x]
    print(f'  [{key}] ' + (ls_[0][:220] if ls_ else '(absent)') + (f' (x{len(ls_)})' if len(ls_) > 1 else ''))
if drv == 'on' and 'B52 cpp actor driver: on' not in out:
    bad('B52 driver not on')
tb = err.count('Traceback') + out.count('Traceback')
print(f'  Tracebacks: {tb}')
if tb:
    bad('Traceback')
cfg = json.load(open(f'{s0}/config.json'))
for kv in want:
    k, v = kv.split('=', 1)
    got = json.dumps(cfg.get(k)) if k in cfg else '(missing)'
    good = got == v or got.strip('"') == v
    print(f'  config {k} = {got}' + ('' if good else f'  (want {v})'))
    if not good:
        bad(f'config {k}')
print('GATE', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
