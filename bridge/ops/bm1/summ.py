"""bm1: one profile's summary (host python3, standard library), written by prof.sh inside the lane lock.

  python3 summ.py <reports dir> <tag> [lane cpus=0-11,24-35] [gpu=1]

Speed (progress.csv): medians of collect_s / update_s / wait_learner_s / x_real_time over u2-u23 and u10+, the first
and last update; u15's CF counters and the normalised CF columns of cf_rate.py (wall to u15 = sum of wall_s,
done/s, lane calls/s, done per game hour). M0 over the steady window (the last sum(wall_s of u2..) seconds of the run):
memory by process kind (PSS / RSS peaks, the container cgroup's peak, the host's lowest MemAvailable), cores used by
kind and by the container, the trainer's threads (cores, top 12), this GPU's utilisation / memory (median, peak),
the other GPUs and the other CPUs (contamination by other tenants)."""
import csv
import os
import statistics as st
import sys


def f(r, c):
    try:
        return float(r.get(c) or 'nan')
    except (TypeError, ValueError):
        return float('nan')


def med(rows, c):
    v = [f(r, c) for r in rows]
    v = [x for x in v if x == x]
    return st.median(v) if v else float('nan')


def cpus(spec):
    out = set()
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out |= set(range(int(a), int(b or a)+1))
    return out


def main():
    R, tag = sys.argv[1], sys.argv[2]
    lane = cpus(sys.argv[3] if len(sys.argv) > 3 else '0-11,24-35')
    gpu = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    out = []
    p = f'{R}/{tag}/progress.csv'
    rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
    out.append(f'{tag}: {len(rows)} updates, {f(rows[-1], "game_hours") if rows else float("nan"):.2f} game hours')
    if not rows:
        print('\n'.join(out)); return
    a, s = rows[1:23], rows[9:23]
    for name, seg in (('u2-u23', a), ('u10+', s)):
        out.append(f'  {name:7s} collect {med(seg, "collect_s"):6.2f}  update {med(seg, "update_s"):6.2f}  '
                   f'wait_learner {med(seg, "wait_learner_s"):6.2f}  x_real {med(seg, "x_real_time"):6.1f}')
    out.append(f'  first update {f(rows[0], "update_s"):.2f} s, last {f(rows[-1], "update_s"):.2f} s')
    if len(rows) >= 15:
        r = rows[14]
        w = f(r, 'wall_s')
        if not w == w or w < 100:
            w = sum(f(x, 'wall_s') for x in rows[:15])
        gh = f(r, 'game_hours')
        out.append(f'  u15: cf_done {f(r, "cf_done"):.0f}  cf_nobank {f(r, "cf_nobank"):.0f}  cf_lane_calls '
                   f'{f(r, "cf_lane_calls"):.0f}  cf_queue {f(r, "cf_queue"):.0f}  cf_dropped {f(r, "cf_dropped"):.0f}'
                   f'  | wall to u15 {w:.0f} s, done/s {f(r, "cf_done")/w:.3f}, lane calls/s '
                   f'{f(r, "cf_lane_calls")/w:.0f}, done per game hour {f(r, "cf_done")/gh:.2f}')
    win = sum(f(x, 'wall_s') for x in rows[1:] if f(x, 'wall_s') == f(x, 'wall_s'))

    def window(path, tcol):   # rows of a sampler csv inside the last `win` seconds
        if not os.path.exists(path):
            return []
        rr = list(csv.DictReader(open(path)))
        if not rr:
            return []
        t1 = max(f(x, tcol) for x in rr)
        return [x for x in rr if f(x, tcol) >= t1-win]
    m = window(f'{R}/{tag}.mem.csv', 't')
    if len(m) > 1:
        kinds = ('trainer', 'workers', 'game', 'other')
        dt = f(m[-1], 't')-f(m[0], 't')
        out.append(f'  M0 memory (steady window {win:.0f} s; PSS = shared pages split): ' + ', '.join(
            f'{k} PSS peak {max(f(x, k+"_pss_mb") for x in m)/1024:.1f} GB (RSS {max(f(x, k+"_rss_mb") for x in m)/1024:.1f}, '
            f'n {max(f(x, k+"_n") for x in m):.0f})' for k in kinds))
        allm = list(csv.DictReader(open(f'{R}/{tag}.mem.csv')))
        out.append(f'  container cgroup: peak {max(f(x, "cg_peak_mb") for x in allm)/1024:.1f} GB, steady median '
                   f'{med(m, "cg_current_mb")/1024:.1f} GB (anon {med(m, "cg_anon_mb")/1024:.1f}, file '
                   f'{med(m, "cg_file_mb")/1024:.1f}); host MemAvailable min {min(f(x, "host_avail_mb") for x in allm)/1024:.1f} GB')
        out.append('  cores used (steady): ' + ', '.join(
            f'{k} {(f(m[-1], k+"_ticks")-f(m[0], k+"_ticks"))/100/dt:.2f}' for k in kinds)
            + f', container {(f(m[-1], "cg_cpu_usec")-f(m[0], "cg_cpu_usec"))/1e6/dt:.2f}')
    th = window(f'{R}/{tag}.threads.csv', 't')
    if th:
        first, last = {}, {}
        for x in th:
            if x['kind'] == 'thread':
                k = x['name'].rsplit(':', 1)[0]
                key = x['name']
                first.setdefault(key, (f(x, 't'), f(x, 'ticks')))
                last[key] = (f(x, 't'), f(x, 'ticks'))
        per = {}
        for key, (t1, k1) in last.items():
            t0, k0 = first[key]
            if t1 > t0:
                name = key.rsplit(':', 1)[0]
                per[name] = per.get(name, 0)+(k1-k0)/100/(t1-t0)
        top = sorted(per.items(), key=lambda kv: -kv[1])[:12]
        out.append('  trainer threads (cores, steady): ' + ', '.join(f'{n} {c:.2f}' for n, c in top))
    g = []
    if os.path.exists(f'{R}/{tag}.gpu'):
        for line in open(f'{R}/{tag}.gpu'):
            parts = [x.strip() for x in line.split(',')]
            try:
                g.append((float(parts[1].split()[0]), float(parts[2].split()[0])))
            except (IndexError, ValueError):
                pass
    if g:
        k = max(1, int(win))
        gs = g[-k:]
        out.append(f'  GPU {gpu} (1 s samples, steady): util median {st.median(x[0] for x in gs):.0f}% peak '
                   f'{max(x[0] for x in gs):.0f}%, memory median {st.median(x[1] for x in gs)/1024:.1f} GB peak '
                   f'{max(x[1] for x in g)/1024:.1f} GB')
    if os.path.exists(f'{R}/{tag}.allgpu'):
        other = {}
        lines = open(f'{R}/{tag}.allgpu').read().splitlines()[-8*max(1, int(win/5)):]   # (8 GPUs per sample)
        for line in lines:
            parts = [x.strip() for x in line.split(',')]
            try:
                other.setdefault(int(parts[1]), []).append(float(parts[2].split()[0]))
            except (IndexError, ValueError):
                pass
        out.append('  all GPUs util median (steady): ' + ', '.join(f'{i}:{st.median(v):.0f}%' for i, v in sorted(other.items())))
    if os.path.exists(f'{R}/{tag}.mpstat'):
        busy = {}
        lines = open(f'{R}/{tag}.mpstat').read().splitlines()
        hdr = None
        for line in lines:
            w_ = line.split()
            if '%idle' in w_:
                hdr = w_
                continue
            if hdr and len(w_) == len(hdr) and w_[hdr.index('CPU')].isdigit():
                busy.setdefault(int(w_[hdr.index('CPU')]), []).append(100-float(w_[hdr.index('%idle')]))
        n_keep = max(1, int(win/5))
        lb = [st.mean(v[-n_keep:]) for c, v in busy.items() if c in lane]
        ob = [st.mean(v[-n_keep:]) for c, v in busy.items() if c not in lane]
        if lb:
            out.append(f'  mpstat (steady): this lane {len(lb)} CPUs mean busy {st.mean(lb):.0f}% (sum {sum(lb)/100:.1f} cores); '
                       f'other CPUs {len(ob)} mean busy {st.mean(ob) if ob else 0:.0f}% (sum {sum(ob)/100:.1f} cores)')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
