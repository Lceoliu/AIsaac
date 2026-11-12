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

# x86-64 syscall numbers the ledger names (/proc/<pid>/task/<tid>/syscall: the call a task sits in)
SYSCALL = {'0': 'read', '1': 'write', '7': 'poll', '9': 'mmap', '11': 'munmap', '16': 'ioctl', '23': 'select',
           '24': 'sched_yield', '35': 'nanosleep', '44': 'sendto', '45': 'recvfrom', '46': 'sendmsg', '47': 'recvmsg',
           '56': 'clone', '57': 'fork', '59': 'execve', '61': 'wait4', '202': 'futex', '230': 'clock_nanosleep',
           '232': 'epoll_wait', '270': 'pselect6', '271': 'ppoll', '281': 'epoll_pwait', '435': 'clone3',
           'running': 'running', '': 'unreadable'}
WAIT_COLS = (('actor waits learner', 'wait_learner_s'), ('driver GPU sync', 'drv_sync_s'),
             ('driver polls workers', 'drv_poll_s'), ('driver loop', 'drv_loop_s'), ('workers wait (sum over workers)', 'w_wait'),
             ('worker idle share', 'idle_share'), ('learner waits packing', 'b42_learn_pack_crit_s'),
             ('learner waits prefetch', 'b42_learn_prefetch_wait_s'),
             ('learner waits admission', 'b42_learn_admission_wait_s'), ('cf wait (sum over lanes)', 'cf_wait_s'),
             ('v2 wait', 'v2_wait_s'), ('p2 join wait', 'p2_join_wait_s'))


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
    for part in filter(None, spec.split(',')):   # (an empty spec: no lane)
        a, _, b = part.partition('-')
        out |= set(range(int(a), int(b or a)+1))
    return out


def main():
    R, tag = sys.argv[1], sys.argv[2]
    lane = cpus(sys.argv[3] if len(sys.argv) > 3 else '')
    gpu = sys.argv[4] if len(sys.argv) > 4 else '1'
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
    # (the learner's compile integrity: graphs compiled after the warm-up, checkpointed Spatial segments redone because
    #  a graph compiled during their forward; both must be 0 in a steady run)
    sc = rows[-1].get('block_steady_compiles', '')
    rd = sum(f(x, 'b42_learn_ckpt_compile_redo') for x in rows if x.get('b42_learn_ckpt_compile_redo', '') != '')
    flag = '  <-- NOT 0' if ((sc not in ('', '0', '0.0')) or rd > 0) else ''
    out.append(f'  compiles: block_steady_compiles {sc or "-"}, learn_ckpt_compile_redo '
               f'{rd:.0f}{"" if any("b42_learn_ckpt_compile_redo" in x for x in rows[:1]) else " (no column)"}{flag}')
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
    # (the ledger: where each cycle waits, medians u10+ of progress.csv's wait columns present in this run, s per update)
    wl = [(lab, med(s, c)) for lab, c in WAIT_COLS if c in rows[0]]
    if wl:
        out.append('  waits per update (u10+ medians): ' + ', '.join(f'{lab} {v:.2f}' for lab, v in wl if v == v))

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
        out.append('  cores used (steady, medians of 5 s samples): ' + ', '.join(
            f'{k} {med(m, k+"_cores"):.2f}' for k in kinds)
            + f', container {(f(m[-1], "cg_cpu_usec")-f(m[0], "cg_cpu_usec"))/1e6/dt:.2f}')
        if 'cg_sys_ticks' in m[0]:   # (the ledger: the container's cores split, and who they belong to)
            cg_all = (f(m[-1], "cg_cpu_usec")-f(m[0], "cg_cpu_usec"))/1e6/dt
            cg_u = (f(m[-1], 'cg_user_ticks')-f(m[0], 'cg_user_ticks'))/100/dt
            cg_s = (f(m[-1], 'cg_sys_ticks')-f(m[0], 'cg_sys_ticks'))/100/dt
            mean = (lambda c: st.mean([f(x, c) for x in m[1:] if f(x, c) == f(x, c)] or [float('nan')]))
            seen = sum(mean(k+'_cores') for k in kinds)
            reaped = sum(mean(k+'_child_cores') for k in kinds)
            out.append(f'  container cores (means): {cg_all:.2f} = user {cg_u:.2f} + sys {cg_s:.2f}; sampled live '
                       f'processes {seen:.2f} (' + ', '.join(f'{k} {mean(k+"_cores"):.2f} sys {mean(k+"_sys_cores"):.2f}'
                                                            for k in kinds)
                       + f'); reaped children (cutime + cstime of their parents) {reaped:.2f} ('
                       + ', '.join(f'{k} {mean(k+"_child_cores"):.2f}' for k in kinds)
                       + f'; reaped children are booked when they exit, a lump that overlaps the samples); not in a '
                       f'live sample {cg_all-seen:.2f}; nice (most common over the window) ' + ', '.join(
                           f'{k} {max(set(nv), key=nv.count) if nv else "-"}' for k in kinds
                           for nv in [[x.get(k+"_nice") for x in m if x.get(k+"_nice")]]))
        if 'game_sched_wait' in m[0]:   # (the ledger: per kind, on a CPU and runnable-but-waiting, s per s, means)
            out.append('  schedstat by kind (run / waiting for a CPU, s per s): ' + ', '.join(
                f'{k} {mean(k+"_sched_run"):.2f}/{mean(k+"_sched_wait"):.2f}' for k in kinds))
        sc_path = f'{R}/{tag}.sysc.csv'
        if os.path.exists(sc_path):   # (the ledger: where the tasks sit, share of samples per kind)
            t0_ = f(m[0], 't')
            agg = {}
            for x in csv.DictReader(open(sc_path)):
                if f(x, 't') >= t0_:
                    k_ = x['kind'] if not x['kind'].startswith('trainer:') else 'trainer'
                    d_ = agg.setdefault(k_, {})
                    nm_ = SYSCALL.get(x['syscall'], 'sys'+x['syscall'])
                    d_[nm_] = d_.get(nm_, 0)+f(x, 'tasks')
            for k_, d_ in sorted(agg.items()):
                tot_ = sum(d_.values())
                out.append(f'  {k_} tasks by syscall (share of samples): ' + ', '.join(
                    f'{nm_} {100*v_/tot_:.0f}%' for nm_, v_ in sorted(d_.items(), key=lambda kv: -kv[1])[:7]))
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
        if th[0].get('wait_ns'):   # (m0_mem with schedstat: on-CPU and runnable-but-waiting seconds per second)
            s0, s1 = {}, {}
            for x in th:
                if x['kind'] == 'thread' and x.get('wait_ns'):
                    s0.setdefault(x['name'], (f(x, 't'), f(x, 'run_ns'), f(x, 'wait_ns')))
                    s1[x['name']] = (f(x, 't'), f(x, 'run_ns'), f(x, 'wait_ns'))
            sch = {}
            for key, (t1, r1, w1) in s1.items():
                t0, r0, w0 = s0[key]
                if t1 > t0:
                    nm = key.rsplit(':', 1)[0]
                    a_ = sch.setdefault(nm, [0.0, 0.0])
                    a_[0] += (r1-r0)/1e9/(t1-t0)
                    a_[1] += (w1-w0)/1e9/(t1-t0)
            topw = sorted(sch.items(), key=lambda kv: -(kv[1][0]+kv[1][1]))[:12]
            out.append('  trainer threads schedstat (run / waiting for a CPU, s per s): ' + ', '.join(
                f'{n} {r:.2f}/{w:.2f}' for n, (r, w) in topw))
    # this run's GPU(s): 1 s samples; the index is the last column (older files without it: one GPU, `gpu`)
    import datetime as dt_
    g = {}
    if os.path.exists(f'{R}/{tag}.gpu'):
        for line in open(f'{R}/{tag}.gpu'):
            parts = [x.strip() for x in line.split(',')]
            try:
                t = dt_.datetime.strptime(parts[0], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt_.timezone.utc).timestamp()
                idx = parts[5] if len(parts) > 5 else gpu
                g.setdefault(idx, []).append((t, float(parts[1].split()[0]), float(parts[2].split()[0])))
            except (IndexError, ValueError):
                pass
    ends = [float(x.split()[0]) for x in open(f'{R}/{tag}.rows')] if os.path.exists(f'{R}/{tag}.rows') else []
    for idx, gi in sorted(g.items()):
        k = max(1, int(win))
        gs = gi[-k:]
        out.append(f'  GPU {idx} (1 s samples, steady): util median {st.median(x[1] for x in gs):.0f}% peak '
                   f'{max(x[1] for x in gs):.0f}%, memory median {st.median(x[2] for x in gs)/1024:.1f} GB peak '
                   f'{max(x[2] for x in gi)/1024:.1f} GB')
        if ends:
            # cycle phases from the row times (a row is written when its update ends): learner busy = [end - update_s,
            # end] (the actor collects meanwhile unless it waits), learner only = the actor's wait_learner_s before
            # the end
            ph = {'both': [], 'learner only': [], 'actor only': []}
            for kk in range(9, min(len(rows), len(ends))):
                e, up, wl = ends[kk], f(rows[kk], 'update_s'), f(rows[kk], 'wait_learner_s')
                wl = wl if wl == wl else 0.0
                for t, u, _ in gi:
                    if ends[kk-1] < t <= e:
                        ph['learner only' if t > e-wl else 'both' if t >= e-up else 'actor only'].append(u)
            out.append(f'  GPU {idx} util by cycle phase (u10+, nvidia-smi = share of time any kernel runs): ' + ', '.join(
                f'{kk} {st.median(v):.0f}% (n {len(v)})' for kk, v in ph.items() if v))
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
        cat = {}   # (the ledger: per CPU, per mpstat category: usr, nice, sys, iowait, irq + soft, steal, idle)
        cats = (('usr', ('%usr',)), ('nice', ('%nice',)), ('sys', ('%sys',)), ('iowait', ('%iowait',)),
                ('irq', ('%irq', '%soft')), ('steal', ('%steal',)), ('idle', ('%idle',)))
        lines = open(f'{R}/{tag}.mpstat').read().splitlines()
        hdr = None
        for line in lines:
            w_ = line.split()
            if '%idle' in w_:
                hdr = w_
                continue
            if hdr and len(w_) == len(hdr) and w_[hdr.index('CPU')].isdigit():
                busy.setdefault(int(w_[hdr.index('CPU')]), []).append(100-float(w_[hdr.index('%idle')]))
                for nm_, cols_ in cats:
                    cat.setdefault(int(w_[hdr.index('CPU')]), {}).setdefault(nm_, []).append(
                        sum(float(w_[hdr.index(c_)]) for c_ in cols_ if c_ in hdr))
        n_keep = max(1, int(win/5))
        lb = [st.mean(v[-n_keep:]) for c, v in busy.items() if c in lane]
        ob = [st.mean(v[-n_keep:]) for c, v in busy.items() if c not in lane]
        if lb:
            out.append(f'  mpstat (steady): this lane {len(lb)} CPUs mean busy {st.mean(lb):.0f}% (sum {sum(lb)/100:.1f} cores); '
                       f'other CPUs {len(ob)} mean busy {st.mean(ob) if ob else 0:.0f}% (sum {sum(ob)/100:.1f} cores)')
        for lab_, sel_ in (('this lane', [c for c in cat if c in lane]), ('whole host', list(cat))):
            if sel_:
                out.append(f'  mpstat cores by category (steady, {lab_}, {len(sel_)} CPUs): ' + ', '.join(
                    f'{nm_} {sum(st.mean(cat[c][nm_][-n_keep:]) for c in sel_)/100:.1f}' for nm_, _ in cats))
    if os.path.exists(f'{R}/{tag}.freq1') and ends:   # (the ledger: CPU frequency by cycle phase, 1 s samples)
        fq = [list(map(float, x.split())) for x in open(f'{R}/{tag}.freq1').read().splitlines() if len(x.split()) > 1]
        sets = [('lane', sorted(lane))]
        if os.environ.get('FREQ_CPUS'):   # (e.g. the learner's pinned CPUs)
            sets.append((os.environ['FREQ_CPUS'], sorted(cpus(os.environ['FREQ_CPUS']))))
        for lab_, cs_ in sets:
            ph = {'both': [], 'learner only': [], 'actor only': []}
            for kk in range(9, min(len(rows), len(ends))):
                e, up, wl = ends[kk], f(rows[kk], 'update_s'), f(rows[kk], 'wait_learner_s')
                wl = wl if wl == wl else 0.0
                for x in fq:
                    if ends[kk-1] < x[0] <= e and cs_:
                        v_ = st.mean(x[1+c] for c in cs_ if 1+c < len(x))/1000
                        ph['learner only' if x[0] > e-wl else 'both' if x[0] >= e-up else 'actor only'].append(v_)
            out.append(f'  CPU MHz by cycle phase (u10+, 1 s samples, mean over {lab_}): ' + ', '.join(
                f'{kk} {st.median(v):.0f} (n {len(v)})' for kk, v in ph.items() if v))
    if os.path.exists(f'{R}/{tag}.cpufreq'):   # (CPU MHz, every CPU, 5 s, from /proc/cpuinfo)
        fl = [x.split() for x in open(f'{R}/{tag}.cpufreq').read().splitlines()]
        fl = [list(map(float, x)) for x in fl if len(x) > 1]
        if fl:
            t1 = fl[-1][0]
            fs = [x for x in fl if x[0] >= t1-win] or fl
            ln = [st.mean(x[1+c] for c in lane if 1+c < len(x)) for x in fs] if lane else []
            al = [st.mean(x[1:]) for x in fs]
            out.append(f'  CPU MHz (steady, 5 s samples): this lane median {st.median(ln):.0f} (min {min(ln):.0f}), '
                       f'all CPUs median {st.median(al):.0f}' if ln else
                       f'  CPU MHz (steady, 5 s samples): all CPUs median {st.median(al):.0f}')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
