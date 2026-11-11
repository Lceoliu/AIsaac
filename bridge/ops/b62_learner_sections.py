"""BM1 step 4: where a learner update's time goes, per section -- GPU time or host waiting -- from an nsys report of a
train_tok run with ISAAC_BLOCK_NVTX=1 (cuda + nvtx traces).

  python3 b62_learner_sections.py <report.sqlite> [learner device index, default: the busiest in 'ppo']

train_tok's learn() marks its sections with consecutive NVTX start/end ranges 'learner.after_<previous section>'
(sec(name) ends the running range and starts 'learner.after_<name>'), inside the push/pop range 'learner.update'.
A range 'learner.after_X' therefore spans the code of the section that follows X: its name is the next range's
suffix (the last one: 'tail' / the update's end). Per section, summed over every complete learner.update in the
window and divided by their number:
  wall       the section's host time (what the progress.csv u_* columns measure, no synchronisation),
  launched   GPU time (kernels + copies + memsets on the learner's device) of the work launched during it, by any
             thread (the backward's autograd threads included),
  busy       the learner device's busy time (union of its activity) inside the section's time,
  idle       wall - busy: the learner's GPU idle while this section ran (host-side time not hidden by GPU work).
Standard library only (sqlite3)."""
import sqlite3
import sys


def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def covered(u, lo, hi, starts=None):
    """Length of the union u (sorted, disjoint) inside [lo, hi]; starts: [a for a, b in u] (bisect)."""
    import bisect
    k = max(0, bisect.bisect_right(starts, lo)-1) if starts is not None else 0
    s = 0
    for a, b in u[k:]:
        if b <= lo:
            continue
        if a >= hi:
            break
        s += min(b, hi)-max(a, lo)
    return s


def main():
    c = sqlite3.connect(sys.argv[1])
    T = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    names = dict(c.execute('select id, value from StringIds'))
    rows = c.execute('select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null').fetchall()
    ev = [(a, b, (t or names.get(i) or ''), g) for a, b, t, i, g in rows]
    upd = sorted((a, b) for a, b, t, g in ev if t == 'learner.update')
    secs = sorted((a, b, t[len('learner.after_'):]) for a, b, t, g in ev if t.startswith('learner.after_'))
    work = []   # (start, end, device, launch time)
    rt = dict(c.execute('select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME')) \
        if 'CUPTI_ACTIVITY_KIND_RUNTIME' in T else {}
    for tb in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
        if tb in T:
            work += [(a, b, d, rt.get(k, a)) for a, b, d, k in
                     c.execute(f'select start, end, deviceId, correlationId from {tb}')]
    full = [(a, b) for a, b in upd if any(s[0] >= a for s in secs)]
    print(f'learner.update ranges: {len(upd)}; sections ranges: {len(secs)}; GPU activities: {len(work)}')
    # sections inside each complete update, named by the next range's suffix
    named = []
    for a, b in full:
        inside = [s for s in secs if a <= s[0] < b]
        for k, (s0, s1, prev) in enumerate(inside):
            nxt = inside[k+1][2] if k+1 < len(inside) else 'end'
            named.append((s0, min(s1, b), nxt))
    if not named:
        print('no learner sections in the window'); return
    devs = sorted({d for _, _, d, _ in work})
    if len(sys.argv) > 2:
        ldev = int(sys.argv[2])
    else:   # the device with the most GPU time launched during 'ppo'
        ppo = [(s0, s1) for s0, s1, n in named if n == 'ppo']
        tot = {d: sum(e-s for s, e, dd, l in work if dd == d and any(p0 <= l < p1 for p0, p1 in ppo)) for d in devs}
        ldev = max(tot, key=tot.get)
    import bisect
    busy_u = union([[s, e] for s, e, d, _ in work if d == ldev])
    bstart = [a for a, _ in busy_u]
    by_launch = sorted((l, e-s) for s, e, d, l in work if d == ldev)
    lt = [x[0] for x in by_launch]
    cum = [0]
    for _, du in by_launch:
        cum.append(cum[-1]+du)
    agg = {}
    for s0, s1, n in named:
        g = agg.setdefault(n, [0, 0, 0])
        g[0] += s1-s0
        g[1] += cum[bisect.bisect_left(lt, s1)]-cum[bisect.bisect_left(lt, s0)]
        g[2] += covered(busy_u, s0, s1, bstart)
    n_upd = len(full)
    print(f'complete updates: {n_upd}; learner device {ldev} (devices {devs}); seconds per update')
    print(f'{"section":14s} {"wall":>7s} {"launched":>9s} {"busy":>7s} {"idle":>7s}')
    tw = tl = tb = 0
    for n, (w, l, bz) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f'{n:14s} {w/1e9/n_upd:7.2f} {l/1e9/n_upd:9.2f} {bz/1e9/n_upd:7.2f} {(w-bz)/1e9/n_upd:7.2f}')
        tw, tl, tb = tw+w, tl+l, tb+bz
    print(f'{"total":14s} {tw/1e9/n_upd:7.2f} {tl/1e9/n_upd:9.2f} {tb/1e9/n_upd:7.2f} {(tw-tb)/1e9/n_upd:7.2f}')


if __name__ == '__main__':
    main()
