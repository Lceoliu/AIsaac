#!/usr/bin/env python3
"""A training run's event watcher: read-only (the run directory's guard.log and progress / episodes files), it never
touches the trainers or the guard. Every minute it appends new events to <run dir>/events.log, one line each:

    <n> <YYYY-mm-dd HH:MM:SS> <kind> <text>

kinds: guard (a guard.log line about an attempt, a resume, a warning, a stall, a stop or the end -- at once),
u300 (rank 0 reached update 300), milestone (each multiple of 400: train_summary.py's table for that window),
alert (one of the coordinator's report-only thresholds, at most once per 100 updates each), end (the guard ended).
The reader (the operator's waiter) asks for the lines past the last n it saw.

  setsid nohup python3 train_watch.py <run dir> > <run dir>/train_watch.out 2>&1 < /dev/null &

Thresholds (TRAINING.md section 8): KL median of the last 50 updates > 0.05; a head's entropy (median of 10 updates)
down by more than half against 100 updates before; at 2500 game hours in all (both ranks): the zero-progress share of
all episodes not below the u1-400 window's and the floor starts' mean new rooms not above it; ctr_mem_gb > 160 (MEM_GB 176 since the resume of 2026-11-13), or
after update 800 growing more than 2 GB per 100 updates (the coordinator, 2026-11-12: MEM_GB 144, the buffers fill to
about 116 GB); the combined x real time's 100-update median < 300; v2_idle (2026-11-13): a rank holding >= 256
teacher-v2 records with no v2_loss_pi in 20 updates in a row (with --cf-per-clear > 0 in train_args: once per rank). The milestones add the characters by start builds
(each rank's distinct fresh-start (group, seed)) and rank 0's new checkpoint_rotation.log lines.
Standard library only (the host's python3); train_summary.py beside it."""
import csv
import os
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_summary as TS   # noqa: E402

GUARD_KEYS = ('attempt', 'resuming', 'WARNING', 'STALL', 'stopped', 'STOP', 'guard end', 'cause', 'no progress',
              'failed', 'unreadable', 'free disk', 'DRIVER OFF', 'driver_off', 'early_end', 'early end', 'cgroup_mem')
HEADS = ('ent_move', 'ent_shoot', 'ent_bomb', 'ent_item', 'ent_pill')


class Watch:
    def __init__(self, rd):
        self.rd = rd.rstrip('/')
        self.ev = os.path.join(self.rd, 'events.log')
        self.n = 0
        if os.path.exists(self.ev):
            with open(self.ev) as f:
                for line in f:
                    self.n = max(self.n, int(line.split(' ', 1)[0]))
        self.guard_pos = 0
        self.done_u300, self.milestones, self.alert_at, self.rot_pos = False, set(), {}, 0
        self.state = os.path.join(self.rd, 'train_watch.state')
        if os.path.exists(self.state):   # (a restarted watcher: what it already reported)
            with open(self.state) as f:
                for line in f:
                    k, _, v = line.strip().partition(' ')
                    if k == 'guard_pos':
                        self.guard_pos = int(v)
                    elif k == 'u300':
                        self.done_u300 = True
                    elif k == 'milestone':
                        self.milestones.add(int(v))
                    elif k == 'rot_pos':
                        self.rot_pos = int(v)
                    elif k == 'alert':
                        a, _, u = v.partition(' ')
                        self.alert_at[a] = int(u)

    def save(self):
        with open(self.state + '.tmp', 'w') as f:
            f.write(f'guard_pos {self.guard_pos}\n')
            f.write(f'rot_pos {self.rot_pos}\n')
            if self.done_u300:
                f.write('u300 1\n')
            for m in sorted(self.milestones):
                f.write(f'milestone {m}\n')
            for a, u in self.alert_at.items():
                f.write(f'alert {a} {u}\n')
        os.replace(self.state + '.tmp', self.state)

    def emit(self, kind, text):
        self.n += 1
        with open(self.ev, 'a') as f:
            f.write(f"{self.n} {time.strftime('%Y-%m-%d %H:%M:%S')} {kind} {text}\n")

    def progress(self):
        out = {}
        for r in ('0', '1'):
            p = os.path.join(self.rd, f'{os.path.basename(self.rd)}-s{r}', 'progress.csv')
            if os.path.exists(p):
                with open(p) as f:
                    out[r] = list(csv.DictReader(f))
        return out

    def guard(self):
        p = os.path.join(self.rd, 'guard.log')
        if not os.path.exists(p):
            return False
        ended = False
        with open(p) as f:
            f.seek(self.guard_pos)
            for line in f:
                if not line.endswith('\n'):
                    break
                self.guard_pos += len(line.encode())
                if 'container memory 0.0 GB' in line:   # (an old guard's false warning)
                    continue
                if any(k in line for k in GUARD_KEYS):
                    self.emit('guard', line.strip())
                if 'guard end' in line:
                    ended = True
                elif 'guard start' in line:   # (2026-11-13: a later guard on the same run -- an earlier guard's end
                    ended = False             #  read in the same pass is not this guard's)
        return ended

    def builds(self):
        """the characters by start builds: each rank's distinct fresh-start (group, seed), by char0"""
        try:
            eps, _ = TS.load(self.rd)
        except Exception as e:   # noqa: BLE001
            return f'builds: {e}'
        seen = {}
        for e in eps:
            if e.get('start') == 'fresh':
                seen.setdefault((e.get('_rank'), e.get('group'), e.get('seed')), e.get('char0'))
        n = [sum(1 for c in seen.values() if c == k) for k in range(3)]
        return f'start builds by character (Isaac / Magdalene / Cain): {n[0]} / {n[1]} / {n[2]} of {len(seen)}'

    def rotation(self):
        """rank 0's checkpoint_rotation.log lines since the last milestone"""
        p = os.path.join(self.rd, f'{os.path.basename(self.rd)}-s0', 'checkpoint_rotation.log')
        if not os.path.exists(p):
            return 'checkpoint_rotation.log: none'
        with open(p) as fh:
            fh.seek(self.rot_pos)
            new = fh.read()
        self.rot_pos += len(new.encode())
        lines = [x for x in new.splitlines() if x.strip()]
        return 'checkpoint_rotation.log new: ' + (' || '.join(lines) if lines else 'none')

    def per_clear(self):
        """(2026-11-13) the run's trainer flags (<run dir>/train_args) set --cf-per-clear above 0."""
        try:
            a = open(os.path.join(self.rd, 'train_args')).read().split()
        except OSError:
            return False
        v = [a[i + 1] for i, x in enumerate(a[:-1]) if x == '--cf-per-clear']
        try:
            return bool(v) and float(v[-1]) > 0
        except ValueError:
            return False

    def alert(self, name, u, text):
        if u - self.alert_at.get(name, -10 ** 9) >= 100:
            self.alert_at[name] = u
            self.emit('alert', f'{name} u{u}: {text}')

    def check(self, P):
        p0 = P.get('0', [])
        if not p0:
            return
        u = int(p0[-1]['update'])
        f = lambda x, k: float(x[k]) if x.get(k) not in (None, '') else float('nan')
        if u >= 300 and not self.done_u300:
            self.done_u300 = True
            w = p0[-50:]
            mem = [f(x, 'ctr_mem_gb') for x in p0]
            held = ' '.join(f"rank {r} teach_held {P[r][-1].get('teach_held', '?')} v2_held {P[r][-1].get('v2_held', '?')}"
                            for r in sorted(P))
            self.emit('u300', f'u{u}: ctr_mem_gb u250 {mem[min(249, len(mem) - 1)]:.1f} now {mem[-1]:.1f} (expected '
                              f'105-112); {held}; '
                              f'block_gc_s median50 {st.median(f(x, "block_gc_s") for x in w):.2f}; '
                              f'x_real median50 {sum(st.median(f(x, "x_real_time") for x in P[r][-50:]) for r in P):.0f}; '
                              f'kl median50 {st.median(f(x, "kl") for x in w):.4f}; '
                              + ' '.join(f'{h[4:]} {st.median(f(x, h) for x in w):.2f}' for h in HEADS if h in w[-1])
                              + f'; lr {f(w[-1], "lr"):.2e}; mismatches {sum(f(x, "shard_mismatches") for x in p0):.0f}; '
                              f'steady compiles {sum(f(x, "block_steady_compiles") for x in p0):.0f}')
        for m in range(400, u + 1, 400):
            if m not in self.milestones:
                self.milestones.add(m)
                try:
                    out = subprocess.run([sys.executable, os.path.join(HERE, 'train_summary.py'), self.rd,
                                          '--window', '400'], capture_output=True, text=True, timeout=600).stdout
                except Exception as e:   # (a report must not stop the watcher)
                    out = f'train_summary failed: {e}'
                self.emit('milestone', f'u{m}: ' + out.replace('\n', ' | ') + ' | ' + self.builds() + ' | '
                          + self.rotation())
        # the report-only thresholds
        if len(p0) >= 50:
            k = st.median(f(x, 'kl') for x in p0[-50:])
            if k > 0.05:
                self.alert('kl', u, f'KL median of the last 50 updates {k:.4f} > 0.05')
        if len(p0) >= 110:
            for h in HEADS:
                if h in p0[-1]:
                    now_ = st.median(f(x, h) for x in p0[-10:])
                    then = st.median(f(x, h) for x in p0[-110:-100])
                    if then > 0 and now_ < then / 2:
                        self.alert(f'entropy_{h[4:]}', u, f'{h} {then:.2f} -> {now_:.2f} in 100 updates')
        mem = f(p0[-1], 'ctr_mem_gb')
        if mem > 160:
            self.alert('memory', u, f'ctr_mem_gb {mem:.1f} > 160')
        if u > 800:
            back = [x for x in p0 if int(x['update']) == u - 100]
            if back and mem - f(back[0], 'ctr_mem_gb') > 2:
                self.alert('memory_growth', u, f'ctr_mem_gb +{mem - f(back[0], "ctr_mem_gb"):.1f} GB in 100 updates '
                                               f'after update 800')
        pc_ = self.per_clear()
        for r in sorted(P):   # (2026-11-13) the teacher-v2 / CF records held but not trained on (main1: 0 of 824 rows)
            w = P[r][-20:]
            # (--cf-per-clear > 0: the CF records come slowly, a slow v2 is expected -- each rank's alert once a run)
            name_ = f'v2_idle_pc_s{r}' if pc_ else f'v2_idle_s{r}'
            if pc_ and name_ in self.alert_at:
                continue
            if len(w) == 20 and all(f(x, 'v2_held') >= 256 and x.get('v2_loss_pi') in (None, '') for x in w):
                self.alert(name_, u, f'rank {r}: v2_held {w[-1].get("v2_held")} >= 256 and no v2_loss_pi in '
                                     f'the last 20 updates (v2_step {w[-1].get("v2_step")})'
                                     + (' (--cf-per-clear on: reported once)' if pc_ else ''))
        if len(p0) >= 100 and all(len(P[r]) >= 100 for r in P):
            x = sum(st.median(f(y, 'x_real_time') for y in P[r][-100:]) for r in P)
            if x < 300:
                self.alert('speed', u, f'combined x real time, 100-update median {x:.0f} < 300')
        gh = sum(f(P[r][-1], 'game_hours') for r in P)
        if gh >= 2500 and 'learning' not in self.alert_at:
            eps, _ = TS.load(self.rd)
            a = [e for e in eps if int(e.get('update', 0)) <= 400]
            b = [e for e in eps if u - 400 < int(e.get('update', 0)) <= u]
            z = lambda E: sum(TS.zero_progress(e) for e in E) / max(len(E), 1)
            rooms = lambda E: st.mean([float(e.get('rooms') or 0) for e in E if e.get('start') == 'fresh'] or [0.0])
            if z(b) >= z(a) and rooms(b) <= rooms(a):
                self.alert('learning', u, f'at {gh:.0f} game hours: zero-progress {z(a):.3f} -> {z(b):.3f}, fresh '
                                          f'starts\' new rooms {rooms(a):.2f} -> {rooms(b):.2f} (u1-400 -> the last 400)')
            else:
                self.alert_at['learning'] = u   # (checked once, passed)

    def run(self):
        while True:
            try:
                ended = self.guard()
                self.check(self.progress())
                self.save()
                if ended:
                    self.emit('end', 'the guard ended')
                    self.save()
                    return
            except Exception as e:   # (a bad line or a file being rotated: next minute)
                print(time.strftime('%F %T'), 'watch error:', e, flush=True)
            time.sleep(60)


if __name__ == '__main__':
    Watch(sys.argv[1]).run()
