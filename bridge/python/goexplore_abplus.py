"""Go-Explore phase 1 on AB+ rooms (EXPERIMENTS.md A8): per room an archive of cells, returns by batched replay (bridge
abp-0.2.12 play) checked against the hidden-state digest, sticky random exploration (isaac_bridge/abplus_goexplore.py).

Each worker process owns one AB+ instance (render-lite exact mode by default) and serves tasks: a room's root (reset,
first cell, explore), an exploration (return to a cell, check, explore) or the final check of a room's best win (return,
check). The rooms are the seeds' rooms of a tasks file or of one group of a groups file, as in training (TaskSampler on
the seed). --rooms-at-once rooms are explored at a time, each for --iterations explorations; its best win is then
returned to once more as a check, and the room is written out.

Outputs in --out: config.json; progress.jsonl (every --progress-s); results.jsonl (one line per room); rooms/<seed>.json
(summary, every cell without its trajectory, the trajectories of the wins and of the best cell) and
rooms/<seed>.cells.json.gz (every cell with its trajectory: action codes as hex, one byte each, see abplus_goexplore).
--resume skips rooms already written.

usage (from the bridge's python dir, PYTHONPATH=.):
  python goexplore_abplus.py --groups-file ../catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:8 --iterations 1000 --workers 12 --out ../runs/gx/<name>
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from isaac_bridge.abplus import BRIDGE_VERSION
from isaac_bridge.abplus_goexplore import WIN, Archive, GxConfig, worker_main
from isaac_bridge.abplus_groups import load_groups
from isaac_bridge.steam_watch import steam_running

HERE = Path(__file__).resolve().parent
MAX_VERIFY = 3   # wins tried by the final check of a room, best first


def default_bridge_lua():
    """The bridge next to this copy: <copy>/abp_bridge.lua when deployed (python/ inside the bridge dir), else the
    workspace's ../abplus/abp_bridge.lua."""
    for path in (HERE.parent / 'abp_bridge.lua', HERE.parent / 'abplus' / 'abp_bridge.lua'):
        if path.is_file():
            return str(path)
    raise SystemExit('abp_bridge.lua not found next to this script; pass --bridge-lua')


def load_spec(args):
    """{'name', 'spec': {weights, normal, boss}, 'target', 'seconds', 'tasks'} of --tasks or of --groups-file/--group."""
    if args.groups_file:
        groups = {g['name']: g for g in load_groups(args.groups_file)}
        if args.group not in groups:
            raise SystemExit(f'--group must be one of {sorted(groups)}')
        g = groups[args.group]
        seconds = args.seconds or g['seconds']
        return dict(name=g['name'], spec=g['spec'], target=g['target'], seconds=float(seconds), tasks=g['tasks'])
    raw = json.loads(Path(args.tasks).read_text(encoding='utf8'))
    return dict(name=Path(args.tasks).stem, spec={k: raw[k] for k in ('weights', 'normal', 'boss')},
                target=raw.get('target'), seconds=float(args.seconds or 180.0), tasks=str(Path(args.tasks).resolve()))


def parse_seeds(text):
    if text.startswith('range:'):
        start, count = map(int, text.split(':')[1:])
        return list(range(start, start + count))
    raw = json.loads(Path(text).read_text())
    return [int(s['seed'] if isinstance(s, dict) else s) for s in (raw['seeds'] if isinstance(raw, dict) else raw)]


def cell_json(cell, with_actions):
    out = cell.summary()
    if with_actions:
        out['actions'] = cell.actions.hex()
    return out


class Manager:
    def __init__(self, args, cfg, spec, seeds):
        self.args, self.cfg, self.spec = args, cfg, spec
        self.out = Path(args.out)
        (self.out / 'rooms').mkdir(parents=True, exist_ok=True)
        self.pending = list(seeds)
        self.rooms, self.order, self.rr = {}, [], 0
        self.rng = np.random.default_rng(args.rng_seed)
        self.inflight, self.next_id = {}, 0
        self.ctx = mp.get_context('spawn')
        self.results = self.ctx.Queue()
        self.workers = [None] * args.workers
        self.task_queues = [None] * args.workers
        self.fatal = Counter()
        self.idle = set()
        self.sent = [dict() for _ in range(args.workers)]    # worker -> seed -> archive log entries sent
        self.drops = [[] for _ in range(args.workers)]       # worker -> finished rooms it may forget
        self.totals = Counter()
        self.t0 = time.time()
        self.last_progress = 0.0
        self.finished = 0

    # ---------------------------------------------------------------------------------------------------- workers
    def start_worker(self, i):
        q = self.ctx.Queue()
        name, port = f'{self.args.name}{i}', self.args.port + i
        p = self.ctx.Process(target=worker_main, args=(i, name, port, self.cfg, self.spec, q, self.results),
                             name=f'gx-worker-{i}')
        p.start()
        self.workers[i], self.task_queues[i] = p, q
        self.sent[i] = {}
        self.drops[i] = []
        self.idle.add(i)

    def stop_workers(self):
        for q in self.task_queues:
            if q is not None:
                try:
                    q.put(None)
                except Exception:
                    pass
        deadline = time.time() + 90
        for p in self.workers:
            if p is not None:
                p.join(timeout=max(1.0, deadline - time.time()))
        for p in self.workers:
            if p is not None and p.is_alive():
                p.terminate()

    # ------------------------------------------------------------------------------------------------------ rooms
    def activate(self):
        while len(self.rooms) < self.args.rooms_at_once and self.pending:
            seed = self.pending.pop(0)
            self.rooms[seed] = dict(archive=None, root_inflight=False, root_tries=0, inflight=0, verify=None,
                                    verify_attempts=[], verified=None, failed=None, started=time.time(), frames=0,
                                    seconds=Counter())
            self.order.append(seed)

    def room_task(self, seed, room):
        """(kind, fields) of the room's next task, None when it has none to hand out now."""
        args = self.args
        archive = room['archive']
        if archive is None:
            if room['failed'] or room['root_inflight']:
                return None
            return 'root', dict()
        if archive.dispatched < args.iterations:
            cell = archive.select(self.rng)
            if cell is not None:
                archive.dispatched += 1
                return 'explore', dict(key=cell.key, actions=cell.actions, digest=cell.digest, frames=cell.frames,
                                       outcome='running')
        if room['verify'] is None and room['inflight'] == 0:
            # the final check: the best win not tried yet (a win found in an exploration whose state was not the replay's
            # fails it; then the next one, up to MAX_VERIFY)
            tried = {a['key'] for a in room['verify_attempts']}
            wins = [c for c in archive.wins() if c.key not in tried]
            if wins:
                best = wins[0]
                room['verify'] = 'inflight'
                return 'verify', dict(key=best.key, actions=best.actions, digest=best.digest, frames=best.frames,
                                      outcome=WIN)
        return None

    def room_done(self, room):
        archive = room['archive']
        if archive is None:
            return bool(room['failed']) and not room['root_inflight']
        if room['inflight'] or room['verify'] == 'inflight':
            return False
        if archive.dispatched < self.args.iterations and archive.selectable():
            return False
        return room['verify'] is not None or not archive.wins()

    def dispatch(self, w):
        n = len(self.order)
        for k in range(n):
            seed = self.order[(self.rr + k) % n]
            room = self.rooms[seed]
            picked = self.room_task(seed, room)
            if picked is None:
                continue
            self.rr = (self.rr + k + 1) % n
            kind, fields = picked
            tid = self.next_id
            self.next_id += 1
            archive = room['archive']
            task = dict(id=tid, kind=kind, seed=seed, rng=int(self.rng.integers(1 << 62)),
                        drop=self.drops[w], **fields)
            self.drops[w] = []
            if archive is not None:
                start = self.sent[w].get(seed, 0)
                task['delta'] = archive.log[start:]
                task['ctx'] = archive.ctx
                self.sent[w][seed] = len(archive.log)
            else:
                room['root_inflight'] = True
            room['inflight'] += 1
            self.inflight[tid] = dict(worker=w, seed=seed, kind=kind, key=fields.get('key'),
                                      prefix=fields.get('actions', b''), sent=time.time())
            self.task_queues[w].put(task)
            self.idle.discard(w)
            return True
        return False

    def handle(self, res):
        kind = res.get('kind')
        w = res['worker']
        if kind == 'ready':
            return
        if kind == 'recycled':
            self.totals['recycles'] += 1
            return
        if kind == 'fatal':
            self.fatal[w] += 1
            self.log_event(dict(event='worker_fatal', worker=w, error=res.get('error')))
            for tid, t in list(self.inflight.items()):
                if t['worker'] == w:
                    self.lost(tid)
            self.idle.discard(w)
            if self.fatal[w] <= 3:
                self.workers[w].join(timeout=30)
                self.start_worker(w)
            return
        task = self.inflight.pop(res['id'], None)
        if task is None:   # its worker was already given up on
            return
        self.idle.add(w)
        seed = task['seed']
        room = self.rooms[seed]
        room['inflight'] -= 1
        frames = int(res.get('return_frames', 0)) + int(res.get('explore_frames', 0))
        room['frames'] += frames
        self.totals['frames'] += frames
        self.totals['return_frames'] += int(res.get('return_frames', 0))
        self.totals['explore_frames'] += int(res.get('explore_frames', 0))
        for k in ('reset_s', 'return_s', 'explore_s', 'digest_s', 'seconds'):
            room['seconds'][k] += float(res.get(k, 0.0))
            self.totals[k] += float(res.get(k, 0.0))
        self.totals['digests'] += int(res.get('digests', 0))
        self.totals['resets'] += 1 + int(res.get('retry') is not None)
        self.totals['tasks_' + res['status']] += 1
        status = res['status']
        retry = res.get('retry')
        if retry is not None:
            # a digest mismatch repeated in the same instance: matched now / the same state again / another one
            outcome = 'match' if retry['match'] else 'same' if retry['same'] else 'other'
            self.totals['retry_' + outcome] += 1
            if room['archive'] is not None:
                room['archive'].stats['retry_' + outcome] += 1
            self.log_event(dict(event='digest_retry', seed=seed, key=list(task['key']), steps=len(task['prefix']),
                                worker=w, **retry))
        if task['kind'] == 'root':
            room['root_inflight'] = False
            if status == 'ok':
                archive = Archive(seed, res['root'], self.cfg)
                archive.merge(archive.cells[0].key, b'', res, 0)
                room['archive'] = archive
                room['root'] = res['root']
            else:
                room['root_tries'] += 1
                if status == 'unusable' or room['root_tries'] >= 3:
                    room['failed'] = res.get('error') or status
                self.log_event(dict(event='root_' + status, seed=seed, error=res.get('error')))
        elif task['kind'] == 'explore':
            archive = room['archive']
            if status == 'ok':
                archive.merge(task['key'], task['prefix'], res, archive.stats['returns'] + 1)
            else:
                archive.failed(task['key'], error=status == 'error')
                if status == 'error' or self.totals['fail_logged'] < 50:
                    self.totals['fail_logged'] += 1
                    self.log_event(dict(event='return_' + status, seed=seed, key=list(task['key']),
                                        steps=len(task['prefix']), reason=res.get('reason'), played=res.get('played'),
                                        outcome=res.get('outcome'), error=res.get('error')))
        elif task['kind'] == 'verify':
            self.verified(room, task['key'], dict(status=status, reason=res.get('reason'), played=res.get('played'),
                                                  outcome=res.get('outcome'), error=res.get('error'),
                                                  retry=res.get('retry')))

    def verified(self, room, key, attempt):
        """A final check's result: done when it passed, when MAX_VERIFY wins were tried or no other win is left."""
        room['verify_attempts'].append(dict(key=key, **attempt))
        if attempt['status'] == 'ok':
            room['verified'], room['verify'] = key, 'done'
            return
        tried = {a['key'] for a in room['verify_attempts']}
        more = any(c.key not in tried for c in room['archive'].wins())
        room['verify'] = None if more and len(room['verify_attempts']) < MAX_VERIFY else 'done'

    def lost(self, tid):
        """A task whose worker died: counted as an error of its cell (a root is retried)."""
        task = self.inflight.pop(tid)
        room = self.rooms[task['seed']]
        room['inflight'] -= 1
        if task['kind'] == 'root':
            room['root_inflight'] = False
            room['root_tries'] += 1
            if room['root_tries'] >= 3:
                room['failed'] = 'worker died'
        elif task['kind'] == 'explore':
            room['archive'].failed(task['key'], error=True)
        else:
            self.verified(room, task['key'], dict(status='error', error='worker died'))

    def finish_rooms(self):
        for seed in [s for s in self.order if self.room_done(self.rooms[s])]:
            room = self.rooms.pop(seed)
            self.order.remove(seed)
            self.rr = 0
            for d in self.drops:
                d.append(seed)
            for s in self.sent:
                s.pop(seed, None)
            self.write_room(seed, room)
            self.finished += 1

    def write_room(self, seed, room):
        archive = room['archive']
        elapsed = time.time() - room['started']
        rec = dict(seed=seed, group=self.spec['name'], seconds=round(elapsed, 1), frames=room['frames'],
                   game_hours=round(room['frames'] / 30 / 3600, 4),
                   x_realtime=round(room['frames'] / 30 / max(elapsed, 1e-9), 1),
                   worker_seconds={k: round(v, 1) for k, v in room['seconds'].items()})
        if archive is None:
            rec.update(failed=str(room['failed'])[:500])
        else:
            # the best state: the verified win, else the archive's best (then unverified)
            verified = room['verified']
            best = archive.cells[archive.index[verified]] if verified is not None else archive.best()
            summary = archive.summary()
            summary['best'] = dict(outcome=best.outcome, hurt=best.hurt, health=best.health, progress=best.progress,
                                   frames=best.frames, steps=len(best.actions))
            attempts = [dict(a, key=list(a['key'])) for a in room['verify_attempts']]
            rec.update(room=room['root']['room'], hp0=archive.ctx['hp0'], health0=archive.ctx['health0'], **summary,
                       verify=dict(status='ok' if verified is not None else 'fail' if attempts else 'none',
                                   attempts=len(attempts)),
                       verify_attempts=attempts, best_verified=verified is not None, best_cell=cell_json(best, False))
            detail = dict(rec, cells=[cell_json(c, False) for c in archive.cells],
                          wins=[cell_json(c, True) for c in archive.wins()], best=cell_json(best, True))
            (self.out / 'rooms' / f'{seed}.json').write_text(json.dumps(detail, separators=(',', ':')))
            with gzip.open(self.out / 'rooms' / f'{seed}.cells.json.gz', 'wt', encoding='utf8') as f:
                for c in archive.cells:
                    f.write(json.dumps(cell_json(c, True), separators=(',', ':')) + '\n')
        if archive is None:
            (self.out / 'rooms' / f'{seed}.json').write_text(json.dumps(rec))
        with open(self.out / 'results.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
        best = rec.get('best', {})
        print(f"[room {seed}] {rec.get('room', {}).get('name')} cells {rec.get('cells')} wins {rec.get('wins')} "
              f"best {best} verify {rec.get('verify')} fails {rec.get('return_fails', 0)} "
              f"{rec['game_hours']} game h in {rec['seconds']} s", flush=True)

    def log_event(self, rec):
        rec = dict(t=round(time.time() - self.t0, 1), **rec)
        with open(self.out / 'events.jsonl', 'a') as f:
            f.write(json.dumps(rec, default=str) + '\n')

    def progress(self, force=False):
        now = time.time()
        if not force and now - self.last_progress < self.args.progress_s:
            return
        self.last_progress = now
        elapsed = now - self.t0
        t = self.totals
        rooms = {str(s): (r['archive'].summary() if r['archive'] is not None else dict(root_pending=True))
                 for s, r in self.rooms.items()}
        tasks = sum(v for k, v in t.items() if k.startswith('tasks_'))
        rec = dict(t=round(elapsed, 1), rooms_done=self.finished, rooms_pending=len(self.pending), tasks=tasks,
                   tasks_per_s=round(tasks / max(elapsed, 1e-9), 2),
                   ok=t['tasks_ok'], fail=t['tasks_fail'], error=t['tasks_error'],
                   game_hours=round(t['frames'] / 30 / 3600, 4), x_realtime=round(t['frames'] / 30 / max(elapsed, 1e-9), 1),
                   return_share=round(t['return_frames'] / max(t['frames'], 1), 3),
                   reset_ms=round(1000 * t['reset_s'] / max(t['resets'], 1), 1),
                   retries={k[6:]: t[k] for k in ('retry_match', 'retry_same', 'retry_other')},
                   return_fps=round(t['return_frames'] / max(t['return_s'], 1e-9)),
                   explore_fps=round(t['explore_frames'] / max(t['explore_s'], 1e-9)),
                   digest_ms=round(1000 * t['digest_s'] / max(t['digests'], 1), 2),
                   recycles=t['recycles'], active=rooms)
        with open(self.out / 'progress.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
        brief = ' '.join(f"{s}:{r.get('cells', '-')}c/{r.get('wins', '-')}w/{r.get('best', {}).get('hurt', '-')}d"
                         for s, r in rooms.items())
        print(f"[{rec['t']:.0f}s] done {self.finished} tasks {tasks} ({rec['tasks_per_s']}/s) fail {rec['fail']} "
              f"err {rec['error']} retry {rec['retries']} {rec['game_hours']} game h ({rec['x_realtime']}x) "
              f"reset {rec['reset_ms']} ms return {rec['return_fps']} f/s explore {rec['explore_fps']} f/s "
              f"digest {rec['digest_ms']} ms | {brief}", flush=True)

    def run(self):
        for i in range(self.args.workers):
            self.start_worker(i)
        try:
            while True:
                self.activate()
                if not self.rooms:
                    break
                while self.idle:
                    w = min(self.idle)
                    if not self.dispatch(w):
                        break
                if not self.inflight:
                    before = len(self.rooms)
                    self.finish_rooms()
                    if len(self.rooms) < before:
                        continue
                    # a room that is neither done nor able to hand out a task, with nothing in flight: a bug
                    self.log_event(dict(event='stalled', rooms=list(self.rooms)))
                    break
                try:
                    res = self.results.get(timeout=60)
                except queue.Empty:
                    self.check_workers()
                    self.progress()
                    continue
                self.handle(res)
                self.finish_rooms()
                self.progress()
        finally:
            try:
                self.progress(force=True)
            except Exception as exc:
                print(f'progress failed: {exc!r}', flush=True)
            self.stop_workers()

    def check_workers(self):
        for i, p in enumerate(self.workers):
            if p is not None and not p.is_alive():
                self.handle(dict(worker=i, kind='fatal', error=f'process exited ({p.exitcode})'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--tasks', help='an abplus_tasks spec file (mixture)')
    src.add_argument('--groups-file', help='a groups file; with --group')
    p.add_argument('--group', default='normal')
    p.add_argument('--seconds', type=float, default=None, help="episode deadline (default: the group's, else 180)")
    p.add_argument('--seeds', required=True, help='range:START:COUNT or a JSON list / {"seeds": [...]}')
    p.add_argument('--iterations', type=int, default=1000, help='explorations per room')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--rooms-at-once', type=int, default=4)
    p.add_argument('--out', required=True)
    p.add_argument('--resume', action='store_true', help='skip rooms already written to --out')
    p.add_argument('--name', default='gx')
    p.add_argument('--port', type=int, default=27600)
    p.add_argument('--rng-seed', type=int, default=0)
    p.add_argument('--progress-s', type=float, default=30.0)
    p.add_argument('--bridge-lua', default=None)
    p.add_argument('--software-gl', action='store_true', help='Mesa software OpenGL for the instances (B7)')
    for f, v in asdict(GxConfig()).items():
        if f == 'bridge_lua':
            continue
        flag = '--' + f.replace('_', '-')
        if isinstance(v, bool):
            p.add_argument(flag, action='store_true', default=v)
        else:
            p.add_argument(flag, type=type(v), default=v)
    args = p.parse_args()
    if not steam_running():
        raise SystemExit('the Steam client is not running (every AB+ start needs it)')
    if args.software_gl:
        from isaac_bridge.abplus import SOFTWARE_GL_ENV
        os.environ.update(SOFTWARE_GL_ENV)
    cfg = GxConfig(**{f: getattr(args, f) for f in asdict(GxConfig()) if f != 'bridge_lua'},
                   bridge_lua=args.bridge_lua or default_bridge_lua())
    spec = load_spec(args)
    seeds = parse_seeds(args.seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        seeds = [s for s in seeds if not (out / 'rooms' / f'{s}.json').exists()]
    config = dict(args=vars(args), gx=asdict(cfg), bridge=BRIDGE_VERSION, group=spec['name'], tasks=spec['tasks'],
                  seconds=spec['seconds'], target=bool(spec['target']), seeds=len(seeds),
                  started=time.strftime('%Y-%m-%d %H:%M:%S'), host=os.uname().nodename if hasattr(os, 'uname') else '')
    with open(out / 'config.json', 'a') as f:
        f.write(json.dumps(config) + '\n')
    print(json.dumps(dict(event='start', seeds=len(seeds), workers=args.workers, group=spec['name'],
                          bridge=BRIDGE_VERSION, lua=cfg.bridge_lua)), flush=True)
    manager = Manager(args, cfg, spec, seeds)

    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, signal.default_int_handler)   # started in the background: SIGINT arrives ignored
    try:
        manager.run()
    except KeyboardInterrupt:
        print('interrupted', flush=True)
        sys.exit(130)


if __name__ == '__main__':
    main()
