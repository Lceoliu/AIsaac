"""Live dashboard of the AB+ training runs, in the style of Weights & Biases.

Serves one page (isaac_bridge/abplus_dashboard.html) and a read-only JSON API over the run
directories of --train-dir (~/isaac-abplus/train/<run>/): progress.csv (the SB3 log), config.json,
episodes.jsonl and evaluations*/<checkpoint>/[sampled/]results.jsonl, plus an evaluation's
replays.html. Run and checkpoint names must be plain directory names under the train directory;
nothing else is read and nothing is written.

  GET /                          the page
  GET /api/runs                  every run: state, progress, key numbers, config summary
  GET /api/run?run=R&since=N     progress.csv columns from row N (the page appends)
  GET /api/episodes?run=R&tail=N the last N episodes and their summary
  GET /api/evals?run=R           per evaluation checkpoint: greedy and sampled summaries
  GET /api/system                GPU, CPU, memory, disk, Steam client
  GET /replay?run=R&eval=G/C     that evaluation's replays.html

usage: python abplus_dashboard.py [--host 100.76.185.120] [--port 8790] [--train-dir ~/isaac-abplus/train]
"""
import argparse
import csv
import gzip
import io
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
PAGE = HERE / 'isaac_bridge' / 'abplus_dashboard.html'
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
FRAMES_PER_DECISION, GAME_FPS = 2, 30


def number(text):
    if text is None or text == '':
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_progress(path):
    """SB3's progress.csv as {header, n, columns: {key: [float | None]}}; a partial last line is left out."""
    text = path.read_text(encoding='utf8', errors='replace')
    if text and not text.endswith('\n'):
        text = text[:text.rfind('\n') + 1]
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return dict(header=[], n=0, columns={})
    header, body = rows[0], rows[1:]
    columns = {key: [] for key in header}
    for row in body:
        for i, key in enumerate(header):
            columns[key].append(number(row[i]) if i < len(row) else None)
    columns = {k: v for k, v in columns.items() if any(x is not None for x in v)}
    if 'game/hours_this_run' not in columns and 'time/total_timesteps' in columns:
        columns['game/hours_this_run'] = [None if t is None else t * FRAMES_PER_DECISION / GAME_FPS / 3600
                                         for t in columns['time/total_timesteps']]
    return dict(header=header, n=len(body), columns=columns)


def parse_config(path):
    c = json.loads(path.read_text(encoding='utf8'))
    spec = c.get('tasks_spec') or {}
    model = c.get('model') if isinstance(c.get('model'), dict) else {}
    sampling = c.get('room_sampling')
    return dict(
        reward_profile=c.get('reward_profile'), game_hours=c.get('game_hours'),
        episode_seconds=c.get('max_episode_seconds'), rooms=len(spec.get('normal') or ()),
        boss_rooms=len(spec.get('boss') or ()), weights=spec.get('weights'), target=c.get('target'),
        aux=model.get('aux_head'), lineage_mode=c.get('lineage_mode'), miss_cap=c.get('miss_cap'),
        invincible=c.get('invincible'), start_bombs=c.get('start_bombs'),
        room_sampling=sampling.get('method') if isinstance(sampling, dict) else sampling,
        ent_coef=c.get('ent_coef'), gamma=c.get('gamma'), learning_rate=c.get('learning_rate'), envs=c.get('envs'),
        tasks_file=os.path.basename(c.get('tasks_file') or ''), transport=c.get('observation_transport'),
        warm_start=c.get('warm_start'), resume=c.get('resume'), async_train=c.get('async_train'),
        reward=c.get('reward'), reward_options=c.get('reward_options'))


def eval_summary(path):
    results = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
    wins = [r for r in results if r.get('outcome') == 'win']
    shots = sum((r.get('stats') or {}).get('shots', 0) for r in results)
    hits = sum((r.get('stats') or {}).get('tear_hits', 0) for r in results)
    seconds = sum(r.get('frames', 0) for r in results) / GAME_FPS
    return dict(n=len(results), wins=len(wins),
                timeouts=sum(r.get('outcome') == 'time_limit' for r in results),
                deaths=sum(r.get('outcome') == 'death' for r in results),
                median_s=statistics.median([r['frames'] / GAME_FPS for r in wins]) if wins else None,
                accuracy=hits / shots if shots else None, shots_per_s=shots / seconds if seconds else None)


def tail_lines(path, n):
    with open(path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        data = b''
        while size > 0 and data.count(b'\n') <= n:
            step = min(1 << 16, size)
            size -= step
            f.seek(size)
            data = f.read(step) + data
    if data and not data.endswith(b'\n'):
        data = data[:data.rfind(b'\n') + 1]
    return [line for line in data.split(b'\n') if line.strip()][-n:]


class Store:
    """Files parsed once per (mtime, size); the process table and system numbers for a few seconds."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.lock = threading.Lock()
        self.files = {}
        self.timed = {}

    def cached(self, kind, path, build):
        try:
            st = path.stat()
        except FileNotFoundError:
            return None
        stamp = (st.st_mtime_ns, st.st_size)
        key = (kind, str(path))
        with self.lock:
            hit = self.files.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
        value = build(path)
        with self.lock:
            self.files[key] = (stamp, value)
        return value

    def recent(self, key, seconds, build):
        now = time.monotonic()
        with self.lock:
            hit = self.timed.get(key)
        if hit and now - hit[0] < seconds:
            return hit[1]
        value = build()
        with self.lock:
            self.timed[key] = (now, value)
        return value

    def run_dir(self, name):
        if not name or not NAME.match(name):
            raise LookupError(f'unknown run {name!r}')
        d = (self.root / name).resolve()
        if d.parent != self.root or not (d / 'progress.csv').is_file():
            raise LookupError(f'unknown run {name!r}')
        return d

    # -------------------------------------------------------------------------------- runs
    def training_processes(self):
        """Run directory names of the train_abplus.py processes alive now (their --out)."""
        def scan():
            names = set()
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    args = (proc / 'cmdline').read_bytes().split(b'\0')
                except OSError:
                    continue
                if not any(a.endswith(b'train_abplus.py') for a in args) or b'--train' not in args:
                    continue
                for i, a in enumerate(args[:-1]):
                    if a == b'--out':
                        names.add(os.path.basename(os.path.normpath(args[i + 1].decode('utf8', 'replace'))))
            return names
        return self.recent('processes', 5.0, scan)

    def run_summary(self, d):
        progress = self.cached('progress', d / 'progress.csv', parse_progress) or dict(n=0, columns={})
        config = self.cached('config', d / 'config.json', parse_config) or {}
        columns = progress['columns']

        def last(key):
            return next((v for v in reversed(columns.get(key, ())) if v is not None), None)
        updated = (d / 'progress.csv').stat().st_mtime
        if d.name in self.training_processes():
            state = 'running'
        elif any(p.name.endswith('-final') for p in (d / 'checkpoints').glob('*')):
            state = 'finished'
        else:
            state = 'stopped'
        hours, target = last('game/hours_this_run'), config.get('game_hours')
        eta = None
        elapsed = columns.get('time/time_elapsed') or []
        hour_list = columns.get('game/hours_this_run') or []
        pairs = [(t, h) for t, h in zip(elapsed, hour_list) if t is not None and h is not None]
        if state == 'running' and target and len(pairs) >= 3:
            (t0, h0), (t1, h1) = pairs[max(0, len(pairs) - 6)], pairs[-1]
            if h1 > h0 and t1 > t0:
                eta = max(0.0, (target - h1) * (t1 - t0) / (h1 - h0))
        started = (d / 'config.json').stat().st_mtime if (d / 'config.json').exists() else None
        return dict(name=d.name, state=state, updated=updated, started=started, rows=progress['n'],
                    game_hours=hours, target_hours=target, eta_s=eta, speed=last('game/speed_x_realtime'),
                    updates=last('time/iterations'), win_rate=last('task/normal/win_rate'),
                    hit_rate=last('behavior/hit_rate'), errors=last('abplus/worker_errors'), config=config)

    def runs(self):
        out = []
        for d in self.root.iterdir():
            if d.is_dir() and NAME.match(d.name) and (d / 'progress.csv').is_file():
                try:
                    out.append(self.run_summary(d))
                except (OSError, ValueError) as error:
                    out.append(dict(name=d.name, state='error', error=str(error), updated=0))
        out.sort(key=lambda r: r.get('updated') or 0, reverse=True)
        return out

    def progress(self, name, since):
        progress = self.cached('progress', self.run_dir(name) / 'progress.csv', parse_progress)
        since = max(0, min(int(since), progress['n']))
        return dict(run=name, n=progress['n'], since=since, header=progress['header'],
                    columns={k: v[since:] for k, v in progress['columns'].items()})

    def episodes(self, name, tail):
        path = self.run_dir(name) / 'episodes.jsonl'
        if not path.exists():
            return dict(run=name, n=0, summary=None, rows=[])
        episodes = []
        for line in tail_lines(path, tail):
            try:
                episodes.append(json.loads(line))
            except ValueError:
                continue
        wins = [e for e in episodes if e.get('outcome') == 'win']
        shots = sum((e.get('stats') or {}).get('shots', 0) for e in episodes)
        hits = sum((e.get('stats') or {}).get('tear_hits', 0) for e in episodes)
        seconds = sum(e.get('frames', 0) for e in episodes) / GAME_FPS
        clear = sorted(e['frames'] / GAME_FPS for e in wins)
        summary = dict(n=len(episodes), wins=len(wins),
                       timeouts=sum(e.get('outcome') == 'time_limit' for e in episodes),
                       deaths=sum(e.get('outcome') == 'death' for e in episodes),
                       median_s=statistics.median(clear) if clear else None,
                       p90_s=clear[min(len(clear) - 1, int(0.9 * len(clear)))] if clear else None,
                       accuracy=hits / shots if shots else None, shots_per_s=shots / seconds if seconds else None,
                       mean_return=statistics.fmean([e.get('r', 0.0) for e in episodes]) if episodes else None)
        rows = [dict(episode=e.get('episode'), room=e.get('layout'), outcome=e.get('outcome'),
                     seconds=round(e.get('frames', 0) / GAME_FPS, 1), shots=(e.get('stats') or {}).get('shots'),
                     hits=(e.get('stats') or {}).get('tear_hits'), r=e.get('r'),
                     components=e.get('reward_components')) for e in episodes[-40:]]
        return dict(run=name, n=len(episodes), summary=summary, rows=rows[::-1])

    def evals(self, name):
        d = self.run_dir(name)
        out = []
        for group in sorted(p for p in d.iterdir() if p.is_dir() and p.name.startswith('evaluations')):
            for ckpt in sorted(p for p in group.iterdir() if p.is_dir() and NAME.match(p.name)):
                match = re.match(r'update-(\d+)', ckpt.name)
                item = dict(group=group.name, checkpoint=ckpt.name, update=int(match.group(1)) if match else None,
                            replays=(ckpt / 'replays.html').exists())
                for mode, sub in (('greedy', ckpt), ('sampled', ckpt / 'sampled')):
                    results = sub / 'results.jsonl'
                    item[mode] = self.cached('eval', results, eval_summary) if results.exists() else None
                if item['greedy'] or item['sampled']:
                    out.append(item)
        return dict(run=name, evals=out)

    def replay_path(self, name, ref):
        d = self.run_dir(name)
        group, _, ckpt = (ref or '').partition('/')
        if not (group.startswith('evaluations') and NAME.match(group) and NAME.match(ckpt)):
            raise LookupError(f'unknown evaluation {ref!r}')
        path = (d / group / ckpt / 'replays.html').resolve()
        if path.parent.parent.parent != d or not path.is_file():
            raise LookupError(f'no replays for {ref!r}')
        return path

    # ------------------------------------------------------------------------------ system
    def system(self):
        def build():
            info = dict(time=time.time(), cpus=os.cpu_count())
            try:
                out = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3).stdout
                name, util, used, total, temp = [x.strip() for x in out.splitlines()[0].split(',')]
                info['gpu'] = dict(name=name, util=float(util), mem_used=float(used), mem_total=float(total), temp=float(temp))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                info['gpu'] = None
            try:
                info['load'] = [float(x) for x in Path('/proc/loadavg').read_text().split()[:3]]
                mem = {line.split(':')[0]: float(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.split(':')[0] in ('MemTotal', 'MemAvailable')}
                info['mem'] = dict(total_gb=mem['MemTotal'] / 2 ** 20, available_gb=mem['MemAvailable'] / 2 ** 20)
            except (OSError, KeyError, ValueError):
                info['load'] = info['mem'] = None
            disk = shutil.disk_usage(self.root)
            info['disk'] = dict(free_gb=disk.free / 2 ** 30, total_gb=disk.total / 2 ** 30)
            steam = False
            for proc in Path('/proc').iterdir():
                if proc.name.isdigit():
                    try:
                        if (proc / 'comm').read_text().strip() == 'steam':
                            steam = True
                            break
                    except OSError:
                        continue
            info['steam'] = steam
            info['training'] = sorted(self.training_processes())
            return info
        return self.recent('system', 5.0, build)


class Handler(BaseHTTPRequestHandler):
    server_version = 'abplus-dashboard/1'
    store = None

    def log_message(self, fmt, *args):
        pass

    def send_body(self, body, content_type, status=200):
        if len(body) > 32768 and 'gzip' in (self.headers.get('Accept-Encoding') or ''):
            body = gzip.compress(body, 5)
            encoded = True
        else:
            encoded = False
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        if encoded:
            self.send_header('Content-Encoding', 'gzip')
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status=200):
        self.send_body(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf8'),
                       'application/json; charset=utf-8', status)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        s = self.store
        try:
            if url.path in ('/', '/index.html'):
                self.send_body(PAGE.read_bytes(), 'text/html; charset=utf-8')
            elif url.path == '/api/runs':
                self.send_json(s.runs())
            elif url.path == '/api/run':
                self.send_json(s.progress(q.get('run'), q.get('since', 0)))
            elif url.path == '/api/episodes':
                self.send_json(s.episodes(q.get('run'), max(1, min(2000, int(q.get('tail', 200))))))
            elif url.path == '/api/evals':
                self.send_json(s.evals(q.get('run')))
            elif url.path == '/api/system':
                self.send_json(s.system())
            elif url.path == '/replay':
                self.send_body(s.replay_path(q.get('run'), q.get('eval')).read_bytes(), 'text/html; charset=utf-8')
            else:
                self.send_json(dict(error='not found'), 404)
        except LookupError as error:
            self.send_json(dict(error=str(error)), 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:   # the dashboard must keep serving whatever one request hits
            self.send_json(dict(error=f'{type(error).__name__}: {error}'), 500)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--host', default='100.76.185.120', help='address to listen on (the Tailscale one by default)')
    p.add_argument('--port', type=int, default=8790)
    p.add_argument('--train-dir', default=str(Path.home() / 'isaac-abplus' / 'train'))
    args = p.parse_args()
    Handler.store = Store(args.train_dir)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f'dashboard on http://{args.host}:{args.port}/ over {Handler.store.root}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
