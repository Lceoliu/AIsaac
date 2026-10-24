"""Local web server for the floor generators: serves web/static and the JSON API of web/api.py.

    python rl/macro/web/server.py [--port 8792] [--host 127.0.0.1]

It reads each game's afterbirthp.a (read only); nothing is uploaded anywhere. The same API runs in
the browser on the static site (tools/build_site.py), so the page works with either.

  GET /api/run?game=abplus|repplus&seed=DXNH%20NZLG&mode=debug|normal&last=8|11&route=sheol|cathedral
              &coins=0&keys=0&hearts=6&max_hearts=6&soul=0
  GET /api/layout?game=abplus|repplus&stage=<room file id>&type=<room type>&variant=<variant>
  GET /api/sprites?game=abplus|repplus
  GET /api/random
  GET /art/...      the room layouts' game art (web/roomart.py), built into rl/runs/macro/art once
  GET /config.js    the page's settings; with rl/runs/macro/supabase.json the scoreboard is shared
"""
from __future__ import annotations

import argparse
import json
import posixpath
import sys
import threading
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

import api  # noqa: E402
import roomart  # noqa: E402

STATIC = Path(__file__).resolve().parent / 'static'
SUPABASE = Path(__file__).resolve().parents[2] / 'runs' / 'macro' / 'supabase.json'
_art = {'dir': None}
_art_lock = threading.Lock()


def art_dir() -> Path:
    """The game art for room layouts, built on first use (requests wait for it)."""
    with _art_lock:
        if _art['dir'] is None:
            try:
                _art['dir'] = roomart.ensure(roomart.DEFAULT_OUT, log=lambda m: print(f'art: {m}', flush=True))
            except (FileNotFoundError, OSError, ValueError) as exc:
                print(f'art: not available ({exc}); layouts are drawn as diagrams', flush=True)
                _art['dir'] = roomart.DEFAULT_OUT / 'missing'
    return _art['dir']


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write('%s %s\n' % (self.address_string(), fmt % args))

    def translate_path(self, path):
        url = unquote(urlparse(path).path)
        if url.startswith('/art/'):
            rel = posixpath.normpath(url[len('/art/'):])
            if rel.startswith(('..', '/')):
                return str(STATIC / 'missing')
            return str(art_dir() / rel)
        return super().translate_path(path)

    def end_headers(self):
        if not self.path.startswith('/api/'):
            self.send_header('Cache-Control', 'no-cache')    # always revalidate the page's own files
        super().end_headers()

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _config(self) -> None:
        cfg = dict(backend='server')
        try:
            sb = json.loads(SUPABASE.read_text(encoding='utf-8'))
            if sb.get('url') and sb.get('key'):
                cfg['supabase'] = dict(url=sb['url'].rstrip('/'), key=sb['key'])
        except (OSError, ValueError):
            pass
        body = f'window.MAPGEN_CONFIG = {json.dumps(cfg, ensure_ascii=False)};\n'.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/javascript; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/config.js':
            return self._config()
        if not url.path.startswith('/api/'):
            return super().do_GET()
        method = url.path[len('/api/'):]
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if method not in api.METHODS:
                return self._json(404, dict(error='未知接口'))
            return self._json(200, api.METHODS[method](params))
        except ConnectionError:   # the page went away (reload, new request) before the answer was sent
            return None
        except FileNotFoundError as exc:
            return self._json(500, dict(error=f'找不到游戏资源文件：{exc}'))
        except (ValueError, KeyError) as exc:
            return self._json(400, dict(error=str(exc)))
        except NotImplementedError as exc:
            return self._json(400, dict(error=f'暂不支持：{exc}'))
        except Exception as exc:  # pragma: no cover - surfaced to the page for debugging
            traceback.print_exc()
            return self._json(500, dict(error=f'{type(exc).__name__}: {exc}'))


def main() -> None:
    ap = argparse.ArgumentParser(description='Isaac floor generator web UI (AB+ and Repentance+)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8792)
    args = ap.parse_args()
    for game in api.GAMES.values():
        try:
            game.room_config()
        except FileNotFoundError as exc:
            print(f'{game.title}: game files not found ({exc}); its requests will fail', flush=True)
    threading.Thread(target=art_dir, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Isaac floor generator UI: http://{args.host}:{args.port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
