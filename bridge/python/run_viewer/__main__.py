"""整局录制的帧级回放查看器（不启动游戏，完全从录制文件重建每一帧）。

  python -m run_viewer RUN.jsonl.zst                 本地 http 服务（127.0.0.1，默认端口 8796）并打开浏览器
  python -m run_viewer RUN.jsonl.zst --port 0        随机端口
  python -m run_viewer RUN.jsonl.zst --no-browser    只起服务，不开浏览器
  python -m run_viewer RUN.jsonl.zst --html OUT.html 生成自包含 HTML（数据 gzip+base64 内嵌，可直接双击打开），不起服务
  python -m run_viewer.verify RUN.jsonl.zst          查看器展开结果与 load_run 逐字段比对（需要 node）

工作目录需是 rl/bridge/python（或把它加进 PYTHONPATH）。名称表/道具图标默认从 <仓库外层>/analysis/resources 读，
--resources DIR 指定，--no-names 不用。
真实美术模式（phase6-art）：素材默认从 <仓库外层>/experiments/run-recorder/assets/rep-extracted/resources 读（--assets DIR，
--no-art 关闭）；服务模式按需从磁盘只读提供（/asset/<键>），自包含 HTML 只内嵌本局用到的素材。实录画面层：若会话目录下有
footage/<run>.mp4 + <run>.json（schema isaac-footage-v1）自动启用（--footage MP4 指定），服务模式经 /footage.mp4（支持 Range）
提供，自包含 HTML 用 file:// 绝对路径引用（不内嵌视频）。
"""
from __future__ import annotations

import argparse
import base64
import gzip
import http.server
import json
import mimetypes
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

from .art import ArtIndex, find_footage, norm
from .build import build

STATIC = Path(__file__).resolve().parent / 'static'


def make_html(gz=None):
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    core = (STATIC / 'viewer_core.js').read_text(encoding='utf-8')
    art = (STATIC / 'viewer_art.js').read_text(encoding='utf-8')
    app = (STATIC / 'viewer_app.js').read_text(encoding='utf-8')
    for js in (core, art, app):
        if '</script' in js.lower():
            raise ValueError('脚本里不能出现 </script')
    data = base64.b64encode(gz).decode('ascii') if gz is not None else ''
    # 用 split/join 而不是 replace 链，避免脚本内容里恰好出现占位符
    head, rest = html.split('/*__DATA__*/', 1)
    mid, rest = rest.split('/*__CORE__*/', 1)
    mid2, rest = rest.split('/*__ART__*/', 1)
    mid3, tail = rest.split('/*__APP__*/', 1)
    return (head + data + mid + core + mid2 + art + mid3 + app + tail).encode('utf-8')


def attach_art(data, art_index, embed, footage_mp4, footage_meta, footage_url, log=print):
    """把美术查表（与可选的内嵌素材）、实录画面层的元数据放进查看器数据。返回附加信息（大小统计）。"""
    info = {}
    if art_index is not None and art_index.ok:
        tables = art_index.tables()
        anm2, pngs = art_index.used_assets(data)
        info.update(used_anm2=len(anm2), used_png=len(pngs))
        a = dict(tables=tables, base='asset/', assets=None, used=dict(anm2=len(anm2), png=len(pngs)))
        if embed:
            assets, total = art_index.embed(anm2 + pngs)
            a['assets'] = assets
            info.update(embedded_files=len(assets), embedded_raw_bytes=total)
        data['art'] = a
    if footage_meta is not None:
        data['footage'] = dict(footage_meta, url=footage_url, file=str(footage_mp4))
        info['footage'] = str(footage_mp4)
    return info


def regzip(data):
    blob = json.dumps(data, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return gzip.compress(blob, compresslevel=6), len(blob)


def serve(gz, port, open_browser, art_index=None, footage_mp4=None, log=print):
    class H(http.server.BaseHTTPRequestHandler):
        def _send(self, body, ctype, code=200, extra=None):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store' if ctype.startswith('text/html') else 'max-age=3600')
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            from urllib.parse import unquote
            p = unquote(self.path.split('?', 1)[0].split('#', 1)[0])
            if p in ('/', '/index.html'):
                return self._send(make_html(None), 'text/html; charset=utf-8')   # 每次现读 static/，改界面不用重启
            if p == '/data.json.gz':
                return self._send(gz, 'application/octet-stream')
            if p.startswith('/asset/') and art_index is not None:
                b = art_index.read_bytes(norm(p[len('/asset/'):]))   # 只读；键经 norm，出不了素材根目录
                if b is None:
                    return self.send_error(404)
                ctype = 'image/png' if p.endswith('.png') else 'text/xml; charset=utf-8'
                return self._send(b, ctype)
            if p == '/footage.mp4' and footage_mp4 is not None:
                size = footage_mp4.stat().st_size
                rng = self.headers.get('Range')
                start, end = 0, size - 1
                if rng and rng.startswith('bytes='):
                    a, _, b = rng[6:].partition('-')
                    start = int(a) if a else max(0, size - int(b))
                    end = int(b) if (a and b) else size - 1
                end = min(end, size - 1, start + (8 << 20) - 1)   # 每次最多 8 MB
                with open(footage_mp4, 'rb') as fh:
                    fh.seek(start)
                    body = fh.read(end - start + 1)
                return self._send(body, 'video/mp4', 206 if rng else 200,
                                  {'Accept-Ranges': 'bytes', 'Content-Range': f'bytes {start}-{end}/{size}'} if rng else
                                  {'Accept-Ranges': 'bytes'})
            self.send_error(404)

        def log_message(self, fmt, *args):
            pass

        def handle(self):
            try:
                super().handle()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass   # 浏览器 seek 视频时会主动断开旧的 Range 请求

    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = False   # Windows 上 SO_REUSEADDR 会让两个进程同时占同一端口

    srv = S(('127.0.0.1', port), H)
    url = f'http://127.0.0.1:{srv.server_address[1]}/'
    log(f'[run_viewer] 查看器：{url}   （Ctrl+C 退出）')
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m run_viewer', description='整局录制的帧级回放查看器')
    ap.add_argument('run', help='run-*.jsonl 或 run-*.jsonl.zst')
    ap.add_argument('--port', type=int, default=8796)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--html', help='生成自包含 HTML 到此路径（不起服务）')
    ap.add_argument('--data-out', help='另把查看器数据（gzip JSON）写到此路径')
    ap.add_argument('--resources', help='analysis/resources 目录（名称表与道具图标）')
    ap.add_argument('--no-names', action='store_true')
    ap.add_argument('--assets', help='解包素材根目录（含 entities2.xml 的 resources 目录）')
    ap.add_argument('--no-art', action='store_true', help='不带真实美术模式（只有几何模式）')
    ap.add_argument('--footage', help='实录视频 MP4（同名 .json 为 isaac-footage-v1 元数据）；默认找 <会话>/footage/<run>.mp4')
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    gz, data = build(a.run, res_dir=a.resources, with_names=not a.no_names)
    art_index = None if a.no_art else ArtIndex(a.assets)
    if art_index is not None and not art_index.ok:
        print(f'[run_viewer] 没找到解包素材（{art_index.root}），只有几何模式')
        art_index = None
    mp4, meta = find_footage(a.run, a.footage)
    furl = None
    if mp4 is not None:
        furl = mp4.resolve().as_uri() if a.html else 'footage.mp4'
        print(f'[run_viewer] 实录画面：{mp4}（{meta.get("frames")} 帧，first_frame={meta.get("first_frame")}）')
    info = attach_art(data, art_index, bool(a.html), mp4, meta, furl)
    if info:
        gz, nblob = regzip(data)
        print(f'[run_viewer] 美术/实录：{info}；数据 JSON {nblob / 1e6:.1f} MB -> gzip {len(gz) / 1e6:.1f} MB')
    if a.data_out:
        Path(a.data_out).write_bytes(gz)
    if a.html:
        out = Path(a.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(make_html(gz))
        print(f'[run_viewer] 已写 {out}（{out.stat().st_size / 1e6:.1f} MB）')
        return 0
    serve(gz, a.port, not a.no_browser, art_index=art_index, footage_mp4=mp4)
    return 0


if __name__ == '__main__':
    sys.exit(main())
