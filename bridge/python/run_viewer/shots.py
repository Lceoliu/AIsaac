"""用无头 Chrome/Edge 给查看器指定帧截图（REPORT 的画面检查用）。

  python -m run_viewer.shots VIEWER.html OUT_DIR 3650 5300 14900 ... [--size 1600x950]

VIEWER.html 是 `python -m run_viewer RUN --html VIEWER.html` 生成的自包含页面；每帧开一次无头浏览器，
URL 带 #f=<帧号>，等页面加载（虚拟时间预算）后截图为 OUT_DIR/f<帧号>.png。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CANDIDATES = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
]


def find_browser():
    for c in CANDIDATES:
        if Path(c).is_file():
            return c
    for n in ('chrome', 'msedge', 'chromium'):
        p = shutil.which(n)
        if p:
            return p
    raise RuntimeError('找不到 Chrome/Edge')


def shot(html, out_png, frame, size='1600x950', budget_ms=30000):
    w, h = size.split('x')
    url = Path(html).resolve().as_uri() + f'#f={frame}'
    prof = tempfile.mkdtemp(prefix='rv_shot_')
    try:
        subprocess.run([find_browser(), '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--user-data-dir={prof}',
                        f'--window-size={w},{h}', f'--virtual-time-budget={budget_ms}', f'--screenshot={Path(out_png).resolve()}',
                        url], check=True, capture_output=True, timeout=180)
    finally:
        shutil.rmtree(prof, ignore_errors=True)   # 本次新建的临时浏览器配置目录


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('html')
    ap.add_argument('out')
    ap.add_argument('frames', nargs='+', type=int)
    ap.add_argument('--size', default='1600x950')
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for f in a.frames:
        p = Path(a.out) / f'f{f}.png'
        shot(a.html, p, f, a.size)
        print(p, p.stat().st_size if p.exists() else 'missing')
    return 0


if __name__ == '__main__':
    sys.exit(main())
