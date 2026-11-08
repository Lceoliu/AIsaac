"""查看器加载与拖动性能测量（无头 Chrome，经 Playwright 驱动；用系统 Chrome，channel="chrome"）。

  python -m run_viewer.bench VIEWER.html [--report out.json] [--size 1600x950]

测量：
- 加载：页面里 window.__rv.timing（读数据、gunzip、JSON.parse、建索引、首帧渲染、总计），以及从 goto 到 ready 的墙钟；
- 随机跳帧 500 次（每次完整渲染：房间画面 + 面板 + 两条进度条）；
- 整局扫动：沿全局进度条从头到尾 1500 步（相当于把进度条从左拖到右）；
- 顺序播放 900 帧；
- 真实鼠标拖动：在全局进度条上按下、分 300 步拖到右端、松开，统计期间每次渲染耗时（RV.renderMs）。
渲染耗时都是同步 JS 时间（不含合成/上屏），单位毫秒。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

STATS = """(a) => { a = a.slice().sort((x, y) => x - y); const q = (p) => +a[Math.min(a.length - 1, Math.floor(a.length * p))].toFixed(2);
  return {n: a.length, mean: +(a.reduce((s, x) => s + x, 0) / a.length).toFixed(2), p50: q(0.5), p95: q(0.95), p99: q(0.99), max: q(1)}; }"""


def bench(html, size='1600x950', log=print):
    from playwright.sync_api import sync_playwright
    w, h = (int(x) for x in size.split('x'))
    url = Path(html).resolve().as_uri() + '#f=1'
    out = {}
    with sync_playwright() as p:
        br = p.chromium.launch(channel='chrome', headless=True)
        pg = br.new_page(viewport=dict(width=w, height=h))
        t0 = time.time()
        pg.goto(url)
        pg.wait_for_function('window.__rv && (window.__rv.ready || window.__rv.error)', timeout=120000)
        out['wall_load_s'] = round(time.time() - t0, 2)
        out['error'] = pg.evaluate('window.__rv.error || null')
        out['timing'] = pg.evaluate('window.__rv.timing')
        out['heap_mb'] = pg.evaluate('performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null')
        pg.evaluate('() => { window.__st = ' + STATS + '; }')
        out['random_jump_ms'] = pg.evaluate("""() => { const R = window.__rv, N = R.run.N, a = [];
            for (let k = 0; k < 500; k++) a.push(R.setFrame(Math.floor(Math.random() * N))); return __st(a); }""")
        out['sweep_whole_run_ms'] = pg.evaluate("""() => { const R = window.__rv, N = R.run.N, a = [];
            for (let k = 0; k < 1500; k++) a.push(R.setFrame(Math.floor(k / 1499 * (N - 1)))); return __st(a); }""")
        out['sequential_900_ms'] = pg.evaluate("""() => { const R = window.__rv, a = [];
            for (let k = 0; k < 900; k++) a.push(R.setFrame(30000 + k)); return __st(a); }""")
        # 真实鼠标拖动（经 pointer 事件 -> rAF 渲染）
        box = pg.locator('#tl').bounding_box()
        pg.evaluate('window.__rv.renderMs = []')
        y = box['y'] + 20
        pg.mouse.move(box['x'] + 2, y)
        pg.mouse.down()
        t1 = time.time()
        for k in range(300):
            pg.mouse.move(box['x'] + 2 + (box['width'] - 4) * (k + 1) / 300, y)
            pg.wait_for_timeout(16)
        pg.mouse.up()
        drag_s = time.time() - t1
        pg.wait_for_timeout(200)
        out['mouse_drag'] = pg.evaluate('__st(window.__rv.renderMs)')
        out['mouse_drag']['wall_s'] = round(drag_s, 2)
        out['mouse_drag']['final_frame'] = pg.evaluate('window.__rv.cur + 1')
        out['heap_mb_after'] = pg.evaluate('performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null')
        br.close()
    log(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('html')
    ap.add_argument('--report')
    ap.add_argument('--size', default='1600x950')
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    r = bench(a.html, a.size)
    if a.report:
        Path(a.report).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
    return 0 if not r.get('error') else 1


if __name__ == '__main__':
    sys.exit(main())
