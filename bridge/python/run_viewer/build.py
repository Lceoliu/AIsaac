"""把 isaac-run-v1 录制文件（format 1 = v1，format 2 = v2 带外观字段）整理成查看器数据（isaac-run-viewer-v1，gzip 压缩的 JSON）。

只做"分拣与索引"，不在 Python 里把差分展开成完整行：
- frames：每个 f 行变成定长数组 [n, gf, t, kf, tk, r, p, e, g, dr, mt, dg, sp数, ms]，p/e 保持录制里的
  关键帧 + 差分形式（展开由 viewer_core.js 在浏览器里按关键帧段做，结果缓存）。全部展开成完整行约 78 万实体行，
  JSON 会到几十 MB；保持差分形式 gzip 后约 5 MB，浏览器展开一个关键帧段（≤300 帧）只要几毫秒。
- rooms / levels / items：每行附 at = 它生效的第一帧的数组下标（= 文件里它前面的 f 行数），与 load_run.py
  按"行号先后"取最近一行的口径相同。
- events：ev / hcmd / mark / api 行，按其 n 归帧（与 load_run 相同）。
- names / icons：可选名称表与道具图标（names.py）。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from load_run import SCHEMA, read_lines  # noqa: E402

from .names import load_names  # noqa: E402

VIEWER_SCHEMA = 'isaac-run-viewer-v1'
# frames 数组的列（viewer_core.js 里同名常量）
FRAME_COLS = ['n', 'gf', 't', 'kf', 'tk', 'r', 'p', 'e', 'g', 'dr', 'mt', 'dg', 'nsp', 'ms',
              'pa', 'ea', 'gs', 'ds']   # v2（header.format 2）：外观行与网格/门精灵变化；字符串字典另放 strings


def _float_cells(rows):
    """统计行里的 Python float（JSON 带小数点的数）。JS 里 1.0 与 1 无法区分，而差分规则对"整数"与"非整数"处理不同；
    录制的数值字段都是 ×100 定点整数，应为 0，否则在 stats 里报出来。"""
    c = 0
    for r in rows:
        if isinstance(r, list):
            for v in r:
                if type(v) is float:
                    c += 1
                elif isinstance(v, list):
                    c += _float_cells([v])
    return c


def build(path, res_dir=None, with_names=True, log=print):
    t0 = time.time()
    path = Path(path)
    repaired = []
    lines = read_lines(path, repaired)
    if not lines:
        raise ValueError('空文件')
    header = json.loads(lines[0])
    if header.get('k') != 'header' or header.get('schema') != SCHEMA:
        raise ValueError(f'不是 {SCHEMA} 文件')
    t1 = time.time()
    frames, rooms, levels, items, events, other = [], [], [], [], [], {}
    strings = {}   # v2 字符串字典 id -> 串（sd 条目，来自帧行与进房行）
    end = None
    float_cells = 0
    item_ids, trinket_ids = set(), set()
    for ln in lines[1:]:
        o = json.loads(ln)
        k = o.get('k')
        if 'sd' in o:
            for sid, sv in o['sd']:
                strings[sid] = sv
        if k == 'f':
            if o.get('err'):
                # recorder 0.5.0 的退化帧（观测构建出错，只有输入/MT）：画成空帧（没有玩家/实体），当关键帧处理（不从它向后
                # 差分；mod 保证下一帧本来就是关键帧），房间头沿用上一帧。不拿上一帧的玩家/实体冒充
                o = dict(o, kf=1, p=[], e=[], r=frames[-1][5] if frames else [0, 0, 0, -1])
            e = o.get('e')
            float_cells += _float_cells(o['p']) + (_float_cells(e) if e else 0)
            frames.append([o['n'], o['gf'], o['t'], 1 if o.get('kf') == 1 else 0, o['tk'], o['r'], o['p'],
                           e if 'e' in o else None, o.get('g'), o.get('dr'), o.get('mt'), o.get('dg'),
                           len(o.get('sp', [])), o.get('ms'), o.get('pa'), o.get('ea'), o.get('gs'), o.get('ds')])
            # 本局出现过的道具台 / 饰品（只用来决定嵌哪些图标）
            for row in (e or []):
                if isinstance(row, list) and len(row) > 3 and not isinstance(row[1], list) and row[1] == 5:
                    if row[2] == 100:
                        item_ids.add(row[3])
                    elif row[2] == 350:
                        trinket_ids.add(row[3] & 0x7fff)
        elif k in ('room', 'level', 'items'):
            d = dict(o)
            d.pop('sd', None)
            d['at'] = len(frames)
            {'room': rooms, 'level': levels, 'items': items}[k].append(d)
            if k == 'items':
                item_ids.update(it[0] for it in o['items'])
        elif k in ('ev', 'hcmd', 'mark', 'api'):
            # api：第三方 mod 的效果调用（recorder 0.4.0），与 load_run 一样进 events（查看器在时间轴上不画高频的那几种）
            events.append(o)
        elif k == 'end':
            end = o
        else:
            other[k] = other.get(k, 0) + 1
    for i, fr in enumerate(frames):
        if fr[0] != i + 1:
            raise ValueError(f'帧号不连续：第 {i} 个 f 行 n={fr[0]}')
    # 玩家主动/饰品槽里的道具与饰品也要图标（关键帧是完整行）
    ppl = header['fields']['player']
    ci = {nm: ppl.index(nm) for nm in ('active0', 'active1', 'active2', 'active3', 'trinket0', 'trinket1')}
    for fr in frames:
        if fr[3]:
            for row in fr[6]:
                if isinstance(row, list) and len(row) == len(ppl):
                    for nm in ('active0', 'active1', 'active2', 'active3'):
                        if isinstance(row[ci[nm]], int) and row[ci[nm]] > 0:
                            item_ids.add(row[ci[nm]])
                    for nm in ('trinket0', 'trinket1'):
                        if isinstance(row[ci[nm]], int) and row[ci[nm]] > 0:
                            trinket_ids.add(row[ci[nm]] & 0x7fff)
    t2 = time.time()
    names = load_names(res_dir, dict(c=item_ids, t=trinket_ids)) if with_names else None
    hdr = {k: v for k, v in header.items() if k != 'prestart_spawns'}
    hdr['prestart_spawns_count'] = len(header.get('prestart_spawns') or [])
    raw = path.read_bytes()
    data = dict(schema=VIEWER_SCHEMA, frame_cols=FRAME_COLS,
                source=dict(path=str(path), name=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                            lines=len(lines), repaired_lines=repaired),
                header=hdr, end=end, frames=frames, rooms=rooms, levels=levels, items=items, events=events,
                format=header.get('format', 1), strings={str(k): v for k, v in strings.items()},
                other_lines=other, names=names,
                stats=dict(float_cells=float_cells, parse_s=round(t1 - t0, 3), index_s=round(t2 - t1, 3)))
    blob = json.dumps(data, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    gz = gzip.compress(blob, compresslevel=6)
    t3 = time.time()
    log(f'[run_viewer] {path.name}: {len(frames)} 帧，{len(rooms)} 次进房，{len(levels)} 层，{len(events)} 个事件；'
        f'JSON {len(blob) / 1e6:.1f} MB -> gzip {len(gz) / 1e6:.1f} MB；用时 {t3 - t0:.1f} s'
        + (f'；警告：{float_cells} 个浮点格' if float_cells else ''))
    return gz, data
