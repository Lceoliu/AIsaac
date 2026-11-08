"""查看器正确性校验：用查看器自己的展开代码（static/viewer_core.js，经 node 运行）展开抽样帧，与
load_run.Run.frame(n) 直接展开的结果逐字段比对。

  python -m run_viewer.verify RUN.jsonl.zst [--samples 300] [--seed 1] [--report out.json]

抽样 = 随机帧 + 边界帧（第一帧、最后一帧、每次进房的第一帧及其前一帧、关键帧前后、有网格/门变化的帧、有事件的帧，各取一部分）。
比较字段：n/gf/t/关键帧/MT/digest、房间头 r、完整玩家行与实体行（整数原值）、解码后的玩家/实体字段（含附加字段与 category）、
当前房间行（不含 grid/doors）、网格、门、楼层（生效的 level 行与全部 RoomDescriptor）、道具清单、本帧事件、本帧全部 tick
（kind、12x3 输入状态、changed、extra）、鼠标流；v2 文件另比外观完整行 pa/ea 与解码后的外观字段（字符串列经字典还原）、
网格精灵与门精灵（进房行 + gs/ds 变化）。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from load_run import Run  # noqa: E402

from .build import build  # noqa: E402


def _strkeys(d):
    return {str(k): v for k, v in d.items()}


def _py_state(fs):
    room = fs.room or {}
    return dict(
        n=fs.n, gf=fs.gf, t=fs.t, keyframe=fs.keyframe, mt=fs.mt, digest=fs.digest,
        room_frame=fs.room_frame,
        player_rows=fs.player_rows, entity_rows=fs.entity_rows,
        players=[{k: v for k, v in p.items() if k != '_raw'} for p in fs.players],
        entities=None if fs.entities is None else [{k: v for k, v in e.items() if k != '_raw'} for e in fs.entities],
        room_head={k: v for k, v in room.items() if k not in ('grid', 'doors', 'grid_spr', 'door_spr', 'sd')} if fs.room else None,
        grid=_strkeys(room['grid']) if fs.room else None,
        doors=_strkeys(room['doors']) if fs.room else None,
        level_n=(fs.level or {}).get('n'), level_rooms=(fs.level or {}).get('rooms'),
        items={str(p): _strkeys(v) for p, v in fs.items.items()},
        events=fs.events,
        ticks=[dict(kind=t.kind, inputs=_strkeys(t.inputs), changed=t.changed, extra=t.extra) for t in fs.ticks],
        mouse=fs.mouse,
        # v2 外观（v1 文件全为 None）
        pa_rows=fs.pa_rows, ea_rows=fs.ea_rows,
        player_appearance=None if fs.player_appearance is None else
        [None if a is None else {k: v for k, v in a.items() if k != '_raw'} for a in fs.player_appearance],
        entity_appearance=None if fs.entity_appearance is None else
        [None if a is None else {k: v for k, v in a.items() if k != '_raw'} for a in fs.entity_appearance],
        grid_spr=_strkeys(room['grid_spr']) if fs.room and 'grid_spr' in room else None,
        door_spr=_strkeys(room['door_spr']) if fs.room and 'door_spr' in room else None,
    )


def _js_norm(s):
    s = dict(s)
    s['grid'] = None if s['grid'] is None else {str(g[0]): g for g in s['grid']}
    s['doors'] = None if s['doors'] is None else {str(d[0]): d for d in s['doors']}
    for k in ('grid_spr', 'door_spr'):
        s[k] = None if s.get(k) is None else {str(g[0]): g for g in s[k]}
    s['items'] = {p: _strkeys(v) for p, v in s['items'].items()}
    s['ticks'] = [dict(kind=t['kind'], inputs=_strkeys(t['inputs']), changed=t['changed'], extra=t['extra'])
                  for t in s['ticks']]
    return s


def _eq(a, b):
    """JSON 语义下的相等：int 与等值 float 视为不同（Python 侧），这里按值比较但区分 None。"""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b and type(a) is type(b)
    return a == b


def _diff_fields(py, js):
    """返回 [(字段路径, py 值, js 值)]：对 players/entities 下钻到每个字段，其余按顶层字段。"""
    out = []
    for k in py:
        a, b = py[k], js.get(k)
        if k in ('players', 'entities', 'player_appearance', 'entity_appearance') and isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                if x is None or y is None:
                    if not _eq(x, y):
                        out.append((f'{k}[{i}]', x, y))
                    continue
                for fk in set(x) | set(y):
                    if not _eq(x.get(fk), y.get(fk)):
                        out.append((f'{k}[{i}].{fk}', x.get(fk), y.get(fk)))
        elif not _eq(a, b):
            out.append((k, a, b))
    return out


def pick_frames(run, js_meta, samples, seed):
    rng = random.Random(seed)
    N = run.frames
    pick = set(rng.sample(range(N), min(samples, N)))
    pick.update({0, N - 1})
    edges = []
    for (_, n) in run.room_lines:
        edges += [n - 1, n - 2, n]          # 进房第一帧（下标 n-1）、前一帧、第二帧
    kf = run.keyframes
    for n in rng.sample(kf, min(40, len(kf))):
        edges += [n - 1, n - 2]
    edges += [n - 1 for n in rng.sample(run.change_frames, min(60, len(run.change_frames)))]
    edges += [n - 1 for n in rng.sample(sorted(run.event_lines), min(60, len(run.event_lines)))]
    rng.shuffle(edges)
    pick.update(i for i in edges[:max(60, samples // 2)] if 0 <= i < N)
    return sorted(pick)


def verify(path, samples=300, seed=1, report=None, log=print, node_dir=None):
    t0 = time.time()
    gz, data = build(path, log=log)
    run = Run(path)
    tmp = Path(tempfile.mkdtemp(prefix='run_viewer_verify_'))
    try:
        dp = tmp / 'data.json.gz'
        dp.write_bytes(gz)
        idx = pick_frames(run, data, samples, seed)
        (tmp / 'frames.json').write_text(json.dumps(idx))
        node = shutil.which('node')
        if not node:
            raise RuntimeError('找不到 node（校验需要用 node 运行 viewer_core.js）')
        t1 = time.time()
        cmd = [node, str(Path(node_dir or HERE / 'static') / 'node_dump.js'), str(dp), str(tmp / 'frames.json'),
               str(tmp / 'out.json')]
        node_retries = 0
        for attempt in range(3):
            if dp.read_bytes() != gz:          # 写盘后回读核对（一次见过 node 报 gzip "incorrect data check"）
                dp.write_bytes(gz)
            pr = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if pr.returncode == 0:
                break
            node_retries += 1
            log(f'[verify] node 失败（第 {attempt + 1} 次）：{pr.stderr.strip().splitlines()[-1] if pr.stderr.strip() else pr.returncode}')
        else:
            raise RuntimeError('node_dump.js 连续失败：' + pr.stderr[-2000:])
        t2 = time.time()
        js = json.loads((tmp / 'out.json').read_text(encoding='utf-8'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)   # 自己建的临时目录（只含本次生成的文件）
    res = dict(file=str(path), frames=run.frames, checked=0, frames_with_diff=0, field_diffs=0, by_field={},
               examples=[], node_timing_ms=js['timing_ms'], node_total_entity_rows=js['total_entity_rows'],
               float_cells=data['stats']['float_cells'], compared_fields=None)
    py_rows = 0
    for i in idx:
        fs = run.frame(i + 1)
        py = _py_state(fs)
        jj = _js_norm(js['frames'][str(i)])
        if res['compared_fields'] is None:
            res['compared_fields'] = sorted(py)
        py_rows += len(fs.entity_rows or [])
        diffs = _diff_fields(py, jj)
        res['checked'] += 1
        if diffs:
            res['frames_with_diff'] += 1
            res['field_diffs'] += len(diffs)
            for k, a, b in diffs:
                key = k.split('[')[0] + (k.split('].', 1)[1] if '].' in k else '')
                res['by_field'][key] = res['by_field'].get(key, 0) + 1
                if len(res['examples']) < 10:
                    res['examples'].append(dict(frame=i + 1, field=k, load_run=repr(a)[:200], viewer=repr(b)[:200]))
    res['checked_entity_rows'] = py_rows
    res['checked_player_fields'] = sum(len(run.player_names) for _ in idx)
    res['node_s'] = round(t2 - t1, 2)
    res['node_retries'] = node_retries
    res['total_s'] = round(time.time() - t0, 2)
    res['ok'] = res['field_diffs'] == 0 and res['float_cells'] == 0
    log(json.dumps(res, ensure_ascii=False, indent=1))
    if report:
        Path(report).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description='查看器展开结果与 load_run 逐字段比对')
    ap.add_argument('run')
    ap.add_argument('--samples', type=int, default=300)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--report')
    ap.add_argument('--node-dir', help='用另一份 static 目录里的 node_dump.js/viewer_core.js（变异测试用）')
    a = ap.parse_args(argv)
    return 0 if verify(a.run, a.samples, a.seed, a.report, node_dir=a.node_dir)['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
