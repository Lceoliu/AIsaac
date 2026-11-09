"""回放一致性检查（回归测试；DESIGN_BRIEF §6 D1、§7 E6）。

  python replay_run.py RUN.jsonl[.zst] [--name NAME] [--out-root DIR] [--report REPORT.json]
  python replay_run.py RUN --continue-save SNAPSHOT_DIR   续局回放：先把存档快照（含 gamestate1.dat）拷进副本存档目录
  python replay_run.py RUN --compare REPLAY_RUN           只比较两个已有文件
  python replay_run.py RUN --turbo [gate|clock] [--copy COPY] [--init-save SNAPSHOT_DIR]
      --turbo：加速回放（默认关 = 原速）。rr_mt.dll 在 RR_MT_TURBO=1 时装虚拟时钟（rl/bridge/engine/j746_rec/rr_turbo.cpp）；
               gate（--turbo 不带值时的默认）另外不提交 GPU 绘制/SwapBuffers（CPU 渲染照跑，窗口不刷新），clock 只换时钟。
               比较口径不变（三层 digest + MT；帧行里的 t/os 时间字段不参与比较）。实测见 experiments/run-recorder/phase4-turbo/。
      --copy：隔离副本根目录（默认 game-rec-01）。
      --init-save：非续局回放前把录制会话的存档快照（save/run-...）拷进副本存档目录（不拷 gamestate1.dat），
                   使解锁/道具池与录制开局时相同。
  python replay_run.py RUN --watch --copy <game-rec-03> --init-save SNAPSHOT_DIR        观战模式（README §8.3）
  python replay_run.py RUN --capture OUT.mp4 --copy <game-rec-03> --init-save SNAPSHOT_DIR   实录导出（README §8.4）

流程：新会话目录 → 新进程（默认经注入器带 rr_mt.dll，存档在副本 profile/）→ 进菜单 → harness 执行
`seed <录制种子>` 开局（续局回放：菜单 Enter 续上快照里的局）→ mod 的 replay 模式在录制 header.mt.reseeds 的同一时机
用同一个 salt reseed 全局 MT（StartSeed xor salt）、在 MC_INPUT_ACTION 按 (帧, tick) 注入录制的输入、在
MC_PRE_ENTITY_SPAWN 按 "(gf, type.variant.subtype, 帧内第几次)" 钉回录制的生成种子、在同一帧重放 harness 的脚本命令 →
回放局本身也被完整录下 → 逐帧按数值比对（不比字符串）：
  mt           全局 MT 的 (cursor, hash)（帧行 mt；带 DLL 的录制才有）
  player_room  玩家与房间（digest[0]：Game/Room 帧计数、层、房间号、三种房间种子、清理状态 + 玩家全部数值字段的精确浮点位）
  noncosmetic  + 非外观实体（digest[1]：非 1000.x 实体的类型/位置/速度/HP/NPC State/StateFrame/FrameCount）
  full         全量（digest[2]：+ 特效、实体种子/Index、各类型附加字段、网格、门）
三层口径与第一阶段 divergence_views.py 的 player_room / gameplay_no_grid / full 对应。
输出每层首个分歧帧、分歧帧数，以及首个分歧帧上具体哪些字段不同（按 0.01 定点解码比较）。
带 DLL 且录制有 MT reseed 时期望全部 0 分歧（exact=true）；录制没有 MT（-NoDll 会话）时只能近似（INVESTIGATION §4）。
MouseControl=1 的局（鼠标射击绕过 MC_INPUT_ACTION）、非默认难度/挑战/角色的局列为不可完整回放。

菜单暂停（experiments/run-recorder/phase5-pause/REPORT.md）：录制里真人按 Esc 开暂停菜单的帧（同一帧 pause -> unpause、
开菜单的 tick 有 TRIGGERED MENUBACK/PAUSE=1），回放默认也在同一 (帧, tick) 由 mod 注入 MENUBACK 打开菜单，暂停
MENU_CLOSE_AFTER 个渲染后由本进程发一次真实 Esc 关闭，解除暂停后的 tick 按"相对解除暂停"对齐（暂停长度不影响结果）。
`--no-menu-pause` 关掉（旧行为：回放不暂停，开菜单落在 i2 的帧起分歧）。引擎自己的暂停（换房/弹窗/换层）不注入，但解除暂停后的
tick 同样相对对齐（`--no-pause-align` 关掉）。compare 的 pause_report 列出录制里的菜单暂停、
回放是否也暂停了，首个分歧帧若是回放没有复现的菜单暂停帧，归为已知类别 menu_pause_not_replayed。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_run  # noqa: E402
from isaac_bridge import run_recording as rr  # noqa: E402

LAYERS = ['player_room', 'noncosmetic', 'full']
DEFAULT_OUT = rr.ISAAC_ROOT / 'experiments/run-recorder/sessions/_replays'


# ---------------------------------------------------------------------------------------------- 回放表
def lua_value(v):
    if v is None:
        return 'nil'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'
    raise TypeError(type(v))


def mt_salts(header):
    """录制 header.mt.reseeds 里各时机（POST_PLAYER_INIT / POST_GAME_STARTED）的 salt；没有 MT 时返回 {}。"""
    mt = header.get('mt') or {}
    out = {}
    if not mt.get('available'):
        return out
    for r in mt.get('reseeds') or []:
        if r.get('at') and r.get('salt') is not None and not r.get('stale'):
            out[r['at']] = r['salt']
    return out


def record_start_paused(run: load_run.Run):
    """录制开局时游戏是否暂停（phase12-human4，README §8.9）。返回 (值, 来源)：
    recorder 0.4.2 起 header.mt.reseeds[0].paused（第一次 reseed 当时的 game:IsPaused()）→ ('header')；
    更早的录制按事件推断：本局第一条 pause/unpause 事件是 unpause（没有 pause 就先解除了暂停 = 开局时已暂停，典型是
    玩家在控制台敲 rewind，续上的局开局时控制台还开着）→ True，是 pause → False（'events'）；都没有 → (None, None)。"""
    rs = ((run.header.get('mt') or {}).get('reseeds') or [])
    if rs and 'paused' in rs[0]:
        return bool(rs[0]['paused']), 'header'
    for n in sorted(run.event_lines):
        for i in run.event_lines[n]:
            ln = run.lines[i]
            if b'"k":"ev"' not in ln[:12]:
                continue
            e = json.loads(ln)
            if e.get('ev') == 'unpause':
                return True, 'events'
            if e.get('ev') == 'pause':
                return False, 'events'
    return None, None


# ---------------------------------------------------------------------------------------------- 菜单暂停
# 真人按 Esc（TRIGGERED MENUBACK，实体 nil）打开暂停菜单的那一帧：录制的 tick 序列是 开菜单前的中间 tick + 暂停期间每个渲染一个
# tick（玩家输入冻结）+ 解除暂停后 2 次轮询 + 更新 tick。开菜单落在 i2（本应同时做逻辑更新的那次轮询）时，这一帧玩家多走
# 一个 60 Hz 子步（phase5-pause/REPORT.md）。回放不暂停就没有这些 tick，从这一帧起分歧。
# 回放：mod 在同一 (帧, tick) 对 TRIGGERED MENUBACK 返回 true 打开菜单；菜单里的按键不经过 MC_INPUT_ACTION，Lua 关不掉，
# 所以暂停 MENU_CLOSE_AFTER 个渲染后 mod 写 pausectl.txt 请求，本进程发一次真实 Esc；解除暂停后的 tick 按"相对解除暂停"对齐。
MENU_OPEN_ACTIONS = (15, 12)   # MENUBACK（Esc）、PAUSE
MENU_CLOSE_AFTER = 10          # 回放开菜单后过多少个渲染请求 Esc
ACTION_RESTART = 16            # ButtonAction.ACTION_RESTART（R 键重开）：默认注入它的全局查询（build_replay_table inject_restart）
MENU_CLOSE_RETRY_S = 1.5       # 发了 Esc 仍在暂停时多久重发


def tick_index(kind):
    if kind == 'u':
        return None
    return int(kind[1:].split('-')[-1])


def menu_pauses(run):
    """录制里每个"真人开菜单"的暂停：返回 (plans, skipped)。
    plan = dict(n, open='i<jp>', open_action, close, close_action, pause_rs, unpause_rs, paused_renders)。
    判定：同一帧里 pause -> unpause，且 pause 前一个 tick（序号 = pause 渲染序号 - 1）有 TRIGGERED MENUBACK/PAUSE（实体 nil）=1。
    换房过场、捡道具弹窗、活板门换层这些引擎自己的暂停没有这个按键，回放会自己暂停，不注入。
    控制台暂停（phase10，recorder 0.4.0 的 --allow-console 录制）：同一帧 pause -> unpause 之间有玩家提交的控制台命令
    （ev cmd src=console，r = 暂停开始的渲染序号）也按菜单暂停注入（console=True）：回放里没人开控制台，用同一 tick 的菜单暂停
    复现"这一次轮询起暂停"的 tick 结构，命令在暂停中执行（replay.lua console_render），再由 Python 发 Esc 关菜单。
    旧录制的 pause 事件没有第三个参数（渲染序号）：解除暂停的渲染序号 = 本帧最后一个中间 tick 序号 - 1（解除暂停后固定 2 次轮询）。"""
    plans, skipped = [], []
    for n, idxs in sorted(run.event_lines.items()):
        evs = [json.loads(run.lines[i]) for i in idxs]
        cons_r = {e.get('r') for e in evs if e.get('k') == 'ev' and e.get('ev') == 'cmd' and e.get('src') == 'console'}
        # （暂停跨过本帧更新的控制台命令没有本帧的 unpause，下面的 pause/unpause 配对自然不会把它当成菜单暂停）
        evs = [e for e in evs if e.get('k') == 'ev' and e.get('ev') in ('pause', 'unpause')]
        if not evs or n not in run.frame_line:
            continue
        o = run.frame_obj(n)
        ticks = {}
        kmax = 0
        for e in o['tk']:
            kind = e if isinstance(e, str) else e[0]
            j = tick_index(kind)
            if j is not None:
                kmax = max(kmax, j)
            if not isinstance(e, str) and len(e) > 2:
                ticks[j] = e[2]
        i = 0
        while i < len(evs):
            if evs[i]['ev'] != 'pause' or i + 1 >= len(evs) or evs[i + 1]['ev'] != 'unpause':
                i += 1
                continue
            pa, ua = evs[i]['a'], evs[i + 1]['a']
            P = ua[0] - pa[0]
            if len(ua) > 2 and len(pa) > 2:
                U, prs = ua[2], pa[2]
            else:
                U = kmax - 1
                prs = U - P
            jp = prs - 1
            trig = [x for x in (ticks.get(jp) or []) if x[0] == 1 and x[1] in MENU_OPEN_ACTIONS and x[2] == -1 and x[3]]
            rec = dict(n=n, pause_rs=prs, unpause_rs=U, paused_renders=P, open_tick=jp, last_tick=kmax)
            if trig and jp >= 1:
                rec.update(open=f'i{jp}', open_action=trig[0][1])
                plans.append(rec)
            elif prs in cons_r and jp >= 1:
                # 控制台暂停：开控制台的按键不经过 MC_INPUT_ACTION（没有 MENUBACK），用 MENUBACK 注入同一 tick 的菜单暂停
                rec.update(open=f'i{jp}', open_action=MENU_OPEN_ACTIONS[0], console=True)
                plans.append(rec)
            else:
                skipped.append(rec)
            i += 2
    return plans, skipped


def continue_start_players(run: load_run.Run):
    """续局（header.continued）：录制第 1 帧每个玩家的位置 [(num, x100, y100), ...]（定点 x100 原值，= load_run 解码前的整数）；
    非续局或没有帧时返回 []。"""
    if not run.header.get('continued') or run.frames < 1:
        return []
    names = (run.header.get('fields') or {}).get('player') or []
    ix, iy = (names.index('x'), names.index('y')) if 'x' in names and 'y' in names else (None, None)
    out = []
    for p in run.frame(1).players:
        raw = p.get('_raw')
        if raw is not None and ix is not None:
            x100, y100 = int(raw[ix]), int(raw[iy])   # 玩家行的整数原值（定点 x100）
        else:
            x100, y100 = int(round(p['x'] * 100)), int(round(p['y'] * 100))
        out.append((int(p['num']), x100, y100))
    return out


def mouse_stream(run: load_run.Run):
    """录制的鼠标流（帧行 ms，phase8-mouse）：返回 (init, per_frame)。init = 第 1 帧第一条 [sx100, sy100, buttons]（开局时的光标；
    mod 的 ms 只在变化时记，第一条总会记），per_frame = {n: [[r, sx100, sy100, buttons], ...]}（其余条目，坐标为定点原值）。
    没有任何 ms 条目时返回 (None, {})。"""
    init, per = None, {}
    for n in range(1, run.frames + 1):
        o = json.loads(run.lines[run.frame_line[n]])
        for m in o.get('ms') or ():
            e = [int(m[0]), int(m[1]), int(m[2]), int(m[5])]
            if init is None:
                init = e[1:]
                continue
            per.setdefault(n, []).append(e)
    return init, per


def mouse_table_lines(run: load_run.Run):
    init, per = mouse_stream(run)
    if init is None:
        return [], dict(mouse_init=None, mouse_frames=0, mouse_entries=0)
    lines = ['mouse_init = {' + ','.join(str(x) for x in init) + '},', 'mouse = {']
    for n, lst in sorted(per.items()):
        lines.append(f'[{n}]={{' + ','.join('{' + ','.join(str(x) for x in e) + '}' for e in lst) + '},')
    lines.append('},')
    return lines, dict(mouse_init=init, mouse_frames=len(per), mouse_entries=sum(len(v) for v in per.values()),
                       mouse_buttons_pressed=sum(1 for v in per.values() for e in v if e[3]))


# ---------------------------------------------------------------------------------------------- 第三方 mod 的调用（api 行）
# recorder 0.4.0（phase10-mods-console）：录制里别的 mod（GoodTrip 小地图传送等）调用的被监视方法（README §4.7 api 行、§8.6）。
# 回放时把这些 mod 临时禁用（副本 mods/<目录>/disable.it，回放结束后挪进回放会话目录，不删），由 mod 的 replay.lua 在录制的
# 同一位置原样调用。结束本局的命令（restart / seed …）不回放（录制以它结束，回放由帧上限在同一帧数截止）。
API_ENDS_RUN = ('restart', 'seed', 'challenge', 'quit', 'exit')


def api_records(run: load_run.Run, replayed=False):
    """api 行（dict 列表，按文件顺序）；replayed=True 只取回放局里 replay.lua 执行的（rp），False 只取录制下来的。"""
    out = []
    for i in getattr(run, 'api_lines', ()):
        d = json.loads(run.lines[i])
        if bool(d.get('rp')) == replayed:
            out.append(d)
    return out


def api_mods(run: load_run.Run):
    """录制里有 api 调用的 mod 目录名（来源不明的 "?" 除外）= 回放时要禁用、由 replay.lua 代为调用的 mod。"""
    return sorted({d.get('src') for d in api_records(run) if d.get('src') not in (None, '?')})


class Unreplayable(ValueError):
    pass


def api_arg_lua(v):
    """api 行里的一个参数（JSON）-> replay.lua 的 {类型, 值} 表（api.lua 的 dec）。"""
    if v is None:
        return '{"n"}'
    if isinstance(v, bool):
        return '{"b",' + lua_value(v) + '}'
    if isinstance(v, int):
        return '{"i",' + str(v) + '}'
    if isinstance(v, str):
        return '{"s",' + lua_value(v) + '}'
    if isinstance(v, dict):
        if 'f' in v and v['f'] is not None:
            return '{"f",' + repr(float(v['f'])) + '}'
        if 'p' in v and v['p'] is not None:
            return '{"p",' + str(int(v['p'])) + '}'
        if 'ref' in v:
            r = v['ref']
            if r is None:
                return '{"ref"}'
            if isinstance(r, dict) and r.get('p') is not None:
                return '{"ref",' + str(int(r['p'])) + '}'
        if 'v' in v and isinstance(v['v'], list) and len(v['v']) == 2 and None not in v['v']:
            return '{"v",' + repr(float(v['v'][0])) + ',' + repr(float(v['v'][1])) + '}'
    raise Unreplayable(f'参数无法回放：{v!r}')


def api_table_lines(run: load_run.Run, mods):
    """回放表的 api 段：只含 mods（回放时禁用的 mod）的调用。返回 (lines, info)。"""
    mods = set(mods or ())
    per, skipped, foreign = {}, [], 0
    for d in api_records(run):
        if d.get('src') not in mods:
            foreign += 1
            continue
        f = [f'id={int(d["id"])}', f'ph={lua_value(d.get("ph"))}', f'm={lua_value(d.get("m"))}',
             f'src={lua_value(d.get("src"))}']
        for k in ('r', 'af', 'rc'):
            if d.get(k) is not None:
                f.append(f'{k}={int(d[k])}')
        if d.get('set'):
            f.append('set=1')
        if isinstance(d.get('self'), dict) and d['self'].get('p') is not None:
            f.append(f'self={int(d["self"]["p"])}')
        if d.get('in') is not None:
            f.append(f'parent={int(d["in"])}')
        skip = None
        try:
            args = [api_arg_lua(a) for a in d.get('a') or []]
        except Unreplayable as e:
            args, skip = [], str(e)
        if d.get('m') == 'Isaac.ExecuteCommand' and d.get('a') and isinstance(d['a'][0], str):
            word = d['a'][0].strip().split(' ')[0].lower()
            if word in API_ENDS_RUN:
                skip = f'结束本局的命令（{word}）：录制以它结束，回放由帧上限截止'
        if skip:
            f.append(f'skip={lua_value(skip)}')
            skipped.append(dict(id=d['id'], n=d['n'], m=d.get('m'), reason=skip))
        f.append('a={' + ','.join(args) + '}')
        per.setdefault(int(d['n']), []).append('{' + ','.join(f) + '}')
    if not per:
        return [], dict(api_calls=0, api_mods=sorted(mods), api_skipped=skipped, api_other_mods_calls=foreign)
    lines = ['api = {']
    for n, lst in sorted(per.items()):
        lines.append(f'[{n}]={{' + ','.join(lst) + '},')
    lines.append('},')
    return lines, dict(api_calls=sum(len(v) for v in per.values()), api_frames=len(per), api_mods=sorted(mods),
                       api_skipped=skipped, api_other_mods_calls=foreign)


def console_commands(run: load_run.Run):
    """录制里玩家在调试控制台提交的命令（recorder 0.4.0，ev cmd src=console）：[(n, r, line, paused, same_frame), ...]。
    same_frame = 控制台暂停在同一帧里就解除了（pause 与 unpause 都在第 n 帧；回放按菜单暂停注入，见 menu_pauses）；
    False = 暂停跨过了第 n 帧的逻辑更新（关控制台与更新在同一次迭代，unpause 记在第 n+1 帧，con4 实测）。"""
    out = []
    for n, idxs in sorted(run.event_lines.items()):
        evs = [json.loads(run.lines[i]) for i in idxs]
        for k, d in enumerate(evs):
            if d.get('k') == 'ev' and d.get('ev') == 'cmd' and d.get('src') == 'console':
                same = any(e.get('k') == 'ev' and e.get('ev') == 'unpause' for e in evs[k + 1:])
                out.append((int(d['n']), int(d.get('r') or 0), d.get('line') or ' '.join(d.get('a') or []), d.get('paused'),
                            same))
    return out


def start_character(run: load_run.Run):
    """回放开局的角色（phase11-human3）：非续局时返回预备命令 `restart <录制的玩家 0 角色>`（harness start 的 pre），
    `seed` 命令沿用当前局的角色与难度，所以先用 restart 换成录制的角色、再 seed。续局 / 录制没有玩家信息时返回 None。
    Jacob & Esau（19/20）：restart 19 开出的就是两人；其它多玩家（合作）不支持，见 replayability 的 issue。"""
    h = run.header
    if h.get('continued'):
        return None
    players = (h.get('meta') or {}).get('players') or []
    if not players or players[0].get('type') is None:
        return None
    return f'restart {int(players[0]["type"])}'


def record_custom_run(run: load_run.Run):
    """录制是否自定义种子局（header.meta.seeds.is_custom_run）；续局 / 没有这个字段时 None（回放不动标记）。"""
    h = run.header
    if h.get('continued'):
        return None
    v = ((h.get('meta') or {}).get('seeds') or {}).get('is_custom_run')
    return None if v is None else bool(v)


def start_meta(header):
    """开局元数据里与回放开局有关的部分：难度、挑战、贪婪、各玩家角色、是否自定义种子局（compare 对照录制/回放用）。"""
    m = (header or {}).get('meta') or {}
    return dict(difficulty=m.get('difficulty'), challenge=m.get('challenge'), greed_mode=m.get('greed_mode'),
                players=[p.get('type') for p in m.get('players') or []],
                is_custom_run=((m.get('seeds') or {}).get('is_custom_run')))


def build_replay_table(run: load_run.Run, path_out, inject_pauses=True, align_engine_pauses=True, inject_extras=False,
                       mouse=True, api=None, inject_restart=True):
    """mouse：把录制的鼠标流写进回放表（mod 经 rr_mt.dll 的 RRMT_SetMouse 还原给引擎，phase8-mouse）。
    api：回放时禁用、由 replay.lua 代为调用的 mod 目录名（api_mods(run)）；None/空 = 回放表没有 api 段。"""
    h = run.header
    seed = h['meta']['seeds']['start_seed_string']
    salts = mt_salts(h)
    start = 'menu:continue' if h.get('continued') else 'seed ' + seed
    lines = ['return {', f'start = {lua_value(start)},', f'frames = {run.frames},',
             f'source = {lua_value(str(run.path))},']
    if salts:
        lines.append('mt = {' + ','.join(f'{k} = {int(v)}' for k, v in sorted(salts.items())) + '},')
    custom = record_custom_run(run)
    if custom is not None:
        # phase11-human3：录制是不是自定义种子局；false 时回放局（`seed` 命令开出，引擎记为自定义种子局）在楼层生成前
        # 经 rr_mt.dll 把标记清掉（replay.lua R.fix_custom_run；自定义种子局商店不刷捐款机）
        lines.append(f'custom_run = {lua_value(custom)},')
    start_paused, start_paused_src = record_start_paused(run)
    if start_paused:
        # phase12：录制开局时游戏暂停（控制台开着）→ 回放局从早期 reseed 到 POST_GAME_STARTED 经 rr_mt.dll 让
        # Game::IsPaused 返回 true（replay.lua R.start_pause_on/off），开局前那一轮实体预更新与录制相同
        lines.append('start_paused = true,')
    start_players = continue_start_players(run)
    if start_players:
        # 续局：引擎给玩家选出生位置用的是 reseed 之前的随机数（phase7-continue/REPORT），回放在续局的
        # MC_POST_GAME_STARTED 里把每个玩家钉回录制第 1 帧的位置（定点 x100 原值，Lua 里 /100），速度清零
        lines.append('start_players = {' + ','.join(
            f'[{num}]={{x100={x100},y100={y100}}}' for num, x100, y100 in start_players) + '},')
    lines.append('ticks = {')
    n_tick_changes = 0
    spawns = {}
    occ_gf, occ = None, {}
    cmds = {}
    extras = {}  # 实验（--inject-extras）：录制里非玩家 12 动作的非零查询 {n: {kind: [[hook, action, who, value], ...]}}
    for n in range(1, run.frames + 1):
        o = json.loads(run.lines[run.frame_line[n]])
        per = []
        for e in o['tk']:
            if isinstance(e, list) and len(e) > 2 and e[2]:
                if inject_extras:
                    extras.setdefault(n, {}).setdefault(e[0], []).extend(e[2])
                elif inject_restart:
                    # phase11-human3：按住 R 重开（ACTION_RESTART 的全局查询）也注入，录制最后一帧（重开在这一帧的更新里
                    # 触发）才能复现；其它 extra（地图键、菜单键…）仍不注入
                    rs = [x for x in e[2] if x[1] == ACTION_RESTART]
                    if rs:
                        extras.setdefault(n, {}).setdefault(e[0], []).extend(rs)
        for e in o['tk']:
            if isinstance(e, list) and e[1]:
                pl = []
                for pnum, triples in e[1]:
                    arr = [None] * 36
                    for a, (v, p, t) in enumerate(triples):
                        arr[a] = p
                        arr[12 + a] = t
                        arr[24 + a] = v
                    pl.append(f'[{pnum}]={{' + ','.join(f'[{i + 1}]={lua_value(x)}' for i, x in enumerate(arr) if x is not None) + '}')
                    n_tick_changes += 1
                per.append(f'["{e[0]}"]={{' + ','.join(pl) + '}')
        if per:
            lines.append(f'[{n}]={{' + ','.join(per) + '},')
        for t, v, s, sd, gf in o.get('sp', []):
            if gf != occ_gf:
                occ_gf, occ = gf, {}
            k = f'{gf}:{t}.{v}.{s}'
            occ[k] = occ.get(k, 0) + 1
            spawns[f'{k}:{occ[k]}'] = sd
    for lst in run.event_lines.values():
        for i in lst:
            d = json.loads(run.lines[i])
            if d['k'] == 'hcmd' and d.get('tag') == 'script':
                cmds.setdefault(d['n'], []).append(d['cmd'])
    lines.append('},')
    lines.append('spawns = {')
    for k, sd in spawns.items():
        lines.append(f'["{k}"]={sd},')
    lines.append('},')
    lines.append('cmds = {')
    for n, lst in sorted(cmds.items()):
        lines.append(f'[{n}]={{' + ','.join(lua_value(c) for c in lst) + '},')
    lines.append('},')
    if extras:
        lines.append('extras = {')
        for n, per in sorted(extras.items()):
            lines.append(f'[{n}]={{' + ','.join(f'["{k}"]={{' + ','.join(
                '{' + ','.join(lua_value(x) for x in ex) + '}' for ex in lst) + '}' for k, lst in per.items()) + '},')
        lines.append('},')
    plans, skipped = menu_pauses(run)
    # 引擎自己的暂停（换房过场、捡道具弹窗、活板门换层）不注入，回放里自己发生；但它们解除暂停后的 tick 也按"相对解除暂停"
    # 对齐（录制与回放的暂停长度可以不同：真人提前关掉的道具弹窗；换层过场偶尔整体错开一个渲染，human1 第 30982 帧）
    engine = [p for p in skipped if p['unpause_rs'] >= 1] if align_engine_pauses else []
    if (inject_pauses and plans) or engine:
        lines.append('pauses = {')
        by_n = {}
        for pl in (plans if inject_pauses else []) + engine:
            by_n.setdefault(pl['n'], []).append(pl)
        for n, lst in sorted(by_n.items()):
            lines.append(f'[{n}]={{' + ','.join(
                (f'{{open="{pl["open"]}",open_action={pl["open_action"]},unpause_rs={pl["unpause_rs"]},'
                 f'close_after={MENU_CLOSE_AFTER}}}') if 'open' in pl else
                f'{{engine=true,unpause_rs={pl["unpause_rs"]}}}' for pl in lst) + '},')
        lines.append('},')
    mouse_info = dict(mouse=False)
    if mouse:
        ml, mouse_info = mouse_table_lines(run)
        mouse_info['mouse'] = bool(ml)
        lines.extend(ml)
    cons = console_commands(run)
    if cons:
        # 控制台命令（phase10）：同一帧窗口里经 Isaac.ExecuteCommand 执行。暂停在本帧内解除的：第 r 次渲染（回放在同一 tick
        # 注入菜单暂停，命令在暂停中执行）；暂停跨过本帧更新的：不注入暂停，在第 min(r, 2) 次渲染之后执行（普通帧窗口在更新前
        # 只有 2 次渲染；con4 实测 r=3 的 spawn 与 r=2 的 giveitem 都要在第 2 次渲染之后、紧接着本帧更新之前）
        lines.append('console = {')
        per = {}
        for n, r, ln, _p, same in cons:
            per.setdefault(n, []).append(f'{{r={r if same else min(r, 2)},line={lua_value(ln)}}}')
        for n, lst in sorted(per.items()):
            lines.append(f'[{n}]={{' + ','.join(lst) + '},')
        lines.append('},')
    api_info = dict(api_calls=0, console_cmds=[dict(n=n, r=r, line=ln, paused=p, pause_same_frame=same)
                                                for n, r, ln, p, same in cons])
    if api:
        al, ai = api_table_lines(run, api)
        api_info.update(ai)
        lines.extend(al)
    lines.append('}')
    Path(path_out).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return dict(start=start, pre=start_character(run), custom_run=custom, start_paused=start_paused,
                start_paused_from=start_paused_src, frames=run.frames, tick_changes=n_tick_changes, spawns=len(spawns),
                cmds={str(k): v for k, v in cmds.items()}, mt_salts=salts, continued=bool(h.get('continued')),
                start_players=[dict(num=num, x=x100 / 100, y=y100 / 100) for num, x100, y100 in start_players],
                inject_pauses=inject_pauses, menu_pauses=plans, engine_pauses=len(skipped), engine_pauses_aligned=len(engine),
                inject_extras=inject_extras, inject_restart=inject_restart and not inject_extras,
                extra_ticks=sum(len(v) for v in extras.values()), **mouse_info, **api_info)


def replayability(run: load_run.Run, continue_save=None):
    h = run.header
    m = h['meta']
    issues = []
    if h.get('continued') and not continue_save:
        issues.append('续局：需要 --continue-save 给出续局前的存档快照（gamestate1.dat）')
    if not (h.get('mt') or {}).get('available'):
        issues.append('录制没有全局 MT reseed（-NoDll 或 DLL 不可用）：回放只能近似，不期望 0 分歧')
    if m.get('difficulty') not in (0, None):
        # phase11-human3：难度没有控制台命令可选；`restart`/`seed` 沿用当前局的难度，菜单开出的第一局用存档里记住的上次选择，
        # 所以 --init-save 给录制会话的存档快照时与录制相同（README §8.8）；回放后 compare 的 start_meta 核对
        issues.append(f'难度 {m.get("difficulty")}：不能用命令指定，回放沿用副本存档（--init-save）里上次选的难度；'
                      'compare 的 start_meta_match 核对')
    if m.get('challenge') not in (0, None):
        issues.append(f'挑战 {m.get("challenge")}：未实现')
    if m.get('greed_mode'):
        issues.append('贪婪模式：未实现（回放开局沿用存档里上次的模式）')
    types = [p.get('type') for p in m.get('players', [])]
    if len(types) > 1 and sorted(types) != [19, 20]:
        issues.append(f'多玩家 {types}：回放开局只按玩家 0 的角色 `restart`，合作玩家未实现')
    if (m.get('options') or {}).get('MouseControl'):
        issues.append('MouseControl 开启：鼠标射击绕过 MC_INPUT_ACTION，无法注入')
    cons = console_commands(run)
    if cons:
        issues.append(f'{len(cons)} 条控制台命令（ev cmd src=console）：回放在同一帧窗口执行，控制台暂停用同一 tick 的菜单暂停'
                      '复现（README §8.7；--no-menu-pause 时不复现，控制台在本应做逻辑更新的那次轮询打开的帧会分歧）')
    unknown = [d for d in api_records(run) if d.get('src') in (None, '?')]
    if unknown:
        issues.append(f'{len(unknown)} 个来源不明的第三方调用（api 行 src="?"，例如 {unknown[0].get("m")}，'
                      f'{unknown[0].get("srcfile")}）：不回放')
    return issues


# ---------------------------------------------------------------------------------------------- 比较
def entity_key_noncosmetic(e):
    ex = e['extra'] or {}
    st, sf = (ex.get('state'), ex.get('state_frame')) if e['category'] == 'npc' else (-1, -1)
    return (e['type'], e['variant'], e['subtype'], e['x'], e['y'], e['vx'], e['vy'], e['hp'], e['max_hp'], e['frame'], st, sf)


def strip(p):
    return {k: v for k, v in p.items() if k != '_raw'}


def diff_dicts(a, b, limit=8):
    out = []
    for k in a:
        if a.get(k) != b.get(k):
            out.append([k, a.get(k), b.get(k)])
            if len(out) >= limit:
                break
    return out


def explain(fa, fb, layer):
    """首个分歧帧上，解码状态（0.01 定点）里哪些字段不同。"""
    ex = {}
    if fa.room_frame != fb.room_frame:
        ex['room_frame'] = diff_dicts(fa.room_frame, fb.room_frame)
    pd = []
    for i, (pa, pb) in enumerate(zip(fa.players, fb.players)):
        d = diff_dicts(strip(pa), strip(pb))
        if d:
            pd.append([i, d])
    if pd:
        ex['players'] = pd
    if layer in ('noncosmetic', 'full') and fa.entities is not None and fb.entities is not None:
        ea = [entity_key_noncosmetic(e) for e in fa.entities if e['type'] != 1000]
        eb = [entity_key_noncosmetic(e) for e in fb.entities if e['type'] != 1000]
        if ea != eb:
            sa, sb = set(ea), set(eb)
            ex['entities_record_only'] = [list(x) for x in ea if x not in sb][:4]
            ex['entities_replay_only'] = [list(x) for x in eb if x not in sa][:4]
    if layer == 'full' and fa.entities is not None and fb.entities is not None:
        fa_fx = sorted((e['variant'], e['x'], e['y']) for e in fa.entities if e['type'] == 1000)
        fb_fx = sorted((e['variant'], e['x'], e['y']) for e in fb.entities if e['type'] == 1000)
        if fa_fx != fb_fx:
            ex['effects'] = dict(record=len(fa_fx), replay=len(fb_fx))
        # 特效（1000.x）按 Index 逐字段对照（速度、帧号、附加字段等；只比位置会漏掉"位置相同、速度不同"的首个分歧帧）
        da = {e['idx']: e for e in fa.entities if e['type'] == 1000}
        db = {e['idx']: e for e in fb.entities if e['type'] == 1000}
        fx_fields = []
        for i in sorted(set(da) & set(db)):
            d = diff_dicts({k: v for k, v in da[i].items() if k != '_raw'}, {k: v for k, v in db[i].items() if k != '_raw'})
            if d:
                fx_fields.append([i, da[i]['variant'], d])
                if len(fx_fields) >= 4:
                    break
        if fx_fields:
            ex['effect_fields'] = fx_fields
        sa = [(e['type'], e['init_seed'], e['idx']) for e in fa.entities]
        sb = [(e['type'], e['init_seed'], e['idx']) for e in fb.entities]
        if sa != sb:
            ex['entity_seeds_or_index_differ'] = True
        if (fa.room or {}).get('grid') != (fb.room or {}).get('grid'):
            ex['grid_differs'] = True
    return ex


def zero_frame_compatible(rec_frames, rep_frames, rep_end):
    """录制 0 帧的局（按住 R 在第一次逻辑更新之前就重开，human3 run-0025/0031）：0.4.1 起回放也是 0 帧（harness 帧上限 0
    当场关局）；0.4.0 的回放会多跑 1 帧才被帧上限截止。两种都算与录制等长（没有可比的帧，只比 header 的 MT reseed）。"""
    return rec_frames == 0 and rep_frames <= 1 and rep_end == 'frames'


def compare(rec_path, rep_path):
    A, B = load_run.Run(rec_path), load_run.Run(rep_path)
    zero = zero_frame_compatible(A.frames, B.frames, (B.end or {}).get('reason'))
    n = 0 if zero else min(A.frames, B.frames)
    res = dict(record=str(rec_path), replay=str(rep_path), record_frames=A.frames, replay_frames=B.frames, compared=n,
               record_end=(A.end or {}).get('reason'), replay_end=(B.end or {}).get('reason'),
               replay_pins=(B.end or {}).get('replay'))
    # 开局元数据（角色、难度…）：回放开局选错角色/难度时第 1 帧就分歧，这里直接给出原因
    sa, sb = start_meta(A.header), start_meta(B.header)
    res['start_meta'] = dict(record=sa, replay=sb)
    res['start_meta_match'] = sa == sb
    res['zero_frame_record'] = A.frames == 0
    firsts = {k: None for k in LAYERS}
    counts = {k: 0 for k in LAYERS}
    mt_first, mt_count, mt_missing = None, 0, 0
    ia, ib = A.iter_frames(), B.iter_frames()
    fixed_first = {k: None for k in LAYERS}
    examples = {}
    tick_diff = None
    death = dict(record=None, replay=None)
    mouse_first, mouse_count = None, 0   # phase8-mouse：回放局自己读到的鼠标流与录制逐帧比较（诊断，不进 exact）
    for _ in range(n):
        fa, fb = next(ia), next(ib)
        if fa.mt is None or fb.mt is None:
            mt_missing += 1
        elif [int(x) for x in fa.mt[:2]] != [int(x) for x in fb.mt[:2]]:  # 第三个数（读的次数）不比
            mt_count += 1
            if mt_first is None:
                mt_first = fa.n
                examples['mt'] = dict(frame=fa.n, record=fa.mt, replay=fb.mt)
        for li, layer in enumerate(LAYERS):
            if int(fa.digest[li]) != int(fb.digest[li]):
                counts[layer] += 1
                if firsts[layer] is None:
                    firsts[layer] = fa.n
                    examples[layer] = dict(frame=fa.n, gf_record=fa.gf, gf_replay=fb.gf, diff=explain(fa, fb, layer))
        # 0.01 定点口径（解码后）：玩家与房间
        if fixed_first['player_room'] is None and (fa.room_frame != fb.room_frame or
                                                  [strip(p) for p in fa.players] != [strip(p) for p in fb.players]):
            fixed_first['player_room'] = fa.n
        if fa.mouse != fb.mouse:
            mouse_count += 1
            if mouse_first is None:
                mouse_first = fa.n
                examples['mouse'] = dict(frame=fa.n, record=fa.mouse, replay=fb.mouse)
        for nm, fr in (('record', fa), ('replay', fb)):
            if death[nm] is None and fr.players and fr.players[0].get('dead'):
                death[nm] = fr.n
        if tick_diff is None and [(t.kind, t.inputs) for t in fa.ticks] != [(t.kind, t.inputs) for t in fb.ticks]:
            # 菜单暂停帧的 tick 列表可有上千项：只留前后各 6 个与总数
            def kinds(fr):
                ks = [t.kind for t in fr.ticks]
                return ks if len(ks) <= 14 else ks[:6] + [f'...({len(ks) - 12})...'] + ks[-6:]
            tick_diff = dict(frame=fa.n, record=kinds(fa), replay=kinds(fb), n_record=len(fa.ticks), n_replay=len(fb.ticks))
    res['first_divergent_frame'] = firsts
    res['differing_frames'] = counts
    res['mt'] = dict(first_divergent_frame=mt_first, differing_frames=mt_count, frames_without_mt=mt_missing,
                     reseeds_match=reseeds_match(A.header, B.header))
    res['first_divergent_frame_fixed_0p01'] = fixed_first
    res['mouse'] = dict(first_divergent_frame=mouse_first, differing_frames=mouse_count,
                        replay_stats=((B.end or {}).get('replay') or {}).get('mouse'))
    res['examples'] = examples
    res['first_tick_mismatch'] = tick_diff
    res['first_dead_frame'] = death  # 玩家 0 的 dead 字段第一次为真的帧（比较范围内）
    res['same_death_frame'] = death['record'] is not None and death['record'] == death['replay']
    same_len = A.frames == B.frames or zero
    res['same_length'] = same_len
    res['same_end'] = same_len and res['record_end'] == res['replay_end']
    res['player_room_identical_whole_run'] = firsts['player_room'] is None and same_len
    # 结束方式相容：相同；或录制由玩家退出/重开结束（exit / exit_save / restart，这些是菜单/按键操作，回放不注入），
    # 回放在同一帧数被 harness 帧上限截止（frames）
    # 录制截断（没有 end 行：进程被强杀/崩溃，phase9-robust 场景 C）时，录下的最后一帧之后没有可比的东西，
    # 回放在同一帧数被帧上限截止同样算相容；record_partial_tail = 读取时丢掉的末尾不完整行
    res['record_truncated'] = A.end is None
    res['record_partial_tail'] = A.partial_tail or None
    res['end_compatible'] = res['same_end'] or (same_len and res['replay_end'] == 'frames'
                                                and (res['record_end'] in ('exit', 'exit_save', 'restart', 'frames', 'stop')
                                                     or res['record_truncated']))
    # E6：三层 + MT 全部 0 分歧、同长度、结束方式相容、MT 每帧都有、reseed 记录一致
    # phase11：开局元数据（角色/难度/挑战/贪婪）也要相同（0 帧局没有帧可比，角色选错也查得出来）
    res['exact'] = (all(firsts[k] is None for k in LAYERS) and mt_first is None and mt_missing == 0 and same_len
                    and res['end_compatible'] and res['mt']['reseeds_match'] is True and res['start_meta_match'])
    res['pause_report'] = pause_report(A, B, firsts, mt_first)
    # phase12：录制开局时暂停（控制台开着）而回放没能在开局窗口暂停（旧 DLL 没有 RRMT_ForcePause 等）→ 已知类别
    res['start_paused'] = start_paused_report(A, B)
    if not res['exact'] and res['start_paused']['record'] and not res['start_paused']['replay_forced']:
        res['pause_report']['first_divergence_category'] = 'start_paused_not_replayed'
    res['first_divergence_category'] = res['pause_report']['first_divergence_category'] if not res['exact'] else None
    res['api'] = api_report(A, B)
    return res


def start_paused_report(A, B):
    """录制开局时是否暂停（record_start_paused）、回放局第一次 reseed 当时是否暂停（header，0.4.2 起）、回放有没有经
    RRMT_ForcePause 在开局窗口强制暂停（end.replay.start_paused.ok）。诊断；不进 exact（结果已由 MT/digest 比较体现）。"""
    rec, src = record_start_paused(A)
    rs = ((B.header.get('mt') or {}).get('reseeds') or [])
    rep_paused = rs[0].get('paused') if rs else None
    st = ((B.end or {}).get('replay') or {}).get('start_paused')
    return dict(record=rec, record_from=src, replay=rep_paused, replay_forced=bool(st and st.get('ok')), replay_stats=st)


def api_report(A, B):
    """第三方 mod 调用（api 行）的对照（诊断，不进 exact）：录制下来的调用、回放局里 replay.lua 实际执行的（rp，按录制 id
    对上），执行位置（帧、ph、渲染序号）与录制不同的，以及回放局里没被禁用的 mod 自己发起的调用（live）。"""
    ra = api_records(A)
    if not ra and not getattr(B, 'api_lines', None):
        return None
    rb = api_records(B, replayed=True)
    live = api_records(B)
    done = {d.get('id'): d for d in rb}
    moved = []
    for d in ra:
        e = done.get(d.get('id'))
        if e and (e.get('n') != d.get('n') or e.get('ph') != d.get('ph') or e.get('r') != d.get('r')
                  or e.get('m') != d.get('m')):
            moved.append(dict(id=d['id'], m=d.get('m'), record=[d.get('n'), d.get('ph'), d.get('r'), d.get('af')],
                              replay=[e.get('n'), e.get('ph'), e.get('r'), e.get('af'), e.get('where')]))
    by_m = {}
    for d in ra:
        by_m[d.get('m')] = by_m.get(d.get('m'), 0) + 1
    return dict(record_calls=len(ra), record_by_method=by_m, record_mods=sorted({d.get('src') for d in ra}),
                replay_executed=len(rb), replay_live_calls=len(live),
                not_executed=[d['id'] for d in ra if d.get('id') not in done][:50],
                position_changed=moved[:20], position_changed_count=len(moved),
                replay_stats=((B.end or {}).get('replay') or {}).get('api'))


def pause_report(A, B, firsts, mt_first):
    """录制里的菜单暂停（真人开菜单），以及回放有没有在同一帧暂停；首个分歧帧若正好是一个回放没有复现的菜单暂停帧，
    归为已知类别 menu_pause_not_replayed（录制本身逐帧状态不受影响，只是回放从这一帧起没有可比性）。"""
    plans, _ = menu_pauses(A)
    rep_pauses = {}
    for n, idxs in B.event_lines.items():
        evs = [json.loads(B.lines[i]) for i in idxs]
        if any(e.get('k') == 'ev' and e.get('ev') == 'pause' for e in evs):
            rep_pauses[n] = [e['a'] for e in evs if e.get('k') == 'ev' and e.get('ev') in ('pause', 'unpause')]
    items = []
    for p in plans:
        items.append(dict(frame=p['n'], open_tick=p['open'], paused_renders=p['paused_renders'], console=bool(p.get('console')),
                          replay_paused=p['n'] in rep_pauses, replay_pause_events=rep_pauses.get(p['n'])))
    cands = [x for x in list(firsts.values()) + [mt_first] if x is not None]
    first = min(cands) if cands else None
    cat = None
    if first is not None:
        hit = [x for x in items if x['frame'] == first]
        if hit and not hit[0]['replay_paused']:
            cat = 'menu_pause_not_replayed'
        elif hit:
            cat = 'menu_pause_replayed_but_diverged'
        else:
            cat = 'unknown'
    return dict(menu_pauses=items, first_divergent_frame_any=first, first_divergence_category=cat)


def reseeds_match(ha, hb):
    """两个文件 header.mt.reseeds 的时机、seeded_with、reseed 后的 (cursor, hash) 是否逐项相同；第二次起 reseed 前的
    (cursor, hash) 也要相同（第一次 reseed 前的 MT 是进程开机以来的历史，本来就不同）。
    返回 True / False / None（任一方没有 MT）。"""
    ma, mb = ha.get('mt') or {}, hb.get('mt') or {}
    if not (ma.get('available') and mb.get('available')):
        return None
    ra, rb = ma.get('reseeds') or [], mb.get('reseeds') or []
    def key(i, r):
        return (r.get('at'), r.get('seeded_with'), tuple(r.get('before') or []) if i > 0 else None, tuple(r.get('after') or []))
    return len(ra) == len(rb) and all(key(i, a) == key(i, b) for i, (a, b) in enumerate(zip(ra, rb)))


# ---------------------------------------------------------------------------------------------- 回放进程
TURBO_MODES = ('gate', 'clock')
# 加速回放的 stall 判断：原速 25 s 没有帧推进 = stalled；加速时每秒几百帧，10 s 不动就是卡住了
STALL_S = dict(off=25.0, turbo=10.0)


def turbo_env(turbo):
    """--turbo 对应的 rr_mt.dll 环境变量（rl/bridge/engine/j746_rec/rr_turbo.cpp）。
    gate  = 虚拟时钟 + GPU 提交门控（DrawElements/DrawArrays/SwapBuffers 不提交，CPU 渲染照跑；窗口不再刷新）
    clock = 只有虚拟时钟（完整渲染，窗口照常显示）"""
    if not turbo:
        return {}
    if turbo not in TURBO_MODES:
        raise ValueError(f'turbo 模式只能是 {TURBO_MODES}：{turbo!r}')
    env = {'RR_MT_TURBO': '1'}
    if turbo == 'gate':
        env['RR_MT_TURBO_GPU_GATE'] = '1'
    return env


def affinity_mask(affinity):
    """--affinity：'auto' = 逻辑 CPU 0 一个核（掩码 1；本机实测 9.5–9.9×，最后一个逻辑 CPU（E 核）只有 6.4–6.7×）；
    '0'/'off'/None = 不限制；其它 = 掩码（十进制或 0x 十六进制）。与录制侧共用 run_recording.affinity_mask。"""
    return rr.affinity_mask(affinity)


def replay_env(turbo, mouse, affinity=None):
    """回放进程的额外环境变量：turbo_env + RR_MT_MOUSE=1（rr_mt.dll 才导出可用的 RRMT_SetMouse）+ RR_MT_AFFINITY
    （DLL 启动即把进程限制到这些 CPU：多核并行时引擎的 1000.x 环境特效偶发不确定，phase8-mouse/FINDINGS.md §H4）。"""
    env = dict(turbo_env(turbo))
    if mouse:
        env['RR_MT_MOUSE'] = '1'
    m = affinity_mask(affinity)
    if m:
        env['RR_MT_AFFINITY'] = hex(m)
    # phase12：装 Game::IsPaused 钩子（只在回放表 start_paused = true 的局、开局窗口里由 mod 打开；其余时候不起作用）
    env['RR_MT_FORCEPAUSE'] = '1'
    return env


def disable_mods_in_copy(copy: rr.GameCopy, mods):
    """回放前给副本 mods/<目录> 放空的 disable.it（目录不存在或已禁用的跳过）。返回实际放了的目录名列表。"""
    placed = []
    for m in sorted(set(mods or ())):
        d = copy.mods / m
        if d.is_dir() and not (d / 'disable.it').exists():
            (d / 'disable.it').write_bytes(b'')
            placed.append(m)
    return placed


def restore_disabled_mods(copy: rr.GameCopy, placed, keep_dir):
    """回放后把放进去的 disable.it 挪到 keep_dir/<目录>/disable.it（不删除），mod 恢复启用。返回挪动记录。"""
    out = []
    for m in placed:
        src = copy.mods / m / 'disable.it'
        if src.exists():
            dst = Path(keep_dir) / m
            dst.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst / 'disable.it'))
            out.append(dict(mod=m, moved_to=str(dst / 'disable.it')))
    return out


def replay_run_name(proc, index):
    """mod 给第 index 个开出的局起的名字（main.lua open_run：run-<proc>-<NNNN>）。"""
    return f'run-{proc}-{int(index):04d}'


def start_target(runs0, pre):
    """回放局在本进程里的序号：已开局数 + 1；有 pre（restart <角色>）时预备局再占一个序号。"""
    return int(runs0 or 0) + (2 if pre else 1)


def start_replay_run(g, table, limit_frames, timeout=30):
    """发 harness start（table['pre'] 非空时先 restart <角色> 开预备局，再 seed），等回放局开出来；返回 (回放局名, 状态)。
    回放局名预先算好（start_target）；等待条件是 status.runs 到了这个序号——即回放局已经开过，不要求还在局里：
    极短的局（录制 2–9 帧，turbo 下几十毫秒就跑完回到"没有局"的状态）在 Python 轮询到 in_game 之前就结束了
    （phase11-human3 第一轮 7 局因此误判为没开出来、重试 6 次），这种情况直接进入收尾。"""
    s0 = None
    for _ in range(50):
        s0 = g.status()
        if s0 is not None:
            break
        time.sleep(0.02)
    if s0 is None:
        raise TimeoutError('读不到 status.json')
    target = start_target(s0.get('runs'), table.get('pre'))
    cmd = dict(cmd='start', start=table['start'], replay=True, frames=limit_frames)
    if table.get('pre'):
        cmd['pre'] = table['pre']
    cid = g.command(**cmd)
    g.wait_ack(cid, timeout)
    st = g.wait(lambda st: int(st.get('runs') or 0) >= target, timeout)
    return replay_run_name(st.get('proc') or s0.get('proc'), target), st


def run_replay(rec_path, name, out_root, attempts=6, timeout_s=None, use_dll=True, continue_save=None,
               copy_root=rr.DEFAULT_COPY, turbo=None, init_save=None, forced_options=None, inject_pauses=True,
               align_engine_pauses=True, inject_extras=False, mouse=True, affinity='auto', disable_mods='auto',
               inject_restart=True):
    """mouse：还原录制的鼠标流（phase8-mouse；需要带 RRMT_SetMouse 的 rr_mt.dll，进程环境 RR_MT_MOUSE=1）。
    affinity：回放进程的 CPU 亲和（默认 'auto' = 一个核；'0' = 不限制；见 affinity_mask），只在带 DLL 时生效。
    continue_save：续局回放用的存档快照目录（含 gamestate1.dat），拷进本会话的存档目录后菜单 Enter 续局。
    copy_root：隔离副本（默认 game-rec-01）。turbo：None（原速，默认）/ 'gate' / 'clock'（见 turbo_env，需要 DLL）。
    init_save：非续局回放时先把这个存档快照（例如录制会话 save/run-...，含 persistentgamedata*.dat 与 options.ini）
    拷进副本存档目录，使解锁/道具池与录制开局时相同；不拷 gamestate1.dat（否则菜单 Enter 会先续上旧局）。
    disable_mods：'auto'（默认）= 录制里有 api 调用的第三方 mod（api_mods）；'none' = 不禁用（这些 mod 的调用也不代为执行）；
    或目录名列表。回放期间给这些 mod 放 disable.it，结束后挪进回放会话目录 disabled-mods/（README §8.6）。"""
    if turbo and not use_dll:
        raise ValueError('--turbo 需要 rr_mt.dll（不能与 --no-dll 同用）')
    turbo_env(turbo)  # 模式名不对就在启动前报错
    run = load_run.Run(rec_path)
    issues = replayability(run, continue_save)
    continued = bool(run.header.get('continued'))
    if continued:
        init_from, exclude = continue_save, ()
    else:
        init_from, exclude = init_save, ('gamestate1.dat',)
    # 回放进程没人看着：失焦暂停会让回放停住（真人会话 20261106-human1 的第一次回放在第 1068 帧因此被判 stalled），所以强制关掉
    sess = rr.Session(name, out_root, copy_root=copy_root, raw_policy='keep', use_dll=use_dll,
                      init_save_from=init_from, init_save_exclude=exclude, extra_env=replay_env(turbo, mouse and use_dll, affinity if use_dll else None),
                      forced_options=forced_options or dict(rr.FORCED_OPTIONS, PauseOnFocusLost='0'))
    if disable_mods == 'auto':
        api_set = api_mods(run)
    elif disable_mods in (None, 'none', ''):
        api_set = []
    else:
        api_set = [x for x in (disable_mods if isinstance(disable_mods, (list, tuple, set))
                               else str(disable_mods).split(',')) if x]
    placed = disable_mods_in_copy(sess.copy, api_set)
    try:
        sess.prepare()
    except Exception:
        restore_disabled_mods(sess.copy, placed, sess.dir / 'disabled-mods')
        raise
    # 录制以死亡结束时多给 90 帧，让回放局自己走到 MC_POST_GAME_END（比较"是否同帧死亡"）
    limit_frames = run.frames + (90 if (run.end or {}).get('reason') == 'death' else 0)
    ctl = dict(purpose='replay', source=str(rec_path))
    if continued:
        # 续局：本进程第一个开出的局（菜单 Enter 续上的局）就是回放局
        ctl['menu_run'] = dict(replay=True, frames=limit_frames)
    rr.write_control(sess.irr_dir, ctl)
    info = dict(source=str(rec_path), issues=issues, use_dll=use_dll, frame_limit=limit_frames,
                affinity=hex(affinity_mask(affinity)) if use_dll and affinity_mask(affinity) else None,
                continue_save=str(continue_save) if continue_save else None, copy=str(sess.copy.root),
                turbo=turbo or 'off', init_save=str(init_save) if init_save and not continued else None,
                api_mods=list(api_set), mods_disabled=placed)
    stall_s = STALL_S['turbo' if turbo else 'off']
    try:
        # 放在 try 里：建表失败也要走 finish()（还原 options.ini、停监视线程、写 session.json）
        table = build_replay_table(run, sess.irr_dir / 'irr_replay.lua', inject_pauses=inject_pauses,
                                   align_engine_pauses=align_engine_pauses, inject_extras=inject_extras,
                                   mouse=mouse, api=api_set, inject_restart=inject_restart)
        info['table'] = table
        # 进菜单之后、回放局开出来之前进程崩溃（例如 0xC0000409）或超时：同样算一次启动失败，换新进程重试并计数
        info['start_failures'] = []
        for k in range(attempts):
            g, s = sess.launch_and_enter(attempts=attempts)
            if continued:
                rep_run = s.get('run')
                break
            try:
                time.sleep(1.0)
                rep_run, _st = start_replay_run(g, table, limit_frames)
                break
            except (rr.LaunchFailed, TimeoutError) as e:
                info['start_failures'].append(dict(pid=g.pid, error=repr(e)))
                sess.mark_menu_failure(g, f'开回放局前失败：{e!r}')
                # 旧进程留下的命令文件挪走（不删），免得新进程的 harness 一进菜单就执行旧的 start
                stale = sess.irr_dir / f'stale-cmds-{k + 1}'
                stale.mkdir(exist_ok=True)
                for f in sess.irr_dir.glob('cmd-*.json'):
                    f.replace(stale / f.name)
                time.sleep(2.0)
        else:
            raise rr.LaunchFailed(f'{attempts} 次都没能开出回放局：{info["start_failures"]}')
        info['replay_run'] = rep_run
        t0 = time.time()
        # 超时按原速估（加速失效时也不会被误判为超时）
        limit = timeout_s or (run.frames / 30.0 * 1.5 + 60)
        last_n, last_t = -1, time.time()
        samples = []  # [墙钟秒（相对 t0）, 回放局帧号]，约每秒一个，用来算实测倍速
        esc_log = []   # 菜单暂停：mod 请求关菜单 -> 发真实 Esc
        esc_sent = {}  # 请求标识 -> 上次发 Esc 的时间
        ctl = sess.irr_dir / 'pausectl.txt'
        info['menu_pause_esc'] = esc_log
        while True:
            if not g.alive():
                info['ended'] = 'process_exit'
                break
            st = g.status()
            if st is None:  # status.json 正被重写
                time.sleep(0.05)
                continue
            if st.get('run') != rep_run or not st.get('in_game'):
                info['ended'] = 'run_closed'
                break
            now = time.time()
            if st.get('n') != last_n:
                last_n, last_t = st.get('n'), now
                if not samples or now - t0 - samples[-1][0] >= 1.0:
                    samples.append([round(now - t0, 2), st.get('n')])
            elif now - last_t > stall_s:
                info['ended'] = 'stalled'
                break
            if now - t0 > limit:
                info['ended'] = 'timeout'
                break
            pending = False
            if table.get('menu_pauses') and inject_pauses:
                try:
                    req = ctl.read_text(encoding='utf-8').strip()
                except OSError:
                    req = ''
                if req:
                    pending = True
                    last_t = now  # 暂停中帧不推进，不算 stalled
                    t_prev = esc_sent.get(req)
                    if t_prev is None or now - t_prev > MENU_CLOSE_RETRY_S:
                        g.tap('ESC', hold=0.05)
                        esc_sent[req] = time.time()
                        esc_log.append(dict(req=req, t=round(now - t0, 3), retry=t_prev is not None))
            time.sleep(0.01 if pending or (table.get('menu_pauses') and inject_pauses) else (0.2 if not turbo else 0.05))
        info['replay_s'] = round(time.time() - t0, 1)
        info['progress_samples'] = samples
        if len(samples) >= 2 and samples[-1][0] > samples[0][0]:
            fps = (samples[-1][1] - samples[0][1]) / (samples[-1][0] - samples[0][0])
            info['logic_fps'] = round(fps, 1)
            info['speedup_x_realtime'] = round(fps / 30.0, 2)
        time.sleep(1.0 if not turbo else 0.2)
        info['close'] = g.close()
        dl = g.dll_log() if use_dll else None
        if dl:
            txt = dl.read_text(encoding='utf-8', errors='replace').splitlines()
            info['dll_status'] = [ln for ln in txt if 'status=' in ln][-1:]
            info['dll_turbo_log'] = [ln for ln in txt if 'turbo:' in ln][-6:]
    except Exception as e:
        info['error'] = repr(e)
    finally:
        sess.info['replay'] = info
        try:
            sess.finish()
        finally:
            # 回放结束（无论成败）：挪走临时放的 disable.it，第三方 mod 恢复启用
            info['mods_restored'] = restore_disabled_mods(sess.copy, placed, sess.dir / 'disabled-mods')
    rep_file = None
    for r in sess.info['runs']:
        if r.get('lua_name') == info.get('replay_run'):
            rep_file = sess.dir / r['file']
    info['replay_file'] = str(rep_file) if rep_file else None
    return sess, info


# ---------------------------------------------------------------------------------------------- 观战模式 / 实录导出（phase6-watch）
# --watch：像 MOBA 回放那样看真实游戏画面。rr_mt.dll 装观战钩子（RR_MT_WATCH=1：虚拟时钟 + 速率旋钮 + 呈现节流），mod 加载
# watch.lua（irr_watch.json）：热键、观战 GUI、自由镜头、往前跳 = 不限速跑到目标帧；往回跳 = mod 写 watchreq.txt，本进程结束
# 当前游戏进程、把开局存档快照重新拷回副本存档目录、重开回放进程从头精确重跑（新进程按 start_target 全速到目标帧）。
# --capture OUT.mp4：加速回放时每个逻辑帧取"MC_POST_UPDATE 之后、本帧最后一次渲染"的画面（DLL 在 SwapBuffers 前读后台缓冲），
# DLL 自己启动 ffmpeg 编码 30 fps，旁边写 OUT.json（isaac-footage-v1）。见 README §8.3 / §8.4、experiments/run-recorder/phase6-watch/。
WATCH_FORCED_OPTIONS = dict(rr.FORCED_OPTIONS, PauseOnFocusLost='0', MouseControl='0')
WATCH_SPEEDS = (0.25, 0.5, 1, 2, 4, 8, 16)
# 给游戏窗口发键用（PostMessage，不抢前台）：vk, scancode, extended（GLFW 按 scancode 翻译）
WATCH_KEYS = dict(rr.KEYS, PERIOD=(0xBE, 0x34, 0), EQUAL=(0xBB, 0x0D, 0), MINUS=(0xBD, 0x0C, 0),
                  LBRACKET=(0xDB, 0x1A, 0), RBRACKET=(0xDD, 0x1B, 0), PGUP=(0x21, 0x49, 1), PGDN=(0x22, 0x51, 1),
                  HOME=(0x24, 0x47, 1), G=(0x47, 0x22, 0), C=(0x43, 0x2E, 0), H=(0x48, 0x23, 0), F12=(0x7B, 0x58, 0),
                  BACK=(0x08, 0x0E, 0), M=(0x4D, 0x32, 0),
                  **{str(d): (0x30 + d, 0x0B if d == 0 else 0x01 + d, 0) for d in range(10)})


def post_key(g, name, down):
    vk, sc, ext = WATCH_KEYS[name]
    lp = 1 | (sc << 16) | (ext << 24)
    if not down:
        lp |= (1 << 30) | (1 << 31)
    for h in g.hwnds():
        rr.u32.PostMessageW(h, rr.WM_KEYDOWN if down else rr.WM_KEYUP, vk, lp)


def tap_key(g, name, hold=0.06):
    post_key(g, name, True)
    time.sleep(hold)
    post_key(g, name, False)


def build_watch_timeline(run: load_run.Run, path_out):
    """观战 GUI 的进度条数据：层（起始帧、层号、名字）、房间（首帧、房间号、类型、形状）、事件（受伤、Boss 房、死亡、菜单暂停）。"""
    levels, rooms, events = [], [], []
    for ln, n in run.level_lines:
        o = json.loads(run.lines[ln])
        levels.append(dict(n=n, stage=o.get('stage'), stage_type=o.get('stage_type'), name=o.get('name') or ''))
    for ln, n in run.room_lines:
        o = json.loads(run.lines[ln])
        desc = o.get('desc') or []
        rooms.append(dict(n=n, idx=o.get('room_index'), type=o.get('type'), shape=o.get('shape'),
                          name=desc[10] if len(desc) > 10 and isinstance(desc[10], str) else ''))
        if o.get('type') == 5:
            events.append(dict(n=n, k='boss'))
    for n, idxs in sorted(run.event_lines.items()):
        for i in idxs:
            d = json.loads(run.lines[i])
            if d.get('k') != 'ev':
                continue
            if d.get('ev') == 'damage':
                events.append(dict(n=n, k='damage'))
            elif d.get('ev') == 'game_end' and (d.get('a') or [False])[0]:
                events.append(dict(n=n, k='death'))
    for p in menu_pauses(run)[0]:
        events.append(dict(n=p['n'], k='pause'))
    events.sort(key=lambda e: e['n'])
    rooms.sort(key=lambda r: r['n'])
    levels.sort(key=lambda l: l['n'])

    def tbl(items):
        return '{' + ','.join('{' + ','.join(f'{k}={lua_value(v)}' for k, v in it.items() if v is not None) + '}'
                              for it in items) + '}'
    text = f'return {{frames={run.frames},\nlevels={tbl(levels)},\nrooms={tbl(rooms)},\nevents={tbl(events)}}}\n'
    Path(path_out).write_text(text, encoding='utf-8')
    return dict(levels=len(levels), rooms=len(rooms), events=len(events),
                damage=sum(1 for e in events if e['k'] == 'damage'), boss=sum(1 for e in events if e['k'] == 'boss'))


def install_watch_resources(copy):
    """mod 自带的 resources/（观战 GUI 的 1 像素贴图）拷进副本的 mod 目录；GameCopy.install_mod 只拷顶层文件。"""
    src = rr.MOD_SRC / 'resources'
    dst = copy.mods / rr.MOD_NAME / 'resources'
    files = []
    if src.exists():
        for f in sorted(src.rglob('*')):
            if f.is_file():
                d = dst / f.relative_to(src)
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, d)
                files.append(str(f.relative_to(src)).replace('\\', '/'))
    return files


def restore_init_save(sess, k):
    """重启回放进程前：把上一个进程可能写过的存档挪开（gamestate1.dat 等，不删），重新拷入开局快照并强制观战选项，
    保证每个回放进程的解锁/道具池都与录制开局时相同（同 Session.prepare 的 init_save 规则）。"""
    sd = sess.save_dir
    moved, copied = [], []
    disp = sess.dir / f'save/displaced-relaunch-{k:02d}'
    for f in sorted(sd.iterdir()):
        if f.is_file() and f.name.lower() in sess.init_save_exclude:
            disp.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(disp / f.name))
            moved.append(f.name)
    if sess.init_save_from:
        for f in sorted(sess.init_save_from.iterdir()):
            if (f.is_file() and f.suffix.lower() != '.dmp' and f.name.lower() not in ('log.txt', 'manifest.json')
                    and f.name.lower() not in sess.init_save_exclude):
                shutil.copy2(f, sd / f.name)
                copied.append(f.name)
    opt = sd / 'options.ini'
    if opt.exists() and sess.enforce_options:
        rr.set_options(opt, sess.forced_options)
    return dict(moved=moved, copied=copied)


def pick_encoder(ffmpeg):
    """h264_nvenc 能用就用（快），否则 libx264。返回 (ffmpeg 输出参数串, 编码器名)。"""
    nv = '-c:v h264_nvenc -preset p5 -rc vbr -cq 24 -b:v 0 -pix_fmt yuv420p -movflags +faststart'
    try:
        p = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc=size=1280x720:rate=30',
                            '-t', '0.3', '-c:v', 'h264_nvenc', '-f', 'null', '-'], capture_output=True, timeout=30)
        if p.returncode == 0:
            return nv, 'h264_nvenc'
    except Exception:
        pass
    return '-c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p -movflags +faststart', 'libx264'


def find_ffmpeg():
    f = shutil.which('ffmpeg')
    if f:
        return f
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError('找不到 ffmpeg：PATH 里没有，也没有 imageio-ffmpeg（pip install imageio-ffmpeg）')


def proc_memory(pid):
    try:
        import psutil
        m = psutil.Process(pid).memory_info()
        return dict(rss_mb=round(m.rss / 2**20, 1), private_mb=round(getattr(m, 'private', m.vms) / 2**20, 1),
                    peak_wset_mb=round(getattr(m, 'peak_wset', 0) / 2**20, 1))
    except Exception as e:
        return dict(error=repr(e))


class WatchControl:
    """写 watchctl.txt（每行 "<id> <命令> [参数]"，id 递增；mod 只执行比上次大的 id）。每个新进程重新从 1 开始。"""

    def __init__(self, irr_dir):
        self.path = Path(irr_dir) / 'watchctl.txt'
        self.reset()

    def _write(self, text):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(text, encoding='utf-8')
        tmp.replace(self.path)

    def reset(self):
        self.id = 0
        self.lines = []
        self._write('')

    def send(self, cmd):
        self.id += 1
        self.lines.append(f'{self.id} {cmd}')
        self.lines = self.lines[-40:]
        self._write('\n'.join(self.lines) + '\n')
        return self.id


def _read_new_events(path, pos):
    if not path.exists() or path.stat().st_size <= pos:
        return pos, []
    with open(path, 'rb') as f:
        f.seek(pos)
        chunk = f.read()
    cut = chunk.rfind(b'\n') + 1
    out = []
    for ln in chunk[:cut].decode('utf-8', 'replace').splitlines():
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return pos + cut, out


def run_watch(rec_path, name, out_root, copy_root=rr.DEFAULT_COPY, init_save=None, mode='watch', capture_out=None,
              speed=1, paused=False, driver=None, console=False, shot_at=None, max_launches=30, attempts=6,
              shots_dir=None, capture_vargs=None, hint=True, extra_cfg=None, frames_limit=None, rewind='restart'):
    """观战（mode='watch'）或实录导出（mode='capture'）。返回 (session, info, 各回放进程的回放局文件列表)。
    driver(ctx)：自动化测试回调，在轮询循环里约每 20 ms 调一次；ctx 有 g（当前游戏进程）、ctl（WatchControl）、
    state()（watchstate.json）、events（watchlog 事件，带本进程收到的时间 t_py）、key(name)；返回 'quit' 结束观战。
    console=True：从 stdin 读命令（数字 = 跳到该帧；s 4 = 4 倍速；p = 暂停/继续；q = 退出；其它原样写进 watchctl）。"""
    if mode not in ('watch', 'capture'):
        raise ValueError(mode)
    run = load_run.Run(rec_path)
    if run.header.get('continued'):
        raise ValueError('观战/导出暂不支持续局录制')
    issues = replayability(run)
    env = {'IRR_HUD': '0', 'RR_MT_MOUSE': '1', 'RR_MT_FORCEPAUSE': '1'}  # phase8-mouse：观战/导出同样还原录制的鼠标；phase12 开局暂停
    capture = None
    if mode == 'watch':
        env['RR_MT_WATCH'] = '1'
    if capture_out:
        capture_out = Path(capture_out).resolve()
        capture_out.parent.mkdir(parents=True, exist_ok=True)
        old = Path(str(capture_out) + '.dll.json')
        if old.exists():  # 上一次导出留下的 DLL 记录挪开（不删），否则会被当成"已结束"
            old.replace(old.with_name(old.name + f'.old-{time.strftime("%H%M%S")}'))
        ff = find_ffmpeg()
        vargs, enc = (capture_vargs, 'custom') if capture_vargs else pick_encoder(ff)
        env.update(RR_MT_CAPTURE='1', RR_CAPTURE_OUT=str(capture_out), RR_CAPTURE_FFMPEG=ff, RR_CAPTURE_VARGS=vargs,
                   RR_CAPTURE_LOG=str(capture_out.with_suffix('.ffmpeg.log')))
        capture = dict(out=str(capture_out), ffmpeg=ff, vargs=vargs, encoder=enc)
    sess = rr.Session(name, out_root, copy_root=copy_root, raw_policy='keep', use_dll=True, init_save_from=init_save,
                      init_save_exclude=('gamestate1.dat',), extra_env=env, forced_options=WATCH_FORCED_OPTIONS)
    # phase10：录制里有 api 调用的第三方 mod（GoodTrip 等）观战/导出时同样禁用、由 mod 代为执行（与 run_replay 相同）
    api_set = api_mods(run)
    placed = disable_mods_in_copy(sess.copy, api_set)
    try:
        sess.prepare()
    except Exception:
        restore_disabled_mods(sess.copy, placed, sess.dir / 'disabled-mods')
        raise
    info = dict(source=str(rec_path), mode=mode, issues=issues, copy=str(sess.copy.root), api_mods=api_set, mods_disabled=placed,
                init_save=str(init_save) if init_save else None, capture=capture, launches=[], rewinds=[], events=[],
                menu_pause_esc=[], memory=[])
    t_session = time.time()
    console_q = []
    if console:
        import threading

        def reader():
            for ln in sys.stdin:
                console_q.append(ln.strip())
        threading.Thread(target=reader, daemon=True).start()
    try:
        info['resources'] = install_watch_resources(sess.copy)
        rr.write_control(sess.irr_dir, dict(purpose=mode, source=str(rec_path)))
        table = build_replay_table(run, sess.irr_dir / 'irr_replay.lua', api=api_set)
        info['table'] = {k: v for k, v in table.items() if k != 'cmds'}
        info['timeline'] = build_watch_timeline(run, sess.irr_dir / 'irr_watch_timeline.lua')
        # 观战：帧上限 = 录制帧数（到头 mod 自己停住）；导出：同 run_replay（死亡局多 90 帧让回放走到 GAME_END）
        died = (run.end or {}).get('reason') == 'death'
        limit_frames = run.frames if mode == 'watch' else run.frames + (90 if died else 0)
        if frames_limit:  # 测试用：只看/只导出前 frames_limit 帧
            limit_frames = min(limit_frames, int(frames_limit))
        info['frame_limit'] = limit_frames
        ctl = WatchControl(sess.irr_dir)
        start_target, cur_speed, cur_paused, cur_hint = None, speed, paused, hint
        pending_rewind = None
        log_path = sess.irr_dir / 'watchlog.jsonl'
        for k in range(1, max_launches + 1):
            # 上一个进程留下的控制/状态文件挪开（不删）
            stale = sess.irr_dir / f'stale-launch-{k:02d}'
            for pat in ('cmd-*.json', 'watchreq.txt', 'pausectl.txt', 'watchstate.json', 'capture_done.txt'):
                for f in sess.irr_dir.glob(pat):
                    stale.mkdir(exist_ok=True)
                    f.replace(stale / f.name)
            ctl.reset()
            launch = dict(index=k, start_target=start_target, t_begin=round(time.time() - t_session, 3))
            if k > 1:
                launch['restore_save'] = restore_init_save(sess, k)
            wcfg = dict(mode=mode, rewind=rewind, frames=min(run.frames, limit_frames), speed=cur_speed, paused=cur_paused, hint=cur_hint,
                        timeline=str(sess.irr_dir / 'irr_watch_timeline.lua').replace('\\', '/'),
                        shots_dir=str(Path(shots_dir).resolve()).replace('\\', '/') if shots_dir else None)
            wcfg.update(extra_cfg or {})
            if start_target and start_target > 1:
                wcfg.update(start_target=start_target, start_kind='rewind')
            if shot_at:
                wcfg['shot_at'] = {str(n): str(Path(p).resolve()).replace('\\', '/') for n, p in shot_at.items()}
            rr.write_json(sess.irr_dir / 'irr_watch.json', {kk: v for kk, v in wcfg.items() if v is not None}, compact=True)
            g = rep_run = None
            for a in range(attempts):
                g, s0 = sess.launch_and_enter(attempts=attempts)
                try:
                    time.sleep(1.0)
                    rep_run, _st = start_replay_run(g, table, limit_frames)
                    break
                except (rr.LaunchFailed, TimeoutError) as e:
                    launch.setdefault('start_failures', []).append(dict(pid=g.pid, error=repr(e)))
                    sess.mark_menu_failure(g, f'开回放局前失败：{e!r}')
                    sd = sess.irr_dir / f'stale-cmds-{k:02d}-{a + 1}'
                    sd.mkdir(exist_ok=True)
                    for f in sess.irr_dir.glob('cmd-*.json'):
                        f.replace(sd / f.name)
                    time.sleep(2.0)
            if rep_run is None:
                raise rr.LaunchFailed(f'第 {k} 次启动没能开出回放局')
            launch.update(pid=g.pid, replay_run=rep_run, t_run_open=round(time.time() - t_session, 3))
            if pending_rewind is not None:
                pending_rewind['t_new_run_open'] = time.time()
                if pending_rewind['target'] <= 1:
                    # 回到开局：新进程开出回放局就算到达（没有快进）
                    pr = pending_rewind
                    rec = {kk: v for kk, v in pr.items() if not kk.startswith('t_')}
                    rec.update(t_request=round(pr['t_request'] - t_session, 3),
                               seconds_total=round(pr['t_new_run_open'] - pr['t_request'], 2),
                               seconds_relaunch=round(pr['t_new_run_open'] - pr['t_request'], 2), seconds_fast_forward=0.0)
                    info['rewinds'].append(rec)
                    pending_rewind = None
            info['launches'].append(launch)
            log_pos = log_path.stat().st_size if log_path.exists() else 0
            esc_sent = {}
            pctl = sess.irr_dir / 'pausectl.txt'
            ended = None
            last_mem = 0
            dctx = dict(g=g, ctl=ctl, sess=sess, events=info['events'], launch=k, run=run, t0=t_session, info=info,
                        key=lambda nm, hold=0.06, _g=g: tap_key(_g, nm, hold), post=lambda nm, d, _g=g: post_key(_g, nm, d),
                        state=lambda: rr.read_json(sess.irr_dir / 'watchstate.json'))
            t_open = time.time()
            seen_state = False
            while True:
                now = time.time()
                if not g.alive():
                    ended = 'process_exit'
                    break
                # watch.lua 没加载（语法错误、DLL 不对）：mod 不会写 watchstate.json，不要无限等
                # （watchstate.json 由 mod 先写临时文件再改名，改名的瞬间可能不存在：只在一次都没见过时判定）
                if not seen_state:
                    seen_state = (sess.irr_dir / 'watchstate.json').exists()
                    if not seen_state and now - t_open > 15:
                        ended = 'watch_not_loaded'
                        info['error'] = 'watch.lua 未加载（见 lua/errors.jsonl）'
                        break
                log_pos, evs = _read_new_events(log_path, log_pos)
                for ev in evs:
                    ev['t_py'] = round(now - t_session, 3)
                    ev['launch'] = k
                    info['events'].append(ev)
                    if ev.get('ev') == 'arrive' and ev.get('kind') == 'rewind' and pending_rewind is not None:
                        pr = pending_rewind
                        rec = {kk: v for kk, v in pr.items() if not kk.startswith('t_')}
                        rec.update(t_request=round(pr['t_request'] - t_session, 3),
                                   seconds_total=round(now - pr['t_request'], 2),
                                   seconds_relaunch=round(pr['t_new_run_open'] - pr['t_request'], 2),
                                   seconds_fast_forward=round(now - pr['t_new_run_open'], 2), in_game_ms=ev.get('ms'))
                        info['rewinds'].append(rec)
                        pending_rewind = None
                # 菜单暂停：mod 请求关菜单 -> 发真实 Esc（同 run_replay）
                try:
                    req = pctl.read_text(encoding='utf-8').strip()
                except OSError:
                    req = ''
                if req:
                    t_prev = esc_sent.get(req)
                    if t_prev is None or now - t_prev > MENU_CLOSE_RETRY_S:
                        g.tap('ESC', hold=0.05)
                        esc_sent[req] = time.time()
                        info['menu_pause_esc'].append(dict(launch=k, req=req, t=round(now - t_session, 3),
                                                           retry=t_prev is not None))
                # 往回跳请求
                wr = sess.irr_dir / 'watchreq.txt'
                if wr.exists():
                    parts = wr.read_text(encoding='utf-8').split()
                    if len(parts) >= 5 and parts[0] == 'rewind':
                        T, si, pz, hn = (int(x) for x in parts[1:5])
                        inproc = len(parts) > 6 and parts[6] == 'inproc'
                        stw = rr.read_json(sess.irr_dir / 'watchstate.json') or {}
                        pending_rewind = dict(target=T, from_launch=k, why=parts[5] if len(parts) > 5 else '',
                                              t_request=now, speed=WATCH_SPEEDS[si - 1], paused=bool(pz),
                                              from_frame=stw.get('n'), mode='inproc' if inproc else 'restart')
                        if inproc:
                            # 进程内倒带：同一进程再发一次 start（seed 命令），等新的回放局开出来
                            nreq = len(list(sess.irr_dir.glob('watchreq-done-*.txt'))) + 1
                            wr.replace(sess.irr_dir / f'watchreq-done-{nreq:03d}.txt')
                            try:
                                rep_run, _st = start_replay_run(g, table, limit_frames)
                            except (rr.LaunchFailed, TimeoutError) as e:
                                info['error'] = f'进程内倒带没能开出新回放局：{e!r}'
                                ended = 'inproc_rewind_failed'
                                break
                            launch.setdefault('replay_runs', [launch['replay_run']]).append(rep_run)
                            pending_rewind['t_new_run_open'] = time.time()
                            if T <= 1:
                                pr = pending_rewind
                                rec = {kk: v for kk, v in pr.items() if not kk.startswith('t_')}
                                rec.update(t_request=round(pr['t_request'] - t_session, 3),
                                           seconds_total=round(pr['t_new_run_open'] - pr['t_request'], 2),
                                           seconds_relaunch=round(pr['t_new_run_open'] - pr['t_request'], 2),
                                           seconds_fast_forward=0.0)
                                info['rewinds'].append(rec)
                                pending_rewind = None
                            continue
                        start_target, cur_speed, cur_paused, cur_hint = T, WATCH_SPEEDS[si - 1], bool(pz), bool(hn)
                        ended = 'rewind'
                        break
                # 导出结束：mod 写 capture_done.txt；或 DLL 在最后一帧之后没有逻辑更新时自己关了 ffmpeg（写 OUT.mp4.dll.json）
                if mode == 'capture' and ((sess.irr_dir / 'capture_done.txt').exists()
                                          or Path(capture['out'] + '.dll.json').exists()):
                    ended = 'capture_done'
                    break
                if now - last_mem > 5:
                    last_mem = now
                    info['memory'].append(dict(launch=k, t=round(now - t_session, 1), **proc_memory(g.pid)))
                while console_q:
                    c = console_q.pop(0)
                    if not c:
                        continue
                    if c.isdigit():
                        ctl.send(f'goto {c}')
                    elif c in ('q', 'quit'):
                        ctl.send('quit')
                    elif c in ('p', 'pause'):
                        ctl.send('toggle')
                    elif c.startswith('s '):
                        ctl.send('speed ' + c[2:])
                    else:
                        ctl.send(c)
                if driver:
                    if driver(dctx) == 'quit':
                        ended = 'driver_quit'
                        break
                stw = rr.read_json(sess.irr_dir / 'watchstate.json') or {}
                if stw.get('quit'):
                    time.sleep(0.5)
                    ended = 'quit'
                    break
                time.sleep(0.02)
            launch['ended'] = ended
            launch['t_end'] = round(time.time() - t_session, 3)
            if ended == 'rewind':
                # 回放文件已由 mod 关好（end 行 reason=rewind）：直接结束进程，不经 WM_CLOSE，免得游戏写存档
                launch['close'] = g.close(wm_close=False, wait_s=0.3)
                continue
            launch['close'] = g.close()
            dl = g.dll_log()
            if dl:
                txt = dl.read_text(encoding='utf-8', errors='replace').splitlines()
                launch['dll_tail'] = [ln for ln in txt if 'watch:' in ln or 'capture:' in ln or 'shot' in ln][-12:]
            break
        info['seconds'] = round(time.time() - t_session, 1)
    except Exception as e:
        import traceback
        info['error'] = repr(e)
        info['traceback'] = traceback.format_exc()
    finally:
        sess.info['watch'] = info
        try:
            sess.finish()
        finally:
            info['mods_restored'] = restore_disabled_mods(sess.copy, placed, sess.dir / 'disabled-mods')
    # 每个回放进程的回放局都与录制比较（往回跳之前的那些是前缀，跳帧重跑的是从第 1 帧开始的完整重跑）
    reps = []
    for L in info['launches']:
        names = L.get('replay_runs') or [L.get('replay_run')]
        for nm in names:
            for r in sess.info['runs']:
                if r.get('lua_name') == nm:
                    L.setdefault('replay_files', []).append(str(sess.dir / r['file']))
                    reps.append(str(sess.dir / r['file']))
        if L.get('replay_files'):
            L['replay_file'] = L['replay_files'][-1]
    if capture:
        dj = rr.read_json(Path(capture['out'] + '.dll.json')) or {}
        capture['dll'] = dj
        try:
            capture['bytes'] = Path(capture['out']).stat().st_size
        except OSError:
            capture['bytes'] = None
        meta = dict(schema='isaac-footage-v1', run=str(rec_path), fps=30, first_frame=dj.get('first_frame'),
                    frames=dj.get('frames'), width=dj.get('width'), height=dj.get('height'), capture_mode='clock',
                    notes=('视频第 k 帧（从 0 数）= 录制逻辑帧 n = k + first_frame，取该帧 MC_POST_UPDATE 之后、下一次逻辑更新之前的'
                           '最后一次渲染（rr_mt.dll 在 SwapBuffers 前从默认帧缓冲的后台缓冲异步读回，Lua 在下一帧 MC_POST_UPDATE '
                           '时写出）。clock = 虚拟时钟不限速、每次渲染都提交绘制（gate 模式不提交绘制，后台缓冲里没有画面），只节流'
                           '真正的 SwapBuffers。不含观战 GUI 与 REC 标记（IRR_HUD=0）。分辨率 = 副本窗口客户区（options.ini '
                           'WindowWidth/WindowHeight，取自录制会话的开局快照）。'),
                    encoder=capture.get('encoder'), dups=dj.get('dups'), source_session=str(sess.dir),
                    seconds=info.get('seconds'))
        rr.write_json(Path(capture['out']).with_suffix('.json'), meta)
        capture['meta'] = meta
    return sess, info, reps


def main(argv=None):
    ap = argparse.ArgumentParser(description='整局录制的回放一致性检查')
    ap.add_argument('run')
    ap.add_argument('--name', default=None)
    ap.add_argument('--out-root', default=str(DEFAULT_OUT))
    ap.add_argument('--report', default=None)
    ap.add_argument('--compare', default=None, help='已有的回放文件（跳过启动游戏）')
    ap.add_argument('--continue-save', default=None, help='续局回放：续局前的存档快照目录（含 gamestate1.dat）')
    ap.add_argument('--no-dll', action='store_true', help='回放进程不注入 DLL（第二阶段行为，存档在真实 Documents）')
    ap.add_argument('--copy', default=str(rr.DEFAULT_COPY), help='隔离副本根目录（默认 game-rec-01）')
    ap.add_argument('--turbo', nargs='?', const='gate', default=None, choices=TURBO_MODES,
                    help='加速回放（默认关 = 原速）：gate（默认，虚拟时钟 + 不提交 GPU 绘制）或 clock（只虚拟时钟）')
    ap.add_argument('--no-menu-pause', action='store_true',
                    help='不在录制的菜单暂停帧注入暂停（旧行为；见 phase5-pause/REPORT.md）')
    ap.add_argument('--no-pause-align', action='store_true',
                    help='引擎自己的暂停（换房/弹窗/换层）解除后不做 tick 相对对齐（旧行为）')
    ap.add_argument('--inject-extras', action='store_true',
                    help='实验：把录制里非玩家 12 动作的非零输入查询（tick 的 extra：地图键、菜单左右/确认等）也按 (帧, tick) 注入')
    ap.add_argument('--no-inject-restart', action='store_true',
                    help='不注入录制里按住 R 重开（ACTION_RESTART）的全局查询（phase11 之前的行为；以重开结束的局最后一帧会分歧）')
    ap.add_argument('--affinity', default='auto',
                    help="回放进程的 CPU 亲和：auto（默认，一个核；多核时引擎特效偶发不确定）、0 = 不限制、或掩码如 0x8000")
    ap.add_argument('--no-mouse', action='store_true',
                    help='不还原录制的鼠标流（旧行为；phase8-mouse 之前的回放不还原，光标一直不动）')
    ap.add_argument('--init-save', default=None,
                    help='非续局回放：先把这个存档快照（录制会话 save/run-...）拷进副本存档目录（不拷 gamestate1.dat）')
    ap.add_argument('--disable-mods', default='auto',
                    help='回放时禁用、由 mod 代为执行其调用（api 行）的第三方 mod：auto（默认，录制里有 api 调用的）、none、'
                         '或逗号分隔的目录名（README §8.6）')
    ap.add_argument('--watch', action='store_true',
                    help='观战模式（README §8.3）：真实游戏画面 + 热键（暂停/变速/单帧/跳房间/跳帧/自由镜头）；需带观战钩子的 DLL')
    ap.add_argument('--watch-speed', type=float, default=1, choices=WATCH_SPEEDS, help='观战初始倍速')
    ap.add_argument('--watch-paused', action='store_true', help='观战开局即暂停')
    ap.add_argument('--no-console', action='store_true', help='观战时不从 stdin 读命令')
    ap.add_argument('--capture', default=None, metavar='OUT.mp4',
                    help='实录导出（README §8.4）：每个逻辑帧的最后一次渲染编码成 30 fps H.264，旁边写 OUT.json')
    ap.add_argument('--capture-vargs', default=None, help='ffmpeg 输出参数（默认 h264_nvenc 可用就用，否则 libx264）')
    ap.add_argument('--frames', type=int, default=None, help='观战/导出只到第 N 帧（测试用）')
    ap.add_argument('--rewind', default='restart', choices=('restart', 'inproc'),
                    help='观战往回跳：restart（默认，重启回放进程，精确）或 inproc（同一进程重开回放局，快但存档状态可能不同）')
    a = ap.parse_args(argv)
    if a.watch or a.capture:
        name = a.name or (('watch-' if a.watch else 'capture-') + Path(a.run).name.split('.')[0] + '-' + time.strftime('%H%M%S'))
        sp = int(a.watch_speed) if a.watch_speed == int(a.watch_speed) else a.watch_speed
        sess, info, reps = run_watch(a.run, name, a.out_root, copy_root=a.copy, init_save=a.init_save,
                                     mode='watch' if a.watch else 'capture', capture_out=a.capture, speed=sp,
                                     paused=a.watch_paused, console=a.watch and not a.no_console,
                                     capture_vargs=a.capture_vargs, frames_limit=a.frames, rewind=a.rewind)
        comps = []
        for r in reps:
            try:
                comps.append(compare(a.run, r))
            except Exception as e:  # load_run 解析失败（例如录制格式在改）：照样输出观战记录
                comps.append(dict(replay=r, error=repr(e), first_divergent_frame={}, mt={}, replay_frames=None))
        res = dict(watch_session=str(sess.dir), info=info, compare=comps)
        text = json.dumps(res, ensure_ascii=False, indent=1)
        if a.report:
            Path(a.report).write_text(text, encoding='utf-8')
        for c in res['compare']:
            if c.get('error'):
                print('回放局', c['replay'], '比较失败：', c['error'])
                continue
            print('回放局', c['replay'], '帧', c['replay_frames'], '三层首个分歧帧', c['first_divergent_frame'],
                  'MT 首个分歧帧', c['mt']['first_divergent_frame'], '精确一致' if c.get('exact') else '（前缀/不精确，见报告）')
        if info.get('error'):
            print('错误：', info['error'])
        return 0
    if a.compare:
        res = compare(a.run, a.compare)
    else:
        name = a.name or ('replay-' + Path(a.run).name.split('.')[0] + '-' + time.strftime('%H%M%S'))
        sess, info = run_replay(a.run, name, a.out_root, use_dll=not a.no_dll, continue_save=a.continue_save,
                                copy_root=a.copy, turbo=a.turbo, init_save=a.init_save,
                                inject_pauses=not a.no_menu_pause, align_engine_pauses=not a.no_pause_align,
                                inject_extras=a.inject_extras, mouse=not a.no_mouse, affinity=a.affinity,
                                disable_mods=a.disable_mods, inject_restart=not a.no_inject_restart)
        res = dict(replay_session=str(sess.dir), info=info)
        if info.get('replay_file'):
            res['compare'] = compare(a.run, info['replay_file'])
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if a.report:
        Path(a.report).write_text(text, encoding='utf-8')
    print(text)
    c = res.get('compare', res)
    if 'first_divergent_frame' in c:
        print('三层首个分歧帧：', c['first_divergent_frame'], '分歧帧数：', c['differing_frames'], 'MT：', c['mt'],
              f'录制 {c["record_frames"]} 帧/{c["record_end"]}，回放 {c["replay_frames"]} 帧/{c["replay_end"]}',
              '精确一致' if c.get('exact') else '不精确')
    return 0


if __name__ == '__main__':
    sys.exit(main())
