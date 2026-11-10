"""把整局录制文件（run-*.jsonl 或 .jsonl.zst，schema isaac-run-v1）的任意帧重建为 Python 对象。

  python load_run.py RUN [--frame N] [--json]      打印第 N 帧的重建结果（默认最后一帧）
  python load_run.py RUN --roundtrip               round-trip 测试（见 roundtrip()）

重建内容（DESIGN_BRIEF §6.3 第 5 项）：该帧之前（含该帧）的全部 60 Hz tick 输入（每个 tick 的完整 12 动作
(value, pressed, triggered) 状态，已按"只写变化"展开）+ 该帧状态（玩家、实体、房间头、网格、门、楼层、道具、
生成种子、鼠标、事件）。玩家/实体行在关键帧写完整行、其余帧写差分行，这里统一还原成完整行；
×100 定点字段解码为 float（像素等），完整整数行保留在 _raw 里便于逐位比较。格式定义见
rl/bridge/mod/isaac_run_recorder/README.md。
第三阶段（recorder 0.2.0）：帧行可带 mt = [cursor, hash]（全局 MT 的只读校验，FrameState.mt）；
IRR_OBS=keyframe 录制的文件在非关键帧没有 e 行，这些帧的 FrameState.entities 为 None（实体状态只能由回放重建）。
v2（recorder 0.3.0，header.format = 2，schema 字符串仍是 isaac-run-v1）：另有外观行 pa/ea（稀疏差分）、字符串字典 sd、
网格/门精灵 room.grid_spr/door_spr 与帧行 gs/ds（README 4.10）。旧字段（p/e/g/dr/dg…）与 v1 完全相同；v1 文件照常读，
外观相关字段为 None。
0.5.0（重建安全，phase13）：观测构建出错的帧是"退化帧"（帧行带 "err":1，只有 tk/sp/ms/mt 等回放必需字段，没有 r/p/e/dg；
"err":2 = 连输入也没取到），其后 mod 强制关键帧并补写一条 "resync":1 的完整 room 行。读取端：Run.degraded_frames /
input_missing_frames / frame_gaps（帧号缺口）/ suspect_frames（差分基准断了、到下一个关键帧之前解出来的值不可信的帧；0.5.0 的
文件应为空）；FrameState.degraded / suspect 标记受影响的帧，退化帧的 players/entities/digest 等为 None，绝不静默返回错值。
"""
from __future__ import annotations

import argparse
import bisect
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = 'isaac-run-v1'
_N_RE = re.compile(rb'"n":(-?\d+)')
_K_RE = re.compile(rb'^\{"k":"([a-z_]+)"')
N_COMMON = 19  # 实体行公共字段数（其后平铺该类型的附加字段）


_HEX_PREFIX = re.compile(rb'0x(-?\d)')


def repair_line(ln):
    """Lua 构建偶发把整数写成 "0x<十进制数字>"（见 isaac_run_recorder/main.lua repair()）：解析失败且含 0x 的行去掉
    这个前缀再试。返回 (行, 是否修过)。recorder 0.2.0 起 mod 写出前已修，这里兼容更早的文件。"""
    if b'0x' not in ln:
        return ln, False
    try:
        json.loads(ln)
        return ln, False
    except ValueError:
        fixed = _HEX_PREFIX.sub(rb'\1', ln)
        try:
            json.loads(fixed)
        except ValueError:
            return ln, False
        return fixed, True


def read_lines(path, repaired=None, partial=None):
    """repaired：可选列表，收集被 repair_line 修过的行号。
    partial：可选列表；文件最后一行没有换行且解析不了（进程被强杀/崩溃时 mod 的 256 KB 写缓冲恰好在行中间写出，
    phase9-robust c2-kill 实测）时丢掉这一行，并把 dict(line=行号, bytes=字节数, kind=行种类或 None) 放进来。"""
    path = Path(path)
    data = path.read_bytes()
    if path.suffix == '.zst':
        import zstandard
        data = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read()
    lines = data.split(b'\n')
    if lines and lines[-1] == b'':
        lines.pop()
    elif lines and len(lines) > 1:
        # 没有换行结尾 = 最后一行可能被截断：解析得了（含 0x 修复）就保留，否则丢掉
        tail = lines[-1].rstrip(b'\r')
        try:
            json.loads(repair_line(tail)[0])
        except ValueError:
            m = _K_RE.match(tail)
            if partial is not None:
                partial.append(dict(line=len(lines) - 1, bytes=len(lines[-1]), kind=m.group(1).decode() if m else None))
            lines.pop()
    if lines and lines[0].endswith(b'\r'):  # 文本模式写出的 CRLF（mod 早期测试文件）
        lines = [ln[:-1] if ln.endswith(b'\r') else ln for ln in lines]
    for i, ln in enumerate(lines):
        if b'0x' in ln:
            fixed, ok = repair_line(ln)
            if ok:
                lines[i] = fixed
                if repaired is not None:
                    repaired.append(i)
    return lines


def _fix(v):
    return None if v is None else v / 100.0


def _unfix(v):
    if v is None:
        return None
    return int(round(v * 100))


def _is_int(v):
    return type(v) is int


def entity_category(t):
    return {2: 'tear', 3: 'familiar', 4: 'bomb', 5: 'pickup', 7: 'laser', 8: 'knife', 9: 'projectile',
            1000: 'effect'}.get(t) or ('npc' if 10 <= t < 1000 else None)


# ---------------------------------------------------------------------------------------------- 行差分
def expand_rows(rows, prev, pcol=None):
    """把 p/e 列表还原成完整行。rows 元素：完整行（list）、差分行 [key, [[列, 值], ...]]、或只有 key（未变）。
    prev = {key: 上一帧完整行}。pcol = 位置预测的 x 列（0 起；y = pcol+1，vx/vy = pcol+2/pcol+3），None = 无预测
    （mod 早期文件）。返回 (完整行列表, 新 prev)。"""
    out, cur = [], {}
    for r in rows:
        if _is_int(r):
            full = list(prev[r])
            pairs = {}
        elif len(r) == 2 and isinstance(r[1], list):
            full = list(prev[r[0]])
            pairs = dict((c, v) for c, v in r[1])
        else:
            full = list(r)
            out.append(full)
            cur[full[0]] = full
            continue
        pos = (pcol, pcol + 1) if pcol is not None else ()
        for col, v in pairs.items():
            if col in pos:
                continue
            a = full[col]
            full[col] = a + v if (_is_int(a) and _is_int(v)) else v
        for col in pos:
            a, vel = full[col], full[col + 2]
            present = col in pairs
            v = pairs.get(col)
            if _is_int(a) and _is_int(vel) and (not present or _is_int(v)):
                full[col] = a + vel + (v if present else 0)      # 预测差：残差缺省为 0
            elif col in pairs:
                full[col] = a + v if (_is_int(a) and _is_int(v)) else v
        out.append(full)
        cur[full[0]] = full
    return out, cur


def compress_rows(full_rows, prev, kf, seed_col=None, pcol=None):
    """expand_rows 的逆：按 mod 的规则把完整行压回 p/e 列表（round-trip 校验用）。"""
    out = []
    pos = (pcol, pcol + 1) if pcol is not None else ()
    for v in full_rows:
        pv = None if kf else prev.get(v[0])
        if pv is None or len(pv) != len(v) or (seed_col is not None and pv[seed_col] != v[seed_col]):
            out.append(list(v))
            continue
        d = []
        for i in range(1, len(v)):
            a, b = pv[i], v[i]
            if i in pos and _is_int(a) and _is_int(b) and _is_int(v[i + 2]):
                r = b - (a + v[i + 2])
                if r != 0:
                    d.append([i, r])
            elif a != b:
                d.append([i, b - a if (_is_int(a) and _is_int(b)) else b])
        out.append([v[0], d] if d else v[0])
    return out


# ---------------------------------------------------------------------------------------------- v2 外观（稀疏差分）
def expand_sparse(rows, prev, keys):
    """v2 外观行 pa/ea：关键帧/新对象 = 完整行；其余只列出变化的对象（差分行 [key, [[列, 值], ...]]，规则同 p/e，
    无位置预测）；未列出的对象沿用上一帧。keys = 本帧存在的对象（p/e 行的 key，按出现顺序），不在 keys 里的对象丢弃。
    返回 (按 keys 顺序的完整行列表（缺失为 None）, 新 prev)。"""
    cur = {k: prev[k] for k in keys if k in prev}
    for r in rows or ():
        if _is_int(r):
            # 格式外的裸整数：只见于数字格式损坏被修过的行（human2b 第 10522 帧 ea 里 "0x924" 修成 "924"，原差分行已丢），
            # 按"该对象外观未变"处理（phase8-mouse）；外观不进 digest，不影响回放比对
            continue
        if len(r) == 2 and isinstance(r[1], list):   # 差分行（完整行至少 26 列）
            full = list(cur[r[0]])
            for col, v in r[1]:
                a = full[col]
                full[col] = a + v if (_is_int(a) and _is_int(v)) else v
            cur[r[0]] = full
        else:
            cur[r[0]] = list(r)
    return [cur.get(k) for k in keys], cur


def compress_sparse(full_rows, prev, kf, new_keys=()):
    """expand_sparse 的逆（round-trip 用）：full_rows = 按出现顺序的完整行；new_keys = 本帧必须写完整行的对象
    （新出现，或 e 行换成了完整行）。"""
    out = []
    for v in full_rows:
        if v is None:
            continue
        pv = prev.get(v[0])
        if kf or pv is None or len(pv) != len(v) or v[0] in new_keys:
            out.append(list(v))
            continue
        d = []
        for i in range(1, len(v)):
            a, b = pv[i], v[i]
            if a != b:
                d.append([i, b - a if (_is_int(a) and _is_int(b)) else b])
        if d:
            out.append([v[0], d])
    return out


@dataclass
class Tick:
    frame: int                 # 所属逻辑帧（本局第几帧，从 1 起）
    kind: str                  # "u" = 有逻辑更新的 tick；"i<j>" = 中间 tick
    inputs: dict               # {玩家编号: [[value, pressed, triggered] x 12]}（完整状态；None = 从未被查询）
    changed: bool              # 本 tick 是否写出了变化
    extra: list = field(default_factory=list)   # [hook, action, who, value]：非玩家/非 12 动作的非零查询


@dataclass
class FrameState:
    n: int
    gf: int
    t: int
    keyframe: bool
    header: dict
    level: dict
    room: dict                 # 当前房间行（解码后），grid/doors 已应用到本帧的变化
    room_frame: dict           # room_frame, clear, alive_enemies, room_index
    players: list
    entities: list
    items: dict                # {玩家编号: {道具 id: 数量}}
    spawns: list               # 本帧的生成 [type, variant, subtype, seed, gf]
    mouse: list                # 本帧的鼠标变化 [render, screen_x, screen_y, game_x, game_y, buttons]（坐标已 /100）
    events: list               # 本帧的 ev / hcmd / mark 行
    ticks: list                # 本帧的 tick（Tick）
    digest: list               # [玩家与房间, +非外观实体, 全量]
    raw: dict                  # 本帧原始帧行（解析后的 dict，p/e 仍是差分形式）
    mt: list = None            # [cursor, hash]：全局 MT 只读校验（带 DLL 的录制才有）
    player_rows: list = None   # 完整玩家行（整数，未解码）
    entity_rows: list = None   # 完整实体行（整数，未解码）
    inputs_before: list = None  # 本帧及之前的全部 tick（Run.frame(n, with_inputs=True) 时填）
    # v2（header.format = 2）外观；v1 文件为 None。*_appearance 与 players/entities 同序（解码后：定点 /100、字符串列
    # 换成字典里的字符串；某对象缺外观时为 None），pa_rows/ea_rows 是整数原值行
    player_appearance: list = None
    entity_appearance: list = None
    pa_rows: list = None
    ea_rows: list = None
    # 0.5.0：退化帧（观测缺省；players/entities/room_frame/digest 为 None）；suspect = 本帧解码值不可信的原因列表
    # （差分基准经过了退化帧/帧号缺口、网格基准断了还没有补写 room 行…）；正常帧为 False / None
    degraded: bool = False
    suspect: list = None


class Run:
    def __init__(self, path):
        self.path = Path(path)
        self.repaired_lines = []   # 读取时修过 "0x" 数字损坏的行号（见 repair_line）
        self.partial_tail = []     # 截断局（进程被杀/崩溃）末尾被丢掉的不完整行（见 read_lines）；正常文件为空
        self.lines = read_lines(path, self.repaired_lines, self.partial_tail)
        if not self.lines:
            raise ValueError('空文件')
        self.header = json.loads(self.lines[0])
        if self.header.get('k') != 'header' or self.header.get('schema') != SCHEMA:
            raise ValueError(f'不是 {SCHEMA} 文件')
        f = self.header['fields']
        self.fixed = f['fixed_x100']
        self.player_names = f['player']
        self.entity_names = f['entity']
        self.extra_names = f['entity_extra']
        self.extra_fixed = self.fixed.get('entity_extra') or {}
        pp = f.get('position_prediction') or {}
        self.pcol_p, self.pcol_e = pp.get('player'), pp.get('entity')  # None = 早期文件，无位置预测
        self._pfix = set(self.fixed['player'])
        self._efix = set(self.fixed['entity'])
        # v2：外观字段（header.format >= 2）
        self.format = self.header.get('format', 1)
        self.app_names = f.get('appearance')                  # None = v1 文件
        self.app_player_extra = f.get('player_appearance_extra') or []
        self._afix = set(f.get('appearance_fixed_x100') or [])
        self._astr = set(f.get('appearance_string_columns') or [])
        self.strings = {}         # 字符串字典 id -> 串（sd 条目，全局唯一编号，随机访问时直接查）
        self.frame_line = {}      # n -> 行号
        self.keyframes = []       # 关键帧帧号（升序）
        self.room_lines = []      # (行号, n)
        self.level_lines = []
        self.items_lines = []     # (行号, n, p)
        self.event_lines = {}     # n -> [行号]
        self.api_lines = []       # api 行的行号（recorder 0.4.0；旧文件为空）
        self.change_frames = []   # 有 g/dr 的帧号（升序）
        # 0.5.0：数据完整性（README §4.4 退化帧）
        self.degraded_frames = []       # 观测缺省的帧（"err":1 或 2）
        self.input_missing_frames = []  # 连 tick 输入也没有的帧（"err":2；回放不可能精确）
        self.frame_gaps = []            # 帧号缺口 [(上一帧, 下一帧)]（帧行没写出：写失败、进程中途…）
        self.resync_rooms = []          # "resync":1 的 room 行（行号, n）：不是换房，只为重置网格/门基准
        self.end = None
        self.other = []
        for i, ln in enumerate(self.lines[1:], 1):
            m = _K_RE.match(ln)
            if not m:
                self.other.append(i)
                continue
            k = m.group(1).decode()
            nm = _N_RE.search(ln)
            n = int(nm.group(1)) if nm else None
            if k == 'f':
                self.frame_line[n] = i
                if b',"kf":1' in ln[:80]:
                    self.keyframes.append(n)
                if b',"err":' in ln[:90]:
                    em = re.search(rb',"err":(\d+)', ln[:90])
                    self.degraded_frames.append(n)
                    if em and em.group(1) != b'1':
                        self.input_missing_frames.append(n)
                if b',"g":[' in ln or b',"dr":[' in ln or b',"gs":[' in ln or b',"ds":[' in ln:
                    self.change_frames.append(n)
                if b',"sd":[' in ln:
                    self._add_strings(json.loads(ln).get('sd'))
            elif k == 'room':
                self.room_lines.append((i, n))
                if b',"resync":1' in ln[:120]:
                    self.resync_rooms.append((i, n))
                if b',"sd":[' in ln:
                    self._add_strings(json.loads(ln).get('sd'))
            elif k == 'level':
                self.level_lines.append((i, n))
            elif k == 'items':
                self.items_lines.append((i, n, json.loads(ln)['p']))
            elif k in ('ev', 'hcmd', 'mark', 'api'):
                # api：第三方 mod 的效果调用（recorder 0.4.0，README §4.7）；与 ev 一样按 n 归帧
                self.event_lines.setdefault(n, []).append(i)
                if k == 'api':
                    self.api_lines.append(i)
            elif k == 'end':
                self.end = json.loads(ln)
            else:
                self.other.append(i)
        self.frames = len(self.frame_line)
        self.last_n = max(self.frame_line, default=0)   # 最后一帧的帧号（有帧号缺口时 != frames）
        self._tick_cache = {}
        ns = sorted(self.frame_line)
        prev = 0
        for n in ns:
            if n != prev + 1:
                self.frame_gaps.append((prev, n))
            prev = n
        self.suspect_frames = self._find_suspect()
        self._suspect_set = set(self.suspect_frames)

    def _find_suspect(self):
        """差分基准断了（退化帧、帧号缺口）之后解码值不可信的帧：玩家/实体/外观到下一个关键帧为止；网格/门/精灵到下一条
        room 行为止。按文件行序扫（room 行写在所属帧的帧行之前）。0.5.0 的 mod 在出错后的下一帧强制关键帧并在它之前补写
        room 行，所以正常应为空。"""
        bad = set(self.degraded_frames)
        kfs = set(self.keyframes)
        events = sorted([(i, 'f', n) for n, i in self.frame_line.items()] + [(i, 'room', n) for i, n in self.room_lines])
        out = []
        obs_broken = grid_broken = False
        prev, room_since = 0, False
        for _i, k, n in events:
            if k == 'room':
                grid_broken = False
                room_since = True
                continue
            if n != prev + 1:
                obs_broken = True
                if not room_since:
                    grid_broken = True
            prev, room_since = n, False
            if n in bad:
                obs_broken = grid_broken = True
                continue
            if n in kfs:
                obs_broken = False
            if obs_broken or grid_broken:
                out.append(n)
        return out

    def data_issues(self):
        """本局数据完整性问题（字符串列表；空 = 没发现问题）。"""
        out = []
        if self.input_missing_frames:
            out.append(f'输入缺失帧 {len(self.input_missing_frames)}（"err":2，回放不可能精确）：{self.input_missing_frames[:10]}')
        if self.degraded_frames:
            out.append(f'退化帧 {len(self.degraded_frames)}（观测缺省）：{self.degraded_frames[:10]}')
        if self.frame_gaps:
            out.append(f'帧号缺口 {len(self.frame_gaps)}：{self.frame_gaps[:10]}')
        if self.suspect_frames:
            out.append(f'解码不可信帧 {len(self.suspect_frames)}：{self.suspect_frames[:10]}')
        if self.end is not None and self.end.get('frames') is not None and self.end.get('frames') != max(self.frame_line, default=0):
            out.append(f'end.frames={self.end.get("frames")} 与最后一帧 {max(self.frame_line, default=0)} 不符')
        if self.end is not None and (self.end.get('errors') or 0) > 0:
            out.append(f'end.errors={self.end.get("errors")}')
        if self.end is not None and (self.end.get('write_errors') or 0) > 0:
            out.append(f'end.write_errors={self.end.get("write_errors")}')
        return out

    def frame_obj(self, n):
        return json.loads(self.lines[self.frame_line[n]])

    def _add_strings(self, entries):
        for sid, s in entries or ():
            if sid in self.strings and self.strings[sid] != s:
                raise ValueError(f'字符串字典 id {sid} 重复定义：{self.strings[sid]!r} / {s!r}')
            self.strings[sid] = s

    def string(self, sid):
        return None if sid is None else self.strings.get(sid)

    # ------------------------------------------------------------------ v2 外观
    def decode_appearance(self, row, player=False):
        if row is None or not self.app_names:
            return None
        names = list(self.app_names) + (list(self.app_player_extra) if player else [])
        out = {}
        for nm, v in zip(names, row):
            if nm in self._afix:
                out[nm] = _fix(v)
            elif nm in self._astr:
                out[nm] = self.string(v)
            else:
                out[nm] = v
        out['_raw'] = row
        return out

    # ------------------------------------------------------------------ 解码 / 编码
    def decode_player(self, row):
        out = {name: (_fix(v) if name in self._pfix else v) for name, v in zip(self.player_names, row)}
        out['_raw'] = row
        return out

    def decode_entity(self, row):
        out = {name: (_fix(v) if name in self._efix else v) for name, v in zip(self.entity_names, row)}
        cat = entity_category(out['type'])
        out['category'] = cat
        tail = row[N_COMMON:]
        names = self.extra_names.get(cat) if cat else None
        if names and len(tail) == len(names):
            fx = set(self.extra_fixed.get(cat, []))
            out['extra'] = {nm: (_fix(v) if nm in fx else v) for nm, v in zip(names, tail)}
        else:
            out['extra'] = None
        out['_raw'] = row
        return out

    def encode_player(self, p):
        return [(_unfix(p[nm]) if nm in self._pfix else p[nm]) for nm in self.player_names]

    def encode_entity(self, e):
        row = [(_unfix(e[nm]) if nm in self._efix else e[nm]) for nm in self.entity_names]
        if e['extra'] is None:
            row.extend(e['_raw'][N_COMMON:])
        else:
            fx = set(self.extra_fixed.get(e['category'], []))
            row.extend((_unfix(v) if nm in fx else v) for nm, v in e['extra'].items())
        return row

    def decode_room(self, o):
        r = dict(o)
        r['top_left'] = [_fix(x) for x in o['top_left']]
        r['bottom_right'] = [_fix(x) for x in o['bottom_right']]
        r['grid'] = {g[0]: g for g in o['grid']}
        r['doors'] = {d[0]: d for d in o['doors']}
        if 'grid_spr' in o:   # v2：[index, file, anim, frame] / [slot, file, anim, frame, rot, ox, oy]（字符串为字典 id）
            r['grid_spr'] = {g[0]: g for g in o['grid_spr']}
            r['door_spr'] = {d[0]: d for d in o.get('door_spr', [])}
        return r

    # ------------------------------------------------------------------ tick 展开
    @staticmethod
    def expand_ticks(n, tk, state):
        """把一帧的 tk 列表展开为 Tick 列表；state = {玩家: 当前 12x3 状态}（就地更新）。"""
        out = []
        for e in tk:
            if isinstance(e, str):
                if '-' in e:  # "iA-B"：连续未变化的中间 tick
                    a, b = e[1:].split('-')
                    for j in range(int(a), int(b) + 1):
                        out.append(Tick(n, f'i{j}', dict(state), False))
                else:
                    out.append(Tick(n, e, dict(state), False))
                continue
            kind, changed = e[0], e[1]
            for pnum, arr in changed:
                state[pnum] = arr
            extra = e[2] if len(e) > 2 else []
            out.append(Tick(n, kind, dict(state), bool(changed), extra))
        return out

    def ticks_until(self, n):
        """第 1..n 帧的全部 tick（完整状态）。"""
        state, out = {}, []
        for k in range(1, n + 1):
            if k in self.frame_line:   # 帧号缺口：那一帧的输入没有（frame_gaps）
                out.extend(self.expand_ticks(k, self.frame_obj(k).get('tk', []), state))
        return out

    def _input_state_before(self, n):
        base = (n - 1) // 500 * 500
        if base not in self._tick_cache:
            start = max([b for b in self._tick_cache if b < base], default=0)
            state = dict(self._tick_cache.get(start, {}))
            for k in range(start + 1, base + 1):
                if k in self.frame_line:
                    self.expand_ticks(k, self.frame_obj(k).get('tk', []), state)
            self._tick_cache[base] = dict(state)
        state = dict(self._tick_cache[base])
        for k in range(base + 1, n):
            if k in self.frame_line:
                self.expand_ticks(k, self.frame_obj(k).get('tk', []), state)
        return state

    # ------------------------------------------------------------------ 帧重建
    @staticmethod
    def _latest(lst, line_no):
        keys = [x[0] for x in lst]
        i = bisect.bisect_left(keys, line_no) - 1
        return lst[i] if i >= 0 else None

    def _make_state(self, o, room, level, items, prows, erows, events, ticks, parows=None, earows=None, suspect=None):
        if o.get('err'):
            # 退化帧（0.5.0）：只有回放必需字段；观测一律 None，不拿上一帧的值冒充
            return FrameState(n=o['n'], gf=o['gf'], t=o['t'], keyframe=False, header=self.header, level=level, room=None,
                              room_frame=None, players=None, entities=None, items=items, spawns=o.get('sp', []),
                              mouse=[[m[0]] + [_fix(x) for x in m[1:5]] + [m[5]] for m in o.get('ms', [])],
                              events=events, ticks=ticks, digest=None, raw=o, mt=o.get('mt'), degraded=True,
                              suspect=['degraded' if o.get('err') == 1 else 'input_missing'])
        fs = FrameState(n=o['n'], gf=o['gf'], t=o['t'], keyframe=o.get('kf') == 1, header=self.header, level=level,
                        room=room, room_frame=dict(zip(self.header['fields']['room_frame'], o['r'])),
                        suspect=suspect or None,
                        players=None if prows is None else [self.decode_player(p) for p in prows],
                        entities=None if erows is None else [self.decode_entity(e) for e in erows],
                        items=items, spawns=o.get('sp', []),
                        mouse=[[m[0]] + [_fix(x) for x in m[1:5]] + [m[5]] for m in o.get('ms', [])],
                        events=events, ticks=ticks, digest=o['dg'], raw=o, player_rows=prows, entity_rows=erows,
                        mt=o.get('mt'))
        if self.app_names:
            fs.pa_rows = parows
            fs.ea_rows = earows
            fs.player_appearance = None if parows is None else [self.decode_appearance(r, player=True) for r in parows]
            fs.entity_appearance = None if earows is None else [self.decode_appearance(r) for r in earows]
        return fs

    def _expand_app(self, o, prows, erows, ppa, pea, first):
        """v2：展开本帧外观行；返回 (parows, earows, ppa, pea)。v1 文件返回 (None, None, {}, {})。"""
        if not self.app_names:
            return None, None, {}, {}
        parows, ppa = expand_sparse(o.get('pa'), {} if first else ppa, [r[0] for r in prows])
        if erows is None:
            return parows, None, ppa, {}
        earows, pea = expand_sparse(o.get('ea'), {} if first else pea, [r[0] for r in erows])
        return parows, earows, ppa, pea

    def frame(self, n, with_inputs=False):
        """随机访问第 n 帧：从不晚于 n 的最近关键帧向前应用差分。"""
        if n not in self.frame_line:
            raise KeyError(f'没有第 {n} 帧（共 {self.frames} 帧）')
        li = self.frame_line[n]
        ki = bisect.bisect_right(self.keyframes, n) - 1
        k0 = self.keyframes[ki] if ki >= 0 else min(self.frame_line)
        pp, pe, ppa, pea = {}, {}, {}, {}
        prows = erows = parows = earows = None
        o = self.frame_obj(n)
        if not o.get('err'):
            for k in range(k0, n + 1):
                if k not in self.frame_line:
                    continue
                ok_ = self.frame_obj(k) if k != n else o
                if ok_.get('err'):
                    continue   # 退化帧没有观测；之后的值由 suspect_frames 标出
                try:
                    prows, pp = expand_rows(ok_['p'], {} if k == k0 else pp, self.pcol_p)
                    if 'e' in ok_:
                        erows, pe = expand_rows(ok_['e'], {} if k == k0 else pe, self.pcol_e)
                    else:  # IRR_OBS=keyframe 的非关键帧
                        erows = None
                    parows, earows, ppa, pea = self._expand_app(ok_, prows, erows, ppa, pea, k == k0)
                except (KeyError, IndexError):
                    # 差分引用了基准里没有的对象（基准断了）：本帧观测不可用
                    prows = erows = parows = earows = None
                    pp, pe, ppa, pea = {}, {}, {}, {}
        rl = self._latest(self.room_lines, li)
        room = self.decode_room(json.loads(self.lines[rl[0]])) if rl else None
        lv = self._latest(self.level_lines, li)
        level = json.loads(self.lines[lv[0]]) if lv else None
        if room is not None:
            i0 = bisect.bisect_left(self.change_frames, rl[1])
            for k in self.change_frames[i0:]:
                if k > n:
                    break
                if self.frame_line[k] > rl[0]:
                    apply_changes(room, o if k == n else self.frame_obj(k))
        items = {}
        for (i, _k, p) in self.items_lines:
            if i < li:
                items[p] = {it[0]: it[1] for it in json.loads(self.lines[i])['items']}
        events = [json.loads(self.lines[i]) for i in self.event_lines.get(n, [])]
        ticks = self.expand_ticks(n, o.get('tk', []), self._input_state_before(n))
        fs = self._make_state(o, room, level, items, prows, erows, events, ticks, parows, earows,
                              self._suspect_reasons(n, prows))
        if with_inputs:
            fs.inputs_before = self.ticks_until(n)
        return fs

    def iter_frames(self):
        """顺序重建所有帧（比逐个 frame(n) 快得多）。"""
        state, room, level, items, events = {}, None, None, {}, {}
        pp, pe, ppa, pea = {}, {}, {}, {}
        for ln in self.lines[1:]:
            m = _K_RE.match(ln)
            k = m.group(1).decode() if m else None
            if k == 'room':
                room = self.decode_room(json.loads(ln))
            elif k == 'level':
                level = json.loads(ln)
            elif k == 'items':
                d = json.loads(ln)
                items[d['p']] = {it[0]: it[1] for it in d['items']}
            elif k in ('ev', 'hcmd', 'mark', 'api'):
                d = json.loads(ln)
                events.setdefault(d['n'], []).append(d)
            elif k == 'f':
                o = json.loads(ln)
                kf = o.get('kf') == 1
                ticks = self.expand_ticks(o['n'], o.get('tk', []), state)
                if o.get('err'):
                    # 退化帧：观测缺省；差分基准保持上一帧（之后的帧若不是关键帧会被标成 suspect）
                    yield self._make_state(o, None, level, {p: dict(v) for p, v in items.items()},
                                           None, None, events.pop(o['n'], []), ticks)
                    continue
                try:
                    prows, pp = expand_rows(o['p'], {} if kf else pp, self.pcol_p)
                    if 'e' in o:
                        erows, pe = expand_rows(o['e'], {} if kf else pe, self.pcol_e)
                    else:  # IRR_OBS=keyframe 的非关键帧
                        erows = None
                    parows, earows, ppa, pea = self._expand_app(o, prows, erows, ppa, pea, kf)
                except (KeyError, IndexError):
                    prows = erows = parows = earows = None
                    pp, pe, ppa, pea = {}, {}, {}, {}
                if room is not None:
                    apply_changes(room, o)
                yield self._make_state(o, _copy_room(room), level, {p: dict(v) for p, v in items.items()},
                                       prows, erows, events.pop(o['n'], []), ticks, parows, earows,
                                       self._suspect_reasons(o['n'], prows))

    def _suspect_reasons(self, n, prows):
        out = []
        if n in self._suspect_set:
            out.append('base_broken')
        if prows is None:
            out.append('undecodable')
        return out or None

    # ------------------------------------------------------------------ 编码（round-trip）
    def encode_frame_line(self, fs: FrameState, prev_p, prev_e, app_prev=None):
        """app_prev（v2）：{'pa': {...}, 'ea': {...}} 上一帧外观完整行，就地更新。"""
        o = fs.raw
        out = {'k': 'f', 'n': fs.n, 'gf': fs.gf, 't': fs.t}
        if fs.degraded:
            # 退化帧（0.5.0）：键序 = mod 的写法 n, gf, t, err, tk, sp, ms, mt, sd；差分基准不变（下一帧是关键帧）
            out['err'] = o['err']
            for k in ('tk', 'sp', 'ms', 'mt', 'sd'):
                if k in o:
                    out[k] = o[k]
            return json.dumps(out, separators=(',', ':'), ensure_ascii=False).encode('utf-8'), prev_p, prev_e
        if fs.keyframe:
            out['kf'] = 1
        out['tk'] = o['tk']
        if 'sp' in o:
            out['sp'] = fs.spawns
        if 'ms' in o:
            out['ms'] = [[m[0]] + [_unfix(x) for x in m[1:5]] + [m[5]] for m in fs.mouse]
        if 'mt' in o:
            out['mt'] = fs.mt
        out['r'] = [fs.room_frame[k] for k in self.header['fields']['room_frame']]
        prows = [self.encode_player(p) for p in fs.players]
        out['p'] = compress_rows(prows, prev_p, fs.keyframe, pcol=self.pcol_p)
        if fs.entities is not None:
            erows = [self.encode_entity(e) for e in fs.entities]
            out['e'] = compress_rows(erows, prev_e, fs.keyframe, seed_col=11, pcol=self.pcol_e)
        else:
            erows = []
        if self.app_names:
            ap = app_prev if app_prev is not None else {}
            ppa = {} if fs.keyframe else ap.get('pa', {})
            pea = {} if (fs.keyframe or fs.entities is None) else ap.get('ea', {})
            if 'sd' in o:
                out['sd'] = o['sd']
            pa = compress_sparse(fs.pa_rows, ppa, fs.keyframe)
            if pa:
                out['pa'] = pa
            if fs.ea_rows is not None:
                # e 行写成完整行的实体（新出现 / 种子或行长变化），外观也写完整行
                new = {r[0] for r in erows if not (prev_e.get(r[0]) is not None and len(prev_e[r[0]]) == len(r)
                                                    and prev_e[r[0]][11] == r[11])}
                ea = compress_sparse(fs.ea_rows, pea, fs.keyframe, new)
                if ea:
                    out['ea'] = ea
            ap['pa'] = {r[0]: r for r in fs.pa_rows if r is not None}
            ap['ea'] = {r[0]: r for r in (fs.ea_rows or []) if r is not None}
        for k in ('g', 'dr', 'gs', 'ds'):
            if k in o:
                out[k] = o[k]
        out['dg'] = fs.digest
        return json.dumps(out, separators=(',', ':'), ensure_ascii=False).encode('utf-8'), \
            {r[0]: r for r in prows}, {r[0]: r for r in erows}


def _copy_room(room):
    if room is None:
        return None
    r = dict(room)
    r['grid'] = dict(room['grid'])
    r['doors'] = dict(room['doors'])
    if 'grid_spr' in room:
        r['grid_spr'] = dict(room['grid_spr'])
        r['door_spr'] = dict(room['door_spr'])
    return r


def apply_changes(room, frame_obj):
    for row in frame_obj.get('g', []):
        if len(row) == 1:
            room['grid'].pop(row[0], None)
        else:
            room['grid'][row[0]] = row
    for row in frame_obj.get('dr', []):
        if len(row) == 1:
            room['doors'].pop(row[0], None)
        else:
            room['doors'][row[0]] = row
    if 'grid_spr' in room:   # v2：网格/门精灵变化
        for row in frame_obj.get('gs', []):
            if len(row) == 1:
                room['grid_spr'].pop(row[0], None)
            else:
                room['grid_spr'][row[0]] = row
        for row in frame_obj.get('ds', []):
            if len(row) == 1:
                room['door_spr'].pop(row[0], None)
            else:
                room['door_spr'][row[0]] = row


# ---------------------------------------------------------------------------------------------- round-trip
def recompress_ticks(ticks, prev_state):
    """把展开后的 tick 按 mod 的规则重新压成 tk 列表。"""
    out = []
    pend = None
    state = dict(prev_state)

    def flush():
        nonlocal pend
        if pend:
            a, b = pend
            out.append(f'i{a}' if a == b else f'i{a}-{b}')
            pend = None
    for t in ticks:
        changed = []
        if t.changed:
            changed = [[p, s] for p, s in sorted(t.inputs.items()) if state.get(p) != s]
            state.update(t.inputs)
        if not t.changed and not t.extra:
            j = int(t.kind[1:]) if t.kind.startswith('i') else None
            if j is not None and pend and pend[1] == j - 1:
                pend = (pend[0], j)
            else:
                flush()
                if j is not None:
                    pend = (j, j)
                else:
                    out.append(t.kind)
            continue
        flush()
        e = [t.kind, changed]
        if t.extra:
            e.append(t.extra)
        out.append(e)
    flush()
    return out


def roundtrip(path, sample_every=97, verbose=True):
    """1) 每帧 解码（差分还原 + 定点转浮点）-> 编码（浮点转定点 + 按 mod 规则重新差分）与原始行逐字节相同；
    2) 抽样帧 随机访问 frame(n) 与顺序重建 iter_frames() 一致（玩家/实体/房间/网格/门/道具/tick/楼层）；
    3) tick 展开后按"只写变化"规则压回与原 tk 一致。"""
    run = Run(path)
    res = dict(file=str(path), frames=run.frames, keyframes=len(run.keyframes), line_mismatch=0,
               random_access_mismatch=0, tick_mismatch=0, checked_random=0, first_mismatch=None,
               degraded_frames=len(run.degraded_frames), suspect_frames=len(run.suspect_frames), frame_gaps=len(run.frame_gaps))
    seq = {}
    prev_state, prev_p, prev_e, app_prev = {}, {}, {}, {}
    for fs in run.iter_frames():
        enc, prev_p, prev_e = run.encode_frame_line(fs, prev_p, prev_e, app_prev)
        orig = run.lines[run.frame_line[fs.n]]
        if enc != orig:
            res['line_mismatch'] += 1
            if res['first_mismatch'] is None:
                i = next((i for i in range(min(len(enc), len(orig))) if enc[i] != orig[i]), min(len(enc), len(orig)))
                res['first_mismatch'] = dict(n=fs.n, at=i, orig=orig[max(0, i - 80):i + 80].decode(errors='replace'),
                                             enc=enc[max(0, i - 80):i + 80].decode(errors='replace'))
        if recompress_ticks(fs.ticks, prev_state) != fs.raw.get('tk', []):
            res['tick_mismatch'] += 1
        if fs.ticks:
            prev_state = dict(fs.ticks[-1].inputs)
        if fs.n % sample_every == 1 or fs.n == run.frames:
            seq[fs.n] = fs
    for n, a in seq.items():
        b = run.frame(n)
        res['checked_random'] += 1
        same = (a.players == b.players and a.entities == b.entities and a.room_frame == b.room_frame
                and (a.room or {}).get('grid') == (b.room or {}).get('grid')
                and (a.room or {}).get('doors') == (b.room or {}).get('doors')
                and a.items == b.items and a.mouse == b.mouse
                and [(t.kind, t.inputs, t.extra) for t in a.ticks] == [(t.kind, t.inputs, t.extra) for t in b.ticks]
                and (a.level or {}).get('rooms') == (b.level or {}).get('rooms')
                and a.pa_rows == b.pa_rows and a.ea_rows == b.ea_rows
                and (a.room or {}).get('grid_spr') == (b.room or {}).get('grid_spr')
                and (a.room or {}).get('door_spr') == (b.room or {}).get('door_spr'))
        if not same:
            res['random_access_mismatch'] += 1
    res['ok'] = res['line_mismatch'] == 0 and res['random_access_mismatch'] == 0 and res['tick_mismatch'] == 0
    if verbose:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description='整局录制文件的任意帧重建')
    ap.add_argument('run')
    ap.add_argument('--frame', type=int, default=None)
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--roundtrip', action='store_true')
    a = ap.parse_args(argv)
    if a.roundtrip:
        return 0 if roundtrip(a.run)['ok'] else 1
    run = Run(a.run)
    n = a.frame or run.frames
    fs = run.frame(n, with_inputs=True)
    if a.json:
        d = dict(n=fs.n, gf=fs.gf, t=fs.t, keyframe=fs.keyframe, room_frame=fs.room_frame,
                 players=[{k: v for k, v in p.items() if k != '_raw'} for p in fs.players],
                 entities=None if fs.entities is None else [{k: v for k, v in e.items() if k != '_raw'} for e in fs.entities],
                 mt=fs.mt,
                 items=fs.items, spawns=fs.spawns, mouse=fs.mouse, events=fs.events,
                 ticks=[t.__dict__ for t in fs.ticks], digest=fs.digest,
                 room={k: v for k, v in (fs.room or {}).items() if k not in ('grid', 'doors', 'grid_spr', 'door_spr')},
                 grid=list((fs.room or {}).get('grid', {}).values()), doors=list((fs.room or {}).get('doors', {}).values()),
                 inputs_before_ticks=len(fs.inputs_before), format=run.format,
                 player_appearance=None if fs.player_appearance is None else
                 [{k: v for k, v in x.items() if k != '_raw'} if x else None for x in fs.player_appearance],
                 entity_appearance=None if fs.entity_appearance is None else
                 [{k: v for k, v in x.items() if k != '_raw'} if x else None for x in fs.entity_appearance],
                 grid_spr=list((fs.room or {}).get('grid_spr', {}).values()) or None,
                 door_spr=list((fs.room or {}).get('door_spr', {}).values()) or None)
        print(json.dumps(d, ensure_ascii=False, indent=1))
    else:
        p = fs.players[0] if fs.players else {}
        print(f'{run.path.name}: {run.frames} 帧，第 {n} 帧 gf={fs.gf} 房间 {fs.room_frame}')
        print(f'  玩家0 位置=({p.get("x")}, {p.get("y")}) 心={p.get("hearts")} 道具数={p.get("collectibles")}')
        print(f'  实体 {len(fs.entities) if fs.entities is not None else "（本帧未记录：keyframe 模式）"} 个；MT {fs.mt}；网格 {len((fs.room or {}).get("grid", {}))} 格；门 {len((fs.room or {}).get("doors", {}))}')
        print(f'  本帧 tick: {[t.kind for t in fs.ticks]}；此前共 {len(fs.inputs_before)} 个 tick')
        print(f'  道具 {fs.items}；事件 {[e.get("ev") or e.get("k") for e in fs.events]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
