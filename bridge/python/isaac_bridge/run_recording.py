"""整局录制会话管理（Repentance v1.7.8a.J746 离线 GOG 构建，Windows）。

职责（DESIGN_BRIEF §6 D5/D6/D8/D10/D11，§7 E1/E2/E7）：
  * 检查/复用隔离副本（默认 experiments/run-recorder/game-rec-01），安装 isaac_run_recorder mod；
  * 默认（E1）经挂起式注入器 <副本>/rec/rr_launch.exe 加载最小 DLL <副本>/rec/rr_mt.dll（源码与构建脚本
    rl/bridge/engine/j746_rec/）：存档重定向到副本内 profile/（E2），mod 用它的导出函数做开局 MT reseed 与每帧 MT 校验；
    存档目录 = <副本>/profile/Documents/My Games/Binding of Isaac Repentance (Galaxy)/；
    use_dll=False（record_run.ps1 -NoDll）回退到第二阶段行为：直接启动 EXE，存档在真实
    %USERPROFILE%/Documents/My Games/Binding of Isaac Repentance (Galaxy)/，回放只能近似；
  * 快照存档目录（排除 *.dmp、log.txt）：会话前、每局开始（mod 写信号文件）、会话后；-InitSaveFrom、cmd_history.txt、
    options.ini 的强制与还原都对"本会话的存档目录"做；
  * 会话期间只读监视真实 Documents\\My Games\\ 下是否出现 (Galaxy) 目录，结果写进 session.json（real_documents_watch）；
  * 录制会话强制 options.ini 的 EnableDebugConsole=0（D8），结束后还原该键；MouseControl 保留用户值（D6，mod 记 mouse 流）；
    allow_console=True（record_run.ps1 -AllowConsole，phase10）时不强制，并给游戏进程设 RR_MT_CONSOLE=1：rr_mt.dll 挂引擎的
    Console::SubmitInput，玩家在控制台提交的每条命令由 mod 写成 ev cmd（src = console，带帧号），回放时在同一帧窗口执行；
  * 启动游戏（--luadebug），自动重试启动失败并计数（注入器失败、进程崩溃、log 洪水、mod 未加载），
    log 洪水检测（只认 mtime 晚于本次启动的 log.txt）；DLL 自己的 rr_mt.<pid>.log 直接写进会话 game/ 目录；
  * 退出后把每局 jsonl 压成 .zst（ARCHIVE_ZSTD）并逐字节校验、逐行解析校验；写 session.json 与会话摘要；校验通过的原始文件
    默认送进 Windows 回收站（raw_policy = recycle；失败退回 stage = 挪到 _raw_staging）。
另外提供测试 harness 客户端（文件 IPC、按种子开局、机器人按键），供回归测试与 replay_run.py 使用。

只会结束本模块自己启动的进程（按 pid/句柄）。永久删除一律不做：原始 jsonl 校验后送回收站或移到暂存目录，由用户处理。
"""
from __future__ import annotations

import ctypes as C
import datetime as _dt
import hashlib
import io
import json
import os
import platform
import random
import re
import shutil
import socket
import subprocess
import threading
import time
from ctypes import wintypes as W
from pathlib import Path

import zstandard

ISAAC_ROOT = Path(__file__).resolve().parents[4]
RL_ROOT = ISAAC_ROOT / 'rl'
MOD_SRC = RL_ROOT / 'bridge/mod/isaac_run_recorder'
MOD_NAME = 'isaac_run_recorder'
DEFAULT_COPY = ISAAC_ROOT / 'experiments/run-recorder/game-rec-01'
LOCAL_RESOURCES = ISAAC_ROOT / 'experiments/vpt-idm-v1/repentance/local-resources'
SAVE_SUBDIR = 'Documents/My Games/Binding of Isaac Repentance (Galaxy)'
# 真实用户的存档目录：-NoDll 会话（第二阶段行为）的存档目录；DLL 会话只读监视它是否出现
REAL_SAVE_DIR = Path(os.environ['USERPROFILE']) / SAVE_SUBDIR
SAVE_DIR = REAL_SAVE_DIR  # 兼容第二阶段工具（phase2/tools 引用 rr.SAVE_DIR）；新代码用 Session.save_dir
DLL_SUBDIR = 'rec'        # <副本>/rec/rr_mt.dll、rr_launch.exe（rl/bridge/engine/j746_rec/build.py 的默认输出）
DLL_NAME, LAUNCHER_NAME = 'rr_mt.dll', 'rr_launch.exe'
# CPU 亲和（phase8-mouse §H4）：多核并行时引擎对 1000.x 环境特效偶发不确定（线程竞争），录制进程里发生过一次就再也
# 无法精确回放（human2b 第 6911 帧、human2 第 10599 帧）。限制到 1 个逻辑 CPU 后同条件 16 次回放 0 次。所以带 DLL 的
# 录制会话和回放都默认设 RR_MT_AFFINITY=0x1（逻辑 CPU 0；本机最后一个逻辑 CPU 是 E 核、明显更慢），DLL 在游戏主线程
# 恢复之前 SetProcessAffinityMask。-NoDll 会话没有 DLL，设不了。
DEFAULT_AFFINITY = 'auto'


def affinity_mask(affinity):
    """'auto' = 逻辑 CPU 0 一个核（掩码 1）；'0'/'off'/None = 不限制；其它 = 掩码（十进制或 0x 十六进制）。"""
    if affinity in (None, '', '0', 'off', 0):
        return 0
    if affinity == 'auto':
        return 1
    return int(str(affinity), 0)


def affinity_env(use_dll, affinity=DEFAULT_AFFINITY):
    """录制/回放进程的 CPU 亲和环境变量：带 DLL 且掩码非 0 时 {'RR_MT_AFFINITY': '0x…'}，否则 {}。"""
    m = affinity_mask(affinity) if use_dll else 0
    return {'RR_MT_AFFINITY': hex(m)} if m else {}


def dll_affinity_line(dll_log_text):
    """DLL 日志里的 `affinity:` 行（SetProcessAffinityMask 的结果与设置后的进程掩码）；没有则 None。"""
    m = re.search(r'affinity: .*', dll_log_text)
    return m.group(0).strip() if m else None
EXE_NAME = 'isaac-ng_repentance.exe'
BINARIES = [EXE_NAME, 'Galaxy.dll', 'Lua5.3.3r.dll']
SCHEMA = 'isaac-run-v1'
LOGIC_HZ = 30
LOG_FLOOD_BYTES = 20 * 1024 * 1024
# D6（实测见 experiments/run-recorder/phase2/ACCEPTANCE.md）：Input.IsMouseBtnPressed / GetMousePosition 能读到
# PostMessage 鼠标消息，所以 mod 另记 mouse 流（帧行 "ms"），MouseControl 保留用户值（=1 时射击来自鼠标，
# 回放无法注入鼠标，replay_run.py 会标出）。录制会话只强制关闭调试控制台（D8）。
FORCED_OPTIONS = {'EnableDebugConsole': '0'}
# phase10：--allow-console 会话的强制项（不再强制关控制台；控制台命令由 DLL 钩子记录，README §8.7）
FORCED_OPTIONS_CONSOLE = {}
# 观测模式（IRR_OBS）：full（默认）/ keyframe（实体行只在关键帧写）/ fxkey（0.5.1：装饰性特效只在关键帧写，README §4.11）
OBS_MODES = ('full', 'keyframe', 'fxkey')
# 真人录制启动器（本模块 CLI 与 rl/record_run.ps1）的默认观测模式（phase15：human1/3/4 全部 44 局以 fxkey 回放重录 exact、
# 与原录像逐帧解码比较 0 差异后改为默认）。只影响启动器：mod 自身（IRR_OBS 未设置）、Session 类、回放进程（replay_run.py
# --obs）、测试 harness 的默认仍是 full。--obs full 可显式选回完整模式。
RECORD_DEFAULT_OBS = 'fxkey'


def obs_env(obs, fx_omit=None):
    """录制/回放进程的观测模式环境变量。obs = 'full' 时返回 {}（mod 默认）；fx_omit = 覆盖 fxkey 的省略名单（逗号分隔的
    EffectVariant 编号；None = mod 内置名单）。"""
    if obs in (None, '', 'full'):
        return {}
    if obs not in OBS_MODES:
        raise ValueError(f'观测模式只能是 {OBS_MODES}：{obs!r}')
    env = {'IRR_OBS': obs}
    if fx_omit is not None:
        env['IRR_FX_OMIT'] = str(fx_omit)
    return env
# 解压端的窗口上限（0.5.1，phase14 B3）：长距离匹配/大窗口压出的帧需要放宽默认上限（zstd 默认 2^27）
ZSTD_MAX_WINDOW = 1 << 31


# ---------------------------------------------------------------------------------------------- 小工具
def now_iso():
    return _dt.datetime.now().astimezone().isoformat(timespec='seconds')


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path, obj, compact=False):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    text = json.dumps(obj, ensure_ascii=False, separators=(',', ':')) if compact else json.dumps(obj, ensure_ascii=False, indent=1)
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        # errors='replace'：mod 写的文件里偶有系统代码页字节（例如 loadlib 的中文错误信息），不能因此读不出来
        return json.loads(Path(path).read_text(encoding='utf-8', errors='replace'))
    except Exception:
        return default


def junction_target(path):
    try:
        return os.readlink(path)
    except OSError:
        return None


# ---------------------------------------------------------------------------------------------- 存档目录
# 0.5.0（phase13 P2）：mod 在 MC_POST_GAME_STARTED 里把这些文件原样拷到 IRR_DIR（savecopy-<run>-<文件名>）。phase13 实测
# （NOTES §P2）：游戏在开局回调之前 3–5 ms 写完 gamestate1.dat 与 persistentgamedata1.dat（每次开局都写；退出/死亡时也写），
# 所以开局回调里读到的就是"这一局开局时的存档"。回放需要的是 persistentgamedataN.dat（新局 --init-save：解锁/道具池/难度记忆）
# 与 gamestateN.dat（续局 --continue-save）；options.ini 一并拷（回放会强制若干选项）。日期前缀的备份文件、崩溃日志不需要。
SAVE_COPY_FILES = ('gamestate1.dat', 'gamestate2.dat', 'gamestate3.dat', 'persistentgamedata1.dat', 'persistentgamedata2.dat',
                   'persistentgamedata3.dat', 'options.ini')
# 判断 Python 侧快照是否"晚了"时只看游戏在开局/退出时写的文件（options.ini 可能被选项菜单在局中改写）
LATE_CHECK_PREFIXES = ('gamestate', 'persistentgamedata')
BLOB_DIR = 'blobs'
# 0.5.1（phase14 B1）：blob 单独 zstd-19 压缩存成 <sha256>.zst（sha256 仍是原始内容的哈希）。实测 human3 34 局：存档快照原始
# 4.16 MB，按内容去重 1.55 MB，再逐个 zstd-19 0.19 MB。0.5.0 的未压缩 blob（<sha256>，无后缀）与更早的直接文件布局照常可读。
BLOB_SUFFIX = '.zst'
BLOB_LEVEL = 19


def _zstd_decompress_all(data):
    """解压一段完整的 zstd 数据（放宽窗口上限，长距离模式/大窗口的帧也能解）。"""
    return zstandard.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW).stream_reader(io.BytesIO(data)).read()


def put_blob(blobs_dir, src):
    """按内容去重存储：blobs_dir/<sha256>.zst（zstd-19，同一内容只存一份）。源文件一次读进内存，哈希与压缩都基于这份
    内容；写临时文件 -> 解压回读校验 sha256 -> 改名入库。已有同内容的 blob（压缩的 <sha>.zst 或 0.5.0 未压缩的 <sha>）就复用。
    返回 (sha256, blob 文件路径)。"""
    blobs_dir = Path(blobs_dir)
    data = Path(src).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    for cand in (blobs_dir / (sha + BLOB_SUFFIX), blobs_dir / sha):
        if cand.exists():
            return sha, cand
    blobs_dir.mkdir(parents=True, exist_ok=True)
    dst = blobs_dir / (sha + BLOB_SUFFIX)
    comp = zstandard.ZstdCompressor(level=BLOB_LEVEL, write_checksum=True, write_content_size=True).compress(data)
    tmp = blobs_dir / f'{sha}.tmp-{os.getpid()}-{threading.get_ident()}'
    tmp.write_bytes(comp)
    back = _zstd_decompress_all(tmp.read_bytes())
    if hashlib.sha256(back).hexdigest() != sha:   # 压缩/写盘出错：不入库（挪成 .mismatch-*，不删）
        os.replace(tmp, blobs_dir / f'{sha}.mismatch-{int(time.time() * 1000)}')
        raise OSError(f'blob 压缩校验失败：{src}')
    os.replace(tmp, dst)
    return sha, dst


def store_blob(blobs_dir, src, sha=None):
    """put_blob 的旧接口：返回 sha256。sha 给了就要求与读到的内容一致（不一致 = 源文件在两次读之间被改写）。"""
    got, _path = put_blob(blobs_dir, src)
    if sha and got != sha:
        raise OSError(f'内容与给定 sha256 不符（读的同时被改写？）：{src}')
    return got


def _blob_ref(dest, blob_path):
    return os.path.relpath(Path(blob_path), Path(dest)).replace('\\', '/')


class SnapFile:
    """快照里的一个文件（snapshot_files 的值）。read_bytes() 总是返回原始内容（压缩 blob 先解压）并按清单的 sha256 校验；
    copy_to(dst) 把原始内容写到 dst。path = 实际存储的文件（可能是 .zst），compressed = 是否压缩。"""

    def __init__(self, name, path, sha256=None, compressed=False, mtime=None):
        self.name, self.path, self.sha256, self.compressed = name, Path(path), sha256, compressed
        self.mtime = mtime   # 原文件的 mtime（清单里有就用；旧的直接文件布局 = 快照文件自己的 mtime，同 shutil.copy2）

    def read_bytes(self):
        data = self.path.read_bytes()
        if self.compressed:
            data = _zstd_decompress_all(data)
        if self.sha256 and hashlib.sha256(data).hexdigest() != self.sha256:
            raise OSError(f'快照文件 {self.name}（{self.path}）内容与清单的 sha256 不符')
        return data

    def copy_to(self, dst):
        dst = Path(dst)
        dst.write_bytes(self.read_bytes())
        if self.mtime:
            os.utime(dst, (self.mtime, self.mtime))
        return dst

    def __fspath__(self):   # 只给"未压缩"的情形当路径用（压缩的拿到的是 .zst 文件，调用方应改用 read_bytes/copy_to）
        if self.compressed:
            raise TypeError(f'{self.path} 是压缩 blob：用 read_bytes() / copy_to()')
        return str(self.path)

    def __repr__(self):
        return f'SnapFile({self.name!r}, {str(self.path)!r}, compressed={self.compressed})'


def snapshot_save(dest, save_dir, blobs=None):
    """复制整个存档目录（排除 *.dmp、log.txt）到 dest，返回清单（文件名、字节、mtime、sha256）。
    blobs（0.5.0）：给了就按内容去重存进 blobs/<sha256>，dest 里只写 manifest.json（files[名].blob = 相对 dest 的路径）；
    读取一律用 snapshot_files()（新旧两种布局都认）。"""
    dest, save_dir = Path(dest), Path(save_dir)
    man = dict(source=str(save_dir), dest=str(dest), time=now_iso(), time_epoch=time.time(), exists=save_dir.exists(),
               files={}, skipped=[], layout='blobs' if blobs else 'files')
    if not save_dir.exists():
        dest.mkdir(parents=True, exist_ok=True)
        write_json(dest / 'manifest.json', man)
        return man
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(save_dir.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() == '.dmp' or f.name.lower() == 'log.txt':
            man['skipped'].append(f.name)
            continue
        for _ in range(5):  # 游戏可能正在写
            try:
                st = f.stat()
                if blobs:
                    sha, bp = put_blob(blobs, f)
                    man['files'][f.name] = dict(bytes=st.st_size, mtime=st.st_mtime, sha256=sha, blob=_blob_ref(dest, bp))
                else:
                    shutil.copy2(f, dest / f.name)
                    man['files'][f.name] = dict(bytes=st.st_size, mtime=st.st_mtime, sha256=sha256_file(dest / f.name))
                break
            except OSError as e:
                man.setdefault('retries', []).append([f.name, repr(e)])
                time.sleep(0.2)
    write_json(dest / 'manifest.json', man)
    return man


def snapshot_files(snap_dir):
    """存档快照目录 -> {文件名: SnapFile}（read_bytes()/copy_to() 拿原始内容）。三种布局都认：
    0.5.1 压缩 blob（manifest.json 的 files[名].blob 指向 save/blobs/<sha256>.zst）、0.5.0 未压缩 blob（指向
    save/blobs/<sha256>）、更早的直接文件布局（文件直接放在快照目录里）；不含 manifest.json、*.dmp、log.txt。"""
    snap_dir = Path(snap_dir)
    man = read_json(snap_dir / 'manifest.json') or {}
    files = man.get('files') or {}
    if any(isinstance(v, dict) and v.get('blob') for v in files.values()):
        out = {}
        for name, v in sorted(files.items()):
            if v.get('blob'):
                path = (snap_dir / v['blob']).resolve()
                if not path.exists():
                    raise FileNotFoundError(f'快照 {snap_dir} 的 {name} 指向的 {path} 不存在')
                out[name] = SnapFile(name, path, v.get('sha256'), compressed=path.name.endswith(BLOB_SUFFIX), mtime=v.get('mtime'))
        return out
    out = {}
    for f in sorted(snap_dir.iterdir()):
        if f.is_file() and f.suffix.lower() != '.dmp' and f.name.lower() not in ('log.txt', 'manifest.json'):
            v = files.get(f.name)
            out[f.name] = SnapFile(f.name, f, v.get('sha256') if isinstance(v, dict) else None, mtime=f.stat().st_mtime)
    return out


def snapshot_status(snap_dir):
    """快照是不是"这一局开局时"的存档（phase13 P2）。返回 dict(source, late（True/False/None=无法判断）, reasons, checked)。
    0.5.0 的快照在 manifest 里有 late 字段；旧会话按同一规则从会话的开局信号推断：快照里 gamestate*/persistentgamedata* 的
    mtime 晚于本局开局信号文件的 mtime（游戏在开局回调之前写完这两类文件，之后再写就是退出/后来的局/局中存档）。"""
    snap_dir = Path(snap_dir)
    man = read_json(snap_dir / 'manifest.json') or {}
    if 'late' in man:
        return dict(source=man.get('kind', 'python'), late=man['late'], reasons=man.get('late_reasons') or [], checked='manifest')
    sess_dir, run = snap_dir.parent.parent, snap_dir.name
    sig = sess_dir / 'lua' / f'signal-{run}.start.json'
    if not sig.exists() or not man.get('files'):
        return dict(source='python', late=None, reasons=['无法判断：找不到开局信号或快照清单'], checked='none')
    t_sig = sig.stat().st_mtime
    reasons = late_reasons(man['files'], t_sig)
    return dict(source='python', late=bool(reasons), reasons=reasons, checked='inferred (pre-0.5.0 session)', signal_mtime=t_sig)


def late_reasons(files, t_sig, margin=0.05):
    """files = manifest.files；t_sig = 本局开局信号的 mtime。返回"快照内容晚于开局"的理由列表（空 = 不晚）。"""
    out = []
    for name, v in sorted(files.items()):
        if name.lower().startswith(LATE_CHECK_PREFIXES) and (v.get('mtime') or 0) > t_sig + margin:
            out.append(f'{name} 的 mtime 晚于开局 {v["mtime"] - t_sig:.3f} s（开局之后又被写过）')
    return out


def wait_save_quiet(save_dir, quiet_s=1.0, max_s=6.0):
    """等存档目录 quiet_s 秒内无变化（开局时游戏会写 persistentgamedata/gamestate）。"""
    def state():
        out = {}
        try:
            for f in save_dir.iterdir():
                if f.is_file() and f.suffix.lower() != '.dmp' and f.name.lower() != 'log.txt':
                    st = f.stat()
                    out[f.name] = (st.st_size, st.st_mtime)
        except OSError:
            pass
        return out
    t0 = time.time()
    last, since = state(), time.time()
    while time.time() - t0 < max_s:
        time.sleep(0.25)
        cur = state()
        if cur != last:
            last, since = cur, time.time()
        elif time.time() - since >= quiet_s:
            break
    return time.time() - t0


def read_options(path):
    vals = {}
    if not Path(path).exists():
        return None
    for ln in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        if '=' in ln and not ln.startswith('['):
            k, v = ln.split('=', 1)
            vals[k.strip()] = v.strip()
    return vals


def set_options(path, pairs):
    """改写 options.ini 的若干键（不存在的键追加），返回改写前的值。"""
    path = Path(path)
    text = path.read_text(encoding='utf-8', errors='replace')
    before = {}
    for k, v in pairs.items():
        m = re.search(rf'(?m)^{re.escape(k)}=(.*)$', text)
        before[k] = m.group(1).strip() if m else None
        if m:
            text = text[:m.start()] + f'{k}={v}' + text[m.end():]
        else:
            text = text.rstrip('\n') + f'\n{k}={v}\n'
    path.write_text(text, encoding='utf-8')
    return before


# ---------------------------------------------------------------------------------------------- 隔离副本
class GameCopy:
    def __init__(self, root=DEFAULT_COPY):
        self.root = Path(root).resolve()  # 绝对路径：注入器以副本 game/ 为工作目录，相对的 DLL 路径会落空
        self.game = self.root / 'game'
        self.exe = self.game / EXE_NAME
        self.mods = self.game / 'mods'
        self.profile = self.root / 'profile'
        self.profile_save_dir = self.profile / SAVE_SUBDIR   # DLL 会话的存档目录（E2）
        self.dll = self.root / DLL_SUBDIR / DLL_NAME
        self.launcher = self.root / DLL_SUBDIR / LAUNCHER_NAME

    def check_dll(self):
        """E1：最小 DLL 与注入器存在；返回 sha256 与构建记录。缺失时抛 RuntimeError（提示先构建）。"""
        for p in (self.dll, self.launcher):
            if not p.exists():
                raise RuntimeError(f'缺少 {p}：先运行 python rl/bridge/engine/j746_rec/build.py（或用 -NoDll）')
        build = read_json(self.dll.parent / 'build.json', {}) or {}
        return dict(dll=str(self.dll), dll_sha256=sha256_file(self.dll), launcher=str(self.launcher),
                    launcher_sha256=sha256_file(self.launcher), profile=str(self.profile),
                    build=dict(time=build.get('time'), ok=build.get('ok'), sources=build.get('sources')),
                    source_dir='rl/bridge/engine/j746_rec')

    def check(self):
        """副本完整性：二进制存在、resources junction 指向 SSD 副本且可读。返回描述 dict，失败抛 RuntimeError。"""
        info = dict(copy=str(self.root), binaries={})
        for b in BINARIES:
            p = self.game / b
            if not p.exists():
                raise RuntimeError(f'副本缺少 {p}')
            info['binaries'][b] = dict(bytes=p.stat().st_size, sha256=sha256_file(p))
        res = self.game / 'resources'
        target = junction_target(res)
        info['resources'] = dict(path=str(res), junction_target=target)
        if target and target.upper().replace('\\\\?\\', '').startswith('E:'):
            raise RuntimeError('resources 仍指向 E: 盘（已掉线），应指向 SSD 副本 ' + str(LOCAL_RESOURCES))
        packed = res / 'packed'
        if not packed.exists() or not any(packed.iterdir()):
            raise RuntimeError(f'resources/packed 不可读：{packed}')
        info['resources']['packed_files'] = len(list(packed.iterdir()))
        return info

    def install_mod(self):
        dst = self.mods / MOD_NAME
        dst.mkdir(parents=True, exist_ok=True)
        files = {}
        for f in sorted(MOD_SRC.iterdir()):
            if f.is_file():
                shutil.copyfile(f, dst / f.name)
                files[f.name] = sha256_file(f)
        (dst / 'recordings').mkdir(exist_ok=True)
        dis = dst / 'disable.it'
        if dis.exists():
            raise RuntimeError(f'{dis} 存在：录制 mod 被禁用')
        return dict(path=str(dst), files=files)

    def mods_manifest(self):
        out = []
        if not self.mods.exists():
            return out
        for d in sorted(self.mods.iterdir()):
            if not d.is_dir():
                continue
            m = dict(directory=d.name, disabled=(d / 'disable.it').exists())
            md = d / 'metadata.xml'
            if md.exists():
                text = md.read_text(encoding='utf-8', errors='replace')
                for tag in ['name', 'id', 'version', 'directory']:
                    mm = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.S)
                    if mm:
                        m['meta_' + tag] = mm.group(1).strip()
            files = {}
            for f in sorted(d.rglob('*')):
                if f.is_file() and 'recordings' not in f.relative_to(d).parts:
                    files[str(f.relative_to(d)).replace('\\', '/')] = sha256_file(f)
            m['files'] = files
            out.append(m)
        return out


# ---------------------------------------------------------------------------------------------- Win32
u32 = C.WinDLL('user32', use_last_error=True)
k32 = C.WinDLL('kernel32', use_last_error=True)
u32.GetClassNameW.argtypes = [W.HWND, W.LPWSTR, C.c_int]
u32.PostMessageW.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPARAM]
u32.GetWindowThreadProcessId.argtypes = [W.HWND, C.POINTER(W.DWORD)]
_EnumProc = C.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
u32.EnumWindows.argtypes = [_EnumProc, W.LPARAM]
k32.OpenProcess.restype = W.HANDLE
k32.TerminateProcess.argtypes = [W.HANDLE, W.UINT]
k32.WaitForSingleObject.argtypes = [W.HANDLE, W.DWORD]
k32.GetExitCodeProcess.argtypes = [W.HANDLE, C.POINTER(W.DWORD)]
k32.GetProcessTimes.argtypes = [W.HANDLE] + [C.POINTER(W.FILETIME)] * 4
k32.CloseHandle.argtypes = [W.HANDLE]
k32.QueryFullProcessImageNameW.argtypes = [W.HANDLE, W.DWORD, W.LPWSTR, C.POINTER(W.DWORD)]
_psapi = C.WinDLL('psapi', use_last_error=True)
_psapi.EnumProcesses.argtypes = [C.POINTER(W.DWORD), W.DWORD, C.POINTER(W.DWORD)]


def pids_of_exe(exe):
    """正在运行、映像路径等于 exe 的进程 pid 列表（只读：EnumProcesses + QueryFullProcessImageNameW）。"""
    want = os.path.normcase(os.path.abspath(str(exe)))
    arr = (W.DWORD * 4096)()
    got = W.DWORD()
    if not _psapi.EnumProcesses(arr, C.sizeof(arr), C.byref(got)):
        return []
    out = []
    for pid in arr[:got.value // C.sizeof(W.DWORD)]:
        if not pid:
            continue
        h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            continue
        try:
            buf = C.create_unicode_buffer(1024)
            n = W.DWORD(1024)
            if k32.QueryFullProcessImageNameW(h, 0, buf, C.byref(n)) and os.path.normcase(buf.value) == want:
                out.append(int(pid))
        finally:
            k32.CloseHandle(h)
    return out

# vk, scancode, extended（默认按键绑定）
KEYS = {
    'W': (0x57, 0x11, 0), 'A': (0x41, 0x1E, 0), 'S': (0x53, 0x1F, 0), 'D': (0x44, 0x20, 0),
    'LEFT': (0x25, 0x4B, 1), 'UP': (0x26, 0x48, 1), 'RIGHT': (0x27, 0x4D, 1), 'DOWN': (0x28, 0x50, 1),
    'E': (0x45, 0x12, 0), 'Q': (0x51, 0x10, 0), 'SPACE': (0x20, 0x39, 0), 'CTRL': (0x11, 0x1D, 0),
    'ENTER': (0x0D, 0x1C, 0), 'ESC': (0x1B, 0x01, 0), 'TAB': (0x09, 0x0F, 0), 'R': (0x52, 0x13, 0),
    'F': (0x46, 0x21, 0), 'P': (0x50, 0x19, 0), 'GRAVE': (0xC0, 0x29, 0),
}
# 测试机器人字母 -> 按键
BOT_KEYS = {'l': 'A', 'r': 'D', 'u': 'W', 'd': 'S', 'L': 'LEFT', 'R': 'RIGHT', 'U': 'UP', 'D': 'DOWN',
            'b': 'E', 'i': 'SPACE', 'p': 'Q', 'x': 'CTRL'}
WM_KEYDOWN, WM_KEYUP, WM_CLOSE = 0x100, 0x101, 0x10
WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_LBUTTONUP = 0x200, 0x201, 0x202


class LaunchFailed(RuntimeError):
    pass


class GameProcess:
    """一个游戏进程：启动、窗口按键、文件 IPC、关闭。只操作自己启动的 pid。"""

    def __init__(self, copy: GameCopy, irr_dir, proc_tag, log_dir, extra_env=None, save_dir=None, use_dll=False):
        self.copy = copy
        self.irr_dir = Path(irr_dir)
        self.irr_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.proc_tag = proc_tag
        self.use_dll = use_dll
        self.save_dir = Path(save_dir) if save_dir else (copy.profile_save_dir if use_dll else REAL_SAVE_DIR)
        prof = copy.profile
        env = os.environ.copy()
        # 不带 DLL 时这些重定向对存档无效（INVESTIGATION §2），只影响显卡缓存等；带 DLL 时存档由 RR_MT_PROFILE 重定向
        env.update(USERPROFILE=str(prof), APPDATA=str(prof / 'AppData/Roaming'), LOCALAPPDATA=str(prof / 'AppData/Local'),
                   TEMP=str(copy.root / 'temp'), TMP=str(copy.root / 'temp'),
                   IRR_DIR=str(self.irr_dir).replace('\\', '/'), IRR_PROC=proc_tag,
                   # 0.5.0（P2）：mod 在开局回调里把这些存档文件原样拷到 IRR_DIR
                   IRR_SAVE_DIR=str(self.save_dir).replace('\\', '/'), IRR_SAVE_FILES=','.join(SAVE_COPY_FILES))
        env_keys = ['USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'TEMP', 'TMP', 'IRR_DIR', 'IRR_PROC', 'IRR_SAVE_DIR', 'IRR_SAVE_FILES']
        # 测试用故障注入、观测模式只在调用方显式给出时生效（不从外层环境继承，免得回放进程也注入/换了观测模式）
        for k in ('IRR_FAULT', 'IRR_OBS', 'IRR_FX_OMIT'):
            env.pop(k, None)
        if use_dll:
            env.update(RR_MT_PROFILE=str(prof), RR_MT_LOG_DIR=str(self.log_dir), IRR_MT_DLL=str(copy.dll).replace('\\', '/'))
            env_keys += ['RR_MT_PROFILE', 'RR_MT_LOG_DIR', 'IRR_MT_DLL']
        else:
            env.pop('IRR_MT_DLL', None)
        env.update(extra_env or {})
        game_cmd = [str(copy.exe), '--luadebug']
        self.cmd = ([str(copy.launcher), '--launch', str(copy.dll)] + game_cmd) if use_dll else game_cmd
        self.out = open(self.log_dir / f'{proc_tag}.stdout.txt', 'wb')
        self.err = open(self.log_dir / f'{proc_tag}.stderr.txt', 'wb')
        self.t0 = time.time()
        self.down = set()
        self.cmd_id = 0
        self.closed = None
        self.handle = None
        cmdrec = dict(command=self.cmd, cwd=str(copy.game), started=self.t0, started_iso=now_iso(), use_dll=use_dll,
                      save_dir=str(self.save_dir), env={k: env[k] for k in env_keys + list((extra_env or {}).keys())})
        if use_dll:
            # 挂起式启动：注入器以 CREATE_SUSPENDED 建进程、注入 DLL、等控制块 Active 后恢复主线程并退出；
            # 游戏进程的 stdout/stderr 不经过这里（这两个文件里是注入器的输出）
            try:
                lp = subprocess.run(self.cmd, cwd=copy.game, env=env, capture_output=True, timeout=90)
            except subprocess.TimeoutExpired as e:
                self.out.close(); self.err.close()
                cmdrec['launcher'] = dict(timeout=True)
                write_json(self.log_dir / f'{proc_tag}.command.json', cmdrec)
                raise LaunchFailed(f'注入器超时：{e!r}')
            self.out.write(lp.stdout)
            self.err.write(lp.stderr)
            self.out.flush(); self.err.flush()
            so = lp.stdout.decode('utf-8', 'replace')
            cmdrec['launcher'] = dict(exit=lp.returncode, stdout=so.strip(), stderr=lp.stderr.decode('utf-8', 'replace').strip())
            m = re.search(r'PASS launched pid=(\d+)', so)
            if lp.returncode != 0 or not m:
                self.out.close(); self.err.close()
                write_json(self.log_dir / f'{proc_tag}.command.json', cmdrec)
                raise LaunchFailed(f'注入器失败 exit={lp.returncode} {cmdrec["launcher"]["stderr"][:200]}')
            self.proc = None
            self.pid = int(m.group(1))
        else:
            self.proc = subprocess.Popen(self.cmd, cwd=copy.game, env=env, stdout=self.out, stderr=self.err)
            self.pid = self.proc.pid
        self.handle = k32.OpenProcess(0x100001 | 0x0400, False, self.pid)  # SYNCHRONIZE|TERMINATE|QUERY_INFORMATION
        if not self.handle:
            raise LaunchFailed(f'OpenProcess({self.pid}) 失败')
        cmdrec['pid'] = self.pid
        write_json(self.log_dir / f'{proc_tag}.command.json', cmdrec)

    def dll_log(self):
        """DLL 自己的日志（RR_MT_LOG_DIR = 会话 game/ 目录）。"""
        p = self.log_dir / f'rr_mt.{self.pid}.log'
        return p if p.exists() else None

    # ---- 进程状态
    def alive(self):
        return k32.WaitForSingleObject(self.handle, 0) != 0

    def exit_code(self):
        code = W.DWORD()
        k32.GetExitCodeProcess(self.handle, C.byref(code))
        return code.value

    def cpu_seconds(self):
        a, b, c, d = W.FILETIME(), W.FILETIME(), W.FILETIME(), W.FILETIME()
        k32.GetProcessTimes(self.handle, C.byref(a), C.byref(b), C.byref(c), C.byref(d))
        f = lambda t: ((t.dwHighDateTime << 32) | t.dwLowDateTime) / 1e7
        return f(c), f(d)  # kernel, user

    def log_state(self):
        """本次启动之后写的 log.txt（mtime 晚于启动）才算数。"""
        p = self.save_dir / 'log.txt'
        try:
            st = p.stat()
        except FileNotFoundError:
            return None
        if st.st_mtime < self.t0 - 1:
            return None
        return st.st_size

    def check_health(self):
        if not self.alive():
            raise LaunchFailed(f'进程退出 exit=0x{self.exit_code():08X}')
        size = self.log_state()
        if size is not None and size > LOG_FLOOD_BYTES:
            raise LaunchFailed(f'log 洪水 {size} 字节')

    # ---- 窗口
    def hwnds(self):
        found = []

        def cb(h, _):
            o = W.DWORD()
            u32.GetWindowThreadProcessId(h, C.byref(o))
            b = C.create_unicode_buffer(128)
            u32.GetClassNameW(h, b, 128)
            if o.value == self.pid and b.value.startswith('GLFW'):
                found.append(h)
            return True
        u32.EnumWindows(_EnumProc(cb), 0)
        return found

    def key(self, name, down):
        vk, sc, ext = KEYS[name]
        lp = 1 | (sc << 16) | (ext << 24)
        if not down:
            lp |= (1 << 30) | (1 << 31)
        for h in self.hwnds():
            u32.PostMessageW(h, WM_KEYDOWN if down else WM_KEYUP, vk, lp)
        (self.down.add if down else self.down.discard)(name)

    def tap(self, name, hold=0.08):
        self.key(name, True)
        time.sleep(hold)
        self.key(name, False)

    def set_keys(self, names):
        names = set(names)
        for n in list(self.down - names):
            self.key(n, False)
        for n in names - self.down:
            self.key(n, True)

    def mouse(self, msg, x, y, wparam=0):
        for h in self.hwnds():
            u32.PostMessageW(h, msg, wparam, (y << 16) | x)

    # ---- mod 文件 IPC
    def boot_seen(self):
        p = self.irr_dir / 'irr_boot.json'
        try:
            return p.stat().st_mtime >= self.t0 - 1 and (read_json(p) or {}).get('proc') == self.proc_tag
        except FileNotFoundError:
            return False

    def status(self):
        s = read_json(self.irr_dir / 'status.json')
        if s and s.get('proc') == self.proc_tag:
            return s
        return None

    def wait(self, pred, timeout, poll=0.1, during=None):
        end = time.time() + timeout
        while time.time() < end:
            self.check_health()
            s = self.status()
            if s is not None and pred(s):
                return s
            if during:
                during()
            time.sleep(poll)
        raise TimeoutError(f'等待超时，最后状态 {self.status()!r}')

    def command(self, **c):
        self.cmd_id += 1
        c['id'] = self.cmd_id
        tmp = self.irr_dir / ('cmd-%06d.tmp' % self.cmd_id)
        tmp.write_text(json.dumps(c), encoding='utf-8')
        os.replace(tmp, self.irr_dir / ('cmd-%06d.json' % self.cmd_id))
        return self.cmd_id

    def wait_ack(self, cid, timeout=30, during=None):
        return self.wait(lambda s: s.get('ack', 0) >= cid, timeout, during=during)

    def enter_game(self, timeout=150):
        """连按 Enter（第 6 次起每三次夹一次 Space）直到 mod 报告局已开始。"""
        n = [0]
        nxt = [time.time() + 3]

        def press():
            if time.time() >= nxt[0]:
                self.tap('SPACE' if n[0] >= 6 and n[0] % 3 == 2 else 'ENTER')
                n[0] += 1
                nxt[0] = time.time() + 0.9
        s = self.wait(lambda s: s.get('in_game'), timeout, during=press)
        return s, n[0]

    # ---- 关闭
    def close(self, wm_close=True, wait_s=8.0):
        if self.closed is not None:
            return self.closed
        rec = dict(pid=self.pid)
        try:
            self.set_keys([])
        except Exception:
            pass
        if self.alive():
            if wm_close:
                for h in self.hwnds():
                    u32.PostMessageW(h, WM_CLOSE, 0, 0)
                rec['wm_close'] = True
            forced = k32.WaitForSingleObject(self.handle, int(wait_s * 1000)) != 0
            if forced:
                k32.TerminateProcess(self.handle, 0xDEAD)
                k32.WaitForSingleObject(self.handle, 10000)
            rec['forced'] = forced
        rec['exit_code'] = self.exit_code()
        rec['exit_hex'] = f'0x{rec["exit_code"]:08X}'
        rec['seconds'] = time.time() - self.t0
        k32.CloseHandle(self.handle)
        self.out.close()
        self.err.close()
        time.sleep(0.5)
        rec.update(self.copy_log())
        if self.use_dll:
            dl = self.dll_log()
            rec['dll_log'] = dl.name if dl else None
        write_json(self.log_dir / f'{self.proc_tag}.exit.json', rec)
        self.closed = rec
        return rec

    def wait_exit(self, poll=0.5, during=None):
        while self.alive():
            if during:
                during()
            time.sleep(poll)
        return self.close(wm_close=False)

    def copy_log(self):
        p = self.save_dir / 'log.txt'
        out = {}
        try:
            st = p.stat()
            if st.st_mtime < self.t0 - 1:
                out['log'] = 'stale'
            elif st.st_size < LOG_FLOOD_BYTES:
                shutil.copyfile(p, self.log_dir / f'{self.proc_tag}.log.txt')
                out['log'] = f'{self.proc_tag}.log.txt'
            else:
                with open(p, 'rb') as f:
                    (self.log_dir / f'{self.proc_tag}.log.head.txt').write_bytes(f.read(200000))
                out['log_flood_bytes'] = st.st_size
        except Exception as e:
            out['log_copy_error'] = repr(e)
        return out


def game_version_from_log(path):
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            for i, ln in enumerate(f):
                m = re.search(r'Binding of Isaac: (Repentance\S*|Afterbirth\S*|Rebirth\S*) (v[0-9][^\s]*)', ln)
                if m:
                    return dict(line=ln.strip(), product=m.group(1), version=m.group(2), line_no=i + 1)
                if i > 400:
                    break
    except FileNotFoundError:
        pass
    return None


def launch_with_retry(copy, irr_dir, log_dir, attempts=6, boot_timeout=90, settle_s=4.0, extra_env=None, tag_prefix='p',
                      save_dir=None, use_dll=False):
    """启动直到 mod 加载（irr_boot.json）且进程存活 settle_s 秒；失败的结束掉并重试。返回 (GameProcess, 记录列表)。
    use_dll=True：经注入器挂起启动（注入器失败 = 一次启动失败，同样重试）。"""
    log = []
    for i in range(attempts):
        tag = f'{tag_prefix}{i + 1:02d}'
        rec = dict(tag=tag, started=now_iso(), use_dll=use_dll)
        try:
            g = GameProcess(copy, irr_dir, tag, log_dir, extra_env, save_dir=save_dir, use_dll=use_dll)
        except LaunchFailed as e:
            rec['result'] = 'failed'
            rec['reason'] = str(e)
            log.append(rec)
            time.sleep(2.0)
            continue
        rec['pid'] = g.pid
        try:
            end = time.time() + boot_timeout
            while not g.boot_seen():
                g.check_health()
                if time.time() > end:
                    raise LaunchFailed('mod 未加载（boot 超时）')
                time.sleep(0.2)
            rec['boot_s'] = round(time.time() - g.t0, 2)
            t = time.time()
            while time.time() - t < settle_s:
                g.check_health()
                time.sleep(0.2)
            ver = game_version_from_log(g.save_dir / 'log.txt') if g.log_state() is not None else None
            rec['version'] = ver
            boot = read_json(Path(irr_dir) / 'irr_boot.json', {}) or {}
            rec['mod_mt'] = boot.get('mt')
            rec['obs'] = boot.get('obs')
            if use_dll:
                dl = g.dll_log()
                rec['dll_log'] = dl.name if dl else None
                if dl:
                    txt = dl.read_text(encoding='utf-8', errors='replace')
                    st = re.findall(r'status=\d+ (.*)', txt)
                    rec['dll_status'] = st[-1] if st else None
                    cl = re.findall(r'console: (?:hook installed|OFF).*', txt)
                    if cl:
                        rec['console_hook'] = cl[-1].strip()
                    # CPU 亲和（phase8-mouse）：要求设了 RR_MT_AFFINITY 时 DLL 日志里有成功的 affinity: 行
                    want = (extra_env or {}).get('RR_MT_AFFINITY')
                    rec['affinity'] = dict(requested=want, dll=dll_affinity_line(txt))
                    if want and (not rec['affinity']['dll'] or 'FAILED' in rec['affinity']['dll']):
                        raise LaunchFailed(f'CPU 亲和未生效（RR_MT_AFFINITY={want}，DLL 日志 {rec["affinity"]["dll"]!r}）')
            rec['result'] = 'ok'
            log.append(rec)
            return g, log
        except LaunchFailed as e:
            rec['result'] = 'failed'
            rec['reason'] = str(e)
            rec['close'] = g.close(wm_close=False, wait_s=1)
            log.append(rec)
            time.sleep(2.0)
    err = LaunchFailed(f'{attempts} 次启动全部失败：{[r.get("reason") for r in log]}')
    err.log = log
    raise err


# ---------------------------------------------------------------------------------------------- 归档
def iter_lines_raw(path):
    with open(path, 'rb') as f:
        for ln in f:
            yield ln


def validate_lines(lines):
    """逐行解析并检查结构：header 在首行、帧号连续、end 行（无则标记 truncated）。返回统计。"""
    st = dict(lines=0, frames=0, header=None, end=None, kinds={}, events={}, first_gf=None, last_gf=None, errors=[],
              truncated=False, bad_json=0, last_line_partial=False, mt_frames=0, repaired_lines=0, console_cmds=0, api_lines=0,
              degraded_frames=[], input_missing_frames=[], frame_gaps=[], resync_rooms=0, last_n=0)
    expect_n = 1
    for i, raw in enumerate(lines):
        st['lines'] += 1
        if not raw.strip():
            continue
        try:
            o = json.loads(raw)
        except ValueError:
            # Lua 构建偶发的 "0x<数字>" 损坏（load_run.repair_line 同一规则）：修得好就算修复行
            fixed = re.sub(rb'0x(-?\d)', rb'\1', raw) if b'0x' in raw else None
            o = None
            if fixed is not None:
                try:
                    o = json.loads(fixed)
                    st['repaired_lines'] += 1
                except ValueError:
                    o = None
        if o is None:
            # 进程崩溃时最后一行可能不完整
            st['bad_json'] += 1
            st['last_line_partial'] = True
            continue
        if st['last_line_partial']:
            st['errors'].append(f'第 {i + 1} 行之前有不完整的行')
            st['last_line_partial'] = False
        k = o.get('k')
        st['kinds'][k] = st['kinds'].get(k, 0) + 1
        if i == 0:
            if k != 'header' or o.get('schema') != SCHEMA:
                st['errors'].append('首行不是 isaac-run-v1 header')
            st['header'] = dict(run=o.get('run'), mode=o.get('mode'), continued=o.get('continued'), gf=o.get('gf'),
                                seed=((o.get('meta') or {}).get('seeds') or {}).get('start_seed_string'),
                                harness=o.get('harness') is not None, obs=o.get('obs'), fx_omit=o.get('fx_omit'),
                                recorder=o.get('recorder'),
                                flush_every=o.get('flush_every'), fault=o.get('fault'), save_copy=o.get('save_copy'),
                                mt=dict(available=(o.get('mt') or {}).get('available'), salt=(o.get('mt') or {}).get('salt'),
                                        reason=(o.get('mt') or {}).get('reason'),
                                        reseeds=[r.get('at') for r in ((o.get('mt') or {}).get('reseeds') or [])]))
        elif k == 'f':
            if 'mt' in o:
                st['mt_frames'] = st.get('mt_frames', 0) + 1
            if o['n'] != expect_n:
                st['errors'].append(f'帧号不连续：期望 {expect_n} 得到 {o["n"]}')
                st['frame_gaps'].append([expect_n - 1, o['n']])
                expect_n = o['n']
            expect_n += 1
            st['last_n'] = o['n']
            if o.get('err'):
                # 0.5.0 退化帧：观测构建出错（"err":1，输入照写）/ 连输入也没有（"err":2）
                st['degraded_frames'].append(o['n'])
                if o.get('err') != 1:
                    st['input_missing_frames'].append(o['n'])
            st['frames'] += 1
            if st['first_gf'] is None:
                st['first_gf'] = o['gf']
            st['last_gf'] = o['gf']
        elif k == 'ev':
            st['events'][o['ev']] = st['events'].get(o['ev'], 0) + 1
            if o.get('src') == 'console':
                st['console_cmds'] += 1
        elif k == 'api':
            st['api_lines'] += 1
        elif k == 'room' and o.get('resync'):
            st['resync_rooms'] += 1
        elif k == 'end':
            st['end'] = dict(reason=o.get('reason'), frames=o.get('frames'), cost=o.get('cost'), errors=o.get('errors'),
                             write_errors=o.get('write_errors'), err_frames=o.get('err_frames'),
                             room_errors=o.get('room_errors'), flush_every=o.get('flush_every'))
            if o.get('frames') != st['frames']:
                st['errors'].append(f'end.frames={o.get("frames")} 与帧行数 {st["frames"]} 不符')
    if st['end'] is None:
        st['truncated'] = True
    st['game_hours'] = st['frames'] / LOGIC_HZ / 3600
    return st


def data_issues(st, end_signal=None):
    """一局数据是否可信（0.5.0，phase13 P1）：返回问题列表（空 = 没发现问题）。`verified` 只表示压缩逐字节校验，
    这里看内容：end.errors>0（mod 在本局记过错误）、写失败、退化帧、帧号缺口、end.frames 与帧行数不符、行间不完整行。
    截断（没有 end 行）本身不算数据错误（已录的帧仍然可信），由 truncated 字段表示。"""
    out = []
    end = st.get('end') or {}
    if st.get('input_missing_frames'):
        out.append(f'输入缺失帧 {len(st["input_missing_frames"])}（回放不可能精确）：{st["input_missing_frames"][:10]}')
    if st.get('degraded_frames'):
        out.append(f'退化帧 {len(st["degraded_frames"])}（观测缺省，输入完整）：{st["degraded_frames"][:10]}')
    if st.get('frame_gaps'):
        out.append(f'帧号缺口 {len(st["frame_gaps"])}：{st["frame_gaps"][:10]}')
    if end and end.get('frames') is not None and end.get('frames') != st.get('frames'):
        out.append(f'end.frames={end.get("frames")} 与帧行数 {st.get("frames")} 不符')
    if (end.get('errors') or 0) > 0:
        out.append(f'end.errors={end.get("errors")}（见会话 lua/errors.jsonl）')
    if (end.get('write_errors') or 0) > 0:
        out.append(f'end.write_errors={end.get("write_errors")}')
    sig = end_signal or {}
    if not end and (sig.get('write_errors') or 0) > 0:
        out.append(f'结束信号 write_errors={sig.get("write_errors")}（end 行没写出）')
    if not end and (sig.get('errors') or 0) > 0:
        out.append(f'结束信号 errors={sig.get("errors")}')
    if any('之前有不完整的行' in e for e in st.get('errors') or []):
        out.append('文件中间有不完整的行')
    return out


def process_exit_summary(closed):
    """GameProcess.close() 的记录 -> session.json 的 game_exit（phase9-robust：以前只在 game/<tag>.exit.json 里）。
    abnormal = 退出码非 0（被任务管理器/TerminateProcess 结束是 1，崩溃是 0xC0000005/0xC0000409 等）；
    forced_by_recorder = 我们发 WM_CLOSE 后等不到退出、自己 TerminateProcess(0xDEAD) 的。"""
    if not closed:
        return None
    code = closed.get('exit_code')
    return dict(pid=closed.get('pid'), exit_code=code, exit_hex=closed.get('exit_hex'),
                wm_close_sent=bool(closed.get('wm_close')), forced_by_recorder=bool(closed.get('forced')),
                seconds=round(closed.get('seconds') or 0, 1), abnormal=code not in (0, None))


def truncated_note(game_exit, end_signal=None, flush_every=None):
    """截断局（没有 end 行）的说明：带上游戏进程的退出码（若知道）。mod 每 flush_every 帧（0.5.0 默认 30 帧 ≈ 1 秒；0.4.x
    是 300 帧 ≈ 10 秒）、进房、暂停时 flush，另外 256 KB 写缓冲满了也会写出（可能停在行中间，读取时丢掉，见
    load_run.read_lines）。有结束信号（mod 关了局）却没有 end 行 = end 行没写出去（磁盘满/写失败），不是进程异常结束。"""
    if end_signal:
        if (end_signal.get('write_errors') or 0) > 0:
            return (f'没有 end 行，但 mod 已关局（结束信号 reason={end_signal.get("reason")}）且记了 '
                    f'{end_signal.get("write_errors")} 次写失败：磁盘满/写失败，不是进程异常结束')
        return f'没有 end 行，但 mod 已关局（结束信号 reason={end_signal.get("reason")}）：end 行没写到文件里（写失败？）'
    secs = '约 1 秒' if (flush_every or 300) <= 30 else '约 10 秒'
    s = f'没有 end 行：进程异常结束，缓冲区里最后{secs}（最后一次 flush 之后）的数据丢失'
    if game_exit and game_exit.get('abnormal'):
        s += f'（游戏进程退出码 {game_exit.get("exit_hex")}）'
    return s


# 归档压缩配置（0.5.1，phase14 B3：真人长局上的耗时/体积表见 experiments/run-recorder/phase14-space/NOTES.md §B3）。
# 键 = zstandard.ZstdCompressionParameters.from_level 的参数（level 另取）。解压端（load_run、open_run_lines、这里的回读校验）
# 一律放宽窗口上限 ZSTD_MAX_WINDOW，旧的单线程 zstd-19 文件照常读。
ARCHIVE_ZSTD = dict(level=19, threads=-1, job_size=4 << 20, overlap_log=9, enable_ldm=True, window_log=27)


def archive_compressor(cfg=None):
    cfg = dict(cfg or ARCHIVE_ZSTD)
    level = cfg.pop('level')
    params = zstandard.ZstdCompressionParameters.from_level(level, write_checksum=True, write_content_size=True, **cfg)
    return zstandard.ZstdCompressor(compression_params=params)


def compress_and_verify(raw, dst, level=None, cfg=None):
    """jsonl -> .zst（配置 ARCHIVE_ZSTD，带内容校验和与原始长度），解压回读逐字节比较 sha256，并逐行解析校验。
    level（旧接口）：给了就用单线程的该级别（0.5.0 及以前的行为）。"""
    raw, dst = Path(raw), Path(dst)
    if cfg is None:
        cfg = dict(level=level, threads=0) if level is not None else dict(ARCHIVE_ZSTD)
    raw_sha = sha256_file(raw)
    cctx = archive_compressor(cfg)
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    t = time.time()
    with open(raw, 'rb') as fi, open(tmp, 'wb') as fo:
        cctx.copy_stream(fi, fo, size=raw.stat().st_size)
    os.replace(tmp, dst)
    secs = time.time() - t
    h = hashlib.sha256()
    dctx = zstandard.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW)
    with open(dst, 'rb') as f, dctx.stream_reader(f) as r:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            h.update(b)
    if h.hexdigest() != raw_sha:
        raise RuntimeError(f'压缩校验失败：{dst}')
    st = validate_lines(iter_lines_raw(raw))
    return dict(raw_bytes=raw.stat().st_size, zst_bytes=dst.stat().st_size, raw_sha256=raw_sha, zst_sha256=sha256_file(dst),
                compress_s=round(secs, 2), level=cfg.get('level'), zstd=cfg, verified=True, validation=st)


def open_run_lines(path):
    """逐行读取 .jsonl 或 .jsonl.zst（bytes 行）。"""
    path = Path(path)
    if path.suffix == '.zst':
        import io
        f = open(path, 'rb')
        r = zstandard.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW).stream_reader(f)
        return io.BufferedReader(r)
    return open(path, 'rb')


def stage_raw(raw, staging_root, session_name):
    """原始 jsonl 校验后移到暂存目录（不删除；由用户确认后自行清理）。"""
    d = Path(staging_root) / session_name
    d.mkdir(parents=True, exist_ok=True)
    target = d / Path(raw).name
    shutil.move(str(raw), str(target))
    return str(target)


# ---------------------------------------------------------------------------------------------- 回收站（0.5.1，phase14 B2）
# 原始 jsonl 与 mod 的开局存档拷贝在"压缩并逐字节校验通过"之后送进 Windows 回收站（不是永久删除，用户可以还原/自己清空）。
# SHFileOperationW(FO_DELETE, FOF_ALLOWUNDO)：FOF_ALLOWUNDO = 进回收站；FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI = 不弹窗。
# 风险：文件超过回收站容量、回收站被设为"直接删除"、或不是本地固定磁盘时，shell 会改成永久删除。所以先检查（盘类型、
# 注册表 BitBucket 的 NukeOnDelete/MaxCapacity），不满足就不调用、退回 stage；另加 FOF_WANTNUKEWARNING 兜底（真要永久删除时
# 弹确认框而不是悄悄删）。调用后核对：文件确实不在原处、回收站的项目数增加了。
FO_DELETE = 3
FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI, FOF_WANTNUKEWARNING = 0x4, 0x10, 0x40, 0x400, 0x4000


class _SHFILEOPSTRUCTW(C.Structure):
    _fields_ = [('hwnd', W.HWND), ('wFunc', W.UINT), ('pFrom', W.LPCWSTR), ('pTo', W.LPCWSTR), ('fFlags', W.WORD),
                ('fAnyOperationsAborted', W.BOOL), ('hNameMappings', C.c_void_p), ('lpszProgressTitle', W.LPCWSTR)]


class _SHQUERYRBINFO(C.Structure):
    _fields_ = [('cbSize', W.DWORD), ('i64Size', C.c_int64), ('i64NumItems', C.c_int64)]


def recycle_bin_state(drive_root):
    """回收站在 drive_root（如 'D:\\\\'）上的 (项目数, 字节数)；查不到返回 None。"""
    try:
        info = _SHQUERYRBINFO()
        info.cbSize = C.sizeof(info)
        if C.windll.shell32.SHQueryRecycleBinW(C.c_wchar_p(drive_root), C.byref(info)) != 0:
            return None
        return int(info.i64NumItems), int(info.i64Size)
    except Exception:   # noqa: BLE001
        return None


def recycle_bin_policy(drive_root):
    """drive_root 上回收站的设置：dict(drive_type, nuke_on_delete, max_capacity_mb, volume, source)。只读注册表与卷信息。"""
    out = dict(drive_type=None, nuke_on_delete=None, max_capacity_mb=None, volume=None, source=None)
    try:
        out['drive_type'] = int(k32.GetDriveTypeW(C.c_wchar_p(drive_root)))
        buf = C.create_unicode_buffer(64)
        if k32.GetVolumeNameForVolumeMountPointW(C.c_wchar_p(drive_root), buf, 64):
            m = re.search(r'\{[0-9a-fA-F-]+\}', buf.value)
            out['volume'] = m.group(0) if m else None
    except Exception:   # noqa: BLE001
        pass
    try:
        import winreg
        base = r'Software\Microsoft\Windows\CurrentVersion\Explorer\BitBucket'
        keys = ([base + '\\Volume\\' + out['volume']] if out['volume'] else []) + [base]
        for kp in keys:
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, kp) as k:
                    for nm, field in (('NukeOnDelete', 'nuke_on_delete'), ('MaxCapacity', 'max_capacity_mb')):
                        try:
                            v, _t = winreg.QueryValueEx(k, nm)
                            if out[field] is None:
                                out[field] = int(v)
                                out['source'] = out['source'] or kp
                        except OSError:
                            pass
            except OSError:
                pass
    except Exception:   # noqa: BLE001
        pass
    return out


def recycle_files(paths):
    """把 paths（同一个盘上的文件）送进回收站。返回 dict(ok, files, error, bin_before, bin_after, policy)。
    ok=False 时文件原样留在原处（调用方退回 stage）。"""
    paths = [Path(p).resolve() for p in paths]
    res = dict(ok=False, files=[str(p) for p in paths], error=None)
    if not paths:
        res['ok'] = True
        return res
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        res['error'] = f'文件不存在：{missing}'
        return res
    drive = paths[0].anchor
    if any(p.anchor.lower() != drive.lower() for p in paths):
        res['error'] = '文件不在同一个盘上'
        return res
    pol = recycle_bin_policy(drive)
    res['policy'] = pol
    total = sum(p.stat().st_size for p in paths)
    if pol.get('drive_type') != 3:   # DRIVE_FIXED：网络盘/可移动盘没有回收站，shell 会直接删除
        res['error'] = f'{drive} 不是本地固定磁盘（GetDriveType={pol.get("drive_type")}），没有回收站'
        return res
    if pol.get('nuke_on_delete'):
        res['error'] = f'{drive} 的回收站设置为"不移到回收站，直接删除"（NukeOnDelete=1）'
        return res
    if pol.get('max_capacity_mb') is not None and total > pol['max_capacity_mb'] * 1024 * 1024:
        res['error'] = f'这批文件 {total} B 超过 {drive} 回收站容量 {pol["max_capacity_mb"]} MB（会被永久删除）'
        return res
    before = recycle_bin_state(drive)
    res['bin_before'] = before
    op = _SHFILEOPSTRUCTW()
    op.hwnd = None
    op.wFunc = FO_DELETE
    op.pFrom = '\0'.join(str(p) for p in paths) + '\0'   # 双 0 结尾（ctypes 再补一个 0）
    op.pTo = None
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI | FOF_WANTNUKEWARNING
    try:
        rc = C.windll.shell32.SHFileOperationW(C.byref(op))
    except Exception as e:   # noqa: BLE001
        res['error'] = f'SHFileOperationW 调用失败：{e!r}'
        return res
    after = recycle_bin_state(drive)
    res['bin_after'] = after
    still = [str(p) for p in paths if p.exists()]
    if rc != 0 or op.fAnyOperationsAborted or still:
        res['error'] = f'SHFileOperationW 返回 0x{rc:X}，aborted={bool(op.fAnyOperationsAborted)}，仍在原处：{still}'
        return res
    if before and after and after[0] < before[0] + 1:
        # 文件已不在原处但回收站项目数没涨：可能被永久删除了（或回收站同时被清理）。如实记录，不当作失败（文件已无法挪回）
        res['warning'] = f'回收站项目数 {before[0]} -> {after[0]}，没有按预期增加'
    res['ok'] = True
    return res


# ---------------------------------------------------------------------------------------------- 会话
class SaveWatcher(threading.Thread):
    """监视 mod 的开局信号文件，记录每局开局时的存档（phase13 P2 重写）。

    主快照：mod 在开局回调里同步拷出的文件（信号里的 save_copy；kind = "mod"，不会晚）。按内容去重存进 save/blobs/<sha256>，
    save/<run>/manifest.json 只记清单。拍不到 mod 拷贝（旧 mod / 拷贝失败）时退回 Python 快照（等存档目录安静后复制），并检测
    是否"晚了"：拍完时已有更新的开局信号，或 gamestate*/persistentgamedata* 的 mtime 晚于本局开局信号（late_reasons）。
    check=True 时对有 mod 拷贝的局另做一次 Python 核对快照（只记哈希与是否一致，内容也去重入库），用来证明两者相同；已有更新的
    开局信号时不做（会晚）。所有新信号先处理 mod 拷贝（毫秒级），再做核对，所以按住 R 连续重开也不会排队拍错。"""

    def __init__(self, irr_dir, save_root, records, save_dir, check=True):
        super().__init__(daemon=True)
        self.irr_dir = Path(irr_dir)
        self.save_root = Path(save_root)
        self.blobs = self.save_root / BLOB_DIR
        self.save_dir = Path(save_dir)
        self.records = records
        self.seen = set()
        self.stop_flag = False
        self.check = check
        self.pending_check = []

    def _signals(self):
        out = []
        for sig in self.irr_dir.glob('signal-*.start.json'):
            try:
                out.append((sig.stat().st_mtime, sig))
            except OSError:
                pass
        out.sort()
        return out

    def ingest_mod_copy(self, run, sc, t_sig):
        """mod 的开局拷贝 -> blobs + 清单。返回清单 dict；拷贝不可用时返回 None。"""
        files = sc.get('files') if isinstance(sc, dict) else None
        if not isinstance(files, dict):
            return None
        dest = self.save_root / run
        man = dict(kind='mod', run=run, time=now_iso(), signal_mtime=t_sig, source=str(self.save_dir), dest=str(dest),
                   files={}, absent=[], errors=[], copy_ms=sc.get('ms'), copy_cpu_ms=sc.get('cpu_ms'), copy_bytes=sc.get('bytes'),
                   layout='blobs', late=False, late_reasons=[],
                   note='mod 在 MC_POST_GAME_STARTED 里同步拷出（开局那一刻的存档）')
        for name, v in sorted(files.items()):
            if v is None:
                man['absent'].append(name)
                continue
            src = self.irr_dir / f'savecopy-{run}-{name}'
            if v == -1 or not src.exists():
                man['errors'].append(f'{name}: mod 写副本失败')
                continue
            if src.stat().st_size != v:
                man['errors'].append(f'{name}: 大小 {src.stat().st_size} 与 mod 报告的 {v} 不符')
                continue
            sha, bp = put_blob(self.blobs, src)
            man['files'][name] = dict(bytes=v, sha256=sha, blob=_blob_ref(dest, bp), copied_from=src.name)
        if man['errors']:
            return man   # 有错误也返回（记进 session.json），调用方据此退回 Python 快照
        dest.mkdir(parents=True, exist_ok=True)
        write_json(dest / 'manifest.json', man)
        return man

    def python_snapshot(self, run, t_sig, dest_name='manifest.json', as_check=False):
        """Python 侧快照（等安静 -> 去重复制）。返回清单（含 late / late_reasons）。"""
        dest = self.save_root / run
        n_before = len(self._signals())
        waited = wait_save_quiet(self.save_dir)
        tmp_dir = dest / ('python-check' if as_check else '')
        man = snapshot_save(tmp_dir, self.save_dir, blobs=self.blobs)
        reasons = late_reasons(man['files'], t_sig)
        newer = [sg.name for mt, sg in self._signals() if mt > t_sig + 1e-6]
        if newer:
            reasons.append(f'拍快照时已有更新的开局信号 {newer[:3]}（共 {len(newer)} 个）')
        man.update(kind='python', run=run, signal_mtime=t_sig, waited_quiet_s=round(waited, 2), late=bool(reasons),
                   late_reasons=reasons, signals_before=n_before)
        write_json(tmp_dir / 'manifest.json', man)
        return man

    def poll_once(self):
        new = []
        for t_sig, sig in self._signals():
            if sig.name in self.seen:
                continue
            self.seen.add(sig.name)
            run = sig.name[len('signal-'):-len('.start.json')]
            info = read_json(sig, {}) or {}
            rec = dict(signal=info, signal_seen=now_iso(), signal_mtime=t_sig, snapshot=str(self.save_root / run))
            try:
                man = self.ingest_mod_copy(run, info.get('save_copy'), t_sig)
            except Exception as e:   # 不让监视线程死掉
                man = dict(kind='mod', errors=[repr(e)])
            if man is not None and not man.get('errors'):
                rec.update(kind='mod', late=False, late_reasons=[], files=sorted(man['files']), absent=man['absent'],
                           copy_ms=man.get('copy_ms'), copy_bytes=man.get('copy_bytes'))
                if self.check:
                    self.pending_check.append(run)
            else:
                rec['mod_copy_errors'] = (man or {}).get('errors') if man else 'no save_copy in signal (recorder < 0.5.0?)'
                new.append(run)
            self.records[run] = rec
        # 没有 mod 拷贝的局：Python 快照（可能晚，检测并标记）
        for run in new:
            rec = self.records[run]
            man = self.python_snapshot(run, rec['signal_mtime'])
            rec.update(kind='python', late=man['late'], late_reasons=man['late_reasons'], files=sorted(man['files']),
                       waited_quiet_s=man.get('waited_quiet_s'), snapshot_delay_s=round(time.time() - rec['signal_mtime'], 2))
        # 有 mod 拷贝的局：Python 核对快照（已有更新的开局信号就不做）
        while self.pending_check:
            run = self.pending_check.pop(0)
            rec = self.records[run]
            newer = [sg for mt, sg in self._signals() if mt > rec['signal_mtime'] + 1e-6]
            if newer:
                rec['python_check'] = dict(skipped='已有更新的开局信号（核对快照会晚）')
                continue
            chk = self.python_snapshot(run, rec['signal_mtime'], as_check=True)
            mod_files = (read_json(self.save_root / run / 'manifest.json') or {}).get('files') or {}
            common = sorted(set(mod_files) & set(chk['files']))
            diff = [n for n in common if mod_files[n]['sha256'] != chk['files'][n]['sha256']]
            only_check = sorted(n for n in set(chk['files']) - set(mod_files)
                                if n.lower() in SAVE_COPY_FILES)
            rec['python_check'] = dict(late=chk['late'], late_reasons=chk['late_reasons'], compared=common, differing=diff,
                                       only_in_python=only_check, same_as_mod=not diff and not only_check,
                                       waited_quiet_s=chk.get('waited_quiet_s'))

    def run(self):
        while not self.stop_flag:
            try:
                self.poll_once()
            except Exception as e:  # 不让监视线程死掉
                self.records.setdefault('_errors', []).append(repr(e))
            time.sleep(0.25)


class RealDocsWatcher(threading.Thread):
    """E2：只读监视真实 %USERPROFILE%\\Documents\\My Games\\ 下是否出现 (Galaxy) 存档目录（只做 stat / listdir，
    从不写）。结果（开始/结束时是否存在、My Games 目录清单、每次存在性变化）写进 session.json 的 real_documents_watch。"""

    def __init__(self, interval=2.0):
        super().__init__(daemon=True)
        self.path = REAL_SAVE_DIR
        self.interval = interval
        self.stop_flag = False
        self.polls = 0
        self.changes = []
        self.ever_existed = False
        self.start_state = self.state()
        self.last = self.start_state['exists']
        self.ever_existed = self.last

    def state(self):
        mg = self.path.parent
        try:
            listing = sorted(x.name for x in mg.iterdir())
        except OSError:
            listing = None
        st = dict(time=now_iso(), exists=self.path.exists(), my_games=listing)
        if st['exists']:
            try:
                st['files'] = sorted(x.name for x in self.path.iterdir())
            except OSError:
                pass
        return st

    def run(self):
        while not self.stop_flag:
            self.poll()
            time.sleep(self.interval)

    def poll(self):
        self.polls += 1
        ex = self.path.exists()
        if ex != self.last:
            self.changes.append(self.state())
            self.last = ex
        self.ever_existed = self.ever_existed or ex

    def result(self):
        self.stop_flag = True
        self.poll()
        return dict(path=str(self.path), interval_s=self.interval, polls=self.polls, at_start=self.start_state,
                    at_end=self.state(), changes=self.changes, ever_existed=self.ever_existed,
                    appeared_during_session=(not self.start_state['exists']) and self.ever_existed,
                    note='只读监视（stat/listdir），不写真实 Documents')


def host_info():
    return dict(hostname=socket.gethostname(), platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), python=platform.python_version(), cpu_count=os.cpu_count())


class Session:
    """一次录制会话（一个 session 目录；其中可以有多次游戏进程启动、多局）。"""

    def __init__(self, name, out_root, copy_root=DEFAULT_COPY, init_save_from=None, raw_policy='recycle', harness=None,
                 enforce_options=True, use_dll=True, extra_env=None, forced_options=None, init_save_exclude=(),
                 affinity=DEFAULT_AFFINITY, allow_console=False):
        self.name = name
        self.out_root = Path(out_root).resolve()  # 绝对路径：IRR_DIR 交给游戏进程（工作目录是副本 game/），相对路径会落空
        self.dir = self.out_root / name
        if self.dir.exists():
            raise RuntimeError(f'会话目录已存在：{self.dir}')
        self.dir.mkdir(parents=True)
        self.irr_dir = self.dir / 'lua'
        self.irr_dir.mkdir()
        self.game_logs = self.dir / 'game'
        self.copy = GameCopy(copy_root)
        self.use_dll = use_dll
        self.extra_env = dict(extra_env or {})
        # CPU 亲和（phase8-mouse §H4）：带 DLL 的会话默认把游戏进程限制到 1 个逻辑 CPU，录下的局才可能精确回放
        self.affinity = affinity_env(use_dll, affinity).get('RR_MT_AFFINITY')
        if self.affinity:
            self.extra_env.setdefault('RR_MT_AFFINITY', self.affinity)
        # E2：带 DLL 时存档在副本 profile/ 下；-NoDll 时是真实 Documents
        self.save_dir = self.copy.profile_save_dir if use_dll else REAL_SAVE_DIR
        self.init_save_from = Path(init_save_from).resolve() if init_save_from else None
        # init_save_from 里不拷的文件名（小写；例如非续局回放不要 gamestate1.dat，免得菜单 Enter 先续上旧局）
        self.init_save_exclude = {x.lower() for x in init_save_exclude}
        self.raw_policy = raw_policy
        self.harness = harness
        self.enforce_options = enforce_options
        # phase10：allow_console 时不强制 EnableDebugConsole=0，并让 DLL 挂控制台提交函数（RR_MT_CONSOLE=1）
        self.allow_console = bool(allow_console)
        default_forced = FORCED_OPTIONS_CONSOLE if self.allow_console else FORCED_OPTIONS
        self.forced_options = dict(forced_options) if forced_options is not None else dict(default_forced)
        if self.allow_console:
            if not use_dll:
                raise ValueError('--allow-console 需要 rr_mt.dll（控制台命令由 DLL 钩子记录）')
            self.extra_env.setdefault('RR_MT_CONSOLE', '1')
        self.info = dict(schema='isaac-run-session-v1', name=name, created=now_iso(), host=host_info(),
                         session_dir=str(self.dir), save_dir=str(self.save_dir), use_dll=use_dll,
                         affinity=self.affinity, allow_console=self.allow_console, launches=[], runs=[], notes=[])
        if self.extra_env:
            self.info['extra_env'] = self.extra_env
        # phase15：本会话游戏进程的观测模式（IRR_OBS；未设置 = mod 默认 full）与省略名单覆盖（IRR_FX_OMIT；None = mod 内置名单）。
        # 每局实际用的模式与名单以 run 文件 header 为准（runs[].validation.header.obs / fx_omit）
        self.info['obs'] = self.extra_env.get('IRR_OBS') or 'full'
        self.info['fx_omit_override'] = self.extra_env.get('IRR_FX_OMIT')
        self.snapshots = {}
        self.watcher = None
        self.docs_watcher = None
        self.g = None
        self.forced_before = None
        self.cmd_history_before = None

    # ---- 会话前
    def prepare(self):
        self.info['copy'] = self.copy.check()
        self.info['mod'] = self.copy.install_mod()
        self.info['mods'] = self.copy.mods_manifest()
        exe_sha = self.info['copy']['binaries'][EXE_NAME]['sha256']
        sd = self.save_dir
        # 只读监视真实 Documents（DLL 会话应全程不出现 (Galaxy) 目录）
        self.docs_watcher = RealDocsWatcher()
        self.docs_watcher.start()
        if self.use_dll:
            self.info['dll'] = self.copy.check_dll()
            sd.mkdir(parents=True, exist_ok=True)
        else:
            self.info['dll'] = dict(enabled=False, note='-NoDll：第二阶段行为，存档在真实 Documents，回放只能近似')
        if self.init_save_from:
            if sd.exists() and any(sd.iterdir()):
                self.info['save_replaced_by_init'] = snapshot_save(self.dir / 'save/before-init', sd)
            sd.mkdir(parents=True, exist_ok=True)
            # 排除的文件（例如 gamestate1.dat）若在存档目录里残留，挪进会话目录（不删），使存档状态 = 来源快照
            displaced = []
            for name in sorted(self.init_save_exclude):
                for f in sd.iterdir():
                    if f.is_file() and f.name.lower() == name:
                        ddir = self.dir / 'save/displaced-by-init'
                        ddir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(f), str(ddir / f.name))
                        displaced.append(f.name)
            copied = []
            # snapshot_files：0.5.1 压缩 blob、0.5.0 未压缩 blob 与旧的直接文件布局都认；copy_to 写原始内容（按清单 sha256 校验）
            for name, f in snapshot_files(self.init_save_from).items():
                if name.lower() not in self.init_save_exclude:
                    f.copy_to(sd / name)
                    copied.append(name)
            self.info['init_save_from'] = dict(source=str(self.init_save_from), dest=str(sd), files=copied,
                                               excluded=sorted(self.init_save_exclude), displaced=displaced,
                                               status=snapshot_status(self.init_save_from),
                                               note='只拷贝进本会话的存档目录，会话结束不写回来源')
        self.info['save_pre'] = snapshot_save(self.dir / 'save/pre', sd, blobs=self.dir / 'save' / BLOB_DIR)
        opt = sd / 'options.ini'
        odir = self.dir / 'options'
        odir.mkdir()
        if opt.exists():
            shutil.copy2(opt, odir / 'options.before.ini')
            if self.enforce_options:
                self.forced_before = set_options(opt, self.forced_options)
            shutil.copy2(opt, odir / 'options.effective.ini')
            self.info['options'] = dict(before=read_options(odir / 'options.before.ini'), forced=self.forced_options if self.enforce_options else {},
                                        forced_before=self.forced_before, effective_file='options/options.effective.ini')
        else:
            self.info['options'] = dict(before=None, note='存档目录里没有 options.ini；游戏首次启动会写默认值（EnableDebugConsole 默认 0）',
                                        forced={})
        hist = sd / 'cmd_history.txt'
        self.cmd_history_before = hist.read_text(encoding='utf-8', errors='replace').splitlines() if hist.exists() else []
        self.write_session_file(exe_sha, None)
        self.write_recover_state('recording')
        self.write_partial()

    # ---- 异常结束的恢复（phase13 P3）
    def write_recover_state(self, state):
        """recover.json：启动器被杀/PowerShell 被关/重启之后，`--recover <会话目录>` 靠它知道还原什么、归档到哪。"""
        write_json(self.dir / 'recover.json', dict(
            schema='isaac-run-recover-v1', state=state, updated=now_iso(), name=self.name, session_dir=str(self.dir),
            out_root=str(self.out_root), copy=str(self.copy.root), save_dir=str(self.save_dir), use_dll=self.use_dll,
            raw_policy=self.raw_policy, allow_console=self.allow_console, launcher_pid=os.getpid(),
            options_file=str(self.save_dir / 'options.ini'), forced_before=self.forced_before,
            forced=self.forced_options if self.enforce_options else {}, cmd_history_before=self.cmd_history_before))

    def write_partial(self):
        """会话进行中先写一份不完整的 session.json（state = recording），finish 时覆盖。"""
        write_json(self.dir / 'session.json', dict(self.info, state='recording',
                                                   note='未收尾：录制进行中，或启动器异常结束（可用 --recover 收尾）'))

    def write_session_file(self, exe_sha, version):
        """mod 在每局开局时把这个文件原样拼进 run header 的 session 字段（必须单行 JSON）。"""
        mods = [dict(directory=m['directory'], disabled=m['disabled'], name=m.get('meta_name'), id=m.get('meta_id'),
                     version=m.get('meta_version'), files_sha256=m['files']) for m in self.info['mods']]
        obj = dict(session=self.name, exe_sha256=exe_sha,
                   binaries={k: v['sha256'] for k, v in self.info['copy']['binaries'].items()},
                   game_version=version, mods=mods, options_forced=self.forced_options if self.enforce_options else {},
                   launch_command=([str(self.copy.launcher), '--launch', str(self.copy.dll)] if self.use_dll else [])
                   + [str(self.copy.exe), '--luadebug'], host=self.info['host']['hostname'],
                   save_dir=str(self.save_dir), use_dll=self.use_dll, allow_console=self.allow_console,
                   dll=dict(sha256=self.info['dll'].get('dll_sha256'), launcher_sha256=self.info['dll'].get('launcher_sha256'),
                            source='rl/bridge/engine/j746_rec') if self.use_dll else None)
        write_json(self.irr_dir / 'irr_session.json', obj, compact=True)

    # ---- 启动
    def launch(self, attempts=6, extra_env=None):
        tag_prefix = f'p{len(self.info["launches"]) + 1}-'
        env = dict(self.extra_env)
        env.update(extra_env or {})
        try:
            g, log = launch_with_retry(self.copy, self.irr_dir, self.game_logs, attempts=attempts, extra_env=env,
                                       tag_prefix=tag_prefix, save_dir=self.save_dir, use_dll=self.use_dll)
        except LaunchFailed as e:
            self.info['launches'].append(dict(attempts=getattr(e, 'log', None), error=str(e)))
            raise
        self.info['launches'].append(dict(attempts=log))
        ver = log[-1].get('version')
        if ver:
            self.info['game_version'] = ver
            self.write_session_file(self.info['copy']['binaries'][EXE_NAME]['sha256'], ver)
        sdp = self.copy.game / 'savedatapath.txt'
        if sdp.exists() and sdp.stat().st_mtime >= g.t0 - 1:
            self.info['savedatapath_txt'] = sdp.read_text(encoding='utf-8', errors='replace').strip().splitlines()
        if self.watcher is None:
            self.watcher = SaveWatcher(self.irr_dir, self.dir / 'save', self.snapshots, self.save_dir)
            self.watcher.start()
        self.g = g
        self.write_partial()
        return g

    def mark_menu_failure(self, g, reason):
        """mod 已加载、但还没开局进程就崩了（例如读取续局存档时 0xC0000005）：记为一次启动失败。"""
        att = self.info['launches'][-1]['attempts'][-1]
        att['result'] = 'failed'
        att['reason'] = f'进局前失败：{reason}'
        att['close'] = g.close(wm_close=False, wait_s=1)

    def runs_started_since(self, t0):
        n = 0
        for sig in self.irr_dir.glob('signal-*.start.json'):
            try:
                if sig.stat().st_mtime >= t0:
                    n += 1
            except OSError:
                pass
        return n

    def launch_and_enter(self, attempts=6, timeout=150):
        """启动并按键进入游戏（自动化用）；进局之前的任何失败都关掉重来并计数。返回 (GameProcess, 状态)。"""
        last = None
        for _ in range(attempts):
            g = self.launch(attempts=attempts)
            try:
                s, presses = g.enter_game(timeout=timeout)
                self.info['launches'][-1]['attempts'][-1]['menu_presses'] = presses
                return g, s
            except (LaunchFailed, TimeoutError) as e:
                last = e
                self.mark_menu_failure(g, repr(e))
                time.sleep(2.0)
        raise LaunchFailed(f'{attempts} 次都没能进入游戏：{last!r}')

    # ---- 会话后
    def finish(self, recovered=None):
        """recovered（phase13 P3）：--recover 收尾时的说明 dict（启动器已不在、游戏进程已结束）。"""
        if self.watcher:
            self.watcher.stop_flag = True
            try:
                self.watcher.poll_once()
            except Exception as e:  # noqa: BLE001
                self.snapshots.setdefault('_errors', []).append(repr(e))
        if self.g is not None and self.g.closed is None:
            self.g.close()
        # 游戏进程怎么结束的（phase9-robust：强杀/崩溃时以前只有 game/<tag>.exit.json 记了退出码）
        if self.g is not None and self.g.closed is not None:
            ex = process_exit_summary(self.g.closed)
            self.info['game_exit'] = ex
            for L in self.info['launches']:
                for att in L.get('attempts') or []:
                    if att.get('pid') == self.g.pid and att.get('result') == 'ok':
                        att['exit'] = ex
        time.sleep(1.0)
        sd = self.save_dir
        self.info['save_post'] = snapshot_save(self.dir / 'save/post', sd, blobs=self.dir / 'save' / BLOB_DIR)
        opt = sd / 'options.ini'
        if opt.exists():
            (self.dir / 'options').mkdir(exist_ok=True)
            shutil.copy2(opt, self.dir / 'options/options.after-game.ini')
            if self.forced_before:
                restore = {k: v for k, v in self.forced_before.items() if v is not None}
                if restore:
                    set_options(opt, restore)
                self.info['options']['restored'] = restore
        hist = sd / 'cmd_history.txt'
        after = hist.read_text(encoding='utf-8', errors='replace').splitlines() if hist.exists() else []
        n0 = len(self.cmd_history_before)
        new = after[n0:] if after[:n0] == self.cmd_history_before else after
        self.info['console'] = dict(cmd_history_new_lines=new, allow_console=self.allow_console,
                                    note=('--allow-console：控制台命令由 DLL 钩子逐条记进 run 文件（ev cmd src=console，带帧号）；'
                                          'cmd_history.txt 只在退出时写、没有帧号，这里仅作对照') if self.allow_console else
                                    '内建控制台命令只出现在退出时写的 cmd_history.txt（无帧号）')
        for lg in sorted(self.game_logs.glob('*.log.txt')):
            v = game_version_from_log(lg)
            if v:
                self.info.setdefault('game_version', v)
        self.archive_runs(console_used=bool(new))
        if self.docs_watcher is not None:
            self.info['real_documents_watch'] = self.docs_watcher.result()
        if recovered:
            self.info['recovered'] = recovered
        self.info['save_snapshot_errors'] = self.snapshots.get('_errors')
        self.info['state'] = 'recovered' if recovered else 'finished'
        self.info['finished'] = now_iso()
        self.info['summary'] = self.summary()
        write_json(self.dir / 'session.json', self.info)
        (self.dir / 'summary.txt').write_text(self.summary_text(), encoding='utf-8')
        self.write_recover_state('recovered' if recovered else 'finished')
        return self.info

    def archive_runs(self, console_used=False):
        runs_dir = self.dir / 'runs'
        runs_dir.mkdir(exist_ok=True)
        raws = sorted(self.irr_dir.glob('run-*.jsonl'))
        for i, raw in enumerate(raws, 1):
            name = raw.stem
            dst = runs_dir / f'run-{i:04d}.jsonl.zst'
            rec = dict(index=i, file=f'runs/{dst.name}', lua_name=name)
            try:
                rec.update(compress_and_verify(raw, dst))
            except Exception as e:
                rec['archive_error'] = repr(e)
            sig_end = read_json(self.irr_dir / f'signal-{name}.end.json')
            rec['end_signal'] = sig_end
            rec['save_snapshot'] = self.snapshots.get(name)
            cc = (rec.get('validation') or {}).get('console_cmds', 0)
            rec['console_cmds'] = cc
            rec['console_used'] = console_used or bool(cc)
            if cc:
                rec['console_note'] = f'本局 {cc} 条控制台命令（ev cmd src=console，带帧号，回放时同帧执行）'
            elif console_used and not self.allow_console:
                rec['console_note'] = '本会话 cmd_history.txt 有新增行，无法定位到具体局，会话内所有局都标记'
            v = rec.get('validation') or {}
            if v.get('truncated'):
                rec['note'] = truncated_note(self.info.get('game_exit'), sig_end, (v.get('header') or {}).get('flush_every'))
            # 0.5.0（phase13）：内容是否可信（verified 只表示压缩逐字节校验）
            issues = data_issues(v, sig_end) if v else [f'归档失败：{rec.get("archive_error")}']
            snap = rec['save_snapshot']
            if not snap:
                issues.append('没有开局存档快照')
            elif snap.get('late'):
                issues.append('开局存档快照晚了（可能含后来的局写过的存档）：' + '；'.join(snap.get('late_reasons') or []))
            elif snap.get('late') is None and snap.get('kind') != 'mod':
                issues.append('开局存档快照无法判断是否晚了')
            rec['data_ok'] = not issues
            rec['data_issues'] = issues
            # 原始文件去向（0.5.1，phase14 B2）：只有压缩逐字节校验通过（verified）的局才处理原始文件；没通过的永远原地保留。
            # mod 的开局存档拷贝只在本局快照取自它（kind = mod，已入库 save/blobs 并校验）时一并处理。
            savecopies = [sc for sc in sorted(self.irr_dir.glob(f'savecopy-{name}-*')) if snap and snap.get('kind') == 'mod']
            policy = self.raw_policy if rec.get('verified') else 'keep'
            if policy == 'recycle':
                rc = recycle_files([raw] + savecopies)
                rec['raw_recycle'] = rc
                if rc['ok']:
                    rec['raw_recycled'] = True
                    rec['raw_disposition'] = 'recycled'
                else:
                    policy = 'stage'
                    rec['raw_recycle_fallback'] = f'送回收站失败，退回 stage（挪到 _raw_staging）：{rc.get("error")}'
            if policy == 'stage':
                rec['raw_staged_to'] = stage_raw(raw, self.out_root / '_raw_staging', self.name)
                for sc in savecopies:
                    stage_raw(sc, self.out_root / '_raw_staging', self.name + '/savecopy')
                rec['raw_disposition'] = 'staged'
            elif policy == 'keep':
                rec['raw_kept'] = str(raw)
                rec['raw_disposition'] = 'kept' if rec.get('verified') else 'kept (压缩校验没通过，原始文件保留)'
            self.info['runs'].append(rec)

    def summary(self):
        runs = self.info['runs']
        frames = sum((r.get('validation') or {}).get('frames', 0) for r in runs)
        zst = sum(r.get('zst_bytes', 0) for r in runs)
        raw = sum(r.get('raw_bytes', 0) for r in runs)
        hours = frames / LOGIC_HZ / 3600
        launches = [a for L in self.info['launches'] for a in (L.get('attempts') or [])]
        return dict(runs=len(runs), frames=frames, game_hours=round(hours, 4), zst_mb=round(zst / 1e6, 3),
                    raw_mb=round(raw / 1e6, 3), zst_mb_per_hour=round(zst / 1e6 / hours, 2) if hours else None,
                    continued_runs=sum(1 for r in runs if ((r.get('validation') or {}).get('header') or {}).get('continued')),
                    truncated_runs=sum(1 for r in runs if (r.get('validation') or {}).get('truncated')),
                    archive_errors=sum(1 for r in runs if 'archive_error' in r),
                    launch_attempts=len(launches), launch_failures=sum(1 for a in launches if a.get('result') != 'ok'),
                    use_dll=self.use_dll,
                    game_exit_hex=(self.info.get('game_exit') or {}).get('exit_hex'),
                    game_exit_abnormal=(self.info.get('game_exit') or {}).get('abnormal'),
                    mt_runs=sum(1 for r in runs if (((r.get('validation') or {}).get('header') or {}).get('mt') or {}).get('available')),
                    # 0.5.0（phase13）：内容有问题的局、晚了的存档快照、退化帧、写失败
                    data_issue_runs=sum(1 for r in runs if r.get('data_ok') is False),
                    late_snapshots=sum(1 for r in runs if (r.get('save_snapshot') or {}).get('late')),
                    mod_snapshots=sum(1 for r in runs if (r.get('save_snapshot') or {}).get('kind') == 'mod'),
                    degraded_frames=sum(len((r.get('validation') or {}).get('degraded_frames') or []) for r in runs),
                    write_errors=sum(((r.get('validation') or {}).get('end') or {}).get('write_errors') or 0 for r in runs),
                    # 0.5.1（phase14 B2）：原始文件去向
                    raw_policy=self.raw_policy,
                    raw_recycled=sum(1 for r in runs if r.get('raw_disposition') == 'recycled'),
                    raw_staged=sum(1 for r in runs if r.get('raw_disposition') == 'staged'),
                    raw_kept=sum(1 for r in runs if str(r.get('raw_disposition') or '').startswith('kept')),
                    raw_recycle_fallbacks=sum(1 for r in runs if r.get('raw_recycle_fallback')),
                    state=self.info.get('state'), recovered=bool(self.info.get('recovered')),
                    real_documents_galaxy_appeared=(self.info.get('real_documents_watch') or {}).get('appeared_during_session'))

    def summary_text(self):
        s = self.info['summary']
        lines = [f'会话 {self.name}', f'局数 {s["runs"]}（续局 {s["continued_runs"]}，截断 {s["truncated_runs"]}）',
                 f'逻辑帧 {s["frames"]}，游戏时间 {s["game_hours"]:.4f} 小时（{s["frames"] / LOGIC_HZ:.1f} 秒）',
                 f'压缩后 {s["zst_mb"]:.3f} MB（原始 {s["raw_mb"]:.3f} MB），{s["zst_mb_per_hour"]} MB/小时',
                 f'启动 {s["launch_attempts"]} 次，失败 {s["launch_failures"]} 次；游戏进程退出码 {s.get("game_exit_hex")}'
                 + ('（异常退出）' if s.get('game_exit_abnormal') else ''),
                 f'DLL {"开" if s.get("use_dll") else "关（-NoDll）"}；有 MT reseed 的局 {s.get("mt_runs")}；存档目录 {self.save_dir}',
                 f'观测模式 {self.info.get("obs")}' + (f'（省略名单覆盖 {self.info.get("fx_omit_override")}）'
                                                       if self.info.get('fx_omit_override') else ''),
                 f'会话期间真实 Documents 出现 (Galaxy) 目录：{s.get("real_documents_galaxy_appeared")}',
                 f'数据有问题的局 {s.get("data_issue_runs")}；开局存档快照：mod 拷贝 {s.get("mod_snapshots")} 局，晚了 '
                 f'{s.get("late_snapshots")} 局；退化帧 {s.get("degraded_frames")}；写失败 {s.get("write_errors")}'
                 + ('；本会话由 --recover 收尾' if s.get('recovered') else ''),
                 f'原始文件（策略 {s.get("raw_policy")}）：送回收站 {s.get("raw_recycled")} 局，挪到暂存 {s.get("raw_staged")} 局，'
                 f'原地保留 {s.get("raw_kept")} 局' + (f'；送回收站失败退回暂存 {s.get("raw_recycle_fallbacks")} 局'
                                                    if s.get('raw_recycle_fallbacks') else ''), '']
        for r in self.info['runs']:
            v = r.get('validation') or {}
            h = v.get('header') or {}
            lines.append(f'{r["file"]}: 帧 {v.get("frames")} 种子 {h.get("seed")} 续局 {h.get("continued")} '
                         f'结束 {(v.get("end") or {}).get("reason")} zst {r.get("zst_bytes")} B 校验 {r.get("verified")} '
                         f'MT {(h.get("mt") or {}).get("available")} salt {(h.get("mt") or {}).get("salt")} '
                         f'obs {h.get("obs")} '
                         f'数据 {"OK" if r.get("data_ok") else "有问题：" + "；".join(r.get("data_issues") or [])}')
        return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------------------------- harness 客户端（测试）
def write_control(irr_dir, cfg=None):
    write_json(Path(irr_dir) / 'irr_control.json', cfg or {}, compact=True)


def drive_bot(g: GameProcess, stall_s=25, max_s=None):
    """把 Lua 机器人想按的键（want.txt）以真实键盘消息发出，直到 harness 局结束。返回结束原因与统计。"""
    want = g.irr_dir / 'want.txt'
    last_n, presses = -1, 0
    last_change, last_seen = time.time(), -1
    t0 = time.time()
    s0 = g.status() or {}
    run0 = s0.get('run')
    while True:
        if not g.alive():
            return dict(ended='process_exit', key_downs=presses)
        s = g.status()
        if s is None:  # status.json 正被 mod 重写（非原子写），下次再读
            time.sleep(0.01)
            continue
        if s.get('run') != run0 or not s.get('in_game'):
            return dict(ended='run_closed', key_downs=presses, status=s)
        if s.get('n', -1) != last_seen:
            last_seen, last_change = s.get('n', -1), time.time()
        elif time.time() - last_change > stall_s:
            return dict(ended='stalled', key_downs=presses, status=s)
        if max_s and time.time() - t0 > max_s:
            return dict(ended='max_s', key_downs=presses, status=s)
        try:
            txt = want.read_text(encoding='utf-8').split(' ', 1)
            n = int(txt[0])
            keys = txt[1] if len(txt) > 1 else ''
        except Exception:
            n, keys = last_n, None
        if keys is not None and n != last_n:
            last_n = n
            names = {BOT_KEYS[c] for c in keys if c in BOT_KEYS}
            before = set(g.down)
            g.set_keys(names)
            presses += len(names - before)
        time.sleep(0.004)


OPPOSITE = {'W': 'S', 'S': 'W', 'A': 'D', 'D': 'A'}


def random_keys(g: GameProcess, seconds, rng, paired=False):
    """随机移动/射击（真实键盘消息，每 0.5 s 换一次）。paired=True 时每一步之后走一步反方向，角色基本留在原地附近。"""
    end = time.time() + seconds
    while time.time() < end:
        g.check_health()
        mv = rng.choice([[], ['W'], ['A'], ['S'], ['D'], ['W', 'A'], ['S', 'D']])
        steps = [mv, [OPPOSITE[k] for k in mv]] if paired else [mv]
        for step in steps:
            sh = rng.choice([[], ['UP'], ['LEFT'], ['RIGHT'], ['DOWN']])
            g.set_keys(step + sh)
            time.sleep(0.5)
    g.set_keys([])


def run_auto_play(sess: Session, seconds=40, restart=True, seed=7, entered=False):
    """人类流程的自动化替身（验收第 6 项）：脚本按键进菜单开局（有存档时是续局）、在开局房间附近随机走动射击、
    按住 R 重开一局、再玩一会儿、关窗口（游戏保存续局状态）。"""
    g = sess.g
    rng = random.Random(seed)
    if not entered:
        s, presses = g.enter_game()
        sess.info['notes'].append(f'auto: 进游戏按键 {presses} 次')
    random_keys(g, seconds, rng, paired=True)
    if restart:
        g.key('R', True)
        try:
            run0 = (g.status() or {}).get('run')
            g.wait(lambda st: st.get('run') not in (None, run0) and st.get('in_game'), 20)
        finally:
            g.key('R', False)
        random_keys(g, seconds, rng, paired=True)
    return g.close(wm_close=True)


def launcher_alive(pid):
    """recover.json 里的 launcher_pid 是否还是一个活着的 Python 进程（另一个窗口正在录）。只读：OpenProcess +
    GetExitCodeProcess + QueryFullProcessImageNameW。映像名不是 python*（pid 被别的程序复用）当作不在。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0 or pid == os.getpid():
        return False
    h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = W.DWORD()
        if not k32.GetExitCodeProcess(h, C.byref(code)) or code.value != 259:   # STILL_ACTIVE
            return False
        buf = C.create_unicode_buffer(1024)
        n = W.DWORD(1024)
        if not k32.QueryFullProcessImageNameW(h, 0, buf, C.byref(n)):
            return False
        return Path(buf.value).name.lower().startswith('python')
    finally:
        k32.CloseHandle(h)


def find_unfinished(out_root, exclude=None):
    """out_root 下没有收尾的会话目录（recover.json 的 state 不是 finished/recovered）。启动器（launcher_pid）还活着的
    会话不算（另一个窗口正在录，phase14 A2）。"""
    out = []
    try:
        dirs = sorted(Path(out_root).iterdir())
    except OSError:
        return out
    for d in dirs:
        if not d.is_dir() or (exclude and d.resolve() == Path(exclude).resolve()):
            continue
        rec = read_json(d / 'recover.json')
        if rec and rec.get('state') not in ('finished', 'recovered') and not launcher_alive(rec.get('launcher_pid')):
            out.append(d)
    return out


def recover_session(sess_dir, force=False):
    """phase13 P3：启动器（本模块）被杀、PowerShell 被关、电脑重启之后收尾一个会话：
    归档 <会话>/lua/ 里的原始 run 文件（截断局照常标 truncated）、把 mod 的开局存档拷贝入库、补写 session.json（state =
    recovered，recovered 字段说明）、按 recover.json 还原 options.ini 被强制的键。游戏进程还在运行时拒绝（先关游戏）。"""
    sess_dir = Path(sess_dir).resolve()
    rec = read_json(sess_dir / 'recover.json')
    if not rec:
        raise RuntimeError(f'{sess_dir} 没有 recover.json（不是 0.5.0 起的录制会话目录？）')
    if rec.get('state') in ('finished', 'recovered') and not force:
        raise RuntimeError(f'{sess_dir} 已收尾（state={rec.get("state")}）；确需重做加 --force')
    # phase14 A1：重做只能基于 lua/ 里的原始 run 文件。原始文件已挪走（送回收站/暂存）而 runs/ 里已有归档时，重做会把
    # session.json 写成 runs=[]（或少了局），毁掉元数据：拒绝
    raws_now = sorted((sess_dir / 'lua').glob('run-*.jsonl'))
    archived = sorted((sess_dir / 'runs').glob('run-*.jsonl.zst'))
    if len(archived) > len(raws_now):
        raise RuntimeError(f'{sess_dir}：runs/ 里已有 {len(archived)} 个归档，lua/ 里只剩 {len(raws_now)} 个原始 run 文件'
                           '（原始文件已送回收站/挪到暂存）；重做会丢掉已归档局的记录，拒绝。要重做请先把原始文件放回 lua/')
    if launcher_alive(rec.get('launcher_pid')):
        raise RuntimeError(f'记录这个会话的启动器（pid {rec.get("launcher_pid")}）还在运行：它会自己收尾')
    copy = GameCopy(rec['copy'])
    pids = pids_of_exe(copy.exe)
    if pids:
        raise RuntimeError(f'副本 {copy.root} 的游戏进程仍在运行（pid {pids}）：先正常关掉游戏窗口，再运行 --recover')
    # 覆盖 session.json 之前先备份（任何情况下；phase14 A1）
    if (sess_dir / 'session.json').exists():
        bak = sess_dir / f'session.before-recover-{_dt.datetime.now().strftime("%Y%m%d-%H%M%S")}.json'
        k = 1
        while bak.exists():
            k += 1
            bak = sess_dir / f'session.before-recover-{_dt.datetime.now().strftime("%Y%m%d-%H%M%S")}-{k}.json'
        shutil.copy2(sess_dir / 'session.json', bak)
    partial = read_json(sess_dir / 'session.json') or {}
    sess = object.__new__(Session)   # 不走 __init__（会话目录已存在）；只设 finish/archive/summary 用到的属性
    sess.name = rec['name']
    sess.out_root = Path(rec['out_root'])
    sess.dir = sess_dir
    sess.irr_dir = sess_dir / 'lua'
    sess.game_logs = sess_dir / 'game'
    sess.copy = copy
    sess.use_dll = rec.get('use_dll', True)
    sess.save_dir = Path(rec['save_dir'])
    sess.raw_policy = rec.get('raw_policy', 'stage')
    sess.allow_console = bool(rec.get('allow_console'))
    sess.forced_before = rec.get('forced_before')
    sess.forced_options = rec.get('forced') or {}
    sess.enforce_options = bool(rec.get('forced'))
    sess.cmd_history_before = rec.get('cmd_history_before') or []
    sess.info = dict(partial)
    sess.info.pop('state', None)
    sess.info.pop('note', None)
    sess.info.setdefault('runs', [])
    sess.info['runs'] = []
    sess.info.setdefault('launches', [])
    sess.info.setdefault('notes', [])
    sess.info.setdefault('options', dict(before=None))
    sess.snapshots = {}
    # 已由在线监视线程入库的快照：清单还在 save/<run>/manifest.json；重新扫描开局信号（mod 拷贝入库是幂等的），
    # 不做 Python 核对快照（游戏早已不在开局时刻）
    sess.watcher = SaveWatcher(sess.irr_dir, sess_dir / 'save', sess.snapshots, sess.save_dir, check=False)
    for t_sig, sig in sess.watcher._signals():
        run = sig.name[len('signal-'):-len('.start.json')]
        man = read_json(sess_dir / 'save' / run / 'manifest.json')
        if man and 'late' in man:
            sess.watcher.seen.add(sig.name)
            sess.snapshots[run] = dict(signal=read_json(sig, {}), signal_mtime=t_sig, snapshot=str(sess_dir / 'save' / run),
                                       kind=man.get('kind'), late=man['late'], late_reasons=man.get('late_reasons') or [],
                                       files=sorted(man.get('files') or {}), from_manifest=True)
    sess.docs_watcher = None
    sess.g = None
    raws = sorted(sess.irr_dir.glob('run-*.jsonl'))
    sess.info['real_documents_watch'] = dict(note='--recover：启动器异常结束，会话后半段没有监视')
    info = sess.finish(recovered=dict(
        time=now_iso(), launcher_pid=rec.get('launcher_pid'), raw_runs_found=[r.name for r in raws],
        options_restored_from='recover.json forced_before',
        note='启动器（Python）没能正常收尾（被结束/PowerShell 被关/电脑重启）；本记录由 --recover 补写：game_exit 未知，'
             '没有 end 行的局标为截断；开局存档以 mod 的开局拷贝为准（没有的局退回到 Python 当时的快照或缺失）'))
    return sess, info


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='整局录制会话（人类会话入口；rl/record_run.ps1 调用）')
    ap.add_argument('--session', default=None, help='会话名（录制时必填）')
    ap.add_argument('--recover', default=None, metavar='SESSION_DIR',
                    help='收尾一个没收尾的会话（启动器被杀/PowerShell 被关/重启之后）：归档原始 run 文件、补写 session.json、还原 options.ini')
    ap.add_argument('--force', action='store_true', help='与 --recover 一起：会话已收尾也重做')
    ap.add_argument('--fault', default=None, help=argparse.SUPPRESS)   # 测试用：IRR_FAULT（frame:<n>,room:<k>,write:<n>）
    ap.add_argument('--out-root', default=str(ISAAC_ROOT / 'experiments/run-recorder/sessions'))
    ap.add_argument('--copy', default=str(DEFAULT_COPY))
    ap.add_argument('--init-save-from', default=None)
    ap.add_argument('--raw', choices=['recycle', 'stage', 'keep'], default='recycle',
                    help='校验通过后原始 jsonl 与 mod 存档拷贝的去向：recycle（默认）= 送 Windows 回收站（失败退回 stage）；'
                         'stage = 挪到 <out-root>/_raw_staging/<会话>；keep = 原地保留')
    ap.add_argument('--auto-seconds', type=float, default=0, help='>0 时由脚本自动按键代替人类（测试用）')
    ap.add_argument('--attempts', type=int, default=6)
    ap.add_argument('--no-dll', action='store_true', help='不注入 rr_mt.dll（第二阶段行为：真实 Documents 存档、无 MT reseed）')
    ap.add_argument('--obs', choices=OBS_MODES, default=RECORD_DEFAULT_OBS,
                    help='观测模式（默认 fxkey = 装饰性特效只在关键帧写，digest 不变，README §4.11；full = 每帧完整观测；'
                         'keyframe = 实体行只在关键帧写）')
    ap.add_argument('--allow-console', action='store_true',
                    help='不强制关闭调试控制台；DLL 记录每条控制台命令（ev cmd src=console），回放时同帧执行（README §8.7）')
    ap.add_argument('--affinity', default=DEFAULT_AFFINITY,
                    help='游戏进程的 CPU 亲和掩码（phase8-mouse §H4；默认 auto = 逻辑 CPU 0 一个核，录下的局才可能精确回放；'
                         '0 = 不限制；0x… = 指定掩码；只在带 DLL 时生效）')
    a = ap.parse_args(argv)
    if a.recover:
        try:
            sess, info = recover_session(a.recover, force=a.force)
        except RuntimeError as e:
            print(f'[恢复] {e}', flush=True)
            return 3
        print(sess.summary_text(), flush=True)
        print(f'[恢复] 已收尾：{sess.dir / "session.json"}（state = recovered）', flush=True)
        return 0 if not info['summary']['archive_errors'] else 2
    if not a.session:
        ap.error('录制需要 --session')
    for d in find_unfinished(a.out_root):
        print(f'[录制] 提示：会话 {d} 上次没有收尾（启动器被结束/重启？）。关掉游戏后可运行 '
              f'python -m isaac_bridge.run_recording --recover "{d}"（或 record_run.ps1 -Recover）', flush=True)
    extra_env = obs_env(a.obs) or None
    if a.fault:
        extra_env = dict(extra_env or {}, IRR_FAULT=a.fault)
    sess = Session(a.session, a.out_root, a.copy, a.init_save_from, raw_policy=a.raw, use_dll=not a.no_dll,
                   extra_env=extra_env, affinity=a.affinity, allow_console=a.allow_console)
    print(f'[录制] 会话目录 {sess.dir}', flush=True)
    sess.prepare()
    print(f'[录制] {"带 DLL（MT reseed、存档在副本 profile/）" if sess.use_dll else "不带 DLL（-NoDll）"}；存档目录 {sess.save_dir}',
          flush=True)
    print(f'[录制] CPU 亲和 {sess.affinity or "不限制（回放将无法保证精确）"}', flush=True)
    if sess.allow_console:
        print('[录制] 允许调试控制台：每条控制台命令记进 run 文件（ev cmd src=console），回放时同帧执行', flush=True)
    print(f'[录制] 存档快照 {len(sess.info["save_pre"]["files"])} 个文件；启动游戏…', flush=True)
    try:
        if a.auto_seconds > 0:
            g, _ = sess.launch_and_enter(attempts=a.attempts)
            print(f'[录制] 自动模式：已进入游戏（pid {g.pid}）', flush=True)
            run_auto_play(sess, seconds=a.auto_seconds, entered=True)
        else:
            relaunches = 0
            while True:
                g = sess.launch(attempts=a.attempts)
                att = sess.info['launches'][-1]['attempts']
                print(f'[录制] 启动成功（第 {len(att)} 次尝试，pid {g.pid}）。正常游玩即可，关闭游戏窗口后自动归档。', flush=True)
                rec = g.wait_exit()
                # 还没开局就异常退出（读续局存档时崩溃等）= 启动失败，自动重启；开过局之后的退出一律视为玩家结束会话
                if rec['exit_code'] != 0 and sess.runs_started_since(g.t0) == 0 and relaunches < a.attempts:
                    relaunches += 1
                    att[-1]['result'] = 'failed'
                    att[-1]['reason'] = f'进局前失败：exit={rec["exit_hex"]}'
                    print(f'[录制] 游戏在开局前异常退出（{rec["exit_hex"]}），自动重启（第 {relaunches} 次）', flush=True)
                    time.sleep(2.0)
                    continue
                if rec['exit_code'] != 0:
                    print(f'[录制] 游戏异常退出（{rec["exit_hex"]}）；已录的局照常归档（缺 end 行的局标为截断）', flush=True)
                break
    except LaunchFailed as e:
        print(f'[录制] 启动失败：{e}', flush=True)
    finally:
        info = sess.finish()
    print(sess.summary_text(), flush=True)
    print(f'[录制] session.json：{sess.dir / "session.json"}', flush=True)
    return 0 if not info['summary']['archive_errors'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
