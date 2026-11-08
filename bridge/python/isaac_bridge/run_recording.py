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
  * 启动游戏（--luadebug），自动重试启动失败并计数（注入器失败、进程崩溃、log 洪水、mod 未加载），
    log 洪水检测（只认 mtime 晚于本次启动的 log.txt）；DLL 自己的 rr_mt.<pid>.log 直接写进会话 game/ 目录；
  * 退出后把每局 jsonl 压成 .zst（zstd-19）并逐字节校验、逐行解析校验；写 session.json 与会话摘要。
另外提供测试 harness 客户端（文件 IPC、按种子开局、机器人按键），供回归测试与 replay_run.py 使用。

只会结束本模块自己启动的进程（按 pid/句柄）。删除一律不做：原始 jsonl 校验后移到暂存目录由用户处理。
"""
from __future__ import annotations

import ctypes as C
import datetime as _dt
import hashlib
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
def snapshot_save(dest, save_dir):
    """复制整个存档目录（排除 *.dmp、log.txt）到 dest，返回清单（文件名、字节、mtime、sha256）。"""
    dest, save_dir = Path(dest), Path(save_dir)
    man = dict(source=str(save_dir), dest=str(dest), time=now_iso(), exists=save_dir.exists(), files={}, skipped=[])
    if not save_dir.exists():
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
                shutil.copy2(f, dest / f.name)
                st = f.stat()
                man['files'][f.name] = dict(bytes=st.st_size, mtime=st.st_mtime, sha256=sha256_file(dest / f.name))
                break
            except OSError as e:
                man.setdefault('retries', []).append([f.name, repr(e)])
                time.sleep(0.2)
    write_json(dest / 'manifest.json', man)
    return man


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
                   IRR_DIR=str(self.irr_dir).replace('\\', '/'), IRR_PROC=proc_tag)
        env_keys = ['USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'TEMP', 'TMP', 'IRR_DIR', 'IRR_PROC']
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
              truncated=False, bad_json=0, last_line_partial=False, mt_frames=0, repaired_lines=0)
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
                                harness=o.get('harness') is not None, obs=o.get('obs'),
                                mt=dict(available=(o.get('mt') or {}).get('available'), salt=(o.get('mt') or {}).get('salt'),
                                        reason=(o.get('mt') or {}).get('reason'),
                                        reseeds=[r.get('at') for r in ((o.get('mt') or {}).get('reseeds') or [])]))
        elif k == 'f':
            if 'mt' in o:
                st['mt_frames'] = st.get('mt_frames', 0) + 1
            if o['n'] != expect_n:
                st['errors'].append(f'帧号不连续：期望 {expect_n} 得到 {o["n"]}')
                expect_n = o['n']
            expect_n += 1
            st['frames'] += 1
            if st['first_gf'] is None:
                st['first_gf'] = o['gf']
            st['last_gf'] = o['gf']
        elif k == 'ev':
            st['events'][o['ev']] = st['events'].get(o['ev'], 0) + 1
        elif k == 'end':
            st['end'] = dict(reason=o.get('reason'), frames=o.get('frames'), cost=o.get('cost'), errors=o.get('errors'))
            if o.get('frames') != st['frames']:
                st['errors'].append(f'end.frames={o.get("frames")} 与帧行数 {st["frames"]} 不符')
    if st['end'] is None:
        st['truncated'] = True
    st['game_hours'] = st['frames'] / LOGIC_HZ / 3600
    return st


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


def truncated_note(game_exit):
    """截断局（没有 end 行）的说明：带上游戏进程的退出码（若知道）。mod 每 300 帧、进房、暂停时 flush，
    另外 256 KB 写缓冲满了也会写出（可能停在行中间，读取时丢掉，见 load_run.read_lines），所以最多丢约 10 秒。"""
    s = '没有 end 行：进程异常结束，缓冲区里最后最多 10 秒的数据丢失'
    if game_exit and game_exit.get('abnormal'):
        s += f'（游戏进程退出码 {game_exit.get("exit_hex")}）'
    return s


def compress_and_verify(raw, dst, level=19):
    """jsonl -> .zst（zstd level 19，带内容校验和），解压回读逐字节比较 sha256，并逐行解析校验。"""
    raw, dst = Path(raw), Path(dst)
    raw_sha = sha256_file(raw)
    cctx = zstandard.ZstdCompressor(level=level, write_checksum=True, write_content_size=True)
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    t = time.time()
    with open(raw, 'rb') as fi, open(tmp, 'wb') as fo:
        cctx.copy_stream(fi, fo)
    os.replace(tmp, dst)
    secs = time.time() - t
    h = hashlib.sha256()
    dctx = zstandard.ZstdDecompressor()
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
                compress_s=round(secs, 2), level=level, verified=True, validation=st)


def open_run_lines(path):
    """逐行读取 .jsonl 或 .jsonl.zst（bytes 行）。"""
    path = Path(path)
    if path.suffix == '.zst':
        import io
        f = open(path, 'rb')
        r = zstandard.ZstdDecompressor().stream_reader(f)
        return io.BufferedReader(r)
    return open(path, 'rb')


def stage_raw(raw, staging_root, session_name):
    """原始 jsonl 校验后移到暂存目录（不删除；由用户确认后自行清理）。"""
    d = Path(staging_root) / session_name
    d.mkdir(parents=True, exist_ok=True)
    target = d / Path(raw).name
    shutil.move(str(raw), str(target))
    return str(target)


# ---------------------------------------------------------------------------------------------- 会话
class SaveWatcher(threading.Thread):
    """监视 mod 的开局信号文件，每局开始时快照存档目录。"""

    def __init__(self, irr_dir, save_root, records, save_dir):
        super().__init__(daemon=True)
        self.irr_dir = Path(irr_dir)
        self.save_root = Path(save_root)
        self.save_dir = Path(save_dir)
        self.records = records
        self.seen = set()
        self.stop_flag = False

    def poll_once(self):
        for sig in sorted(self.irr_dir.glob('signal-*.start.json')):
            if sig.name in self.seen:
                continue
            self.seen.add(sig.name)
            run = sig.name[len('signal-'):-len('.start.json')]
            info = read_json(sig, {})
            t_sig = time.time()
            waited = wait_save_quiet(self.save_dir)
            man = snapshot_save(self.save_root / run, self.save_dir)
            self.records[run] = dict(signal=info, signal_seen=now_iso(), waited_quiet_s=round(waited, 2),
                                     snapshot=str(self.save_root / run), files=sorted(man['files']),
                                     snapshot_delay_s=round(time.time() - t_sig, 2))

    def run(self):
        while not self.stop_flag:
            try:
                self.poll_once()
            except Exception as e:  # 不让监视线程死掉
                self.records.setdefault('_errors', []).append(repr(e))
            time.sleep(0.5)


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

    def __init__(self, name, out_root, copy_root=DEFAULT_COPY, init_save_from=None, raw_policy='stage', harness=None,
                 enforce_options=True, use_dll=True, extra_env=None, forced_options=None, init_save_exclude=(),
                 affinity=DEFAULT_AFFINITY):
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
        self.forced_options = dict(forced_options) if forced_options is not None else dict(FORCED_OPTIONS)
        self.info = dict(schema='isaac-run-session-v1', name=name, created=now_iso(), host=host_info(),
                         session_dir=str(self.dir), save_dir=str(self.save_dir), use_dll=use_dll,
                         affinity=self.affinity, launches=[], runs=[], notes=[])
        if self.extra_env:
            self.info['extra_env'] = self.extra_env
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
            for f in sorted(self.init_save_from.iterdir()):
                if (f.is_file() and f.suffix.lower() != '.dmp' and f.name.lower() not in ('log.txt', 'manifest.json')
                        and f.name.lower() not in self.init_save_exclude):
                    shutil.copy2(f, sd / f.name)
                    copied.append(f.name)
            self.info['init_save_from'] = dict(source=str(self.init_save_from), dest=str(sd), files=copied,
                                               excluded=sorted(self.init_save_exclude), displaced=displaced,
                                               note='只拷贝进本会话的存档目录，会话结束不写回来源')
        self.info['save_pre'] = snapshot_save(self.dir / 'save/pre', sd)
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

    def write_session_file(self, exe_sha, version):
        """mod 在每局开局时把这个文件原样拼进 run header 的 session 字段（必须单行 JSON）。"""
        mods = [dict(directory=m['directory'], disabled=m['disabled'], name=m.get('meta_name'), id=m.get('meta_id'),
                     version=m.get('meta_version'), files_sha256=m['files']) for m in self.info['mods']]
        obj = dict(session=self.name, exe_sha256=exe_sha,
                   binaries={k: v['sha256'] for k, v in self.info['copy']['binaries'].items()},
                   game_version=version, mods=mods, options_forced=self.forced_options if self.enforce_options else {},
                   launch_command=([str(self.copy.launcher), '--launch', str(self.copy.dll)] if self.use_dll else [])
                   + [str(self.copy.exe), '--luadebug'], host=self.info['host']['hostname'],
                   save_dir=str(self.save_dir), use_dll=self.use_dll,
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
    def finish(self):
        if self.watcher:
            self.watcher.poll_once()
            self.watcher.stop_flag = True
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
        self.info['save_post'] = snapshot_save(self.dir / 'save/post', sd)
        opt = sd / 'options.ini'
        if opt.exists():
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
        self.info['console'] = dict(cmd_history_new_lines=new, note='内建控制台命令只出现在退出时写的 cmd_history.txt（无帧号）')
        for lg in sorted(self.game_logs.glob('*.log.txt')):
            v = game_version_from_log(lg)
            if v:
                self.info.setdefault('game_version', v)
        self.archive_runs(console_used=bool(new))
        if self.docs_watcher is not None:
            self.info['real_documents_watch'] = self.docs_watcher.result()
        self.info['finished'] = now_iso()
        self.info['summary'] = self.summary()
        write_json(self.dir / 'session.json', self.info)
        (self.dir / 'summary.txt').write_text(self.summary_text(), encoding='utf-8')
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
            rec['console_used'] = console_used
            if console_used:
                rec['console_note'] = '本会话 cmd_history.txt 有新增行，无法定位到具体局，会话内所有局都标记'
            v = rec.get('validation') or {}
            if v.get('truncated'):
                rec['note'] = truncated_note(self.info.get('game_exit'))
            if self.raw_policy == 'stage' and rec.get('verified'):
                rec['raw_staged_to'] = stage_raw(raw, self.out_root / '_raw_staging', self.name)
            else:
                rec['raw_kept'] = str(raw)
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
                    real_documents_galaxy_appeared=(self.info.get('real_documents_watch') or {}).get('appeared_during_session'))

    def summary_text(self):
        s = self.info['summary']
        lines = [f'会话 {self.name}', f'局数 {s["runs"]}（续局 {s["continued_runs"]}，截断 {s["truncated_runs"]}）',
                 f'逻辑帧 {s["frames"]}，游戏时间 {s["game_hours"]:.4f} 小时（{s["frames"] / LOGIC_HZ:.1f} 秒）',
                 f'压缩后 {s["zst_mb"]:.3f} MB（原始 {s["raw_mb"]:.3f} MB），{s["zst_mb_per_hour"]} MB/小时',
                 f'启动 {s["launch_attempts"]} 次，失败 {s["launch_failures"]} 次；游戏进程退出码 {s.get("game_exit_hex")}'
                 + ('（异常退出）' if s.get('game_exit_abnormal') else ''),
                 f'DLL {"开" if s.get("use_dll") else "关（-NoDll）"}；有 MT reseed 的局 {s.get("mt_runs")}；存档目录 {self.save_dir}',
                 f'会话期间真实 Documents 出现 (Galaxy) 目录：{s.get("real_documents_galaxy_appeared")}', '']
        for r in self.info['runs']:
            v = r.get('validation') or {}
            h = v.get('header') or {}
            lines.append(f'{r["file"]}: 帧 {v.get("frames")} 种子 {h.get("seed")} 续局 {h.get("continued")} '
                         f'结束 {(v.get("end") or {}).get("reason")} zst {r.get("zst_bytes")} B 校验 {r.get("verified")} '
                         f'MT {(h.get("mt") or {}).get("available")} salt {(h.get("mt") or {}).get("salt")}')
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


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='整局录制会话（人类会话入口；rl/record_run.ps1 调用）')
    ap.add_argument('--session', required=True)
    ap.add_argument('--out-root', default=str(ISAAC_ROOT / 'experiments/run-recorder/sessions'))
    ap.add_argument('--copy', default=str(DEFAULT_COPY))
    ap.add_argument('--init-save-from', default=None)
    ap.add_argument('--raw', choices=['stage', 'keep'], default='stage')
    ap.add_argument('--auto-seconds', type=float, default=0, help='>0 时由脚本自动按键代替人类（测试用）')
    ap.add_argument('--attempts', type=int, default=6)
    ap.add_argument('--no-dll', action='store_true', help='不注入 rr_mt.dll（第二阶段行为：真实 Documents 存档、无 MT reseed）')
    ap.add_argument('--obs', choices=['full', 'keyframe'], default='full', help='观测模式（E5；默认 full = 每帧完整观测）')
    ap.add_argument('--affinity', default=DEFAULT_AFFINITY,
                    help='游戏进程的 CPU 亲和掩码（phase8-mouse §H4；默认 auto = 逻辑 CPU 0 一个核，录下的局才可能精确回放；'
                         '0 = 不限制；0x… = 指定掩码；只在带 DLL 时生效）')
    a = ap.parse_args(argv)
    extra_env = {'IRR_OBS': 'keyframe'} if a.obs == 'keyframe' else None
    sess = Session(a.session, a.out_root, a.copy, a.init_save_from, raw_policy=a.raw, use_dll=not a.no_dll,
                   extra_env=extra_env, affinity=a.affinity)
    print(f'[录制] 会话目录 {sess.dir}', flush=True)
    sess.prepare()
    print(f'[录制] {"带 DLL（MT reseed、存档在副本 profile/）" if sess.use_dll else "不带 DLL（-NoDll）"}；存档目录 {sess.save_dir}',
          flush=True)
    print(f'[录制] CPU 亲和 {sess.affinity or "不限制（回放将无法保证精确）"}', flush=True)
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
