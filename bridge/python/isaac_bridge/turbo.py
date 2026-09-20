"""isaac_turbo.dll 的控制端：命名共享内存控制块、注入、带加速层/探针的启动。

控制块布局与 rl/turbo/src/turbo_shared.hpp 一一对应（改一处必须改另一处）：

    0x00 magic 'ITRB'   0x04 version   0x08 pid   0x0C flags(控制端写)
    0x10 render_every   0x14 render_request(控制端递增)   0x18 render_done   0x1C status
    0x20.. 计数器（见 TurboStats）   0x80 status_text[128]   0x100 log_path[256]   0x200.. 附加计数器
    0x228.. 崩溃探针计数器与最近失败摘要（2026-09-20，rl/docs/NATIVE_CRASH_ANALYSIS.md）

DLL 每次 Hook 调用都直接读 flags/render_every/render_request，所以这些字段可以在运行中随时改。
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import mmap
import os
import re
import struct
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .launch import DEFAULT_GAME_DIR, launch

HERE = os.path.dirname(os.path.abspath(__file__))
TURBO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "turbo"))
DEFAULT_DLL = os.path.join(TURBO_ROOT, "build", "isaac_turbo.dll")
DEFAULT_INJECTOR = os.path.join(TURBO_ROOT, "build", "isaac_turbo_inject.exe")

MAGIC = 0x42525449
LAYOUT_VERSION = 1
BLOCK_BYTES = 4096
FLAG_VIRTUAL_CLOCK = 1 << 0
FLAG_SKIP_RENDER = 1 << 1
FLAG_FONT_GUARD = 1 << 2
FLAG_FILE_RETRY = 1 << 3
FLAG_PROBE_DUMP = 1 << 4
STATUS_NAMES = {0: "loading", 1: "active", 2: "identity_failed", 3: "hook_failed", 4: "shm_failed"}
HOOK_NAMES = {1: "glfw_get_time", 2: "manager_update", 4: "game_update", 8: "render_frame",
              16: "push_shader", 32: "font_draw", 64: "file_open_plain", 128: "crt_access"}

_HEAD = struct.Struct("<8I")          # 0x000..0x020
_STATS = struct.Struct("<8QdQQII")    # 0x020..0x080
_TEXT = struct.Struct("<128s256s")    # 0x080..0x200
_EXTRA = struct.Struct("<4QII")       # 0x200..0x228
_PROBE = struct.Struct("<8Q4I256s4Q")   # 0x228..0x398
assert _HEAD.size == 0x20 and _STATS.size == 0x60 and _TEXT.size == 0x180 and _EXTRA.size == 0x28
assert _PROBE.size == 0x170
OFF_FLAGS, OFF_RENDER_EVERY, OFF_RENDER_REQUEST, OFF_RENDER_DONE, OFF_STATUS = 0x0C, 0x10, 0x14, 0x18, 0x1C
OFF_PROBE = 0x228


class TurboError(RuntimeError):
    pass


@dataclass
class TurboStats:
    magic: int
    version: int
    pid: int
    flags: int
    render_every: int
    render_request: int
    render_done: int
    status: int
    manager_update_calls: int
    game_update_calls: int
    game_update_total_us: int
    game_update_max_us: int
    game_update_last_us: int
    render_calls: int
    render_skipped: int
    clock_reads: int
    clock_value: float
    virtual_ticks: int
    wall_us: int
    hooks_mask: int
    draw_parity_ok: int
    status_text: str
    log_path: str
    game_update_fast_calls: int
    game_update_fast_total_us: int
    game_update_min_us: int
    render_opportunities: int
    counter_checks: int
    counter_mismatches: int
    push_calls: int
    push_failures: int
    font_calls: int
    font_skipped: int
    file_open_calls: int
    file_open_failures: int
    file_open_retries: int
    file_open_retry_ok: int
    probe_dumps_written: int
    registry_size_seen: int
    font_shader_flags: int
    probe_reserved: int
    last_fail_text: str
    access_calls: int
    access_failures: int
    access_retries: int
    access_retry_ok: int

    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(self.status, f"unknown({self.status})")

    @property
    def virtual_clock(self) -> bool:
        return bool(self.flags & FLAG_VIRTUAL_CLOCK)

    @property
    def skip_render(self) -> bool:
        return bool(self.flags & FLAG_SKIP_RENDER)

    @property
    def font_guard(self) -> bool:
        return bool(self.flags & FLAG_FONT_GUARD)

    @property
    def file_retry(self) -> bool:
        return bool(self.flags & FLAG_FILE_RETRY)

    @property
    def probe_dump(self) -> bool:
        return bool(self.flags & FLAG_PROBE_DUMP)

    @property
    def hooks(self) -> list:
        return [name for bit, name in HOOK_NAMES.items() if self.hooks_mask & bit]

    @property
    def game_update_fast_avg_us(self) -> float:
        return self.game_update_fast_total_us / self.game_update_fast_calls if self.game_update_fast_calls else 0.0

    def probe_summary(self) -> Dict[str, Any]:
        return {"push_calls": self.push_calls, "push_failures": self.push_failures, "font_calls": self.font_calls,
                "font_skipped": self.font_skipped, "file_open_calls": self.file_open_calls,
                "file_open_failures": self.file_open_failures, "file_open_retries": self.file_open_retries,
                "file_open_retry_ok": self.file_open_retry_ok, "probe_dumps_written": self.probe_dumps_written,
                "registry_size_seen": self.registry_size_seen, "font_shader_flags": hex(self.font_shader_flags),
                "access_calls": self.access_calls, "access_failures": self.access_failures,
                "access_retries": self.access_retries, "access_retry_ok": self.access_retry_ok,
                "last_fail_text": self.last_fail_text}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(status_name=self.status_name, hooks=self.hooks, game_update_fast_avg_us=self.game_update_fast_avg_us)
        return d


def _cstr(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace")


class TurboControl:
    """附着到某个 isaac-ng.exe 进程的控制块。进程尚未注入时也可以先创建并预写配置。"""

    def __init__(self, pid: int):
        self.pid = int(pid)
        self.name = f"IsaacTurbo.{self.pid}"
        # tagname 命名映射：已存在则打开，否则创建（DLL 之后会 OpenFileMapping 到同一对象）
        self.mm = mmap.mmap(-1, BLOCK_BYTES, tagname=self.name)

    # ------------------------------------------------------------------ 读
    def stats(self) -> TurboStats:
        head = _HEAD.unpack_from(self.mm, 0x00)
        stats = _STATS.unpack_from(self.mm, 0x20)
        text = _TEXT.unpack_from(self.mm, 0x80)
        extra = _EXTRA.unpack_from(self.mm, 0x200)
        probe = _PROBE.unpack_from(self.mm, OFF_PROBE)
        return TurboStats(*head, *stats, _cstr(text[0]), _cstr(text[1]), *extra, *probe[:12], _cstr(probe[12]),
                          *probe[13:])

    def is_initialized(self) -> bool:
        return struct.unpack_from("<I", self.mm, 0)[0] == MAGIC

    def status(self) -> int:
        return struct.unpack_from("<I", self.mm, OFF_STATUS)[0]

    # ------------------------------------------------------------------ 写
    def preconfigure(self, virtual_clock: bool = False, skip_render: bool = False, render_every: int = 0,
                     font_guard: bool = False, file_retry: bool = False, probe_dump: bool = False) -> None:
        """注入前写入配置。DLL 打开时若看到 magic/version 匹配就沿用 flags/render_every。"""
        self.mm[0:BLOCK_BYTES] = b"\0" * BLOCK_BYTES
        flags = ((FLAG_VIRTUAL_CLOCK if virtual_clock else 0) | (FLAG_SKIP_RENDER if skip_render else 0)
                 | (FLAG_FONT_GUARD if font_guard else 0) | (FLAG_FILE_RETRY if file_retry else 0)
                 | (FLAG_PROBE_DUMP if probe_dump else 0))
        _HEAD.pack_into(self.mm, 0, MAGIC, LAYOUT_VERSION, self.pid, flags, int(render_every), 0, 0, 0)

    def clear_status(self) -> None:
        """注入（或重新注入）前把状态清回 loading，避免 wait_active 读到上一次残留的失败状态。"""
        struct.pack_into("<I", self.mm, OFF_STATUS, 0)
        self.mm[0x80:0x100] = b"\0" * 0x80

    def _update_flags(self, **bits: Optional[bool]) -> int:
        table = {"virtual_clock": FLAG_VIRTUAL_CLOCK, "skip_render": FLAG_SKIP_RENDER, "font_guard": FLAG_FONT_GUARD,
                 "file_retry": FLAG_FILE_RETRY, "probe_dump": FLAG_PROBE_DUMP}
        flags = struct.unpack_from("<I", self.mm, OFF_FLAGS)[0]
        for name, value in bits.items():
            if value is None:
                continue
            bit = table[name]
            flags = (flags | bit) if value else (flags & ~bit)
        struct.pack_into("<I", self.mm, OFF_FLAGS, flags)
        return flags

    def set_flags(self, virtual_clock: Optional[bool] = None, skip_render: Optional[bool] = None) -> int:
        """只改加速层两位；探针位保持不变。"""
        return self._update_flags(virtual_clock=virtual_clock, skip_render=skip_render)

    def set_probe_flags(self, font_guard: Optional[bool] = None, file_retry: Optional[bool] = None,
                        probe_dump: Optional[bool] = None) -> int:
        return self._update_flags(font_guard=font_guard, file_retry=file_retry, probe_dump=probe_dump)

    def set_turbo(self, enabled: bool) -> int:
        return self.set_flags(virtual_clock=enabled, skip_render=enabled)

    def set_render_every(self, n: int) -> None:
        struct.pack_into("<I", self.mm, OFF_RENDER_EVERY, int(n))

    def request_render(self) -> int:
        req = struct.unpack_from("<I", self.mm, OFF_RENDER_REQUEST)[0] + 1
        struct.pack_into("<I", self.mm, OFF_RENDER_REQUEST, req & 0xFFFFFFFF)
        return req

    def wait_render(self, request: int, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if struct.unpack_from("<I", self.mm, OFF_RENDER_DONE)[0] >= request:
                return True
            time.sleep(0.005)
        return False

    def wait_active(self, timeout: float = 30.0) -> TurboStats:
        deadline = time.monotonic() + timeout
        while True:
            st = self.stats() if self.is_initialized() else None
            if st and st.status == 1:
                return st
            if st and st.status in (2, 3, 4):
                raise TurboError(f"isaac_turbo failed: {st.status_name}: {st.status_text} (log {st.log_path})")
            if time.monotonic() > deadline:
                raise TurboError("isaac_turbo did not become active in time" + (f": {st.status_text}" if st else ""))
            time.sleep(0.05)

    def close(self) -> None:
        self.mm.close()


# ---------------------------------------------------------------------- 注入与启动

def stage_dll(dll: str, pid: int) -> str:
    """把构建产物复制到 %LOCALAPPDATA%\\IsaacRL\\turbo\\stage\\ 下的带 pid 与时间戳的文件名再注入。

    游戏进程会一直锁住被加载的 DLL 文件；注入副本可以让 build.ps1 随时重新生成 build/isaac_turbo.dll，
    也保证同一进程内重复注入（例如修复后再注入）加载的是新模块而不是已加载的旧模块。
    """
    import shutil
    stage_dir = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "IsaacRL", "turbo", "stage")
    os.makedirs(stage_dir, exist_ok=True)
    dst = os.path.join(stage_dir, f"isaac_turbo-{int(pid)}-{int(time.time() * 1000)}.dll")
    shutil.copy2(dll, dst)
    return dst


def inject(pid: int, dll: str = DEFAULT_DLL, injector: str = DEFAULT_INJECTOR, stage: bool = True) -> str:
    for path in (dll, injector):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} (先运行 rl/turbo/build.ps1)")
    target = stage_dll(dll, pid) if stage else os.path.abspath(dll)
    proc = subprocess.run([injector, str(int(pid)), target], capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0 or "PASS" not in out:
        raise TurboError(f"inject failed rc={proc.returncode}: {out}")
    return out


def launch_turbo(port: int, game_dir: str = DEFAULT_GAME_DIR, privileged: bool = True,
                 virtual_clock: bool = False, skip_render: bool = False, render_every: int = 0,
                 inject_after_s: float = 8.0, log_dir: Optional[str] = None,
                 dll: str = DEFAULT_DLL, injector: str = DEFAULT_INJECTOR) -> Tuple[subprocess.Popen, TurboControl]:
    """启动 isaac-ng.exe --luadebug，预写控制块，延迟注入 DLL，等待其报告 active。

    默认以直通模式（不改时钟、不跳渲染）启动；进入一局并连上桥接后再 set_turbo(True)，这样
    加载画面、菜单和第一次开局都按原样进行，也便于同一进程内做"正常 vs 加速"对比。
    """
    extra_env = {"ISAAC_TURBO_LOG_DIR": log_dir} if log_dir else None
    proc = launch(port, game_dir, privileged, extra_env=extra_env)
    ctl = TurboControl(proc.pid)
    ctl.preconfigure(virtual_clock=virtual_clock, skip_render=skip_render, render_every=render_every)
    deadline = time.monotonic() + inject_after_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise TurboError(f"isaac-ng.exe exited early rc={proc.returncode}")
        time.sleep(0.25)
    ctl.clear_status()
    inject(proc.pid, dll, injector)
    ctl.wait_active(30.0)
    return proc, ctl


def attach(pid: int, do_inject: bool = False, dll: str = DEFAULT_DLL, injector: str = DEFAULT_INJECTOR,
           timeout: float = 30.0) -> TurboControl:
    """附着到已运行的游戏；do_inject=True 时先注入（控制块此时按直通配置创建）。"""
    ctl = TurboControl(pid)
    if do_inject:
        if not ctl.is_initialized():
            ctl.preconfigure()
        else:
            ctl.clear_status()
        inject(pid, dll, injector)
    ctl.wait_active(timeout)
    return ctl


_LAUNCH_RE = re.compile(r"PASS launched pid=(\d+) module=0x([0-9a-fA-F]+) status=(\d+)")


def launch_suspended(port: int, game_dir: str = DEFAULT_GAME_DIR, privileged: bool = True,
                     extra_args: Sequence[str] = ("--luadebug",), skip_render: bool = False,
                     virtual_clock: bool = False, font_guard: bool = True, file_retry: bool = True,
                     probe_dump: bool = True, log_dir: Optional[str] = None, engine_velocity: bool = False,
                     dll: str = DEFAULT_DLL, injector: str = DEFAULT_INJECTOR) -> Tuple[int, TurboControl, str]:
    """挂起创建 isaac-ng.exe、注入探针 DLL、等钩子激活后再放行主线程（启动期取证必须用这个）。

    配置经环境变量传给 DLL（进程创建前无法预写控制块）。返回 (pid, 控制块, 注入器输出)。
    进程不是本 Python 的子进程，用 wait_process / process_alive 跟踪。
    """
    for path in (dll, injector):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} (先运行 rl/turbo/build.ps1)")
    exe = os.path.join(game_dir, "isaac-ng.exe")
    if not os.path.isfile(exe):
        raise FileNotFoundError(exe)
    env = dict(os.environ)
    env["ISAAC_RL_PORT"] = str(port)
    env["ISAAC_RL_PRIVILEGED"] = "1" if privileged else "0"
    env["ISAAC_RL_ENGINE_VEL"] = "1" if engine_velocity else "0"
    env["ISAAC_TURBO_SKIP_RENDER"] = "1" if skip_render else "0"
    env["ISAAC_TURBO_CLOCK"] = "1" if virtual_clock else "0"
    env["ISAAC_TURBO_FONT_GUARD"] = "1" if font_guard else "0"
    env["ISAAC_TURBO_FILE_RETRY"] = "1" if file_retry else "0"
    env["ISAAC_TURBO_PROBE_DUMP"] = "1" if probe_dump else "0"
    env.pop("ISAAC_TURBO", None)
    if log_dir:
        env["ISAAC_TURBO_LOG_DIR"] = log_dir
    # 挂起启动时 pid 未知，先复制一份带时间戳的 DLL（和 inject 一样避免锁住构建产物）。
    staged = stage_dll(dll, 0)
    proc = subprocess.run([injector, "--launch", staged, exe, *extra_args], capture_output=True, text=True, env=env,
                          cwd=game_dir)
    out = (proc.stdout + proc.stderr).strip()
    m = _LAUNCH_RE.search(out)
    if not m:
        raise TurboError(f"launch_suspended failed rc={proc.returncode}: {out}")
    pid, status = int(m.group(1)), int(m.group(3))
    ctl = TurboControl(pid)
    if status != 1 or proc.returncode != 0:
        st = ctl.stats() if ctl.is_initialized() else None
        raise TurboError(f"isaac_turbo not active after suspended launch: status={status} rc={proc.returncode} "
                         f"text={st.status_text if st else '?'} out={out}")
    return pid, ctl, out


# ---------------------------------------------------------------------- 进程跟踪（非子进程）

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_SYNCHRONIZE = 0x00100000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_PROCESS_TERMINATE = 0x0001
_STILL_ACTIVE = 259
_WM_CLOSE = 0x0010
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_kernel32.WaitForSingleObject.restype = wintypes.DWORD
_kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
_kernel32.GetExitCodeProcess.restype = wintypes.BOOL
_kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
_kernel32.TerminateProcess.restype = wintypes.BOOL
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
_user32.EnumWindows.argtypes = [_EnumWindowsProc, wintypes.LPARAM]
_user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
_user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_user32.IsWindowVisible.argtypes = [wintypes.HWND]


def process_exit_code(pid: int) -> Optional[int]:
    """None = 仍在运行；否则退出码（进程不存在时返回 -1）。"""
    h = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return -1
    try:
        code = wintypes.DWORD()
        if not _kernel32.GetExitCodeProcess(h, ctypes.byref(code)):
            return -1
        return None if code.value == _STILL_ACTIVE else int(code.value)
    finally:
        _kernel32.CloseHandle(h)


def process_alive(pid: int) -> bool:
    return process_exit_code(pid) is None


def wait_process(pid: int, timeout: float) -> Optional[int]:
    """等待进程退出，返回退出码；超时返回 None。"""
    h = _kernel32.OpenProcess(_SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return -1
    try:
        if _kernel32.WaitForSingleObject(h, int(timeout * 1000)) != 0:
            return None
        code = wintypes.DWORD()
        _kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        return int(code.value)
    finally:
        _kernel32.CloseHandle(h)


def windows_of(pid: int) -> List[int]:
    found: List[int] = []

    def cb(hwnd, _):
        owner = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and _user32.IsWindowVisible(hwnd):
            found.append(int(hwnd))
        return True

    _user32.EnumWindows(_EnumWindowsProc(cb), 0)
    return found


_WM_KEYDOWN, _WM_KEYUP = 0x0100, 0x0101
VK_RETURN, VK_SPACE, VK_ESCAPE = 0x0D, 0x20, 0x1B
_SCANCODES = {VK_RETURN: 0x1C, VK_SPACE: 0x39, VK_ESCAPE: 0x01}


def post_key(pid: int, vk: int = VK_RETURN) -> int:
    """给进程窗口投递一次按键（WM_KEYDOWN/WM_KEYUP，带 GLFW 需要的扫描码），用于跳过开场动画/进入主菜单。"""
    scan = _SCANCODES.get(vk, 0)
    hwnds = windows_of(pid)
    for hwnd in hwnds:
        _user32.PostMessageW(hwnd, _WM_KEYDOWN, vk, (scan << 16) | 1)
        _user32.PostMessageW(hwnd, _WM_KEYUP, vk, (scan << 16) | 1 | (1 << 30) | (1 << 31))
    return len(hwnds)


def post_close(pid: int) -> int:
    """给进程的可见窗口发 WM_CLOSE（GLFW 走正常关闭：游戏日志会有 'Isaac is shutting down'）。返回发送数。"""
    hwnds = windows_of(pid)
    for hwnd in hwnds:
        _user32.PostMessageW(hwnd, _WM_CLOSE, 0, 0)
    return len(hwnds)


def terminate(pid: int) -> bool:
    h = _kernel32.OpenProcess(_PROCESS_TERMINATE, False, int(pid))
    if not h:
        return False
    try:
        return bool(_kernel32.TerminateProcess(h, 1))
    finally:
        _kernel32.CloseHandle(h)
