"""isaac_bridge.turbo 的端到端自测（不需要游戏）：

启动 rl/turbo/build/probe_host.exe（它加载 isaac_turbo.dll，被身份门拦住但会建立命名共享内存并保持存活），
Python 端附着到同一控制块：读到 status=identity_failed、pid、日志路径；写 flags/render_every/render_request，
探针退出时回显这些值，证明共享内存双向可用且布局一致。

    python test_turbo_control.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaac_bridge.turbo import (DEFAULT_DLL, TURBO_ROOT, TurboControl, FLAG_SKIP_RENDER,  # noqa: E402
                                FLAG_VIRTUAL_CLOCK, FLAG_FONT_GUARD, FLAG_FILE_RETRY, FLAG_PROBE_DUMP)

PROBE = os.path.join(TURBO_ROOT, "build", "probe_host.exe")


def main() -> int:
    for path in (PROBE, DEFAULT_DLL):
        if not os.path.isfile(path):
            print("SKIP missing", path, "(run rl/turbo/build.ps1 first)")
            return 2
    env = dict(os.environ)
    env["ISAAC_TURBO_LOG_DIR"] = os.path.join(TURBO_ROOT, "build", "probe-logs")
    proc = subprocess.Popen([PROBE, DEFAULT_DLL, "1500"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env)
    ctl = TurboControl(proc.pid)
    deadline = time.monotonic() + 5.0
    while not ctl.is_initialized() and time.monotonic() < deadline:
        time.sleep(0.02)
    ok = True
    if not ctl.is_initialized():
        print("FAIL control block never initialized")
        ok = False
    else:
        st = ctl.stats()
        print(f"probe pid={proc.pid} status={st.status_name} text=\"{st.status_text}\" log={st.log_path}")
        if st.status_name != "identity_failed" or st.pid != proc.pid or st.hooks_mask != 0:
            print("FAIL unexpected status/pid/hooks:", st.to_dict())
            ok = False
        probe = st.probe_summary()
        if any(probe[k] for k in ("push_calls", "push_failures", "font_calls", "font_skipped", "file_open_calls",
                                  "probe_dumps_written")) or probe["last_fail_text"] != "":
            print("FAIL probe counters not zero on a non-game host:", probe)
            ok = False
        if st.font_shader_flags != 0xFFFFFFFF:
            print("FAIL font_shader_flags should start as 0xFFFFFFFF (unknown):", hex(st.font_shader_flags))
            ok = False
        # 探针位与加速位互不覆盖
        ctl.set_probe_flags(font_guard=True, file_retry=True, probe_dump=True)
        if ctl.stats().flags != (FLAG_FONT_GUARD | FLAG_FILE_RETRY | FLAG_PROBE_DUMP):
            print("FAIL probe flags write-back", hex(ctl.stats().flags))
            ok = False
        ctl.set_probe_flags(font_guard=False, file_retry=False, probe_dump=False)
        flags = ctl.set_flags(virtual_clock=True, skip_render=True)
        ctl.set_render_every(7)
        req = ctl.request_render()
        if flags != (FLAG_VIRTUAL_CLOCK | FLAG_SKIP_RENDER) or req != 1:
            print("FAIL write-back", flags, req)
            ok = False
    out, _ = proc.communicate(timeout=10)
    print(out.strip())
    if "HOLD_END flags=0x3 render_every=7 render_request=1" not in out:
        print("FAIL probe did not observe controller writes")
        ok = False
    if proc.returncode != 0:
        print("FAIL probe rc", proc.returncode)
        ok = False
    ctl.close()
    print("PASS test_turbo_control" if ok else "FAIL test_turbo_control")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
