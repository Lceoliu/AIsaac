"""冷启动循环：反复启动到主菜单再正常关闭，复现启动期 B 类与主菜单 A 类崩溃并取证。

默认模式（probe）：launch_suspended（挂起创建 + 启动即注入探针，字体守卫/文件重试/探针转储按参数）。
对照模式（--no-inject）：普通 launch，不注入任何 DLL，用于判断"正常关闭后退出崩溃"（C 类）是否与 turbo 有关。

每一轮：等游戏日志出现桥接 "listening"（归档挂载、shader 初始化、mod 加载都已过）→ 再等主菜单初始化标记
"Menu Online Awards Init"（历史 A 类冷启动崩溃正好发生在它之后）→ 再等 --menu-seconds → 发 WM_CLOSE 正常退出
→ 记录退出码、探针计数、新增的 WER/游戏转储、turbo 日志里的失败行 → 冷却 --cooldown 秒。不改游戏目录，不改 EXE。

结局分类：clean_exit（日志有 "shut down successfully" 且退出码 0）；exit_crash_after_shutdown（日志有 "shut down
successfully" 但退出码非 0，即 C 类）；crash（其他非 0 退出或出现崩溃转储）；launch_error。

    python -m crash_probe.cold_start_loop --runs 8 --out D:\\...\\cold-start-8
    python -m crash_probe.cold_start_loop --runs 3 --no-inject --out D:\\...\\cold-start-control
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaac_bridge.launch import launch  # noqa: E402
from isaac_bridge.turbo import (launch_suspended, post_close, post_key, process_exit_code, terminate,  # noqa: E402
                                wait_process)

GAME_LOG = Path(os.path.expanduser("~")) / "Documents" / "My Games" / "Binding of Isaac Repentance+" / "log.txt"
GAME_DUMPS = GAME_LOG.parent / "crash_dumps"
WER_DUMPS = Path(os.environ.get("LOCALAPPDATA", "")) / "CrashDumps"
LISTENING = "[IsaacRLBridge] listening on"
MENU_READY = "Menu Online Awards Init"
SHUTDOWN_OK = "Isaac has shut down successfully"
MARKERS = ("Shader stack empty", "Caught exception", "Failed to open archive", "checksum is invalid",
           "Isaac is shutting down", SHUTDOWN_OK, MENU_READY, LISTENING, "Lua stack trace", "playing cutscene")


def snapshot(patterns):
    return {p for pattern in patterns for p in glob.glob(pattern)}


def read_game_log(since: float) -> str:
    try:
        if GAME_LOG.stat().st_mtime >= since:
            return GAME_LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return ""


def wait_for_log_marker(marker: str, timeout: float, since: float, pid: int, press_every: float = 0.0) -> int:
    """返回投递的按键次数（-1 = 未等到）。press_every > 0 时每隔该秒数给窗口投递一次回车（跳过开场动画、离开标题画面）。"""
    deadline = time.monotonic() + timeout
    next_press = time.monotonic() + (press_every if press_every > 0 else 1e9)
    presses = 0
    while time.monotonic() < deadline:
        if marker in read_game_log(since):
            return presses
        if process_exit_code(pid) is not None:
            return -1
        if press_every > 0 and time.monotonic() >= next_press:
            post_key(pid)
            presses += 1
            next_press = time.monotonic() + press_every
        time.sleep(0.25)
    return -1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=8)
    p.add_argument("--port", type=int, default=27015)
    p.add_argument("--menu-seconds", type=float, default=5.0, help="主菜单初始化后再渲染多久")
    p.add_argument("--startup-timeout", type=float, default=90.0)
    p.add_argument("--menu-timeout", type=float, default=45.0, help="等待主菜单初始化标记的上限（期间每 4 秒投递一次回车）")
    p.add_argument("--exit-timeout", type=float, default=40.0)
    p.add_argument("--cooldown", type=float, default=5.0)
    p.add_argument("--no-inject", action="store_true", help="对照模式：普通启动，不注入 DLL")
    p.add_argument("--no-font-guard", action="store_true")
    p.add_argument("--no-file-retry", action="store_true")
    p.add_argument("--no-probe-dump", action="store_true")
    p.add_argument("--set-stage", action="store_true", help="附加 --set-stage=1 直接开局（默认停在主菜单）")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    turbo_log_dir = str(args.out / "turbo-logs")
    os.makedirs(turbo_log_dir, exist_ok=True)
    dump_patterns = [str(WER_DUMPS / "isaac-ng.exe.*.dmp"), str(GAME_DUMPS / "*.dmp"), str(GAME_DUMPS / "*.txt"),
                     os.path.join(turbo_log_dir, "probe-*.dmp")]
    extra_args = ["--luadebug"] + (["--set-stage=1"] if args.set_stage else [])
    runs = []
    report = {"status": "started", "mode": "control_no_inject" if args.no_inject else "probe",
              "requested_runs": args.runs, "extra_args": extra_args,
              "font_guard": not (args.no_font_guard or args.no_inject),
              "file_retry": not (args.no_file_retry or args.no_inject),
              "probe_dump": not (args.no_probe_dump or args.no_inject), "runs": runs}
    report_path = args.out / "report.json"
    try:
        for i in range(args.runs):
            before = snapshot(dump_patterns)
            started = time.time()
            run = {"index": i, "started": started, "mode": report["mode"]}
            ctl = None
            proc = None
            try:
                if args.no_inject:
                    proc = launch(args.port, privileged=True, extra_args=tuple(extra_args))
                    pid = proc.pid
                else:
                    pid, ctl, out = launch_suspended(args.port, extra_args=extra_args, skip_render=False,
                                                     font_guard=not args.no_font_guard,
                                                     file_retry=not args.no_file_retry,
                                                     probe_dump=not args.no_probe_dump, log_dir=turbo_log_dir)
                    run["injector"] = out
            except Exception as error:  # noqa: BLE001
                run.update(outcome="launch_error", error=f"{type(error).__name__}: {error}")
                runs.append(run)
                print("RUN", json.dumps(run, ensure_ascii=False), flush=True)
                time.sleep(args.cooldown)
                continue
            run["pid"] = pid
            run["reached_listening"] = wait_for_log_marker(LISTENING, args.startup_timeout, started - 1, pid) >= 0
            presses = -1
            if run["reached_listening"]:
                time.sleep(3.0)  # 让标题/开场动画先起来，再用回车跳过并进入主菜单
                presses = wait_for_log_marker(MENU_READY, args.menu_timeout, started - 1, pid, press_every=4.0)
            run["reached_menu"] = presses >= 0
            run["enter_presses"] = presses
            exit_code = process_exit_code(pid)
            if run["reached_menu"] and exit_code is None:
                time.sleep(args.menu_seconds)
                exit_code = process_exit_code(pid)
            if exit_code is None:
                run["close_posted"] = post_close(pid)
                exit_code = wait_process(pid, args.exit_timeout)
                if exit_code is None:
                    run["terminated"] = terminate(pid)
                    exit_code = wait_process(pid, 10.0)
            if proc is not None:
                proc.poll()
            run["exit_code"] = exit_code
            run["exit_code_hex"] = hex(exit_code & 0xFFFFFFFF) if isinstance(exit_code, int) else None
            if ctl is not None:
                try:
                    st = ctl.stats()
                    run["turbo"] = {"status": st.status_name, "hooks": st.hooks, "flags": hex(st.flags),
                                    "render_calls": st.render_calls, "manager_update_calls": st.manager_update_calls,
                                    **st.probe_summary()}
                except Exception as error:  # noqa: BLE001
                    run["turbo"] = {"error": repr(error)}
                finally:
                    ctl.close()
            text = read_game_log(started - 1)
            run["game_log_markers"] = {k: text.count(k) for k in MARKERS}
            (args.out / f"game-log-{i:02d}-{pid}.txt").write_text(text, encoding="utf-8")
            new_files = sorted(snapshot(dump_patterns) - before)
            run["new_dump_files"] = new_files
            for f in new_files:
                if f.endswith(".txt"):
                    shutil.copy2(f, args.out / os.path.basename(f))
            turbo_log = os.path.join(turbo_log_dir, f"turbo-{pid}.log")
            if os.path.isfile(turbo_log):
                lines = Path(turbo_log).read_text(encoding="utf-8", errors="replace").splitlines()
                run["turbo_fail_lines"] = [l for l in lines if "fail" in l or "guard" in l or "probe dump" in l
                                           or "retry" in l][:60]
            crash_dumps = [f for f in new_files if f.endswith(".dmp") and "probe-" not in os.path.basename(f)]
            shut_down_ok = SHUTDOWN_OK in text
            if exit_code == 0 and not crash_dumps:
                run["outcome"] = "clean_exit"
            elif shut_down_ok and exit_code not in (0, None):
                run["outcome"] = "exit_crash_after_shutdown"
            else:
                run["outcome"] = "crash"
            runs.append(run)
            print("RUN", json.dumps(run, ensure_ascii=False), flush=True)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            time.sleep(args.cooldown)
        report["status"] = "completed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    finally:
        def total(key):
            return sum(r.get("turbo", {}).get(key, 0) or 0 for r in runs)
        report["summary"] = {
            "runs": len(runs),
            "clean_exit": sum(1 for r in runs if r.get("outcome") == "clean_exit"),
            "exit_crash_after_shutdown": sum(1 for r in runs if r.get("outcome") == "exit_crash_after_shutdown"),
            "crash": sum(1 for r in runs if r.get("outcome") == "crash"),
            "launch_error": sum(1 for r in runs if r.get("outcome") == "launch_error"),
            "reached_menu": sum(1 for r in runs if r.get("reached_menu")),
            "push_failures": total("push_failures"), "font_calls": total("font_calls"), "font_skipped": total("font_skipped"),
            "file_open_failures": total("file_open_failures"), "file_open_retry_ok": total("file_open_retry_ok"),
            "access_failures": total("access_failures"), "access_retry_ok": total("access_retry_ok"),
            "probe_dumps": total("probe_dumps_written"),
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("SUMMARY", json.dumps(report["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
