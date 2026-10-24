"""阶段 3：测真实成本。

单实例（加速开）：逻辑帧/秒（repeat=1 与 repeat=8 两种决策粒度）、重置耗时（均值/最大）、
Game::Update 纯逻辑均值（DLL fast 桶）、进程 RSS。多实例：--instances 1,2,4 依次启动 N 个游戏
（端口 base+i，共用同一用户档案，需要 PauseOnFocusLost=0），并行跑同样的负载，汇总总帧/秒。

    python -m feasibility.stage3_cost --launch --instances 1,2,4 --frames 3000 --resets 5
    python -m feasibility.stage3_cost --pid 12345 --inject          # 只测当前这一个实例
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Any, Dict, List

from feasibility.common import NOOP, Session, add_session_args, digest, fmt_row, make_policy, out_dir, write_json

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def rss_mb(pid: int) -> float:
    if not psutil or not pid:
        return 0.0
    try:
        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 0.0


def measure_instance(s: Session, frames: int, resets: int, policy_name: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"port": s.port}
    s.set_turbo(True)
    # 重置耗时（含 restart + seed + goto 与 settle 帧；开局有 ≥3 s 的真实时间保持，见方案 §2）
    reset_times = []
    for _ in range(max(1, resets)):
        t0 = time.perf_counter()
        s.reset()
        reset_times.append(time.perf_counter() - t0)
    result["reset_seconds"] = {"mean": sum(reset_times) / len(reset_times), "max": max(reset_times), "all": reset_times}
    policy = make_policy(policy_name, 0)
    for repeat in (1, 8):
        before = digest(s.obs())
        shm0 = s.shm()
        t0 = time.perf_counter()
        steps = 0
        while steps * repeat < frames:
            _, _, term, _, _ = s.step(policy(steps), repeat=repeat)
            steps += 1
            if term:
                s.reset()
        dt = time.perf_counter() - t0
        after = digest(s.obs())
        shm1 = s.shm()
        df = (after["game_frame"] or 0) - (before["game_frame"] or 0)
        entry = {"steps": steps, "frames": df, "seconds": dt, "frames_per_second": df / dt if dt else 0.0,
                 "decisions_per_second": steps / dt if dt else 0.0}
        if shm0 and shm1:
            fast = shm1["game_update_fast_calls"] - shm0["game_update_fast_calls"]
            fast_us = shm1["game_update_fast_total_us"] - shm0["game_update_fast_total_us"]
            entry["game_update_fast_avg_us"] = fast_us / fast if fast else None
            entry["game_update_min_us"] = shm1["game_update_min_us"]
            entry["render_skipped"] = shm1["render_skipped"] - shm0["render_skipped"]
            entry["render_calls"] = shm1["render_calls"] - shm0["render_calls"]
        result[f"repeat{repeat}"] = entry
    pid = s.proc.pid if s.proc else (s.args.pid or 0)
    result["rss_mb"] = rss_mb(pid)
    result["shm"] = s.shm()
    s.set_turbo(False)
    return result


def run_parallel(args: argparse.Namespace, n: int) -> Dict[str, Any]:
    sessions: List[Session] = []
    errors: List[str] = []
    for i in range(n):
        port = args.port + i
        try:
            sessions.append(Session(args, port=port))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"port {port}: {exc}")
        time.sleep(args.stagger)
    results: List[Dict[str, Any]] = [None] * len(sessions)  # type: ignore[list-item]

    def worker(idx: int) -> None:
        try:
            results[idx] = measure_instance(sessions[idx], args.frames, args.resets, args.policy)
        except Exception as exc:  # noqa: BLE001
            results[idx] = {"error": str(exc), "port": sessions[idx].port}

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(len(sessions))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    for s in sessions:
        s.close()
    ok = [r for r in results if r and "error" not in r]
    agg = {
        "instances_requested": n, "instances_ok": len(ok), "errors": errors + [r["error"] for r in results if r and "error" in r],
        "wall_seconds": wall,
        "sum_frames_per_second_repeat1": sum(r["repeat1"]["frames_per_second"] for r in ok),
        "sum_frames_per_second_repeat8": sum(r["repeat8"]["frames_per_second"] for r in ok),
        "sum_decisions_per_second_repeat8": sum(r["repeat8"]["decisions_per_second"] for r in ok),
        "mean_reset_seconds": sum(r["reset_seconds"]["mean"] for r in ok) / len(ok) if ok else None,
        "total_rss_mb": sum(r["rss_mb"] for r in ok),
        "per_instance": results,
    }
    return agg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_session_args(p)
    p.add_argument("--instances", default="1", help="逗号分隔，例如 1,2,4（多于 1 时必须 --launch）")
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--resets", type=int, default=5)
    p.add_argument("--policy", choices=["scripted", "random"], default="scripted")
    p.add_argument("--stagger", type=float, default=3.0, help="多实例启动间隔秒")
    args = p.parse_args()
    out = out_dir(args, "stage3")
    counts = [int(x) for x in args.instances.split(",") if x.strip()]
    if any(c > 1 for c in counts) and not args.launch:
        print("FAIL multi-instance measurement needs --launch")
        return 1
    report: List[Dict[str, Any]] = []
    for n in counts:
        agg = run_parallel(args, n)
        report.append(agg)
        ok = agg["instances_ok"] == n
        print(fmt_row(f"instances={n}", ok,
                      f"repeat1 {agg['sum_frames_per_second_repeat1']:.0f} f/s, repeat8 {agg['sum_frames_per_second_repeat8']:.0f} f/s "
                      f"({agg['sum_decisions_per_second_repeat8']:.1f} dec/s), reset {agg['mean_reset_seconds'] or 0:.2f}s, "
                      f"RSS {agg['total_rss_mb']:.0f} MB, errors={agg['errors']}"))
    write_json(os.path.join(out, "stage3.json"), {"args": vars(args), "report": report})
    print(f"DONE stage3 -> {out}")
    return 0 if all(r["instances_ok"] == r["instances_requested"] for r in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
