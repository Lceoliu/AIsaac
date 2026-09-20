"""阶段 1：证明可控。

  A 重置可靠：同一 seed/房间连续 reset 两次，初始观测摘要一致（玩家/实体位置、事件计数清零）。
  B 步进精确：step(action, N) 后 game_frame、room.frame、mod 的 logic_frames 都恰好推进 N（N = 1,2,4,8,30）。
  C 不偷跑：不发指令空等 --idle 秒，game_frame 与 DLL 的 manager_update_calls / game_update_calls 都不变。
  D 加速下重复 B、C（需要控制块；--launch 或 --pid），并粗测 300 帧的逻辑帧/秒作预览。

    python -m feasibility.stage1_control --launch --room "goto d.12"
    python -m feasibility.stage1_control --pid 12345 --inject --port 27015   # 种子默认 auto：第一局的真实种子
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

from feasibility.common import NOOP, Session, add_session_args, digest, fmt_row, out_dir, write_json


def check_reset(s: Session, results: List[Dict[str, Any]]) -> None:
    t0 = time.perf_counter()
    obs1, _ = s.reset()
    t1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    obs2, _ = s.reset()
    t2 = time.perf_counter() - t0
    d1, d2 = digest(obs1), digest(obs2)
    same_player = d1["player_pos"] == d2["player_pos"]
    same_entities = d1["entities"] == d2["entities"]
    events_zero = all(v == 0 for v in d2["events"].values()) if d2["events"] else None
    ok = same_player and same_entities and (events_zero is not False)
    detail = (f"reset {t1:.2f}s / {t2:.2f}s; player {d1['player_pos']} vs {d2['player_pos']}; "
              f"entities {len(d1['entities'])} vs {len(d2['entities'])} same={same_entities}; events_zero={events_zero}")
    print(fmt_row("A reset determinism", ok, detail))
    results.append({"check": "A_reset", "ok": ok, "reset_seconds": [t1, t2], "d1": d1, "d2": d2})


def check_stepping(s: Session, results: List[Dict[str, Any]], label: str) -> None:
    all_ok = True
    rows = []
    for n in (1, 2, 4, 8, 30):
        before = digest(s.obs())
        obs, _, _, _, _ = s.step(NOOP, repeat=n)
        after = digest(obs)
        dg = (after["game_frame"] or 0) - (before["game_frame"] or 0)
        dr = (after["room_frame"] or 0) - (before["room_frame"] or 0)
        dl = (after["logic_frames"] or 0) - (before["logic_frames"] or 0) if after["logic_frames"] is not None else None
        ok = dg == n and dr == n and (dl is None or dl == n)
        all_ok &= ok
        rows.append({"n": n, "d_game_frame": dg, "d_room_frame": dr, "d_logic_frames": dl, "ok": ok})
    detail = " ".join(f"N={r['n']}:{r['d_game_frame']}/{r['d_room_frame']}/{r['d_logic_frames']}" for r in rows)
    print(fmt_row(f"B exact stepping [{label}]", all_ok, detail))
    results.append({"check": f"B_stepping_{label}", "ok": all_ok, "rows": rows})


def check_idle(s: Session, results: List[Dict[str, Any]], label: str, idle: float) -> None:
    before = digest(s.obs())
    shm0 = s.shm()
    time.sleep(idle)
    after = digest(s.obs())
    shm1 = s.shm()
    ok = after["game_frame"] == before["game_frame"]
    detail = f"idle {idle:.1f}s: game_frame {before['game_frame']} -> {after['game_frame']}"
    if shm0 and shm1:
        same_mu = shm1["manager_update_calls"] == shm0["manager_update_calls"]
        same_gu = shm1["game_update_calls"] == shm0["game_update_calls"]
        ok = ok and same_mu and same_gu
        detail += (f"; manager_update {shm0['manager_update_calls']}->{shm1['manager_update_calls']}"
                   f"; game_update {shm0['game_update_calls']}->{shm1['game_update_calls']}")
    print(fmt_row(f"C no hidden progress [{label}]", ok, detail))
    results.append({"check": f"C_idle_{label}", "ok": ok, "before": before, "after": after, "shm0": shm0, "shm1": shm1})


def preview_throughput(s: Session, results: List[Dict[str, Any]], frames: int) -> None:
    before = digest(s.obs())
    shm0 = s.shm()
    t0 = time.perf_counter()
    steps = 0
    while steps * 10 < frames:
        s.step(NOOP, repeat=10)
        steps += 1
    dt = time.perf_counter() - t0
    after = digest(s.obs())
    shm1 = s.shm()
    df = (after["game_frame"] or 0) - (before["game_frame"] or 0)
    detail = f"{df} logic frames in {dt:.2f}s = {df / dt:.0f} frames/s ({df / dt / 30:.1f}x realtime)"
    if shm0 and shm1:
        fast = shm1["game_update_fast_calls"] - shm0["game_update_fast_calls"]
        fast_us = shm1["game_update_fast_total_us"] - shm0["game_update_fast_total_us"]
        detail += f"; Game::Update fast avg {fast_us / fast if fast else 0:.0f} us; render_skipped +{shm1['render_skipped'] - shm0['render_skipped']}"
    print(fmt_row("D throughput preview (turbo)", df > 0, detail))
    results.append({"check": "D_preview", "frames": df, "seconds": dt, "shm0": shm0, "shm1": shm1})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_session_args(p)
    p.add_argument("--idle", type=float, default=3.0)
    p.add_argument("--preview-frames", type=int, default=300)
    args = p.parse_args()
    out = out_dir(args, "stage1")
    s = Session(args)
    results: List[Dict[str, Any]] = []
    try:
        s.set_turbo(False)
        check_reset(s, results)
        check_stepping(s, results, "normal")
        check_idle(s, results, "normal", args.idle)
        if s.set_turbo(True):
            print("[stage1] turbo ON:", s.shm())
            check_stepping(s, results, "turbo")
            check_idle(s, results, "turbo", args.idle)
            preview_throughput(s, results, args.preview_frames)
            s.set_turbo(False)
        else:
            print(fmt_row("D turbo checks", None, "no control block (start with --launch or --pid/--inject)"))
    finally:
        s.close()
    ok = all(r.get("ok", True) for r in results)
    write_json(os.path.join(out, "stage1.json"), {"args": vars(args), "seed_cmd": s.seed_cmd, "hello": s.hello,
                                                  "results": results, "ok": ok})
    print(("PASS" if ok else "FAIL") + f" stage1 -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
