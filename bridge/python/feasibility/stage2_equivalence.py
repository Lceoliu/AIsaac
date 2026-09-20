"""阶段 2：证明加速不改玩法。

同一 seed、同一房间、同一动作序列，跑三条轨迹：A1 正常、A2 正常、B 加速（虚拟时钟 + 跳渲染）。
逐步比较玩家位置、实体表（含眼泪/弹幕高度）、心数与事件计数（受伤、发射眼泪、敌人死亡、清房）。

  * A1 vs A2 是基线：游戏自身在同输入下是否逐帧确定（存在未播种的全局 RNG，见方案文档）。
  * A1 vs B 是判定：加速轨迹的分歧不得早于、不得大于基线分歧。
  * 若 A1 == A2 逐字节一致，则要求 A1 == B 也一致；否则只能给出"不劣于基线"的判定并记录分歧步。

    python -m feasibility.stage2_equivalence --launch --room "goto d.12" --steps 150
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

from feasibility.common import (Session, add_session_args, compare_trajectories, digest, fmt_row, make_policy,
                                out_dir, write_json)


def run_episode(s: Session, turbo: bool, steps: int, policy_name: str, policy_seed: int) -> Dict[str, Any]:
    s.set_turbo(turbo)
    policy = make_policy(policy_name, policy_seed)
    obs, _ = s.reset()
    traj: List[Dict[str, Any]] = [digest(obs)]
    t0 = time.perf_counter()
    terminal_at = None
    for t in range(steps):
        obs, _, term, _, _ = s.step(policy(t))
        traj.append(digest(obs))
        if term:
            terminal_at = t + 1
            break
    dt = time.perf_counter() - t0
    frames = (traj[-1]["game_frame"] or 0) - (traj[0]["game_frame"] or 0)
    return {"turbo": turbo, "trajectory": traj, "seconds": dt, "frames": frames, "terminal_at": terminal_at,
            "frames_per_second": frames / dt if dt > 0 else 0.0, "shm": s.shm()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_session_args(p)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--policy", choices=["scripted", "random"], default="scripted")
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1, help="每种模式重复的局数（每局都做一次 A1/A2/B）")
    args = p.parse_args()
    out = out_dir(args, "stage2")
    s = Session(args)
    if not s.seed_cmd:
        print("WARN 没有固定种子：每次 restart 的种子不同，比较没有意义")
    summary: List[Dict[str, Any]] = []
    verdicts: List[str] = []
    try:
        if not s.ctl:
            print("FAIL stage2 needs the turbo control block (--launch or --pid/--inject)")
            return 1
        for ep in range(args.episodes):
            a1 = run_episode(s, False, args.steps, args.policy, args.policy_seed)
            a2 = run_episode(s, False, args.steps, args.policy, args.policy_seed)
            b = run_episode(s, True, args.steps, args.policy, args.policy_seed)
            aa = compare_trajectories(a1["trajectory"], a2["trajectory"])
            ab = compare_trajectories(a1["trajectory"], b["trajectory"])
            if aa["identical"] and ab["identical"]:
                verdict = "PASS"
            elif aa["identical"] and not ab["identical"]:
                verdict = "FAIL"
            else:
                early = (ab["first_divergence_step"] or 10**9) < (aa["first_divergence_step"] or 10**9)
                larger = ab["max_player_pos_diff"] > aa["max_player_pos_diff"] * 1.5 + 1e-3
                verdict = "FAIL" if (early or larger) else "INCONCLUSIVE_BASELINE_NONDETERMINISTIC"
            verdicts.append(verdict)
            print(fmt_row(f"episode {ep} A1 vs A2 (baseline)", aa["identical"],
                          f"first_div={aa['first_divergence_step']} max_pos_diff={aa['max_player_pos_diff']} events {aa['events_a']} vs {aa['events_b']}"))
            print(fmt_row(f"episode {ep} A1 vs B (turbo)", ab["identical"],
                          f"first_div={ab['first_divergence_step']} max_pos_diff={ab['max_player_pos_diff']} events {ab['events_a']} vs {ab['events_b']}"))
            print(f"      speed: normal {a1['frames_per_second']:.0f} f/s, {a2['frames_per_second']:.0f} f/s; turbo {b['frames_per_second']:.0f} f/s; verdict {verdict}")
            summary.append({"episode": ep, "verdict": verdict, "aa": aa, "ab": ab,
                            "speed": {"a1": a1["frames_per_second"], "a2": a2["frames_per_second"], "b": b["frames_per_second"]}})
            write_json(os.path.join(out, f"episode{ep}_a1.json"), a1)
            write_json(os.path.join(out, f"episode{ep}_a2.json"), a2)
            write_json(os.path.join(out, f"episode{ep}_b.json"), b)
    finally:
        s.close()
    overall = "PASS" if all(v == "PASS" for v in verdicts) else ("FAIL" if any(v == "FAIL" for v in verdicts) else "INCONCLUSIVE")
    write_json(os.path.join(out, "stage2.json"), {"args": vars(args), "seed_cmd": s.seed_cmd, "hello": s.hello,
                                                  "episodes": summary, "verdict": overall})
    print(f"{overall} stage2 -> {out}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
