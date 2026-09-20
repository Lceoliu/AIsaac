"""Exercise natural episode termination and reset through the Gymnasium adapter."""
import argparse
import json
from pathlib import Path
import time
from dataclasses import asdict

import numpy as np

from isaac_bridge.monstro_gym import MonstroGymEnv
from isaac_bridge.env import BridgeError
from isaac_bridge.rendering import configure_rendering
from isaac_bridge.turbo import TurboError, launch_suspended, process_exit_code, HOOK_CAPTURE_OVERLAY


def tracking_action(obs):
    """Visible-only diagnostic controller, not a learned policy or benchmark score."""
    p = obs["player"]
    rows = obs["entities"].reshape(-1, 14)[obs["mask"].astype(bool)]
    enemies = rows[rows[:, 7] > 0]
    if not len(enemies):
        return np.array([0, 0, 0, 0])
    boss = enemies[0]
    dx, dy = boss[2] - p[0], boss[3] - p[1]
    # Stay vertically separated, align to shoot, retreat from the landing point.
    vx = float(np.clip(dx * 3, -1, 1))
    vy = (1 if dy < 0 else -1) * max(0., .48 - abs(dy)) * 4
    if p[1] > .85: vy -= (p[1] - .85) * 8
    if p[1] < .15: vy += (.15 - p[1]) * 8
    hazards = rows[(rows[:, 8] > 0) | (rows[:, 7] > 0)]
    for e in hazards:
        ex, ey = p[0] - e[2], p[1] - e[3]
        d2 = ex * ex + ey * ey
        if d2 < .05:
            vx += ex / (d2 + .002) * .12
            vy += ey / (d2 + .002) * .12
    mx, my = int(abs(vx) > .12) * int(np.sign(vx)), int(abs(vy) > .12) * int(np.sign(vy))
    moves = {(0, 0): 0, (0, -1): 1, (1, -1): 2, (1, 0): 3,
             (1, 1): 4, (0, 1): 5, (-1, 1): 6, (-1, 0): 7, (-1, -1): 8}
    shoot = (1 if dy < 0 else 3) if abs(dy) >= abs(dx) else (2 if dx > 0 else 4)
    return np.array([moves[(mx, my)], shoot, 0, 0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=27015)
    p.add_argument("--launch", action="store_true")
    p.add_argument("--pid", type=int, help="existing worker PID when attaching")
    p.add_argument("--render-mode", choices=("headless", "visible"), default="headless")
    p.add_argument("--connect-timeout", type=float, default=30.)
    p.add_argument("--frames", type=int, default=5400)
    p.add_argument("--policies", default="idle,track,track")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.launch == (args.pid is not None):
        p.error("specify exactly one of --launch or --pid")
    args.out.mkdir(parents=True, exist_ok=True)
    pid = args.pid
    launch_output = None
    if args.launch:
        pid, startup_control, launch_output = launch_suspended(
            args.port, extra_args=("--luadebug", "--set-stage=1"), skip_render=args.render_mode == "headless",
            virtual_clock=False, font_guard=False, file_retry=False, probe_dump=False,
            log_dir=str(args.out / "native"))
        startup_control.close()
    print("PID", pid, "render_mode", args.render_mode, flush=True)
    env = MonstroGymEnv(port=args.port, max_episode_frames=args.frames)
    env.bridge.connect_timeout = args.connect_timeout
    report = {"episodes": [], "status": "started", "turbo": False, "pid": pid,
              "render_mode": args.render_mode, "virtual_clock": False, "launch_output": launch_output}
    render_control = None
    start = time.perf_counter()
    try:
        env.bridge.connect()
        env.connected = True
        render_control = configure_rendering(pid, args.render_mode)
        report["render_before"] = asdict(render_control.stats()) if render_control else None
        report["capture_overlay_isolation"] = bool(render_control and report["render_before"]["hooks_mask"] & HOOK_CAPTURE_OVERLAY)
        for n, policy in enumerate(args.policies.split(",")):
            if policy not in ("idle", "track"): raise ValueError(policy)
            obs, info = env.reset(seed=0)
            entry = {"episode": n, "policy": policy, "initial_info": info}
            report["episodes"].append(entry)
            print("RESET", n, policy, flush=True)
            with (args.out / f"episode{n}.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({"obs": env.raw_obs, "info": info}) + "\n")
                while True:
                    action = np.zeros(4, dtype=np.int64) if policy == "idle" else tracking_action(obs)
                    obs, reward, terminal, truncated, info = env.step(action)
                    f.write(json.dumps({"action": action.tolist(), "obs": env.raw_obs, "reward": reward,
                                        "terminated": terminal, "truncated": truncated, "info": info}) + "\n")
                    if terminal or truncated: break
            entry.update(outcome=info["outcome"], frames=info["elapsed_frames"], reward=reward,
                         hearts=env.raw_obs["players"][0]["hearts"], final_info=info)
            print("END", n, entry["outcome"], entry["frames"], "reward", reward, flush=True)
        report["status"] = "rollouts_completed"
    except (OSError, BridgeError) as error:
        report.update(status="worker_error", error={"type": type(error).__name__, "message": str(error)})
        raise
    except TurboError as error:
        report.update(status="render_control_error", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        try:
            env.close()
            report["cleanup"] = env.cleanup_outcome
            if report["status"] == "rollouts_completed": report["status"] = "completed"
        finally:
            report["process_exit"] = process_exit_code(pid) if args.launch else None
            if render_control:
                report["render_after"] = asdict(render_control.stats())
                render_control.close()
            report["wall_seconds"] = time.perf_counter() - start
            (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
