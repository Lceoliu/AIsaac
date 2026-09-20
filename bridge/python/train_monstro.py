"""Bounded SB3 PPO wiring test on the original engine, not a convergence claim."""
import argparse
import json
from pathlib import Path
import time
from dataclasses import asdict

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from isaac_bridge.monstro_gym import MonstroGymEnv, OBSERVATION_SCHEMA
from isaac_bridge.launch import launch
from isaac_bridge.env import BridgeError
from isaac_bridge.rendering import configure_rendering
from isaac_bridge.turbo import TurboError


class EpisodeLog(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodes = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                entry = {**info["episode"], "outcome": info["outcome"], "frames": info["elapsed_frames"]}
                self.episodes.append(entry)
                print("EPISODE", json.dumps(entry), flush=True)
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=27015)
    p.add_argument("--launch", action="store_true")
    p.add_argument("--pid", type=int, help="existing worker PID when attaching")
    p.add_argument("--render-mode", choices=("headless", "visible"), default="headless")
    p.add_argument("--connect-timeout", type=float, default=30.)
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--episode-frames", type=int, default=960)
    p.add_argument("--hit-reward", type=float, default=0.05)
    p.add_argument("--damage-reward", type=float, default=1.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.launch == (args.pid is not None):
        p.error("specify exactly one of --launch or --pid")
    args.out.mkdir(parents=True, exist_ok=True)
    proc = launch(args.port, privileged=True, extra_args=("--luadebug", "--set-stage=1")) if args.launch else None
    pid = proc.pid if proc else args.pid
    print("PID", pid, "render_mode", args.render_mode, flush=True)
    torch.set_num_threads(1)
    env = Monitor(MonstroGymEnv(port=args.port, max_episode_frames=args.episode_frames,
                              hit_reward=args.hit_reward, damage_reward=args.damage_reward),
                  filename=str(args.out / "monitor.csv"))
    env.unwrapped.bridge.connect_timeout = args.connect_timeout
    callback = EpisodeLog()
    report = {"status": "started", "requested_steps": args.steps, "turbo": False,
              "pid": pid, "render_mode": args.render_mode, "virtual_clock": False,
              "observation_schema": OBSERVATION_SCHEMA,
              "reward": {"hurt": -1, "hit": args.hit_reward, "normalized_damage": args.damage_reward,
                         "clear": 1, "extra_death": 0, "time_limit": "truncation"}}
    start = time.perf_counter()
    render_control = None
    try:
        env.unwrapped.bridge.connect()
        env.unwrapped.connected = True
        render_control = configure_rendering(pid, args.render_mode)
        report["render_before"] = asdict(render_control.stats()) if render_control else None
        model = PPO("MultiInputPolicy", env, n_steps=128, batch_size=64, n_epochs=4,
                    learning_rate=3e-4, seed=0, device="cpu", verbose=1,
                    policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])})
        initial = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
        model.learn(total_timesteps=args.steps, callback=callback)
        delta = sum(float(torch.sum((v - initial[k]) ** 2)) for k, v in model.policy.state_dict().items()) ** .5
        if delta == 0: raise RuntimeError("PPO did not change policy/value parameters")
        model.save(args.out / "ppo_monstro")
        restored = PPO.load(args.out / "ppo_monstro", device="cpu")
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        restored_action, _ = restored.predict(obs, deterministic=True)
        np.testing.assert_array_equal(action, restored_action)
        _, reward, terminated, truncated, info = env.step(restored_action)
        report.update(status="completed", actual_steps=model.num_timesteps,
                      training_epochs=model._n_updates, parameter_l2_change=delta,
                      episodes=callback.episodes, checkpoint=str(args.out / "ppo_monstro.zip"),
                      reload_action=restored_action.tolist(), reload_step_outcome=info["outcome"],
                      reload_step_reward=reward)
    except (OSError, BridgeError) as error:
        report.update(status="worker_error", error={"type": type(error).__name__, "message": str(error)},
                      episodes=callback.episodes)
        raise
    except TurboError as error:
        report.update(status="render_control_error", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        try:
            env.close()
            report["cleanup"] = env.unwrapped.cleanup_outcome
        finally:
            report["process_exit"] = proc.poll() if proc else None
            if render_control:
                report["render_after"] = asdict(render_control.stats())
                render_control.close()
            report["wall_seconds"] = time.perf_counter() - start
            (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
