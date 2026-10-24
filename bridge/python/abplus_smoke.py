"""AB+ bridge smoke test: one instance, simulator arena seeds, random legal actions.

Checks the protocol end to end (connect, arena reset, 15 Hz steps, termination), that every
observation encodes through the policy's VisibleHistory (combat-v1 deadline schema), which
Monstro animation names the AB+ probe reports, and the decision rate.

usage: python abplus_smoke.py [--mode exact|skip|render] [--episodes 6] [--port 27100]
"""
import argparse
import collections
import json
import time

import numpy as np

from isaac_bridge.abplus import AbplusTransformerEnv, launch_abplus, sim_arena, stop_abplus


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="exact", choices=("exact", "skip", "render"))
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--port", type=int, default=27100)
    p.add_argument("--seed", type=int, default=2**31 + 5000)
    p.add_argument("--name", default="bsmoke")
    args = p.parse_args()
    proc = launch_abplus(args.name, args.port, args.mode)
    env = AbplusTransformerEnv(port=args.port)
    # Time split: bridge round trip (Lua observation + JSON + socket) vs Python-side encoding.
    timing = {"bridge_s": 0.0, "bridge_calls": 0, "json_bytes": 0}
    raw_step, raw_recv = env.bridge.step, env.bridge._recv

    def timed_step(*a, **k):
        t = time.perf_counter()
        try:
            return raw_step(*a, **k)
        finally:
            timing["bridge_s"] += time.perf_counter() - t
            timing["bridge_calls"] += 1

    def sized_recv():
        before = len(env.bridge._buf)
        msg = raw_recv()
        timing["json_bytes"] += max(0, len(json.dumps(msg)) if msg.get("type") == "obs" else 0)
        return msg

    env.bridge.step, env.bridge._recv = timed_step, sized_recv
    rng = np.random.default_rng(0)
    anims, outcomes, results = collections.Counter(), collections.Counter(), []
    steps_total, t_start = 0, None
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            obs, info = env.reset(options={"arena_seed": seed})
            if t_start is None:
                t_start = time.monotonic()
                raw = env.raw_obs
                first = {"player": {k: raw["players"][0][k] for k in ("pos", "hearts", "range", "damage", "speed",
                                                                        "fire_delay_max", "active", "bombs")},
                         "room": {k: raw["room"].get(k) for k in ("type", "variant", "top_left", "bottom_right", "clear")},
                         "doors": raw["doors"], "entities": [(e["type"], e["variant"], e["pos"], e.get("anim"))
                                                            for e in raw["entities"]],
                         "arena": info.get("arena"), "arena_setup": info.get("arena_setup"),
                         "obs_keys": sorted(obs.keys())}
                print("first reset:", json.dumps(first))
            done, steps, ep_t0 = False, 0, time.monotonic()
            while not done:
                joint = int(rng.integers(45))
                bomb = int(rng.random() < 0.02 and env.action_masks()[46])
                obs, reward, terminated, truncated, info = env.step(np.array([joint, bomb, 0]))
                for e in env.raw_obs["entities"]:
                    if e["type"] == 20:
                        anims[e.get("anim")] += 1
                steps += 1
                done = terminated or truncated
            steps_total += steps
            outcomes[info["outcome"]] += 1
            arena = sim_arena(seed)
            results.append({"seed": seed, "variant": arena["variant"], "outcome": info["outcome"],
                            "frames": info["elapsed_frames"], "steps": steps,
                            "seconds": round(time.monotonic() - ep_t0, 2)})
            print(json.dumps(results[-1]))
        elapsed = time.monotonic() - t_start
        calls = max(1, timing["bridge_calls"])
        print(json.dumps({"mode": args.mode, "episodes": args.episodes, "decisions": steps_total,
                          "seconds": round(elapsed, 1), "decisions_per_s": round(steps_total / elapsed, 1),
                          "bridge_ms_per_step": round(1000 * timing["bridge_s"] / calls, 3),
                          "python_ms_per_step": round(1000 * (elapsed - timing["bridge_s"]) / max(1, steps_total), 3),
                          "obs_json_kb": round(timing["json_bytes"] / 1024 / calls, 1),
                          "outcomes": outcomes, "monstro_animations": anims}))
    finally:
        try:
            env.close()
        finally:
            stop_abplus(proc, args.name)


if __name__ == "__main__":
    main()
