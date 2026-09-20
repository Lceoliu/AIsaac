"""Stage-0 probe: connect to a running game with the bridge mod, reset into a room, take random
actions, and record observations + privileged info to JSONL. Exercises every protocol command.

    python smoke_test.py --port 27015 --room "goto d.12" --steps 200 --out runs/probe.jsonl
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaac_bridge import IsaacBridgeEnv, Action, TrajectoryRecorder  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=27015)
    p.add_argument("--room", default="goto d.12", help="console command run after restart")
    p.add_argument("--seed", default="", help="optional 'seed XXXX XXXX' command run after restart")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--repeat", type=int, default=4)
    p.add_argument("--out", default="runs/probe.jsonl")
    args = p.parse_args()

    env = IsaacBridgeEnv(port=args.port, action_repeat=args.repeat)
    hello = env.connect()
    print("hello:", hello)
    phases = [["restart"]]
    if args.seed:
        phases.append([args.seed])
    phases.append([args.room])
    t0 = time.perf_counter()
    obs, info = env.reset(phases=phases)
    print(f"reset ok in {time.perf_counter() - t0:.2f}s; room={obs['room']}; "
          f"entities={len(obs['entities'])}; grid={len(obs['grid'])}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rec = TrajectoryRecorder(args.out)
    rec.write(obs, None, info, meta={"event": "reset", "hello": hello})
    rng = random.Random(0)
    t0 = time.perf_counter()
    n = 0
    for _ in range(args.steps):
        a = Action(move=rng.randrange(9), shoot=rng.randrange(5))
        obs, reward, term, trunc, info = env.step(a)
        truth = env.query_info()
        rec.write(obs, a.__dict__, {**info, "truth": truth})
        n += 1
        if term:
            outcome = "dead" if all(pl["dead"] for pl in obs["players"]) else "clear"
            print("terminal:", outcome, "at step", n)
            break
    dt = time.perf_counter() - t0
    print(f"{n} steps, {n * args.repeat} game frames, {n / dt:.1f} steps/s, {n * args.repeat / dt:.1f} frames/s")
    rec.close()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
