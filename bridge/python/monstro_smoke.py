"""Validate the formal original-engine Monstro arena, without Turbo injection."""
import argparse
import dataclasses
import json
from pathlib import Path
import time

from feasibility.common import digest, scripted_action
from isaac_bridge.launch import launch
from isaac_bridge.training import IsaacTrainingEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--port", type=int, default=27015)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", default="9AM0 7PRP")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    proc = launch(args.port, privileged=True) if args.launch else None
    print("PID", proc.pid if proc else "attached", "port", args.port, flush=True)
    env = IsaacTrainingEnv(port=args.port, connect_timeout=240, recv_timeout=15)
    report = {"args": {**vars(args), "out": str(args.out)},
              "pid": proc.pid if proc else None, "episodes": [], "turbo": False}
    try:
        report["hello"] = env.connect()
        for n in range(args.episodes):
            obs, info = env.reset_monstro(args.seed)
            entry = {"episode": n, "initial_obs": obs, "initial_info": info,
                     "trajectory": [digest(obs)], "terminal": False}
            report["episodes"].append(entry)
            print("READY", n, "room", obs["room"]["room_idx"], "alive", obs["room"]["alive"],
                  "player", obs["players"][0]["pos"], "NPCs", info.get("npcs"), flush=True)
            start = time.perf_counter()
            for step in range(args.steps):
                before = obs["logic_frames"]
                obs, _, terminated, _, _ = env.step(dataclasses.asdict(scripted_action(step)), repeat=4)
                if obs["logic_frames"] - before != 4:
                    raise RuntimeError("A four-frame action did not advance exactly four logic frames")
                entry["trajectory"].append(digest(obs))
                if terminated:
                    entry["terminal"] = True
                    break
            entry["seconds"] = time.perf_counter() - start
            entry["frames"] = entry["trajectory"][-1]["logic_frames"] - entry["trajectory"][0]["logic_frames"]
            (args.out / f"episode{n}.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
            print("DONE", n, "frames", entry["frames"], "terminal", entry["terminal"],
                  "hearts", entry["trajectory"][-1]["hearts"], flush=True)
        obs, info = env.reset_safe(args.seed)
        report["safe_end"] = {"obs": obs, "info": info}
        report["status"] = "completed"
        print("SAFE_END enemy-free starting room; game process retained", flush=True)
    finally:
        env.close()
        report["process_exit_at_finish"] = proc.poll() if proc else None
        (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
