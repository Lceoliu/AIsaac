"""Start one isaac-ng.exe instance for the bridge.

Requirements (all outside this repo, none are changed here):
  * Steam is running and logged in (the exe only imports SteamInternal_SteamAPI_Init).
  * The mod folder rl/bridge/mod/isaac_rl_bridge is present under <game>/mods and enabled in the
    in-game Mods menu (EnableMods=1 in options.ini of the Repentance+ profile).
  * --luadebug is passed so the mod can require("socket").
  * For background instances set PauseOnFocusLost=0 in options.ini (user profile, not install dir).
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional, Sequence

DEFAULT_GAME_DIR = r"D:\Steam\steamapps\common\The Binding of Isaac Rebirth"


def launch(port: int, game_dir: str = DEFAULT_GAME_DIR, privileged: bool = False,
           extra_args: Sequence[str] = ("--luadebug",), engine_velocity: bool = False,
           extra_env: Optional[dict] = None) -> subprocess.Popen:
    exe = os.path.join(game_dir, "isaac-ng.exe")
    if not os.path.isfile(exe):
        raise FileNotFoundError(exe)
    env = dict(os.environ)
    env["ISAAC_RL_PORT"] = str(port)
    env["ISAAC_RL_PRIVILEGED"] = "1" if privileged else "0"
    env["ISAAC_RL_ENGINE_VEL"] = "1" if engine_velocity else "0"
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    return subprocess.Popen([exe, *extra_args], cwd=game_dir, env=env)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="launch isaac-ng.exe with the RL bridge port")
    p.add_argument("--port", type=int, default=27015)
    p.add_argument("--game-dir", default=DEFAULT_GAME_DIR)
    p.add_argument("--privileged", action="store_true")
    args = p.parse_args(argv)
    proc = launch(args.port, args.game_dir, args.privileged)
    print(f"started pid={proc.pid} port={args.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
