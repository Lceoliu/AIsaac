"""Python client for the IsaacRLBridge mod (rl/bridge/mod/isaac_rl_bridge).

Nothing here launches or modifies the game installation by itself; see launch.py
for starting isaac-ng.exe with --luadebug and a per-instance port.
"""
import sys
from .env import IsaacBridgeEnv, BridgeError, Action, TrajectoryRecorder

__all__ = ["IsaacBridgeEnv", "BridgeError", "Action", "TrajectoryRecorder"]
# The headless simulator and TCP client are portable; injection is Windows-only.
if sys.platform == "win32":
    from .turbo import TurboControl, TurboError, TurboStats, attach, inject, launch_turbo
    __all__ += ["TurboControl", "TurboError", "TurboStats", "attach", "inject", "launch_turbo"]
