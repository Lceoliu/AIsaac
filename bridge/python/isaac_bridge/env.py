"""Synchronous environment wrapper over the IsaacRLBridge TCP protocol.

The game blocks inside MC_POST_UPDATE until we answer, so every ``step`` here is
exactly ``repeat`` game logic frames (30 Hz frames). The wrapper is deliberately
framework-free: it returns plain dicts and follows the Gymnasium 5-tuple shape so
a thin adapter can turn it into a gymnasium.Env later.

Observation policy: everything in ``obs`` is what the mod's whitelist considers
player-visible. Privileged truth (enemy HP, AI state, RNG) only arrives through
``info`` when the mod runs with ISAAC_RL_PRIVILEGED=1 or via ``query_info()`` and
must never be fed to the actor.
"""
from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


class BridgeError(RuntimeError):
    pass


@dataclass
class Action:
    move: int = 0      # 0 stop, 1 up, 2 up-right, 3 right, 4 down-right, 5 down, 6 down-left, 7 left, 8 up-left
    shoot: int = 0     # 0 none, 1 up, 2 right, 3 down, 4 left
    bomb: int = 0
    item: int = 0
    pill: int = 0
    drop: int = 0

    @classmethod
    def from_any(cls, a: Any) -> "Action":
        if isinstance(a, Action):
            return a
        if isinstance(a, dict):
            return cls(**{k: int(v) for k, v in a.items() if k in cls.__dataclass_fields__})
        seq = list(a)
        return cls(*[int(x) for x in seq[:6]])


class IsaacBridgeEnv:
    """One game instance == one env. Connect after the game has loaded the mod."""

    def __init__(self, port: int = 27015, host: str = "127.0.0.1", action_repeat: int = 4,
                 connect_timeout: float = 600.0, recv_timeout: Optional[float] = 120.0):
        self.host, self.port = host, port
        self.action_repeat = action_repeat
        self.connect_timeout = connect_timeout
        self.recv_timeout = recv_timeout
        self._sock: Optional[socket.socket] = None
        self._buf = b""
        self.last_obs: Optional[Dict[str, Any]] = None
        self.last_info: Dict[str, Any] = {}
        self.hello: Dict[str, Any] = {}

    # ------------------------------------------------------------------ transport
    def connect(self) -> Dict[str, Any]:
        deadline = time.monotonic() + self.connect_timeout
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                s = socket.create_connection((self.host, self.port), timeout=5.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(self.recv_timeout)
                self._sock = s
                self.hello = self._recv()
                if self.hello.get("type") != "hello":
                    raise BridgeError(f"unexpected first message: {self.hello}")
                return self.hello
            except (OSError, BridgeError) as exc:  # game not listening yet
                last_err = exc
                if self._sock:
                    self._sock.close()
                    self._sock = None
                time.sleep(1.0)
        raise BridgeError(f"could not connect to {self.host}:{self.port}: {last_err}")

    def close(self) -> None:
        if self._sock:
            try:
                self._send({"cmd": "close"})
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def _send(self, msg: Dict[str, Any]) -> None:
        if not self._sock:
            raise BridgeError("not connected")
        self._sock.sendall((json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8"))

    def _recv(self) -> Dict[str, Any]:
        if not self._sock:
            raise BridgeError("not connected")
        while b"\n" not in self._buf:
            chunk = self._sock.recv(1 << 16)
            if not chunk:
                raise BridgeError("connection closed by game")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        msg = json.loads(line.decode("utf-8"))
        if msg.get("type") == "error":
            raise BridgeError(msg.get("msg", "error"))
        return msg

    def _expect_obs(self) -> Dict[str, Any]:
        msg = self._recv()
        while msg.get("type") != "obs":  # skip stray acks
            msg = self._recv()
        self.last_obs = msg["obs"]
        self.last_info = msg.get("info", {})
        self.last_info["event"] = msg.get("event")
        self.last_info["seq"] = msg.get("seq")
        return self.last_obs

    # ------------------------------------------------------------------ env API
    def reset(self, phases: Sequence[Sequence[str]] = (("restart",),), settle: int = 2) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run console-command phases; each phase waits for MC_POST_NEW_ROOM before the next.

        Typical single-room curriculum: (("restart",), ("goto d.12",)). Multi-phase is needed
        because ``restart`` only takes effect on the next game update.
        """
        obs = None
        for cmds in phases:
            self._send({"cmd": "reset", "commands": list(cmds), "settle": int(settle), "wait_room": True})
            obs = self._expect_obs()
        if obs is None:
            self._send({"cmd": "obs"})
            obs = self._expect_obs()
        return obs, dict(self.last_info)

    def step(self, action: Any, repeat: Optional[int] = None) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        a = Action.from_any(action)
        msg = {"cmd": "step", "repeat": int(repeat or self.action_repeat)}
        msg.update(asdict(a))
        self._send(msg)
        obs = self._expect_obs()
        terminated = self.is_terminal(obs)
        reward = 0.0  # reward shaping lives in the trainer (needs privileged info or obs deltas)
        return obs, reward, terminated, False, dict(self.last_info)

    def exec(self, command: str) -> None:
        self._send({"cmd": "exec", "command": command})
        msg = self._recv()
        if msg.get("type") != "ok":
            raise BridgeError(f"exec failed: {msg}")

    def query_info(self) -> Dict[str, Any]:
        self._send({"cmd": "info"})
        msg = self._recv()
        return msg.get("info", {})

    def query_obs(self) -> Dict[str, Any]:
        """Re-send the current observation without advancing the game (event == "query")."""
        self._send({"cmd": "obs"})
        return self._expect_obs()

    def set_control(self, enabled: bool) -> None:
        self._send({"cmd": "control", "enabled": bool(enabled)})
        self._recv()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def is_terminal(obs: Dict[str, Any]) -> bool:
        players = obs.get("players") or []
        if players and all(p.get("dead") for p in players):
            return True
        room = obs.get("room") or {}
        return bool(room.get("clear"))

    @staticmethod
    def visible_enemies(obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [e for e in obs.get("entities", []) if e.get("enemy")]


class TrajectoryRecorder:
    """JSONL recorder for calibration data: one line per decision boundary."""

    def __init__(self, path: str):
        self._f = open(path, "a", encoding="utf-8")

    def write(self, obs: Dict[str, Any], action: Optional[Dict[str, Any]], info: Dict[str, Any],
              meta: Optional[Dict[str, Any]] = None) -> None:
        rec = {"t": time.time(), "obs": obs, "action": action, "info": info}
        if meta:
            rec["meta"] = meta
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._f.close()
