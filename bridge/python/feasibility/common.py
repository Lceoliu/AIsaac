"""L1 可行性实验共用：会话（启动或附着）、观测摘要、轨迹比较、报告落盘。

所有脚本共用同一组命令行参数（add_session_args）：
  --launch            由脚本启动 isaac-ng.exe --luadebug 并注入 isaac_turbo.dll（默认直通模式）
  --pid N [--inject]  附着到已运行的游戏；--inject 表示尚未注入，先注入
  --port              桥接 mod 监听端口（ISAAC_RL_PORT）
  --seed-cmd          重置时在 restart 之后执行的种子命令，例如 "seed 4JH8 P2N9"
  --room              重置时最后执行的房间命令，例如 "goto d.12"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # rl/bridge/python
from isaac_bridge import Action, IsaacBridgeEnv  # noqa: E402
from isaac_bridge.launch import DEFAULT_GAME_DIR  # noqa: E402
from isaac_bridge.turbo import TurboControl, TurboError, attach, launch_turbo  # noqa: E402

RL_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_RUNS = os.path.join(RL_ROOT, "runs", "l1")
NOOP = Action()


def add_session_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--port", type=int, default=27015)
    p.add_argument("--pid", type=int, default=0, help="附着到已运行的 isaac-ng.exe")
    p.add_argument("--inject", action="store_true", help="附着时先注入 isaac_turbo.dll")
    p.add_argument("--launch", action="store_true", help="由脚本启动游戏并注入")
    p.add_argument("--game-dir", default=DEFAULT_GAME_DIR)
    p.add_argument("--log-dir", default=None, help="DLL 日志目录（默认 %%LOCALAPPDATA%%\\IsaacRL\\turbo）")
    p.add_argument("--seed-cmd", default="auto",
                   help='"auto"（默认）= 第一次 restart 后读取游戏生成的起始种子并在之后每次重置固定使用；'
                        '也可给定 "seed XXXX XXXX"（必须是游戏认可的带校验的种子串）；空串 = 不固定种子')
    p.add_argument("--room", default="goto d.12", help="重置后进入的房间命令；为空则留在起始房")
    p.add_argument("--settle", type=int, default=2)
    p.add_argument("--repeat", type=int, default=4)
    p.add_argument("--out", default="", help="输出目录；默认 rl/runs/l1/<时间戳>-<阶段>")
    p.add_argument("--keep", action="store_true", help="结束时不关闭由脚本启动的游戏")


def out_dir(args: argparse.Namespace, stage: str) -> str:
    path = args.out or os.path.join(DEFAULT_RUNS, time.strftime("%Y%m%d-%H%M%S") + "-" + stage)
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class Session:
    """一个游戏实例 = 一个桥接连接 + （可选）一个加速控制块。"""

    def __init__(self, args: argparse.Namespace, port: Optional[int] = None):
        self.args = args
        self.port = port or args.port
        self.proc = None
        self.ctl: Optional[TurboControl] = None
        if args.launch:
            self.proc, self.ctl = launch_turbo(self.port, args.game_dir, privileged=True, log_dir=args.log_dir)
            print(f"[session] launched pid={self.proc.pid} port={self.port} turbo={self.ctl.stats().status_name}")
        elif args.pid:
            self.ctl = attach(args.pid, do_inject=args.inject)
            print(f"[session] attached pid={args.pid} turbo={self.ctl.stats().status_name}")
        self.env = IsaacBridgeEnv(port=self.port, action_repeat=args.repeat)
        self.hello = self.env.connect()
        print(f"[session] bridge hello: {self.hello}")
        self.seed_cmd: str = args.seed_cmd
        if self.seed_cmd == "auto":
            self.seed_cmd = self._resolve_seed()

    def _resolve_seed(self) -> str:
        """restart 一次，读取游戏自己生成的起始种子串（带校验，保证 `seed` 命令接受它）。"""
        self.env.reset(phases=[["restart"]], settle=self.args.settle)
        info = self.env.query_info()
        seed = info.get("start_seed")
        if not seed:
            print("[session] WARN start_seed unavailable; running without a fixed seed")
            return ""
        cmd = f"seed {seed}"
        print(f"[session] fixed seed for all resets: {cmd}")
        return cmd

    # ------------------------------------------------------------------ 控制
    def set_turbo(self, on: bool, render_every: int = 0) -> bool:
        if not self.ctl:
            return False
        self.ctl.set_turbo(on)
        self.ctl.set_render_every(render_every if on else 0)
        return True

    def shm(self) -> Optional[Dict[str, Any]]:
        return self.ctl.stats().to_dict() if self.ctl else None

    # ------------------------------------------------------------------ 环境
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        phases: List[List[str]] = [["restart"]]
        if self.seed_cmd:
            phases.append([self.seed_cmd])
        if self.args.room:
            phases.append([self.args.room])
        return self.env.reset(phases=phases, settle=self.args.settle)

    def step(self, action: Action, repeat: Optional[int] = None):
        return self.env.step(action, repeat=repeat)

    def obs(self) -> Dict[str, Any]:
        return self.env.query_obs()

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass
        if self.ctl:
            try:
                self.ctl.set_turbo(False)
            except Exception:  # noqa: BLE001
                pass
        if self.proc and not self.args.keep:
            self.proc.terminate()


# ---------------------------------------------------------------------- 摘要与比较

def digest(obs: Dict[str, Any]) -> Dict[str, Any]:
    players = obs.get("players") or []
    p = players[0] if players else {}
    pos = p.get("pos") or [0.0, 0.0]
    ents = sorted(obs.get("entities") or [], key=lambda e: (e.get("id", 0), e.get("type", 0)))
    room = obs.get("room") or {}
    return {
        "game_frame": obs.get("game_frame"),
        "room_frame": room.get("frame"),
        "logic_frames": obs.get("logic_frames"),
        "player_pos": [round(pos[0], 3), round(pos[1], 3)],
        "hearts": p.get("hearts"), "soul": p.get("soul"), "invulnerable": p.get("invulnerable"),
        "entities": [[e.get("id"), e.get("type"), e.get("variant"), round(e["pos"][0], 3), round(e["pos"][1], 3),
                      round(float(e.get("height") or 0.0), 3)] for e in ents if e.get("pos")],
        "events": dict(obs.get("events") or {}),
        "clear": room.get("clear"), "alive": room.get("alive"),
    }


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compare_trajectories(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """逐步比较两条摘要轨迹：任一字段首次不同的步、玩家位置首次不同的步、最大玩家位置差、事件总数。"""
    n = min(len(a), len(b))
    first_any = None
    first_player = None
    max_player = 0.0
    entity_mismatch_steps = 0
    for i in range(n):
        da, db = a[i], b[i]
        d = _dist(da["player_pos"], db["player_pos"])
        max_player = max(max_player, d)
        if first_player is None and d > 1e-6:
            first_player = i
        if da["entities"] != db["entities"]:
            entity_mismatch_steps += 1
        if first_any is None and (d > 1e-6 or da["entities"] != db["entities"] or da["hearts"] != db["hearts"]
                                  or da["events"] != db["events"]):
            first_any = i
    return {
        "steps_compared": n, "len_a": len(a), "len_b": len(b),
        "first_divergence_step": first_any, "first_player_divergence_step": first_player,
        "max_player_pos_diff": round(max_player, 4), "entity_mismatch_steps": entity_mismatch_steps,
        "events_a": a[-1]["events"] if a else {}, "events_b": b[-1]["events"] if b else {},
        "identical": first_any is None and len(a) == len(b),
    }


# ---------------------------------------------------------------------- 策略

def scripted_action(t: int) -> Action:
    """确定性脚本：每 40 步一轮，四个方向各 10 步，边走边朝反方向射击。"""
    phase = (t % 40) // 10
    return [Action(move=3, shoot=4), Action(move=1, shoot=3), Action(move=7, shoot=2), Action(move=5, shoot=1)][phase]


class SeededRandomPolicy:
    def __init__(self, seed: int):
        import random
        self.rng = random.Random(seed)

    def __call__(self, t: int) -> Action:
        return Action(move=self.rng.randrange(9), shoot=self.rng.randrange(5))


def make_policy(name: str, seed: int = 0):
    return scripted_action if name == "scripted" else SeededRandomPolicy(seed)


def fmt_row(name: str, ok: Optional[bool], detail: str) -> str:
    mark = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
    return f"{mark:5s} {name:34s} {detail}"


__all__ = ["Action", "IsaacBridgeEnv", "NOOP", "Session", "TurboError", "add_session_args", "compare_trajectories",
           "digest", "fmt_row", "make_policy", "out_dir", "scripted_action", "write_json"]
