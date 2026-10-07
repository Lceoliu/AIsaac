"""Native lean observation check: is abp_turbo's native observation (abp_bridge.lua `abp_native_lean`, mode 1) the same
bytes as the Lua code's (mode 0), and is the game unchanged by it?

Per seed: reset (room mode: a room of the group, `warm` decisions played; floor mode: a whole floor's start room), then
three lean clones of that state:
  lua     native_obs mode 0: the Lua code only (the reference);
  native  mode 1: the native code (the Lua code only where the native one hands over: a laser, a cancelled lethal hit);
  check   mode 2: both in the same process on the same state, the Lua result is sent, the bridge counts differences
          of the fixed part and of the terrain check (ABP_NATIVE_OBS).
All three get the same actions; at every decision the raw payloads of the three must be equal byte for byte, and every
`check` decisions and at the end the hidden-state digests (abplus_goexplore.DIGEST_LUA) of the three must be equal:
the native observation changes what the bridge reads, not the game.
Room mode: sticky random actions (abplus_probe_fork.sticky_actions) until the player dies or `--steps`, going on for
`--after-clear` decisions after a clear (the drops). Floor mode: legs: walk to a random open door of the room (as
abplus_probe_floor_lean.py), then sticky random actions for `--steps` decisions in the new room (or until it is clear and
`--after-clear` more), repeated `--legs` times: room changes, map and door blocks, terrain resends, other rooms' enemies.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_native_obs.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:32 --out <dir>
  python abplus_probe_native_obs.py --groups-file ../abplus/catalog/scaling2_groups.json --floor --seeds range:2147600000:4 \
      --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from abplus_probe_fork import digest, split_code, sticky_actions
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder

MOVES = {(0, 0): 0, (0, -1): 1, (1, -1): 2, (1, 0): 3, (1, 1): 4, (0, 1): 5, (-1, 1): 6, (-1, 0): 7, (-1, -1): 8}
MODES = (('lua', 0), ('native', 1), ('check', 2))


def read_raw(bridge):
    """The next lean payload of a clone (raw bytes)."""
    while True:
        line = bridge._read_line()
        if line.startswith(b'L '):
            return bridge._read_exact(int(line.split(b' ')[1]))
        msg = json.loads(line.decode('utf-8'))
        if msg.get('type') == 'error':
            raise RuntimeError(msg.get('msg', 'error'))


def native_obs(bridge, mode=None):
    bridge._send({"cmd": "native_obs"} if mode is None else {"cmd": "native_obs", "mode": mode})
    msg = bridge._recv()
    if msg.get('type') != 'ok' or msg.get('cmd') != 'native_obs':
        raise RuntimeError(f'native_obs failed: {msg}')
    return msg


def first_diff(a, b):
    n = min(len(a), len(b))
    at = next((i for i in range(n) if a[i] != b[i]), n)
    return dict(at=at, len=(len(a), len(b)), a=a[at:at + 24].hex(), b=b[at:at + 24].hex())


class Trio:
    """The three clones, stepped together; compares their payloads at every decision."""

    def __init__(self, parent, check_every):
        self.clones = {}
        for name, mode in MODES:
            c = parent.fork(tag=name, lean=True, alarm=600)
            native_obs(c, mode)
            self.clones[name] = c
        self.decoder = LeanDecoder()
        self.check_every = check_every
        self.decisions = 0
        self.payloads = 0
        self.mismatch = None
        self.digests_equal = True
        self.digest_checks = 0
        self.lasers = 0
        self.obs = self._collect({"cmd": "obs"})

    def _collect(self, cmd):
        for c in self.clones.values():
            c._send(cmd)
        raw = {k: read_raw(c) for k, c in self.clones.items()}
        self.payloads += 1
        if self.mismatch is None:
            for k in ('native', 'check'):
                if raw[k] != raw['lua']:
                    self.mismatch = dict(decision=self.decisions, clone=k, **first_diff(raw['lua'], raw[k]))
                    break
        o = self.decoder.decode(raw['lua'])
        self.lasers += len(o.lasers) > 0
        return o

    def step(self, action, repeat):
        self.decisions += 1
        self.obs = self._collect({"cmd": "step", "repeat": repeat, "move": int(action[0]), "shoot": int(action[1]),
                                  "bomb": int(action[2]), "item": int(action[3])})
        if self.decisions % self.check_every == 0:
            self.compare_digests()
        return self.obs

    def lua(self, code):
        for c in self.clones.values():
            if c.lua(code) != 'ok':
                raise RuntimeError('lua command failed')

    def compare_digests(self):
        # all three (the digest's lua command makes the next observation carry the terrain: same for all)
        d = {k: digest(c)[0] for k, c in self.clones.items()}
        self.digest_checks += 1
        self.digests_equal = self.digests_equal and d['lua'] == d['native'] == d['check']

    def finish(self):
        self.compare_digests()
        stats = {k: native_obs(c) for k, c in self.clones.items()}
        for c in self.clones.values():
            c.close()
        return dict(decisions=self.decisions, payloads=self.payloads, payload_mismatch=self.mismatch,
                    digests_equal=self.digests_equal, digest_checks=self.digest_checks, laser_obs=self.lasers,
                    native_calls=stats['native']['native'], native_fallbacks=stats['native']['fallbacks'],
                    last_fallback=stats['native'].get('last_fallback'),
                    check_checks=stats['check']['checks'], check_mismatches=stats['check']['mismatches'],
                    check_first=stats['check'].get('first'), available=stats['native']['available'],
                    turbo=stats['native'].get('turbo'))


def room_seed(inst, seed, args, rep):
    rng = np.random.default_rng(seed)
    warm, test = sticky_actions(rng, args.warm), sticky_actions(rng, args.steps)
    inst.reset(seed)
    _, _, stop = inst.env.bridge.play(warm, repeat=rep, stop_clear=True)
    if stop != 'done':
        return dict(seed=seed, skipped='episode ended in the warm-up')
    trio = Trio(inst.env.bridge, args.check)
    after = None
    for code in test:
        o = trio.step(split_code(code), rep)
        if o.dead:
            break
        if o.clear and after is None:
            after = args.after_clear
        if after is not None:
            after -= 1
            if after < 0:
                break
    return dict(seed=seed, entities_last=len(trio.obs.entities), **trio.finish())


def move_to(px, py, x, y):
    sx = 0 if abs(x - px) < 6 else (1 if x > px else -1)
    sy = 0 if abs(y - py) < 6 else (1 if y > py else -1)
    return MOVES[(sx, sy)]


def floor_seed(inst, seed, args, rep):
    inst.env.bridge.reset_mode = 'floor'
    inst.reset(seed)
    rng = np.random.default_rng(seed)
    trio = Trio(inst.env.bridge, args.check)
    legs, rooms, kills = 0, set(), 0
    o = trio.obs
    for _ in range(args.legs):
        doors = [d for d in o.doors if d['open'] and not d['locked']]
        if not doors or o.dead:
            break
        d = doors[int(rng.integers(len(doors)))]
        here = o.room[4]
        for _ in range(150):
            pl = o.players[0]
            o = trio.step((move_to(pl['x'], pl['y'], float(d['x']), float(d['y'])), 0, 0, 0), rep)
            if o.room[4] != here or o.dead:
                break
        if o.room[4] == here or o.dead:
            break
        legs += 1
        rooms.add(o.room[4])
        after = None
        for code in sticky_actions(rng, args.steps):
            move, _, bomb, item = split_code(code)
            o = trio.step((move, aim(o), bomb, item), rep)
            if o.dead:
                break
            if o.clear and after is None:
                after = args.after_clear
            if after is not None:
                after -= 1
                if after < 0:
                    break
        if not o.clear and not o.dead:
            # what is left of the room's enemies dies (the same lua command to all three clones), so the walk goes on
            trio.lua(KILL_LUA)
            kills += 1
            for _ in range(args.after_clear):
                o = trio.step((0, 0, 0, 0), rep)
    return dict(seed=seed, legs=legs, rooms=len(rooms), kill_commands=kills, **trio.finish())


KILL_LUA = ("for _, e in ipairs(Isaac.GetRoomEntities()) do "
            "if e:IsActiveEnemy(false) then e:Kill() end end return 'ok'")


def aim(o):
    """Shoot along the main axis towards the nearest enemy (flag 1), 0 without one."""
    pl = o.players[0]
    best = None
    for e in o.entities:
        if e['kind'] == 6 and e['flags'] & 1:
            d = (float(e['x']) - pl['x']) ** 2 + (float(e['y']) - pl['y']) ** 2
            if best is None or d < best[0]:
                best = (d, float(e['x']) - pl['x'], float(e['y']) - pl['y'])
    if best is None:
        return 0
    _, dx, dy = best
    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 4
    return 3 if dy > 0 else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:16')
    p.add_argument('--floor', action='store_true')
    p.add_argument('--legs', type=int, default=6)
    p.add_argument('--warm', type=int, default=10)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--after-clear', type=int, default=15)
    p.add_argument('--check', type=int, default=25)
    p.add_argument('--port', type=int, default=31000)
    p.add_argument('--name', default='nobprobe')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                   stub_list=stub if Path(stub).is_file() else '')
    rep = cfg.frames_per_decision
    inst = Instance(args.name, args.port, cfg, spec)
    rows = []
    t0 = time.perf_counter()
    try:
        for seed in parse_seeds(args.seeds):
            row = (floor_seed if args.floor else room_seed)(inst, seed, args, rep)
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
    finally:
        inst.close()
        done = [r for r in rows if 'skipped' not in r]
        summary = dict(group='floor' if args.floor else args.group, seeds=len(rows), tested=len(done),
                       decisions=sum(r['decisions'] for r in done), payloads=sum(r['payloads'] for r in done),
                       payload_equal=sum(r['payload_mismatch'] is None for r in done),
                       digests_equal=sum(r['digests_equal'] for r in done),
                       native_calls=sum(r['native_calls'] for r in done),
                       native_fallbacks=sum(r['native_fallbacks'] for r in done),
                       check_checks=sum(r['check_checks'] for r in done),
                       check_mismatches=sum(r['check_mismatches'] for r in done),
                       laser_obs=sum(r['laser_obs'] for r in done), seconds=round(time.perf_counter() - t0, 1))
        (out / 'rows.json').write_text(json.dumps(rows, indent=1, default=str))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
