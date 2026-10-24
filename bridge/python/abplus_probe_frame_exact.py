"""Frame-cost exactness probe (2026-10-04): does a speed-up of the engine's logic frame leave the game and the
observations unchanged?

Per seed (room mode: a room of the group, sticky random actions; --floor: legs through doors as
abplus_probe_native_step.py), two lean clones of the same state are stepped with step lines (the sampler's path,
native step on):
  ref   the instance as launched (plus --ref-lua)
  fast  the same state with the speed-ups switched on in that clone only (--fast-lua: Lua run in the clone first, e.g.
        "return tostring(os.getenv('ABP_STUBS:<file>'))" applies a stub list to the clone's own code pages,
        "return tostring(os.getenv('ABP_FAST:<mask>'))" abp_turbo's fast paths (the PNG cache, bit 2, needs its shared
        region from the start: --env ABP_PNG_CACHE_MB=512), or a bridge setting such as "ABP_PU.enabled = true")
At every decision the raw lean payloads must be byte for byte equal, every --check decisions and at the end the
hidden-state digests (abplus_goexplore.DIGEST_LUA: player, every entity incl. NPC AI state, seeds, grid, global MT).
Both clones' CPU time (/proc/<pid>/schedstat) is summed per decision: a speed comparison on the same states, with the
two clones stepped alternately (same machine load). Results in <out>/rows.json and summary.json.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_frame_exact.py --group normal --seeds range:2147500000:32 --out <dir> \
      --fast-lua "return tostring(os.getenv('ABP_STUBS:/path/stub_render_h.txt'))"
  python abplus_probe_frame_exact.py --floor --seeds range:2147600000:4 --out <dir> --fast-lua ...
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from abplus_probe_fork import digest, split_code, sticky_actions
from abplus_probe_native_obs import KILL_LUA, aim, first_diff, move_to, read_raw
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder


def cpu_ns(pid):
    try:
        return int(open(f'/proc/{pid}/schedstat').read().split()[0])
    except (OSError, ValueError, IndexError):
        return 0


class Pair:
    def __init__(self, parent, check_every, setups):
        self.clones = {}
        self.setup_answers = {}
        for name in ('ref', 'fast'):
            c = parent.fork(tag=name, lean=True, alarm=1800)
            self.setup_answers[name] = [c.lua(code) for code in setups[name]]
            self.clones[name] = c
        self.decoder = LeanDecoder()
        self.check_every = check_every
        self.decisions = self.payloads = self.digest_checks = 0
        self.mismatch = None
        self.digest_mismatch = None
        self.cpu = dict.fromkeys(self.clones, 0)
        self.obs = self._collect(None)

    def _collect(self, step):
        raw = {}
        for k, c in self.clones.items():   # one after the other: each clone's CPU is its own step only
            c0 = cpu_ns(c.pid)
            if step is None:
                c._send({"cmd": "obs"})
            else:
                c._sock.sendall(b"S %d %d %d %d %d\n" % (step[4], step[0], step[1], step[2], step[3]))
            raw[k] = read_raw(c)
            self.cpu[k] += cpu_ns(c.pid) - c0
        self.payloads += 1
        if self.mismatch is None and raw['fast'] != raw['ref']:
            self.mismatch = dict(decision=self.decisions, **first_diff(raw['ref'], raw['fast']))
        return self.decoder.decode(raw['ref'])

    def step(self, action, repeat):
        self.decisions += 1
        self.obs = self._collect(tuple(int(v) for v in action) + (int(repeat),))
        if self.decisions % self.check_every == 0:
            self.compare_digests()
        return self.obs

    def lua(self, code):
        for c in self.clones.values():
            if c.lua(code) != 'ok':
                raise RuntimeError('lua command failed')

    def compare_digests(self):
        d = {k: digest(c)[0] for k, c in self.clones.items()}
        self.digest_checks += 1
        if d['ref'] != d['fast'] and self.digest_mismatch is None:
            self.digest_mismatch = self.decisions

    def finish(self, status_lua):
        self.compare_digests()
        status = {k: c.lua(status_lua) if status_lua else None for k, c in self.clones.items()}
        for c in self.clones.values():
            c.close()
        return dict(decisions=self.decisions, payloads=self.payloads, payload_mismatch=self.mismatch,
                    digests_equal=self.digest_mismatch is None, digest_mismatch_at=self.digest_mismatch,
                    digest_checks=self.digest_checks, setup=self.setup_answers, status=status,
                    cpu_ms_ref=round(self.cpu['ref'] / 1e6, 3), cpu_ms_fast=round(self.cpu['fast'] / 1e6, 3))


def room_seed(inst, seed, args, setups):
    rng = np.random.default_rng(seed)
    warm, test = sticky_actions(rng, args.warm), sticky_actions(rng, args.steps)
    inst.reset(seed)
    _, _, stop = inst.env.bridge.play(warm, repeat=args.fpd, stop_clear=True)
    if stop != 'done':
        return dict(seed=seed, skipped='episode ended in the warm-up')
    g = Pair(inst.env.bridge, args.check, setups)
    after = None
    for code in test:
        o = g.step(split_code(code), args.fpd)
        if o.dead:
            break
        if o.clear and after is None:
            after = args.after_clear
        if after is not None:
            after -= 1
            if after < 0:
                break
    return dict(seed=seed, entities_last=len(g.obs.entities), **g.finish(args.status_lua))


def floor_seed(inst, seed, args, setups):
    inst.env.bridge.reset_mode = 'floor'
    inst.reset(seed)
    rng = np.random.default_rng(seed)
    g = Pair(inst.env.bridge, args.check, setups)
    legs, rooms = 0, set()
    o = g.obs
    for _ in range(args.legs):
        doors = [d for d in o.doors if d['open'] and not d['locked']]
        if not doors or o.dead:
            break
        d = doors[int(rng.integers(len(doors)))]
        here = o.room[4]
        for _ in range(150):
            pl = o.players[0]
            o = g.step((move_to(pl['x'], pl['y'], float(d['x']), float(d['y'])), 0, 0, 0), args.fpd)
            if o.room[4] != here or o.dead:
                break
        if o.room[4] == here or o.dead:
            break
        legs += 1
        rooms.add(o.room[4])
        after = None
        for code in sticky_actions(rng, args.steps):
            move, _, bomb, item = split_code(code)
            o = g.step((move, aim(o), bomb, item), args.fpd)
            if o.dead:
                break
            if o.clear and after is None:
                after = args.after_clear
            if after is not None:
                after -= 1
                if after < 0:
                    break
        if not o.clear and not o.dead:
            g.lua(KILL_LUA)
            for _ in range(args.after_clear):
                o = g.step((0, 0, 0, 0), args.fpd)
    return dict(seed=seed, legs=legs, rooms=len(rooms), **g.finish(args.status_lua))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='../abplus/catalog/scaling2_groups.json')
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
    p.add_argument('--fpd', type=int, default=4)
    p.add_argument('--fast-lua', action='append', default=[])
    p.add_argument('--ref-lua', action='append', default=[])
    p.add_argument('--status-lua', default='')
    p.add_argument('--stub-list', default='', help="ABP_STUB_LIST of the instance ('' = the copy's stub_render_g.txt)")
    p.add_argument('--env', default='', help='extra instance environment, k=v,k=v')
    p.add_argument('--port', type=int, default=37200)
    p.add_argument('--name', default='frmexact')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    for kv in filter(None, args.env.split(',')):
        k, v = kv.split('=', 1)
        os.environ[k] = v
    stub = args.stub_list or str(Path(default_preload()).parent / 'stub_render_g.txt')
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0, stub_list=stub)
    setups = dict(ref=args.ref_lua, fast=args.fast_lua)
    inst = Instance(args.name, args.port, cfg, spec)
    rows = []
    t0 = time.perf_counter()
    load0 = os.getloadavg()
    try:
        for seed in parse_seeds(args.seeds):
            row = (floor_seed if args.floor else room_seed)(inst, seed, args, setups)
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
    finally:
        inst.close()
        done = [r for r in rows if 'skipped' not in r]
        dec = sum(r['decisions'] for r in done)
        cpu_ref = sum(r['cpu_ms_ref'] for r in done)
        cpu_fast = sum(r['cpu_ms_fast'] for r in done)
        summary = dict(group='floor' if args.floor else args.group, fpd=args.fpd, seeds=len(rows), tested=len(done),
                       decisions=dec, payloads=sum(r['payloads'] for r in done),
                       payload_equal=sum(r['payload_mismatch'] is None for r in done),
                       digests_equal=sum(r['digests_equal'] for r in done),
                       digest_checks=sum(r['digest_checks'] for r in done),
                       legs=sum(r.get('legs', 0) for r in done),
                       cpu_ms_per_decision_ref=round(cpu_ref / max(dec, 1), 4),
                       cpu_ms_per_decision_fast=round(cpu_fast / max(dec, 1), 4),
                       fast_lua=args.fast_lua, ref_lua=args.ref_lua, stub_list=stub, env=args.env,
                       load_start=load0, load_end=os.getloadavg(), seconds=round(time.perf_counter() - t0, 1))
        (out / 'rows.json').write_text(json.dumps(rows, indent=1, default=str))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
