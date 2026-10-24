"""Step line and native step check (2026-10-04): are the observations byte for byte the same and is the game unchanged
when a lean clone is stepped with the step line ("S ..."), with abp_turbo's native step (abp_native_step) or both?

Per seed (as abplus_probe_native_obs.py: room mode with sticky random actions, or --floor legs through doors), four lean
clones of the same state get the same actions:
  ref     JSON step command, native step off, terrain block cache off (the bridge as before: pack_lean, luasocket,
          json.decode, every terrain block built anew)
  json1   JSON step command, native step on (the observation goes out natively, the JSON command is read by Lua),
          terrain block cache in check mode (built anew and compared with the cached one)
  line0   step line, native step off (the Lua parse of the step line), terrain block cache on
  line1   step line, native step on (the sampler's path), terrain block cache on
At every decision the raw payloads must be equal, every `check` decisions and at the end the hidden-state digests
(abplus_goexplore.DIGEST_LUA) too. The clones' native_step counters say how many observations went out natively.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_native_step.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:24 --out <dir>
  python abplus_probe_native_step.py --groups-file ../abplus/catalog/scaling2_groups.json --floor \
      --seeds range:2147600000:4 --out <dir>
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

# (name, command, native step, terrain block cache 0 off / 1 on / 2 check)
CLONES = (('ref', 'json', 0, 0), ('json1', 'json', 1, 2), ('line0', 'line', 0, 1), ('line1', 'line', 1, 1))


def set_mode(c, cmd, mode=None, **extra):
    c._send({"cmd": cmd, **extra} if mode is None else {"cmd": cmd, "mode": mode, **extra})
    msg = c._recv()
    if msg.get('type') != 'ok':
        raise RuntimeError(f'{cmd} failed: {msg}')
    return msg


class Group:
    def __init__(self, parent, check_every):
        self.clones = {}
        for name, how, mode, cache in CLONES:
            c = parent.fork(tag=name, lean=True, alarm=900)
            set_mode(c, 'native_step', mode, terrain_cache=cache)
            if how == 'line' and not c.hello.get('step_line'):
                raise RuntimeError('the bridge offers no step line')
            self.clones[name] = (c, how)
        self.decoder = LeanDecoder()
        self.check_every = check_every
        self.decisions = self.payloads = self.digest_checks = 0
        self.mismatch = None
        self.digests_equal = True
        self.obs = self._collect(None)

    def _collect(self, step):
        for c, how in self.clones.values():
            if step is None:
                c._send({"cmd": "obs"})
            elif how == 'line':
                c._sock.sendall(b"S %d %d %d %d %d\n" % (step[4], step[0], step[1], step[2], step[3]))
            else:
                c._send({"cmd": "step", "repeat": step[4], "move": step[0], "shoot": step[1], "bomb": step[2],
                         "item": step[3]})
        raw = {k: read_raw(c) for k, (c, _) in self.clones.items()}
        self.payloads += 1
        if self.mismatch is None:
            for k in raw:
                if raw[k] != raw['ref']:
                    self.mismatch = dict(decision=self.decisions, clone=k, **first_diff(raw['ref'], raw[k]))
                    break
        return self.decoder.decode(raw['ref'])

    def step(self, action, repeat):
        self.decisions += 1
        self.obs = self._collect(tuple(int(v) for v in action) + (int(repeat),))
        if self.decisions % self.check_every == 0:
            self.compare_digests()
        return self.obs

    def lua(self, code):
        for c, _ in self.clones.values():
            if c.lua(code) != 'ok':
                raise RuntimeError('lua command failed')

    def compare_digests(self):
        d = {k: digest(c)[0] for k, (c, _) in self.clones.items()}
        self.digest_checks += 1
        self.digests_equal = self.digests_equal and len(set(d.values())) == 1

    def finish(self):
        self.compare_digests()
        stats = {k: set_mode(c, 'native_step') for k, (c, _) in self.clones.items()}
        for c, _ in self.clones.values():
            c.close()
        return dict(decisions=self.decisions, payloads=self.payloads, payload_mismatch=self.mismatch,
                    digests_equal=self.digests_equal, digest_checks=self.digest_checks,
                    native_steps={k: v.get('steps') for k, v in stats.items()},
                    native_blocks={k: v.get('blocks') for k, v in stats.items()},
                    native_fallbacks={k: v.get('fallbacks') for k, v in stats.items()},
                    available=stats['line1'].get('available'), turbo=stats['line1'].get('turbo'),
                    terrain_cache={k: (v.get('terrain_cache'), v.get('tc_hits'), v.get('tc_builds'), v.get('tc_checks'),
                                       v.get('tc_mismatches')) for k, v in stats.items()})


def room_seed(inst, seed, args, rep):
    rng = np.random.default_rng(seed)
    warm, test = sticky_actions(rng, args.warm), sticky_actions(rng, args.steps)
    inst.reset(seed)
    _, _, stop = inst.env.bridge.play(warm, repeat=rep, stop_clear=True)
    if stop != 'done':
        return dict(seed=seed, skipped='episode ended in the warm-up')
    g = Group(inst.env.bridge, args.check)
    after = None
    for code in test:
        o = g.step(split_code(code), rep)
        if o.dead:
            break
        if o.clear and after is None:
            after = args.after_clear
        if after is not None:
            after -= 1
            if after < 0:
                break
    return dict(seed=seed, entities_last=len(g.obs.entities), **g.finish())


def floor_seed(inst, seed, args, rep):
    inst.env.bridge.reset_mode = 'floor'
    inst.reset(seed)
    rng = np.random.default_rng(seed)
    g = Group(inst.env.bridge, args.check)
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
            o = g.step((move_to(pl['x'], pl['y'], float(d['x']), float(d['y'])), 0, 0, 0), rep)
            if o.room[4] != here or o.dead:
                break
        if o.room[4] == here or o.dead:
            break
        legs += 1
        rooms.add(o.room[4])
        after = None
        for code in sticky_actions(rng, args.steps):
            move, _, bomb, item = split_code(code)
            o = g.step((move, aim(o), bomb, item), rep)
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
                o = g.step((0, 0, 0, 0), rep)
    return dict(seed=seed, legs=legs, rooms=len(rooms), **g.finish())


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
    p.add_argument('--fpd', type=int, default=4)
    p.add_argument('--port', type=int, default=34100)
    p.add_argument('--name', default='stpprobe')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                   stub_list=stub if Path(stub).is_file() else '')
    inst = Instance(args.name, args.port, cfg, spec)
    rows = []
    t0 = time.perf_counter()
    try:
        for seed in parse_seeds(args.seeds):
            row = (floor_seed if args.floor else room_seed)(inst, seed, args, args.fpd)
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
    finally:
        inst.close()
        done = [r for r in rows if 'skipped' not in r]
        summary = dict(group='floor' if args.floor else args.group, fpd=args.fpd, seeds=len(rows), tested=len(done),
                       decisions=sum(r['decisions'] for r in done), payloads=sum(r['payloads'] for r in done),
                       payload_equal=sum(r['payload_mismatch'] is None for r in done),
                       digests_equal=sum(r['digests_equal'] for r in done),
                       native_steps_line1=sum(r['native_steps']['line1'] or 0 for r in done),
                       native_steps_json1=sum(r['native_steps']['json1'] or 0 for r in done),
                       native_steps_ref=sum(r['native_steps']['ref'] or 0 for r in done),
                       blocks_line1=sum(r['native_blocks']['line1'] or 0 for r in done),
                       fallbacks_line1=sum(r['native_fallbacks']['line1'] or 0 for r in done),
                       tc_hits_line1=sum(r['terrain_cache']['line1'][1] or 0 for r in done),
                       tc_builds_line1=sum(r['terrain_cache']['line1'][2] or 0 for r in done),
                       tc_checks_json1=sum(r['terrain_cache']['json1'][3] or 0 for r in done),
                       tc_mismatches_json1=sum(r['terrain_cache']['json1'][4] or 0 for r in done),
                       tc_builds_ref=sum(r['terrain_cache']['ref'][2] or 0 for r in done),
                       seconds=round(time.perf_counter() - t0, 1))
        (out / 'rows.json').write_text(json.dumps(rows, indent=1, default=str))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
