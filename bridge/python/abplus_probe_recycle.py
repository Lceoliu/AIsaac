"""Does a recycled root build the same start states? (2026-10-06, tok_sampler.recycle_root)

A worker's root that is replaced at a state boundary builds its next floor as the first build of a fresh process,
where the old root would have built it after many floors. Per test seed: root A (one instance for the whole probe,
--history floors built first, then the test seeds one after the other) and root B (launched fresh for each test seed:
its first build is the seed) reset the same floor; their hidden-state digests are compared (abplus_goexplore DIGEST_LUA:
player, every entity incl. AI state, seeds, grid, global MT); then a lean clone of each plays the same actions (legs
through doors, sticky random actions aimed at the nearest enemy, as abplus_probe_frame_exact.py --floor), and at every
decision the raw lean payloads must be equal apart from the header's frame counters (bytes 4-11: the bridge's logic
frame count and the game's frame count, which count the process's history), every --check decisions and at the end the
digests too. Results in <out>/rows.json, summary.json.
usage (PYTHONPATH=.): python abplus_probe_recycle.py --seeds range:2147600400:8 --history 6 --out ../runs/recycle
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
from isaac_bridge.tok_obs import ROW, EpisodeState, encode_row
from isaac_bridge.tok_sampler import apply_instance_defaults, default_stub_list


def masked(raw):
    return raw[:4] + raw[12:]


def obs_diff(a, b):
    """The first field in which two decoded lean observations differ (None: equal). The frame counters and the terrain
    block's version counter count the process's history and are left out; the terrain is compared as parsed (its JSON
    text lists the keys in the order of the Lua table, which depends on the process's history too)."""
    for name in ('clear', 'paused', 'room', 'damage_taken', 'monsters_hp', 'blocking_hp', 'blocking_count'):
        if getattr(a, name) != getattr(b, name):
            return name
    for name in ('players', 'doors', 'entities', 'map'):
        x, y = getattr(a, name), getattr(b, name)
        if x.shape != y.shape or x.tobytes() != y.tobytes():
            return name
    if len(a.lasers) != len(b.lasers) or any(
            {k: v for k, v in la.items() if k != 'samples'} != {k: v for k, v in lb.items() if k != 'samples'}
            or not np.array_equal(la['samples'], lb['samples']) for la, lb in zip(a.lasers, b.lasers)):
        return 'lasers'
    ta = {k: v for k, v in (a.terrain or {}).items() if k != 'version'}
    tb = {k: v for k, v in (b.terrain or {}).items() if k != 'version'}
    if ta != tb:
        return 'terrain'
    if a.grid != b.grid:
        return 'grid'
    return None


class Pair:
    def __init__(self, parents, check_every):
        self.clones = {k: p.fork(tag=k, lean=True, alarm=1800) for k, p in parents.items()}
        self.decoders = {k: LeanDecoder() for k in self.clones}
        self.states = {k: EpisodeState(3600, floor=True, stall=450) for k in self.clones}
        self.rows = {k: np.zeros((1,), ROW) for k in self.clones}
        self.check_every = check_every
        self.decisions = self.payloads = self.digest_checks = 0
        self.mismatch = self.digest_mismatch = self.row_mismatch = None
        self.raw_equal = 0   # payloads equal byte for byte apart from the frame counters
        self.obs = self._collect(None)

    def _collect(self, step):
        raw, obs = {}, {}
        for k, c in self.clones.items():
            if step is None:
                c._send({"cmd": "obs"})
            else:
                c._sock.sendall(b"S %d %d %d %d %d\n" % (step[4], step[0], step[1], step[2], step[3]))
            raw[k] = read_raw(c)
            obs[k] = self.decoders[k].decode(raw[k])
            encode_row(obs[k], self.states[k], self.rows[k], self.decisions)
        self.payloads += 1
        self.raw_equal += masked(raw['a']) == masked(raw['b'])
        if self.mismatch is None:
            field = obs_diff(obs['a'], obs['b'])
            if field is not None:
                self.mismatch = dict(decision=self.decisions, field=field,
                                     **first_diff(masked(raw['a']), masked(raw['b'])))
        if self.row_mismatch is None and self.rows['a'].tobytes() != self.rows['b'].tobytes():
            self.row_mismatch = self.decisions
        return obs['a']

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
        if d['a'] != d['b'] and self.digest_mismatch is None:
            self.digest_mismatch = self.decisions

    def finish(self):
        self.compare_digests()
        for c in self.clones.values():
            c.close()
        return dict(decisions=self.decisions, payloads=self.payloads, payload_mismatch=self.mismatch,
                    row_mismatch_at=self.row_mismatch, raw_equal_payloads=self.raw_equal,
                    digests_equal=self.digest_mismatch is None, digest_mismatch_at=self.digest_mismatch,
                    digest_checks=self.digest_checks)


def play_floor(parents, seed, args):
    rng = np.random.default_rng(seed)
    g = Pair(parents, args.check)
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
    return dict(legs=legs, rooms=len(rooms), **g.finish())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='../abplus/catalog/scaling2_groups.json')
    p.add_argument('--seeds', default='range:2147600400:8')
    p.add_argument('--history', type=int, default=6, help='floors root A builds before the first test seed')
    p.add_argument('--legs', type=int, default=6)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--after-clear', type=int, default=15)
    p.add_argument('--check', type=int, default=25)
    p.add_argument('--fpd', type=int, default=4)
    p.add_argument('--port', type=int, default=40200)
    p.add_argument('--name', default='memrc')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    os.environ.update(FORK_ENV)
    apply_instance_defaults()   # the workers' instance defaults
    preload = default_preload()
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=preload, al_stopped=True, nice=0,
                   stub_list=default_stub_list(preload))
    a = Instance(f'{args.name}a', args.port, cfg, spec)
    a.env.bridge.reset_mode = 'floor'
    rows = []
    t0 = time.perf_counter()
    try:
        for k in range(args.history):
            a.reset(1000 + 16 * k)
        for seed in parse_seeds(args.seeds):
            a.reset(seed)
            b = Instance(f'{args.name}b', args.port + 1, cfg, spec)
            try:
                b.env.bridge.reset_mode = 'floor'
                b.reset(seed)
                da, db = a.digest(), b.digest()
                row = dict(seed=seed, root_digest_equal=da == db, root_digest_a=da, root_digest_b=db,
                           **play_floor({'a': a.env.bridge, 'b': b.env.bridge}, seed, args))
            finally:
                b.kill()
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
    finally:
        a.kill()
        summary = dict(seeds=len(rows), root_digests_equal=sum(r['root_digest_equal'] for r in rows),
                       decisions=sum(r['decisions'] for r in rows),
                       observation_equal=sum(r['payload_mismatch'] is None for r in rows),
                       rows_equal=sum(r['row_mismatch_at'] is None for r in rows),
                       raw_equal_payloads=sum(r['raw_equal_payloads'] for r in rows),
                       payloads=sum(r['payloads'] for r in rows),
                       digests_equal=sum(r['digests_equal'] for r in rows),
                       digest_checks=sum(r['digest_checks'] for r in rows), legs=sum(r['legs'] for r in rows),
                       history=args.history, seconds=round(time.perf_counter() - t0, 1))
        (out / 'rows.json').write_text(json.dumps(rows, indent=1, default=str))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
