"""malloc_trim in a parked clone: how much memory does it give back, and is the game the same afterwards? (2026-10-06)

Per seed: the root resets the floor; an episode clone walks through doors (sticky random actions aimed at the nearest
enemy, as abplus_probe_frame_exact.py --floor); at the entry of room --at it is cloned twice (two parked clones of the
same moment, as tok_floor parks a room entry); the episode then plays --after more decisions and is closed, as it is in
training. One parked clone is trimmed (abp_turbo ABP_MALLOC_INFO:TRIM: glibc malloc_trim(0), then mallinfo2), the other
not; their Pss / Private_Dirty (smaps_rollup) are read before and after. Then each parked clone gives an episode clone
with the same reseed, and the two play the same actions (--legs more legs): the raw lean payloads must be equal at every
decision, the hidden-state digests every --check decisions and at the end.
usage (PYTHONPATH=.): python abplus_probe_trim.py --seeds range:2147600500:8 --out ../runs/trim
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
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.tok_sampler import apply_instance_defaults, default_stub_list

TRIM = "return tostring(os.getenv('ABP_MALLOC_INFO:TRIM'))"
INFO = "return tostring(os.getenv('ABP_MALLOC_INFO'))"


def roll(pid):
    out = {}
    try:
        for line in open(f'/proc/{pid}/smaps_rollup'):
            k, _, v = line.partition(':')
            if k in ('Rss', 'Pss', 'Private_Dirty', 'Shared_Dirty'):
                out[k] = int(v.split()[0])
    except OSError:
        pass
    return out


class Walker:
    """Steps one or two lean clones with the same actions; payloads compared when two."""

    def __init__(self, clones, check):
        self.clones, self.check = clones, check
        self.decoder = LeanDecoder()
        self.decisions = 0
        self.mismatch = self.digest_mismatch = None
        self.digest_checks = 0
        self.obs = self._collect(None)

    def _collect(self, step):
        raw = []
        for c in self.clones:
            if step is None:
                c._send({"cmd": "obs"})
            else:
                c._sock.sendall(b"S %d %d %d %d %d\n" % (step[4], step[0], step[1], step[2], step[3]))
            raw.append(read_raw(c))
        if len(raw) == 2 and self.mismatch is None and raw[0] != raw[1]:
            self.mismatch = dict(decision=self.decisions, **first_diff(raw[0], raw[1]))
        return self.decoder.decode(raw[0])

    def step(self, action, fpd):
        self.decisions += 1
        self.obs = self._collect(tuple(int(v) for v in action) + (int(fpd),))
        if len(self.clones) == 2 and self.decisions % self.check == 0:
            self.compare()
        return self.obs

    def compare(self):
        d = [digest(c)[0] for c in self.clones]
        self.digest_checks += 1
        if d[0] != d[1] and self.digest_mismatch is None:
            self.digest_mismatch = self.decisions

    def lua(self, code):
        for c in self.clones:
            c.lua(code)


def leg(w, rng, args, fight=True):
    """Walk to a random open door, then (fight) sticky random actions aimed at the nearest enemy. True: a new room."""
    o = w.obs
    doors = [d for d in o.doors if d['open'] and not d['locked']]
    if not doors or o.dead:
        return False
    d = doors[int(rng.integers(len(doors)))]
    here = o.room[4]
    for _ in range(150):
        pl = o.players[0]
        o = w.step((move_to(pl['x'], pl['y'], float(d['x']), float(d['y'])), 0, 0, 0), args.fpd)
        if o.room[4] != here or o.dead:
            break
    if o.room[4] == here or o.dead:
        return False
    if fight:
        fight_room(w, rng, args)
    return True


def fight_room(w, rng, args):
    o = w.obs
    after = None
    for code in sticky_actions(rng, args.steps):
        move, _, bomb, item = split_code(code)
        o = w.step((move, aim(o), bomb, item), args.fpd)
        if o.dead:
            break
        if o.clear and after is None:
            after = args.after_clear
        if after is not None:
            after -= 1
            if after < 0:
                break
    if not o.clear and not o.dead:
        w.lua(KILL_LUA)
        for _ in range(args.after_clear):
            o = w.step((0, 0, 0, 0), args.fpd)


def one_seed(inst, seed, args):
    inst.env.bridge.reset_mode = 'floor'
    inst.reset(seed)
    rng = np.random.default_rng(seed)
    if args.at == 0:   # the floor's start, parked as the template is (a clone of the root right after its reset)
        snaps = [inst.env.bridge.fork(tag='template', lean=True, alarm=1800) for _ in range(2)]
        inst.reset(seed + 1)   # the root moves on to another floor, as in training
    else:
        ep = inst.env.bridge.fork(tag='ep', lean=True, alarm=1800)
        w = Walker([ep], args.check)
        entered = 0
        while entered < args.at:
            if not leg(w, rng, args, fight=entered + 1 < args.at):
                ep.close()
                return dict(seed=seed, skipped=f'no room {entered + 1}')
            entered += 1
        snaps = [ep.fork(tag='snap', lean=True, alarm=1800) for _ in range(2)]   # the room entry, parked twice
        fight_room(w, rng, args)   # the episode plays on in this room
        for _ in range(args.after_legs):
            if not leg(w, rng, args):
                break
        ep.close()
    time.sleep(0.5)
    before = [roll(s.pid) for s in snaps]
    t0 = time.perf_counter()
    trim = snaps[0].lua(TRIM)   # the other parked clone gets no command at all (the control)
    trim_s = time.perf_counter() - t0
    after = [roll(s.pid) for s in snaps]
    reseed = int(rng.integers(1, 2 ** 31 - 1))
    eps = [s.fork(tag='e', lean=True, alarm=1800, reseed=reseed) for s in snaps]
    w2 = Walker(eps, args.check)
    rng2 = np.random.default_rng(seed + 7)
    fight_room(w2, rng2, args)
    legs = 0
    for _ in range(args.legs):
        if not leg(w2, rng2, args):
            break
        legs += 1
    w2.compare()
    info1 = snaps[1].lua(INFO)   # (after the comparison: the control took no command before it)
    for c in eps + snaps:
        c.close()
    return dict(seed=seed, trim_info=trim, untrimmed_info=info1, trim_ms=round(1000 * trim_s, 1),
                before=before, after=after, decisions=w2.decisions, legs=legs, payload_mismatch=w2.mismatch,
                digests_equal=w2.digest_mismatch is None, digest_checks=w2.digest_checks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='../abplus/catalog/scaling2_groups.json')
    p.add_argument('--seeds', default='range:2147600500:8')
    p.add_argument('--at', type=int, default=2, help='the room whose entry is parked (1 = the first door)')
    p.add_argument('--after-legs', type=int, default=2, help='legs the episode plays on after parking')
    p.add_argument('--legs', type=int, default=4, help='legs the two episodes from the parked clones play')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--after-clear', type=int, default=15)
    p.add_argument('--check', type=int, default=25)
    p.add_argument('--fpd', type=int, default=4)
    p.add_argument('--port', type=int, default=40210)
    p.add_argument('--name', default='memtr')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    os.environ.update(FORK_ENV)
    apply_instance_defaults()
    preload = default_preload()
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=preload, al_stopped=True, nice=0,
                   stub_list=default_stub_list(preload))
    inst = Instance(args.name, args.port, cfg, spec)
    rows = []
    t0 = time.perf_counter()
    try:
        for seed in parse_seeds(args.seeds):
            row = one_seed(inst, seed, args)
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
    finally:
        inst.kill()
        done = [r for r in rows if 'skipped' not in r]

        def mean(key, i, field):
            xs = [r[key][i].get(field, 0) for r in done]
            return round(sum(xs) / max(len(xs), 1) / 1024, 1)
        # the two parked clones share almost every page with each other, so Pss moves between them; the trimmed
        # clone's Rss drop is what it no longer maps (an upper bound of what trimming frees once its relatives are gone)
        summary = dict(seeds=len(rows), tested=len(done), decisions=sum(r['decisions'] for r in done),
                       rss_mib_trimmed_before=mean('before', 0, 'Rss'), rss_mib_trimmed_after=mean('after', 0, 'Rss'),
                       rss_mib_untrimmed=mean('after', 1, 'Rss'),
                       payload_equal=sum(r['payload_mismatch'] is None for r in done),
                       digests_equal=sum(r['digests_equal'] for r in done),
                       digest_checks=sum(r['digest_checks'] for r in done),
                       pss_mib_trimmed_before=mean('before', 0, 'Pss'), pss_mib_trimmed_after=mean('after', 0, 'Pss'),
                       pss_mib_untrimmed=mean('after', 1, 'Pss'),
                       private_dirty_mib_trimmed_before=mean('before', 0, 'Private_Dirty'),
                       private_dirty_mib_trimmed_after=mean('after', 0, 'Private_Dirty'),
                       private_dirty_mib_untrimmed=mean('after', 1, 'Private_Dirty'),
                       trim_ms=round(sum(r['trim_ms'] for r in done) / max(len(done), 1), 1),
                       seconds=round(time.perf_counter() - t0, 1))
        (out / 'rows.json').write_text(json.dumps(rows, indent=1, default=str))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
