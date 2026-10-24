"""A19 (gate 0), part 4: does a clone stay an exact copy across room changes (new rooms loaded inside the clone)?

Per seed: a real Basement I floor from its start room (bridge._restart_floor), the player made invincible (the walk
must not end in a death), then a clone. Parent and clone each go start -> A -> B -> back to A through the doors
(Level.LeaveDoor + Game():StartRoomTransition, which enters a room as walking through the door does, EXPERIMENTS.md A9)
and play the same `steps` random decisions in each room; the hidden-state digest after every stage must be the same.
A and B are 1x1 normal rooms (chain_pairs). The clone's counters show whether it tried to write files (the continue
save at a room change), which abp_turbo sends to /dev/null in a clone.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_fork_rooms.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:16 --out <dir>
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from abplus_probe_fork import digest, sticky_actions
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import CHAIN_REROLL_CURSES, FORK_ENV, chain_pairs, room_seed
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.env import BridgeError


def walk(bridge, legs, plans, repeat):
    """The stages' (room index, digest, played, stop) for the legs [(slot, target)] and their action plans."""
    out = []
    for (slot, target), plan in zip(legs, plans):
        bridge.lua(f"local l = Game():GetLevel(); l.LeaveDoor = {slot}; Game():StartRoomTransition({target}, {slot}, 0); "
                   "return 'ok'")
        obs, _, stop = bridge.play([0] * 90, repeat=1, stop_clear=False)   # ends at the new room's first logic frame
        if stop != 'room' or obs['room']['room_idx'] != target:
            out.append((obs['room']['room_idx'], 'no transition', 0, stop))
            break
        obs, played, stop = bridge.play(plan, repeat=repeat, stop_clear=False)
        out.append((obs['room']['room_idx'], digest(bridge)[0], played, stop))
        if stop != 'done':
            break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:16')
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--port', type=int, default=27985)
    p.add_argument('--name', default='fkrooms')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0)
    inst = Instance(args.name, args.port, cfg, spec)
    rows = []
    try:
        for seed in parse_seeds(args.seeds):
            row = dict(seed=seed)
            inst.reset(seed)   # connects and installs the digest function
            bridge = inst.env.bridge
            curses, start, floor = bridge._restart_floor(room_seed(seed, -2))
            variants = {r['variant'] for r in floor.values()}
            pairs = chain_pairs(floor, start, variants)
            if curses & CHAIN_REROLL_CURSES or not pairs:
                row['skipped'] = 'curse' if curses & CHAIN_REROLL_CURSES else 'no start-A-B chain of 1x1 normal rooms'
                rows.append(row)
                continue
            slot_a, a, slot_b, b = pairs[0]
            bridge.lua("AbpSetInvincible(true); AbpSetMissCap(0); return 'ok'")
            bridge.step({}, repeat=1)
            legs = [(slot_a, a), (slot_b, b), ((slot_b + 2) % 4, a)]
            rng = np.random.default_rng(seed)
            plans = [sticky_actions(rng, args.steps) for _ in legs]
            row.update(start=start, a=[a, floor[a]['variant']], b=[b, floor[b]['variant']])
            d0 = digest(bridge)[0]
            clone = bridge.fork(tag=str(seed))
            row['clone_start_equal'] = digest(clone)[0] == d0
            try:
                parent = walk(bridge, legs, plans, cfg.frames_per_decision)
                child = walk(clone, legs, plans, cfg.frames_per_decision)
                row['stages'] = len(parent)
                row['rooms'] = [s[0] for s in parent]
                row['equal'] = parent == child
                if parent != child:
                    row['parent'], row['clone'] = parent, child
                row['clone_counters'] = clone.lua("return tostring(os.getenv('ABP_FORK_COUNTERS'))")
            except (BridgeError, OSError) as exc:
                row['error'] = repr(exc)
            finally:
                try:
                    clone.close()
                except OSError:
                    pass
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        inst.close()
        done = [r for r in rows if 'skipped' not in r]
        summary = dict(seeds=len(rows), tested=len(done), equal=sum(r.get('equal') is True for r in done),
                       three_stages=sum(r.get('stages') == 3 for r in done), errors=sum('error' in r for r in done),
                       clone_start_equal=sum(r.get('clone_start_equal') is True for r in done))
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
