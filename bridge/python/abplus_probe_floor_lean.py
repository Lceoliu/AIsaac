"""C57 probe: a whole floor through the lean observation and clones.

One instance in the bridge's floor mode. Per seed: the floor's start is parked (template), a lean clone walks from
the start room through each of its open doors in turn (a straight line to the door: start rooms are empty) and the
probe reports
  - the map block (the rooms the minimap shows, before and after the first room change), the doors' seen flags, the
    room index, the events encode_row derives (a new room, a clear);
  - clones at a room entry: two clones of the state the walker entered a room in play the same random actions for
    --steps decisions; their observations must be equal byte for byte (players, entities, totals);
  - a restore from that parked entry with one `play` against the same actions stepped one by one;
  - the memory the parked clones hold (Pss of the instance's descendants).

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_floor_lean.py --groups-file ../abplus/catalog/scaling2_groups.json --seeds 2147600000:4 --out <dir>

--items (2026-10-06, Phase A): per seed a scripted run with items instead of the door walks. The root's bridge sends
the inventory block (ISAAC_RL_LEAN_ITEMS=1) and compares its cached inventory with a full scan every decision
(ISAAC_RL_INV_CHECK=1); the walker clone builds its observation in Lua and in C and compares them (native_obs mode 2).
Every record goes through encode_row and FastRow (abplus_probe_fast_row.Pair, run + items: byte-equal records). The
walker takes a passive pedestal, an active pedestal (Yum Heart), a pill, a card and a trinket spawned next to it, uses
the active item, the pill and the card (the buttons through the step line), then a trapdoor is spawned and the floor's
start parked there (entry). From entry: two clones play the same thing (walk into the trapdoor, the next floor, sticky
random actions with the item / pill buttons): every raw payload equal, and the hidden-state digest at the end; the
walker itself does the same (unchanged by its clones); a restore with `play` up to the floor change and then on equals
the stepped clone. On the next floor: Curse of the Lost added (the map block must become empty) and Curse of the Blind
with a pedestal spawned (its ROW ent_item must be ITEM_UNKNOWN). Last, the restore-clone map check: a clone parked
with lean_items off / on, a clone of it whose first observation is a play's: does it carry the map block?
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV, action_code
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.tok_obs import ROW, EpisodeState, encode_row
from isaac_bridge.tok_sampler import lean_step, play_actions

# the bridge's move codes: 0 none, then clockwise from up
MOVES = {(0, 0): 0, (0, -1): 1, (1, -1): 2, (1, 0): 3, (1, 1): 4, (0, 1): 5, (-1, 1): 6, (-1, 0): 7, (-1, -1): 8}


def digest(o):
    return (o.players.tobytes(), o.entities.tobytes(), o.room, o.damage_taken, o.monsters_hp, o.clear)


def move_to(px, py, x, y):
    sx = 0 if abs(x - px) < 6 else (1 if x > px else -1)
    sy = 0 if abs(y - py) < 6 else (1 if y > py else -1)
    return MOVES[(sx, sy)]


def descendants(pid):
    out = subprocess.run(['ps', '-e', '-o', 'pid=,ppid='], capture_output=True, text=True).stdout
    kids = {}
    for line in out.splitlines():
        a, b = line.split()
        kids.setdefault(int(b), []).append(int(a))
    found, todo = [], [pid]
    while todo:
        for k in kids.get(todo.pop(), []):
            found.append(k)
            todo.append(k)
    return found


def pss_mib(pid):
    try:
        for line in open(f'/proc/{pid}/smaps_rollup'):
            if line.startswith('Pss:'):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


SPAWN_LUA = ("local p = Isaac.GetPlayer(0) local room = Game():GetRoom() "
             "local pos = room:FindFreeTilePosition(p.Position + Vector({dx}, {dy}), 0) "
             "Isaac.Spawn(5, {variant}, {sub}, pos, Vector(0, 0), nil) return tostring(pos.X) .. ',' .. tostring(pos.Y)")


def items_seed(inst, seed, args, fn):
    """The --items scenario of one seed (see the module docstring)."""
    from abplus_probe_fast_row import Pair, TRAPDOOR_LUA, enter_trapdoor, step
    from abplus_probe_native_obs import native_obs, read_raw
    from isaac_bridge.tok_obs import EV_ACTIVE, EV_EXIT, EV_ITEM, EV_PILL, ITEM_UNKNOWN
    rep_ = 4
    inst.env.bridge.reset_mode = 'floor'
    inst.reset(seed)
    rec = dict(seed=seed)
    template = inst.env.bridge.fork(tag='template', alarm=0)
    parked = [template]
    walker = template.fork(lean=True, alarm=900, reseed=seed % 1000 + 1)
    native_obs(walker, 2)   # Lua and C both build every observation and are compared
    pair = Pair(fn, 10 ** 6, True, 450, run=True, items=True)
    walker._send({"cmd": "obs"})
    o, _ = pair.record(read_raw(walker))
    rec['start_inventory'] = dict(head=list(o.inv_head), held=o.inv.tolist())
    log = []

    def take(variant, sub, dx, dy, bit):
        nonlocal o
        x, y = (float(v) for v in walker.lua(SPAWN_LUA.format(variant=variant, sub=sub, dx=dx, dy=dy)).split(','))
        got = False
        for k in range(120):
            pl = o.players[0]
            o, _ = pair.record(step(walker, (move_to(pl['x'], pl['y'], x, y), 0, 0, 0, 0), rep_))
            if bit and pair.last_row['events'][0] & bit:
                got = True
                break
            if not bit and (o.inv_head[5] or o.inv_head[7] or o.inv_head[3]):
                got = True
                break
        for _ in range(8):   # the pickup animation
            o, _ = pair.record(step(walker, (0, 0, 0, 0, 0), rep_))
        log.append(dict(variant=variant, sub=sub, taken=got, decisions=k + 1, head=list(o.inv_head),
                        held=o.inv.tolist()[:6]))
        return got

    def use(item, pill, bit):
        nonlocal o
        seen = False
        o, _ = pair.record(step(walker, (0, 0, 0, item, pill), rep_))
        seen = bool(pair.last_row['events'][0] & bit)
        for _ in range(12):
            o, _ = pair.record(step(walker, (0, 0, 0, 0, 0), rep_))
            seen = seen or bool(pair.last_row['events'][0] & bit)
        log.append(dict(use='item' if item else 'pill', event=seen, head=list(o.inv_head),
                        pinv=[round(float(v), 3) for v in pair.last_row['pinv'][0]]))
        return seen

    take(100, 1, 0, 80, EV_ITEM)          # The Sad Onion (passive)
    take(100, 45, 80, 0, EV_ITEM)         # Yum Heart (active, 4 charges; the D6 is dropped on a pedestal if held)
    use(1, 0, EV_ACTIVE)
    take(70, 3, -80, 0, 0)                # a pill
    use(0, 1, EV_PILL)
    take(300, 2, 0, -80, 0)               # The Magician
    use(0, 1, EV_PILL)
    take(350, 1, 0, 80, 0)                # Swallowed Penny (trinket)
    rec['items_log'] = log
    rec['inventory_before_trapdoor'] = dict(head=list(o.inv_head), held=o.inv.tolist(),
                                            pitem=pair.last_row['pitem'][0].tolist())
    # the trapdoor, then the parked state the clones start from
    tx, ty = (float(v) for v in walker.lua(TRAPDOOR_LUA.format(dx=0, dy=-80)).split(','))
    for _ in range(30):
        o, _ = pair.record(step(walker, (0, 0, 0, 0, 0), rep_))
    entry = walker.fork(tag='entry', alarm=0)
    parked.append(entry)
    stage0 = o.room[5]

    def policy(obs, rng, phase):
        """Deterministic in (observation, rng state): to the trapdoor, then sticky random with the item buttons."""
        if obs.room[5] == stage0:
            pl = obs.players[0]
            return (move_to(pl['x'], pl['y'], tx, ty), 0, 0, 0, 0)
        if rng.random() < 0.1 or phase[0] is None:
            phase[0] = (int(rng.integers(9)), int(rng.integers(5)))
        return (phase[0][0], phase[0][1], 0, int(rng.random() < 0.1), int(rng.random() < 0.1))

    def run_clone(c, n):
        dec = LeanDecoder()
        c._send({"cmd": "obs"})
        raw = read_raw(c)
        oc = dec.decode(raw)
        trace, acts, rng, phase = [raw], [], np.random.default_rng(seed), [None]
        changed_at = None
        for j in range(n):
            if oc.dead:
                break
            a = policy(oc, rng, phase)
            acts.append(a)
            raw = step(c, a, rep_)
            oc = dec.decode(raw)
            trace.append(raw)
            if changed_at is None and oc.room[5] != stage0:
                changed_at = j + 1
        return trace, acts, changed_at, oc
    finals = []
    for _ in range(2):
        c = entry.fork(lean=True, alarm=300)
        finals.append(run_clone(c, args.steps))
        finals[-1] = finals[-1] + (c.lua('return ABPGX_DIGEST(false)'),)
        c.close()
    (t1, a1, k1, o1, d1), (t2, a2, k2, o2, d2) = finals
    rec['clones'] = dict(decisions=len(a1), floor_change_at=k1, stage_after=int(o1.room[5]), payloads_equal=t1 == t2,
                         actions_equal=a1 == a2, digest_equal=d1 == d2,
                         first_diff=next((j for j, (x, y) in enumerate(zip(t1, t2)) if x != y), None))
    # play restore: up to the floor change (play stops at the room change), then on
    if k1:
        c = entry.fork(lean=True, alarm=300)
        dec = LeanDecoder()
        c._send({"cmd": "play", "actions": [action_code(*a) for a in a1[:k1]], "repeat": rep_, "stop_clear": False})
        r1 = read_raw(c)
        r2, left, plays = r1, [action_code(*a) for a in a1[k1:]], 1
        while left:   # play stops at every room change: go on with the rest (as the teacher's restores never need)
            c._send({"cmd": "play", "actions": left, "repeat": rep_, "stop_clear": False})
            played = None
            while True:
                line = c._read_line()
                if line.startswith(b'L '):
                    r2 = c._read_exact(int(line.split(b' ')[1]))
                    break
                msg = json.loads(line.decode('utf-8'))
                if msg.get('type') == 'error':
                    raise RuntimeError(msg.get('msg'))
                if msg.get('cmd') == 'play':
                    played = int(msg['played'])
            plays += 1
            left = left[played:] if played else []
        o_play = dec.decode(r1)
        rec['play'] = dict(at_change=digest(dec.decode(r1)) == digest(LeanDecoder().decode(t1[k1])),
                           at_end=digest(dec.decode(r2)) == digest(LeanDecoder().decode(t1[-1])),
                           digest_end=c.lua('return ABPGX_DIGEST(false)') == d1, stage=int(o_play.room[5]),
                           plays=plays)
        c.close()
    # the walker itself: the same policy from the same state (its forks changed nothing)
    rng_w, phase_w, same = np.random.default_rng(seed), [None], True
    for j in range(len(a1)):
        a = policy(o, rng_w, phase_w)
        o, _ = pair.record(step(walker, a, rep_))
        same = same and a == a1[j] and digest(o) == digest(LeanDecoder().decode(t1[j + 1]))
    rec['walker_equal'] = same
    rec['walker_stage'] = int(o.room[5])
    # the next floor: Curse of the Lost hides the map; Curse of the Blind hides a pedestal's item
    if o.room[5] != stage0 and not o.dead:
        walker.lua("Game():GetLevel():AddCurse(4, false) return 'ok'")
        for _ in range(35):
            o, _ = pair.record(step(walker, (0, 0, 0, 0, 0), rep_))
        lost = dict(map_rooms=len(o.map), row_map_cells=int(pair.last_row['map'][0].sum()),
                    pinv_lost=float(pair.last_row['pinv'][0][9]))
        walker.lua("Game():GetLevel():AddCurse(64, false) return 'ok'")
        walker.lua(SPAWN_LUA.format(variant=100, sub=1, dx=0, dy=120))
        for _ in range(6):
            o, _ = pair.record(step(walker, (0, 0, 0, 0, 0), rep_))
        r = pair.last_row
        ped = [int(r['ent_item'][0][j][0]) for j in range(int(r['n_ent'][0]))
               if r['ent_id'][0][j][0] == 5 and r['ent_id'][0][j][1] == 100]
        rec['curses'] = dict(lost=lost, blind_pedestal_ids=ped, blind_unknown=ITEM_UNKNOWN in ped,
                             pinv_blind=float(r['pinv'][0][11]))
    nat = native_obs(walker)
    li = walker._send({"cmd": "lean_items"}) or walker._recv()
    rec['native'] = dict(checks=nat.get('checks'), mismatches=nat.get('mismatches'), fallbacks=nat.get('fallbacks'),
                         first=nat.get('first'))
    rec['inventory_check'] = dict(scans=li.get('scans'), checks=li.get('checks'), misses=li.get('misses'),
                                  first_miss=li.get('first_miss'), ids=li.get('ids'))
    rec['fastrow'] = dict(records=pair.t, fast=pair.enc.fast, slow=pair.enc.slow, mismatch=pair.mismatch,
                          events={str(k): v for k, v in pair.counts.items()}, stages=sorted(pair.stages))
    walker.close()
    # restore-clone map check (items off as in floor mode / on)
    maps = {}
    for on in (False, True):
        t = template.fork(lean=True, alarm=300)
        t._send({"cmd": "lean_items", "enabled": on})
        t._recv()
        t._send({"cmd": "obs"})
        read_raw(t)
        for _ in range(40):
            read_raw_ = step(t, (int(np.random.default_rng(0).integers(9)), 0, 0, 0), rep_)
        park = t.fork(tag='park', alarm=0)
        c = park.fork(lean=True, alarm=300)
        c._send({"cmd": "play", "actions": [0] * 12, "repeat": rep_, "stop_clear": False})
        raw = read_raw(c)
        maps['items_on' if on else 'items_off'] = dict(map_block=bool(raw[12] & 8), inventory_block=bool(raw[12] & 16))
        for x in (c, park, t):
            x.close()
    rec['restore_clone_first_obs'] = maps
    for x in parked:
        x.close()
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147600000:4')
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--port', type=int, default=29100)
    p.add_argument('--name', default='flp')
    p.add_argument('--out', required=True)
    p.add_argument('--items', action='store_true', help='2026-10-06: the items / run scenario (module docstring)')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    os.environ.update(FORK_ENV)
    if args.items:
        os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
        os.environ['ISAAC_RL_INV_CHECK'] = "1"
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    first, count = (int(v) for v in args.seeds.split(':'))
    inst = Instance(args.name, args.port, gx, spec)
    rng = np.random.default_rng(7)
    report = []
    parked = []
    try:
        if args.items:
            from isaac_bridge.tok_obs import fast_row_function
            fn = fast_row_function(default_preload())
            for seed in range(first, first + count):
                rec = items_seed(inst, seed, args, fn)
                report.append(rec)
                print(json.dumps(rec, default=str), flush=True)
            (out / 'items.json').write_text(json.dumps(report, default=str, indent=1))
            return
        for seed in range(first, first + count):
            inst.env.bridge.reset_mode = 'floor'
            t0 = time.perf_counter()
            _, info = inst.reset(seed)
            rec = dict(seed=seed, reset_s=round(time.perf_counter() - t0, 2), start_room=info['start_room'],
                       curses=info['floor_curses'], rooms={k: (v['type'], v['shape']) for k, v in info['floor'].items()})
            template = inst.env.bridge.fork(tag='template', alarm=0)
            parked.append(template)
            probe = template.fork(lean=True, alarm=300, reseed=seed % 1000 + 1)
            decoder = LeanDecoder()
            probe._send({"cmd": "obs"})
            o = read_lean(probe, decoder)
            rec['start'] = dict(room=o.room[:8], clear=o.clear, doors=[tuple(int(d[k]) for k in ('slot', 'open', 'locked',
                                                                                             'seen', 'target_type'))
                                                                  for d in o.doors],
                                map=[tuple(int(v) for v in r) for r in o.map])
            doors = [(int(d['slot']), float(d['x']), float(d['y'])) for d in o.doors if d['open'] and not d['locked']]
            probe.close()
            rec['walks'] = []
            for slot, x, y in doors:
                walker = template.fork(lean=True, alarm=300, reseed=seed % 1000 + 1)
                decoder = LeanDecoder()
                walker._send({"cmd": "obs"})
                o = read_lean(walker, decoder)
                st = EpisodeState(2250, floor=True, stall=450)
                row = np.zeros((1,), ROW)
                encode_row(o, st, row, 0)
                start_idx, walk = o.room[4], dict(slot=slot, door=(x, y), path=[])
                for t in range(1, 150):
                    pl = o.players[0]
                    if t <= 12 or t % 10 == 0:
                        walk['path'].append((t - 1, round(float(pl['x'])), round(float(pl['y'])), o.logic_frames))
                    o = lean_step(walker, decoder, (move_to(pl['x'], pl['y'], x, y), 0, 0, 0), 4)
                    done = encode_row(o, st, row, t)
                    if o.room[4] != start_idx:
                        walk.update(decisions=t, room=o.room[:8], clear=o.clear, events=int(row['events'][0]),
                                    done=int(done), monsters_hp=o.monsters_hp, entities=len(o.entities),
                                    doors=[tuple(int(d[k]) for k in ('slot', 'open', 'locked', 'seen', 'target_type'))
                                           for d in o.doors],
                                    map=[tuple(int(v) for v in r) for r in o.map],
                                    map_cells=int(row['map'][0, 0].sum()), map_visited=int(row['map'][0, 1].sum()),
                                    map_current=np.argwhere(row['map'][0, 3]).tolist(),
                                    exits=len(st.exits), n_ent=int(row['n_ent'][0]))
                        break
                else:
                    walk['room'] = None
                    rec['walks'].append(walk)
                    walker.close()
                    continue
                # clones of the room entry
                entry = walker.fork(tag='entry', alarm=0)
                parked.append(entry)
                acts = [(int(rng.integers(9)), int(rng.integers(5)), 0, 0) for _ in range(args.steps)]
                finals = []
                for _ in range(2):
                    c = entry.fork(lean=True, alarm=300)
                    d = LeanDecoder()
                    c._send({"cmd": "obs"})
                    oc = read_lean(c, d)
                    trace = [digest(oc)]
                    for a in acts:
                        if oc.dead or oc.room[4] != o.room[4]:
                            break
                        oc = lean_step(c, d, a, 4)
                        trace.append(digest(oc))
                    finals.append(trace)
                    c.close()
                walk['clone_steps'] = len(finals[0]) - 1
                walk['clones_equal'] = finals[0] == finals[1]
                # the walker itself goes on with the same actions: the parent is unchanged by its clones
                ow, same = o, True
                for j, a in enumerate(acts[:len(finals[0]) - 1]):
                    ow = lean_step(walker, decoder, a, 4)
                    same = same and digest(ow) == finals[0][j + 1]
                walk['walker_equal'] = same
                # a restore with one play
                k = len(finals[0]) - 1
                if k > 0:
                    c = entry.fork(lean=True, alarm=300)
                    oc = play_actions(c, LeanDecoder(), acts[:k], 4)
                    walk['play_equal'] = digest(oc) == finals[0][k]
                    c.close()
                rec['walks'].append(walk)
                walker.close()
            pids = descendants(inst.proc.pid)
            rec['parked'] = len(parked)
            rec['descendants'] = len(pids)
            rec['pss_mib'] = [round(pss_mib(q), 1) for q in pids]
            rec['root_pss_mib'] = round(pss_mib(inst.proc.pid), 1)
            report.append(rec)
            print(json.dumps(rec), flush=True)
        ok = [w for r in report for w in r['walks'] if w.get('room')]
        summary = dict(seeds=count, walks=sum(len(r['walks']) for r in report), entered=len(ok),
                       clones_equal=sum(bool(w.get('clones_equal')) for w in ok),
                       walker_equal=sum(bool(w.get('walker_equal')) for w in ok),
                       play_equal=sum(bool(w.get('play_equal')) for w in ok),
                       new_room_events=sum(bool(w['events'] & 2) for w in ok))
        print('SUMMARY', json.dumps(summary))
        (out / 'probe.json').write_text(json.dumps(dict(summary=summary, seeds=report), default=str))
    finally:
        for c in parked:
            try:
                c.close()
            except OSError:
                pass
        inst.close()


if __name__ == '__main__':
    main()
