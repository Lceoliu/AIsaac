"""The bridge's privileged frames (abp-0.2.19, tok_priv) on the real engine, per seed. Needs an instance; read-only for the
game. Checks, with a JSON line per seed and a summary:
  - off is off: a clone that did not ask gets no frame, before or after a clone of it asked; a clone of a connection that
    asked starts without (`fork_clone`);
  - the visibility audit at the source: two clones of one state, same reseed, same actions for --steps decisions, one with
    the frames, one without: every lean payload equal byte for byte, and the same in reverse order of asking (the frames
    change nothing in the game: no RNG, no state), through room changes when the scripted walk finds a door;
  - the content: every room of the level's list on the map (with its secret and super secret rooms, which the policy's
    minimap does not show), the map's current room == the lean map's, the frame's total HP == the lean totals' monsters_hp,
    NPC HP plausible against a Lua query, pedestals' true ids under Curse of the Blind (the curse added, a pedestal
    spawned), a layout block at each room change and none in between, leftovers remembered when a room is left;
  - frames only on a change (the first after the command always): the record kept without a frame still agrees with a Lua
    query at every record of a fight, a pickup spawned shows, frames per record reported;
  - the time: frame bytes, the worker's decode, and a worker's step (step, read, FastRow-like decode, the record) with / without
    frames, alternating, the same actions: --fight of them with monsters spawned (the costly case), the walk's share without.
usage (bridge's python dir, PYTHONPATH=.): python abplus_probe_priv.py --groups-file ../abplus/catalog/scaling2_groups.json --seeds 2147600000:4 --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV, action_code  # noqa: F401
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean_raw
from isaac_bridge.tok_obs import SHAPE_CELLS
from isaac_bridge.tok_priv import PRIV, PrivDecoder, priv_record
from isaac_bridge.tok_sampler import step_action

MOVES = {(0, 0): 0, (0, -1): 1, (1, -1): 2, (1, 0): 3, (1, 1): 4, (0, 1): 5, (-1, 1): 6, (-1, 0): 7, (-1, -1): 8}
ROOMS_LUA = ("local rooms = Game():GetLevel():GetRooms() local o = {} for i = 0, rooms.Size - 1 do local r = rooms:Get(i) "
             "o[#o + 1] = r.GridIndex .. ':' .. r.Data.Type .. ':' .. r.Data.Shape .. ':' .. r.DisplayFlags end return table.concat(o, ' ')")
HP_LUA = ("local o = {} for _, e in ipairs(Isaac.GetRoomEntities()) do if e.Type >= 10 and e.Type < 1000 and e:ToNPC() "
          "and e.HitPoints > 0 and e:IsActiveEnemy(false) then o[#o + 1] = tostring(e.HitPoints) end end return table.concat(o, ' ')")
NPC_LUA = ("local r = Game():GetRoom() local p = Isaac.GetPlayer(0) for k = 1, 3 do local pos = r:FindFreeTilePosition(p.Position + "
           "Vector(120 * k - 240, 120), 0) Isaac.Spawn(10, 0, 0, pos, Vector(0, 0), nil) end local pos = r:FindFreeTilePosition("
           "p.Position + Vector(0, -120), 0) Isaac.Spawn(20, 0, 0, pos, Vector(0, 0), nil) return 'ok'")
BLIND_LUA = ("Game():GetLevel():AddCurse(64, false) local p = Isaac.GetPlayer(0) local pos = Game():GetRoom():FindFreeTilePosition("
             "p.Position + Vector(80, 0), 0) Isaac.Spawn(5, 100, 33, pos, Vector(0, 0), nil) return 'ok'")
FIGHT_LUA = ("local r = Game():GetRoom() local p = Isaac.GetPlayer(0) p:AddHearts(24) for k = 1, 8 do local pos = r:FindFreeTilePosition(p.Position + "
             "Vector(60 * k - 270, 160), 0) Isaac.Spawn(10, 0, 0, pos, Vector(0, 0), nil) end return 'ok'")
PICKUP_LUA = ("local p = Isaac.GetPlayer(0) local pos = Game():GetRoom():FindFreeTilePosition(p.Position + Vector(0, 120), 0) "
              "Isaac.Spawn(5, 20, 1, pos, Vector(0, 0), nil) return 'ok'")


def acts_of(seed, n):
    """Sticky random moves and shots: deterministic in the seed."""
    rng = np.random.default_rng(seed)
    out, cur = [], (0, 0)
    for _ in range(n):
        if rng.random() < 0.08:
            cur = (int(rng.integers(9)), int(rng.integers(5)))
        out.append((cur[0], cur[1], 0, 0, 0))
    return out


def start(c, priv):
    c.priv_frame = None
    if priv:
        c._send({"cmd": "priv", "enabled": True})
    c._send({"cmd": "obs"})
    return read_lean_raw(c)


def run_clone(template, reseed, acts, priv, rep=4):
    """(lean payloads, privileged frames) of a clone that plays acts."""
    c = template.fork(alarm=300, reseed=reseed, lean=True)
    raws, frames = [start(c, priv)], [c.priv_frame]
    c.priv_frame = None
    t0 = time.perf_counter()
    for a in acts:
        step_action(c, a, True, rep)
        raws.append(read_lean_raw(c))
        frames.append(c.priv_frame)
        c.priv_frame = None
    dt = time.perf_counter() - t0
    c.close()
    return raws, frames, dt


def worker_steps(template, reseed, acts, priv, spawn=None, rep=4):
    """(seconds, decisions, frames): what a worker does per decision: step, read, decode, the privileged record."""
    c = template.fork(alarm=300, reseed=reseed, lean=True)
    if spawn:
        c.lua(spawn)
    lean = LeanDecoder()
    lean.decode(start(c, priv))
    dec, slots = PrivDecoder(), np.zeros(2, PRIV)
    views = (slots[:1], slots[1:2])
    priv_record(dec, c.priv_frame, views, 0) if priv else None
    c.priv_frame = None
    n, frames = 0, 0
    t0 = time.perf_counter()
    for t, a in enumerate(acts):
        step_action(c, a, True, rep)
        o = lean.decode(read_lean_raw(c))
        if priv:
            frames += c.priv_frame is not None
            priv_record(dec, c.priv_frame, views, t + 1)
            c.priv_frame = None
        n += 1
        if o.players['dead'][0]:
            break
    dt = time.perf_counter() - t0
    c.close()
    return dt, n, frames


def door_walk(o, target):
    pl = o.players[0]
    sx = 0 if abs(target[0] - pl['x']) < 6 else (1 if target[0] > pl['x'] else -1)
    sy = 0 if abs(target[1] - pl['y']) < 6 else (1 if target[1] > pl['y'] else -1)
    return MOVES[(sx, sy)]


def walk_rooms(template, reseed, dec_rows, rec, steps=400):
    """A clone with frames walks through doors: a layout block at each room change only, leftovers when a room is left."""
    c = template.fork(alarm=300, reseed=reseed, lean=True)
    lean, pd = LeanDecoder(), PrivDecoder()
    raw = start(c, True)
    o = lean.decode(raw)
    row = np.zeros(1, PRIV)
    pd.decode(c.priv_frame, row)
    c.priv_frame = None
    v0 = int(row['map_ver'][0])
    no_frame = 0
    c.lua(PICKUP_LUA)   # a coin left behind in the start room
    changes, layouts, room0 = 0, 0, o.room[4]
    room, clear = o.room[4], o.clear
    left_seen = 0.0
    first_cur = row['map'][0][3].copy()
    maps_equal = bool((row['map'][0][3] == first_cur).all())
    for _ in range(steps):
        doors = o.doors
        d = doors[doors['open'] != 0] if len(doors) else doors
        if not len(d):
            move = 0
        else:
            j = int(np.argmin(np.hypot(d['x'] - o.players[0]['x'], d['y'] - o.players[0]['y'])))
            move = door_walk(o, (float(d['x'][j]), float(d['y'][j])))
        step_action(c, (move, 1, 0, 0, 0), True, 4)
        raw = read_lean_raw(c)
        o = lean.decode(raw)
        ver_before = int(row['map_ver'][0])
        if c.priv_frame is not None:
            pd.decode(c.priv_frame, row)
        else:
            no_frame += 1
        c.priv_frame = None
        moved = o.room[4] != room or o.clear != clear
        layouts += int(row['map_ver'][0]) != ver_before
        if moved != (int(row['map_ver'][0]) != ver_before):
            rec.setdefault('layout_mismatch', []).append((int(o.room[4]), bool(o.clear)))
        if o.room[4] != room:
            changes += 1
            room = o.room[4]
            left_seen = max(left_seen, float(row['scal'][0][15:24].sum()))
            # the frame's current room == the lean map's
            cur = row['map'][0][3]
            for r in o.map:
                if r['flags'] & 4:
                    maps_equal = maps_equal and cur[int(r['idx']) // 13, int(r['idx']) % 13] == 1
        clear = o.clear
        if o.players['dead'][0]:
            break
    rec['walk'] = dict(steps_without_frame=no_frame, room_changes=changes, layouts=layouts, map_versions=int(row['map_ver'][0]) - v0, left_seen=round(left_seen, 2),
                       current_room_matches_lean_map=bool(maps_equal))
    c.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147600000:4')
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--fight', type=int, default=150)
    p.add_argument('--repeat', type=int, default=4)
    p.add_argument('--out', required=True)
    p.add_argument('--port', type=int, default=33720)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(FORK_ENV)
    os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
    spec = load_spec(argparse.Namespace(groups_file=a.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0, frames_per_decision=4,
                  start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    inst = Instance('privprobe', a.port, gx, spec)
    s0, n = (int(v) for v in a.seeds.split(':'))
    all_ok = True
    times = []
    times_w = []
    try:
        for seed in range(s0, s0 + n):
            inst.env.bridge.reset_mode = 'floor'
            inst.reset(seed)
            template = inst.env.bridge.fork(tag='template', alarm=0)
            rec = dict(seed=seed)
            acts = acts_of(seed, a.steps)
            r_off, f_off, t_off = run_clone(template, seed % 1000 + 1, acts, False)
            r_on, f_on, t_on = run_clone(template, seed % 1000 + 1, acts, True)
            r_off2, f_off2, t_off2 = run_clone(template, seed % 1000 + 1, acts, False)
            rec['frames_off'] = sum(f is not None for f in f_off) + sum(f is not None for f in f_off2)
            rec['frames_on'] = sum(f is not None for f in f_on)
            rec['first_has_frame'] = f_on[0] is not None
            rec['records'] = len(r_on)
            rec['lean_equal'] = r_on == r_off and r_off2 == r_off
            rec['first_diff'] = next((i for i, (x, y) in enumerate(zip(r_on, r_off)) if x != y), None)
            rec['s_per_decision_ms'] = dict(off=round(1000 * (t_off + t_off2) / 2 / a.steps, 3), on=round(1000 * t_on / a.steps, 3))
            times.append((t_off, t_off2, t_on))
            # a clone of a connection that asked starts without
            c = template.fork(alarm=300, reseed=7, lean=True)
            start(c, True)
            kid = c.fork(alarm=300, lean=True)
            kid.priv_frame = None
            start(kid, False)
            rec['clone_of_asked_has_frame'] = kid.priv_frame is not None
            kid.close()
            # content
            c.priv_frame = None
            c._send({"cmd": "obs"})
            raw = read_lean_raw(c)
            pd, row = PrivDecoder(), np.zeros(1, PRIV)
            c._send({"cmd": "priv", "enabled": True})   # (a fresh frame with its layout)
            c._send({"cmd": "obs"})
            raw = read_lean_raw(c)
            lean_dec = LeanDecoder()
            lean = lean_dec.decode(raw)
            pd.decode(c.priv_frame, row)
            c.priv_frame = None
            rooms = [tuple(int(x) for x in s.split(':')) for s in c.lua(ROOMS_LUA).split()]
            m = row['map'][0]
            present = sum(any(m[0][(g + c_) // 13, (g + c_) % 13] == 1 for c_ in SHAPE_CELLS.get(sh, (0,)))
                          for g, _, sh, _ in rooms if 0 <= g < 169)   # (shape 9 does not cover its GridIndex)
            rec['rooms'] = dict(listed=sum(0 <= r[0] < 169 for r in rooms), off_grid=sum(not 0 <= r[0] < 169 for r in rooms), on_map=present, secret=sum(r[1] == 7 for r in rooms),
                                super_secret=sum(r[1] == 8 for r in rooms), hidden_not_displayed=sum(r[3] == 0 for r in rooms),
                                ch_secret=int(m[8].sum()), ch_super=int(m[9].sum()))
            rec['hp'] = dict(frame_total=round(float(row['scal'][0][0]) * 50, 2), lean_total=round(float(lean.monsters_hp), 2))
            # Curse of the Blind: the true id in the frame, ITEM_UNKNOWN stays the policy's
            c.lua(BLIND_LUA)
            for _ in range(2):
                step_action(c, (0, 0, 0, 0, 0), True, 4)
                raw = read_lean_raw(c)
                if c.priv_frame is not None:
                    pd.decode(c.priv_frame, row)
                c.priv_frame = None
            rec['blind'] = dict(items=row['items'][0].tolist()[:3], blind_flag=float(row['scal'][0][5]))
            # monsters: Gapers and a boss spawned; the frame's HP against the lean totals and a Lua query, as they fight
            c.lua(NPC_LUA)
            bad_hp, seen, fight_frames = 0, 0, 0
            for k_ in range(30):
                step_action(c, (0, int(k_ % 4) + 1, 0, 0, 0), True, 4)
                raw = read_lean_raw(c)
                lean = lean_dec.decode(raw)
                fight_frames += c.priv_frame is not None
                if c.priv_frame is not None:
                    pd.decode(c.priv_frame, row)
                c.priv_frame = None
                if lean.monsters_hp > 0:
                    seen += 1
                    hp_lua = sorted((float(v) for v in c.lua(HP_LUA).split()), reverse=True)
                    got = (row['hp'][0] * 50.0).tolist()[:len(hp_lua)]
                    bad_hp += abs(float(row['scal'][0][0]) * 50 - lean.monsters_hp) > 0.01 or \
                        any(abs(x - y) > 0.01 for x, y in zip(got, hp_lua))
            coins0 = float(row['scal'][0][7])
            c.lua(PICKUP_LUA)
            for _ in range(2):
                step_action(c, (0, 0, 0, 0, 0), True, 4)
                raw = read_lean_raw(c)
                if c.priv_frame is not None:
                    pd.decode(c.priv_frame, row)
                c.priv_frame = None
            rec['pickup_shows'] = bool(float(row['scal'][0][7]) > coins0)
            rec['hp_fight'] = dict(records=seen, frames=int(fight_frames), mismatches=int(bad_hp), last_total=round(float(row['scal'][0][0]) * 50, 2),
                                   monsters=round(float(row['scal'][0][1]) * 16), boss_hp=round(float(row['scal'][0][2]) * 500, 1),
                                   boss_max=round(float(row['scal'][0][3]) * 500, 1), hp=[round(float(v) * 50, 1) for v in row['hp'][0][:4]])
            c.close()
            walk_rooms(template, seed % 1000 + 2, row, rec)
            sizes = [len(f) for f in f_on if f is not None]
            rec['frame_bytes'] = dict(mean=round(float(np.mean(sizes)), 1), max=int(np.max(sizes)), per_record=round(len(sizes) / len(f_on), 3))
            t0 = time.perf_counter()
            dec = PrivDecoder()
            for f in f_on:
                if f is not None:
                    dec.decode(f, row)
            rec['decode_us'] = round(1e6 * (time.perf_counter() - t0) / max(len(sizes), 1), 1)
            # the worker's cost, alternating: a walk's actions (mostly quiet rooms), and with monsters spawned
            t_w = {False: [], True: []}
            t_f = {False: [], True: []}
            fr_w = fr_f = 0
            for k_ in range(a.repeat):
                for on_ in ((False, True) if k_ % 2 == 0 else (True, False)):
                    dt_, n_, f_ = worker_steps(template, seed % 1000 + 1, acts, on_)
                    t_w[on_].append(dt_ / max(n_, 1))
                    fr_w = f_ / max(n_, 1) if on_ else fr_w
                    dt_, n_, f_ = worker_steps(template, seed % 1000 + 1, acts[:a.fight], on_, FIGHT_LUA)
                    t_f[on_].append(dt_ / max(n_, 1))
                    fr_f = f_ / max(n_, 1) if on_ else fr_f
            rec['worker_ms'] = dict(quiet_off=round(1000 * float(np.median(t_w[False])), 4), quiet_on=round(1000 * float(np.median(t_w[True])), 4),
                                    quiet_frames=round(fr_w, 3), fight_off=round(1000 * float(np.median(t_f[False])), 4),
                                    fight_on=round(1000 * float(np.median(t_f[True])), 4), fight_frames=round(fr_f, 3))
            times_w.append((rec['worker_ms']['quiet_off'], rec['worker_ms']['quiet_on'], rec['worker_ms']['fight_off'], rec['worker_ms']['fight_on']))
            ok = rec['frames_off'] == 0 and rec['first_has_frame'] and 0 < rec['frames_on'] <= rec['records'] and rec['lean_equal'] and not rec['clone_of_asked_has_frame'] \
                and rec['pickup_shows'] \
                and rec['rooms']['on_map'] == rec['rooms']['listed'] and rec['blind']['items'][0] == 33 and rec['blind']['blind_flag'] == 1.0 \
                and abs(rec['hp']['frame_total'] - rec['hp']['lean_total']) < 0.01 and not rec.get('layout_mismatch') \
                and rec['hp_fight']['records'] > 0 and rec['hp_fight']['mismatches'] == 0
            rec['ok'] = bool(ok)
            all_ok = all_ok and ok
            print(json.dumps(rec), flush=True)
            template.close()
        t = np.array(times)
        w = np.array(times_w)
        print(json.dumps(dict(summary=True, all_ok=bool(all_ok), off_s=round(float(t[:, :2].mean()), 3), on_s=round(float(t[:, 2].mean()), 3),
                              ratio=round(float(t[:, 2].mean() / t[:, :2].mean()), 4),
                              worker_quiet_ratio=round(float(w[:, 1].sum() / w[:, 0].sum()), 4),
                              worker_fight_ratio=round(float(w[:, 3].sum() / w[:, 2].sum()), 4))), flush=True)
    finally:
        inst.close()
    raise SystemExit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
