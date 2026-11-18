"""visible-v3 native encoder check (package C, 2026-11-17): the check mode (ISAAC_RL_VISIBLE_V3=3) on AB+ -- every lean
payload carries both the Lua raw block (flag 32) and the native frame (flag 64, abp_native_v3); here every record's
native frame is compared with the reference (visible_producer.observation_frame_v3 of the raw block) section by section
(--sections: the stages done so far: canvas, P, local, E, N, S, minimap; header). Lean clones stepped with step lines
(the sampler's path) from whole-floor starts over --seeds, --steps decisions each (restarting at an episode end).
Counts per section: compared, equal, the first differences (record, float index, native, reference); the bridge's
native / fallback counters; and per decision the game CPU (modes 0 v2, 1 Lua raw, 2 native) and the worker CPU
(LeanDecoder.decode + encode_row: v2 with its C encoder, against v3 native) for the timing comparison (--timing 1).
Floors at stage 1 + k mod --stages, a ChangeRoom into the floor's next room every --hop decisions.

usage (bridge python dir, PYTHONPATH=.): python abplus_probe_v3_native.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from abplus_probe_native_obs import read_raw   # noqa: E402
from goexplore_abplus import default_bridge_lua, default_preload, load_spec   # noqa: E402
from isaac_bridge import schema_v3 as V3, visible_producer as VP   # noqa: E402
from isaac_bridge.abplus import FLOOR_LUA, FORK_ENV   # noqa: E402
from isaac_bridge.tok_obs import ROW, EpisodeState, encode_row   # noqa: E402
from isaac_bridge.abplus_goexplore import GxConfig, Instance   # noqa: E402
from isaac_bridge.abplus_lean import LeanDecoder   # noqa: E402

ENTER_LUA = "Game():ChangeRoom(%d) return 'ok'"
MOVES = [(3, 0), (3, 2), (5, 0), (5, 3), (7, 1), (1, 4), (0, 0), (2, 2), (4, 1), (6, 3), (8, 2), (0, 4)]


def sections(payload):
    f = np.frombuffer(payload, '<f4')
    p, e, n, s = (int(v) for v in f[2:6])
    at = V3.HEADER
    out = {'header': f[:V3.HEADER], 'frame': f}
    for name, size in (('canvas', V3.CANVAS_H * V3.CANVAS_W * V3.ROOM_WORDS), ('minimap', V3.MAP_SIDE * V3.MAP_SIDE),
                       ('P', p * V3.PF), ('local', p * V3.LOCAL), ('E', e * V3.EF), ('N', n * V3.NF), ('S', s * V3.SF)):
        out[name] = f[at:at + size]
        at += size
    return out


def compare(native, ref, names, stats, rec):
    a, b = sections(native), sections(ref)
    for name in names:
        st = stats.setdefault(name, dict(compared=0, equal=0, first=[]))
        st['compared'] += 1
        x, y = a[name], b[name]
        if name == 'frame':   # (every byte)
            x, y = np.frombuffer(native, '<f4'), np.frombuffer(ref, '<f4')
        if x.shape == y.shape and x.tobytes() == y.tobytes():
            st['equal'] += 1
        elif len(st['first']) < 5:
            if x.shape != y.shape:
                st['first'].append(dict(record=rec, shape=[int(x.size), int(y.size)]))
            else:
                k = int(np.flatnonzero(x.view(np.uint32) != y.view(np.uint32))[0])
                st['first'].append(dict(record=rec, index=k, native=float(x[k]), reference=float(y[k])))


def classify_raw_only(raw, n_fb, why, fallbacks, allowed):
    """A decoded record without the native frame (mode 3): ('raw_only_allowed', why) only when it carries a raw v3
    block, the bridge's fallback counter moved by exactly one at it (n_fb: after, fallbacks: before), the reason is an
    allowed one, and the reference encoder encodes the block (2026-11-18, Codex round 4: as the worker's write_v3 --
    a block of another version or a short grid is refused there); else ('other_missing', why)."""
    if not VP.is_raw_v3(raw):
        return 'other_missing', f'{why}; no raw v3 block'
    if n_fb != fallbacks + 1:
        return 'other_missing', f'{why}; the fallback counter moved by {n_fb - fallbacks}'
    if why not in allowed:
        return 'other_missing', why
    try:
        VP.observation_frame_v3(raw)
    except Exception as exc:   # (VisibleV3Fatal, ValueError, a missing field: the worker would not take it)
        return 'other_missing', f'{why}; the reference encoder: {type(exc).__name__}: {str(exc)[:120]}'
    return 'raw_only_allowed', why


def set_mode(c, mode):
    c._send({"cmd": "visible_v3", "mode": mode})
    ack = c._recv()
    assert ack.get('type') == 'ok' and ack.get('mode') == mode, ack
    return ack


def game_cpu(c):
    return float(c.lua("return tostring(os.clock())"))


def hop_plan(rooms, cover):
    """The rooms to visit in turn: first those of a shape not seen yet, then a boss room (type 5), a shop (2), a devil
    room (14) not seen yet, then the rest in the floor's order."""
    seen_shapes, seen_types = cover['shape'], cover['room_type']
    first = [r for r in rooms if str(r[3]) not in seen_shapes]
    first += [r for r in rooms if r[1] in (5, 2, 14) and str(r[1]) not in seen_types and r not in first]
    return [r[0] for r in first] + [r[0] for r in rooms if r not in first]


COVER = dict(shape={}, room_type={}, priced_records=0, negative_price_records=0, entities=0, npc_records=0, bytes=0)


def walk(inst, seed, steps, mode, names, stats, rec0, hop=0, stage=1, cover=COVER):
    """A lean clone of seed's floor (at `stage` by the stage command) stepped `steps` decisions; every `hop` decisions
    it is sent into the floor's next room (ChangeRoom), so rooms with monsters come in. mode 3: every record compared;
    modes 0 / 2: the worker's side timed (LeanDecoder.decode + encode_row, thread CPU)."""
    b = inst.env.bridge
    b.reset_mode, b.character = 'floor', 0
    inst.reset(seed)
    if stage > 1:
        b.reset(phases=[[f'stage {stage}']], settle=2)
    rooms = [tuple(int(x) for x in row.split(',')[:4]) for row in b.lua(FLOOR_LUA).split(';')[2:]]   # idx, type, var, shape
    plan = hop_plan(rooms, cover)
    c = b.fork(tag='v3native', lean=True, alarm=1800)
    rec = rec0
    worker = 0.0
    configured = VP.VISIBLE_V3_MODE
    try:
        set_mode(c, mode)
        VP.VISIBLE_V3_MODE = mode   # (encode_row: a v3 worker refuses a record without a v3 frame, 2026-11-18)
        dec = LeanDecoder()
        st = EpisodeState(limit=10 ** 6, floor=True, stall=120)
        row = np.zeros(1, ROW)
        fallbacks = int(c.lua("return tostring(ABP_NATIVE_V3.fallbacks)")) if mode == 3 else 0
        c._send({"cmd": "obs"})
        payload = read_raw(c)
        cpu0 = game_cpu(c)
        for t in range(steps):
            w0 = time.thread_time()
            o = dec.decode(payload)
            if mode in (0, 2):
                encode_row(o, st, row, t)
                worker += time.thread_time() - w0
            if mode == 3:
                # (2026-11-18, Codex round 3: every decoded record accounted for -- both blocks (compared), a raw-only
                # hand-back of an allowed reason (the bridge's fallback counter up by one at this record, its reason
                # read), or another missing frame (a failure); not the bridge's own counts, which include the payload
                # after the last decision)
                acct = stats.setdefault('records', dict(both=0, raw_only_allowed=0, other_missing=0, other_first=[]))
                if o.visible_v3_frame is None:
                    n_fb, why = c.lua("return tostring(ABP_NATIVE_V3.fallbacks) .. '|' .. "
                                      "tostring(ABP_NATIVE_V3.last_fallback)").split('|', 1)
                    kind, why = classify_raw_only(o.visible, int(n_fb), why, fallbacks, ALLOWED_FALLBACKS)
                    fallbacks = int(n_fb)
                    acct[kind] += 1
                    if kind != 'raw_only_allowed' and len(acct['other_first']) < 5:
                        acct['other_first'].append(dict(record=rec, raw_v3=VP.is_raw_v3(o.visible), reason=why))
                elif not VP.is_raw_v3(o.visible):
                    acct['other_missing'] += 1
                    if len(acct['other_first']) < 5:
                        acct['other_first'].append(dict(record=rec, raw_v3=False, reason='frame without a raw block'))
                else:
                    acct['both'] += 1
                    compare(o.visible_v3_frame, VP.observation_frame_v3(o.visible), names, stats, rec)
                    raw = o.visible
                    cover['shape'][str(raw['shape'])] = cover['shape'].get(str(raw['shape']), 0) + 1
                    cover['room_type'][str(raw['room_type'])] = cover['room_type'].get(str(raw['room_type']), 0) + 1
                    prices = [e['price'] for e in raw['entities'] if e['kind'] == 5 and e['price'] != 0]
                    if prices:
                        cover['priced_records'] += 1
                        cover['negative_price_records'] += any(v < 0 for v in prices)
                    cover['entities'] += len(raw['entities'])
                    cover['npc_records'] += any(e['kind'] == 6 for e in raw['entities'])
                    cover['bytes'] += len(o.visible_v3_frame)
            rec += 1
            if hop and t % hop == hop - 1 and plan:
                c.lua(ENTER_LUA % plan[(t // hop) % len(plan)])
            m, s_ = MOVES[(t * 7 + seed) % len(MOVES)]
            c._sock.sendall(b'S 4 %d %d %d 0\n' % (m, s_, int(t % 97 == 50)))
            payload = read_raw(c)
        cpu = game_cpu(c) - cpu0
        ack = set_mode(c, mode)
    finally:
        VP.VISIBLE_V3_MODE = configured
        c.close()
    return rec, cpu, ack, worker


ALLOWED_FALLBACKS = {'lethal hit cancelled'}   # (main: --allowed-fallbacks)


def failures(stats, fallback_reasons, allowed):
    """The run's verdict (2026-11-18, Codex rounds 2 / 3): None when every decoded record is accounted for -- both
    blocks, or a raw-only hand-back of an allowed reason; no other missing frame -- every compared section equal and
    compared once for each record with both blocks (> 0), and every reason the bridge reports allowed; else what
    failed."""
    acct = stats.get('records') or dict(both=0, raw_only_allowed=0, other_missing=0)
    parts = {k: v for k, v in stats.items() if k != 'records' and isinstance(v, dict)}
    bad = sorted(k for k, v in parts.items() if v['equal'] != v['compared'])
    short = sorted(k for k, v in parts.items() if v['compared'] != acct['both'])
    if 'frame' not in parts:   # (2026-11-18, round 4: the whole frame is the verdict, whatever --sections lists)
        short.append('frame (not compared)')
    empty = not parts or acct['both'] == 0
    unknown = sorted(set(fallback_reasons) - {r for r in allowed if r})
    if bad or short or empty or unknown or acct['other_missing']:
        return dict(differ=bad, compared_short=short, nothing_compared=empty, reasons_not_allowed=unknown,
                    other_missing=acct['other_missing'])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--seeds', default='2148000000:4')
    ap.add_argument('--steps', type=int, default=250)
    ap.add_argument('--sections', default='frame,header,canvas,minimap,P,local,E,N,S')
    ap.add_argument('--timing', type=int, default=1)
    ap.add_argument('--hop', type=int, default=25, help='decisions per room (ChangeRoom to the next room of the floor)')
    ap.add_argument('--stages', type=int, default=6, help='seed k plays stage 1 + k mod stages')
    ap.add_argument('--port', type=int, default=44540)
    ap.add_argument('--allowed-fallbacks', default='lethal hit cancelled',
                    help='native hand-back reasons that do not fail the run (comma-separated)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    # (2026-11-18, Codex round 3: the defaults here, not at import -- a test importing failures() leaves the
    # environment alone; the worker's configured mode follows the one the game is launched with)
    os.environ.setdefault('ISAAC_RL_VISIBLE_V3', '3')
    os.environ.setdefault('ISAAC_RL_NATIVE_VISIBLE', '1')   # (mode 0 = the v2 native path: G mode 1 + abp_visible.c)
    VP.VISIBLE_V3_MODE = VP.configured_mode(os.environ['ISAAC_RL_VISIBLE_V3'])
    ALLOWED_FALLBACKS.clear()
    ALLOWED_FALLBACKS.update(r for r in args.allowed_fallbacks.split(',') if r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # ('frame', every byte, is always compared -- the verdict's; --sections adds the per-section diagnostics)
    names = ['frame'] + [v for v in args.sections.split(',') if v and v != 'frame']
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    inst = Instance('v3native', args.port, gx, spec)
    stats, timing, wtiming, rec, acks = {}, {}, {}, 0, []
    first, count = (int(v) for v in args.seeds.split(':'))
    t0 = time.perf_counter()
    try:
        for seed in range(first, first + count):
            stage = 1 + (seed - first) % args.stages
            rec, _, ack, _ = walk(inst, seed, args.steps, 3, names, stats, rec, hop=args.hop, stage=stage)
            acks.append(dict(seed=seed, native=ack.get('native'), fallbacks=ack.get('fallbacks'),
                             last_fallback=ack.get('last_fallback'), available=ack.get('native_available'),
                             reasons=ack.get('reasons') if isinstance(ack.get('reasons'), dict) else {}))
            if args.timing:
                for mode in (0, 1, 2):
                    _, cpu, _, worker = walk(inst, seed, args.steps, mode, names, {}, 0, hop=args.hop, stage=stage,
                                             cover=dict(shape={}, room_type={}, priced_records=0,
                                                        negative_price_records=0, entities=0, npc_records=0, bytes=0))
                    timing.setdefault(mode, []).append(1000 * cpu / args.steps)
                    wtiming.setdefault(mode, []).append(1000 * worker / args.steps)
    finally:
        inst.kill()
    summary = dict(records=rec, sections=stats, bridge=acks, coverage=COVER,
                   frame_bytes_mean=round(COVER['bytes'] / max(1, (stats.get('records') or {}).get('both', 0)), 1),
                   fallback_reasons={k: sum(int((a.get('reasons') or {}).get(k, 0)) for a in acks)
                                     for k in sorted({k for a in acks for k in (a.get('reasons') or {})})},
                   game_cpu_ms_per_decision={f'mode{m}': round(float(np.mean(v)), 4) for m, v in timing.items()},
                   worker_cpu_ms_per_decision={f'mode{m}': round(float(np.mean(v)), 4) for m, v in wtiming.items()
                                               if m != 1},
                   seconds=round(time.perf_counter() - t0, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True))
    print('SUMMARY', json.dumps(summary, sort_keys=True), flush=True)
    fail = failures(stats, summary['fallback_reasons'], args.allowed_fallbacks.split(','))
    if fail:
        print('FAIL', json.dumps(fail), flush=True)
        sys.exit(1)
    print('PASS', flush=True)


if __name__ == '__main__':
    main()
