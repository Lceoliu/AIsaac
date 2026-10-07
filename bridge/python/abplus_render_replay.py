"""Real game footage of a recorded whole-floor episode (eval_tok.py --record, tok_record.py): the game's own rendering,
not the observation viewer.

One AB+ instance is launched in the rendering mode (abplus.MODES 'render': the render path draws real pixels with
Mesa's software OpenGL, FORK_ENV, on the host's X display; audio off: OpenAL's null backend, ABP_AL_STOPPED). It
resets the episode's floor exactly as the evaluation's root did (Instance.reset with reset_mode 'floor'), then does in
place what the evaluation's episode clone did right after its fork: the global MT reseeded with
seed % (2**31 - 1) + 1 (ABP_RESEED), lean mode on (native input), one observation. The recorded actions are then
replayed with the same one-decision lag (the step after record t applies the action decided at record t-1; the first
step is a no-op), frames_per_decision logic frames per step, a step ending early at a room change (the bridge's rule).
Every replayed record is encoded with tok_obs.encode_row and compared with the recorded one (player features, entity
rows, doors, hurt, done): the match report says whether the replay is the recorded game.

Frames: abp_turbo's frame grab (getenv ABP_GRAB_ON, analysis/scripts/abplus/abp_turbo.c) reads every <every>-th
presented frame back from the window's back buffer and writes it into a FIFO that ffmpeg encodes. The virtual clock
presents 60 frames per game second (logic runs on every other loop iteration), so --fps 30 keeps every second frame and
the video runs at exactly game speed whatever the replay's real speed. Frames are scaled by an integer factor with
nearest-neighbour sampling (480x270 x 4 = 1920x1080), h264 yuv420p.

usage (host, from a sandbox's python dir, PYTHONPATH=.):
  python abplus_render_replay.py --record ../runs/rec-c63-2147493000/rec --seed 2147493015 \
      --out ../runs/footage/floor_2147493015.mp4
  --from-record a --to-record b: a clip of records a..b (the whole replay still runs from the floor's start, the frames
  are grabbed from record a on; a clip that ends with the episode gets --tail-seconds of no-op steps after it).
  Without --out: the match check alone (no frame grab). --mode exact --stub-list <tools>/stub_render_g.txt: the same
  replay in the evaluation's own render-lite mode (a reference: is a divergence the rendering's or the replay's?).
  The report (<out>.json) has per record the logic frame, the player's position, and in the clip the video frame index
  at which the record's state is shown (frames_at).
  frames_at costs a lua command per record in the clip; a lua command marks the bridge's terrain block dirty, so the next
  lean observation resends it. The game is unchanged, but the floor-exit token (built from the terrain block's grid
  list) can then appear in an encoded record a few decisions earlier than in the recording (seen after a boss clear:
  'ent' differs, 'player' does not). The exact record comparison is the run without --out (no lua command per record).
Video frames vs game time: the game presents 60 frames per game second during play and also during room and floor
transitions (which take no logic frames), so the video is longer than the logic frames / 30 of the episode: it runs at
the speed a player sees.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.tok_obs import ROW, EpisodeState, encode_row
from isaac_bridge.tok_sampler import apply_instance_defaults, lean_step

WIDTH, HEIGHT = 480, 270   # run_instance.sh's window (options.ini WindowWidth / WindowHeight)


def load_episode(record, seed):
    record = Path(record)
    files = sorted(record.glob(f'ep*_{seed}.npz'))
    if len(files) != 1:
        raise SystemExit(f'{len(files)} episode files of seed {seed} in {record}')
    ep = dict(np.load(files[0]))
    ep['meta'] = json.loads(str(ep['meta']))
    run = json.loads((record / 'run.json').read_text())
    return files[0].name, ep, run


def recorded_row(ep, t, offsets):
    """The entity rows / ids of record t (stored concatenated, n_ent per record)."""
    a, b = offsets[t], offsets[t + 1]
    return ep['ent'][a:b], ep['ent_id'][a:b]


def compare(ep, t, row, offsets):
    """Which parts of replayed record t differ from the recorded one ('' when equal)."""
    bad = []
    if not np.array_equal(row['player'][0], ep['player'][t]):
        bad.append('player')
    n = int(row['n_ent'][0])
    ent, ids = recorded_row(ep, t, offsets)
    if n != len(ent) or not np.array_equal(row['ent'][0][:n], ent) or not np.array_equal(row['ent_id'][0][:n], ids):
        bad.append('ent')
    k = int(row['n_doors'][0])
    if k != int(ep['n_doors'][t]) or not np.array_equal(row['doors'][0][:k], ep['doors'][t][:k]):
        bad.append('doors')
    if float(row['hurt'][0]) != float(ep['hurt'][t]):
        bad.append('hurt')
    if int(row['done'][0]) != int(ep['done'][t]):
        bad.append('done')
    if int(row['events'][0]) != int(ep['events'][t]):
        bad.append('events')
    if not np.array_equal(row['grid'][0], ep['grids'][ep['grid_idx'][t]]):
        bad.append('grid')
    if not np.array_equal(row['map'][0], ep['maps'][ep['map_idx'][t]]):
        bad.append('map')
    return ','.join(bad)


def start_encoder(fifo, out, fps, scale, crf, log):
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{WIDTH}x{HEIGHT}', '-r', str(fps), '-i', str(fifo),
           '-vf', f'vflip,scale={WIDTH * scale}:{HEIGHT * scale}:flags=neighbor',
           '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           '-an', str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log)


def grab_on(bridge, every, fifo, tries=50):
    for _ in range(tries):
        answer = bridge.lua(f"return os.getenv('ABP_GRAB_ON:{every}:{WIDTH}:{HEIGHT}:{fifo}')")
        if answer:
            return answer
        time.sleep(0.1)   # the encoder has not opened the FIFO yet
    raise RuntimeError('frame grab could not open the FIFO (no reader, or a library without ABP_GRAB)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--record', required=True, help='eval_tok --record directory')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', default='', help='mp4 to write (default: no video)')
    p.add_argument('--report', default='', help='JSON report (default: <out>.json)')
    p.add_argument('--fps', type=int, default=30, choices=(30, 60))
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--crf', type=int, default=18)
    p.add_argument('--tail-seconds', type=float, default=2.0,
                   help='no-op steps played (and filmed) after the last record, e.g. the death animation')
    p.add_argument('--from-record', type=int, default=0, help='the clip starts at this record (decision)')
    p.add_argument('--to-record', type=int, default=-1, help='... and ends at this one (-1: the episode\'s end, then '
                                                             'the tail)')
    p.add_argument('--check-all', action='store_true', help='replay and compare the records after the clip too')
    p.add_argument('--real-death', action='store_true',
                   help='ISAAC_RL_BLOCK_LETHAL=0: the lethal hit is not cancelled by the bridge (it is during the '
                        'evaluation: the player stays alive with dead=true), so the game plays its death animation; '
                        'every hit before it is the same, the last record may differ')
    p.add_argument('--mode', default='render', choices=('render', 'exact'))
    p.add_argument('--stub-list', default='', help='ABP_STUB_LIST (exact mode reference: the evaluation used '
                                                   'stub_render_g.txt)')
    p.add_argument('--groups-file', default='../abplus/catalog/scaling2_groups.json')
    p.add_argument('--name', default='rend')
    p.add_argument('--port', type=int, default=29950)
    p.add_argument('--bridge-lua', default='')
    p.add_argument('--preload', default='')
    args = p.parse_args()

    name, ep, run = load_episode(args.record, args.seed)
    fpd = int(run['frames_per_decision'])
    run_mode = run.get('mode') == 'run'   # 2026-10-06: a run (on through the trapdoors) with or without items
    items = bool(run.get('items'))
    limit = int(round(float(run['run_seconds'] if run_mode else run['floor_seconds']) * 30 / fpd))
    stall = int(round(float(run['floor_stall_seconds']) * 30 / fpd))
    T = len(ep['t'])
    acts = ep['act'].astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(ep['n_ent'].astype(np.int64))])
    video = bool(args.out) and args.mode == 'render'
    report_path = Path(args.report or (args.out + '.json' if args.out else f'replay_{args.seed}.json'))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.update(FORK_ENV)
    apply_instance_defaults()
    if items:   # the evaluation's roots sent the inventory block (abp_bridge.lua lean_items)
        os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
    if args.real_death:
        os.environ['ISAAC_RL_BLOCK_LETHAL'] = '0'
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    gx = GxConfig(bridge_lua=args.bridge_lua or default_bridge_lua(), preload=args.preload or default_preload(),
                  al_stopped=True, nice=5, frames_per_decision=fpd, start_hp=6, bombs=1, mode=args.mode,
                  stub_list=args.stub_list)
    reseed = args.seed % (2 ** 31 - 1) + 1
    enc = fifo = log = None
    inst = None
    t_begin = time.perf_counter()
    out = dict(seed=args.seed, file=name, records=T, outcome=ep['meta'], mode=args.mode, reseed=reseed, fps=args.fps)
    try:
        inst = Instance(f'{args.name}0', args.port, gx, spec)
        inst.env.bridge.reset_mode = 'floor'
        inst.reset(args.seed)
        bridge = inst.env.bridge
        # what the evaluation's episode clone did after its fork (abp_bridge.lua fork_clone): reseed, lean mode
        if bridge.lua(f"return os.getenv('ABP_RESEED:{reseed}')") != '1':
            raise RuntimeError('ABP_RESEED unavailable')
        bridge._send({"cmd": "lean", "enabled": True})
        ack = bridge._recv()
        if ack.get('type') != 'ok' or not ack.get('enabled'):
            raise RuntimeError(f'lean mode: {ack}')
        decoder = LeanDecoder()
        bridge._send({"cmd": "obs"})
        obs = read_lean(bridge, decoder)
        first = max(0, args.from_record)
        last = T - 1 if args.to_record < 0 else min(args.to_record, T - 1)
        out.update(from_record=first, to_record=last)
        grabbing = False

        def grabbed():
            """Frames written so far (abp_turbo's grab counter; a lua command changes nothing in the game)."""
            return int(bridge.lua("return os.getenv('ABP_GRAB_STATUS')").split()[0].split('=')[1])

        if video:
            fifo = Path(args.out).resolve().with_suffix('.fifo')   # the game's working directory is another one
            if fifo.exists():
                fifo.unlink()   # a FIFO of an earlier run (no data in it)
            os.mkfifo(fifo)
            log = open(Path(args.out).with_suffix('.ffmpeg.log'), 'w')
            enc = start_encoder(fifo, args.out, args.fps, args.scale, args.crf, log)
        st = EpisodeState(limit, floor=True, stall=stall, run=run_mode, items=items)
        row = np.zeros(1, ROW)
        lf0 = obs.logic_frames
        lfs, mism, first_bad, positions, frames_at, parts = [], [], None, [], {}, {}
        action = (0,) * acts.shape[1]
        t = 0
        while True:
            done = encode_row(obs, st, row, t)
            lfs.append(int(obs.logic_frames - lf0))
            p0 = obs.players[0]
            positions.append((float(p0['x']), float(p0['y'])))
            if t < T:
                bad = compare(ep, t, row, offsets)
                if bad:
                    mism.append(t)
                    parts[t] = bad
                    if first_bad is None:
                        first_bad = dict(t=t, parts=bad, replay_player=row['player'][0][:4].tolist(),
                                         recorded_player=ep['player'][t][:4].tolist())
            if video and t == first:   # the clip starts with this record's picture
                out['grab_on'] = grab_on(bridge, 60 // args.fps, fifo)
                grabbing = True
            if grabbing:
                frames_at[t] = grabbed()   # the video frame showing record t's state comes next
            if video and grabbing and t >= last and last < T - 1:
                out['grab_off'] = bridge.lua("return os.getenv('ABP_GRAB_OFF')")
                grabbing = False
            if t >= T - 1 or done or (t >= last and not args.check_all):
                break
            if t >= 1:
                action = tuple(int(v) for v in acts[t - 1])
            obs = lean_step(bridge, decoder, action, fpd)
            t += 1
        out.update(replayed=t + 1, done=int(done), recorded_done=int(ep['done'][T - 1]), mismatches=len(mism),
                   first_mismatch=first_bad, mismatch_list=mism[:50], logic_frames=lfs, positions=positions,
                   frames_at=frames_at, mismatch_parts=dict(list(parts.items())[:200]),
                   player_mismatches=sum('player' in v for v in parts.values()))
        out['match_rate'] = round(1.0 - len(mism) / max(1, min(T, t + 1)), 6)
        # the tail (a clip that ends with the episode): no-op steps after the last record (the death animation, the
        # next floor's first moments)
        tail = int(round(args.tail_seconds * 30 / fpd)) if grabbing else 0
        died = args.real_death and bool(obs.dead)
        if died:
            # after a real death the game stops calling the bridge once the death animation is under way: the step
            # never answers while the game goes on presenting frames (the death animation, the death diary). The wait
            # is cut after a few real seconds and the video trimmed to tail_seconds after the last record below.
            bridge._sock.settimeout(8.0)
        played = 0
        try:
            for _ in range(tail):
                obs = lean_step(bridge, decoder, (0, 0, 0), fpd)
                played += 1
        except (TimeoutError, OSError):
            if not died:
                raise
            out['tail_cut'] = int(frames_at[last] + round(args.tail_seconds * 30))
        out['tail_steps'] = played
        out['end_logic_frames'] = int(obs.logic_frames - lf0)
        if grabbing and 'tail_cut' not in out:
            out['grab_off'] = bridge.lua("return os.getenv('ABP_GRAB_OFF')")
    except Exception as exc:
        out['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if inst is not None:
            inst.close()
        if enc is not None:
            if 'grab_on' not in out:
                enc.kill()   # it waits for a writer that never came
            try:
                enc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                enc.kill()
            out['ffmpeg_exit'] = enc.returncode
        if log is not None:
            log.close()
        if fifo is not None and fifo.exists():
            fifo.unlink()
        if out.get('tail_cut') and out.get('ffmpeg_exit') == 0:   # the death clip: kept up to tail_seconds
            full = Path(args.out).with_suffix('.untrimmed.mp4')
            os.replace(args.out, full)
            cut = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(full), '-frames:v', str(out['tail_cut']),
                                  '-c', 'copy', '-movflags', '+faststart', str(args.out)])
            out['trimmed'] = dict(frames=out['tail_cut'], exit=cut.returncode, untrimmed=str(full))
        out['seconds'] = round(time.perf_counter() - t_begin, 1)
        report_path.write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k not in ('logic_frames', 'positions', 'mismatch_list')}))


if __name__ == '__main__':
    main()
