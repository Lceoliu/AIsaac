"""Evaluation of a token-policy checkpoint (train_tok.py) on held-out seeds, by group.

Every seed is one episode: the group's room of that seed, full health, the episode clone reseeded with the seed
itself, the same time limit and the same one-decision action lag as in training. Actions are the policy's most likely
ones (--greedy) or sampled. Per group: how many rooms were cleared, lost by death or by the time limit, half hearts
lost per room, rooms cleared without a hit, game seconds per cleared room.
--random evaluates uniformly random actions instead of a checkpoint (the floor of the scale).

--mode floor: every seed is one whole Basement I floor from a fresh run's start (isaac_bridge/tok_floor.py; no
archive, nothing scripted): floors cleared (the next floor reached), deaths, time-outs, boss rooms cleared, rooms
cleared and entered, half hearts lost per floor and per cleared room, game seconds per cleared floor.
--mode run (2026-10-06): every seed is one run from Basement I on through the trapdoors (tok_floor run mode, limit
--run-seconds): the deepest stage reached (histogram), floors cleared per run, the stage of each death, game time,
and (items) collectibles gained, active item and pill / card uses. An items checkpoint (config items) evaluates with
the item-aware observation and its item / pill heads; --items forces it for a random policy.

usage (from the bridge's python dir, PYTHONPATH=.):
  python eval_tok.py --checkpoint <run>/checkpoints/last.pt --groups-file ../abplus/catalog/scaling2_groups.json \
      --groups normal:6,boss:4,normal_big:2 --seeds 2147490000:128 --greedy --out <dir>
  python eval_tok.py --checkpoint ... --groups-file ... --mode floor --workers 8 --seeds 2147490000:128 --out <dir>
  --characters "0,7,13" (2026-10-08, floor / run modes): the floors are played as these PlayerTypes, each seed's
  character drawn from the seed alone (tok_floor.character_of_seed); summary 'by_character'. Default: Isaac.
  --record <dir>: every record of every episode and the actions decided, one npz per episode (tok_record.py; the HTML
  replay viewer: abplus_tok_replay_build.py).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.tok_obs import EV_ACTIVE, EV_BOSS, EV_CLEARED, EV_EXIT, EV_ITEM, EV_NEW_ROOM, EV_PILL
from isaac_bridge.tok_sampler import TokSampler, TokSamplerConfig


def main():
    # torch is imported here, not at the top: the sampler's workers are spawned and import this module too
    import torch
    from isaac_bridge.tok_policy import NOOP, TokPolicy, load_compatible, to_batch
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='')
    p.add_argument('--random', action='store_true')
    p.add_argument('--groups-file', required=True)
    p.add_argument('--groups', default='normal:6,boss:4,normal_big:2', help='group:workers, ...')
    p.add_argument('--mode', default='room', choices=('room', 'floor', 'run'))
    p.add_argument('--run-seconds', type=float, default=1800.0, help='run mode: game seconds a run may last')
    p.add_argument('--items', action='store_true', help='item-aware observation (on by itself for an items checkpoint)')
    p.add_argument('--workers', type=int, default=8, help='floor mode: instances')
    p.add_argument('--seeds', default='2147490000:128', help='first seed:count, the same block for every group')
    p.add_argument('--greedy', action='store_true')
    p.add_argument('--frames-per-decision', type=int, default=4)
    p.add_argument('--episode-seconds', type=float, default=90.0)
    p.add_argument('--floor-seconds', type=float, default=480.0, help='as in training: the policy sees the share used')
    p.add_argument('--floor-stall-seconds', type=float, default=60.0)
    p.add_argument('--port', type=int, default=28500)
    p.add_argument('--name', default='tke')
    p.add_argument('--stub-list', default='')
    p.add_argument('--out', required=True)
    p.add_argument('--dump-final', action='store_true', help="save each episode's last record (final_rows.npy)")
    p.add_argument('--record', default='', help='save every record of every episode (and the actions decided) to '
                                                'this directory, one compressed npz per episode (tok_record.py)')
    p.add_argument('--characters', default='',
                   help='2026-10-08, floor / run modes: the PlayerTypes the floors are played as (train_tok '
                        '--characters syntax; each seed\'s character is drawn from the seed alone: '
                        'tok_floor.character_of_seed). Default: Isaac')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    run = args.mode == 'run'
    floor = args.mode == 'floor' or run
    if args.characters:   # 2026-10-08 (a bad spec fails here, not in the workers)
        if not floor:
            p.error('--characters needs --mode floor/run')
        from isaac_bridge.tok_floor import parse_characters
        try:
            parse_characters(args.characters)
        except ValueError as exc:
            p.error(str(exc))
    ck = None
    if not args.random:
        ck = torch.load(args.checkpoint, map_location=os.environ.get('TOK_EVAL_DEVICE', 'cpu'))
    items = bool(args.items or (ck is not None and ck.get('config', {}).get('items')))
    names, assign, eval_seeds = [], [], []
    first, count = (int(v) for v in args.seeds.split(':'))
    groups = f'{"run" if run else "floor"}:{args.workers}' if floor else args.groups
    for gi, part in enumerate(groups.split(',')):
        name, _, k = part.partition(':')
        k = int(k or 1)
        names.append(name)
        seeds = list(range(first, first + count))
        for j in range(k):
            assign.append(gi)
            eval_seeds.append(seeds[j::k])
    specs = [load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal' if floor else n, tasks='',
                                          seconds=0.0)) for n in names]
    stub = args.stub_list or str(Path(default_preload()).parent / 'stub_render_g.txt')
    cfg = TokSamplerConfig(workers=len(assign), specs=specs, assign=assign,
                           frames_per_decision=args.frames_per_decision, episodes_per_state=1,
                           episode_seconds=args.episode_seconds, eval_seeds=eval_seeds, name=args.name, port=args.port,
                           mode=args.mode, floor_seconds=args.floor_seconds,
                           floor_stall_seconds=args.floor_stall_seconds, items=items, run_seconds=args.run_seconds,
                           characters=args.characters if floor else '',
                           bridge_lua=default_bridge_lua(), preload=default_preload(),
                           stub_list=stub if Path(stub).is_file() else '', nice=0)
    # CPU by default: a second CUDA process on the trainer's GPU can crash a trainer that replays CUDA graphs (B12)
    device = torch.device(os.environ.get('TOK_EVAL_DEVICE', 'cpu'))
    model, info = None, dict(policy='random')
    rng = np.random.default_rng(0)
    if not args.random:
        model = TokPolicy(**dict(ck['config'], items=items)).to(device).eval()   # --items on an items-free one: padded
        changed = load_compatible(model, ck['model'])
        info = dict(policy=str(args.checkpoint), update=ck.get('update'), decisions=ck.get('decisions'),
                    greedy=args.greedy, weights_not_exact=changed)
    sampler = TokSampler(cfg)
    n = cfg.workers
    recorder = None
    if args.record:
        from tok_record import Recorder
        recorder = Recorder(args.record, n, dict(info, mode=args.mode, seeds=args.seeds, groups=names,
                                                 frames_per_decision=args.frames_per_decision,
                                                 floor_seconds=args.floor_seconds,
                                                 floor_stall_seconds=args.floor_stall_seconds,
                                                 episode_seconds=args.episode_seconds, items=items,
                                                 run_seconds=args.run_seconds), n_heads=5 if items else 3)
    nh = 5 if items else 3
    pending = np.zeros((n, nh), np.int64)
    hurt = np.zeros(n)
    events = np.zeros((n, 3), np.int64)   # rooms cleared, rooms entered, boss rooms cleared
    extra = np.zeros((n, 6), np.int64)    # run / items: exits, collectibles, active uses, pill uses, stage0, deepest
    char0 = np.zeros(n, np.int64)         # 2026-10-08: ROW pchar at the episode's first record
    episodes = []
    finals = []
    t0 = time.perf_counter()
    try:
        while sampler.live:
            idx, slots = sampler.poll(want=max(1, n // 2), max_wait=0.0005)
            if not idx:
                continue
            rows = sampler.rows[slots]
            act_idx, act_rows = [], []
            for j, i in enumerate(idx):
                r = rows[j]
                if r['first']:
                    pending[i], hurt[i], events[i], extra[i] = 0, 0.0, 0, 0
                    extra[i, 4] = extra[i, 5] = int(r['stage'])
                    char0[i] = int(r['pchar'])
                if recorder is not None:
                    recorder.add(i, r)
                hurt[i] += float(r['hurt'])
                ev = int(r['events'])
                events[i] += (bool(ev & EV_CLEARED), bool(ev & EV_NEW_ROOM), bool(ev & EV_BOSS))
                extra[i, :4] += (bool(ev & EV_EXIT), bool(ev & EV_ITEM), bool(ev & EV_ACTIVE), bool(ev & EV_PILL))
                extra[i, 5] = max(extra[i, 5], int(r['stage']))
                if r['done']:
                    if args.dump_final:
                        finals.append(r.copy())
                    episodes.append(dict(group=names[int(r['group'])], seed=int(r['seed']), done=int(r['done']),
                                         decisions=int(r['t']), hurt=float(hurt[i]), rooms=int(events[i, 0]),
                                         entered=int(events[i, 1]), boss=int(events[i, 2])))
                    if run or items:
                        episodes[-1].update(exits=int(extra[i, 0]), items=int(extra[i, 1]), uses=int(extra[i, 2]),
                                            pills=int(extra[i, 3]), stage0=int(extra[i, 4]), stage=int(extra[i, 5]))
                    if args.characters:   # 2026-10-08
                        episodes[-1]['char0'] = int(char0[i])
                    if recorder is not None:
                        recorder.finish(i, episodes[-1])
                    continue
                act_idx.append(i)
                act_rows.append(j)
            if act_idx:
                if model is None:
                    actions = np.stack([rng.integers(9, size=len(act_idx)), rng.integers(5, size=len(act_idx))]
                                       + [np.zeros(len(act_idx), np.int64)] * (nh - 2), 1)
                else:
                    with torch.inference_mode():
                        actions = model.act(to_batch(rows[act_rows], pending[act_idx], device),
                                            greedy=args.greedy)[0].cpu().numpy()
                for j, i in enumerate(act_idx):
                    pending[i] = actions[j]
                    sampler.actions[i, :nh] = actions[j]
                    if recorder is not None:
                        recorder.act(i, actions[j])
            sampler.reply(idx)
        seconds = round(time.perf_counter() - t0, 1)
        game_s = sum(e['decisions'] for e in episodes) * args.frames_per_decision / 30
        summary = dict(info, mode=args.mode, seeds=args.seeds, frames_per_decision=args.frames_per_decision,
                       episode_seconds=(args.run_seconds if run else args.floor_seconds) if floor
                       else args.episode_seconds, seconds=seconds,
                       game_hours=game_s / 3600, x_real_time=game_s / max(seconds, 1e-9),
                       errors=float(sampler.stats[:, 5].sum()), empty=float(sampler.stats[:, 9].sum()), groups={})
        if run or items:
            summary['items'] = items
        for name in names:
            eps = [e for e in episodes if e['group'] == name]
            m = max(len(eps), 1)
            wins = [e for e in eps if e['done'] == 1]
            if run:
                stages = [e['stage'] for e in eps]
                deaths = [e for e in eps if e['done'] == 2]
                summary['groups'][name] = dict(
                    episodes=len(eps), death=len(deaths) / m, timeout=sum(e['done'] == 3 for e in eps) / m,
                    stage_reached={str(k): stages.count(k) for k in sorted(set(stages))},
                    floors_cleared=sum(e['exits'] for e in eps) / m,
                    death_stage={str(k): [e['stage'] for e in deaths].count(k)
                                 for k in sorted(set(e['stage'] for e in deaths))},
                    rooms_cleared=sum(e['rooms'] for e in eps) / m, boss_clear=sum(e['boss'] for e in eps) / m,
                    hurt_per_run=sum(e['hurt'] for e in eps) / m,
                    seconds=sum(e['decisions'] for e in eps) / m * args.frames_per_decision / 30,
                    items_taken=sum(e['items'] for e in eps) / m, active_uses=sum(e['uses'] for e in eps) / m,
                    pills_used=sum(e['pills'] for e in eps) / m)
                continue
            if floor:
                rooms = sum(e['rooms'] for e in eps)
                summary['groups'][name] = dict(
                    episodes=len(eps), floor_clear=len(wins) / m, death=sum(e['done'] == 2 for e in eps) / m,
                    timeout=sum(e['done'] == 3 for e in eps) / m, boss_clear=sum(e['boss'] > 0 for e in eps) / m,
                    rooms_cleared=rooms / m, rooms_entered=sum(e['entered'] for e in eps) / m,
                    hurt_per_floor=sum(e['hurt'] for e in eps) / m,
                    hurt_per_cleared_room=sum(e['hurt'] for e in eps) / max(rooms, 1),
                    clear_without_hit=sum(e['hurt'] == 0 for e in wins) / m,
                    seconds_per_clear=(sum(e['decisions'] for e in wins) / max(len(wins), 1)
                                       * args.frames_per_decision / 30))
                continue
            summary['groups'][name] = dict(
                episodes=len(eps), clear=len(wins) / m, death=sum(e['done'] == 2 for e in eps) / m,
                timeout=sum(e['done'] == 3 for e in eps) / m, hurt_per_room=sum(e['hurt'] for e in eps) / m,
                clear_without_hit=sum(e['hurt'] == 0 for e in wins) / m,
                seconds_per_clear=(sum(e['decisions'] for e in wins) / max(len(wins), 1)
                                   * args.frames_per_decision / 30))
        if args.characters:   # 2026-10-08: per character (the PlayerType of the episodes' first records)
            summary['characters'] = args.characters
            summary['by_character'] = {}
            for c_id in sorted({e['char0'] for e in episodes}):
                eps = [e for e in episodes if e['char0'] == c_id]
                m = len(eps)
                summary['by_character'][str(c_id)] = dict(
                    episodes=m, win=sum(e['done'] == 1 for e in eps) / m, death=sum(e['done'] == 2 for e in eps) / m,
                    timeout=sum(e['done'] == 3 for e in eps) / m, boss_clear=sum(e['boss'] > 0 for e in eps) / m,
                    rooms_cleared=sum(e['rooms'] for e in eps) / m, hurt=sum(e['hurt'] for e in eps) / m,
                    **({'floors_cleared': sum(e['exits'] for e in eps) / m} if run else {}))
        (out / 'episodes.json').write_text(json.dumps(episodes))
        if recorder is not None:
            recorder.close()
        if args.dump_final and finals:
            np.save(out / 'final_rows.npy', np.stack(finals))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))
    finally:
        sampler.close()


if __name__ == '__main__':
    main()
