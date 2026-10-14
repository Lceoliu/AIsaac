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
  Memory checkpoints (2026-10-11, train_tok --memory gru; config 'memory'): the GRU state is carried per worker and
  zeroed at each episode's first record, as in training.
  Item-token / global-token checkpoints (2026-10-12, train_tok --item-tokens 1 / --global-token 1; config 'item_tokens',
  'global_token'): built from the config like every other option (the extra tokens need no other input; value_global
  is not used for acting); summary 'item_tokens' / 'global_token'.
A28 (2026-10-14), evaluation-only switches (all off by default: the evaluation is the one before):
  --fused-override 1: a separate-term (B21 / B22 / B23) checkpoint evaluated in the fused layout (B24,
  tok_policy.convert_to_fused, as train_tok's resume with --item-tokens-fused 1; approximate, no training);
  --item-slots N overrides the held-item token slots (default: the checkpoint's). Summary 'item_tokens_fused',
  'fused_override' (and 'item_slots').
  --ablate items|global|both: items = every item token absent (ItemTokens' present mask all False, so the old tokens,
  the global token and the memory see none; separate-term layout: ItemTokens.gate zeroed as well; the player token's
  own item inputs stay); global = GlobalToken.gate zeroed (separate: its term for the old tokens; fused: the multiplier
  of its value vector), mem_in zeroed (the GRU's input without it) and its key masked in every attention (no
  token attends to it); apply_ablation. Summary 'ablate'.
  --no-pedestals 1 (floor / run modes): every collectible pedestal (type 5, variant 100) removed from the room before
  the step of each record that shows one (tok_floor, tok_pickup.NO_PEDESTAL_LUA through the bridge's lua command), so
  the policy never picks one up. Summary 'no_pedestals', 'pedestal_runs', 'pedestals_removed'.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.tok_obs import EV_ACTIVE, EV_BOSS, EV_CLEARED, EV_EXIT, EV_ITEM, EV_NEW_ROOM, EV_PILL
from isaac_bridge.tok_sampler import ST, TokSampler, TokSamplerConfig

ABLATE = ('', 'items', 'global', 'both')


def build_eval_model(ck, items, device, fused_override=False, item_slots=None):
    """(A28) The evaluated policy of a checkpoint: (model in eval mode, load_compatible's changed names, the conversion
    note or None). Without fused_override / item_slots: TokPolicy(**config, items) + load_compatible, as before.
    fused_override: a separate-term checkpoint (config item_tokens / global_token, not item_tokens_fused) built fused and
    its state converted by convert_to_fused (every tensor as saved). item_slots: the held-item token slots (needs
    item_tokens)."""
    from isaac_bridge.tok_policy import TokPolicy, convert_to_fused, load_compatible
    cfg = dict(ck['config'], items=items)
    if fused_override:
        if cfg.get('item_tokens_fused'):
            raise ValueError('--fused-override: the checkpoint is fused already')
        if not (cfg.get('item_tokens') or cfg.get('global_token')):
            raise ValueError('--fused-override needs an item-token or global-token checkpoint')
        cfg['item_tokens_fused'] = True
    if item_slots is not None:
        if not cfg.get('item_tokens'):
            raise ValueError('--item-slots needs an item-token checkpoint')
        cfg['item_slots'] = int(item_slots)
    model = TokPolicy(**cfg).to(device).eval()
    state, note = ck['model'], None
    if fused_override:
        state, note = convert_to_fused(model, state)
    return model, load_compatible(model, state), note


def apply_ablation(model, what):
    """(A28) Remove the item tokens ('items'), the global token ('global') or both ('both') from an evaluated model in
    place ('' : nothing). items: ItemTokens.forward wrapped to return its tokens with an all-False present mask (every
    item token a masked key in both layouts: the old tokens, the global token and the memory see none; the player
    token's own item inputs, the item_emb multi-hot and the active / trinket / pocket embeddings, stay); separate-term
    layout: ItemTokens.gate zeroed too (the extra term exactly 0). global: GlobalToken.gate zeroed (separate: its term
    for the old tokens; fused: the multiplier of its value vector) and mem_in zeroed (the GRU's input no longer carries
    it), and its key masked in every attention (separate: the extra tokens' key mask; fused: the blocks' key mask;
    the fused packed layers then refuse to run); value_global is not used for acting. Raises ValueError if the model
    lacks the tokens."""
    import torch
    if what not in ABLATE:
        raise ValueError(f'--ablate {what!r}: one of {ABLATE[1:]}')
    if what in ('items', 'both'):
        it = getattr(model, 'itok', None)
        if it is None:
            raise ValueError('--ablate items needs an item-token checkpoint (config item_tokens)')
        inner = it.forward

        def masked(b, compact=False):
            tok, present = inner(b, compact)
            return tok, torch.zeros_like(present)
        it.forward = masked   # (an instance attribute: nn.Module.__call__ runs it)
        if not model.fused:
            with torch.no_grad():
                it.gate.zero_()
    if what in ('global', 'both'):
        g = getattr(model, 'gtok', None)
        if g is None:
            raise ValueError('--ablate global needs a global-token checkpoint (config global_token)')
        with torch.no_grad():
            g.gate.zero_()
            if g.mem_in is not None:
                g.mem_in.weight.zero_()
                g.mem_in.bias.zero_()
        # and its KEY masked in every attention: it neither feeds the item tokens (separate layout: the extra tokens'
        # key mask ybias) nor takes softmax mass (fused: the blocks' key mask, the global token is the last token)
        if not model.fused:
            inner_ext = model._ext_tokens

            def ext_tokens(b):
                ext = inner_ext(b)
                if ext is not None and ext['glob'] is not None:
                    col = torch.tensor([ext['glob']], device=ext['ybias'].device)
                    ext['ybias'] = ext['ybias'].index_fill(-1, col, -1e9)
                return ext
            model._ext_tokens = ext_tokens
        else:
            for block in model.blocks:
                def fwd(x, bias, ext=None, vscale=None, _inner=block.forward):
                    return _inner(x, bias.index_fill(-1, torch.tensor([bias.shape[-1] - 1], device=bias.device),
                                                     -1e9), ext, vscale)

                def packed(*a, **k):
                    raise NotImplementedError('--ablate global (fused): the padded forward only (eval_tok act)')
                block.forward, block.forward_packed = fwd, packed


def main():
    # torch is imported here, not at the top: the sampler's workers are spawned and import this module too
    import torch
    from isaac_bridge.tok_policy import to_batch
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
    p.add_argument('--fused-override', type=int, default=0,
                   help='A28: 1 = a separate-term checkpoint evaluated in the fused layout (convert_to_fused)')
    p.add_argument('--item-slots', type=int, default=None,
                   help="A28: held-item token slots of the evaluated model (default: the checkpoint's)")
    p.add_argument('--ablate', default='', choices=ABLATE,
                   help='A28: evaluate with the item tokens / the global token / both removed (apply_ablation)')
    p.add_argument('--no-pedestals', type=int, default=0,
                   help='A28, floor / run modes: 1 = every collectible pedestal removed from the rooms entered')
    args = p.parse_args()
    if args.random and (args.fused_override or args.item_slots is not None or args.ablate):
        p.error('--fused-override / --item-slots / --ablate need a checkpoint')
    if args.no_pedestals and args.mode not in ('floor', 'run'):
        p.error('--no-pedestals needs --mode floor or run')
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
                           characters=args.characters if floor else '', no_pedestals=bool(args.no_pedestals),
                           bridge_lua=default_bridge_lua(), preload=default_preload(),
                           stub_list=stub if Path(stub).is_file() else '', nice=0)
    # CPU by default: a second CUDA process on the trainer's GPU can crash a trainer that replays CUDA graphs (B12)
    device = torch.device(os.environ.get('TOK_EVAL_DEVICE', 'cpu'))
    model, info = None, dict(policy='random')
    rng = np.random.default_rng(0)
    if not args.random:
        # (--items on an items-free one: padded; A28 --fused-override / --item-slots: build_eval_model)
        try:
            model, changed, note = build_eval_model(ck, items, device, bool(args.fused_override), args.item_slots)
            apply_ablation(model, args.ablate)
        except ValueError as exc:
            p.error(str(exc))
        if note:
            print(note, flush=True)
        info = dict(policy=str(args.checkpoint), update=ck.get('update'), decisions=ck.get('decisions'),
                    greedy=args.greedy, weights_not_exact=changed)
        for key_ in ('item_tokens', 'global_token', 'item_tokens_fused'):   # 2026-10-12 (B21 / B22), 10-14 (B24)
            if model.config.get(key_):
                info[key_] = True
        if args.fused_override:   # A28
            info['fused_override'] = True
        if args.item_slots is not None:
            info['item_slots'] = int(args.item_slots)
        if args.ablate:
            info['ablate'] = args.ablate
            print(f'ablate {args.ablate}: gates now {model.gate_values()}', flush=True)
    if args.no_pedestals:   # A28
        info['no_pedestals'] = True
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
    # 2026-10-11 (memory checkpoints, config 'memory'): the GRU state per eval worker, zeroed at an episode's first
    # record (as in training: not at a room or floor change)
    mem_dim = int(getattr(model, 'memory', 0) or 0) if model is not None else 0
    mem_h = torch.zeros((n, mem_dim), device=device) if mem_dim else None
    mem_reset = np.zeros(n, bool)
    if mem_dim:
        info['memory'] = mem_dim
        print(f'memory: GRU state of size {mem_dim} per worker, reset at each episode start', flush=True)
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
                    mem_reset[i] = True
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
                elif mem_h is not None:   # (memory) the workers' states in, the new ones kept
                    ai = torch.tensor(act_idx, dtype=torch.int64, device=device)
                    h_in = mem_h.index_select(0, ai) * torch.from_numpy(~mem_reset[act_idx]).float().to(device)[:, None]
                    mem_reset[act_idx] = False
                    with torch.inference_mode():
                        out_ = model.act(to_batch(rows[act_rows], pending[act_idx], device), greedy=args.greedy,
                                         mem=h_in)
                    mem_h.index_copy_(0, ai, out_[3])
                    actions = out_[0].cpu().numpy()
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
        if args.no_pedestals:   # A28: records whose removal ran, pedestals removed (tok_floor)
            summary['pedestal_runs'] = float(sampler.stats[:, ST['np_runs']].sum())
            summary['pedestals_removed'] = float(sampler.stats[:, ST['np_removed']].sum())
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
