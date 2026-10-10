"""Character randomisation through the token sampler (2026-10-08, EXPERIMENTS.md A25): tok_floor workers with
TokSamplerConfig.characters and uniformly random actions (no policy, no GPU), as train_tok / eval_tok would run them.

Training mode (--seconds of wall time): every floor start's character is drawn by the worker (tok_floor's prepare, its
own rng); reported per worker: the ROW pchar of each episode's first record (floor starts and archive starts), the
counter char_starts, errors. Evaluation mode (--eval-seeds first:count): each seed once, its character from
tok_floor.character_of_seed; the first record's pchar must be that character.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_characters_sampler.py --groups-file ../abplus/catalog/scaling2_groups.json \
      --characters "0,4,7,9,13" --workers 2 --seconds 120 --eval-seeds 2147600000:6 --out <dir>
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.tok_floor import character_of_seed, parse_characters
from isaac_bridge.tok_sampler import ST, TokSampler, TokSamplerConfig, default_stub_list


def run(cfg, seconds, rng):
    sampler = TokSampler(cfg)
    n = cfg.workers
    firsts, ends = [], Counter()
    t0 = time.perf_counter()
    try:
        while sampler.live and (seconds <= 0 or time.perf_counter() - t0 < seconds):
            idx, slots = sampler.poll(want=1, max_wait=0.0005)
            if not idx:
                continue
            rows = sampler.rows[slots]
            for j, i in enumerate(idx):
                r = rows[j]
                if r['first']:
                    firsts.append(dict(worker=int(i), seed=int(r['seed']), pchar=int(r['pchar']),
                                       hearts=round(float(r['player'][4]) * 12, 2),
                                       soul=round(float(r['player'][5]) * 12, 2), stage=int(r['stage'])))
                if r['done']:
                    ends[int(r['done'])] += 1
                sampler.actions[i, :3] = (int(rng.integers(9)), int(rng.integers(5)), 0)
            sampler.reply(idx)
        stats = sampler.stats.copy()
    finally:
        sampler.close()
    return dict(firsts=firsts, ends=dict(ends), seconds=round(time.perf_counter() - t0, 1),
                char_starts=float(stats[:, ST['char_starts']].sum()), states=float(stats[:, ST['states']].sum()),
                errors=float(stats[:, ST['errors']].sum()), archive_starts=float(stats[:, ST['archive_starts']].sum()),
                decisions=float(stats[:, ST['decisions']].sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--characters', default='0,4,7,9,13')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seconds', type=float, default=120.0)
    p.add_argument('--eval-seeds', default='2147600000:6')
    p.add_argument('--port', type=int, default=44430)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chars = parse_characters(args.characters)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    base = dict(workers=args.workers, specs=[spec], assign=[0] * args.workers, frames_per_decision=4, mode='floor',
                floor_seconds=60.0, floor_stall_seconds=10.0, items=True, characters=args.characters,
                bridge_lua=default_bridge_lua(), preload=default_preload(),
                stub_list=default_stub_list(default_preload()), nice=10)
    rng = np.random.default_rng(5)
    train = run(TokSamplerConfig(**base, name='chrsp', port=args.port, seed=777, episodes_per_state=2,
                                 archive_prob=0.3, snapshot_prob=0.5), args.seconds, rng)
    first, count = (int(v) for v in args.eval_seeds.split(':'))
    seeds = list(range(first, first + count))
    ev = run(TokSamplerConfig(**base, name='chrse', port=args.port + 8, episodes_per_state=1,
                              eval_seeds=[seeds[k::args.workers] for k in range(args.workers)]), 0, rng)
    want = {s: character_of_seed(s, chars) for s in seeds}
    summary = dict(characters=args.characters,
                   train=dict({k: v for k, v in train.items() if k != 'firsts'}, episodes=len(train['firsts']),
                              pchar_counts=dict(Counter(f['pchar'] for f in train['firsts'])),
                              pchar_not_listed=sum(f['pchar'] not in set(chars[0].tolist()) for f in train['firsts'])),
                   eval=dict({k: v for k, v in ev.items() if k != 'firsts'}, episodes=len(ev['firsts']),
                             want=want, got={f['seed']: f['pchar'] for f in ev['firsts']},
                             match=sum(f['pchar'] == want[f['seed']] for f in ev['firsts'])))
    (out / 'firsts.json').write_text(json.dumps(dict(train=train['firsts'], eval=ev['firsts']), indent=1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
