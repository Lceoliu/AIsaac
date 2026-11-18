"""main1 held-out evaluations: one block per evaluation directory (summary.json, episodes.json, entropy.json,
config.json of evtools/eval_ent.py). Ends split by episodes.json: death (done 2), the run's time cap (done 3 with
decisions x 4 / 30 >= run seconds - 1) and no progress (done 3 earlier: the floor's stall rule); floors cleared = exits
per run; new rooms = rooms entered (EV_NEW_ROOM) per run. A 512-seed one also shows its first 128 seeds.
usage: python3 ev_table.py <dir> [<dir> ...]"""
import json
import sys
from pathlib import Path


def block(eps, run_s, fpd):
    m = max(len(eps), 1)
    cap = [e for e in eps if e['done'] == 3 and e['decisions'] * fpd / 30 >= run_s - 1]
    stall = [e for e in eps if e['done'] == 3 and e not in cap]
    st = {}
    for e in eps:
        st[e['stage']] = st.get(e['stage'], 0) + 1
    return (f"n {len(eps)}  floors {sum(e['exits'] for e in eps) / m:.3f}  stage {dict(sorted(st.items()))}  "
            f"3+ {sum(e['stage'] >= 3 for e in eps)}  4+ {sum(e['stage'] >= 4 for e in eps)}  "
            f"boss {sum(e['boss'] for e in eps) / m:.3f}  rooms {sum(e['rooms'] for e in eps) / m:.2f}  "
            f"new rooms {sum(e['entered'] for e in eps) / m:.2f} (>=1 {sum(e['entered'] >= 1 for e in eps) / m:.3f})  "
            f"pills {sum(e['pills'] for e in eps) / m:.2f}  items {sum(e['items'] for e in eps) / m:.2f}  "
            f"active {sum(e['uses'] for e in eps) / m:.2f}  hurt {sum(e['hurt'] for e in eps) / m:.2f}  "
            f"seconds {sum(e['decisions'] for e in eps) / m * fpd / 30:.0f}  "
            f"death {sum(e['done'] == 2 for e in eps) / m:.3f} cap {len(cap) / m:.3f} stall {len(stall) / m:.3f}"
            f" other {sum(e['done'] not in (2, 3) for e in eps) / m:.3f}")


for d in map(Path, [a for a in sys.argv[1:] if not a.startswith("--")]):
    if not (d / 'summary.json').is_file():
        print(f'== {d.name}: no summary.json')
        continue
    s = json.loads((d / 'summary.json').read_text())
    eps = json.loads((d / 'episodes.json').read_text())
    ent = json.loads((d / 'entropy.json').read_text()) if (d / 'entropy.json').is_file() else {}
    cfg = json.loads((d / 'config.json').read_text()) if (d / 'config.json').is_file() else {}
    run_s, fpd = float(s['episode_seconds']), int(s['frames_per_decision'])
    print(f"== {d.name}: update {s.get('update')} seeds {s['seeds']} characters {s.get('characters', '(Isaac)')} "
          f"greedy {s.get('greedy')} errors {s['errors']} empty {s['empty']} wall {s['seconds']:.0f} s "
          f"x_real {s['x_real_time']:.0f} game h {s['game_hours']:.2f} block_memory {s.get('block_memory')}")
    print('   ' + block(eps, run_s, fpd))
    if len(eps) > 128:
        first = int(s['seeds'].split(':')[0])
        print('   first 128: ' + block([e for e in eps if e['seed'] < first + 128], run_s, fpd))
    if ent:
        print('   entropy ' + ' '.join(f'{k} {v:.3f}' for k, v in ent.items() if isinstance(v, float))
              + f" (decisions {ent.get('decisions')}, greedy {ent.get('greedy')})")
        if 'use_prob' in ent:
            print('   p(use) after the gate, all decisions / gate open (open share): ' + '  '.join(
                f"{k} {v['all']:.4f} / {v['open']:.4f} ({v['open_share']:.3f})" for k, v in ent['use_prob'].items()))
            print('   sampled "use" decisions per run: ' + '  '.join(
                f'{k} {v / max(len(eps), 1):.2f}' for k, v in ent['use_actions'].items())
                  + '  (actual events per run: active ' + f"{sum(e['uses'] for e in eps) / max(len(eps), 1):.2f}"
                  + f", pills {sum(e['pills'] for e in eps) / max(len(eps), 1):.2f}; bombs placed are not in episodes.json)")
    if 'by_character' in s:
        print('   by_character ' + json.dumps(s['by_character']))
    if cfg.get('config') and '--config' in sys.argv:
        print('   config ' + json.dumps(cfg['config']))
