"""Floor-runner replays as replay-viewer pages (abplus_replay_view.py shows one room shape per episode): every floor replay of
abplus_floor_run.py --replays is cut at its room changes into one episode per room visit, written as a pseudo evaluation
directory (meta.json, results.jsonl, replays/seed-<id>.jsonl.gz, id = seed * 100 + visit) plus a notes file naming the
floor, the visit and what happened in it; then abplus_replay_view.py builds the page.

  python abplus_floor_replays.py OUT.html RUN_DIR[=LABEL] ... [--title T] [--subtitle S]
"""
import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

KIND = {1: '普通房', 4: '宝箱房', 5: 'Boss 房'}


def split_run(run, label, work):
    meta = json.loads((run / 'meta.json').read_text(encoding='utf8'))
    results = {json.loads(l)['seed']: json.loads(l) for l in (run / 'results.jsonl').read_text(encoding='utf8').splitlines()}
    out = work / label
    (out / 'replays').mkdir(parents=True, exist_ok=True)
    pseudo, notes = [], {}
    for path in sorted((run / 'replays').glob('seed-*.jsonl.gz')):
        rows = [json.loads(l) for l in gzip.open(path, 'rt', encoding='utf8')]
        seed = rows[0]['metadata']['seed']
        result = results.get(seed, {})
        visits, current = [], None
        for row in rows[1:]:
            room = row['obs']['room']['room_idx']
            stage = row['obs']['room']['stage']
            if stage != 1:
                break
            if current is None or room != current[0]:
                current = (room, [])
                visits.append(current)
            current[1].append(row)
        combats = result.get('combats', [])
        for k, (room, steps) in enumerate(visits):
            if len(steps) < 2:
                continue
            rid = seed * 100 + k
            kind = steps[0]['obs']['room']['type']
            fights = [c for c in combats if c['room'] == room]
            last = k == len(visits) - 1
            if last:
                end = {'cleared': '通关（进入活板门）', 'death': '死亡'}.get(result.get('outcome'), result.get('outcome'))
            else:
                end = '离开'
            hp = [s['obs']['players'][0]['hearts'] + s['obs']['players'][0].get('soul', 0) for s in steps]
            text = (f"种子 {seed} · 第 {k + 1}/{len(visits)} 段 · {KIND.get(kind, f'类型 {kind}')} {room} · 血量 {hp[0]}→{hp[-1]} 个半心 · "
                    f"{'战斗 ' + '、'.join(c['outcome'] for c in fights) + ' · ' if fights else ''}{end}"
                    f"{' · 整层结果：' + str(result.get('outcome')) if last else ''}")
            notes[f'{label}|{rid}'] = text
            with gzip.open(out / 'replays' / f'seed-{rid}.jsonl.gz', 'wt', encoding='utf8') as f:
                f.write(json.dumps({'metadata': {'seed': rid, 'format': 'abplus-raw-obs-v1', 'goal': False}}) + '\n')
                for s in steps:
                    f.write(json.dumps({'action': s['action'], 'obs': s['obs']}, separators=(',', ':')) + '\n')
            pseudo.append(dict(seed=rid, task=KIND.get(kind, str(kind)), outcome=end, frames=4 * (len(steps) - 1)))
    (out / 'meta.json').write_text(json.dumps(dict(meta, frames_per_decision=4, deterministic=meta.get('deterministic'))))
    (out / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pseudo))
    return out, notes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path)
    p.add_argument('runs', nargs='+')
    p.add_argument('--title', default='整层回放')
    p.add_argument('--subtitle', default='')
    args = p.parse_args()
    work = args.out.parent / (args.out.stem + '-rooms')
    specs, notes = [], {}
    for spec in args.runs:
        run, _, label = spec.partition('=')
        run = Path(run)
        label = label or run.name
        out, n = split_run(run, label, work)
        notes.update(n)
        specs.append(f'{out}={label}')
    (work / 'notes.json').write_text(json.dumps(notes, ensure_ascii=False), encoding='utf8')
    view = Path(__file__).parent / 'abplus_replay_view.py'
    subprocess.run([sys.executable, str(view), str(args.out), *specs, '--notes', str(work / 'notes.json'),
                    '--title', args.title, '--subtitle', args.subtitle], check=True)


if __name__ == '__main__':
    main()
