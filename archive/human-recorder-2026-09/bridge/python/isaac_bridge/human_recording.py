"""Read-only validation of native human records; deliberately not a PPO data loader."""
import gzip
import json
import os
from pathlib import Path


def read_episode(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def validate_episode(path, *, encode_actor=False):
    rows = read_episode(path)
    if len(rows) < 3 or rows[0].get('type') != 'header' or rows[-1].get('type') != 'end':
        raise ValueError('Missing header, frames or final record; incomplete capture')
    header, end = rows[0], rows[-1]
    if header['schema'] != 'isaac-human-v1':
        raise ValueError('Unsupported capture schema')
    previous = header['obs']; queries = 0; active = 0; max_entities = 0
    histories = None
    if encode_actor:
        from .transformer_obs import VisibleHistory
        histories = VisibleHistory(history=2, deadline=True)
        histories.append(previous)
    for index, row in enumerate(rows[1:-1], 1):
        if row['type'] != 'frame' or row['frame'] != index:
            raise ValueError(f'Frame sequence discontinuity at {index}')
        obs = row['obs']
        if row['previous_game_frame'] != previous['game_frame'] or obs['game_frame'] != previous['game_frame'] + 1:
            raise ValueError(f'Native frame discontinuity at {index}')
        if obs['logic_frames'] != index:
            raise ValueError(f'Wrong observation/action boundary at {index}')
        if not obs['players'] or not obs['terrain']['cells']:
            raise ValueError(f'Missing player/map at {index}')
        for sample in row['queries']:
            if sample['action'] not in header['action_ids'] or sample['hook'] not in header['hooks'].values():
                raise ValueError('Unrecognized native input query')
            value = sample['value']
            if not isinstance(value, (bool, int, float)) or not 0 <= value <= 1:
                raise ValueError('Invalid input value')
            queries += 1; active += bool(value)
        if histories is not None:
            histories.append(obs)  # actual production actor encoder, no hidden labels
        max_entities = max(max_entities, len(obs['entities']))
        previous = obs
    frames = len(rows) - 2
    if end['frames'] != frames or not queries:
        raise ValueError('Frame count mismatch or no game input queries captured')
    if end['outcome'] == 'death' and not previous['players'][0]['dead']:
        raise ValueError('Death outcome contradicts final observation')
    if end['outcome'] == 'timeout' and frames != header['max_logic_frames']:
        raise ValueError('Timeout did not reach the declared deadline')
    if end['outcome'] == 'win' and not rows[-2].get('terminal_evidence',{}).get('boss_dead',False):
        raise ValueError('Win outcome lacks native Boss-death evidence')
    return dict(file=Path(path).name,valid=True,frames=frames,seconds=frames/header['logic_hz'],
                outcome=end['outcome'],source=header['source'],queries=queries,active_queries=active,
                warnings=[] if active else ['No nonzero player input in this episode'],
                max_entities=max_entities,actor_encoding_checked=encode_actor,
                input_resampling='not performed; native query timeline preserved')


def archive_episode(path, *, encode_actor=False):
    path = Path(path)
    result = validate_episode(path, encode_actor=encode_actor)
    # Completed raw file remains untouched. Flush OS buffers before reporting saved.
    with path.open('r+b') as stream:
        os.fsync(stream.fileno())
    compressed = path.with_suffix('.jsonl.gz')
    temporary = compressed.with_suffix('.tmp')
    with path.open('rb') as src, gzip.open(temporary, 'wb') as dst:
        import shutil
        shutil.copyfileobj(src, dst)
    with temporary.open('r+b') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, compressed)
    checked = validate_episode(compressed)
    if result['frames'] != checked['frames']:
        raise ValueError('Compressed record failed round-trip validation')
    return result


def make_replay(path, output):
    rows = read_episode(path)
    template = Path(__file__).with_name('human_replay.html').read_text(encoding='utf8')
    payload = json.dumps(rows, separators=(',', ':')).replace('</', '<\\/')
    Path(output).write_text(template.replace('/*DATA*/', payload), encoding='utf8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('--check-actor', action='store_true')
    parser.add_argument('--prepare-to', type=Path, help='Prepare this session directory for BC; does not train')
    parser.add_argument('--validation-episode', default='episode-0006.jsonl.gz', help='Entire episode held out from training')
    args = parser.parse_args()
    if args.prepare_to:
        from .human_dataset import prepare_dataset
        result = prepare_dataset(args.record, args.prepare_to, args.validation_episode)
    else:
        result = validate_episode(args.record, encode_actor=args.check_actor)
    print(json.dumps(result, ensure_ascii=False))
