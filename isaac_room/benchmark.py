"""Bounded, seeded smoke evaluation. Worker processes never load pygame."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import time

from .env import Config, RoomEnv
from .policy import scripted_action


def run_episode(job):
    seed, layout, policy, max_ticks = job
    env = RoomEnv(Config(layout=layout, max_ticks=max_ticks))
    obs, info = env.reset(seed=seed)
    rng = random.Random(seed + 1000000)
    total_reward, steps = 0.0, 0
    while info['outcome'] == 'running':
        if policy == 'scripted':
            action = scripted_action(obs)
        elif policy == 'random':
            action = rng.randrange(9), rng.randrange(5)
        else:
            action = 0, 0
        obs, reward, _, _, info = env.step(action)
        total_reward += reward
        steps += 1
    env.close()
    return {'seed': seed, 'layout': layout, 'policy': policy, **info,
            'steps': steps, 'reward': round(total_reward, 6)}


def evaluate(*, episodes=12, workers=1, policy='scripted', seed=0, max_ticks=3600):
    if episodes < 1 or workers < 1 or max_ticks < 1:
        raise ValueError('episodes, workers and max_ticks must be positive')
    if policy not in ('idle', 'random', 'scripted'):
        raise ValueError('Unknown policy')
    jobs = [(seed + i, ('empty', 'pillars')[i % 2], policy, max_ticks) for i in range(episodes)]
    start = time.perf_counter()
    if workers == 1:
        rows = list(map(run_episode, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(run_episode, jobs))
    wall = time.perf_counter() - start
    outcomes = {name: sum(row['outcome'] == name for row in rows) for name in ('cleared', 'dead', 'timeout')}
    return {'model': 'approximate-room-v0', 'trained': False, 'workers': workers,
            'episodes': episodes, 'policy': policy, 'max_ticks': max_ticks,
            'outcomes': outcomes, 'clear_rate': outcomes['cleared'] / episodes,
            'mean_damage_taken': sum(row['damage_taken'] for row in rows) / episodes,
            'wall_seconds': wall,
            'physics_ticks_per_second': sum(row['ticks'] for row in rows) / wall,
            'env_steps_per_second': sum(row['steps'] for row in rows) / wall,
            'episodes_detail': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--policy', choices=('idle', 'random', 'scripted'), default='scripted')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-ticks', type=int, default=3600)
    parser.add_argument('--output', type=Path, default=Path('runs/benchmark.json'))
    args = parser.parse_args()
    report = evaluate(episodes=args.episodes, workers=args.workers, policy=args.policy,
                      seed=args.seed, max_ticks=args.max_ticks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes_detail'}, indent=2))
    print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()
