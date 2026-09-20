"""One MaskablePPO learner, independent native engines, completed-episode budget."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import psutil
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback

from isaac_bridge.launch import DEFAULT_GAME_DIR
from isaac_bridge.parallel import prepare_worker, make_worker, EngineVecEnv
from isaac_bridge.transformer_policy import CombatTransformer
from isaac_bridge.turbo import (launch_suspended, HOOK_WORKER_PROFILE, post_close,
                                wait_process, terminate)


class EpisodeLog(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.episodes = []
        self.overlap_steps = 0
        self.vector_steps = 0
        self.sampling_seconds = 0.

    def _on_rollout_start(self):
        self.rollout_start = time.perf_counter()

    def _on_rollout_end(self):
        self.sampling_seconds += time.perf_counter()-self.rollout_start

    def _on_step(self):
        infos = self.locals['infos']
        self.vector_steps += 1
        self.overlap_steps += int(max(x['step_start'] for x in infos) < min(x['step_end'] for x in infos))
        for info in infos:
            if 'episode' in info:
                entry = {**info['episode'], 'worker': info['worker'], 'outcome': info['outcome'],
                         'frames': info['elapsed_frames'], 'learner_step': self.num_timesteps}
                self.episodes.append(entry)
                with self.path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(entry)+'\n')
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, choices=(2, 4), required=True)
    p.add_argument('--episodes', type=int, default=512)
    p.add_argument('--updates', type=int, help='bounded benchmark instead of episode budget')
    p.add_argument('--episode-frames', type=int, default=3600)
    p.add_argument('--port', type=int, default=27115)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--history', type=int, default=64)
    p.add_argument('--capacity', type=int, default=256)
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; install a CUDA-enabled PyTorch build')
        torch.cuda.reset_peak_memory_stats()
    report = {'status': 'starting', 'learner_pid': os.getpid(), 'workers': [], 'config': vars(args).copy()}
    report['config'] = {k: str(v) if isinstance(v, Path) else v for k, v in report['config'].items()}
    def write_report():
        (args.out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    controls, environments, vec, model = [], [], None, None
    samples, stop = [], threading.Event()
    start = time.perf_counter()
    callback = EpisodeLog(args.out/'episodes.jsonl')
    profile_source = Path.home()/'Documents/My Games/Binding of Isaac Repentance+'
    try:
        factories = []
        for rank in range(args.workers):
            root = args.out/f'worker-{rank}'
            runtime, profile, savedata = prepare_worker(root, DEFAULT_GAME_DIR, profile_source)
            pid, ctl, output = launch_suspended(args.port+rank, game_dir=str(runtime),
                worker_profile=str(profile), extra_args=('--luadebug', '--set-stage=1'),
                skip_render=True, virtual_clock=False, font_guard=False, file_retry=False,
                probe_dump=False, log_dir=str(root/'native'))
            controls.append(ctl)
            report['workers'].append({'rank': rank, 'pid': pid, 'port': args.port+rank,
                'profile': str(profile), 'runtime': str(runtime), 'savedata': str(savedata), 'launch': output})
            write_report()
            if not ctl.stats().hooks_mask & HOOK_WORKER_PROFILE:
                raise RuntimeError('Native worker profile isolation hook is not active')
            # Finish each engine's startup before starting another: simultaneous
            # asset/Workshop initialization caused real handshake timeouts at N=4.
            env = make_worker(args.port+rank, rank, str(root), args.episode_frames, args.history, args.capacity)
            environments.append(env)
            env.reset()
            factories.append(lambda env=env: env)
        vec = EngineVecEnv(factories)
        obs = vec.reset()
        # Advancing worker 0 must not advance the other engines or their histories.
        before = vec.env_method('snapshot')
        vec.env_method('step', np.array([15, 0, 0]), indices=0)
        after = vec.env_method('snapshot')
        assert after[0]['logic_frames'] == before[0]['logic_frames']+2
        assert after[1:] == before[1:]
        report['independence_probe'] = {'before': before, 'after': after}
        for worker in report['workers']:
            savedata = Path(worker['savedata'])
            savedatapath = (Path(worker['runtime'])/'savedatapath.txt').read_text()
            assert str(savedata).replace('\\','/') in savedatapath.replace('\\','/')
            assert (savedata/'log.txt').exists()
            scripts = [line for line in (savedata/'log.txt').read_text(encoding='utf-8').splitlines()
                       if 'Running Lua Script:' in line and '/mods/' in line.lower().replace('\\','/')]
            assert all('/mods/isaac_rl_bridge/' in line.lower().replace('\\','/') for line in scripts), scripts
            worker['save_path_verified'] = savedatapath.strip()
            worker['mod_scripts_verified'] = scripts
        processes = [psutil.Process(os.getpid())]
        processes += [psutil.Process(x['pid']) for x in report['workers']]
        for proc in processes:
            proc.cpu_percent()
        def sample_resources():
            with (args.out/'resources.jsonl').open('w', encoding='utf-8') as stream:
                while not stop.wait(1):
                    row = {'elapsed': time.perf_counter()-start, 'processes': [
                        {'pid': proc.pid, 'rss': proc.memory_info().rss, 'cpu_percent': proc.cpu_percent()}
                        for proc in processes]}
                    samples.append(row)
                    stream.write(json.dumps(row)+'\n'); stream.flush()
        sampler = threading.Thread(target=sample_resources, daemon=True)
        sampler.start()
        # Keep total rollout size 256 transitions for a fair 2 vs 4 comparison.
        steps = 256//args.workers
        if args.checkpoint:
            model = MaskablePPO.load(args.checkpoint, env=vec, device=args.device, n_steps=steps)
        else:
            model = MaskablePPO('MultiInputPolicy', vec, n_steps=steps, batch_size=8, n_epochs=4,
                learning_rate=3e-4, seed=0, device=args.device, verbose=0,
                policy_kwargs={'features_extractor_class': CombatTransformer,
                    'features_extractor_kwargs': {'features_dim': 256, 'layers': 4, 'heads': 8},
                    'net_arch': {'pi': [128], 'vf': [128]}, 'normalize_images': False})
        initial = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
        initial_steps = model.num_timesteps
        report['initial_model_steps'] = initial_steps
        report['torch_version'] = torch.__version__
        updates = 0
        report['status'] = 'training'
        training_start = time.perf_counter()
        while (updates < args.updates if args.updates is not None else len(callback.episodes) < args.episodes):
            model.learn(total_timesteps=256, reset_num_timesteps=False, callback=callback)
            updates += 1
            elapsed = time.perf_counter()-training_start
            report.update(completed_episodes=len(callback.episodes), outcomes=dict(Counter(x['outcome'] for x in callback.episodes)),
                per_worker_episodes=dict(Counter(x['worker'] for x in callback.episodes)), updates=updates,
                actual_steps=model.num_timesteps-initial_steps, training_seconds=elapsed,
                decisions_per_second=(model.num_timesteps-initial_steps)/elapsed,
                sampling_seconds=callback.sampling_seconds,
                optimization_and_overhead_seconds=elapsed-callback.sampling_seconds,
                overlap_steps=callback.overlap_steps, vector_steps=callback.vector_steps)
            if updates % 8 == 0:
                model.save(args.out/'ppo_monstro')
            write_report()
            print(json.dumps({k: report[k] for k in ('updates','completed_episodes','outcomes','decisions_per_second')}), flush=True)
        model.save(args.out/'ppo_monstro')
        delta = sum(float(torch.sum((v-initial[k])**2)) for k,v in model.policy.state_dict().items())**.5
        assert delta > 0
        obs = vec.reset()
        masks = get_action_masks(vec)
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        restored = MaskablePPO.load(args.out/'ppo_monstro', device=args.device)
        restored_action, _ = restored.predict(obs, deterministic=True, action_masks=masks)
        np.testing.assert_array_equal(action, restored_action)
        vec.step(restored_action)
        report.update(status='completed', parameter_l2_change=delta,
                      training_epochs=model._n_updates,
                      reload_actions=restored_action.tolist(), checkpoint=str(args.out/'ppo_monstro.zip'))
    except BaseException as error:
        report.update(status='error', error={'type': type(error).__name__, 'message': str(error)})
        if model is not None:
            model.save(args.out/'interrupted')
        raise
    finally:
        stop.set()
        if 'sampler' in locals():
            sampler.join()
        if samples:
            report['resources'] = {'samples': len(samples),
                'peak_total_rss_mb': max(sum(x['rss'] for x in s['processes']) for s in samples)/2**20,
                'mean_cpu_core_equivalents': sum(sum(x['cpu_percent'] for x in s['processes']) for s in samples)/len(samples)/100,
                'logical_cpus': psutil.cpu_count()}
        if args.device == 'cuda':
            report['cuda'] = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
                'peak_allocated_mb': torch.cuda.max_memory_allocated()/2**20,
                'peak_reserved_mb': torch.cuda.max_memory_reserved()/2**20}
        try:
            if vec is not None:
                vec.close()
            else:
                for env in environments:
                    env.close()
        finally:
            for worker, ctl in zip(report['workers'], controls):
                worker['native_final'] = asdict(ctl.stats())
                ctl.close()
                post_close(worker['pid'])
                worker['exit'] = wait_process(worker['pid'], 40)
                worker['forced_termination'] = worker['exit'] is None
                if worker['forced_termination']:
                    terminate(worker['pid'], 9)
                    worker['exit'] = wait_process(worker['pid'], 10)
            report['wall_seconds'] = time.perf_counter()-start
            write_report()
    if any(w['exit'] != 0 or w['forced_termination'] for w in report['workers']):
        raise RuntimeError('Worker cleanup was not a normal exit; see report')


if __name__ == '__main__':
    main()
