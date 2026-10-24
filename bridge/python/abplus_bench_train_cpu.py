"""CPU cost of one PPO training pass (forward + backward of evaluate_actions) on full 64-frame windows.

Upper bound for CPU training: the GPU learner encodes each frame once per segment (frame dedup),
while this full-window path re-encodes all 64 frames per sample; the temporal Transformer is the same.
usage: python abplus_bench_train_cpu.py <replay.jsonl.gz> <checkpoint dir> <threads> [batch]
"""
import gzip, json, sys, time
import numpy as np, torch
from abplus_eval import load_policy
from isaac_bridge.transformer_obs import VisibleHistory

replay, ckpt, threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
batch = int(sys.argv[4]) if len(sys.argv) > 4 else 32
torch.set_num_threads(threads)
rows = [json.loads(x) for x in gzip.open(replay, 'rt')][1:]
history = VisibleHistory(64, 256, deadline=True)
windows = []
for i, row in enumerate(rows):
    w = history.append(row['obs'])
    if i >= 64:
        windows.append(w)
    if len(windows) == batch:
        break
policy = load_policy(ckpt, 'cpu')
policy.set_training_mode(True)
obs = {k: torch.as_tensor(np.stack([w[k] for w in windows])) for k in windows[0]}
actions = torch.zeros((batch, 3), dtype=torch.long)
masks = np.ones((batch, 49), bool)
masks[:, 48] = False
times = []
for rep in range(4):
    policy.zero_grad(set_to_none=True)
    t = time.perf_counter()
    values, log_prob, entropy = policy.evaluate_actions(obs, actions, action_masks=masks)
    loss = -log_prob.mean() + values.pow(2).mean() - 0.01 * entropy.mean()
    loss.backward()
    times.append(time.perf_counter() - t)
per_sample = 1000 * min(times[1:]) / batch
print(json.dumps({'threads': threads, 'batch': batch, 'ms_per_sample_epoch': round(per_sample, 2),
                  'ms_per_sample_2_epochs': round(2 * per_sample, 2)}))
