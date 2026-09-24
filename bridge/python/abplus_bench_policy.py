"""Per-decision policy latency on CPU/GPU from a recorded AB+ replay (cached-feature path)."""
import gzip, json, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from abplus_eval import load_policy, CachedPolicy
from isaac_bridge.transformer_obs import VisibleHistory

replay, device, threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.set_num_threads(threads)
rows = [json.loads(x) for x in gzip.open(replay, 'rt')][1:]
policy = load_policy(sys.argv[4], device)
h = VisibleHistory(64, 256, deadline=True)
cp = CachedPolicy(policy, 64, True)
mask = np.array([True] * 45 + [True, True] + [True, False])
ts = []
for i, row in enumerate(rows[:400]):
    h.append(row['obs'])
    t = time.perf_counter()
    d = cp.distribution(h.frames[-1], mask)
    a = d.get_actions(deterministic=True)[0].cpu().numpy()
    ts.append(time.perf_counter() - t)
ts = np.array(ts[100:]) * 1000
print(json.dumps({'device': device, 'threads': threads, 'ms_mean': round(ts.mean(), 2), 'ms_p50': round(np.median(ts), 2), 'ms_p95': round(np.percentile(ts, 95), 2)}))
