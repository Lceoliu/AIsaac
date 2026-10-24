"""Independent Actor graphs can reuse scratch, but never overwrite live outputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from isaac_bridge.rebuild_actor import RebuildGraphActor

torch.manual_seed(9)
actor = object.__new__(RebuildGraphActor)
actor.capture = torch.cuda.Stream()
actor.sdpa_math = False
actor.graph_pool = None if '--separate' in sys.argv else torch.cuda.graph_pool_handle()
x = torch.randn(4 * 1024 * 1024, device='cuda')
torch.cuda.synchronize()
torch.cuda.empty_cache()
before = torch.cuda.memory_reserved()
graphs = []
with torch.no_grad():
    for scale in range(1, 7):
        graphs.append(actor._capture(lambda: ((x * scale).sin().sum().reshape(1),)))
torch.cuda.synchronize()
reserved = torch.cuda.memory_reserved() - before
for graph, _ in graphs:graph.replay()
saved = [out[0].clone() for _, out in graphs]
for index in [5, 0, 4, 1, 3, 2, 0, 5, 2]:
    graph, out = graphs[index]
    graph.replay()
    torch.testing.assert_close(out[0], (x * (index + 1)).sin().sum().reshape(1))
    for (_, other), expected in zip(graphs, saved):
        torch.testing.assert_close(other[0], expected, rtol=0, atol=0)
print('graph_count=6 scratch_reserved_bytes=', reserved, flush=True)
assert reserved < 3 * x.nbytes, 'Each Actor graph still retains its own large scratch pool'
print('PASS arbitrary sequential bucket order, persistent outputs, bounded shared scratch')
