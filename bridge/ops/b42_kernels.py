"""B42 glue-layer profile: GPU kernels and synced time per call of the block-memory model parts.

Standalone (no game, no training threads): real inputs saved by a short run with
ISAAC_BLOCK_DUMP=<dir> (model.pt, actor_rows/pending.npy, learner_rows/pending.npy).

  python b42_kernels.py <dump dir> [label]

Each part runs alone. Kernels: one call of the part's eager body under torch.profiler (CUDA
activity; kernels vs memcpy/memset). A CUDA graph replays exactly the kernels it captured, so
graph-replayed parts are counted on the body they capture (profiling a replay itself stalled
CUPTI for minutes on the 4090). ms: median of 20 GPU-synced calls of the path production
runs (graph replay where production replays a graph). Parts:
  actor.decode      from_rows (host decode + upload + canvas expansion)
  actor.spatial     encode_block, bf16 inference (production: SpatialEncoderGraph replay)
  actor.reader      one Reader step on real slot inputs (production: ReaderStepGraph replay)
  actor.writer      BlockWriter on one completed block per actor row, bf16 inference
  actor.heads       TokPolicy.act heads/sampling on a fixed Reader result, bf16 inference
  actor.step        a whole SlotBlockActor.step, no block boundary (ms only: mixed graph/host)
  learn.spatial.fwd / learn.spatial.fwdbwd   encode_block on one 512-row training microbatch, bf16
  learn.writer.fwd  BlockWriter on 16 blocks, bf16 autocast, with grad
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.structured_input import from_rows
from isaac_bridge.block_paged import SlotBlockActor


def count_kernels(fn, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    k = m = 0
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            low = e.name.lower()
            if 'memcpy' in low or 'memset' in low:
                m += 1
            else:
                k += 1
    return k, m


def timed(fn, warm=3, reps=20):
    for _ in range(warm):
        fn()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter()
        fn()
        torch.cuda.synchronize(); times.append((time.perf_counter()-t)*1e3)
    return round(statistics.median(times), 3)


def part(out, name, body, production=None, count=True):
    t0 = time.perf_counter()
    r = {}
    if count:
        r['kernels'], r['copies'] = count_kernels(body)
    r['ms'] = timed(production or body)
    out[name] = r
    print('part', name, r, round(time.perf_counter()-t0, 1), 's', file=sys.stderr, flush=True)


def main():
    root = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else 'run'
    dev = torch.device('cuda')
    ck = torch.load(os.path.join(root, 'model.pt'), map_location='cpu')
    model = TokPolicy(**ck['config']).to(dev).eval()
    model.load_state_dict(ck['model'])
    learner = TokPolicy(**ck['config']).to(dev).train()
    learner.load_state_dict(ck['model'])
    ar, ap = np.load(os.path.join(root, 'actor_rows.npy')), np.load(os.path.join(root, 'actor_pending.npy'))
    lr, lp = np.load(os.path.join(root, 'learner_rows.npy')), np.load(os.path.join(root, 'learner_pending.npy'))
    n = len(ar)
    actor = SlotBlockActor(model, n, graph_entities=128, bf16=True)
    memory = model.block_memory
    cfg = memory.config
    slots = list(range(n))
    u = torch.rand((n, sum(model.heads)), device=dev)
    print('warm-up steps', file=sys.stderr, flush=True)
    bf16 = dict(device_type='cuda', dtype=torch.bfloat16)
    with torch.autocast(**bf16):
        for t in range(5*cfg.block_steps+7):
            rows = ar.copy(); rows['first'] = int(t == 0)
            actor.step(rows, ap, slots, uniform=u)
    torch.cuda.synchronize()
    out = {'label': label, 'actor_rows': n, 'learner_rows': len(lr), 'torch': torch.__version__,
           'step_graph': getattr(actor, 'step_graph', None) is not None}
    rows = ar.copy(); rows['first'] = 0
    part(out, 'actor.decode', lambda: from_rows(rows, dev, ap))
    b = from_rows(rows, dev, ap)

    def sp():
        with torch.inference_mode(), torch.autocast(**bf16):
            return model.encode_block(b)
    enc = getattr(actor, 'encoder', None)
    part(out, 'actor.spatial', sp, (lambda: enc(rows, ap)) if enc is not None else None)
    z = sp()
    # Reader inputs of these slots, as the actor gathers them (read_inputs), at bf16.
    with torch.inference_mode(), torch.autocast(**bf16):
        ids = actor.state._ids(slots)[1]
        steps, prefixes, bank, ends, kp = actor.state.read_inputs(memory, ids, z.dtype)
    args = (z[:, None], steps[:, None], prefixes, bank, ends, kp)

    def rd():
        with torch.inference_mode(), torch.autocast(**bf16):
            return memory.read(*args)
    rgraph = getattr(actor, 'reader', None)

    def rg():
        with torch.inference_mode(), torch.autocast(**bf16):
            return rgraph(*args)
    part(out, 'actor.reader', rd, rg if rgraph is not None else None)
    partial = torch.randn((n, cfg.block_steps, cfg.latents, cfg.latent_width), device=dev)

    def wr():
        with torch.inference_mode(), torch.autocast(**bf16):
            return memory.writer(partial)
    part(out, 'actor.writer', wr)
    hidden, tokens, current, allowed, mask = rd()
    fixed = dict(hidden=hidden, tokens=tokens, local_mask=allowed, memory_mask=mask, positions=steps[:, None])

    def heads():
        with torch.inference_mode(), torch.autocast(**bf16):
            ctx = {'memory_call': lambda _z: fixed, 'encoded': z, 'version': 0, 'reset_done': True}
            return model.act(b, u=u, need_vg=True, temporal=ctx)
    part(out, 'actor.heads', heads)

    def step():
        with torch.autocast(**bf16):
            actor.step(rows, ap, slots, uniform=u)
    # Whole step: ms only (warm-up/timing calls cross block boundaries: their Writer is included).
    part(out, 'actor.step', step, count=False)
    sg = getattr(actor, 'step_graph', None)
    if sg is not None:
        # Kernels of the captured whole-step body (the same kernels the graph replays), on a
        # throwaway actor so the timed actor's state is untouched.
        probe = SlotBlockActor(model, n, graph_entities=128, bf16=True)
        from isaac_bridge.structured_input import decode_frame_host
        host = decode_frame_host(rows, packed=True)
        host['pending'] = np.asarray(ap, np.int64)
        cap = 1 << (n-1).bit_length()
        layout, size = probe.step_graph._layout(host, cap, sum(model.heads))
        flat = torch.zeros(size, dtype=torch.uint8, device=dev)
        st = probe.step_graph._views(flat, layout, True)
        st['ids'].fill_(probe.state.slots-1)

        def body():
            with torch.inference_mode(), torch.autocast(**bf16):
                return probe.step_graph._body(st, False)
        out['actor.step_graph_body'] = dict(zip(('kernels', 'copies'), count_kernels(body)))
    lb = from_rows(lr, dev, lp)

    def lf():
        with torch.autocast(**bf16):
            return learner.encode_block(lb)
    part(out, 'learn.spatial.fwd', lf)

    def lfb():
        z = lf()
        z.float().square().mean().backward()
    part(out, 'learn.spatial.fwdbwd', lfb)
    # The learner's production call: token-sorted microbatches (encode_sorted; row-packed when
    # the code has B42 glue 4), from the host counts.
    from isaac_bridge import block_paged
    from isaac_bridge.structured_input import decode_frame_host, host_counts
    hl = decode_frame_host(lr, packed=True)
    lcounts = host_counts(hl)

    def lsf():
        with torch.autocast(**bf16):
            return block_paged.encode_sorted(learner, lb, 512, lcounts)
    part(out, 'learn.encode_sorted.fwd', lsf)

    def lsfb():
        lsf().float().square().mean().backward()
    part(out, 'learn.encode_sorted.fwdbwd', lsfb)
    hb = torch.randn((16, cfg.block_steps, cfg.latents, cfg.latent_width), device=dev, requires_grad=True)

    def lw():
        with torch.autocast(**bf16):
            return learner.block_memory.writer(hb)
    part(out, 'learn.writer.fwd', lw)
    for k in ('v_body', 'v_local', 'v_entity', 'v_inventory', 'v_slots'):
        out['shape.learn.'+k] = list(lb[k].shape[:2])
        out['shape.actor.'+k] = list(b[k].shape[:2])
    out['real.learn.entities_mean'] = float(lb['v_entity_mask'].float().sum(1).mean())
    print(json.dumps(out))


if __name__ == '__main__':
    main()
