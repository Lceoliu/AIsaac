"""Where a v3 update's time goes (seq-v1, the coordinator 2026-11-18: the first v3 arm runs at about half T1's speed;
the learner is the bottleneck, then the actor's inference). Measures on one GPU, from a train_tok --dump-rollouts
directory (real records and the model's configuration):

learner (one minibatch of --minibatch records in --micro parts, as train_tok's PPO pass):
  rows -> device + decode_rows (the lazy v3 batch: the frames' bytes and checked counts), take (a part's rows),
  materialize (the part's decode), the frame encoder's forward, the rest of the forward (heads; the temporal side is
  not run here -- T1 has it too), the backward of both; each with CUDA events. Also: the frame encoder in bf16
  autocast, and the tokens per record -- real (unpadded) vs padded -- i.e. what packing would save.
actor (records per call --actor-b, the E bucket the calls hit most): the host side (rows_counts), and the device side
  captured as CUDA graphs and replayed -- the decode alone, the decode + the frame encoder, the whole forward; with
  --t1 <checkpoint> T1's ROW model's whole forward on the same records for comparison.

   usage: python tools/v3_speed.py <rollouts dir> [--t1 ckpt.pt] [--minibatch 4096] [--micro 1024] [--actor-b 8,32,128]
   prints one JSON object (and --out <file>)."""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def _ev(torch):
    return torch.cuda.Event(enable_timing=True)


def timed(torch, fn, iters=10, warm=3):
    """Median ms of fn() over iters (CUDA events), after warm calls."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        a, b = _ev(torch), _ev(torch)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return float(np.median(out))


def graph_ms(torch, fn, iters=50):
    """Median ms of a CUDA graph replay of fn (captured after three eager warm-up calls on a side stream)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return timed(torch, g.replay, iters, 3)


def load(paths, n):
    z = np.load(paths[0], allow_pickle=False)
    keep = np.arange(z['rows'].shape[1])[None, :] < z['length'][:, None]
    rows, pend = z['rows'][keep], z['pending'][keep]
    for p in paths[1:]:
        if len(rows) >= n:
            break
        z = np.load(p, allow_pickle=False)
        keep = np.arange(z['rows'].shape[1])[None, :] < z['length'][:, None]
        rows, pend = np.concatenate([rows, z['rows'][keep]]), np.concatenate([pend, z['pending'][keep]])
    return np.ascontiguousarray(rows[:n]), np.ascontiguousarray(pend[:n])


def learner(torch, TP, VI, m, rows, pend, micro, dev):
    res = {}
    n = len(rows)
    raw_h = torch.from_numpy(rows.view(np.uint8).reshape(n, -1))
    pend_t = torch.from_numpy(pend.astype(np.int64)).to(dev)

    def prep():
        return TP.decode_rows(raw_h.to(dev, non_blocking=False), pending=pend_t)
    res['prep_ms (rows to the device + decode_rows, the minibatch)'] = timed(torch, prep, 5, 1)
    d = prep()
    parts = [torch.arange(i, min(i + micro, n), device=dev) for i in range(0, n, micro)]
    acc = dict(take=0.0, decode=0.0, frame_fwd=0.0, rest_fwd=0.0, step=0.0, frame_fwd_bf16=0.0, step_bf16=0.0)
    real_tok, pad_tok = [], []
    m.train()
    for idx in parts:
        t = {}
        acc['take'] += timed(torch, lambda: t.__setitem__('b', TP.take(d, idx)), 5, 1)
        b = t['b']
        acc['decode'] += timed(torch, lambda: t.__setitem__('v', VI.materialize(VI.get_batch(b))), 5, 1)
        v = t['v']
        pm = v['P_mask'] & ~v['local_pad']
        real = v['P_mask'].sum(1) + pm.sum(1) + v['E_mask'].sum(1) + v['N_mask'].sum(1) + v['S_mask'].sum(1) + 30
        real_tok.append(real.float().mean().item())
        pad_tok.append(2 * v['P_mask'].shape[1] + v['E_mask'].shape[1] + v['N_mask'].shape[1] + v['S_mask'].shape[1] + 30)

        def fwd(bf16=False):
            with torch.autocast('cuda', torch.bfloat16, enabled=bf16):
                x, _pad, g = m.frame_v3(v, b['pending'], m.gtok.tok)
            return x, g

        def step(bf16=False):   # (bf16: the whole forward under autocast, as train_tok --amp bf16)
            m.zero_grad(set_to_none=True)
            with torch.autocast('cuda', torch.bfloat16, enabled=bf16):
                x, g = fwd(bf16)
                out = m._heads(x[:, 0], False, None, x[:, g], True, None)
                loss = sum(o.float().sum() for o in out if torch.is_tensor(o) and o.is_floating_point())
            loss.backward()
        acc['frame_fwd'] += timed(torch, lambda: fwd(False), 5, 1)
        x0, g0 = fwd(False)
        acc['rest_fwd'] += timed(torch, lambda: m._heads(x0[:, 0], False, None, x0[:, g0], True, None), 5, 1)
        def safe(fn):   # (an out-of-memory measurement is nan, the others go on)
            try:
                return timed(torch, fn, 5, 1)
            except torch.OutOfMemoryError:
                m.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                return float('nan')
        acc['step'] += safe(lambda: step(False))
        acc['frame_fwd_bf16'] += safe(lambda: fwd(True))
        torch.cuda.reset_peak_memory_stats()
        acc['step_bf16'] += safe(lambda: step(True))
        acc['peak_gb_bf16'] = max(acc.get('peak_gb_bf16', 0.0), torch.cuda.max_memory_allocated() / 2 ** 30)
    k = len(parts)
    res['parts'] = k
    res['take_ms'] = acc['take']
    res['decode_ms (materialize, all parts)'] = acc['decode']
    res['frame_encoder_fwd_ms'] = acc['frame_fwd']
    res['rest_fwd_ms (heads, no temporal)'] = acc['rest_fwd']
    res['step_ms (fwd + bwd, all parts)'] = acc['step']
    res['backward_ms (step - frame fwd - rest fwd)'] = acc['step'] - acc['frame_fwd'] - acc['rest_fwd']
    res['frame_encoder_fwd_ms bf16'] = acc['frame_fwd_bf16']
    res['step_ms bf16 (whole forward under autocast, as --amp bf16)'] = acc['step_bf16']
    res['peak_gb bf16 step (one part)'] = acc['peak_gb_bf16']
    res['tokens per record: real mean'] = float(np.mean(real_tok))
    res['tokens per record: padded'] = float(np.mean(pad_tok))
    return res


BF16_ACTOR = [True]   # (train_tok --actor-bf16 1: the actor graphs under bfloat16 autocast)


def actor(torch, TP, VI, m, rows, pend, bs, dev, t1=None):
    res = {}
    m.eval()
    for B in bs:
        r = rows[:B]
        p = torch.from_numpy(pend[:B].astype(np.int64)).to(dev)
        t0 = time.perf_counter()
        for _ in range(50):
            counts, eb = VI.rows_counts(r, stats=False)
        host = (time.perf_counter() - t0) / 50 * 1000
        w = int(np.asarray(r['visible_bytes']).max()) // 4
        F = torch.from_numpy(np.ascontiguousarray(r['visible_frame'].view('<f4')[:, :VI.V3.FRAME_BYTES // 4])).to(dev)
        c = torch.from_numpy(counts).to(dev)
        b = TP.to_batch(r, pend[:B], dev)
        st = {}
        with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16, enabled=bool(BF16_ACTOR[0])):
            dec = lambda: st.__setitem__('v', VI.decode_frames_device(F, c, eb, VI.N_MAX_V3, VI.S_MAX_V3, check=False))  # noqa: E731
            dec()
            v = st['v']
            enc = lambda: m.frame_v3(VI.decode_frames_device(F, c, eb, VI.N_MAX_V3, VI.S_MAX_V3, check=False), p,  # noqa: E731
                                     m.gtok.tok)
            full = lambda: m(b, need_vg=True)   # noqa: E731
            row = dict(e_bucket=eb, frame_floats=w, host_rows_counts_ms=host,
                       graph_decode_ms=graph_ms(torch, dec), graph_decode_encoder_ms=graph_ms(torch, enc),
                       graph_full_forward_ms=graph_ms(torch, full),
                       eager_full_forward_ms=timed(torch, full, 20, 3))
            if t1 is not None:
                bt = TP.to_batch(_row_only(r), pend[:B], dev)
                row['t1_graph_full_forward_ms'] = graph_ms(torch, lambda: t1(bt, need_vg=True))
                row['t1_eager_full_forward_ms'] = timed(torch, lambda: t1(bt, need_vg=True), 20, 3)
        res[f'B={B}'] = row
    return res


def _row_only(r):
    """The records with their visible frame dropped (T1's ROW model reads no frame: visible_schema 0)."""
    r = r.copy()
    r['visible_schema'] = 0
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('rollouts')
    ap.add_argument('--t1', default=None)
    ap.add_argument('--minibatch', type=int, default=4096)
    ap.add_argument('--micro', default='1024,2048')
    ap.add_argument('--actor-fp32', action='store_true', help='the actor graphs in float32 (default: bf16, as --actor-bf16 1)')
    ap.add_argument('--actor-b', default='8,32,128')
    ap.add_argument('--out', default=None)
    ap.add_argument('--actor-only', action='store_true')
    a = ap.parse_args(argv)
    import torch
    from isaac_bridge import tok_policy as TP
    from isaac_bridge import v3_input as VI
    dev = torch.device('cuda')
    paths = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.npz')))
    pts = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.pt')))
    ck = torch.load(pts[0], map_location='cpu', weights_only=False)
    m = TP.TokPolicy(**ck['config']).to(dev)
    m.load_state_dict(ck['model'])
    rows, pend = load(paths, max(a.minibatch, 128))
    out = dict(records=len(rows), gpu=torch.cuda.get_device_name(0))
    out['learner'] = {f'micro {mi}': learner(torch, TP, VI, m, rows[:a.minibatch], pend[:a.minibatch], mi, dev)
                      for mi in [int(x) for x in str(a.micro).split(',')]} if not a.actor_only else 'skipped'
    BF16_ACTOR[0] = not a.actor_fp32
    t1 = None
    if a.t1:
        c1 = torch.load(a.t1, map_location='cpu', weights_only=False)
        t1 = TP.TokPolicy(**c1['config']).to(dev).eval()
        t1.load_state_dict(c1['model'])
    out['actor'] = actor(torch, TP, VI, m, rows, pend, [int(x) for x in a.actor_b.split(',')], dev, t1)
    s = json.dumps(out, indent=1)
    print(s)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(s + '\n')


if __name__ == '__main__':
    main()
