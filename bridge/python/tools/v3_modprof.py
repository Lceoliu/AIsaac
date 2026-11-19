"""Where the v3 frame encoder's learner time goes, part by part (seq-v1 L-1, 2026-11-19; the coordinator: seq1's u_ppo
15 s against T1's 5.9). From a train_tok --dump-rollouts directory (real records, the model's configuration and
weights), one GPU, the learner's precision (bf16 autocast, train mode), the minibatch's parts of --micro records as the
PPO pass takes them (lazy decode, the part's E / N / S buckets):

  whole     the frame encoder's forward, and forward + backward (a sum of its outputs as the loss)
  tokens    tokens(): the token MLPs, the CNNs, the embeddings and the glue between them -- and the blocks
  modules   each submodule alone on the inputs it saw in that forward (captured by hooks): forward, and forward +
            backward (a random output gradient); what the parts do not cover is the glue (cat / where / one-hot /
            log inputs / masks)
  tokens    real tokens per record (masks) against the padded width, per token type: what packing would save
  compile   (--compile) torch.compile of the frame encoder on the part's fixed shapes: the first call's time and the
            forward + backward after it, modes default and reduce-overhead (CUDA graphs)

  blocks_t  (--blocks-t 86,70,44,31,24) the encoder's 4 blocks alone on random tokens [micro, T, width] (no padding),
            forward + backward per T: what a frame of T tokens costs in the layers (e.g. fewer room tokens)

   usage: python tools/v3_modprof.py <rollouts dir> [--micro 2048] [--parts 2] [--compile] [--blocks-t ...] [--out f]"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v3_speed import load, timed   # noqa: E402

MODULES = ('p_mlp', 'e_mlp', 'n_mlp', 's_mlp', 'l_cnn', 'r_cnn', 'r_out', 'm_cnn', 'id_emb', 'item_emb', 'cat_emb',
           'anim_emb', 'icon_emb')


def capture(torch, enc, run):
    """{name: [(args, output)]} of enc's MODULES and blocks over one call of run()."""
    seen, hooks = {}, []
    names = [(n, getattr(enc, n)) for n in MODULES] + [(f'block{i}', b) for i, b in enumerate(enc.blocks)]
    for n, mod in names:
        def hook(_m, args, out, n=n):
            seen.setdefault(n, []).append((tuple(a.detach() if torch.is_tensor(a) else a for a in args),
                                           out.detach() if torch.is_tensor(out) else out))
        hooks.append(mod.register_forward_hook(hook))
    try:
        run()
    finally:
        for h in hooks:
            h.remove()
    return seen, dict(names)


def module_times(torch, mods, seen, bf16=True):
    """Per module: forward ms and forward + backward ms (summed over its calls in the forward), alone."""
    res = {}
    for n, calls in seen.items():
        mod = mods[n]
        f_ms = fb_ms = 0.0
        for args, out in calls:
            def mk():
                return tuple(a.clone().requires_grad_(True) if torch.is_tensor(a) and a.is_floating_point() else a
                             for a in args)
            g = torch.randn_like(out.float()).to(out.dtype)

            def fwd():
                with torch.autocast('cuda', torch.bfloat16, enabled=bf16):
                    return mod(*mk())

            def fb():
                mod.zero_grad(set_to_none=True)
                with torch.autocast('cuda', torch.bfloat16, enabled=bf16):
                    o = mod(*mk())
                o.backward(g.to(o.dtype))
            f_ms += timed(torch, fwd, 10, 2)
            fb_ms += timed(torch, fb, 10, 2)
        res[n] = dict(calls=len(calls), fwd_ms=round(f_ms, 3), fwd_bwd_ms=round(fb_ms, 3))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('rollouts')
    ap.add_argument('--micro', type=int, default=2048)
    ap.add_argument('--parts', type=int, default=2)
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--blocks-t', default='')
    ap.add_argument('--packed', action='store_true', help='the packed forward (forward_packed) in whole and compile')
    ap.add_argument('--out', default=None)
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
    m.train()
    enc = m.frame_v3
    rows, pend = load(paths, a.micro * a.parts)
    n = len(rows)
    raw_h = torch.from_numpy(rows.view(np.uint8).reshape(n, -1))
    d = TP.decode_rows(raw_h.to(dev), pending=torch.from_numpy(pend.astype(np.int64)).to(dev))
    out = dict(records=n, micro=a.micro, gpu=torch.cuda.get_device_name(0), parts=[])
    for lo in range(0, n, a.micro):
        idx = torch.arange(lo, min(lo + a.micro, n), device=dev)
        b = TP.take(d, idx)
        v = VI.materialize(VI.get_batch(b))
        pe = b['pending']
        part = dict(records=len(idx))
        widths = {t: int(v[f'{t}_mask'].shape[1]) for t in 'PENS'}
        real = {t: float(v[f'{t}_mask'].sum(1).float().mean()) for t in 'PENS'}
        real['L'] = float((v['P_mask'] & ~v['local_pad']).sum(1).float().mean())
        widths['L'] = widths['P']
        part['tokens'] = dict(width=widths, real_mean=real, padded_total=sum(widths.values()) + 30,
                              real_total=round(sum(real.values()) + 30, 1))

        call = enc.forward_packed if a.packed else enc

        def fwd():
            with torch.autocast('cuda', torch.bfloat16):
                return call(v, pe, m.gtok.tok)

        def fwd_bwd():
            m.zero_grad(set_to_none=True)
            with torch.autocast('cuda', torch.bfloat16):
                x, _p, _g = call(v, pe, m.gtok.tok)
            x.float().sum().backward()

        def tok():
            with torch.autocast('cuda', torch.bfloat16):
                return enc.tokens(v, pe, m.gtok.tok)
        torch.cuda.reset_peak_memory_stats()
        part['whole'] = dict(fwd_ms=timed(torch, fwd, 10, 3), fwd_bwd_ms=timed(torch, fwd_bwd, 10, 3),
                             tokens_fwd_ms=timed(torch, tok, 10, 3),
                             peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
        seen, mods = capture(torch, enc, fwd)
        mt = module_times(torch, mods, seen)
        part['modules'] = mt
        blocks = {k: v_ for k, v_ in mt.items() if k.startswith('block')}
        toks = {k: v_ for k, v_ in mt.items() if not k.startswith('block')}
        w = part['whole']
        part['split'] = dict(
            blocks_fwd_ms=round(sum(x['fwd_ms'] for x in blocks.values()), 3),
            blocks_fwd_bwd_ms=round(sum(x['fwd_bwd_ms'] for x in blocks.values()), 3),
            token_modules_fwd_ms=round(sum(x['fwd_ms'] for x in toks.values()), 3),
            token_modules_fwd_bwd_ms=round(sum(x['fwd_bwd_ms'] for x in toks.values()), 3))
        s = part['split']
        s['glue_fwd_ms (tokens() - its modules)'] = round(w['tokens_fwd_ms'] - s['token_modules_fwd_ms'], 3)
        s['rest_fwd_bwd_ms (whole - blocks - token modules)'] = round(
            w['fwd_bwd_ms'] - s['blocks_fwd_bwd_ms'] - s['token_modules_fwd_bwd_ms'], 3)
        if a.compile:
            comp = {}
            for mode in ('default', 'reduce-overhead'):
                try:
                    import torch._dynamo
                    torch._dynamo.reset()
                    ce = torch.compile(call, mode=mode, dynamic=bool(a.packed))

                    def cfb(ce=ce):
                        m.zero_grad(set_to_none=True)
                        with torch.autocast('cuda', torch.bfloat16):
                            x, _p, _g = ce(v, pe, m.gtok.tok)
                        x.float().sum().backward()
                    t0 = time.perf_counter()
                    cfb()
                    torch.cuda.synchronize()
                    first = time.perf_counter() - t0
                    comp[mode] = dict(first_call_s=round(first, 1), fwd_bwd_ms=timed(torch, cfb, 10, 3))
                except Exception as e:  # noqa: BLE001
                    comp[mode] = dict(error=repr(e)[:300])
            part['compile'] = comp
        out['parts'].append(part)
        print(json.dumps(part, indent=1), flush=True)
    if a.blocks_t:   # the layers' cost per token count (random tokens, bf16, train mode)
        bt = {}
        D = enc.blocks[0].qkv.in_features
        for T in [int(x) for x in a.blocks_t.split(',')]:
            x0 = torch.randn((a.micro, T, D), device=dev)
            bias = torch.zeros((a.micro, 1, 1, T), device=dev)

            def fb(x0=x0, bias=bias):
                m.zero_grad(set_to_none=True)
                x = x0.clone().requires_grad_(True)
                with torch.autocast('cuda', torch.bfloat16):
                    for blk in enc.blocks:
                        x = blk(x, bias)
                x.float().sum().backward()
            bt[T] = round(timed(torch, fb, 10, 3), 2)
        out['blocks_t_fwd_bwd_ms'] = bt
        print(json.dumps(dict(blocks_t_fwd_bwd_ms=bt)), flush=True)
    s = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(s + '\n')


if __name__ == '__main__':
    main()
