"""B42 glue layer: compiled fixed-bucket fast paths of the visible-v2 encoder.

Token dimensions are static (structured_input.BUCKETS: a handful of shapes chosen on the
host); only the batch dimension is symbolic (torch._dynamo.mark_dynamic), so every
bucket compiles once and a batch-size change never recompiles. Batches of one row are
padded to two (dynamo specialises size-1 dimensions). Compiles go through a counting
backend: STATS['compiles'] (all) and STATS['steady_compiles'] (after mark_steady(), i.e.
after the start-up warm-up; must stay 0) are written to progress.csv.
"""
import os
import threading

import numpy as np
import torch

STATS = dict(compiles=0, steady_compiles=0, calls=0)
_LOCK = threading.Lock()
_STEADY = [False]
_CONFIGURED = [False]


def configure():
    """Fixed Inductor cache dir (Triton kernels reused across runs); a dynamo cache large enough
    for every bucket."""
    if _CONFIGURED[0]:
        return
    _CONFIGURED[0] = True
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',
                          os.path.join(os.path.expanduser('~'), '.cache', 'isaac-b42-inductor'))
    import torch._dynamo
    import torch._inductor.config as inductor_config
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
    # The FX-tracing flag is process-global: a compile in the learner thread made a compiled
    # call in the actor thread raise (B42 prof-f0). Such a nested call runs eagerly instead.
    torch._dynamo.config.error_on_nested_fx_trace = False
    if hasattr(torch._dynamo.config, 'accumulated_cache_size_limit'):
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit, 1024)
    # FX graph cache off: a cached graph carries the size ranges of the compile that produced it
    # (B42 steady check: guards like 2 <= batch <= 19 from a warm-up batch), so every real batch
    # size recompiled; it also saved only ~9% of the cold warm-up (1118 vs 1225 s).
    inductor_config.fx_graph_cache = False
    # pad_mm decides per matmul size whether to pad, which bounds the symbolic batch (guards like
    # 193 <= batch, B42 steady check): every other size recompiled.
    inductor_config.shape_padding = False
    # Distinct symbolic sizes even when equal on the first call (row-packed lists: equal counts of
    # entities and items must not become one symbol and a recompile on the next batch).
    import torch.fx.experimental._config as fx_config
    if hasattr(fx_config, 'use_duck_shape'):
        fx_config.use_duck_shape = False


def mark_steady():
    _STEADY[0] = True


def _backend(gm, example_inputs):
    with _LOCK:
        STATS['compiles'] += 1
        if _STEADY[0]:
            STATS['steady_compiles'] += 1
            # B44: which compiled call recompiled, with its input shapes (diagnosis only)
            import traceback
            where = [f'{f.filename.rsplit("/", 1)[-1]}:{f.lineno} {f.name}' for f in traceback.extract_stack()
                     if 'isaac_bridge' in f.filename or 'train_tok' in f.filename][-4:]
            shapes = [tuple(x.shape) if hasattr(x, 'shape') else x for x in example_inputs[:8]]
            print('B44 steady compile', where, shapes, flush=True)
    from torch._inductor.compile_fx import compile_fx
    return compile_fx(gm, example_inputs)


def compile_batch_dynamic(fn):
    """torch.compile(fn, dynamic=False, fullgraph=True) with only dim 0 of every tensor
    argument symbolic. Arguments: tensors or (nested) dicts/tuples of tensors (made by callers
    as normal tensors: dynamo guards on dispatch keys, an inference tensor would recompile)."""
    configure()
    compiled = torch.compile(fn, backend=_backend, dynamic=False, fullgraph=True)

    def mark(x):
        if isinstance(x, torch.Tensor):
            if x.dim():
                # maybe_: a model whose ops pin the batch (e.g. broadcast against a fixed size)
                # gets its own specialised graph instead of a constraint error.
                torch._dynamo.maybe_mark_dynamic(x, 0)
        elif isinstance(x, dict):
            for v in x.values():
                mark(v)
        elif isinstance(x, (tuple, list)):
            for v in x:
                mark(v)

    def pad(x):
        if isinstance(x, torch.Tensor):
            if x.dim() and x.shape[0] == 1:
                with torch.inference_mode(False):
                    return torch.cat((x, x))
            return x
        if isinstance(x, dict):
            return {k: pad(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return type(x)(pad(v) for v in x)
        return x

    def unpad(y):
        if isinstance(y, torch.Tensor):
            return y[:1]
        return type(y)(unpad(v) for v in y)

    def call(*args):
        STATS['calls'] += 1
        one = any(isinstance(a, torch.Tensor) and a.dim() and a.shape[0] == 1 for a in args) or any(
            isinstance(a, dict) and any(v.dim() and v.shape[0] == 1 for v in a.values()) for a in args)
        if one:
            args = tuple(pad(a) for a in args)
        for a in args:
            mark(a)
        out = compiled(*args)
        return unpad(out) if one else out
    call.compiled = compiled
    return call


def _spatial(module, x):
    return module.forward_eager(x)


_SPATIAL = []
_ROWS = []


def _spatial_rows(module, x):
    return module.forward_rows(x)


def spatial_rows_compiled():
    """StructuredSpatial.forward_rows compiled: one graph per row-total bucket T and mode; every
    list length and the row count symbolic (packed_rows_batch keeps them >= 2)."""
    if not _ROWS:
        _ROWS.append(compile_batch_dynamic(_spatial_rows))
    return _ROWS[0]


def spatial_compiled():
    """StructuredSpatial.forward_eager compiled (module passed as an argument: the parameters
    are graph inputs, so the learner, actor and teacher copies share one compiled set)."""
    if not _SPATIAL:
        _SPATIAL.append(compile_batch_dynamic(_spatial))
    return _SPATIAL[0]


def synthetic_visible(n, body, entities, device, inventory=32, slots=32 // 2):
    """A decoded visible-v2 batch of the given buckets (one real slot each, rest padding)."""
    from .structured_input import PF, EF, NF, SF, RC
    f = dict(device=device)

    def ragged(name, width, k):
        mask = torch.zeros((n, k), dtype=torch.bool, **f)
        mask[:, 0] = True
        return {'v_'+name: torch.zeros((n, k, width), **f), 'v_'+name+'_mask': mask}
    b = {'v_room': torch.zeros((n, RC, 16, 28), **f), 'v_map': torch.zeros((n, 8, 13, 13), **f),
         'pending': torch.zeros((n, 5), dtype=torch.long, **f)}
    b.update(ragged('body', PF, body))
    b.update(ragged('local', RC*81, body))
    b.update(ragged('entity', EF, entities))
    b.update(ragged('inventory', NF, inventory))
    b.update(ragged('slots', SF, slots))
    return b


def warm_spatial(learner=None, actors=(), bf16=True):
    """Compile every (entity, body) bucket before training: the learner's training mode with
    gradients (forward+backward) and without, the actors' bf16 inference mode. Returns the
    number of compiles it caused; afterwards mark_steady() counts any further compile."""
    from .structured_input import BUCKETS, canvas_decoder
    before = STATS['compiles']
    combos = [(p, e) for e in BUCKETS['entity'] for p in BUCKETS['body']]
    model0 = learner if learner is not None else (actors[0] if actors else None)
    if model0 is not None:
        dev = next(model0.parameters()).device
        with torch.inference_mode(False), torch.no_grad(), torch.autocast('cuda', enabled=False):
            for p in BUCKETS['body']:
                for n in (3, 6):
                    canvas_decoder()(torch.zeros((n, 1792), device=dev), torch.zeros((n, p, 324), device=dev),
                                     torch.zeros((n, 169), device=dev))
    if learner is not None:
        dev = next(learner.parameters()).device
        was = learner.training
        learner.train()
        from .block_paged import SPATIAL_PACKED
        if SPATIAL_PACKED:
            # the learner's row-packed Spatial: one graph per row-total bucket T
            from .structured_input import T_BUCKETS, packed_rows_batch
            for T in T_BUCKETS:
                for n in (3, 6):
                    inv = max(2, T-30-2-2-2-128) if T > 164 else 2
                    ent = min(128, T-30-2-2-inv)
                    b = synthetic_visible(n, 1, 128, dev, inventory=max(32, inv), slots=16)
                    counts = np.array([[1, 1, ent, inv, 2]]*n)
                    # the learner's three modes: training (grad), no_grad passes (teacher
                    # targets) and inference_mode passes (start-of-update value recompute)
                    for mode in ('grad', 'no_grad', 'inference'):
                        ctx = (torch.inference_mode() if mode == 'inference' else
                               torch.set_grad_enabled(mode == 'grad'))
                        with ctx, torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                            part = packed_rows_batch(b, np.arange(n), counts, dev)
                            z = learner.encode_block(part)
                        if mode == 'grad':
                            z.float().sum().backward()
        for p, e in (() if SPATIAL_PACKED else combos):   # typed learner Spatial only when unpacked
            for n in (3, 6):
                b = synthetic_visible(n, p, e, dev)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                    x, _ = learner.structured_spatial(b)
                x.float().sum().backward()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                    learner.structured_spatial(b)
        learner.zero_grad(set_to_none=True)
        learner.train(was)
    for model in actors:
        dev = next(model.parameters()).device
        params = list(model.parameters())
        flags = [q.requires_grad for q in params]
        # the actor's parameters as they are, and frozen as the teacher generations' deep copies
        # (TeacherGenerations: requires_grad_(False)); dynamo guards on parameters' requires_grad
        for frozen in sorted({False, True} if any(flags) else {True}):
            for q, f in zip(params, flags):
                q.requires_grad_(f and not frozen)
            for p, e in combos:
                for n in (3, 6):
                    b = synthetic_visible(n, p, e, dev)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                        model.structured_spatial(b)
        for q, f in zip(params, flags):
            q.requires_grad_(f)
    torch.cuda.synchronize()
    return STATS['compiles']-before
