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

import torch

STATS = dict(compiles=0, steady_compiles=0, calls=0)
_LOCK = threading.Lock()
_STEADY = [False]
_CONFIGURED = [False]


def configure():
    """Fixed Inductor cache dir + FX graph cache; a dynamo cache large enough for every bucket."""
    if _CONFIGURED[0]:
        return
    _CONFIGURED[0] = True
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',
                          os.path.join(os.path.expanduser('~'), '.cache', 'isaac-b42-inductor'))
    os.environ.setdefault('TORCHINDUCTOR_FX_GRAPH_CACHE', '1')
    import torch._dynamo
    import torch._inductor.config as inductor_config
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
    # The FX-tracing flag is process-global: a compile in the learner thread made a compiled
    # call in the actor thread raise (B42 prof-f0). Such a nested call runs eagerly instead.
    torch._dynamo.config.error_on_nested_fx_trace = False
    if hasattr(torch._dynamo.config, 'accumulated_cache_size_limit'):
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit, 1024)
    inductor_config.fx_graph_cache = True


def mark_steady():
    _STEADY[0] = True


def _backend(gm, example_inputs):
    with _LOCK:
        STATS['compiles'] += 1
        if _STEADY[0]:
            STATS['steady_compiles'] += 1
    from torch._inductor.compile_fx import compile_fx
    return compile_fx(gm, example_inputs)


def compile_batch_dynamic(fn):
    """torch.compile(fn, dynamic=False, fullgraph=True) with only dim 0 of every tensor
    argument symbolic. Arguments: tensors or (nested) dicts/tuples of tensors."""
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
            return torch.cat((x, x)) if x.dim() and x.shape[0] == 1 else x
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
