"""Compiled source FFN: discard only GELU outputs, never replay a matrix multiply.

Selective checkpoint tags are consumed by AOTAutograd at compile time. This is
not eager selective checkpointing (whose saved-op queue permits only one VJP).
The resulting compiled backward can run repeatedly on retained source graphs.
"""
from functools import partial

import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy


def native_graph(gm, inputs):
    return make_boxed_func(gm.forward)


# Keep input CNNs on their original ATen/cuDNN kernels. Inductor lowering
# changed their BF16 outputs in the recorded-checkpoint parity test.
native_aot=aot_autograd(fw_compiler=native_graph,bw_compiler=native_graph,
    partition_fn=min_cut_rematerialization_partition)


def pointwise_policy(ctx, op, *args, **kwargs):
    if op == torch.ops.aten.gelu.default:
        return CheckpointPolicy.MUST_RECOMPUTE
    return CheckpointPolicy.MUST_SAVE


def source_ff(ff, x):
    return checkpoint(ff, x, use_reentrant=False, preserve_rng_state=False,
        context_fn=partial(create_selective_checkpoint_contexts, pointwise_policy))


def compile_source_component(template):
    """One code cache across immutable weight versions, weights as inputs."""
    names=tuple(n for n,_ in template.named_parameters())
    original_forward=template.forward
    def checkpoint_forward(x):
        return source_ff(original_forward,x)
    template.forward=checkpoint_forward
    def functional_forward(x,parameters):
        # Functionalize outside the checkpoint body: PyTorch 2.7 cannot trace
        # the reparameterization context manager inside that higher-order op.
        return torch.func.functional_call(template,dict(zip(names,parameters)),(x,))
    return torch.compile(functional_forward,dynamic=True,fullgraph=True,backend=native_aot)
