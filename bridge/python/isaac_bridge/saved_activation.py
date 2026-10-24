"""Experimental 8-bit storage for BF16 backward activations, never forward values."""
from functools import lru_cache
import weakref

import torch


@lru_cache(None)
def _code(device):
    from bitsandbytes.functional import create_linear_map
    return create_linear_map().to(device)


class SavedActivation8bit:
    """Preserve view layouts and share one quantization per live backing storage.

    Only sizeable BF16 activations are compressed. Three-dimensional attention
    Q/K/V/output and Writer sequence views keep their BF16 values: quantizing
    those failed the recorded-checkpoint gradient-error gate. FP32 statistics/masks, integer
    indices and both FP32/BF16 immutable parameter storages remain untouched.
    Unpacking is repeated for each VJP; no decompressed history is retained.
    """
    def __init__(self, replica, blocksize):
        from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
        self.quantize, self.dequantize = quantize_blockwise, dequantize_blockwise
        self.blocksize = blocksize
        tensors = list(replica.parameters()) + list(getattr(replica, '_source_cast_values', {}).values())
        self.weights = {t.untyped_storage().data_ptr() for t in tensors if not t.is_meta}
        self.cache = {}
        self.raw_bytes = self.packed_bytes = self.tensors = self.cache_hits = 0
        self.hooks = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)

    def pack(self, tensor):
        storage = tensor.untyped_storage()
        if (tensor.device.type != 'cuda' or tensor.dtype != torch.bfloat16 or tensor.ndim == 3 or
                storage.nbytes() < 8192 or storage.data_ptr() in self.weights):
            return tensor.detach()
        key = storage.data_ptr()
        cached = self.cache.get(key)
        if cached is not None and cached[0]() is storage:
            value = cached[1]
            self.cache_hits += 1
        else:
            flat = tensor.detach().as_strided((storage.nbytes() // tensor.element_size(),), (1,), 0)
            value = self.quantize(flat, code=_code(tensor.device), blocksize=self.blocksize)
            self.cache[key] = (weakref.ref(storage), value)
            self.raw_bytes += storage.nbytes()
            self.packed_bytes += value[0].numel() + value[1].absmax.numel() * 4
            self.tensors += 1
        return value, tensor.shape, tensor.stride(), tensor.storage_offset()

    def unpack(self, record):
        if isinstance(record, torch.Tensor):
            return record
        (value, state), shape, stride, offset = record
        return self.dequantize(value, quant_state=state).as_strided(shape, stride, offset)

    def __enter__(self):
        self.hooks.__enter__()
        return self

    def __exit__(self, *args):
        self.cache.clear()  # only saved records, not the hook, own packed buffers
        return self.hooks.__exit__(*args)

    def stats(self):
        return dict(activation_raw_bytes=self.raw_bytes, activation_packed_bytes=self.packed_bytes,
                    activation_packs=self.tensors, activation_pack_hits=self.cache_hits)
