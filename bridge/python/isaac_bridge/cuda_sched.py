"""B62: how the process's CUDA host threads wait for the GPU (cudaSetDeviceFlags scheduling), set before the CUDA
context exists. ISAAC_CUDA_BLOCKING_SYNC: unset / '' / 'auto' (the driver's default: spin while the GPU is busy when there
are more cores than GPUs), 'spin', 'yield', 'blocking' (or '1': the waiting thread sleeps on an OS primitive and wakes
when the work is done). Process-wide: the learner, the actor main thread, the C++ driver and lane threads all wait this
way. No effect on values: only how a host thread waits."""
import ctypes
import os

FLAGS = {'auto': 0, 'spin': 1, 'yield': 2, 'blocking': 4}
NAMES = {v: k for k, v in FLAGS.items()}
_LIB = []


def _cudart():
    """torch's own CUDA runtime (already loaded with torch: dlopen by soname returns that handle)."""
    if _LIB:
        return _LIB[0]
    import torch   # noqa: F401  (loads libcudart)
    for name in ('libcudart.so.11.0', 'libcudart.so.12', 'libcudart.so'):
        try:
            _LIB.append(ctypes.CDLL(name))
            return _LIB[0]
        except OSError:
            continue
    return None


def requested():
    v = os.environ.get('ISAAC_CUDA_BLOCKING_SYNC', '').strip().lower()
    if v in ('', '0', 'auto'):
        return None
    return 'blocking' if v == '1' else v


def apply():
    """Set the requested scheduling flag (before any CUDA context). Returns (mode, cudaError code) or (None, None)."""
    mode = requested()
    if mode is None:
        return None, None
    if mode not in FLAGS:
        raise ValueError(f'ISAAC_CUDA_BLOCKING_SYNC={mode!r}: one of {sorted(FLAGS)}')
    lib = _cudart()
    if lib is None:
        return mode, -1
    return mode, int(lib.cudaSetDeviceFlags(ctypes.c_uint(FLAGS[mode])))


def effective():
    """The scheduling flag the current device's context runs with (after it exists): 'auto' / 'spin' / 'yield' /
    'blocking', or None when it cannot be read."""
    lib = _cudart()
    if lib is None:
        return None
    f = ctypes.c_uint(0)
    if lib.cudaGetDeviceFlags(ctypes.byref(f)) != 0:
        return None
    return NAMES.get(f.value & 0x7, hex(f.value & 0x7))
