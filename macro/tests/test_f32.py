"""isaac_macro.f32: float32 arithmetic without numpy must match numpy's bit for bit (the web page runs
the generators under Pyodide without numpy)."""
import random
import struct

import pytest

from isaac_macro.f32 import F32


def _bits(x) -> int:
    return struct.unpack('<I', struct.pack('<f', x))[0]


def test_rounds_like_a_c_cast():
    assert F32(0.1) == 0.10000000149011612
    assert F32(2.3283061589829401e-10) == struct.unpack('<f', struct.pack('<f', 2.3283061589829401e-10))[0]
    assert F32(4294967295) == 4294967296.0          # a uint32 seed becomes float32 first (RNG::Random)
    assert F32(16777217) == 16777216.0              # ties to even


def test_operations_match_numpy():
    np = pytest.importorskip('numpy')
    rnd = random.Random(1)
    for _ in range(20000):
        a = F32(rnd.uniform(-1e6, 1e6) * rnd.choice((1e-6, 1e-3, 1, 1e3)))
        b = F32(rnd.uniform(-1e3, 1e3) * rnd.choice((1e-6, 1e-3, 1, 1e3)) or 1.0)
        na, nb = np.float32(a), np.float32(b)
        assert _bits(F32(a + b)) == _bits(float(na + nb))
        assert _bits(F32(a - b)) == _bits(float(na - nb))
        assert _bits(F32(a * b)) == _bits(float(na * nb))
        assert _bits(F32(a / b)) == _bits(float(na / nb))
    for _ in range(5000):                            # RNG::Random(): float32(state) * the float32 constant
        seed = rnd.getrandbits(32)
        want = np.float32(seed) * np.float32(2.3283061589829401e-10)
        assert _bits(F32(F32(seed) * F32(2.3283061589829401e-10))) == _bits(float(want))
