"""IEEE single precision without numpy, for the engine's float32 arithmetic.

F32(x) rounds a number to the nearest float32 (ties to even), as np.float32(x) and a C cast do. A
sum, difference, product or quotient of two float32 values computed in double and then rounded once
is the exact float32 result, because double carries more than 2 * 24 + 2 significand bits. So
F32(a * b) matches np.float32(a) * np.float32(b) bit for bit, as long as every such operation is
wrapped in F32(). The generators wrap them all; this is what lets the web page run without numpy.
"""
from __future__ import annotations

import struct

_FLOAT = struct.Struct('<f')


def F32(x) -> float:
    """x rounded to float32, returned as a Python float."""
    return _FLOAT.unpack(_FLOAT.pack(x))[0]
