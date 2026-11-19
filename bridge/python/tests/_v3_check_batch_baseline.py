"""Frozen scalar check_batch from 5a212ec (for equivalence tests)."""

import numpy as np

from isaac_bridge.v3_input import V3, _MAIN, _refuse, check_frame_header_v3


def check_batch(headers, versions, sizes, p_max=None, n_max=None, s_max=None, p0=None):
    """Host check of a batch: headers [B, >= 8] float32 (each record's first 8 words), versions [B] (visible_schema),
    sizes [B] (visible_bytes). Every record must pass split_frame's checks (VisibleV3Fatal otherwise). Returns
    (counts [B, 4] int64 p, e, n, s, truncated: dict(n=records with more inventory rows than n_max, s=... slots))."""
    headers = np.asarray(headers, np.float32)
    versions, sizes = np.asarray(versions), np.asarray(sizes)
    counts = np.zeros((len(headers), 4), np.int64)
    for i in range(len(headers)):
        vs = versions[i]
        if not np.isfinite(float(vs)) or float(vs) != V3.VERSION:
            _refuse(f'visible_schema {vs} is not schema_v3.VERSION {V3.VERSION} (record {i})')
        n = int(sizes[i])
        if n % 4 or n < V3.HEADER * 4 or n > V3.FRAME_BYTES:   # (Codex round 10: checked before the allocation)
            _refuse(f'{n} bytes is not a whole frame of floats within {V3.FRAME_BYTES} (record {i})')
        f = np.zeros(n // 4, np.float32)
        f[:V3.HEADER] = headers[i, :V3.HEADER]
        f[1] = headers[i, 1]
        counts[i] = check_frame_header_v3(f, exact=True)
    if p0 is not None:   # (Codex round 10: the batched entries as split_frame -- P row 0 the controlled body, located)
        p0 = np.asarray(p0, np.float32)
        bad = np.nonzero(~((p0[:, 0] == _MAIN) & (p0[:, 1] == 1)))[0]
        if len(bad):
            i = int(bad[0])
            _refuse(f'P row 0 is not the controlled body with a position (body_role and position_known exactly 1, got '
                    f'{float(p0[i, 0])}, {float(p0[i, 1])}; record {i}, {len(bad)} such)')
    if p_max is not None and (counts[:, 0] > p_max).any():
        _refuse(f'{int(counts[:, 0].max())} bodies, the padded layout holds {p_max}')
    trunc = dict(n=int((counts[:, 2] > n_max).sum()) if n_max else 0, s=int((counts[:, 3] > s_max).sum()) if s_max else 0)
    return counts, trunc
