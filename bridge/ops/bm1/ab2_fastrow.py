#!/usr/bin/env python3
"""ab2 (C107, 2026-11-14): FastRow of a tree with a libabp_turbo.so, the workers' own test (tok_obs.fast_row_function:
abp_row_encode_version and the layout constants must match, else None and the workers fall back to the Lua path).
  (in the container, from <tree>/bridge/python) python ab2_fastrow.py <libabp_turbo.so> [...]"""
import ctypes
import sys

from isaac_bridge import tok_obs

want = (tok_obs._ROW_VERSION << 16) | (tok_obs.RC_COUNT << 8) | len(tok_obs._ROW_FIELDS)
print(f'tree: FAST_ROW {tok_obs.FAST_ROW}, wants abp_row_encode_version 0x{want:06x} (row version {tok_obs._ROW_VERSION}, '
      f'RC_COUNT {tok_obs.RC_COUNT}, fields {len(tok_obs._ROW_FIELDS)})')
for p in sys.argv[1:]:
    v = ctypes.CDLL(p).abp_row_encode_version
    v.restype = ctypes.c_int
    fn = tok_obs.fast_row_function(p)
    print(f'{p}: abp_row_encode_version 0x{v():06x}, fast_row_function {"ON" if fn is not None else "None (OFF)"}')
