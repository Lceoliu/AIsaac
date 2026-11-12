#!/usr/bin/env python3
"""The gate's first step (2026-11-12, the coordinator): every native component built and loaded in this tree -- their
tests skip without them, so a failed build passed a gate silently (5ee81f3 to m0301cca: the C++ actor driver). Run from
the tree's bridge/python in the container. One line each, 'EXT <name> ok' or 'EXT <name> FAIL <why>'; exit 1 on any FAIL.
  actor    actor_native (bridge/engine/abp_actor.cpp: the driver, the native consumer, lineage, lanes, PageStore)
  learn    learn_native (abp_learn.c: the learner's host packing)
  visible  visible_native (the C visible encoder)"""
import os
import sys

sys.path.insert(0, os.getcwd())
bad = 0
from isaac_bridge import actor_native, learn_native, visible_native   # noqa: E402

for name, load, why in (('actor', actor_native.load, lambda: actor_native.REASON),
                        ('learn', learn_native.load, lambda: 'build / load failed or ISAAC_BLOCK_NATIVE_PACK=0'),
                        ('visible', visible_native.load, lambda: 'build / load failed or ISAAC_VISIBLE_NATIVE=0')):
    try:
        ok = load() is not None
    except Exception as e:   # noqa: BLE001
        ok, why = False, (lambda e=e: f'{type(e).__name__}: {e}')
    print('EXT', name, 'ok' if ok else 'FAIL ' + str(why())[-1500:].replace('\n', ' | '), flush=True)
    bad += not ok
sys.exit(1 if bad else 0)
