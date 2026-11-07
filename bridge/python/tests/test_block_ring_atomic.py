"""Deterministic slot overwrite between row.copy() and meta/reference reads.

Runs the real TokSampler readers; only the row ndarray is instrumented.
No game process, learner, CUDA, or source-code modification is involved.
"""
from types import SimpleNamespace
import json

import numpy as np
from isaac_bridge.tok_obs import ROW
from isaac_bridge.tok_sampler import TokSampler, ST, TEACH_RING, TEACH_META
from isaac_bridge.tok_teacher2 import V2_RING, V2_META


class CopyThenOverwrite:
    def __init__(self, rows, overwrite):
        self.rows, self.overwrite = rows, overwrite

    def __getitem__(self, index):
        owner = self

        class Record:
            def copy(self):
                result = owner.rows[index].copy()
                owner.overwrite()
                return result

        return Record()


def reproduce(kind, interleave):
    s = SimpleNamespace(cfg=SimpleNamespace(workers=1, memory_dim=0))
    s.stats = np.zeros((1, max(ST.values()) + 1))
    rows = np.zeros((1, 1), ROW)
    rows[0, 0]["episode"], rows[0, 0]["t"] = 11, 3
    events = []
    if kind == "injury":
        s.teach_rows = rows
        s.teach_meta = np.zeros((1, 1, TEACH_META), np.int32)
        s.teach_meta[0, 0, 0] = 101
        s.teach_seen = np.zeros(1, np.int64)
        s.block_arrays = [None] * 5 + [np.ones((1, 1), np.int64)]
        s.stats[0, ST["teach"]] = 1
        contexts = {(0, 11, 3): "old", (0, 22, 7): "new"}
        s.block_lane_service = SimpleNamespace(injury_context=lambda *key: contexts[key])

        def overwrite():
            s.block_arrays[5][0, 0] = 0
            rows[0, 0]["episode"], rows[0, 0]["t"] = 22, 7
            s.teach_meta[0, 0, 0] = 201
            s.block_arrays[5][0, 0] = TEACH_RING + 1
            s.stats[0, ST["teach"]] = TEACH_RING + 1
            events.append("writer published next slot generation after old row copy")

        if interleave:
            s.teach_rows = CopyThenOverwrite(rows, overwrite)
        got = TokSampler.teacher_records(s, with_block=True)
        stamp = int(s.block_arrays[5][0, 0])
    else:
        s.v2_rows = rows
        s.v2_meta = np.zeros((1, 1, V2_META), np.float64)
        s.v2_meta[0, 0, 0] = 101
        s.v2_seen = np.zeros(1, np.int64)
        s.v2_h = None
        s.v2_t = np.array([[[1., 0., 1.]]], np.float32)
        s.lane_t = np.zeros((1, 3), np.float32)
        s.stats[0, ST["v2_records"]] = 1
        contexts = {1: "old", 2: "new"}
        s.block_contexts = SimpleNamespace(get=contexts.__getitem__, retire=lambda live: None)

        def overwrite():
            s.v2_t[0, 0, -1] = 0
            rows[0, 0]["episode"], rows[0, 0]["t"] = 22, 7
            s.v2_meta[0, 0, 0] = 201
            s.v2_t[0, 0, :2] = [2., 0.]
            s.v2_t[0, 0, -1] = V2_RING + 1
            s.stats[0, ST["v2_records"]] = V2_RING + 1
            events.append("writer published next slot generation after old row copy")

        if interleave:
            s.v2_rows = CopyThenOverwrite(rows, overwrite)
        got = TokSampler.v2_records(s, with_block=True)
        stamp = int(s.v2_t[0, 0, -1])
    rr, mm, cc = got
    accepted = len(rr)
    result = dict(kind=kind, interleave=interleave, accepted=accepted,
                  episode=int(rr[0]["episode"]) if accepted else None,
                  meta_marker=int(mm[0, 0]) if accepted else None,
                  context=cc[0] if accepted else None, final_stamp=stamp,
                  torn_accepted=bool(interleave and accepted and int(rr[0]["episode"]) == 11
                                     and (int(mm[0, 0]) != 101 or cc[0] != "old")), events=events)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result




def test_injury_ring_rejects_overwrite_during_copy():
    assert reproduce('injury',False)['accepted']==1
    assert reproduce('injury',True)['accepted']==0


def test_cf_ring_rejects_overwrite_during_copy():
    assert reproduce('cf',False)['context']=='old'
    assert reproduce('cf',True)['accepted']==0
