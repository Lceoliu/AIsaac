"""--source-replay 1 guards (2026-10-27) — CPU, threads and processes.

1. With replay, layer checkpointing, checkpoint blocks, 8-bit saved activations, the VJP merge and streams are forced
   off (a note is printed); without replay they are kept.
2. source_replay_raw_bytes (horizon_stats) counts the raw rows and pending kept for replay: > 0 after an encode,
   equal to the rows' and pending's bytes of the encoded frames, 0 without replay.
  python tests/test_source_replay_guards.py [--process]
"""
import contextlib, io, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch
from isaac_bridge.source_pipeline import SourcePipeline


def main(processes):
    torch.set_num_threads(2); torch.manual_seed(5)
    cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
               temporal_heads=4, temporal_window=4, rebuild_memory=True, memory_block=4, memory_tokens=2, source_cache=True)
    model = TokPolicy(**cfg).eval()
    flags = dict(layer_checkpoint=True, checkpoint_blocks=[0], saved_8bit=64, vjp_merge=1, streams=2)
    out = {}
    for replay in (False, True):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            pipe = SourcePipeline(model, ['cpu'], period=0, micro=8192, update_rollouts=1, processes=processes,
                                  replay=replay, **flags)
        kept = dict(layer_checkpoint=pipe.layer_checkpoint, checkpoint_blocks=pipe.checkpoint_blocks,
                    saved_8bit=pipe.saved_8bit, vjp_merge=pipe.vjp_merge, streams=pipe.streams)
        if replay:
            assert kept == dict(layer_checkpoint=False, checkpoint_blocks=None, saved_8bit=0, vjp_merge=0, streams=0), kept
            assert 'replay=1:' in log.getvalue() and 'forced off' in log.getvalue(), log.getvalue()
        else:
            assert kept['layer_checkpoint'] and kept['vjp_merge'] == 1 and kept['streams'] == 2, kept
        pipe.close()
        # raw bytes: an encode without the incompatible options
        pipe = SourcePipeline(model, ['cpu'], period=0, micro=8192, update_rollouts=1, processes=processes, replay=replay)
        rng = np.random.default_rng(5)
        seq = TS.make_seq(rng, nw=3, T=9, changes={(0, 2)}, resets=set(), exits=set())
        hs = [RawHistory.empty() for _ in range(3)]; req = []
        for t, rows in enumerate(seq):
            pend = M.synth_pending(rng, 3, model.heads)
            if t == len(seq) - 1: req = [(hs[w], rows[w], pend[w]) for w in range(3)]
            for w in range(3): hs[w] = hs[w].append(rows[w], pend[w], 4)
        with torch.no_grad(): SourceBatch(model, req, cache=False, pipeline=pipe)
        raw = pipe.horizon_stats()['source_replay_raw_bytes']
        frames = sum(c['rows'] for c in pipe.chunks if c is not None)
        row_bytes = seq[0][0].nbytes + np.asarray(M.synth_pending(rng, 1, model.heads)[0]).nbytes
        if replay: assert raw > 0 and raw == frames * row_bytes, (raw, frames, row_bytes)
        else: assert raw == 0, raw
        out[replay] = dict(kept=str(kept), raw_bytes=raw, frames=frames)
        pipe.close()
    print(json.dumps(dict(result='PASS', processes=processes, **{f'replay{int(k)}': v for k, v in out.items()})))


if __name__ == '__main__':
    if sys.platform == 'win32': os.environ.setdefault('ISAAC_HOST_TRIM_INTERVAL', '0')
    main('--process' in sys.argv)
