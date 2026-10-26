"""--source-balance: the assignment weights follow measured source work (2026-10-26) — CPU, no GPU.

1. observe_busy: with card 0 three times as busy as card 1 for several updates, card 0's weight rises above
   card 1's (geometric mean 1, clamped to [1/4, 4]); with balance off the weights stay 1.
2. schedule: two cpu workers, equal memory budgets: with balance off new episodes' blocks spread over both;
   with card 0 weighted 4 and card 1 1/4, card 1 takes most of the rows (> 85%).
3. The encoded sources are the same values wherever they are computed (Reader outputs equal).
  python tests/test_source_balance.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch, SourceView
from isaac_bridge.source_pipeline import SourcePipeline


def main():
    torch.set_num_threads(2); torch.manual_seed(23)
    model = TokPolicy(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32,
                      temporal_layers=2, temporal_heads=4, temporal_window=4, rebuild_memory=True, memory_block=4,
                      memory_tokens=2, source_cache=True).eval()
    # 1. the controller
    for balance in (False, True):
        pipe = SourcePipeline(model, ['cpu', 'cpu'], micro=8, balance=balance)
        pipe.devices = [torch.device('cpu', 0), torch.device('cpu', 1)]   # (two distinct device names for the events)
        for _ in range(6):
            pipe.events = [dict(kind='source_backward', device='cpu:0', start=0., end=3.),
                           dict(kind='encode', device='cpu:1', start=0., end=1.)]
            busy = pipe.observe_busy()
        assert busy == {'cpu:0': 3., 'cpu:1': 1.}, busy
        w = pipe.balance_weight
        if balance: assert w[0] > 1.5 and w[1] < 1 / 1.5 and abs(np.log(w).sum()) < 1e-9, w
        else: assert (w == 1).all(), w
        pipe.close()
    # 2./3. assignment, and the values wherever computed
    rng = np.random.default_rng(23)
    seq = TS.make_seq(rng, nw=6, T=24, changes={(0, 4), (1, 8)}, resets=set(), exits=set())
    outs, workers = {}, {}
    for name in ('off', 'busy0'):
        pipe = SourcePipeline(model, ['cpu', 'cpu'], micro=8, balance=name != 'off')
        if name == 'busy0': pipe.balance_weight = np.array([4.0, 0.25])
        hs = [RawHistory.empty() for _ in range(6)]; req = []
        for t, rows in enumerate(seq):
            pending = M.synth_pending(np.random.default_rng(t), 6, model.heads)
            if t == len(seq) - 1: req = [(hs[w], rows[w], pending[w]) for w in range(6)]
            for w in range(6): hs[w] = hs[w].append(rows[w], pending[w], 4)
        with torch.no_grad():
            bank = SourceBatch(model, req, cache=False, pipeline=pipe)
            out = SourceView(bank, np.arange(len(req))).forward(np.arange(len(req)))
        outs[name] = out[0].clone()
        rows = [sum(c['rows'] for c in pipe.chunks if c is not None and c['worker'] == i) for i in (0, 1)]
        workers[name] = rows[1] / sum(rows)   # (card 1's share of the encoded rows)
        pipe.close()
    # (weights 4 : 1/4: card 1 takes groups until its load is ~16 times card 0's)
    assert 0.2 < workers['off'] < 0.8 and workers['busy0'] > 0.85 and workers['busy0'] > workers['off'] + 0.2, workers
    assert torch.allclose(outs['off'], outs['busy0'], rtol=1e-5, atol=1e-6), float((outs['off'] - outs['busy0']).abs().max())
    print('{"result": "PASS", "card1_share": %s, "max_out_diff": %.3g}' % (
        {k: round(v, 3) for k, v in workers.items()}, float((outs['off'] - outs['busy0']).abs().max())))


if __name__ == '__main__':
    main()
