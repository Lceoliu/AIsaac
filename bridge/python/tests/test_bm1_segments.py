"""BM1: learn_segments -- off: no event, nothing recorded; on (CUDA): named segments with host time, stream span and lag,
read after one synchronisation the caller makes (none added)."""
import time

import pytest
import torch

from isaac_bridge import learn_segments as ls


def test_off_records_nothing(monkeypatch):
    monkeypatch.setattr(ls, 'ON', False)
    s = ls.Segments()
    monkeypatch.setattr(s, '_event', lambda: (_ for _ in ()).throw(AssertionError('an event while off')))
    s.begin()
    s.mark(start='a')
    s.sub(('a',), 'b')
    s.mark(end='c')
    assert s.finish(None, time.perf_counter()) == {} and not s.recs


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_on_names_spans_and_lags(monkeypatch):
    monkeypatch.setattr(ls, 'ON', True)
    s = ls.Segments()
    x = torch.randn(2048, 2048, device='cuda')
    s.begin()
    y = x @ x                      # unnamed: closed by mark(end='first')
    s.mark(end='first')
    s.mark(start='work')
    for _ in range(20):
        y = y @ x / 2048           # queued GPU work: span well above the host time, the lag positive
    s.sub(('work',), 'tail')
    s.mark(end='closing')          # ('tail' was opened named: it keeps its name; the short unnamed gap between
    #                                 mark(end='first') and mark(start='work') is the closing section's)
    end = s._event()
    torch.cuda.current_stream().synchronize()
    res = s.finish(end, time.perf_counter())
    assert set(res) == {'first', 'work', 'tail', 'closing'}
    h, span, lag, n = res['work']
    assert n == 1 and span > h and lag > 0
    assert ls.columns(res)['sg_work_n'] == 1
