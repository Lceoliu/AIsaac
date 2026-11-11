"""BM1: per-segment timing of a learner update (measurement only, ISAAC_BLOCK_TIMING=1; off: every call is one bool
test, no event, no CUDA call).

The learner thread marks segment boundaries. Each boundary records the host time and a CUDA event on the thread's
current stream (the learner's device and stream: no device is named here). Nothing synchronises: after the update's
own last synchronisation (learn()'s tail: torch.cuda.current_stream().synchronize(), with an event recorded just before
it and the host time right after it) every event is complete, the elapsed times are read and the segments summed.
Per segment name, per update:
  host  the host's time in the segment (entry to the next boundary, no synchronisation: what u_* measure),
  span  the elapsed time between the segment's two events on the learner's stream: when the stream reached the
        segment's start and its end; it includes the stream's idle gaps -- not "GPU busy time",
  lag   at the segment's end, how long its last work was still queued on the stream: the end event's completion
        (anchored on the update's last synchronisation) minus the host time the segment returned; negative when the
        stream had finished before the host did (the host was the slower one).
span clearly above host: the GPU's work bounds the segment; host close to span with a small lag: the host's
submission does.
Boundaries: mark(start=name) opens a segment named `name` (closing the open one); mark(end=name) closes the open
segment and names it `name` when it was opened unnamed (learn()'s sec(name) closes the code since the previous section
under that section's name); sub(parents, name) splits the open segment when it is one of `parents` (a PPO part's forward:
its plan, the host packing it waits for, its dispatch)."""
import os
import time

ON = os.environ.get('ISAAC_BLOCK_TIMING') == '1' and os.environ.get('ISAAC_LEARN_SEGMENTS', '1') == '1'


class Segments:
    def __init__(self):
        self.recs = []      # (name, host start, host end, start event, end event)
        self.open = None    # (name or None, host start, start event)
        self.active = False
        self.thread = None  # (the learner thread that began the update: marks from any other thread are ignored)

    def _event(self):
        import torch
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def begin(self):
        """An update starts (learner thread). A CPU-only process stays inactive (CUDA events need a device)."""
        if not ON:
            return
        import threading
        import torch
        if not torch.cuda.is_available():
            return
        self.recs, self.active, self.thread = [], True, threading.get_ident()
        self.open = (None, time.perf_counter(), self._event())

    def mark(self, start=None, end=None):
        if not (ON and self.active):
            return
        import threading
        if threading.get_ident() != self.thread:
            return
        t, e = time.perf_counter(), self._event()
        name, t0, e0 = self.open
        if end is not None:   # (unnamed code closed by a section's end is that section's, with every unnamed piece a
            #                    start mark cut off since the previous section's end)
            self.recs = [(end if r[0] is None else r[0],)+r[1:] for r in self.recs]
            name = name or end
        self.recs.append((name, t0, t, e0, e))
        self.open = (start, t, e)

    def sub(self, parents, name):
        """Split the open segment when its name is one of `parents`: the rest is a segment called `name`."""
        if ON and self.active and self.open is not None and self.open[0] in parents:
            self.mark(start=name)

    def finish(self, end_event, t_sync):
        """After the update's last synchronisation: end_event recorded just before it, t_sync the host time right
        after it. Returns {segment: (host s, span s, mean lag s, count)} and stops."""
        if not (ON and self.active):
            return {}
        self.active = False
        out = {}
        for name, t0, t1, e0, e1 in self.recs:
            name = name or 'unnamed'   # (code after the last section's end)
            try:
                span = e0.elapsed_time(e1)/1e3
                done = t_sync - e1.elapsed_time(end_event)/1e3   # (e1's completion on the host clock)
            except RuntimeError:   # (an event never recorded on this stream's work: skipped)
                continue
            h, s, lag, n = out.get(name, (0.0, 0.0, 0.0, 0))
            out[name] = (h+t1-t0, s+span, lag+(done-t1), n+1)
        self.recs = []
        return {k: (h, s, lag/max(n, 1), n) for k, (h, s, lag, n) in out.items()}


SEG = Segments()


def columns(res):
    """progress.csv columns sg_<name>_host / _span / _lag / _n."""
    row = {}
    for name, (h, s, lag, n) in res.items():
        k = 'sg_'+name.replace('.', '_').replace(':', '_')
        row.update({k+'_host': h, k+'_span': s, k+'_lag': lag, k+'_n': n})
    return row
