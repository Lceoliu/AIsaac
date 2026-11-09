"""B50 phase 0: NVTX ranges on the actor and learner threads, and the nsys capture window.

ISAAC_BLOCK_NVTX=1 turns the ranges on (torch.cuda.nvtx.range_push / range_pop, about 1-2 us a pair); otherwise
push / pop / span are no-ops. Names are dotted: actor.* (the main loop's phases), step.* (SlotBlockActor.step),
g.* (ActorStepGraph.__call__), pages.* (PagedStore writes from _pages), lane.* (the lane service), learner.*.

ISAAC_NSYS_RANGE=a:b (hand-off counts of the actor loop): cudaProfilerStart() when the a-th collect begins,
cudaProfilerStop() when the b-th begins, for `nsys profile --capture-range=cudaProfilerApi`. Without it nothing is
called. (Measurement only: nothing here changes what the actor or learner compute.)
"""
import os

ON = os.environ.get('ISAAC_BLOCK_NVTX', '0') == '1'

if ON:
    import torch
    _push, _pop = torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop

    def push(name):
        _push(name)

    def pop():
        _pop()
else:
    def push(name):
        pass

    def pop():
        pass


class span:
    """with span('name'): ... (a no-op context manager when NVTX is off)."""
    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        push(self.name)

    def __exit__(self, *exc):
        pop()


def profiler_window():
    """(start, stop) hand-off counts of ISAAC_NSYS_RANGE, or None."""
    v = os.environ.get('ISAAC_NSYS_RANGE', '')
    if not v:
        return None
    a, b = v.split(':')
    return int(a), int(b)


def profiler_mark(handoffs, window):
    """Called at the start of every collect with the hand-offs done so far: starts / stops the CUDA profiler at the
    window's bounds (nsys --capture-range=cudaProfilerApi). Returns 'start', 'stop' or None."""
    if window is None:
        return None
    import torch
    if handoffs == window[0]:
        torch.cuda.cudart().cudaProfilerStart()
        print(f'B50 nsys: cudaProfilerStart at hand-off {handoffs}', flush=True)
        return 'start'
    if handoffs == window[1]:
        torch.cuda.cudart().cudaProfilerStop()
        print(f'B50 nsys: cudaProfilerStop at hand-off {handoffs}', flush=True)
        return 'stop'
    return None


def start(name):
    """A start/end range (independent of the push/pop stack): its id, 0 when off."""
    if not ON:
        return 0
    import torch
    return torch.cuda.nvtx.range_start(name)


def end(rid):
    if ON and rid:
        import torch
        torch.cuda.nvtx.range_end(rid)
