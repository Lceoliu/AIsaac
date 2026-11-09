"""B50 phase 0: NVTX ranges on the actor and learner threads, and the nsys capture window.

ISAAC_BLOCK_NVTX=1 turns the ranges on (torch.cuda.nvtx.range_push / range_pop, about 1-2 us a pair); otherwise
push / pop / span are no-ops. Names are dotted: actor.* (the main loop's phases), step.* (SlotBlockActor.step),
g.* (ActorStepGraph.__call__), pages.* (PagedStore writes from _pages), lane.* (the lane service), learner.*.

ISAAC_NSYS_RANGE=a:b (hand-off counts of the actor loop): the capture window from the a-th collect's start to the
b-th's. ISAAC_NSYS_TRIGGER=nvtx (default): an NVTX start/end range named B50window in the default domain, for
`nsys profile --capture-range=nvtx --nvtx-capture=B50window` (no CUPTI: on the 4090 nsys 2023.4's libcupti 12.4
segfaults in train_tok at cudaProfilerStart, prof-b50m / prof-b50m3); ISAAC_NSYS_TRIGGER=cupti: cudaProfilerStart /
Stop for --capture-range=cudaProfilerApi. Without ISAAC_NSYS_RANGE nothing is called.
With ISAAC_BLOCK_NVTX=1 the cyclic garbage collector's passes are ranges too ('gc.collect', gc.callbacks, on the
thread that triggered the pass). (Measurement only: nothing here changes what the actor or learner compute.)
"""
import os

ON = os.environ.get('ISAAC_BLOCK_NVTX', '0') == '1'
TRIGGER = os.environ.get('ISAAC_NSYS_TRIGGER', 'nvtx')
_WINDOW = [0]

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
    """Called at the start of every collect with the hand-offs done so far: opens / closes the capture window at
    its bounds (the B50window NVTX range, or cudaProfilerStart / Stop with ISAAC_NSYS_TRIGGER=cupti). Returns
    'start', 'stop' or None."""
    if window is None:
        return None
    import torch
    if handoffs == window[0]:
        if TRIGGER == 'cupti':
            torch.cuda.cudart().cudaProfilerStart()
        else:
            _WINDOW[0] = torch.cuda.nvtx.range_start('B50window')
        print(f'B50 nsys: capture window opens at hand-off {handoffs} ({TRIGGER})', flush=True)
        return 'start'
    if handoffs == window[1]:
        if TRIGGER == 'cupti':
            torch.cuda.cudart().cudaProfilerStop()
        elif _WINDOW[0]:
            torch.cuda.nvtx.range_end(_WINDOW[0])
        print(f'B50 nsys: capture window closes at hand-off {handoffs} ({TRIGGER})', flush=True)
        return 'stop'
    return None


def _gc_callback(phase, info):
    if phase == 'start':
        _push('gc.collect')
    else:
        _pop()


if ON:
    import gc as _gc
    _gc.callbacks.append(_gc_callback)


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
